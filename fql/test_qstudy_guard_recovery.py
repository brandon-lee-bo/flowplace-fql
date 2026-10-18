"""Software-only checks for the narrow, real-FAILED recovery admission."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_guard_recovery as recovery


class GuardRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root/'plan.json'
        self.plan.write_text('{}')
        self.row = dict(flat_index=0, task={'sample_dir': str(self.root)}, attempts=[dict(
            metadata={'job_id': '111', 'restart_count': 1}, attempt_sha256='old-sha')])

    def guarded(self, states):
        return recovery.guarded_states(self.row, self.plan, real_states=lambda job: states)

    def fixtures(self):
        return (patch.object(recovery, 'checked_plan', return_value=({}, {'failed_rows': [self.row]}, [])),
                patch.object(recovery, 'validate_original', return_value={'restart_count': 1}),
                patch.dict(os.environ, {'SLURM_JOB_ID': '222', 'SLURM_RESTART_COUNT': '0'}))

    def test_scoped_failed_preserves_actual_state_and_admission(self):
        a, b, c = self.fixtures()
        with a, b, c:
            states = self.guarded(['FAILED'])('111')
        self.assertEqual(states, ['FAILED'])
        value = json.loads((self.root/'guard_failure_recovery/222_restart_0.json').read_text())
        self.assertEqual(value['actual_observed_scheduler_states'], ['FAILED'])
        self.assertTrue(value['admitted_under_original_sample_flock'])
        self.assertEqual(value['previous_job_id'], '111')

    def test_new_failed_attempt_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Undiagnosed FAILED'):
            self.guarded(['FAILED'])('222')

    def test_mixed_failed_live_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Undiagnosed FAILED'):
            self.guarded(['FAILED', 'RUNNING'])('111')

    def test_live_and_unknown_are_not_relabelled(self):
        for observed in (['RUNNING'], ['PENDING'], [], ['UNKNOWN'], ['PREEMPTED']):
            self.assertEqual(self.guarded(observed)('111'), observed)

    def test_evidence_tamper_rejected_before_admission(self):
        with patch.object(recovery, 'checked_plan', side_effect=ValueError('SHA changed')):
            with self.assertRaisesRegex(ValueError, 'SHA changed'):
                self.guarded(['FAILED'])('111')
        self.assertFalse((self.root/'guard_failure_recovery').exists())

    def test_inside_lock_row_change_rejected(self):
        other = copy.deepcopy(self.row)
        other['task']['sample_dir'] = '/other/request'
        with patch.object(recovery, 'checked_plan', return_value=({}, {'failed_rows': [other]}, [])):
            with self.assertRaisesRegex(ValueError, 'row changed'):
                self.guarded(['FAILED'])('111')

    def test_changed_old_manifest_rejected(self):
        a, _, c = self.fixtures()
        with a, c, patch.object(recovery, 'validate_original', side_effect=ValueError('old manifest changed')):
            with self.assertRaisesRegex(ValueError, 'old manifest changed'):
                self.guarded(['FAILED'])('111')

    def test_repeated_identical_admission_is_safe(self):
        a, b, c = self.fixtures()
        with a, b, c:
            self.guarded(['FAILED'])('111')
            first = (self.root/'guard_failure_recovery/222_restart_0.json').read_bytes()
            self.guarded(['FAILED'])('111')
            self.assertEqual(first, (self.root/'guard_failure_recovery/222_restart_0.json').read_bytes())

    def test_admission_tamper_rejected(self):
        a, b, c = self.fixtures()
        with a, b, c:
            self.guarded(['FAILED'])('111')
            path = self.root/'guard_failure_recovery/222_restart_0.json'
            value = json.loads(path.read_text())
            value['previous_job_id'] = '333'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'admission changed'):
                self.guarded(['FAILED'])('111')

    def test_exact_frozen_bytecode_private_globals(self):
        original = recovery.backend.measure
        before = original.__globals__['RESTARTABLE']
        original_states = original.__globals__['states']
        guard = lambda job: ['FAILED']
        copied = recovery.private_measure(original, guard)
        self.assertIs(copied.__code__, original.__code__)
        self.assertIsNot(copied.__globals__, original.__globals__)
        self.assertIs(copied.__globals__['states'], guard)
        self.assertIn('FAILED', copied.__globals__['RESTARTABLE'])
        self.assertIs(before, original.__globals__['RESTARTABLE'])
        self.assertNotIn('FAILED', original.__globals__['RESTARTABLE'])
        self.assertIs(original_states, original.__globals__['states'])

    def test_completed_native_outcome_reuses_frozen_early_path(self):
        sample = self.root/'sample'
        sample.mkdir()
        (sample/'outcome.json').write_text('{"status":"success"}')
        called = []
        copied = recovery.private_measure(recovery.backend.measure,
            lambda job: self.fail('A completed outcome must never consult/restart attempts'))
        copied.__globals__.update(context=lambda exp: ({}, None, {}),
            load_requests=lambda *args: ([{}], self.root/'tasks.jsonl'),
            checked_request=lambda *args: sample,
            checked_existing=lambda *args: called.append(args[0]),
            subprocess=type('NoProcesses', (), {'run': lambda *args: self.fail('No physical process allowed')})())
        copied(argparse.Namespace(experiment='unused', seed=0, round=0, index=0))
        self.assertEqual(called, [{'status': 'success'}])

    def test_fingerprint_tamper(self):
        path = self.root/'evidence'
        path.write_text('fixed')
        expected = recovery.sha(path)
        self.assertEqual(recovery.bound(path, expected), path)
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            recovery.bound(path, expected)

    def test_plan_tamper_rejected_inside_original_lock(self):
        expected = recovery.sha(self.plan)
        guard = recovery.guarded_states(self.row, self.plan, real_states=lambda job: ['FAILED'],
            plan_sha256=expected)
        self.plan.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            guard('111')

    def test_scope_is_exact_original_25(self):
        self.assertEqual(len(recovery.INDICES), 25)
        self.assertEqual(len(set(recovery.INDICES)), 25)
        self.assertNotIn(8, recovery.INDICES)  # Existing successful evaluation.
        self.assertNotIn(95, recovery.INDICES)

    def test_real_audited_evidence_and_frozen_backend_still_match(self):
        evidence, tasks = recovery.diagnosis()
        self.assertEqual(len(tasks), 96)
        self.assertEqual(len(evidence['failed_rows']), 25)
        self.assertTrue(all(not r['attempts'][0]['manifest_time_end'] for r in evidence['failed_rows']))

    def stream_fixture(self):
        folder = self.root/'seed_0/round_0/requests'
        folder.mkdir(parents=True)
        tasks = folder/'tasks.jsonl'
        tasks.write_text('{"sample_dir":'+json.dumps(str(self.root))+'}\n')
        (self.root/'request.json').write_text('{}')
        row = copy.deepcopy(self.row)
        row['task'].update(experiment=str(self.root), seed=0, round=0, index=0,
            tasks_sha256=recovery.sha(tasks))
        row['request_sha256'] = recovery.sha(self.root/'request.json')
        return row, tasks

    def test_stream_manifest_change_rejected_before_no_attempt_path(self):
        row, tasks = self.stream_fixture()
        load, _ = recovery.target_guards(row, real_load=lambda *args: ([{'sample_dir': str(self.root)}], tasks))
        tasks.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            load(str(self.root), 0, 0)

    def test_unchanged_stream_cannot_resolve_to_another_sample(self):
        row, tasks = self.stream_fixture()
        load, _ = recovery.target_guards(row, real_load=lambda *args: ([{'sample_dir': '/other'}], tasks))
        with self.assertRaisesRegex(ValueError, 'another sample'):
            load(str(self.root), 0, 0)

    def test_checked_request_target_guard_runs_after_original_validation(self):
        row, tasks = self.stream_fixture()
        called = []
        _, check = recovery.target_guards(row, real_check=lambda *args: (called.append(True) or Path('/other')))
        with self.assertRaisesRegex(ValueError, 'unaudited recovery request'):
            check({}, argparse.Namespace(seed=0, round=0), {}, {})
        self.assertEqual(called, [True])

    def test_target_guard_covers_completed_outcome_and_empty_attempts(self):
        row, tasks = self.stream_fixture()
        load, check = recovery.target_guards(row,
            real_load=lambda *args: ([{'sample_dir': str(self.root)}], tasks),
            real_check=lambda *args: self.root)
        parsed, _ = load(str(self.root), 0, 0)
        self.assertEqual(check(parsed[0], argparse.Namespace(seed=0, round=0), {}, {}), self.root)
        (self.root/'request.json').write_text('{"different":true}')
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            check(parsed[0], argparse.Namespace(seed=0, round=0), {}, {})


if __name__ == '__main__':
    unittest.main()
