"""Software-only quota recovery: no model, native runner or physical label."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_quota_recovery as recovery


class QuotaRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sample = self.root/'sample'
        self.sample.mkdir()
        self.tasks = self.root/'seed_0/round_0/requests/tasks.jsonl'
        self.tasks.parent.mkdir(parents=True)
        self.request = dict(sample_dir=str(self.sample),seed=0,round=0)
        self.tasks.write_text(json.dumps(self.request)+'\n')
        (self.sample/'request.json').write_text(json.dumps(self.request))
        self.old = self.sample/'backend_attempts/old'
        self.old.mkdir(parents=True)
        (self.old/'attempt.json').write_text('{}')
        self.row = dict(kind='source',array='1797162',flat_index=15,
            task=dict(experiment=str(self.root),seed=0,round=0,index=0,
                sample_dir=str(self.sample),tasks_sha256=recovery.sha(self.tasks)),
            request=self.request,request_sha256=recovery.sha(self.sample/'request.json'),
            excluded_terminal_infra_attempts=[str(self.old/'attempt.json')])
        self.plan = self.root/'plan.json'
        self.plan.write_text('{}')

    def guards(self):
        return recovery.target_guards(self.row,
            real_load=lambda *args: ([self.request],self.tasks),
            real_check=lambda *args: self.sample)

    def path(self):
        _,check = self.guards()
        return check(self.request,argparse.Namespace(seed=0,round=0),{}, {})

    def test_exact_scope_81_excludes_original_completed_15(self):
        self.assertEqual(sum(len(s['indices']) for s in recovery.SPECS.values()),81)
        self.assertEqual(recovery.SPECS['source']['indices'],list(range(15,48)))
        self.assertEqual(recovery.SPECS['flow']['indices'],list(range(48)))

    def test_iterator_omits_only_exact_old_quota_metadata(self):
        later = self.sample/'backend_attempts/new'
        later.mkdir();(later/'attempt.json').write_text('{}')
        paths = list((self.path()/'backend_attempts').glob('*/attempt.json'))
        self.assertEqual(paths,[later/'attempt.json'])
        self.assertTrue((self.old/'attempt.json').exists())

    def test_noattempt_rows_omit_nothing(self):
        self.row['excluded_terminal_infra_attempts'] = []
        self.assertEqual(list((self.path()/'backend_attempts').glob('*/attempt.json')),[self.old/'attempt.json'])

    def test_other_glob_pattern_and_directory_are_not_filtered(self):
        self.assertEqual(list((self.path()/'backend_attempts/old').glob('*.json')),[self.old/'attempt.json'])
        self.assertEqual(list((self.path()/'backend_attempts').glob('*/*')),[self.old/'attempt.json'])

    def test_loader_calls_original_then_validates_exact_request(self):
        called = []
        load,_ = recovery.target_guards(self.row,
            real_load=lambda *args: (called.append(True) or ([dict(self.request,seed=99)],self.tasks)))
        with self.assertRaisesRegex(ValueError,'another request'):
            load(str(self.root),0,0)
        self.assertEqual(called,[True])

    def test_wrong_stream_identity_refused_before_original_load(self):
        load,_ = recovery.target_guards(self.row,real_load=lambda *args:self.fail('must not load'))
        with self.assertRaisesRegex(ValueError,'unaudited experiment'):
            load(str(self.root),1,0)

    def test_stream_and_request_tamper_refused_even_noattempt(self):
        load,check = self.guards()
        self.tasks.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):
            load(str(self.root),0,0)
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):
            check(self.request,argparse.Namespace(seed=0,round=0),{}, {})

    def test_request_tamper_refused_on_completed_outcome_early_path(self):
        (self.sample/'outcome.json').write_text('{}')
        (self.sample/'request.json').write_text('{"changed":true}')
        _,check = self.guards()
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):
            check(self.request,argparse.Namespace(seed=0,round=0),{}, {})

    def test_original_checker_runs_before_path_subclass(self):
        called = []
        _,check = recovery.target_guards(self.row,
            real_check=lambda *args:(called.append(True) or Path('/another')))
        with self.assertRaisesRegex(ValueError,'unaudited recovery request'):
            check(self.request,argparse.Namespace(seed=0,round=0),{}, {})
        self.assertEqual(called,[True])

    def lock(self):
        handle = (self.sample/'.qstudy_backend.lock').open('a')
        self.addCleanup(handle.close)
        return handle

    def test_lock_acquired_before_evidence_recheck(self):
        events = []
        proxy = recovery.lock_guard(self.row,self.plan,recovery.sha(self.plan),
            real_flock=lambda *args:events.append('acquired'))
        with patch.object(recovery,'checked_plan',return_value={'rows':[self.row]}), \
                patch.object(recovery,'validate_original',side_effect=lambda row:events.append('validated')), \
                patch.dict(os.environ,{'SLURM_JOB_ID':'222'}):
            proxy.flock(self.lock(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertEqual(events,['acquired','validated'])
        record = recovery.read(self.sample/'quota_infrastructure_recovery/222.json')
        self.assertEqual(record['actual_original_scheduler_states'],['FAILED'])
        self.assertTrue(record['original_sample_flock_held'])

    def test_lock_time_row_tamper_prevents_admission(self):
        wrong = dict(self.row,flat_index=16)
        proxy = recovery.lock_guard(self.row,self.plan,recovery.sha(self.plan),real_flock=lambda *args:None)
        with patch.object(recovery,'checked_plan',return_value={'rows':[wrong]}):
            with self.assertRaises(StopIteration):
                proxy.flock(self.lock(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertFalse((self.sample/'quota_infrastructure_recovery').exists())

    def test_original_live_or_unknown_scheduler_is_never_admitted(self):
        proxy = recovery.lock_guard(self.row,self.plan,recovery.sha(self.plan),real_flock=lambda *args:None)
        with patch.object(recovery,'checked_plan',return_value={'rows':[self.row]}), \
                patch.object(recovery,'validate_original',side_effect=RuntimeError('truly FAILED required')):
            with self.assertRaisesRegex(RuntimeError,'truly FAILED'):
                proxy.flock(self.lock(),fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_admission_repeated_identically_keeps_first_record(self):
        proxy = recovery.lock_guard(self.row,self.plan,recovery.sha(self.plan),real_flock=lambda *args:None)
        with patch.object(recovery,'checked_plan',return_value={'rows':[self.row]}), \
                patch.object(recovery,'validate_original'),patch.dict(os.environ,{'SLURM_JOB_ID':'222'}):
            proxy.flock(self.lock(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=(self.sample/'quota_infrastructure_recovery/222.json').read_bytes()
            proxy.flock(self.lock(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertEqual(old,(self.sample/'quota_infrastructure_recovery/222.json').read_bytes())

    def test_exact_measure_bytecode_no_shared_global_or_failed_extension(self):
        original = recovery.backend.measure
        before = dict(original.__globals__)
        copied = recovery.private_measure(original,object(),object(),object())
        self.assertIs(copied.__code__,original.__code__)
        self.assertIsNot(copied.__globals__,original.__globals__)
        self.assertIs(copied.__globals__['states'],before['states'])
        self.assertIs(copied.__globals__['RESTARTABLE'],before['RESTARTABLE'])
        self.assertNotIn('FAILED',copied.__globals__['RESTARTABLE'])
        self.assertIs(original.__globals__['fcntl'],before['fcntl'])

    def test_frozen_completed_outcome_early_return_never_invokes_native(self):
        (self.sample/'outcome.json').write_text('{"status":"success"}')
        observed=[]
        load,check=self.guards()
        copied=recovery.private_measure(recovery.backend.measure,load,check,fcntl)
        copied.__globals__.update(context=lambda exp:({},None,{}),
            checked_existing=lambda *args:observed.append(args[0]),
            subprocess=type('NoProcess',(),{'run':lambda *args:self.fail('physical rerun forbidden')})())
        copied(argparse.Namespace(experiment=str(self.root),seed=0,round=0,index=0))
        self.assertEqual(observed,[{'status':'success'}])

    def test_new_failed_attempt_is_visible_and_originally_not_restartable(self):
        new=self.sample/'backend_attempts/new';new.mkdir();(new/'attempt.json').write_text('{}')
        self.assertIn(new/'attempt.json',list((self.path()/'backend_attempts').glob('*/attempt.json')))
        self.assertNotIn('FAILED',recovery.backend.RESTARTABLE)

    def prepared_measure(self):
        self.request.update(run_id='go2o_test',arm='arm',role='evaluation',index=0,
            raw_action_sha256='raw',action_sha256='executed',noise_sha256='noise',
            raw_action=[[0.,0.]],executed_action=[[0.,0.]],checkpoint_sha256='checkpoint',
            training_feedback_eligible=False)
        self.tasks.write_text(json.dumps(self.request)+'\n')
        (self.sample/'request.json').write_text(json.dumps(self.request))
        self.row['request_sha256']=recovery.sha(self.sample/'request.json')
        self.row['task']['tasks_sha256']=recovery.sha(self.tasks)
        load,check=self.guards()
        copied=recovery.private_measure(recovery.backend.measure,load,check,fcntl)
        binding=dict(config_sha256='config',dataset_sha256='dataset',
            dataset_manifest_sha256='dataset_manifest',reward_scale=10.,physical_reference={})
        copied.__globals__.update(context=lambda exp:({'design':'bp_fe','physical_failure_reward':-4},None,binding),
            checked_existing=lambda *args:None,archive_measurement=lambda *args:{'status':'retained_test'},
            subprocess=type('NoProcess',(),{'run':lambda *args,**kwargs:self.fail('physical rerun forbidden')})())
        return copied

    def test_completed_new_native_attempt_reuses_frozen_recovery_no_process(self):
        copied=self.prepared_measure()
        complete=self.sample/'backend_attempts/new_complete';complete.mkdir()
        run=self.root/'finished_native';run.mkdir()
        (run/'manifest.json').write_text('{"time_end":1,"success":true}')
        (complete/'attempt.json').write_text(json.dumps(dict(physical_run=str(run),
            request_sha256=self.row['request_sha256'],config_sha256='config',job_id='333',restart_count=0)))
        old_sha=recovery.sha(self.old/'attempt.json')
        calls=[]
        copied.__globals__['verify_run']=lambda *args:(calls.append(args[0]) or dict(status='success',
            metrics={'rWL_um':20.},physical_run=str(run),physical_manifest_sha256='native'))
        with patch.dict(os.environ,{'SLURM_JOB_ID':'222','SLURM_RESTART_COUNT':'0'}):
            copied(argparse.Namespace(experiment=str(self.root),seed=0,round=0,index=0,storage='scratch'))
        outcome=recovery.read(self.sample/'outcome.json')
        self.assertEqual(calls,[run])
        self.assertEqual(outcome['reward'],-2.)
        self.assertEqual(outcome['recovered_completed_attempt'],str(complete/'attempt.json'))
        self.assertEqual(recovery.sha(self.old/'attempt.json'),old_sha)

    def test_new_unfinished_failed_attempt_retains_frozen_refusal(self):
        copied=self.prepared_measure()
        later=self.sample/'backend_attempts/new_failed';later.mkdir()
        run=self.root/'unfinished_native';run.mkdir()
        (run/'manifest.json').write_text('{"success":false}')
        (later/'attempt.json').write_text(json.dumps(dict(physical_run=str(run),
            request_sha256=self.row['request_sha256'],config_sha256='config',job_id='333',restart_count=0)))
        copied.__globals__['states']=lambda job:['FAILED']
        with patch.dict(os.environ,{'SLURM_JOB_ID':'222','SLURM_RESTART_COUNT':'0'}):
            with self.assertRaisesRegex(RuntimeError,'Do not duplicate'):
                copied(argparse.Namespace(experiment=str(self.root),seed=0,round=0,index=0,storage='scratch'))
        self.assertFalse((self.sample/'outcome.json').exists())

    def test_absent_and_present_evidence_cannot_be_changed(self):
        p=self.root/'unknown';absent=recovery.fingerprint(p)
        p.write_text('old')
        with self.assertRaisesRegex(ValueError,'presence changed'):
            recovery.check_fingerprint(absent)
        present=recovery.fingerprint(p);p.write_text('new')
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):
            recovery.check_fingerprint(present)

    def test_scheduler_queries_exact_array_element_not_raw_parent(self):
        with patch.object(recovery.backend,'states',return_value=['FAILED']) as states:
            self.assertEqual(recovery.original_states(dict(array='1797375',flat_index=47)),['FAILED'])
        states.assert_called_once_with('1797375_47')

    def test_readonly_real_diagnosis_exact_81_and_six_native_attempts(self):
        value=recovery.diagnosis()
        self.assertEqual(len(value['rows']),81)
        self.assertEqual(sum(a['attempt_json_exists'] for r in value['rows'] for a in r['attempts']),6)
        self.assertFalse(any(r['outcome_exists'] for r in value['rows']))


if __name__ == '__main__':
    unittest.main()
