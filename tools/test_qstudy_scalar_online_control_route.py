"""Route glue contracts; no model execution or physical calls."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).with_name('qstudy_scalar_online_control_route.py')
spec = importlib.util.spec_from_file_location('_route_contract_test', SOURCE)
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)
BASE_MANIFEST = Path('/scratch/bohanli/flowplace_graph/qstudy_scalar_online_control_v1/manifest.json')


class Contracts(unittest.TestCase):
    def setUp(self):
        self.old = json.loads(BASE_MANIFEST.read_text())
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.folder = Path(self.temp.name)
        self.adapter = dict(output=str(self.folder), selection='fixed', base_manifest=dict(path='base', sha256='base'))

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_fixed72_and_two_roots(self):
        with patch.object(route, 'original', return_value=self.old):
            tasks = route.phase_tasks(self.adapter, 'route')
        expected = [dict(t, run_id='go2o_'+t['run_id']) for t in self.old['physical_drt_requests']]
        self.assertEqual([t['task'] for t in tasks], expected)
        self.assertEqual(len(tasks), 72)
        self.assertEqual(len({t['task']['output'] for t in tasks}), 72)
        self.assertEqual(sum(t['task']['round'] == 0 for t in tasks), 24)
        self.assertEqual({t['task']['group'] for t in tasks if t['task']['round'] == 0}, {'bc_t0', 'fql_t0'})
        self.assertEqual({t['task']['noise_index'] for t in tasks}, {0, 1, 2, 3})
        self.assertFalse(any(t['task']['round'] == 1 for t in tasks))

    def test_plan_native_contract_and_budget(self):
        with patch.object(route, 'original', return_value=self.old), patch.object(route, 'bound', return_value=dict(path='route', sha256='digest')):
            plan = route.route_plan(self.adapter)
        expected = [dict(t, run_id='go2o_'+t['run_id']) for t in self.old['physical_drt_requests']]
        self.assertEqual(plan['tasks'], expected)
        self.assertEqual(plan['original_tasks_sha256'], route.task_inventory_sha256(self.old))
        self.assertEqual(plan['run_id_translation'], route.RUN_ID_TRANSLATION)
        self.assertEqual((plan['requested'], plan['detailed_route_end_iteration'], plan['backend_threads']), (72, 20, 8))
        self.assertEqual(plan['backend_storage'], 'scratch')
        self.assertIs(plan['training_feedback_eligible'], False)
        self.assertIs(plan['fresh_confirmation'], False)

    def test_private_dispatch_does_not_mutate_base(self):
        self.assertEqual(route.base.PHASES, ('prepare0', 'export0', 'native0'))
        self.assertEqual(route.dispatch['PHASES'], ('prepare', 'route'))
        self.assertIs(route.dispatch['resources'], route.resources)
        self.assertIs(route.dispatch['checked_receipt'].__globals__, route.dispatch)
        self.assertIsNot(route.dispatch['checked_receipt'], route.base.checked_receipt)
        options, concurrency = route.resources('route')
        self.assertEqual(concurrency, 2)
        self.assertIn('--cpus-per-task=8', options)
        self.assertIn('--time=02:30:00', options)

    def source_fixture(self):
        task = self.old['physical_drt_requests'][0]
        grt = next(r for r in self.old['physical_grt_requests'] if r['physical_key'] == task['group']
                   and r['seed'] == task['seed'] and r['role'] == 'evaluation' and r['index'] == task['noise_index'])
        sample = self.folder/'sample'
        sample.mkdir()
        task = dict(task, source_outcome=str(sample/'outcome.json'))
        old = copy.deepcopy(self.old)
        old['physical_drt_requests'][0] = task
        task = dict(task, run_id='go2o_'+task['run_id'])
        checkpoint = self.folder/'actor.pt'
        checkpoint.write_bytes(b'checkpoint fixture; not a torch model')
        checkpoint_binding = dict(path=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        request = dict(evaluation_purpose='development_validation', training_feedback_eligible=False,
                       source_checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_binding['sha256'],
                       raw_action_sha256='raw', action_sha256='executed', noise_sha256='noise', noise_seed=81400000)
        for name in ('outcome.json', 'request.json', 'tasks.jsonl', 'completed.json'):
            (sample/name).write_text('{}')
        array = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
        np.savez(sample/'actions.npz', noise=array, raw_action=array, action=array)
        physical = dict(config_sha256='config', dataset_sha256='dataset', physical_flow_revision='fixed')
        outcome = dict(status='success', metrics=dict(rWL_um=123.))
        plan = dict(base_manifest=dict(path='base', sha256='base'))
        return old, task, request, outcome, sample, physical, plan, grt

    def source_call(self, values):
        old, task, request, outcome, sample, physical, plan, _ = values
        import fql.qstudy_actor_report as report
        # The original export/native validators are integration-tested elsewhere;
        # this fixture tests only this adapter's added namespace and sealing glue.
        with patch.object(route.base, 'checked_manifest', return_value=old), \
             patch.object(route.base, 'checked_export', return_value=({}, {}, {}, [], sample/'tasks.jsonl')), \
             patch.object(route.base, 'inspect_native', return_value=(request, outcome, sample/'tasks.jsonl', sample, physical)), \
             patch.object(report, 'native_fingerprints', return_value={'physical_run': 'native'}), \
             patch.object(route, 'bound', wraps=route.bound) as guard:
            result = route.inspect_source(plan, task, {})
        return result, guard

    def test_source_seals_exact_checkpoint_and_arrays(self):
        values = self.source_fixture()
        result, _ = self.source_call(values)
        self.assertEqual(result['actor']['checkpoint'], dict(path=values[2]['source_checkpoint'], sha256=values[2]['checkpoint_sha256']))
        self.assertEqual(result['noise_sha256'], 'noise')
        self.assertEqual(result['native']['physical_run'], 'native')
        self.assertEqual(result['exact_coordinates_and_noise']['action'], [[float(np.float32(.1)), float(np.float32(.2))], [float(np.float32(.3)), float(np.float32(.4))]])

    def test_changed_checkpoint_is_rejected(self):
        values = self.source_fixture()
        Path(values[2]['source_checkpoint']).write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'Changed bound artifact'):
            self.source_call(values)

    def test_collect_alias_cannot_be_source(self):
        values = self.source_fixture()
        values[2]['training_feedback_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'declared evaluation'):
            self.source_call(values)

    def test_unplanned_source_rejected_before_validation(self):
        task = dict(self.old['physical_drt_requests'][0], run_id='changed')
        with patch.object(route.base, 'checked_manifest', return_value=self.old), \
             patch.object(route.base, 'checked_export') as validate:
            with self.assertRaisesRegex(ValueError, 'predeclared physical'):
                route.inspect_source(dict(base_manifest=dict(path='base', sha256='base')), task, {})
            validate.assert_not_called()

    def test_noise_mismatch_invalidates_both_sources(self):
        from fql import qstudy_final_route as native
        old_inspector = native.inspect_source
        tasks = [dict(key='a', noise_index=0), dict(key='b', noise_index=0)]
        def fake(plan, task, cache):
            return dict(noise_index=0, noise_sha256=task['key'])
        with patch.object(route, 'inspect_source', side_effect=fake):
            records = route.inspect_sources(self.folder, dict(tasks=tasks), seal=True)
        self.assertEqual([r['state'] for r in records], ['invalid', 'invalid'])
        self.assertIs(native.inspect_source, old_inspector)
        self.assertFalse((self.folder/'source_bindings').exists())

    def test_source_missing_is_not_physical_failure(self):
        with patch.object(route, 'inspect_source', side_effect=FileNotFoundError('not measured')):
            records = route.inspect_sources(self.folder, dict(tasks=[dict(key='a', noise_index=0)]), seal=True)
        self.assertEqual(records[0]['state'], 'missing')
        self.assertIsNone(records[0]['binding'])


if __name__ == '__main__':
    unittest.main()
