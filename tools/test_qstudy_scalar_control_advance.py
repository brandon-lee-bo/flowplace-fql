"""Pure scheduler/receipt fixtures: no model, array, EDA or real sbatch."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).with_name('qstudy_scalar_control_advance.py')
spec = importlib.util.spec_from_file_location('_advance_contracts', SOURCE)
advance = importlib.util.module_from_spec(spec); spec.loader.exec_module(advance)


def fixture():
    slots, jobs, items = {}, {}, []
    base = SimpleNamespace(LIVE={'PENDING', 'RUNNING'}, adopt=Mock())
    base.scheduler_states = lambda job: (jobs.get(job, ['ACCOUNTING_PENDING']), {'job': job})
    definitions = [('base', ('prepare0', 'export0', 'native0')),
                   ('continuation', ('train1', 'gate1', 'prepare1', 'export1', 'native1',
                                     'train2', 'prepare2', 'export2', 'native2')),
                   ('route', ('prepare', 'route'))]
    for name, phases in definitions:
        module = base if name == 'base' else SimpleNamespace()
        module.PHASES = phases
        plan = {'name': name}
        for phase in phases:
            job = name+'_'+phase
            slots[name+':'+phase] = {'missing': [], 'attempts': [{'job': job}]}
            jobs[job] = ['COMPLETED']
        dispatch = {
            'missing_indices': lambda p, ph: slots[p['name']+':'+ph]['missing'],
            'stage_state': lambda p, ph: (slots[p['name']+':'+ph], None)}
        module.stage_submit = Mock()
        items.append((name, module, plan, dispatch, Path(name+'.json'), name+'_digest'))
    return items, slots, jobs


class AdvanceContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.output = Path(self.temp.name)
        self.output_patch = patch.object(advance, 'OUTPUT', self.output)
        self.output_patch.start()
        (self.output/'admission.json').write_text('{}')
        self.digest = advance.sha(self.output/'admission.json')

    def tearDown(self):
        self.output_patch.stop(); self.temp.cleanup()

    def test_base_preemption_recovers_before_anchor_or_train1(self):
        items, slots, jobs = fixture()
        row = slots['base:native0']; row['missing'] = [7]
        jobs[row['attempts'][-1]['job']] = ['PREEMPTED']
        def recover(args):
            self.assertEqual(args.phase, 'native0')
            row['attempts'].append({'job': 'new_native0'})
            jobs['new_native0'] = ['RUNNING']
        items[0][1].stage_submit.side_effect = recover
        anchor = Mock(side_effect=AssertionError('must not inspect anchor yet'))
        result = advance.advance_once(items, {}, lambda: None, anchor=anchor)
        self.assertEqual(result['job'], 'new_native0')
        items[0][1].stage_submit.assert_called_once()
        items[1][1].stage_submit.assert_not_called()

    def test_live_array_even_with_all_receipts_is_awaited(self):
        items, slots, jobs = fixture()
        jobs['base_native0'] = ['RUNNING']
        result = advance.advance_once(items, {}, lambda: None)
        self.assertEqual(result, dict(kind='wait', job='base_native0', accounting=False, stage='base:native0'))
        for item in items: item[1].stage_submit.assert_not_called()

    def test_actual_train1_submitter_wait_and_accounting_delay(self):
        items, slots, jobs = fixture()
        slots['continuation:train1']['missing'] = [0]
        state = {}
        result = advance.advance_once(items, state, lambda: None,
            anchor=lambda b: dict(states=['PENDING'], obsolete=False))
        self.assertEqual(result['job'], advance.ANCHOR)
        items[1][1].stage_submit.assert_not_called()
        jobs['base_native0'] = ['ACCOUNTING_PENDING']
        result = advance.advance_once(items, {}, lambda: None)
        self.assertTrue(result['accounting'])

    def test_unknown_failure_is_not_recovered_or_labeled(self):
        items, slots, jobs = fixture()
        slots['continuation:train1']['missing'] = [2]
        jobs['continuation_train1'] = ['FAILED']
        items[1][1].stage_submit.side_effect = RuntimeError('requires diagnosis: FAILED')
        with self.assertRaisesRegex(RuntimeError, 'requires diagnosis'):
            advance.advance_once(items, {'anchor_closed': True}, lambda: None)
        items[1][1].stage_submit.assert_called_once()
        items[2][1].stage_submit.assert_not_called()

    def test_original_ambiguous_intent_uses_delayed_retry_not_resubmit(self):
        items, slots, jobs = fixture()
        row = slots['continuation:train1']; row.update(missing=[0], attempts=[{'job': None}])
        items[1][1].stage_submit.side_effect = RuntimeError('unresolved original intent')
        result = advance.advance_once(items, {'anchor_closed': True}, lambda: None)
        self.assertEqual(result['job'], None)
        self.assertTrue(result['accounting'])

    def test_callback_intent_precedes_sbatch_and_strips_restart(self):
        state = {'callbacks': [], 'consecutive_delays': 0}
        snapshots = []
        def submit(command, **kwargs):
            self.assertTrue((self.output/'intents/0.json').is_file())
            self.assertEqual(len(snapshots[-1]['callbacks']), 1)
            self.assertNotIn('SLURM_RESTART_COUNT', kwargs['env'])
            self.assertIn('--dependency=afterany:123', command)
            return SimpleNamespace(returncode=0, stdout='456\n', stderr='')
        with patch.dict(os.environ, {'SLURM_RESTART_COUNT': '9'}), \
             patch.object(advance.subprocess, 'run', side_effect=submit):
            job = advance.schedule(state, self.digest, {'kind': 'wait', 'job': '123', 'accounting': False},
                lambda: snapshots.append(copy.deepcopy(state)), None)
        self.assertEqual(job, '456')
        self.assertEqual(advance.read(self.output/'intents/0.json')['job'], None)

    def test_rejected_dependency_race_delays_but_unknown_response_does_not_repeat(self):
        state = {'callbacks': [], 'consecutive_delays': 0}
        calls = [SimpleNamespace(returncode=1, stdout='', stderr='Job dependency problem'),
                 SimpleNamespace(returncode=0, stdout='789\n', stderr='')]
        with patch.object(advance.subprocess, 'run', side_effect=calls) as submit:
            advance.schedule(state, self.digest, {'job': '123', 'accounting': False}, lambda: None, None)
        self.assertEqual(submit.call_count, 2)
        self.assertIn('--begin=now+2minutes', state['callbacks'][1]['command'])
        self.assertNotIn('--dependency=afterany:123', state['callbacks'][1]['command'])
        self.assertTrue(state['callbacks'][0]['rejected'])

    def test_six_consecutive_accounting_delays_are_bounded(self):
        state = {'callbacks': [], 'consecutive_delays': 6}
        with patch.object(advance.subprocess, 'run') as submit:
            with self.assertRaisesRegex(RuntimeError, 'six delayed'):
                advance.schedule(state, self.digest, {'job': None, 'accounting': True}, lambda: None, None)
            submit.assert_not_called()

    def test_completed_all14_has_no_new_submission(self):
        items, _, _ = fixture()
        result = advance.advance_once(items, {'anchor_closed': True}, lambda: None)
        self.assertEqual(result['kind'], 'complete')
        for item in items: item[1].stage_submit.assert_not_called()

    def test_obsolete443_requires_exact_identity_and_terminal_failed_original(self):
        items, _, jobs = fixture(); base = items[0][1]
        jobs[advance.ANCHOR] = ['PENDING']; jobs['1800396'] = ['COMPLETED', 'PREEMPTED']
        text = ('JobId=1800443 JobName=soc_continue_after_native0 UserId=tester(1) '
                'JobState=PENDING Reason=DependencyNeverSatisfied Dependency=afterok:1800396 '
                'Command='+str(advance.ROOT/'tools/qstudy_scalar_online_control_continue.slurm'))
        with patch.dict(os.environ, {'USER': 'tester'}), patch.object(advance.subprocess, 'run',
                return_value=SimpleNamespace(stdout=text)), patch.object(advance, 'anchor_bindings', return_value=['pinned declaration']):
            self.assertTrue(advance.anchor_view(base)['obsolete'])
            jobs['1800396'] = ['RUNNING']
            self.assertFalse(advance.anchor_view(base)['obsolete'])

    def test_aged_out_squeue_uses_exact_sacct_children_without_state_fakery(self):
        error = advance.subprocess.CalledProcessError(1, ['squeue', '-j', '123', '-h'],
            stderr='slurm_load_jobs error: Invalid job id specified\n')
        original = Mock(side_effect=error)
        raw = ('123_0|900|COMPLETED|0:0|job\n'
               '123_1|901|PREEMPTED|0:0|job\n')
        with patch.object(advance.subprocess, 'run', return_value=SimpleNamespace(stdout=raw)):
            states, evidence = advance.scheduler_states('123', original)
        self.assertEqual(states, ['COMPLETED', 'PREEMPTED'])
        self.assertEqual(evidence['child_states'], {'0': ['COMPLETED'], '1': ['PREEMPTED']})
        with patch.object(advance.subprocess, 'run', return_value=SimpleNamespace(stdout='')):
            self.assertEqual(advance.scheduler_states('123', original)[0], ['ACCOUNTING_PENDING'])

    def test_other_scheduler_error_stays_an_error(self):
        error = advance.subprocess.CalledProcessError(1, ['squeue', '-j', '123'], stderr='Connection refused')
        with patch.object(advance.subprocess, 'run') as fallback:
            with self.assertRaises(advance.subprocess.CalledProcessError):
                advance.scheduler_states('123', Mock(side_effect=error))
            fallback.assert_not_called()

    def test_private_scheduler_reaches_both_submitters_without_global_mutation(self):
        def body(args):
            return scheduler_states('old')
        raw = Mock(side_effect=RuntimeError('old global must remain untouched'))
        safe = Mock(return_value=(['COMPLETED'], {}))
        namespace = {'scheduler_states': raw}
        for name in ('checked_receipt', 'stage_state', 'missing_indices', 'stage_submit', 'status'):
            namespace[name] = FunctionType(body.__code__, namespace, name)
        module = SimpleNamespace(**namespace)
        proxy = advance.private_driver(module, safe)
        self.assertEqual(proxy.stage_submit(None)[0], ['COMPLETED'])
        self.assertIs(module.stage_submit.__globals__['scheduler_states'], raw)
        # The continuation has an outer submitter and inner base-derived dispatch.
        def outer(args):
            base.scheduler_states('prior')
            return dispatch['stage_submit'](args)
        globals_outer = {'base': module, 'dispatch': namespace}
        child = SimpleNamespace(dispatch=namespace, stage_submit=FunctionType(outer.__code__, globals_outer), PHASES=('train1',))
        child_proxy = advance.private_driver(child, safe, proxy)
        self.assertEqual(child_proxy.stage_submit(None)[0], ['COMPLETED'])
        raw.assert_not_called()
        self.assertEqual(safe.call_args_list[-2][0], ('prior',))
        self.assertEqual(safe.call_args_list[-1][0], ('old',))

    def test_existing_live_callback_is_adopted_without_stage_submission(self):
        items, _, jobs = fixture(); jobs['456'] = ['PENDING']
        intent = advance.callback_intent(0, self.digest, {'job': '123', 'accounting': False})
        advance.immutable(self.output/'intents/0.json', intent)
        state = dict(version=advance.VERSION, admission=advance.bound(self.output/'admission.json'),
            callbacks=[dict(intent, intent=advance.bound(self.output/'intents/0.json'), job='456')],
            consecutive_delays=0, visits=[])
        advance.atomic(self.output/'state.json', state)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}), \
             patch.object(advance, 'checked_admission'), patch.object(advance, 'drivers', return_value=items), \
             patch.object(advance, 'advance_once') as step:
            advance.run(self.digest)
            step.assert_not_called()

    def test_orphan_callback_intent_is_adopted_without_second_sbatch(self):
        items, _, jobs = fixture(); base = items[0][1]
        base.adopt.return_value = ('456', {'exact_name': True}); jobs['456'] = ['PENDING']
        intent = advance.callback_intent(0, self.digest, {'job': '123', 'accounting': False})
        advance.immutable(self.output/'intents/0.json', intent)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}), \
             patch.object(advance, 'checked_admission'), patch.object(advance, 'drivers', return_value=items), \
             patch.object(advance.subprocess, 'run') as submit:
            advance.run(self.digest)
            submit.assert_not_called()
        saved = advance.read(self.output/'state.json')
        self.assertEqual(saved['callbacks'][0]['job'], '456')


if __name__ == '__main__':
    unittest.main()
