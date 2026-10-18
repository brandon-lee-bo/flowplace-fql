"""Bounded runner/gradient/resume contracts, without a graph or EDA invocation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest

import torch

from . import qstudy_scalar_online as core
from . import qstudy_scalar_online_train as runner
from .test_qstudy_scalar_online import ScalarOnlineTests


class ScalarRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.slurm()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        # Reuse the already independently tested tiny scalar/Adam prefix. These
        # fixtures never invoke the production source loader or backend.
        self.fixture = ScalarOnlineTests(methodName='test_prefix_optimizer_rng_and_named_states_restored_exactly')
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.binding = dict(version=runner.VERSION,request=dict(path='fixture',sha256='a'*64))
        runner.STOP=False

    def test_gradient_probe_is_finite_and_preserves_full_prefix_and_existing_gradients(self):
        state=self.fixture.core()
        self.fixture.update(state)
        before=state.state()
        gradients={name:[None if p.grad is None else p.grad.clone() for p in model.parameters()]
            for name,model in state.models.items()}
        result=runner.gradient_probe(state)
        self.assertFalse(result['consumes_training_rng'])
        self.assertEqual(result['phase'],'after_update_peek_next_noise')
        self.assertGreater(result['q_ascent_grad_norm'],0.)
        core.equal_tree(before,state.state())
        for name,model in state.models.items():
            core.equal_tree(gradients[name],[p.grad for p in model.parameters()])

    def test_zero_q_probe_reports_zero_pressure_without_updating_the_critic(self):
        settings=dict(self.fixture.settings,q_weight=0.)
        state=self.fixture.core(settings=settings)
        before=copy.deepcopy(state.models['critic'].state_dict())
        self.fixture.update(state,settings)
        result=runner.gradient_probe(state)
        self.assertEqual(result['q_ascent_grad_norm'],0.)
        self.assertEqual(result['q_to_distill_grad_ratio'],0.)
        self.assertEqual(result['weighted_q'],0.)
        core.equal_tree(before,state.models['critic'].state_dict())

    def test_atomic_partial_restart_matches_uninterrupted_models_adam_and_rng(self):
        full=self.fixture.core()
        runner.run_loop(self.root/'full',full,self.fixture.offline,self.fixture.replay,
            self.binding,diagnostic_interval=2)
        partial=self.fixture.core()
        history=[]
        for _ in range(2):
            row=self.fixture.update(partial)
            if row['step']==1 or row['step']%2==0:
                row['gradient_diagnostics']=runner.gradient_probe(partial)
            history.append(row)
        folder=self.root/'partial';folder.mkdir()
        runner.save_state(folder,partial,self.binding,history)
        payload=runner.checked_resume(folder,self.binding,partial.binding)
        resumed=self.fixture.core(payload=payload,resume=True)
        runner.run_loop(folder,resumed,self.fixture.offline,self.fixture.replay,
            self.binding,payload['history'],diagnostic_interval=2)
        core.equal_tree(full.state(),resumed.state())
        self.assertEqual(json.loads((self.root/'full/learning_curve.json').read_text()),
            json.loads((folder/'learning_curve.json').read_text()))
        done=json.loads((folder/'completed.json').read_text())
        self.assertEqual(done['steps'],4)
        self.assertEqual(done['actor_updates'],4)
        self.assertEqual(done['critic_updates'],4)
        self.assertEqual(done['teacher_updates'],0)
        self.assertEqual(done['backend_calls'],0)
        self.assertFalse((folder/'resume.pt').exists())

    def test_resume_rejects_changed_inputs_or_incomplete_history(self):
        state=self.fixture.core();folder=self.root/'partial';folder.mkdir()
        row=self.fixture.update(state)
        runner.save_state(folder,state,self.binding,[row])
        with self.assertRaisesRegex(ValueError,'inputs, settings'):
            runner.checked_resume(folder,dict(self.binding,changed=True),state.binding)
        payload=torch.load(folder/'resume.pt',weights_only=False)
        payload['history']=[];torch.save(payload,folder/'resume.pt')
        with self.assertRaisesRegex(ValueError,'complete step history'):
            runner.checked_resume(folder,self.binding,state.binding)

    def test_completed_reentry_rejects_changed_marker_status_counts_and_units(self):
        state=self.fixture.core();folder=self.root/'finished'
        runner.run_loop(folder,state,self.fixture.offline,self.fixture.replay,
            self.binding,diagnostic_interval=2)
        payload=runner.checked_resume(folder,self.binding,state.binding)
        runner.checked_completion(folder,payload,self.binding,state.binding)
        path=folder/'completed.json';original=json.loads(path.read_text())
        for key,value in (('status','partial'),('steps',3),('parent_steps',0),
                ('teacher_updates',1),('critic_updates',0),('q_units','legacy normalized GRT')):
            changed=dict(original);changed[key]=value;path.write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'marker/checkpoint'):
                runner.checked_completion(folder,payload,self.binding,state.binding)

    def test_stop_saves_exact_current_update_and_no_completion(self):
        state=self.fixture.core();folder=self.root/'stopped'
        runner.STOP=True
        with self.assertRaises(SystemExit) as stopped:
            runner.run_loop(folder,state,self.fixture.offline,self.fixture.replay,
                self.binding,diagnostic_interval=2)
        self.assertEqual(stopped.exception.code,85)
        payload=runner.checked_resume(folder,self.binding,state.binding)
        self.assertEqual(payload['step'],1)
        self.assertFalse((folder/'completed.json').exists())
        self.assertEqual(len(payload['history']),1)

    def test_offline_labels_use_the_original_full_train_schema_including_b4(self):
        source=copy.deepcopy(self.fixture.source)
        source['refit']=dict(context=dict(x=self.fixture.x[:2],
            cost_values=[[10.]*6,[float('nan')]*6],cost_mask=[[True]*6,[False]*6],
            statuses=['success','physical_failure']))
        actual=runner.offline_labels(source)
        self.assertEqual(actual['derived_B'].tolist(),[[1.],[4.]])
        self.assertEqual(actual['utility_spec_sha256'],source['schema']['utility_spec_sha256'])
        source['refit']['context']['statuses'][1]='infrastructure_failure'
        with self.assertRaisesRegex(ValueError,'wholly admitted'):runner.offline_labels(source)

    def test_request_cannot_silently_choose_settings_or_write_inside_inputs(self):
        value=dict(version=runner.VERSION,experiment=str(self.root/'physical'),config_sha256='a'*64,
            declaration_sha256='b'*64,seed=0,tasks_sha256='c'*64,final_outcomes={},
            settings=self.fixture.settings,diagnostic_interval=100,output=str(self.root/'trained'))
        self.assertEqual(runner.request_settings(value),value)
        for change in ('missing','nested','fractional_seed','implicit_final','negative_interval'):
            bad=copy.deepcopy(value)
            if change=='missing':del bad['settings']
            elif change=='nested':bad['output']=str(self.root/'physical'/'child')
            elif change=='fractional_seed':bad['seed']=.5
            elif change=='implicit_final':bad['final_outcomes']={'01':{'path':'relative','sha256':'d'*64}}
            else:bad['diagnostic_interval']=0
            with self.subTest(change=change),self.assertRaises(ValueError):runner.request_settings(bad)


if __name__=='__main__':unittest.main()
