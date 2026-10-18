"""Synthetic genuine-BC-prefix contracts; no production graph or EDA call."""
import copy
import json
import unittest

import torch

from . import qstudy_scalar_online as core
from . import qstudy_scalar_online_train as common
from . import qstudy_scalar_online_bc_train as runner
from .test_qstudy_scalar_online_bc import BCSourceTests


class BCTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BCSourceTests.setUpClass()

    def setUp(self):
        self.fixture = BCSourceTests(methodName='test_real_bc_identity_and_all_preserved_states_with_two_source_binding')
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.source = self.fixture.load()
        self.settings = dict(total_updates=4, batch_size=4, alpha=10., q_weight=0.,
            flow_steps=10, update_critic=False, huber_delta=1., grad_clip=1.)
        self.runner_binding = dict(version=runner.VERSION, request={'path':'fixture','sha256':'a'*64})
        runner.STOP = False

    def state(self, payload=None):
        initial, replay = self.fixture.make_core(self.source, self.settings)
        if payload is not None:
            initial = core.Continuation(initial.models, initial.opts, initial.generators,
                payload, initial.binding, frozen_reference=self.source, resume=True)
        return initial, replay

    def test_zero_q_loop_matches_independent_bc_and_preserves_critic_and_true_prefix(self):
        state, replay = self.state()
        before = copy.deepcopy(self.source['payload'])
        reference = copy.deepcopy(self.fixture.actor)
        teacher = copy.deepcopy(self.fixture.flow)
        opt = torch.optim.Adam(reference.parameters(), lr=1e-5)
        opt.load_state_dict(copy.deepcopy(before['optimizers']['actor']))
        gen = torch.Generator(); gen.set_state(before['generators']['actor'])
        for _ in range(4):
            noise = 2*torch.rand(4,3,2,generator=gen)-1
            with torch.no_grad(): target = teacher.sample_flow(noise,10)
            loss = 10*(reference(noise,clip=False)[:,reference.movable]-target[:,reference.movable]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True); opt.step()
        folder = self.root/'finished'
        runner.run_loop(folder,state,self.fixture.offline,replay,self.runner_binding,diagnostic_interval=2)
        core.equal_tree(state.models['actor'].state_dict(),reference.state_dict())
        core.equal_tree(state.opts['actor'].state_dict(),opt.state_dict())
        core.equal_tree(state.generators['actor'].get_state(),gen.get_state())
        core.equal_tree(state.models['critic'].state_dict(),before['critic'])
        core.equal_tree(state.opts['critic'].state_dict(),before['optimizers']['critic'])
        core.equal_tree(before,self.source['payload'])
        done = common.read(folder/'completed.json')
        self.assertEqual(done['version'],runner.VERSION)
        self.assertEqual(done['parent_steps'],1000)
        self.assertEqual(done['actor_updates'],4)
        self.assertEqual(done['critic_updates'],0)
        for row in common.read(folder/'learning_curve.json'):
            if 'gradient_diagnostics' in row:
                self.assertEqual(row['gradient_diagnostics']['q_ascent_grad_norm'],0.)

    def test_signal_resume_preserves_full_adam_rng_history_and_distinct_bc_marker(self):
        full,replay = self.state()
        runner.run_loop(self.root/'full',full,self.fixture.offline,replay,self.runner_binding,diagnostic_interval=2)
        interrupted,replay = self.state();folder=self.root/'resumed'
        runner.STOP=True
        with self.assertRaises(SystemExit) as stopped:
            runner.run_loop(folder,interrupted,self.fixture.offline,replay,self.runner_binding,diagnostic_interval=2)
        self.assertEqual(stopped.exception.code,85)
        self.assertFalse((folder/'completed.json').exists())
        payload=common.checked_resume(folder,self.runner_binding,interrupted.binding)
        self.assertEqual(payload['step'],1)
        resumed,replay=self.state(payload)
        runner.STOP=False
        runner.run_loop(folder,resumed,self.fixture.offline,replay,self.runner_binding,
            payload['history'],diagnostic_interval=2)
        core.equal_tree(full.state(),resumed.state())
        self.assertEqual(common.read(self.root/'full/learning_curve.json'),common.read(folder/'learning_curve.json'))
        payload=common.checked_resume(folder,self.runner_binding,resumed.binding)
        runner.checked_completion(folder,payload,self.runner_binding,resumed.binding)
        original=common.read(folder/'completed.json')
        for key,value in (('version',common.VERSION),('critic_updates',1),('parent_steps',0),('q_units','legacy')):
            changed=dict(original,**{key:value});(folder/'completed.json').write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.checked_completion(folder,payload,self.runner_binding,resumed.binding)

    def test_request_rejects_non_bc_loss_and_mutated_common_request_body(self):
        request=dict(version=runner.VERSION,experiment=str(self.root/'physical'),
            config_sha256='a'*64,declaration_sha256='b'*64,seed=0,tasks_sha256='c'*64,
            final_outcomes={},settings=self.settings,diagnostic_interval=2,output=str(self.root/'trained'))
        self.assertIs(runner.request_settings(request),request)
        for item in ('q','critic','version','nested','implicit_setting','fractional_seed'):
            bad=copy.deepcopy(request)
            if item=='q':bad['settings']['q_weight']=1.
            elif item=='critic':bad['settings']['update_critic']=True
            elif item=='version':bad['version']=common.VERSION
            elif item=='nested':bad['output']=str(self.root/'physical/child')
            elif item=='implicit_setting':del bad['settings']
            else:bad['seed']=.5
            with self.subTest(item=item),self.assertRaises(ValueError):runner.request_settings(bad)

    def test_loop_cannot_silently_train_q_under_a_bc_completion_marker(self):
        state,replay=self.fixture.make_core(self.source,dict(self.settings,q_weight=1.))
        with self.assertRaisesRegex(ValueError,'must not update or optimize'):
            runner.run_loop(self.root/'bad',state,self.fixture.offline,replay,
                self.runner_binding,diagnostic_interval=2)
        self.assertFalse((self.root/'bad').exists())


    def test_bc_resume_rejects_changed_frozen_weights_optimizer_and_unused_generators(self):
        state,replay=self.state()
        row=state.update(self.fixture.offline,replay,
            **{k:v for k,v in self.settings.items() if k!='total_updates'})
        original=state.state()
        self.assertEqual(row['critic_updates'],0)
        runner.checked_frozen_source(original,self.source)
        for change in ('weights','optimizer','teacher','q_rng','flow_rng'):
            payload=copy.deepcopy(original)
            if change=='weights':
                key=next(name for name,_ in state.models['critic'].named_parameters())
                payload['critic'][key].flatten()[0]+=.1
            elif change=='optimizer':payload['optimizers']['critic']['state']={0:{'step':torch.tensor(1.)}}
            elif change=='teacher':payload['optimizers']['flow']['param_groups'][0]['lr']=1.
            else:payload['generators'][change[:-4]][0]^=1
            with self.subTest(change=change),self.assertRaises(ValueError):
                runner.checked_frozen_source(payload,self.source)


if __name__ == '__main__':
    unittest.main()
