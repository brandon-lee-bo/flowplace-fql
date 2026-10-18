"""Bounded CPU contracts; these fixtures do not evaluate a physical design."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import qstudy_scalar_online as online
from . import qstudy_backend as backend
from .qstudy_data import action_sha, sha
from .qstudy_models import MultiMetricCritic, frozen_critic


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 6)
        self.register_buffer('fixed', torch.zeros(3, 2))
        self.register_buffer('movable', torch.tensor([True, True, False]))

    def restore(self, x, clip=False):
        return torch.where(self.movable[None,:,None], x.clamp(-1,1) if clip else x, self.fixed[None])

    def forward(self, x, clip=True):
        return self.restore(self.linear(x.flatten(1)).reshape_as(x), clip)

    def sample_flow(self, noise, steps):
        return self(noise)


class ScalarOnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        online.slurm()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='scalar-online-', dir=os.environ['TMPDIR'])
        self.root = Path(self.temp.name)
        self.cache = dict(node_id_to_name=['a','b','c'], ratio_x=10/224, ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.) for n in 'abc'},
            net_info={'n':{'nodes':{n:{} for n in 'abc'}}})
        self.x = torch.rand(12,3,2,generator=torch.Generator().manual_seed(19))*.8-.4
        self.x[:,2] = 0
        rule = online.screen_for('rule')
        spec = dict(version=rule.SPEC_VERSION, objective='grt_wl',cost_names=list(rule.COST_NAMES),
            weights=list(rule.OBJECTIVES['grt_wl']),scales=[10.]*6,
            reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64),
            rule_definition_sha256=rule.json_sha(rule.rule_definition()),success_cap=2.,failure_cost=4.)
        self.schema = online.schema('rule',spec)
        self.costs = np.tile([10.,1.,1.,1.,1.,1.],(12,1))
        self.costs[:,0] += np.arange(12)
        self.offline = dict(x=self.x, **online.targets(self.schema,self.costs,['success']*12))
        self.critic = MultiMetricCritic(self.cache,self.x,self.offline['derived_B'].float(),
            torch.tensor([True,True,False]),('bounded_cost',),width=12,seed=31)
        torch.manual_seed(51)
        self.models = dict(flow=TinyPolicy(),actor=TinyPolicy(),critic=self.critic)
        self.opts = {k: (torch.optim.AdamW(v.parameters(),lr=3e-4,weight_decay=1e-4) if k=='critic' else
                          torch.optim.Adam(v.parameters(),lr=1e-3)) for k,v in self.models.items()}
        # Actual Adam prefix, not an empty optimizer alleged to be resumed.
        for _ in range(2):
            self.opts['actor'].zero_grad()
            self.models['actor'](self.x).square().mean().backward()
            self.opts['actor'].step()
        self.generators = {k:torch.Generator().manual_seed(100+i) for i,k in enumerate(online.GENERATOR_KEYS)}
        self.payload = copy.deepcopy(dict(**{k:m.state_dict() for k,m in self.models.items()},
            optimizers={k:o.state_dict() for k,o in self.opts.items()}, rng=torch.get_rng_state(),cuda_rng=[],
            generators={k:g.get_state() for k,g in self.generators.items()},step=2,binding=dict(seed=0,round=0)))
        self.source = dict(payload=self.payload,schema=self.schema,
            binding=dict(schema=self.schema,checkpoint=dict(path='fixture',sha256='c'*64)))
        target = online.targets(self.schema,[[10.]*6,[np.nan]*6],['success','physical_failure'])
        self.replay = dict(x=self.x[:2],**target,ready=True,binding=dict(schema=self.schema,seed=0,
            checkpoint_sha256='c'*64,source_round=0,records=[{'index':0},{'index':1}]))
        self.settings = dict(total_updates=4,batch_size=4,alpha=10.,q_weight=1.,flow_steps=10,
                             update_critic=True,huber_delta=1.,grad_clip=1.)

    def tearDown(self):
        self.temp.cleanup()

    def core(self, *, payload=None, resume=False, settings=None):
        models = copy.deepcopy(self.models)
        opts = {k:(torch.optim.AdamW(v.parameters(),lr=99.) if k=='critic' else
                   torch.optim.Adam(v.parameters(),lr=99.)) for k,v in models.items()}
        generators = {k:torch.Generator().manual_seed(999) for k in online.GENERATOR_KEYS}
        binding = online.continuation_binding(self.source,self.replay,settings or self.settings)
        return online.Continuation(models,opts,generators,payload or self.payload,binding,
                                   frozen_reference=self.source,resume=resume)

    def update(self, core, settings=None):
        settings = dict(settings or self.settings)
        settings.pop('total_updates')
        return core.update(self.offline,self.replay,**settings)

    def test_success_and_failure_same_scalar_huber_and_preserved_native_nan(self):
        result = online.targets(self.schema,[[10.]*6,[np.nan]*6,[np.nan]*6],
                                ['success','physical_failure','infrastructure_failure'])
        self.assertTrue(torch.equal(result['mask'],torch.tensor([[True],[True],[False]])))
        self.assertEqual(result['derived_B'][0].item(),1.)
        self.assertEqual(result['derived_B'][1].item(),4.)
        self.assertTrue(torch.isnan(result['derived_B'][2]).all())
        x = self.x[:2]
        loss, meta = self.critic.loss(x,result['derived_B'][:2],result['mask'][:2],regression='huber')
        standardized = self.critic.standardized(x)
        truth = ((result['derived_B'][:2].float()-self.critic.target_mean)/self.critic.target_std)[:,None]
        reference = torch.nn.functional.huber_loss(standardized,truth.expand_as(standardized))
        self.assertAlmostEqual(float(loss),float(reference),places=7)
        self.assertEqual(meta['observed_labels'],2)

    def test_fixed_spec_rejects_shift_and_missing_active_but_inactive_nan_allowed(self):
        original = copy.deepcopy(self.schema)
        result = online.targets(self.schema,[[10.,np.nan,np.nan,np.nan,np.nan,np.nan]],['success'])
        self.assertEqual(result['cost'].item(),1.)
        self.assertFalse(online.targets(self.schema,[[np.nan]*6],['success'])['valid'].item())
        bad = copy.deepcopy(self.schema); bad['utility_spec']['scales'][0]=11.
        with self.assertRaisesRegex(ValueError,'scale identity'):
            online.targets(bad,self.costs,['success']*12)
        self.assertEqual(self.schema,original)

    def test_q_sign_units_and_actor_gradient_no_normalizer_division(self):
        x=self.x[:2].clone().requires_grad_()
        with frozen_critic(self.critic):
            q=online.scalar_q(self.critic,x)
            self.assertTrue(torch.equal(q,-self.critic(x).mean((1,2))))
            q.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in self.critic.parameters()))

    def test_prefix_optimizer_rng_and_named_states_restored_exactly(self):
        core=self.core()
        state=core.state()
        for name in online.MODEL_KEYS:
            online.equal_tree(state[name],self.payload[name])
            online.equal_tree(state['optimizers'][name],self.payload['optimizers'][name])
        online.equal_tree(state['generators'],self.payload['generators'])
        online.equal_tree(state['rng'],self.payload['rng'])
        self.assertEqual(state['parent_steps'],2)
        self.assertEqual(state['step'],0)

    def test_resume_equals_uninterrupted_all_tensors_adam_rng_and_teacher(self):
        original=copy.deepcopy(self.payload)
        whole=self.core()
        for _ in range(4): self.update(whole)
        expected=whole.state()
        split=self.core()
        for _ in range(2): self.update(split)
        path=self.root/'resume.pt';torch.save(split.state(),path)
        resumed=self.core(payload=torch.load(path,weights_only=False),resume=True)
        for _ in range(2): self.update(resumed)
        online.equal_tree(expected,resumed.state())
        online.equal_tree(original,self.payload)
        online.equal_tree(expected['flow'],self.payload['flow'])
        online.equal_tree(online.normalization(resumed.models['critic']),online.normalization(self.critic))
        self.assertTrue(all(p.grad is None for p in resumed.models['critic'].parameters()))
        self.assertTrue(all(p.grad is None for p in resumed.models['flow'].parameters()))

    def test_frozen_q_counterfactual_preserves_critic_and_its_optimizer(self):
        settings=dict(self.settings,update_critic=False)
        core=self.core(settings=settings)
        before=core.state()
        result=self.update(core,settings)
        after=core.state()
        online.equal_tree(before['critic'],after['critic'])
        online.equal_tree(before['optimizers']['critic'],after['optimizers']['critic'])
        self.assertEqual(result['critic_updates'],0)
        with self.assertRaises(ValueError): online.equal_tree(before['actor'],after['actor'])

    def test_q_zero_bc_uses_exact_same_actor_noise_stream_despite_q_draws(self):
        bc_settings=dict(self.settings,q_weight=0.)
        bc=self.core(settings=bc_settings); fql=self.core()
        self.update(bc,bc_settings); self.update(fql)
        self.assertTrue(torch.equal(bc.generators['actor'].get_state(),fql.generators['actor'].get_state()))
        online.equal_tree(bc.models['critic'].state_dict(),self.payload['critic'])

    def test_incomplete_replay_changed_target_scale_or_resume_buffer_refused(self):
        core=self.core();self.replay['ready']=False
        with self.assertRaisesRegex(ValueError,'Unresolved'):
            self.update(core)
        self.replay['ready']=True
        self.replay['utility_spec_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'scalar B'):
            self.update(core)
        self.replay['utility_spec_sha256']=self.schema['utility_spec_sha256']
        saved=core.state();saved['critic']['target_mean']+=.1
        with self.assertRaisesRegex(ValueError,'Tensor'):
            self.core(payload=saved,resume=True)

    def test_resume_binding_and_unbound_budget_change_refused(self):
        core=self.core();saved=core.state();saved['binding']['settings']['alpha']=3.
        with self.assertRaisesRegex(ValueError,'binding changed'):
            self.core(payload=saved,resume=True)
        with self.assertRaisesRegex(ValueError,'bound prospective'):
            self.update(core,dict(self.settings,alpha=3.))

    def test_first_prefix_refuses_foreign_actor_optimizer_or_named_rng(self):
        for key in ('actor','optimizers','generators','step'):
            bad=copy.deepcopy(self.payload)
            if key=='actor': bad['actor']['linear.bias']+=.2
            elif key=='optimizers': bad['optimizers']['actor']['param_groups'][0]['lr']=.5
            elif key=='generators': bad['generators']['q']=torch.Generator().manual_seed(991).get_state()
            else: bad['step']+=1
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.core(payload=bad)

    def native_fixture(self):
        exp=self.root/'experiment';exp.mkdir()
        config=dict(design='bp_fe',arm='future',physical_failure_reward=-4)
        (exp/'config.json').write_text(json.dumps(config))
        cp=self.root/'parent.pt';cp.write_bytes(b'externally-pinned-parent')
        folder=exp/'seed_0/round_0/requests';folder.mkdir(parents=True)
        rows=[]
        for role,noiseval in [('collect',.15),('evaluation',.25)]:
            sample=folder/'future'/role/'0';sample.mkdir(parents=True)
            raw=np.full((3,2),.1,dtype=np.float32);noise=np.full((3,2),noiseval,dtype=np.float32)
            np.savez(sample/'actions.npz',raw_action=raw,action=raw,noise=noise)
            row=dict(sample_dir=str(sample),source_checkpoint=str(cp),checkpoint_sha256=sha(cp),graph='bp_fe',
                seed=0,round=0,arm='future',role=role,index=0,terminal_mask=0,training_feedback_eligible=role=='collect',
                config_sha256=sha(exp/'config.json'),dataset_sha256='d'*64,noise_seed=123 if role=='collect' else 456,
                raw_action=raw.tolist(),executed_action=raw.tolist(),raw_action_sha256=action_sha(raw),
                action_sha256=action_sha(raw),noise_sha256=action_sha(noise),run_id=role+'0')
            (sample/'request.json').write_text(json.dumps(row));rows.append(row)
        tasks=folder/'tasks.jsonl';tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (folder/'completed.json').write_text(json.dumps(dict(tasks_sha256=sha(tasks),config_sha256=sha(exp/'config.json'),count=2)))
        row=rows[0];sample=Path(row['sample_dir'])
        out=dict(row,status='geometry_failure',metrics=None,reward=-4,
            request_sha256=sha(sample/'request.json'),tasks_sha256=sha(tasks))
        (sample/'outcome.json').write_text(json.dumps(out))
        binding=dict(config_sha256=sha(exp/'config.json'),dataset_sha256='d'*64,reward_scale=100.)
        args=dict(seed=0,source_round=0,count=1,tasks_sha256=sha(tasks),config_sha256=sha(exp/'config.json'),
            checkpoint_sha256=sha(cp),collection_noise_seed=123,evaluation_noise_sha256=[rows[1]['noise_sha256']],
            value=self.schema,cache=self.cache)
        return exp,config,binding,args,rows,out

    def test_real_request_existing_native_geometry_failure_admits_b4_keeps_raw_reward(self):
        exp,cfg,binding,args,rows,out=self.native_fixture()
        with mock.patch.object(backend,'context',return_value=(cfg,None,binding)):
            admitted=online.admit_collect(exp,**args)
        self.assertTrue(admitted['ready'])
        self.assertEqual(admitted['derived_B'].item(),4.)
        self.assertEqual(admitted['records'][0]['raw_native_reward'],-4)
        self.assertIsNone(admitted['records'][0]['native_grt_metrics'])
        self.assertFalse(admitted['success'].item())

    def test_evaluation_alias_foreign_parent_noise_and_raw_coordinate_tamper_refused(self):
        exp,cfg,binding,args,rows,out=self.native_fixture()
        expected=dict(arm=cfg['arm'],seed=0,round=0,config_sha256=binding['config_sha256'],
            checkpoint_sha256=args['checkpoint_sha256'],collection_noise_seed=123,
            evaluation_noise_sha256=args['evaluation_noise_sha256'])
        for key,val in [('role','evaluation'),('training_feedback_eligible',False),('checkpoint_sha256','e'*64),
                        ('noise_seed',456),('noise_sha256',rows[1]['noise_sha256']),('raw_action',[[0.,0.]]*3)]:
            row=copy.deepcopy(rows[0]);row[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):
                online.validate_collect_identity(row,out,expected)

    def test_missing_collect_not_silently_filtered_and_denominator_is_fixed(self):
        exp,cfg,binding,args,rows,out=self.native_fixture()
        (Path(rows[0]['sample_dir'])/'outcome.json').unlink()
        with mock.patch.object(backend,'context',return_value=(cfg,None,binding)):
            admitted=online.admit_collect(exp,**args)
            self.assertFalse(admitted['ready']);self.assertFalse(admitted['mask'].item())
            self.assertEqual(len(admitted['records']),1)
            with self.assertRaisesRegex(ValueError,'denominator'):
                online.admit_collect(exp,**dict(args,count=2))

    def test_final_requires_drt_and_native_qa_even_for_grt_wl_objective(self):
        final=online.screen_for('final')
        spec=dict(version=final.SPEC_VERSION,objective='grt_wl',cost_names=list(final.COST_NAMES),
            weights=list(final.OBJECTIVES['grt_wl']),scales=[1.]*9,
            reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64),
            rule_definition_sha256=final.json_sha(final.rule_definition()),success_cap=2.,failure_cost=4.,
            feasibility='completed_final_DRC0; final_DRCpositive=physical_failure_for_all_objectives')
        # Final spec reference has its own exact schema: derive it from the
        # production constructor so no fixture makes a weak validator pass.
        context=dict(cost_values=np.array([[1.+i]*9 for i in range(4)]),cost_mask=np.ones((4,9),bool),
            statuses=np.array(['success']*4),binding=dict(dataset_sha256='a'*64),source_ids=['a','b','c','d'])
        spec,_=final.specification(context,list(range(4)),'grt_wl')
        with self.assertRaisesRegex(ValueError,'power conditions'):
            online.schema('final',spec)
        value=online.schema('final',spec,final_conditions={'clock':'fixture'})
        exp,cfg,binding,args,rows,out=self.native_fixture()
        run=self.root/'grt';run.mkdir();(run/'backend_macro_audit.json').write_text('{}')
        out.update(status='success',physical_run=str(run),physical_manifest_sha256='f'*64,
                   physical_flow_revision='r',metrics={'rWL_um':100},reward=-1.)
        (Path(rows[0]['sample_dir'])/'outcome.json').write_text(json.dumps(out))
        verified={k:out[k] for k in ('status','metrics','physical_manifest_sha256','physical_flow_revision')}
        with mock.patch.object(backend,'context',return_value=(cfg,None,binding)),\
             mock.patch.object(backend,'verify_run',return_value=verified):
            admitted=online.admit_collect(exp,**dict(args,value=value))
        self.assertFalse(admitted['ready'])
        self.assertEqual(admitted['records'][0]['status'],'missing_final')
        self.assertIsNone(admitted['records'][0]['derived_B'])
        self.assertEqual(admitted['records'][0]['raw_native_reward'],-1.)

        # Confirmed positive native final DRC is a valid scalar B4 with the
        # actual six PPA values retained, rather than a fabricated bad vector.
        metrics=dict(DRT_WL=101.,WNS=-.2,TNS=-3.,Power=.1,StdCellArea=1000.,DRC=2)
        native_binding={0:dict(path='declared-final',sha256='a'*64)}
        with mock.patch.object(backend,'context',return_value=(cfg,None,binding)),\
             mock.patch.object(backend,'verify_run',return_value=verified),\
             mock.patch.object(online,'checked_final',return_value=(metrics,{'audited':True})),\
             mock.patch.object(online,'_costs',return_value=np.full(9,np.nan)):
            admitted=online.admit_collect(exp,**dict(args,value=value,final_outcomes=native_binding))
        self.assertTrue(admitted['ready']);self.assertEqual(admitted['derived_B'].item(),4.)
        self.assertFalse(admitted['success'].item())
        self.assertEqual(admitted['records'][0]['native_final_metrics'],metrics)
        self.assertEqual(admitted['records'][0]['raw_native_reward'],-1.)

    def test_final_native_source_and_power_conditions_are_real_admission_gates(self):
        from . import graph_o2o_final_evidence as evidence
        from analysis import qstudy_final_power_audit as power
        exp,cfg,binding,args,rows,grt=self.native_fixture()
        sample=Path(rows[0]['sample_dir'])
        grt.update(physical_run=str(self.root/'physical'),physical_manifest_sha256='f'*64)
        path=self.root/'final.json'
        value=dict(final_conditions={'clock':'TRAIN'})
        out=dict(status='completed',metrics={'DRC':0},spec=dict(source_outcome=str(sample/'outcome.json'),
            source_outcome_sha256=sha(sample/'outcome.json'),source_run=grt['physical_run'],
            source_manifest_sha256=grt['physical_manifest_sha256']))
        path.write_text(json.dumps(out))
        with mock.patch.object(evidence,'audit_final_outcome',return_value={'physical_identity':{'cap':20}}) as audit,\
             mock.patch.object(power,'checked_native_power',return_value={'common_conditions':{'clock':'OTHER'}}):
            with self.assertRaisesRegex(ValueError,'power conditions'):
                online.checked_final(path,sha(path),sample,grt,value)
            audit.assert_called_once_with(out,design='bp_fe',route_cap=20)
        out['spec']['source_outcome_sha256']='e'*64;path.write_text(json.dumps(out))
        with mock.patch.object(evidence,'audit_final_outcome') as audit:
            with self.assertRaisesRegex(ValueError,'exact completed collect'):
                online.checked_final(path,sha(path),sample,grt,value)
            audit.assert_not_called()


if __name__=='__main__':
    unittest.main()
