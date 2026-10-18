"""Synthetic CPU-only dual-source and actual tiny-optimizer contracts."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_scalar_online as core
from . import qstudy_scalar_online_bc as bc
from .qstudy_data import bind, sha
from .qstudy_models import MultiMetricCritic, frozen_critic


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
        self.register_buffer('fixed', torch.zeros(3, 2))
        self.register_buffer('movable', torch.tensor([True, True, False]))

    def restore(self, x, clip=False):
        return torch.where(self.movable[None,:,None], x.clamp(-1,1) if clip else x,
                           self.fixed[None])

    def forward(self, x, clip=True):
        return self.restore(self.linear(x.flatten(1)).reshape_as(x), clip)

    def sample_flow(self, noise, steps):
        return self(noise)


class BCSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.slurm()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(39)
        cls.actor = TinyPolicy()
        opt = torch.optim.Adam(cls.actor.parameters(), lr=1e-5)
        x = torch.ones(2,3,2)*.1
        for _ in range(1000):
            opt.zero_grad()
            cls.actor(x,clip=False).square().mean().backward()
            opt.step()
        cls.actor_optimizer = copy.deepcopy(opt.state_dict())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='scalar-bc-', dir=os.environ['TMPDIR'])
        self.root = Path(self.temp.name)
        self.old = self.root/'bc'; self.other = self.root/'rule'
        self.old.mkdir(); self.other.mkdir()
        self.x = torch.rand(8,3,2,generator=torch.Generator().manual_seed(21))*.6-.3
        self.x[:,2] = 0
        cache = dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.) for n in 'abc'},
            net_info={'n':{'nodes':{n:{} for n in 'abc'}}})
        screen = core.screen_for('rule')
        spec = dict(version=screen.SPEC_VERSION,objective='grt_wl',cost_names=list(screen.COST_NAMES),
            weights=list(screen.OBJECTIVES['grt_wl']),scales=[10.]*6,
            reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64),
            rule_definition_sha256=screen.json_sha(screen.rule_definition()),success_cap=2.,failure_cost=4.)
        self.schema = core.schema('rule',spec)
        costs = np.tile([10.,1.,1.,1.,1.,1.],(8,1)); costs[:,0]+=np.arange(8)
        self.offline = dict(x=self.x,**core.targets(self.schema,costs,['success']*8))
        self.model = MultiMetricCritic(cache,self.x,self.offline['derived_B'].float(),
            torch.tensor([True,True,False]),('bounded_cost',),width=8,seed=17)
        self.flow = TinyPolicy()
        self.legacy = torch.nn.Linear(6,2)
        self.dataset = self.root/'dataset.npz'; self.dataset.write_bytes(b'synthetic dataset binding')
        self.data = dict(behavior_action=np.zeros((256,22),dtype=np.float32),
                         behavior_family=np.arange(256))
        self.common = self.root/'common.pt'
        torch.save(dict(flow=self.flow.state_dict(),actor=self.actor.state_dict()),self.common)
        self.old_critic = self.root/'legacy.pt'
        torch.save(dict(model=self.legacy.state_dict()),self.old_critic)
        cfg = dict(bc.FIXED,arm='bc',run_prefix='bcfr',dataset_path=str(self.dataset),
            dataset_sha256=sha(self.dataset),cache_sha256='f'*64,
            common_checkpoints={'0':str(self.common)},common_checkpoint_sha256={'0':sha(self.common)},
            critic_checkpoints={'0':str(self.old_critic)},critic_checkpoint_sha256={'0':sha(self.old_critic)})
        self.write(self.old/'config.json',cfg); self.cfg=cfg
        self.target_cfg = dict(cfg,arm='rule_grt_wl',run_prefix='rrv10',q_weight=1.,
            critic_variant='rule_scalar_huber',weights=[1.,0.,0.,0.,0.,0.],
            rule_behavior=bc._behavior_identity(self.data))
        self.write(self.other/'config.json',self.target_cfg)
        generators={n:torch.Generator().manual_seed(100+i).get_state()
                    for i,n in enumerate(core.GENERATOR_KEYS)}
        oldbinding=dict(config_sha256=sha(self.old/'config.json'),dataset_sha256=sha(self.dataset),
            common_sha256=sha(self.common),initial_critic_sha256=sha(self.old_critic),
            round=0,seed=0,parent=None,replay=[],source_sha256={n:sha(bc.ROOT/n) for n in bc.OLD_SOURCE_NAMES})
        self.payload = dict(flow=copy.deepcopy(self.flow.state_dict()),actor=copy.deepcopy(self.actor.state_dict()),
            critic=copy.deepcopy(self.legacy.state_dict()),
            optimizers=dict(actor=copy.deepcopy(self.actor_optimizer),
                flow=torch.optim.Adam(self.flow.parameters(),lr=1e-5).state_dict(),
                critic=torch.optim.AdamW(self.legacy.parameters(),lr=3e-4,weight_decay=1e-4).state_dict()),
            rng=torch.get_rng_state(),cuda_rng=[],generators=generators,step=1000,binding=oldbinding)
        self.folder=self.old/'seed_0/round_0'; self.folder.mkdir(parents=True)
        torch.save(self.payload,self.folder/'checkpoint.pt')
        self.write(self.folder/'binding.json',oldbinding)
        self.write(self.folder/'completed.json',dict(status='completed',steps=1000,binding=oldbinding,
            checkpoint_sha256=sha(self.folder/'checkpoint.pt'),offline_critic_frozen=True,online_flow_update=False))
        targetpayload=copy.deepcopy(self.payload)
        targetpayload['critic']=copy.deepcopy(self.model.state_dict())
        targetpayload['optimizers']['critic']=torch.optim.AdamW(self.model.parameters(),lr=3e-4,weight_decay=1e-4).state_dict()
        targetpayload['binding']=dict(seed=0,round=0)
        targetcheckpoint=self.other/'checkpoint.pt';torch.save(targetpayload,targetcheckpoint)
        refit=self.root/'refit.pt';torch.save(dict(model=self.model.state_dict()),refit)
        self.target=dict(payload=targetpayload,schema=self.schema,
            refit=dict(model=self.model,checkpoint=bind(refit)),
            binding=dict(schema=self.schema,config=bind(self.other/'config.json'),
                checkpoint=bind(targetcheckpoint),common=bind(self.common)))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path,value):
        Path(path).write_text(json.dumps(value))

    def load(self,**kwargs):
        opts=dict(expected_bc_config_sha256=sha(self.old/'config.json'),
            expected_bc_checkpoint_sha256=sha(self.folder/'checkpoint.pt'),
            expected_scalar_config_sha256=sha(self.other/'config.json'),
            expected_scalar_checkpoint_sha256=self.target['binding']['checkpoint']['sha256'])
        opts.update(kwargs)
        with mock.patch.object(core,'load_offline_source',return_value=self.target) as verified, \
                mock.patch.object(bc,'load_dataset',return_value=(self.data,{})):
            result=bc.load_bc_source('rule',self.old,self.other,0,**opts)
            verified.assert_called_once_with('rule',self.other.resolve(),0,opts['expected_scalar_checkpoint_sha256'])
            return result

    def test_real_bc_identity_and_all_preserved_states_with_two_source_binding(self):
        old=copy.deepcopy(self.payload); target=copy.deepcopy(self.target['payload'])
        result=self.load(); new=result['payload']
        for key in ('flow','actor','rng','cuda_rng','generators','binding','step'):
            core.equal_tree(new[key],old[key])
        for key in ('actor','flow'):
            core.equal_tree(new['optimizers'][key],old['optimizers'][key])
        core.equal_tree(new['critic'],self.model.state_dict())
        self.assertFalse(new['optimizers']['critic']['state'])
        self.assertEqual(result['binding']['checkpoint'],bind(self.folder/'checkpoint.pt'))
        self.assertEqual(result['binding']['target_scalar_source'],self.target['binding'])
        self.assertEqual(result['binding']['assembled_payload_sha256'],bc.state_sha(new))
        self.assertEqual(new['binding']['initial_critic_sha256'],sha(self.old_critic))
        core.equal_tree(self.payload,old);core.equal_tree(self.target['payload'],target)

    def test_behavior_identity_matches_both_frozen_producer_helpers(self):
        from . import qstudy_rule_reward_actor as rule
        from . import qstudy_final_feedback_actor as final
        self.assertEqual(bc._behavior_identity(self.data),rule.behavior_identity(self.data))
        self.assertEqual(bc._behavior_identity(self.data),final.behavior_identity(self.data))

    def test_external_config_checkpoint_and_changed_source_bindings_refused(self):
        for key in ('expected_bc_config_sha256','expected_bc_checkpoint_sha256',
                    'expected_scalar_config_sha256'):
            with self.subTest(key=key), self.assertRaises(ValueError): self.load(**{key:'0'*64})
        binding=json.loads((self.folder/'binding.json').read_text())
        binding['source_sha256']['fql/qstudy_actor.py']='0'*64
        self.write(self.folder/'binding.json',binding)
        with self.assertRaisesRegex(ValueError,'completion/binding'):self.load()

    def test_changed_teacher_behavior_rng_or_budget_refused(self):
        for item in ('teacher','behavior','rng','budget'):
            with self.subTest(item=item):
                before=copy.deepcopy(self.target);cfg=copy.deepcopy(self.target_cfg)
                if item=='teacher':self.target['payload']['flow']['linear.bias']+=.1
                if item=='behavior':cfg['rule_behavior']['actions_sha256']='0'*64
                if item=='rng':self.target['payload']['generators']['actor']=torch.Generator().manual_seed(2).get_state()
                if item=='budget':cfg['alpha']=30.
                self.write(self.other/'config.json',cfg)
                self.target['binding']['config']=bind(self.other/'config.json')
                with self.assertRaises(ValueError):self.load()
                self.target=before;self.write(self.other/'config.json',self.target_cfg)

    def test_missing_or_fabricated_actor_adam_and_replay_prefix_refused(self):
        for item in ('empty','step','replay'):
            with self.subTest(item=item):
                payload=copy.deepcopy(self.payload)
                if item=='empty':payload['optimizers']['actor']['state']={}
                if item=='step':next(iter(payload['optimizers']['actor']['state'].values()))['step']=torch.tensor(999.)
                if item=='replay':payload['binding']['replay']=[{'foreign':True}]
                torch.save(payload,self.folder/'checkpoint.pt')
                done=json.loads((self.folder/'completed.json').read_text())
                done['checkpoint_sha256']=sha(self.folder/'checkpoint.pt');self.write(self.folder/'completed.json',done)
                with self.assertRaises(ValueError):self.load()

    def test_learned_or_nonempty_target_critic_optimizer_cannot_be_hidden(self):
        self.target['payload']['critic']['target_mean']+=.1
        with self.assertRaisesRegex(ValueError,'Tensor state'):self.load()
        self.target['payload']['critic']=copy.deepcopy(self.model.state_dict())
        self.target['payload']['optimizers']['critic']['state']={0:{'step':torch.tensor(1.)}}
        with self.assertRaises(ValueError):self.load()

    def make_core(self,source,settings):
        models=dict(flow=copy.deepcopy(self.flow),actor=copy.deepcopy(self.actor),critic=copy.deepcopy(self.model))
        opts={k:(torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) if k=='critic' else
                 torch.optim.Adam(m.parameters(),lr=1e-5)) for k,m in models.items()}
        generators={k:torch.Generator().manual_seed(999) for k in core.GENERATOR_KEYS}
        replay=dict(x=self.x[:2],**core.targets(self.schema,[[11.,1.,1.,1.,1.,1.],[12.,1.,1.,1.,1.,1.]],['success']*2),
            ready=True,binding=dict(schema=self.schema,seed=0,source_round=0,
                checkpoint_sha256=source['binding']['checkpoint']['sha256']))
        binding=core.continuation_binding(source,replay,settings)
        obj=core.Continuation(models,opts,generators,source['payload'],binding,frozen_reference=source)
        return obj,replay

    def test_zero_q_tiny_step_exact_real_bc_adam_rng_continuation(self):
        source=self.load(); settings=dict(total_updates=2,batch_size=4,alpha=10.,q_weight=0.,
            flow_steps=10,update_critic=False,huber_delta=1.,grad_clip=1.)
        obj,replay=self.make_core(source,settings)
        reference=copy.deepcopy(self.actor);teacher=copy.deepcopy(self.flow)
        opt=torch.optim.Adam(reference.parameters(),lr=1e-5)
        opt.load_state_dict(copy.deepcopy(self.payload['optimizers']['actor']))
        gen=torch.Generator();gen.set_state(self.payload['generators']['actor'])
        for _ in range(2):
            noise=2*torch.rand(4,3,2,generator=gen)-1
            with torch.no_grad():target=teacher.sample_flow(noise,10)
            prediction=reference(noise,clip=False)
            loss=10*(prediction[:,reference.movable]-target[:,reference.movable]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True);opt.step()
            row=obj.update(self.offline,replay,**{k:v for k,v in settings.items() if k!='total_updates'})
            self.assertTrue(np.isfinite(row['q']))
            self.assertEqual(row['critic_updates'],0)
        core.equal_tree(obj.models['actor'].state_dict(),reference.state_dict())
        core.equal_tree(obj.opts['actor'].state_dict(),opt.state_dict())
        core.equal_tree(obj.generators['actor'].get_state(),gen.get_state())
        core.equal_tree(obj.models['critic'].state_dict(),source['payload']['critic'])

    def test_same_prefix_frozen_q_and_updated_q_actual_steps_and_resume(self):
        source=self.load(); original=copy.deepcopy(source['payload'])
        states=[]
        for update in (False,True):
            settings=dict(total_updates=2,batch_size=4,alpha=10.,q_weight=1.,flow_steps=10,
                update_critic=update,huber_delta=1.,grad_clip=1.)
            obj,replay=self.make_core(source,settings)
            row=obj.update(self.offline,replay,**{k:v for k,v in settings.items() if k!='total_updates'})
            states.append(obj.state())
            self.assertEqual(row['critic_updates'],int(update))
            self.assertEqual(row['parent_steps'],1000)
            core.equal_tree(obj.models['flow'].state_dict(),original['flow'])
            core.equal_tree(core.normalization(obj.models['critic']),core.normalization(self.model))
        core.equal_tree(states[0]['critic'],original['critic'])
        with self.assertRaises(ValueError):core.equal_tree(states[1]['critic'],original['critic'])
        core.equal_tree(states[0]['generators']['actor'],states[1]['generators']['actor'])
        core.equal_tree(source['payload'],original)

    def test_assembled_state_hash_covers_optimizer_rng_and_original_binding(self):
        result=self.load(); expected=result['binding']['assembled_payload_sha256']
        for item in ('rng','optimizer','binding'):
            value=copy.deepcopy(result['payload'])
            if item=='rng':value['rng'][0]^=1
            elif item=='optimizer':next(iter(value['optimizers']['actor']['state'].values()))['exp_avg']+=.01
            else:value['binding']['common_sha256']='0'*64
            self.assertNotEqual(bc.state_sha(value),expected)


if __name__=='__main__':
    unittest.main()
