"""Small CPU contracts and real frozen-loop resume; no GPU or EDA calls."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_final_feedback_actor as adapter
from . import qstudy_actor as frozen
from .qstudy_branch import atomic_json,tree_sha
from .qstudy_data import bind,sha
from . import test_qstudy_actor as existing_tests


class FinalFeedbackActorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run software tests under Slurm')
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.root=Path(self.tmp.name)
        self.stack=ExitStack()
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch('subprocess.run',side_effect=AssertionError('No production calls')))
        self.data=dict(behavior_action=np.zeros((256,22),np.float32),behavior_family=np.arange(256))
        self.data_path=self.root/'data/dataset.npz'
        self.data_path.parent.mkdir()
        self.data_path.write_bytes(b'synthetic measured dataset')
        self.plan_path=self.root/'screen/plan.json'
        self.plan_path.parent.mkdir()
        atomic_json(self.plan_path,dict(synthetic=True))
        self.marker=self.plan_path.parent/'checked_refit.json'
        atomic_json(self.marker,dict(synthetic=True,steps=2000,all_five_fits_checked=True))
        self.base=dict(adapter.FIXED,design='bp_fe',backend_storage='scratch',
            dataset_path=str(self.data_path),dataset_sha256=sha(self.data_path),cache_sha256='c'*64,
            common_checkpoints={},common_checkpoint_sha256={},weights=[1,0,0,0,0],
            critic_variant='legacy',online_rounds=2,online_updates=600,online_samples=8,
            collection_noise_seed=4500000,evaluation_purpose='development_validation')
        for seed in range(3):
            exp=self.root/f'common_{seed}'
            folder=exp/f'seed_{seed}/round_0'
            folder.mkdir(parents=True)
            atomic_json(exp/'config.json',dict(seed=seed))
            path=folder/'common.pt'
            path.write_bytes(('independent source '+str(seed)).encode())
            atomic_json(folder/'training_status.json',dict(status='completed',seed=seed,round=0,
                graph_cache_sha256='c'*64,no_heldout_labels_used=True,config_sha256=sha(exp/'config.json')))
            self.base['common_checkpoints'][str(seed)]=str(path)
            self.base['common_checkpoint_sha256'][str(seed)]=sha(path)
        self.base_path=self.root/'base/config.json'
        self.base_path.parent.mkdir()
        atomic_json(self.base_path,self.base)
        self.args=SimpleNamespace(plan=self.plan_path,plan_sha256=sha(self.plan_path),
            base_config=self.base_path,output=self.root/'actors')
        self.stack.enter_context(mock.patch.object(adapter,'load_dataset',return_value=(self.data,{})))
        self.stack.enter_context(mock.patch.object(adapter,'sources',return_value={'synthetic_source':'a'*64}))
        self.refit_calls=[]
        self.stack.enter_context(mock.patch.object(adapter.screen,'checked_plan',return_value=(
            dict(inputs=self.inputs()),{},self.context())))
        self.stack.enter_context(mock.patch.object(adapter.screen,'checked_refit',side_effect=self.refit))

    def refit(self,plan,objective,seed,expected_sha256,**_):
        self.assertEqual(expected_sha256,self.args.plan_sha256)
        self.refit_calls.append((objective,seed))
        normalization=dict(train_only=True)
        model=SimpleNamespace(normalization_metadata=lambda:normalization)
        model.cuda=lambda:model
        context=self.context()
        spec=dict(weights=list(adapter.screen.OBJECTIVES[objective]),objective=objective)
        return dict(model=model,context=context,spec=spec,normalization=normalization,
            plan=dict(inputs=self.inputs()),
            checkpoint=bind(self.marker),completion=bind(self.marker),evidence={str(self.marker):sha(self.marker)})

    def context(self):
        return dict(x=torch.zeros(64,11,2),families=np.arange(64)//2,
            binding=dict(dataset_sha256=sha(self.data_path),
                final_label_plan_sha256='f'*64,final_report_sha256='e'*64,
                source_rows=list(range(64)),all_rows_original_train=True))

    def inputs(self):
        return dict(dataset_path=self.base['dataset_path'],dataset_sha256=self.base['dataset_sha256'],
            cache=dict(path=str(self.root/'cache.pt'),sha256=self.base['cache_sha256']),
            collector_plan=dict(path='synthetic_final_plan',sha256='f'*64),
            collector_report=dict(path='synthetic_final_report',sha256='e'*64),context_sha256='d'*64)

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def test_all18_predeclared_refits_publish_without_selection(self):
        adapter.prepare(self.args)
        self.assertEqual(self.refit_calls,[(o,s) for o in adapter.OBJECTIVES for s in range(3)])
        for objective in adapter.OBJECTIVES:
            arm,prefix=adapter.ARMS[objective]
            cfg=adapter.validate_config(self.args.output/arm,0)
            self.assertEqual(cfg['run_prefix'],prefix)
            self.assertEqual(cfg['weights'],list(adapter.screen.OBJECTIVES[objective]))
            self.assertFalse(cfg['online_flow_update'])
        self.assertFalse(adapter.read(self.args.output/'preparation.json')['automatic_selection'])
        preparation=adapter.read(self.args.output/'preparation.json')
        self.assertEqual(preparation['critic_inputs']['rows'],64)
        self.assertEqual(preparation['behavior']['targets'],256)
        self.assertEqual(preparation['evaluation']['grt_requests'],144)
        self.assertEqual(preparation['evaluation']['drt_requests'],72)
        self.assertEqual(preparation['evaluation']['drt_indices'],[0,1,2,3])
        self.assertEqual(adapter.read(self.args.output/'ff_grt_wl/config.json')['dataset_path'],str(self.data_path))

    def test_same_config_reprepare_is_immutable_but_source_change_refused(self):
        adapter.prepare(self.args)
        before=sha(self.args.output/'preparation.json')
        adapter.prepare(self.args)
        self.assertEqual(before,sha(self.args.output/'preparation.json'))
        with mock.patch.object(adapter,'sources',return_value={'synthetic_source':'b'*64}):
            with self.assertRaisesRegex(ValueError,'immutable'):
                adapter.prepare(self.args)

    def test_missing_refit_does_not_publish_any_actor(self):
        def missing(plan,objective,seed,*args,**kwargs):
            if objective=='drt_power' and seed==1:
                raise FileNotFoundError('Unfinished fixed refit')
            return self.refit(plan,objective,seed,*args,**kwargs)
        with mock.patch.object(adapter.screen,'checked_refit',side_effect=missing):
            with self.assertRaises(FileNotFoundError):
                adapter.prepare(self.args)
        self.assertFalse(self.args.output.exists())

    def test_common_identity_and_wrong_online_budget_are_rejected(self):
        broken=copy.deepcopy(self.base)
        broken['common_checkpoints']['1']=broken['common_checkpoints']['0']
        broken['common_checkpoint_sha256']['1']=broken['common_checkpoint_sha256']['0']
        with self.assertRaisesRegex(ValueError,'initialization differs'):
            adapter.common_sources(broken)
        self.assertRaises(ValueError,adapter.fixed,dict(self.base,offline_actor_steps=999))
        self.assertRaises(ValueError,adapter.validate_config,Path('unused'),0,1)

    def test_config_mutation_and_changed_common_bytes_are_rejected(self):
        adapter.prepare(self.args)
        exp=self.args.output/'ff_drt_timing'
        cfg=adapter.read(exp/'config.json')
        changed=dict(cfg,weights=[1,0,0,0,0,0])
        atomic_json(exp/'config.json',changed)
        self.assertRaises(ValueError,adapter.validate_config,exp,0)
        atomic_json(exp/'config.json',cfg)
        Path(cfg['common_checkpoints']['0']).write_bytes(b'changed')
        self.assertRaises(ValueError,adapter.validate_config,exp,0)

    def test_measured384_cannot_replace_separate_behavior256(self):
        self.assertRaises(ValueError,adapter.behavior_identity,
                          dict(behavior_action=np.zeros((384,22)),behavior_family=np.arange(384)))

    def test_final64_identity_not_original384_or_historical_validation(self):
        plan=dict(inputs=self.inputs())
        context=self.context()
        self.assertEqual(adapter.critic_source_identity(plan,context,self.base)['families'],32)
        for changed in (
            dict(context,x=torch.zeros(384,11,2)),
            dict(context,families=np.arange(64)),
            dict(context,families=np.r_[np.arange(32),np.zeros(32,dtype=int)]),
            dict(context,x=torch.full((64,11,2),float('nan'))),
            dict(context,binding=dict(context['binding'],dataset_sha256='wrong'))):
            with self.assertRaises(ValueError):
                adapter.critic_source_identity(plan,changed,self.base)
        with self.assertRaises(ValueError):
            adapter.critic_source_identity(dict(inputs=dict(plan['inputs'],cache=dict(sha256='wrong'))),context,self.base)

    def test_exact_screen_plan_sha_required_before_loading_refits(self):
        self.plan_path.write_text('changed final label source plan')
        with self.assertRaises(ValueError):
            adapter.prepare(self.args)
        self.assertFalse(self.refit_calls)
        self.assertFalse(self.args.output.exists())

    def test_preparation_budget_and_refit_evidence_are_immutable(self):
        adapter.prepare(self.args)
        exp=self.args.output/'ff_drt_timing'
        cfg=adapter.read(exp/'config.json')
        self.assertEqual(cfg['final_feedback_critic_inputs']['binding'],self.context()['binding'])
        changed=dict(cfg,final_feedback_evaluation=dict(adapter.FINAL_EVALUATION,drt_indices=[0]))
        atomic_json(exp/'config.json',changed)
        with self.assertRaises(ValueError):
            adapter.validate_config(exp,0)

    def test_late_source_mutation_is_caught_before_config_publication(self):
        original=self.refit
        def mutate(plan,objective,seed,*args,**kwargs):
            result=original(plan,objective,seed,*args,**kwargs)
            if objective==adapter.OBJECTIVES[-1] and seed==2:
                self.marker.write_text('changed source evidence')
            return result
        with mock.patch.object(adapter.screen,'checked_refit',side_effect=mutate):
            with self.assertRaises(ValueError):
                adapter.prepare(self.args)
        self.assertFalse((self.args.output/'preparation.json').exists())

    def test_setup_uses_raw64_but_separate256_behavior_and_rechecks_inputs(self):
        adapter.prepare(self.args)
        exp=self.args.output/'ff_drt_timing'
        labels=dict(cost=torch.arange(64,dtype=torch.float32),valid=torch.ones(64,dtype=torch.bool))
        problem=SimpleNamespace(x=torch.zeros(11,2))
        original_as_tensor=torch.as_tensor
        def as_tensor(*args,**kwargs):
            if kwargs.get('device')=='cuda':
                kwargs['device']='cpu'
            return original_as_tensor(*args,**kwargs)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.cuda,'is_available',return_value=True))
            stack.enter_context(mock.patch.object(torch.Tensor,'cuda',lambda value:value))
            stack.enter_context(mock.patch.object(torch,'as_tensor',side_effect=as_tensor))
            stack.enter_context(mock.patch('baseline.api.load_problem',return_value=problem))
            label_call=stack.enter_context(mock.patch.object(adapter.screen,'labels',return_value=labels))
            cfg,actual_problem,x,costs,mask,behavior,critic,families=adapter.setup(exp,0)
            self.assertIs(actual_problem,problem)
            self.assertEqual(tuple(x.shape),(64,11,2))
            self.assertEqual(tuple(costs.shape),(64,1))
            self.assertEqual(tuple(mask.shape),(64,1))
            self.assertEqual(tuple(behavior.shape),(256,11,2))
            self.assertEqual(tuple(families.shape),(64,))
            self.assertEqual(label_call.call_args.args[1],list(range(64)))
            self.assertEqual(cfg['dataset_path'],str(self.data_path))
            self.data_path.write_bytes(b'changed original backend data')
            with self.assertRaises(ValueError):
                adapter.setup(exp,0)

    def test_isolated_bytecode_and_no_second_q_normalization(self):
        first,second=adapter.isolated_runtime(),adapter.isolated_runtime()
        for name in ('train','export','stop'):
            self.assertIs(first[name].__code__,getattr(frozen,name).__code__)
            self.assertIs(first[name].__globals__,first)
        first['stop']()
        self.assertFalse(second['STOP'])
        class Critic(torch.nn.Module):
            def forward(self,x):
                return torch.stack((x.sum((1,2)),2*x.sum((1,2))),1)[:,:,None]
        x=torch.ones(2,3,2,requires_grad=True)
        q=adapter.reward(Critic(),x,None,None)
        torch.testing.assert_close(q,torch.full((2,),-9.))
        q.sum().backward()
        torch.testing.assert_close(x.grad,torch.full_like(x,-1.5))

    def test_real_frozen_train_resume_preserves_flow_critic_and_streams(self):
        fixture=existing_tests.ActorRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.cfg.update(critic_variant='final_feedback_scalar_huber',online_rounds=0,alpha=10.,q_weight=1.)
        exp=fixture.experiment('rules')
        args=fixture.args(exp)
        def namespace():
            ns=adapter.isolated_runtime()
            ns.update(setup=fixture.setup,GraphPolicy=existing_tests.TinyPolicy,read_replay=lambda *_:None)
            return ns
        ns=namespace()
        with fixture.cpu_fixture():
            ns['train'](args)
        reference=fixture.checkpoint(exp)
        common=torch.load(fixture.common,weights_only=False)
        self.assertEqual(tree_sha(reference['flow']),tree_sha(common['flow']))
        self.assertEqual(tree_sha(reference['critic']),tree_sha(fixture.critic_template.state_dict()))
        self.assertNotEqual(tree_sha(reference['actor']),tree_sha(common['actor']))
        shutil.rmtree(exp/'seed_0')
        ns=namespace()
        calls=[]
        original_step=torch.optim.Adam.step
        def interrupt(opt,*a,**kw):
            out=original_step(opt,*a,**kw)
            calls.append(1)
            if len(calls)==3:
                ns['stop']()
            return out
        with fixture.cpu_fixture(),mock.patch.object(torch.optim.Adam,'step',new=interrupt),self.assertRaises(SystemExit):
            ns['train'](args)
        ns=namespace()
        with fixture.cpu_fixture():
            ns['train'](args)
        actual=fixture.checkpoint(exp)
        for key in ('flow','actor','critic','optimizers','rng','cuda_rng','generators','step','binding'):
            self.assertEqual(tree_sha(reference[key]),tree_sha(actual[key]),key)


if __name__=='__main__':
    unittest.main()
