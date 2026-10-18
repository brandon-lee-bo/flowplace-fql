"""Tiny real fits and artifact/normalization safety tests; Slurm CPU only."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_bounded_screen as screen
from .qstudy_bounded_utility import assert_common_scale, derive_cost_utility
from .qstudy_models import MultiMetricCritic


class BoundedScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run bounded screen tests in a small Slurm CPU allocation')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='qstudy-bounded-test-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.root=Path(self.temporary.name)
        self.cache=dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.2) for n in 'abc'},
            net_info={'n':{'nodes':{'a':{'x_offset':.2},'b':{},'c':{'y_offset':-.3}}}})
        generator=torch.Generator().manual_seed(19)
        x=torch.rand((12,3,2),generator=generator)*1.2-.6
        native=torch.stack((100+3*x[:,0,0],-(.4+.1*x[:,1,0]),-(25+x[:,2,1]),
            (3*x[:,0,1]).clamp_min(0),(4*x[:,1,1]).clamp_min(0),
            torch.ones(12)*.2,torch.ones(12)*1000),-1).double().numpy()
        self.context=dict(x=x,native=native,mask=np.isfinite(native),
            executed=(x*.9).numpy().reshape(12,-1),statuses=np.full(12,'success',dtype='<U32'),
            families=np.arange(6).repeat(2),source_ids=[f'train_family{i//2}_row{i}' for i in range(12)],
            movable=torch.ones(3,dtype=torch.bool),cache=self.cache,
            binding=dict(dataset_sha256='a'*64,manifest_sha256='b'*64,cache_sha256='c'*64))
        plan=screen.family_folds(self.context['families'])[0]
        self.train=np.asarray(plan['train_indices'])
        self.target=np.asarray(plan['target_indices'])
        screen.STOP=False

    def tearDown(self):
        screen.STOP=False
        self.temporary.cleanup()

    def args(self,name='fit',objective='timing',steps=6):
        return SimpleNamespace(output=self.root/name,objective=objective,seed=2,
            steps=steps,batch_size=8,lr=3e-4,weight_decay=1e-4)

    def fit(self,args,context=None,target_data=None,intercept=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return screen.fit(args,context or self.context,'cv_0',self.train,self.target,
                target_data or self.context,intercept=intercept)

    def nested_equal(self,a,b):
        if isinstance(a,torch.Tensor):
            self.assertTrue(torch.equal(a,b),'checkpoint tensor differs')
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                self.nested_equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):
                self.nested_equal(left,right)
        else:
            self.assertEqual(a,b)

    def test_grouped_folds_match_frozen_partition_and_have_exhaustive_targets(self):
        plans=screen.family_folds(self.context['families'])
        groups=np.unique(self.context['families'])
        np.random.default_rng(41307).shuffle(groups)
        observed=[]
        for plan,held in zip(plans,np.array_split(groups,3)):
            self.assertEqual(plan['target_families'],held.tolist())
            train,target=plan['train_indices'],plan['target_indices']
            self.assertFalse(set(self.context['families'][train]) & set(self.context['families'][target]))
            observed+=target
        self.assertEqual(sorted(observed),list(range(12)))

    def test_objectives_share_fold_train_physical_scales_and_ignore_target_values(self):
        specs=[screen.specification(self.context,self.train,o)[0] for o in screen.OBJECTIVES]
        for spec in specs:
            assert_common_scale(specs[0],spec)
        mutated=copy.deepcopy(self.context)
        mutated['native'][self.target,:5]*=1000
        for objective,spec in zip(screen.OBJECTIVES,specs):
            self.assertEqual(screen.specification(mutated,self.train,objective)[0],spec)
        self.assertEqual(specs[0]['reference']['train_rows_sha256'],
            screen.json_sha([self.context['source_ids'][i] for i in self.train]))
        self.assertTrue(all(s>0 for s in specs[0]['scales']))

    def test_physical_scales_exclude_failures_missing_labels_and_use_zero_fallback(self):
        native=np.asarray([[10.,-.2,-20.,0.,np.nan,1.,1.],
                           [30.,-.4,-60.,0.,0.,1.,1.],
                           [999.,-99.,-99.,999.,999.,1.,1.]])
        scales,audit=screen.fit_scales(native,np.asarray(['success','success','physical_failure']),np.isfinite(native))
        np.testing.assert_allclose(scales,[20.,.3,40.,1.,1.])
        self.assertEqual(audit['observed_counts'],[2,2,2,2,1])
        self.assertEqual(audit['fallback_to_one'],[False,False,False,True,True])

    def test_true_scalar_q_units_finite_input_gradients_and_no_secondary_normalization(self):
        spec,_=screen.specification(self.context,self.train,'timing')
        target=screen.labels(self.context,self.train,spec)['cost'].float()
        model=MultiMetricCritic(self.cache,self.context['x'][self.train],target[:,None],
            self.context['movable'],['bounded_cost'],seed=2)
        probe=self.context['x'][self.target]
        torch.testing.assert_close(screen.scalar_reward(model,probe),-model(probe).mean((1,2)))
        torch.testing.assert_close(screen.scalar_reward(model,probe),model.reward(probe,[1.])*model.objective_scale[0])
        self.assertFalse(torch.allclose(screen.scalar_reward(model,probe),model.reward(probe,[1.])))
        audit=screen.input_gradient_audit(model,probe,self.context['movable'])
        self.assertTrue(audit['finite'])
        self.assertEqual(audit['nonzero_rows'],len(probe))
        self.assertGreater(audit['rms_mean'],0)

    def test_objective_fits_use_identical_minibatches_and_fixed_architecture(self):
        batches={}
        for objective in screen.OBJECTIVES:
            batches[objective]=[]
            args=self.args(objective,objective,steps=3)
            done=self.fit(args,intercept=lambda x:batches[objective].append(x.detach().clone()))
            saved=torch.load(args.output/'cv_0/resume.pt',weights_only=False)
            self.assertEqual(saved['spec']['metric_names'],['bounded_cost'])
            self.assertEqual((saved['spec']['width'],saved['spec']['depth'],saved['spec']['ensemble_size']),(128,2,2))
            self.assertFalse(saved['spec']['hpwl_residual'])
            self.assertEqual(saved['actor_contract']['beta'],0)
            self.assertEqual(done['backend_calls'],0)
            self.assertFalse(done['actor_advanced'])
        for left,right in zip(batches['wl'],batches['congestion']):
            torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_heldout_values_do_not_change_fit_or_feature_and_target_normalizers(self):
        args=self.args('original',steps=3)
        done=self.fit(args)
        changed=copy.deepcopy(self.context)
        changed['native'][self.target,:5]*=1000
        other=self.args('changed',steps=3)
        revised=self.fit(other,context=changed,target_data=changed)
        self.assertEqual(done['normalization'],revised['normalization'])
        self.assertEqual(done['utility_spec'],revised['utility_spec'])
        self.assertEqual(done['constant_train_B'],revised['constant_train_B'])
        a=torch.load(args.output/'cv_0/resume.pt',weights_only=False)
        b=torch.load(other.output/'cv_0/resume.pt',weights_only=False)
        self.nested_equal(a['model'],b['model'])
        with np.load(args.output/'cv_0/predictions.npz') as first,np.load(other.output/'cv_0/predictions.npz') as second:
            np.testing.assert_array_equal(first['prediction'],second['prediction'])
            self.assertFalse(np.array_equal(first['truth'],second['truth']))

    def test_failures_and_unresolved_rows_keep_native_nan_and_exact_denominators(self):
        context=copy.deepcopy(self.context)
        bad_train=[self.train[0],self.train[1]]
        bad_target=[self.target[0],self.target[1]]
        for indices in [bad_train,bad_target]:
            context['statuses'][indices]=['physical_failure','infrastructure_failure']
            context['native'][indices]=np.nan
            context['mask'][indices]=False
        args=self.args(steps=3)
        done=self.fit(args,context=context,target_data=context)
        self.assertEqual(done['train_rows'],len(self.train))
        self.assertEqual(done['admitted_train_rows'],len(self.train)-1)
        self.assertEqual(done['target_rows'],len(self.target))
        self.assertEqual(done['valid_target_rows'],len(self.target)-1)
        with np.load(args.output/'cv_0/predictions.npz') as saved:
            self.assertEqual(saved['truth'][0],4.)
            self.assertTrue(np.isnan(saved['truth'][1]))
            np.testing.assert_array_equal(saved['valid'],[True,False,True,True])
            self.assertTrue(np.isnan(saved['native_metrics'][:2]).all())
            self.assertFalse(saved['native_metric_mask'][:2].any())
            self.assertTrue(np.isfinite(saved['prediction']).all())
            self.assertEqual(saved['native_metrics'].shape,(len(self.target),7))
            np.testing.assert_array_equal(saved['native_metrics'],context['native'][self.target])

    def test_interruption_resume_matches_parameters_optimizer_and_both_rngs(self):
        full=self.args('full')
        self.fit(full)
        resumed=self.args('resumed')
        calls=[]
        def interrupt(_):
            calls.append(1)
            if len(calls)==3:
                screen.STOP=True
        with self.assertRaises(SystemExit) as stopped:
            self.fit(resumed,intercept=interrupt)
        self.assertEqual(stopped.exception.code,85)
        self.assertFalse((resumed.output/'cv_0/completed.json').exists())
        self.assertEqual(torch.load(resumed.output/'cv_0/resume.pt',weights_only=False)['step'],3)
        screen.STOP=False
        self.fit(resumed)
        left=torch.load(full.output/'cv_0/resume.pt',weights_only=False)
        right=torch.load(resumed.output/'cv_0/resume.pt',weights_only=False)
        for key in ['model','optimizer','rng','batch_rng','step','normalization','binding','spec','utility_spec']:
            self.nested_equal(left[key],right[key])
        with np.load(full.output/'cv_0/predictions.npz') as first,np.load(resumed.output/'cv_0/predictions.npz') as second:
            for key in first.files:
                np.testing.assert_array_equal(first[key],second[key])

    def test_completed_corruption_and_changed_resume_source_are_refused(self):
        args=self.args()
        self.fit(args)
        path=args.output/'cv_0/predictions.npz'
        path.write_bytes(path.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            self.fit(args)
        args=self.args('interrupted')
        with self.assertRaises(SystemExit):
            self.fit(args,intercept=lambda _:setattr(screen,'STOP',True))
        screen.STOP=False
        context=copy.deepcopy(self.context)
        context['source_ids'][self.train[0]]='changed_identity'
        with self.assertRaisesRegex(ValueError,'binding changed'):
            self.fit(args,context=context)

    def test_family_collision_and_invalid_training_set_are_refused(self):
        context=copy.deepcopy(self.context)
        context['families'][self.train[0]]=context['families'][self.target[0]]
        with self.assertRaisesRegex(ValueError,'family crosses'):
            self.fit(self.args(),context=context)
        context=copy.deepcopy(self.context)
        context['statuses'][self.train]='missing'
        with self.assertRaisesRegex(ValueError,'at least two valid TRAIN'):
            self.fit(self.args(),context=context)

    def test_constant_comparator_and_fold_summary_do_not_pool_fold_utilities(self):
        done=self.fit(self.args(steps=3))
        with np.load(self.args().output/'cv_0/predictions.npz') as saved:
            self.assertAlmostEqual(done['constant_train_B'],float(saved['train_truth'].mean()),places=6)
            self.assertTrue((saved['constant_prediction']==done['constant_train_B']).all())
        other=copy.deepcopy(done)
        other['critic']['mae']=done['critic']['mae']+2
        report=screen.summarize([done,other])
        self.assertAlmostEqual(report['critic']['mae']['mean'],done['critic']['mae']+1)
        self.assertEqual(report['cv_aggregation'],'equal_fold_mean_of_within_fold_metrics')
        self.assertIn('Do not pool',report['warning'])
        self.assertNotIn('sd',report['critic']['mae'])

    def test_plan_fixes_nine_tasks_and_refuses_config_source_or_input_changes(self):
        data={}
        for split,n,offset in [('train',384,0),('validation',16,200)]:
            repeats=np.arange(n)%12
            source={k:(v[repeats].clone() if isinstance(v,torch.Tensor) else v[repeats].copy())
                    for k,v in self.context.items() if k in
                    ['x','native','mask','executed','statuses','families']}
            source['families']=np.arange(n//2).repeat(2)+offset
            source['source_ids']=[f'{split}_row{i}_family{source["families"][i]}' for i in range(n)]
            data[split]=source
        inputs=dict(dataset_path=str(self.root/'dataset.npz'),dataset_sha256='a'*64,
            manifest_sha256='b'*64,rows_sha256='c'*64,cache_path=str(self.root/'cache.pt'),cache_sha256='d'*64)
        args=SimpleNamespace(dataset=self.root/'dataset.npz',output=self.root/'plan')
        with mock.patch.object(screen,'load_inputs',return_value=(data,dict(design='bp_fe'),inputs)),\
             contextlib.redirect_stdout(io.StringIO()):
            screen.prepare(args)
            screen.prepare(args)
            plan_path=args.output/'plan.json'
            plan=json.loads(plan_path.read_text())
            self.assertEqual(len(plan['tasks']),9)
            self.assertEqual({t['objective'] for t in plan['tasks']},set(screen.OBJECTIVES))
            self.assertEqual({t['seed'] for t in plan['tasks']},{0,1,2})
            self.assertEqual(plan['config']['steps'],2000)
            self.assertEqual(len(plan['folds']),4)
            invocation=SimpleNamespace(plan=plan_path,index=0)
            bad=copy.deepcopy(plan)
            bad['config']['steps']=1
            plan_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                screen.run(invocation)
            plan_path.write_text(json.dumps(plan))
            bad_inputs=dict(inputs,dataset_sha256='e'*64)
            with mock.patch.object(screen,'load_inputs',return_value=(data,dict(design='bp_fe'),bad_inputs)):
                with self.assertRaisesRegex(ValueError,'input/cache binding changed'):
                    screen.run(invocation)
            source=plan['source_sha256'].copy()
            first=next(iter(source))
            snapshot=args.output/'source_snapshot'/first
            snapshot.write_bytes(snapshot.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError,'source changed'):
                screen.run(invocation)


if __name__=='__main__':
    unittest.main()
