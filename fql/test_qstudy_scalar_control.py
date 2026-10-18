"""Slurm-only, tiny real scalar fits and frozen-comparator contract tests."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_scalar_control as scalar
from . import qstudy_screen as screen
from .qstudy_models import MultiMetricCritic, frozen_critic


class ScalarControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run scalar training tests in a bounded Slurm allocation')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='qstudy-scalar-test-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.root=Path(self.temporary.name)
        self.cache=dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.2) for n in 'abc'},
            net_info={'n':{'nodes':{'a':{'x_offset':.2},'b':{},'c':{'y_offset':-.3}}}})
        generator=torch.Generator().manual_seed(19)
        x=torch.rand((12,3,2),generator=generator)*1.2-.6
        costs=torch.stack((100+3*x[:,0,0],.4+.1*x[:,1,0],25+x[:,2,1],
                           (3*x[:,0,1]).clamp_min(0),(4*x[:,1,1]).clamp_min(0)),-1)
        self.family=torch.arange(6).repeat_interleave(2)
        self.binding=dict(dataset_sha256='synthetic_dataset',manifest_sha256='synthetic_manifest',
            cache_sha256='synthetic_cache',fold_seed=scalar.FOLD_SEED,heldout_used=False,backend_calls=0,
            source_sha256={name:scalar.sha(scalar.ROOT/name) for name in
                ['fql/qstudy_screen.py','fql/qstudy_models.py','fql/qstudy_data.py',
                 'fql/graph_o2o_models.py','fql/physical_ridge.py','fql/analytic.py']})
        self.context=dict(x=x,costs=costs,mask=torch.ones_like(costs,dtype=torch.bool),
            families=self.family,movable=torch.ones(3,dtype=torch.bool),cache=self.cache,
            source_ids=[f'train_family{int(f)}_row{i}' for i,f in enumerate(self.family)],binding=self.binding)
        plan=scalar.family_folds(self.family.numpy())[0]
        self.train=np.asarray(plan['train_indices'])
        self.target=np.asarray(plan['target_indices'])
        scalar.STOP=False
        screen.STOP=False

    def tearDown(self):
        scalar.STOP=False
        screen.STOP=False
        self.temporary.cleanup()

    def arguments(self,name='new',steps=6):
        return SimpleNamespace(output=self.root/name,screen_root=self.root/'comparator',seed=3,
            steps=steps,folds=3,batch_size=8,lr=3e-4,weight_decay=1e-4)

    def comparator(self,args,*,capture=None):
        old=SimpleNamespace(output=args.screen_root/f'seed_{args.seed}',seed=args.seed,steps=args.steps,
            folds=args.folds,batch_size=args.batch_size,lr=args.lr,weight_decay=args.weight_decay,
            variant='multi_huber')
        binding=copy.deepcopy(self.binding)
        binding['config']={k:str(v) if isinstance(v,Path) else v for k,v in vars(old).items()}
        context={**self.context,'binding':binding}
        original_make=screen.make_model
        def make(*positional,**keyword):
            model,normalization=original_make(*positional,**keyword)
            if capture is not None:
                loss=model.loss
                def recorded(x,*rest,**options):
                    capture.append(x.detach().clone())
                    return loss(x,*rest,**options)
                model.loss=recorded
            return model,normalization
        with mock.patch.object(screen,'make_model',side_effect=make),contextlib.redirect_stdout(io.StringIO()):
            return screen.fit(old,context,'cv_0',self.train,self.target,self.context)

    def fit(self,args,*,context=None,intercept=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return scalar.fit(args,context or self.context,'cv_0',self.train,self.target,
                              self.context,intercept=intercept)

    def assert_nested_equal(self,a,b):
        if isinstance(a,torch.Tensor):
            self.assertTrue(torch.equal(a,b),'checkpoint tensor differs')
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                self.assert_nested_equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for first,second in zip(a,b):
                self.assert_nested_equal(first,second)
        else:
            self.assertEqual(a,b)

    def test_utility_uses_train_scales_and_ignores_only_zero_weight_missing_labels(self):
        train=torch.tensor([[10.,.2,20.,float('nan'),0.],[30.,.4,60.,float('nan'),0.]])
        mask=torch.isfinite(train)
        scales=scalar.fit_utility_scales(train,mask)
        torch.testing.assert_close(scales,torch.tensor([20.,.3,40.,1.,1.]))
        values,valid=scalar.utility(train,scales,mask)
        self.assertTrue(valid.all())
        torch.testing.assert_close(values,torch.tensor([.775,2.225]))
        held=torch.tensor([[1000.,.3,40.,float('nan'),float('nan')]])
        result,_=scalar.utility(held,scales)
        self.assertAlmostEqual(result.item(),50.5,places=5)
        held[0,1]=float('nan')
        result,valid=scalar.utility(held,scales)
        self.assertFalse(valid.item())
        self.assertTrue(torch.isnan(result).all())

    def test_scalar_q_keeps_original_utility_units_and_action_gradients(self):
        x=self.context['x'][self.train]
        costs=self.context['costs'][self.train]
        scales=scalar.fit_utility_scales(costs,self.context['mask'][self.train])
        u,_=scalar.utility(costs,scales)
        self.assertAlmostEqual(u.mean().item(),1.5,places=6)
        model=MultiMetricCritic(self.cache,x,u[:,None],self.context['movable'],
            ['weighted_utility'],seed=3)
        probe=x[:2].clone().requires_grad_()
        with frozen_critic(model):
            q=scalar.scalar_reward(model,probe)
            torch.testing.assert_close(q,-model(probe).mean((1,2)))
            torch.testing.assert_close(q,model.reward(probe,[1.])*model.objective_scale[0])
            q.sum().backward()
        self.assertTrue(torch.isfinite(probe.grad).all())
        self.assertGreater(probe.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(torch.allclose(q,model.reward(probe,[1.])))
        penalized=scalar.scalar_reward(model,probe,uncertainty_beta=1.)
        self.assertTrue((penalized<=q+1e-7).all())

    def test_family_fold_plan_matches_frozen_screen_and_keeps_related_rows_together(self):
        plans=scalar.family_folds(self.family.numpy())
        groups=np.unique(self.family.numpy())
        np.random.default_rng(41307).shuffle(groups)
        frozen=np.array_split(groups,3)
        seen=[]
        for plan,held in zip(plans,frozen):
            train,target=plan['train_indices'],plan['target_indices']
            self.assertEqual(plan['target_families'],held.tolist())
            self.assertFalse(set(self.family[train].tolist())&set(self.family[target].tolist()))
            seen+=target
        self.assertEqual(sorted(seen),list(range(len(self.family))))

    def test_actual_scalar_and_multihead_fits_have_identical_minibatches(self):
        args=self.arguments()
        old_batches=[]
        self.comparator(args,capture=old_batches)
        scalar_batches=[]
        done=self.fit(args,intercept=lambda x:scalar_batches.append(x.detach().clone()))
        self.assertEqual(len(old_batches),args.steps)
        self.assertEqual(len(scalar_batches),args.steps)
        for old,new in zip(old_batches,scalar_batches):
            torch.testing.assert_close(old,new,atol=0,rtol=0)
        saved=torch.load(args.output/'cv_0/resume.pt',weights_only=False)
        self.assertEqual(saved['spec']['metric_names'],['weighted_utility'])
        self.assertEqual(saved['spec']['width'],128)
        self.assertEqual(saved['spec']['depth'],2)
        self.assertFalse(saved['spec']['hpwl_residual'])
        self.assertFalse(done['actor_advanced'])
        self.assertEqual(done['backend_calls'],0)
        self.assertAlmostEqual(done['normalization']['target_mean'][0],1.5,places=6)

    def test_interrupted_resume_matches_full_parameters_optimizer_and_both_rng_states(self):
        full=self.arguments('full')
        self.comparator(full)
        resumed=self.arguments('resumed')
        self.fit(full)
        steps=[]
        def interrupt(x):
            steps.append(1)
            if len(steps)==3:
                scalar.STOP=True
        with self.assertRaises(SystemExit) as stopped:
            self.fit(resumed,intercept=interrupt)
        self.assertEqual(stopped.exception.code,85)
        self.assertFalse((resumed.output/'cv_0/completed.json').exists())
        saved=torch.load(resumed.output/'cv_0/resume.pt',weights_only=False)
        self.assertEqual(saved['step'],3)
        scalar.STOP=False
        self.fit(resumed)
        a=torch.load(full.output/'cv_0/resume.pt',weights_only=False)
        b=torch.load(resumed.output/'cv_0/resume.pt',weights_only=False)
        for key in ['model','optimizer','rng','batch_rng','step','normalization','binding','spec']:
            self.assert_nested_equal(a[key],b[key])
        with np.load(full.output/'cv_0/predictions.npz') as left,np.load(resumed.output/'cv_0/predictions.npz') as right:
            for key in left.files:
                np.testing.assert_array_equal(left[key],right[key])

    def test_target_changes_do_not_fit_scales_and_bad_comparator_bindings_are_refused(self):
        args=self.arguments()
        self.comparator(args)
        scales=scalar.fit_utility_scales(self.context['costs'][self.train],self.context['mask'][self.train])
        compare,binding=scalar.load_comparison(args,self.context,'cv_0',self.train,self.target,self.context,scales)
        expected,_=scalar.utility(self.context['costs'][self.target],scales)
        self.assertEqual(compare.shape,tuple(expected.shape))
        modified=copy.deepcopy(self.context)
        modified['costs'][self.target,0]*=100
        with self.assertRaisesRegex(ValueError,'target costs'):
            scalar.load_comparison(args,self.context,'cv_0',self.train,self.target,modified,scales)
        torch.testing.assert_close(scales,scalar.fit_utility_scales(modified['costs'][self.train],modified['mask'][self.train]))
        path=args.screen_root/f'seed_{args.seed}/cv_0/completed.json'
        done=json.loads(path.read_text())
        for update in [dict(steps=999),dict(seed=99),dict(variant='legacy')]:
            with self.subTest(update=update):
                path.write_text(json.dumps(dict(done,**update)))
                with self.assertRaisesRegex(ValueError,'variant/seed/training budget'):
                    self.fit(args)
        path.write_text(json.dumps(done))
        checkpoint=path.with_name('resume.pt')
        checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):
            self.fit(args)

    def test_missing_active_targets_and_family_leak_are_refused(self):
        args=self.arguments()
        self.comparator(args)
        changed=copy.deepcopy(self.context)
        changed['mask'][self.train[0],1]=False
        with self.assertRaisesRegex(ValueError,'all active'):
            self.fit(args,context=changed)
        changed=copy.deepcopy(self.context)
        changed['families'][self.train[0]]=self.family[self.target[0]]
        with self.assertRaisesRegex(ValueError,'family crosses'):
            self.fit(args,context=changed)

    def test_changed_resume_source_ids_and_completed_artifact_corruption_are_refused(self):
        args=self.arguments()
        self.comparator(args)
        def interrupt(_):
            scalar.STOP=True
        with self.assertRaises(SystemExit):
            self.fit(args,intercept=interrupt)
        scalar.STOP=False
        changed=copy.deepcopy(self.context)
        changed['source_ids'][self.train[0]]+='foreign'
        with self.assertRaisesRegex(ValueError,'resume input/config/source'):
            self.fit(args,context=changed)
        self.fit(args)
        done=self.fit(args)
        self.assertEqual(done['status'],'completed')
        prediction=args.output/'cv_0/predictions.npz'
        prediction.write_bytes(prediction.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            self.fit(args)

    def test_fold_summary_reports_fold_metrics_instead_of_pooled_utility_rank(self):
        outcomes=[]
        for i in range(3):
            metrics=dict(mae=.2+i*.1,spearman=.1+i*.2,kendall=.1,
                top_regret=.05,family_pairs=4,family_pair_accuracy=.5)
            outcomes.append(dict(scope='train_family_cv',scalar=metrics,multihead=dict(metrics,spearman=.1)))
        report=scalar.summarize(outcomes)
        self.assertEqual(report['folds'],3)
        self.assertAlmostEqual(report['scalar']['spearman']['mean'],.3)
        self.assertAlmostEqual(report['deltas_scalar_minus_multihead']['spearman'],.2)
        self.assertEqual(report['scalar']['family_pair_accuracy_pooled_within_fold'],.5)
        self.assertIn('no pooled',report['warning'])
        self.assertFalse(report['actor_advanced'])

    def test_complete_run_binds_sources_folds_and_reports_and_reentry_does_not_train(self):
        # The audit loader and public problem loader supply explicit synthetic
        # fixtures; actual fitting/checkpoints/report publication are exercised.
        fake_root=self.root/'source_tree'
        for name in scalar.SOURCE_NAMES:
            destination=fake_root/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(scalar.ROOT/name,destination)
        cache_path=fake_root/'baseline/runtime/benchmark_cache/bp_fe.pt'
        cache_path.parent.mkdir(parents=True)
        torch.save(self.cache,cache_path)
        dataset=self.root/'audited_fixture/dataset.npz'
        dataset.parent.mkdir()
        dataset.write_bytes(b'synthetic audited loader fixture')
        dataset.with_name('manifest.json').write_text('{"design":"bp_fe","fixture":true}')
        args=self.arguments('full_run')
        args.dataset=dataset
        self.binding.update(dataset_sha256=scalar.sha(dataset),
            manifest_sha256=scalar.sha(dataset.with_name('manifest.json')),cache_sha256=scalar.sha(cache_path))
        old=SimpleNamespace(output=args.screen_root/f'seed_{args.seed}',seed=args.seed,steps=args.steps,
            folds=args.folds,batch_size=args.batch_size,lr=args.lr,weight_decay=args.weight_decay,
            variant='multi_huber')
        old_binding=copy.deepcopy(self.binding)
        old_binding['config']={k:str(v) if isinstance(v,Path) else v for k,v in vars(old).items()}
        train={**self.context,'binding':old_binding}
        validation=copy.deepcopy(self.context)
        validation['x']+=.03
        validation['costs']*=1.01
        validation['families']+=10
        fixture={}
        for split,source in [('train',self.context),('validation',validation)]:
            metrics=source['costs'].clone()
            metrics[:,1:3]*=-1
            metrics=torch.cat((metrics,metrics.new_ones((len(metrics),2))),-1).numpy()
            fixture[split]=dict(raw_action=source['x'].flatten(1).numpy(),metrics=metrics,
                metric_mask=np.ones_like(metrics,dtype=bool),family=source['families'].numpy(),
                source_row=np.arange(len(metrics)),variant=np.arange(len(metrics))%2)
        with contextlib.redirect_stdout(io.StringIO()):
            for plan in scalar.family_folds(self.family.numpy()):
                screen.fit(old,train,plan['name'],np.asarray(plan['train_indices']),
                           np.asarray(plan['target_indices']),train)
            screen.fit(old,train,'refit',np.arange(len(self.family)),np.arange(len(self.family)),validation)
        problem=SimpleNamespace(cond=SimpleNamespace(is_ports=torch.zeros(3,dtype=torch.bool)))
        with mock.patch.object(scalar,'ROOT',fake_root), \
             mock.patch.object(scalar,'load_dataset',return_value=(fixture,dict(design='bp_fe'))), \
             mock.patch('baseline.api.load_problem',return_value=problem), \
             contextlib.redirect_stdout(io.StringIO()):
            scalar.run(args)
            directory=args.output/f'seed_{args.seed}'
            report=json.loads((directory/'summary.json').read_text())
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['folds'],3)
            self.assertEqual(report['weights'],list(scalar.WEIGHTS))
            self.assertEqual(len(report['outcomes']),4)
            self.assertFalse(report['actor_advanced'])
            self.assertEqual(report['binding']['fold_plan_sha256'],scalar.json_sha(report['binding']['folds']))
            self.assertEqual(len(set(report['binding']['train_source_ids'])),len(self.family))
            self.assertEqual(report['historical_validation']['scope'],'historical_validation_previously_inspected')
            for name,digest in report['binding']['source_sha256'].items():
                self.assertEqual(scalar.sha(directory/'source_snapshot'/name),digest)
            marker=json.loads((directory/'completed.json').read_text())
            self.assertEqual(marker['summary_sha256'],scalar.sha(directory/'summary.json'))
            with mock.patch.object(torch.optim,'AdamW',side_effect=AssertionError('completed run retrained')):
                scalar.run(args)
            self.assertFalse((directory/f'seed_{args.seed}').exists())


if __name__=='__main__':
    unittest.main()
