"""Small real fits and provenance fixtures, run only in bounded CPU Slurm."""
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

from . import qstudy_final_feedback_screen as screen
from . import qstudy_bounded_screen as bounded


class FinalFeedbackScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Use CPU Slurm for real numerical tests')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='final-feedback-test-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.root=Path(self.tmp.name)
        self.cache=dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.2) for n in 'abc'},
            net_info={'n':dict(nodes={'a':dict(x_offset=.2),'b':{},'c':dict(y_offset=-.3)})},
            macro_clusters=[['a','b']],id2index={n:i for i,n in enumerate('abc')},
            dataflow_mat=np.asarray([[0.,1.,2.],[1.,0.,0.],[2.,0.,0.]]))
        x=torch.rand((16,3,2),generator=torch.Generator().manual_seed(35))*1.2-.6
        row=np.arange(16,dtype=float)
        native=np.column_stack((100+row,-.3-row/100,-20-row,row+.1,row+.2,.2+row/100,1000+row))
        final=np.column_stack((130+2*row,-.4-row/50,-30-3*row,.25+row/100,1100+2*row,np.zeros(16)))
        costs=np.column_stack((native[:,0],final[:,0],-final[:,2],final[:,3],final[:,4],
            .1+row/100,.15+row/90,.2+row/80,.3+row/70))
        self.context=dict(x=x,executed=(x*.9).numpy().reshape(16,-1),native=native,
            mask=np.isfinite(native),final_native=final,final_mask=np.isfinite(final),
            cost_values=costs,cost_mask=np.isfinite(costs),statuses=np.full(16,'success',dtype='<U32'),
            native_statuses=np.full(16,'completed',dtype='<U32'),families=np.arange(8).repeat(2),
            fold=np.arange(8).repeat(2)%4,source_ids=[f'TRAIN_f{i//2}_v{i%2}' for i in range(16)],
            cache=self.cache,movable=torch.ones(3,dtype=torch.bool),binding=dict(dataset_sha256='a'*64))
        self.folds=screen.folds_for(self.context)
        self.train=self.folds[0]['train_indices']
        self.target=self.folds[0]['target_indices']
        screen.STOP=False

    def tearDown(self):
        screen.STOP=False
        self.tmp.cleanup()

    def args(self,name='one',objective='drt_multi_ppa',steps=4):
        return SimpleNamespace(output=self.root/name,objective=objective,seed=2,steps=steps,
            batch_size=8,lr=3e-4,weight_decay=1e-4)

    def fit(self,args,name='cv_0',intercept=None):
        tr,ta=(self.train,self.target) if name.startswith('cv_') else (list(range(16)),list(range(16)))
        with contextlib.redirect_stdout(io.StringIO()):
            return screen.fit(args,self.context,name,tr,ta,self.context,intercept=intercept)

    def nested_equal(self,left,right):
        if isinstance(left,torch.Tensor):
            self.assertTrue(torch.equal(left,right))
        elif isinstance(left,dict):
            self.assertEqual(left.keys(),right.keys())
            for key in left:self.nested_equal(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.nested_equal(a,b)
        else:self.assertEqual(left,right)

    def test_six_prospective_targets_and_four_inherited_family_folds(self):
        self.assertEqual(list(screen.OBJECTIVES),['grt_wl','drt_wl','drt_timing','drt_power','drt_multi_ppa','drt_expert'])
        self.assertEqual(len(screen.tasks()),18)
        self.assertEqual(screen.OBJECTIVES['drt_expert'],(0.,.8,0.,0.,0.,.05,.05,.05,.05))
        self.assertEqual(len(self.folds),5)
        self.assertEqual(self.folds[-1]['train_indices'],self.folds[-1]['target_indices'])
        for fold in self.folds[:4]:
            self.assertFalse(set(self.context['families'][fold['train_indices']]) &
                             set(self.context['families'][fold['target_indices']]))
        bad=copy.deepcopy(self.context)
        bad['fold'][0]=1
        with self.assertRaisesRegex(ValueError,'family crosses'):screen.folds_for(bad)
        self.assertEqual({k:v for k,v in screen.CONFIG.items() if k!='folds'},
                         {k:v for k,v in bounded.CONFIG.items() if k!='folds'})

    def test_success_only_train_scales_no_target_leakage_and_shared_scale(self):
        first,_=screen.specification(self.context,self.train,'grt_wl')
        altered=copy.deepcopy(self.context)
        altered['cost_values'][self.target]*=1e6
        for objective in screen.OBJECTIVES:
            spec,_=screen.specification(altered,self.train,objective)
            self.assertEqual(spec['scales'],first['scales'])
            self.assertEqual(spec['reference'],first['reference'])
        failed=copy.deepcopy(self.context)
        failed['statuses'][self.train[0]]='physical_failure'
        failed['cost_values'][self.train[0]]=1e9
        spec,_=screen.specification(failed,self.train,'drt_multi_ppa')
        np.testing.assert_allclose(spec['scales'],failed['cost_values'][self.train[1:]].mean(0))

    def test_exact_C_B_units_all_six_and_equal_expert_normalization(self):
        for objective,weights in screen.OBJECTIVES.items():
            spec,_=screen.specification(self.context,self.train,objective)
            out=screen.labels(self.context,self.train,spec)
            c=(self.context['cost_values'][self.train]/np.asarray(spec['scales'])*weights).sum(1)
            np.testing.assert_allclose(out['composite_cost'],c,rtol=1e-14)
            np.testing.assert_allclose(out['cost'],2*c/(1+c),rtol=1e-14)
            self.assertAlmostEqual(c.mean(),1.)
            self.assertTrue(torch.equal(out['utility'],-out['cost']))
        spec,_=screen.specification(self.context,self.train,'drt_expert')
        norm=self.context['cost_values'][self.train]/spec['scales']
        np.testing.assert_allclose(screen.labels(self.context,self.train,spec)['composite_cost'],
                                  .8*norm[:,1]+.2*norm[:,5:].mean(1))

    def test_cost_construction_uses_executed_geometry_and_final_TNS(self):
        context=copy.deepcopy(self.context)
        context['final_native'][0,2]=5.
        screen.add_costs(context,self.cache)
        self.assertEqual(context['cost_values'][0,2],0.)
        np.testing.assert_array_equal(context['cost_values'][:,0],self.context['native'][:,0])
        np.testing.assert_array_equal(context['cost_values'][:,1],self.context['final_native'][:,0])
        changed=copy.deepcopy(context)
        changed['x']*=-.1
        screen.add_costs(changed,self.cache)
        np.testing.assert_array_equal(changed['cost_values'],context['cost_values'])
        changed['executed']*=.5
        screen.add_costs(changed,self.cache)
        self.assertFalse(np.array_equal(changed['cost_values'][:,5:],context['cost_values'][:,5:]))

    def test_shared_physical_failure_unknown_NA_and_inactive_NaN(self):
        changed=copy.deepcopy(self.context)
        changed['statuses'][self.target[:3]]=['physical_failure','geometry_failure','tool_error']
        changed['cost_values'][self.target[:3]]=np.nan
        changed['cost_mask'][self.target[:3]]=False
        for objective in screen.OBJECTIVES:
            spec,_=screen.specification(self.context,self.train,objective)
            out=screen.labels(changed,self.target,spec)
            self.assertEqual(out['cost'][:2].tolist(),[4.,4.])
            self.assertTrue(torch.isnan(out['cost'][2]))
            self.assertFalse(bool(out['valid'][2]))
        spec,_=screen.specification(self.context,self.train,'grt_wl')
        changed=copy.deepcopy(self.context)
        changed['cost_values'][:,1:]=np.nan
        changed['cost_mask'][:,1:]=False
        self.assertTrue(bool(screen.labels(changed,self.train,spec)['valid'].all()))
        with self.assertRaisesRegex(ValueError,'Unresolved'):screen.admit(changed|dict(statuses=np.full(16,'tool_error')),self.folds)

    def test_active_constant_or_missing_refuses_entire_six_objective_plan(self):
        screen.admit(self.context,self.folds)
        for value in (.1,np.nan):
            changed=copy.deepcopy(self.context)
            changed['cost_values'][:,-1]=value
            changed['cost_mask'][:,-1]=np.isfinite(value)
            screen.specification(changed,self.train,'grt_wl')
            with self.assertRaisesRegex(ValueError,'whole protocol'):screen.admit(changed,self.folds)
        changed=copy.deepcopy(self.context)
        changed['cost_values'][self.train[0],3]=np.nan
        changed['cost_mask'][self.train[0],3]=False
        with self.assertRaisesRegex(ValueError,'unavailable'):screen.admit(changed,self.folds)

    def test_true_resume_exact_optimizer_rng_and_paired_batches(self):
        complete=self.args('full')
        resumed=self.args('resume')
        batches=[]
        self.fit(complete,intercept=lambda batch:batches.append(batch.clone()))
        called=[]
        def interrupt(batch):
            called.append(batch.clone())
            if len(called)==2:screen._stop()
        with self.assertRaises(SystemExit) as error:self.fit(resumed,intercept=interrupt)
        self.assertEqual(error.exception.code,85)
        screen.STOP=False
        self.fit(resumed,intercept=lambda batch:called.append(batch.clone()))
        for a,b in zip(batches,called):self.assertTrue(torch.equal(a,b))
        a=torch.load(complete.output/'cv_0/resume.pt',weights_only=False)
        b=torch.load(resumed.output/'cv_0/resume.pt',weights_only=False)
        for key in ('model','optimizer','rng','batch_rng','normalization','utility_spec','step'):
            self.nested_equal(a[key],b[key])
        paired=[]
        self.fit(self.args('other','drt_power'),intercept=lambda batch:paired.append(batch.clone()))
        for a,b in zip(batches,paired):self.assertTrue(torch.equal(a,b))
        self.assertFalse(bounded.STOP)

    def test_full_refit_is_explicitly_in_sample_preserves_both_native_stages(self):
        args=self.args()
        done=self.fit(args,'refit')
        self.assertEqual(done['scope'],'train_in_sample_refit')
        self.assertTrue(done['full_refit_is_in_sample'])
        self.assertFalse(done['historical_validation_used'])
        with np.load(args.output/'refit/predictions.npz',allow_pickle=False) as values:
            np.testing.assert_array_equal(values['native_metrics'],self.context['native'])
            np.testing.assert_array_equal(values['final_native'],self.context['final_native'])
            self.assertEqual(values['native_metrics'].shape[1],7)
            self.assertEqual(values['final_native'].shape[1],6)
            self.assertTrue(np.isfinite(values['composite_cost']).all())
        summary=screen.summarize([done])
        self.assertNotIn('historical_validation',summary)
        self.assertEqual(summary['full_train_in_sample_refit'],done)
        self.assertEqual(self.fit(args,'refit'),done)

    def test_completion_scope_and_native_artifact_tamper_refused(self):
        args=self.args()
        self.fit(args,'refit')
        path=args.output/'refit/completed.json'
        original=json.loads(path.read_text())
        bad=dict(original,scope='historical_validation_previously_inspected')
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'scope'):self.fit(args,'refit')
        path.write_text(json.dumps(original))
        artifact=args.output/'refit/predictions.npz'
        with np.load(artifact,allow_pickle=False) as values:
            saved={k:values[k].copy() for k in values.files}
        saved['final_native'][0,0]+=1.
        np.savez_compressed(artifact,**saved)
        original['predictions_sha256']=screen.sha(artifact)
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.fit(args,'refit')

    def test_scalar_Q_gradients_direct_B_and_frozen_parameters(self):
        args=self.args()
        done=self.fit(args)
        spec,_=screen.specification(self.context,self.train,args.objective)
        with mock.patch.dict(screen.CONFIG,steps=args.steps):
            model=screen.checked_model(self.context,spec,args.seed,self.train,args.output/'cv_0/resume.pt',done)
        for parameter in model.parameters():parameter.requires_grad_(False)
        actions=self.context['x'][self.target].clone().requires_grad_(True)
        q=screen.scalar_reward(model,actions)
        self.assertTrue(torch.equal(q,-model(actions).mean((1,2))))
        gradient,=torch.autograd.grad(q.sum(),actions)
        self.assertTrue(bool(gradient.isfinite().all()))
        self.assertGreater(float(gradient.square().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def collection_fixture(self):
        plan_path=self.root/'collection_plan.json'
        report_path=self.root/'collection_report.json'
        tasks=[dict(index=i,key=str(i),output=str(self.root/f'native_{i}'),provenance=dict(i=i),native=dict(i=i)) for i in range(64)]
        plan=dict(tasks=tasks,selection=dict(selected_families=[dict(family=i) for i in range(32)]))
        plan_path.write_text(json.dumps(plan))
        records=[dict(index=i,key=str(i),state='completed',DRC_zero=True,outcome=dict(path=f'o{i}',sha256='b'*64),attempts=[]) for i in range(64)]
        sources={'source.py':'c'*64}
        report=dict(version=screen.collector.VERSION,status='complete',requested=64,counts={'completed':64},records=records,
            plan=screen.bind(plan_path),source_sha256=sources,family_folds=plan['selection']['selected_families'],
            source_split='original384TRAIN_only',policy_evaluation_or_old_DRT_used=False,DRC_zero=64)
        report_path.write_text(json.dumps(report))
        return plan_path,report_path,plan,records,sources

    def test_complete_report_reinspects_all64_and_refuses_missing_or_changed_report(self):
        pp,rp,plan,records,sources=self.collection_fixture()
        inspected=[]
        def inspect(_,task):
            inspected.append(task['index'])
            return {k:v for k,v in records[task['index']].items() if k!='attempts'}
        with mock.patch.object(screen.collector,'checked_plan',return_value=plan), \
             mock.patch.object(screen.collector,'inspect_result',side_effect=inspect), \
             mock.patch.object(screen.collector,'attempt_evidence',return_value=[]), \
             mock.patch.object(screen.collector,'sources',return_value=sources), \
             mock.patch.object(screen.collector,'native_source',side_effect=lambda row:dict(i=row['i'])):
            screen.checked_collection(pp,screen.sha(pp),rp,screen.sha(rp))
            self.assertEqual(inspected,list(range(64)))
            original_sha=screen.sha(rp)
            rp.write_text(rp.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Changed TRAIN source'):
                screen.checked_collection(pp,screen.sha(pp),rp,original_sha)
            records[12]['state']='missing'
            with self.assertRaisesRegex(ValueError,'not completely resolved'):
                screen.checked_collection(pp,screen.sha(pp),rp,screen.sha(rp))

    def test_cv_family_leak_refused_before_update(self):
        args=self.args()
        with self.assertRaisesRegex(ValueError,'family crosses'):
            screen.fit(args,self.context,'cv_0',self.train,self.train[:2],self.context)

    def test_all_five_fits_verified_before_exposing_in_sample_refit(self):
        output=self.root/'study'
        output.mkdir()
        task=next(t for t in screen.tasks() if t['objective']=='drt_multi_ppa' and t['seed']==2)
        plan=dict(tasks=screen.tasks(),output=str(output),folds=self.folds,
            inputs=dict(dataset_sha256='a'*64),source_sha256={'source':'b'*64},
            fold_plan_sha256=screen.json_sha(self.folds),input_evidence=dict(path='evidence',sha256='c'*64))
        pp=output/'plan.json'
        pp.write_text(json.dumps(plan))
        self.context['binding']=screen.run_binding(plan,pp,task)
        args=self.args(objective=task['objective'])
        args.output=output/task['objective']/f"seed_{task['seed']}"
        outcomes=[]
        with contextlib.redirect_stdout(io.StringIO()):
            for fold in self.folds:
                outcomes.append(screen.fit(args,self.context,fold['name'],fold['train_indices'],
                    fold['target_indices'],self.context))
        summary=dict(status='completed',binding=self.context['binding'],objective=task['objective'],
                     seed=task['seed'],outcomes=outcomes,**screen.summarize(outcomes))
        screen.collector.atomic_json(args.output/'summary.json',summary)
        screen.collector.atomic_json(args.output/'completed.json',dict(status='completed',binding=self.context['binding'],
            summary_sha256=screen.sha(args.output/'summary.json'),backend_calls=0,actor_advanced=False,
            historical_validation_used=False))
        with mock.patch.dict(screen.CONFIG,steps=args.steps,batch_size=args.batch_size):
            result=screen.checked_refit(pp,args.objective,args.seed,screen.sha(pp),
                loaded=(plan,dict(train=self.context),self.context))
            self.assertEqual(result['done']['scope'],'train_in_sample_refit')
            self.assertEqual(len([p for p in result['evidence'] if p.endswith('resume.pt')]),5)
            self.assertNotIn('validation',result['data'])
            # Even a byte-bound, rehashed checkpoint may not substitute feature
            # buffers fitted from different rows. Do not trust self-description.
            fold_dir=args.output/'cv_2'
            saved=torch.load(fold_dir/'resume.pt',weights_only=False)
            buffer=next(k for k,v in saved['model'].items() if isinstance(v,torch.Tensor)
                        and 'feature_mean' in k)
            saved['model'][buffer]+=1
            torch.save(saved,fold_dir/'resume.pt')
            summary['outcomes'][2]['checkpoint_sha256']=screen.sha(fold_dir/'resume.pt')
            screen.collector.atomic_json(fold_dir/'completed.json',summary['outcomes'][2])
            screen.collector.atomic_json(args.output/'summary.json',summary)
            completion=screen.read(args.output/'completed.json')
            completion['summary_sha256']=screen.sha(args.output/'summary.json')
            screen.collector.atomic_json(args.output/'completed.json',completion)
            with self.assertRaisesRegex(ValueError,'TRAIN/cache buffer'):
                screen.checked_refit(pp,args.objective,args.seed,screen.sha(pp),
                    loaded=(plan,dict(train=self.context),self.context))

    def test_prepare_pin_source_and_complete_input_evidence_are_immutable(self):
        source=self.root/'source.py'
        source.write_text('# pinned fixture\n')
        collector_plan=self.root/'native_plan.json'
        collector_plan.write_text(json.dumps(dict(output=str(self.root/'native'))))
        output=self.root/'new_study'
        args=SimpleNamespace(collector_plan=collector_plan,collector_plan_sha256=screen.sha(collector_plan),
            collector_report=self.root/'native_report.json',collector_report_sha256='b'*64,
            power_audit=self.root/'power.json',power_audit_sha256='c'*64,output=output)
        inputs=dict(dataset_sha256='a'*64,collector_plan=dict(path=str(collector_plan),sha256=screen.sha(collector_plan)),
            collector_report=dict(path=str(args.collector_report),sha256='b'*64),
            power_audit=dict(path=str(args.power_audit),sha256='c'*64))
        sources={'source.py':screen.sha(source)}
        with mock.patch.object(screen,'ROOT',self.root), \
             mock.patch.object(screen,'source_binding',return_value=sources), \
             mock.patch.object(screen,'load_inputs',return_value=(self.context,inputs)), \
             mock.patch('baseline.api.load_problem',return_value=SimpleNamespace(cond=SimpleNamespace(is_ports=torch.zeros(3,dtype=torch.bool)))):
            with contextlib.redirect_stdout(io.StringIO()):screen.prepare(args)
            path=output/'plan.json'
            plan,data,context=screen.checked_plan(path,screen.sha(path))
            self.assertEqual(set(data),{'train'})
            self.assertEqual(plan['full_refit_scope'],'train_in_sample_refit')
            with self.assertRaisesRegex(ValueError,'externally pinned'):screen.checked_plan(path,None)
            evidence=Path(plan['input_evidence']['path'])
            original=evidence.read_text()
            evidence.write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'Changed TRAIN source'):
                screen.checked_plan(path,screen.sha(path))
            evidence.write_text(original)
            (output/'source_snapshot/source.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'Changed TRAIN source'):
                screen.checked_plan(path,screen.sha(path))


if __name__=='__main__':
    unittest.main()
