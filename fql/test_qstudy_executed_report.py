"""Synthetic saved-artifact reporting tests; no fits, model load or EDA."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_executed_control as control
from . import qstudy_executed_report as paired
from .qstudy_models import _masked_stats


class ExecutedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run report tests through Slurm')
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='executed-report-test-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.root=Path(self.temporary.name)
        self.output=self.root/'executed'
        self.output.mkdir()
        self.plan_path=self.output/'plan.json'
        self.sources=control.source_bindings()
        for name in self.sources:
            p=self.output/'source_snapshot'/name
            p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(control.ROOT/name,p)
        self.tasks=[]
        self.arrays={}
        for dataset,design in [('bp_fe384','bp_fe'),('bp_be64','bp_be')]:
            path,data=self.dataset(dataset,design)
            arrays,account=control.assemble_arrays(data)
            self.arrays[dataset]=arrays
            for seed in (0,1,2):
                raw_output=self.root/'raw'/dataset/control.VARIANT/f'seed_{seed}'
                task=dict(dataset=str(path),dataset_sha256=control.sha(path),
                    manifest_sha256=control.sha(path.with_name('manifest.json')),
                    rows_sha256=control.sha(path.with_name('rows.jsonl')),
                    cache_sha256=control.sha(control.ROOT/f'baseline/runtime/benchmark_cache/{design}.pt'),
                    design=design,seed=seed,variant=control.VARIANT,config=control.CONFIG,
                    input_representation=control.INPUT,folds=control.folds(arrays),accounting=account,
                    output=str(self.output/dataset/f'seed_{seed}'),raw_output=str(raw_output))
                binding=dict(dataset_sha256=task['dataset_sha256'],manifest_sha256=task['manifest_sha256'],
                    cache_sha256=task['cache_sha256'],config=dict(dataset=str(path),output=str(raw_output),
                        seed=seed,variant=control.VARIANT,**control.CONFIG),
                    source_sha256={k:v for k,v in self.sources.items() if k in
                        ('fql/qstudy_screen.py','fql/qstudy_models.py','fql/qstudy_data.py',
                         'fql/graph_o2o_models.py','fql/physical_ridge.py','fql/analytic.py')},
                    fold_seed=41307,heldout_used=False,backend_calls=0)
                raw_output.mkdir(parents=True)
                control.atomic_json(raw_output/'binding.json',binding)
                outcomes=[self.fit_artifacts(raw_output,fold,arrays,binding,False) for fold in task['folds']]
                control.atomic_json(raw_output/'completed.json',dict(status='completed',binding=binding,outcomes=outcomes))
                task['raw_comparison']=control.raw_comparisons(task,arrays,self.sources)
                self.tasks.append(task)
        control.atomic_json(self.output/'tasks.json',self.tasks)
        self.plan=dict(task_count=6,datasets=list(control.DATASETS),seeds=[0,1,2],variant=control.VARIANT,
            config=control.CONFIG,input_representation=control.INPUT,source_sha256=self.sources,
            raw_matrix=str(self.root/'raw'),data_root=str(self.root/'data_audit'),
            tasks_sha256=control.sha(self.output/'tasks.json'))
        control.atomic_json(self.plan_path,self.plan)
        for task in self.tasks:
            arrays=self.arrays[Path(task['dataset']).parent.name]
            binding=paired.expected_binding(self.plan_path,self.plan,task)
            folder=Path(task['output'])
            folder.mkdir(parents=True)
            control.atomic_json(folder/'binding.json',binding)
            inventories=[]
            for fold in task['folds']:
                self.fit_artifacts(folder,fold,arrays,binding,True)
                done=control.read_json(folder/fold['name']/'completed.json')
                inventory=control.verify_prediction(folder/fold['name'],done,binding,fold,arrays)
                hashes={control.array_sha(row) for row in arrays['train']['x'][fold['train_indices']]}
                inventory.update(raw_comparison=task['raw_comparison']['fits'][fold['name']],
                    executed_target_rows_also_in_train=sum(control.array_sha(row) in hashes
                        for row in arrays[fold['split']]['x'][fold['target_indices']]))
                control.atomic_json(folder/fold['name']/'verified.json',inventory)
                inventories.append(dict(name=fold['name'],**inventory))
            control.atomic_json(folder/'completed.json',dict(status='completed',binding=binding,
                fits=inventories,backend_calls=0,actor_advanced=False))

    def tearDown(self):self.temporary.cleanup()

    def dataset(self,name,design):
        path=self.root/'data_audit'/name/'dataset.npz'
        path.parent.mkdir(parents=True)
        rng=np.random.default_rng(13)
        data={}
        packed={}
        for split,n,family in [('train',12,np.repeat(np.arange(6),2)),
                               ('validation',4,np.repeat([100,101],2))]:
            raw=rng.uniform(-.5,.5,(n,6)).astype(np.float32)
            executed=raw*.7+.02
            executed[2]=executed[0]  # Cross-family collision remains disclosed.
            if n>4:executed[4]=executed[0]  # Three families cannot all share one two-family fold.
            costs=np.stack((100+np.arange(n),.1+.01*np.arange(n),10+np.arange(n),
                1+np.arange(n),3+np.arange(n)),1).astype(np.float64)
            native=np.column_stack((costs,np.ones((n,2))))
            native[:,1:3]*=-1
            data[split]=dict(raw_action=raw,executed_action=executed,metrics=native,
                metric_mask=np.ones((n,7),dtype=bool),family=family,status=np.full(n,'success'))
            packed.update({split+'_'+key:value for key,value in data[split].items()})
        packed.update(behavior_action=np.zeros((2,6),dtype=np.float32),behavior_family=np.asarray([0,1]))
        np.savez_compressed(path,**packed)
        path.with_name('rows.jsonl').write_text('{"fixture":"provenance"}\n')
        control.atomic_json(path.with_name('manifest.json'),dict(dataset_sha256=control.sha(path),
            rows_sha256=control.sha(path.with_name('rows.jsonl')),metric_names=list(control.screen.METRIC_NAMES),
            no_test_or_heldout_exported=True,historical_online_included=False,design=design,
            split_counts={'train':12,'validation':4}))
        return path,data

    def fit_artifacts(self,root,fold,arrays,binding,executed):
        folder=root/fold['name']
        folder.mkdir(parents=True)
        train,target=fold['train_indices'],fold['target_indices']
        source=arrays[fold['split']]
        truth=source['costs'][target].numpy()
        offset=(1+binding['config']['seed']*.5) if executed else (2+binding['config']['seed'])
        prediction=truth+offset
        np.savez_compressed(folder/'predictions.npz',prediction=prediction,truth=truth,
            family=source['families'][target].numpy(),train_indices=np.asarray(train),target_indices=np.asarray(target),
            prediction_train=arrays['train']['costs'][train].numpy()+offset)
        (folder/'resume.pt').write_bytes(b'never-loaded-synthetic-checkpoint')
        costs,mask=arrays['train']['costs'][train],arrays['train']['mask'][train]
        mean,std,scale,count=_masked_stats(costs,mask)
        norm=dict(target_mean=mean.tolist(),target_std=std.tolist(),objective_scale=scale.tolist(),
            train_observed_count=count.tolist(),feature_mean=[float(executed)],feature_std=[1.+float(executed)],
            baseline_slope=[float(executed)]*5,baseline_intercept=[float(executed)]*5)
        done=dict(status='completed',binding=binding,variant=control.VARIANT,seed=binding['config']['seed'],
            steps=2000,train_rows=len(train),target_rows=len(target),normalization=norm,
            checkpoint_sha256=control.sha(folder/'resume.pt'))
        control.atomic_json(folder/'completed.json',done)
        return done

    def build(self):
        with mock.patch.object(torch,'load',side_effect=AssertionError('No checkpoint loads')), \
             mock.patch.object(control.screen,'fit',side_effect=AssertionError('No training')):
            return paired.build(self.plan_path)

    def test_complete_matrix_uses_saved_predictions_all_cost_heads_and_independent_seeds(self):
        result=self.build()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['tasks_complete'],6)
        self.assertEqual(len(result['fold_metrics']),120)
        self.assertFalse(result['integrity_errors'])
        self.assertEqual(result['model_forward_calls'],0)
        self.assertEqual(result['critic_refits'],0)
        self.assertFalse(result['winner_selected'])
        for name in ('fql/qstudy_report.py','fql/qstudy_executed_control.py'):
            self.assertEqual(result['report_dependency_sha256'][name],control.sha(control.ROOT/name))
        for row in result['three_seed_summary']:
            self.assertEqual(row['raw']['defined_seeds'],3)
            if row['statistic']=='mae':
                self.assertAlmostEqual(row['raw']['mean'],3,places=5)
                self.assertAlmostEqual(row['raw']['sd'],1,places=5)
                self.assertAlmostEqual(row['executed']['mean'],1.5,places=5)
                self.assertAlmostEqual(row['delta_executed_minus_raw']['mean'],-1.5,places=5)
                self.assertAlmostEqual(row['delta_executed_minus_raw']['sd'],.5,places=5)
        self.assertEqual(result['primary_metrics'],['WNS_ns','TNS_ns'])
        self.assertIn('max(-native_WNS_ns,0)',result['definitions']['WNS_ns'])

    def test_missing_fold_is_explicit_and_three_seed_summary_withheld(self):
        folder=Path(self.tasks[0]['output'])
        (folder/'completed.json').unlink()
        (folder/'cv_0/completed.json').unlink()
        (folder/'cv_0/verified.json').unlink()
        result=self.build()
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['tasks_complete'],5)
        self.assertFalse(result['integrity_errors'])
        state=next(s for s in result['fit_status'] if s['dataset']=='bp_fe384' and s['seed']==0 and s['fit']=='cv_0')
        self.assertEqual(state['status'],'missing_or_awaiting_inventory')
        for row in result['three_seed_summary']:
            if row['dataset']=='bp_fe384' and row['scope']=='grouped_train_cv':
                self.assertIsNone(row['raw']['mean'])
                self.assertEqual(row['raw']['defined_seeds'],2)
        self.assertTrue(any(r['complete'] for r in result['seed_metrics'] if r['scope']=='historical_refit_development'))

    def test_changed_executed_prediction_bytes_rejects_entire_task(self):
        path=Path(self.tasks[0]['output'])/'cv_0/predictions.npz'
        path.write_bytes(path.read_bytes()+b'changed')
        result=self.build()
        self.assertEqual(result['status'],'invalid_artifacts')
        self.assertTrue(result['integrity_errors'])
        self.assertFalse(any(r['dataset']=='bp_fe384' and r['seed']==0 for r in result['fold_metrics']))

    def test_changed_raw_prediction_or_labels_invalidates_declared_comparator(self):
        path=Path(self.tasks[0]['raw_output'])/'cv_0/predictions.npz'
        with np.load(path,allow_pickle=False) as p:values={k:p[k].copy() for k in p.files}
        values['prediction'][0,0]+=.5
        np.savez_compressed(path,**values)
        result=self.build()
        self.assertEqual(result['status'],'invalid_artifacts')
        self.assertTrue(any('Declared dataset/raw comparator/folds changed' in e['error'] for e in result['integrity_errors']))

    def test_coordinate_prior_changes_and_cross_fold_collisions_are_disclosed(self):
        result=self.build()
        self.assertTrue(all(s['representation_effects']['feature_mean_changed'] for s in result['fit_status']))
        self.assertTrue(all(s['representation_effects']['hpwl_prior_slope_changed'] for s in result['fit_status']))
        self.assertTrue(all(s['target_normalization_identical'] for s in result['fit_status']))
        self.assertGreater(sum(s['executed_target_rows_also_in_train'] for s in result['fit_status']),0)
        self.assertEqual(result['task_status'][0]['declared_failure_accounting']['train']['unique_executed_rows'],10)

    def test_changed_normalization_and_final_inventory_are_refused(self):
        folder=Path(self.tasks[0]['output'])
        path=folder/'cv_0/completed.json'
        done=control.read_json(path)
        done['normalization']['objective_scale'][0]*=2
        control.atomic_json(path,done)
        result=self.build()
        self.assertEqual(result['status'],'invalid_artifacts')
        self.assertTrue(any('normalization' in e['error'] for e in result['integrity_errors']))

    def test_final_task_inventory_tampering_withholds_task(self):
        path=Path(self.tasks[0]['output'])/'completed.json'
        final=control.read_json(path)
        final['fits'][0]['checkpoint_sha256']='changed'
        control.atomic_json(path,final)
        result=self.build()
        self.assertEqual(result['status'],'invalid_artifacts')
        self.assertTrue(any('four-fit task completion' in e['error'] for e in result['integrity_errors']))
        self.assertFalse(any(r['dataset']=='bp_fe384' and r['seed']==0 for r in result['fold_metrics']))

    def test_plan_task_and_source_snapshot_tampering_refused(self):
        path=self.output/'tasks.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'tasks changed'):self.build()
        self.plan['tasks_sha256']=control.sha(path)
        control.atomic_json(self.plan_path,self.plan)
        snapshot=self.output/'source_snapshot/fql/qstudy_screen.py'
        snapshot.write_bytes(snapshot.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'snapshot changed'):self.build()

    def test_fold_scores_are_averaged_before_seed_sd_and_undefined_rank_is_null(self):
        statuses=[]
        rows=[]
        for seed in range(3):
            for fold in range(3):
                statuses.append(dict(dataset='bp_fe384',seed=seed,fit=f'cv_{fold}',status='verified'))
                row=dict(dataset='bp_fe384',seed=seed,fit=f'cv_{fold}',scope='grouped_train_cv',metric='WNS_ns')
                a={key:float(1+fold*2) for key in paired.STATISTICS}
                b={key:a[key]+(seed+1)*2 for key in paired.STATISTICS}
                a['spearman']=None
                row.update(raw=a,executed=b,delta_executed_minus_raw={key:b[key]-a[key] if a[key] is not None else None
                    for key in paired.STATISTICS})
                rows.append(row)
        seeds,summary=paired.aggregate(rows,statuses)
        target=next(r for r in summary if r['dataset']=='bp_fe384' and r['scope']=='grouped_train_cv'
            and r['metric']=='WNS_ns' and r['statistic']=='mae')
        self.assertEqual(target['raw']['mean'],3)
        self.assertEqual(target['delta_executed_minus_raw']['mean'],4)
        self.assertEqual(target['delta_executed_minus_raw']['sd'],2)
        target=next(r for r in summary if r['dataset']=='bp_fe384' and r['scope']=='grouped_train_cv'
            and r['metric']=='WNS_ns' and r['statistic']=='spearman')
        self.assertIsNone(target['raw']['mean'])
        self.assertIsNone(target['delta_executed_minus_raw']['mean'])


if __name__=='__main__':unittest.main()
