"""Bounded CPU Slurm tests with 36 tiny, real, immutable critic fixtures."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_bounded_screen as screen
from . import qstudy_bounded_screen_report as report
from .graph_o2o_archive import atomic_json,sha
from .qstudy_bounded_utility import assert_common_scale
from .qstudy_data import METRIC_NAMES


class BoundedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run numerical report tests through bounded CPU Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temporary=tempfile.TemporaryDirectory(prefix='bounded-report-test-',dir=os.environ['TMPDIR'])
        cls.root=Path(cls.temporary.name)
        cls.original_config=screen.CONFIG
        screen.CONFIG=dict(screen.CONFIG,steps=3,batch_size=8)
        cls.cache=dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.2) for n in 'abc'},
            net_info={'n':{'nodes':{'a':{'x_offset':.2},'b':{},'c':{'y_offset':-.3}}}})
        gen=torch.Generator().manual_seed(19)
        x=torch.rand((12,3,2),generator=gen)*1.2-.6
        native=torch.stack((100+3*x[:,0,0],-(.4+.1*x[:,1,0]),-(25+x[:,2,1]),
            (3*x[:,0,1]).clamp_min(0),(4*x[:,1,1]).clamp_min(0),
            torch.ones(12)*.2,torch.ones(12)*1000),-1).double().numpy()
        cls.data={}
        for split,n,offset in [('train',12,0),('validation',4,10)]:
            cls.data[split]=dict(x=x[:n].clone(),native=native[:n].copy(),mask=np.ones((n,7),bool),
                executed=(x[:n]*.9).numpy().reshape(n,-1),statuses=np.full(n,'success',dtype='<U32'),
                families=np.arange(n//2).repeat(2)+offset,source_ids=[f'{split}_row{i}' for i in range(n)])
        cls.inputs=dict(dataset_path=str(cls.root/'data/dataset.npz'),dataset_sha256='a'*64,
            manifest_sha256='b'*64,rows_sha256='c'*64,cache_path=str(cls.root/'cache.pt'),cache_sha256='d'*64)
        folds=screen.family_folds(cls.data['train']['families'])
        folds.append(dict(name='refit',train_indices=list(range(12)),target_indices=list(range(4))))
        tasks=[dict(index=i,objective=o,seed=s) for i,(o,s) in
               enumerate((o,s) for o in screen.OBJECTIVES for s in range(3))]
        cls.plan=dict(schema='qstudy_bounded_scalar_screen_v1',output=str(cls.root/'fits'),inputs=cls.inputs,
            design='bp_fe',source_sha256={'synthetic_fixture':'immutable'},config=screen.CONFIG,
            objectives={k:list(v) for k,v in screen.OBJECTIVES.items()},tasks=tasks,folds=folds,
            fold_plan_sha256=screen.json_sha(folds),metric_names=list(METRIC_NAMES),
            train_source_ids=cls.data['train']['source_ids'],validation_source_ids=cls.data['validation']['source_ids'],
            train_source_ids_sha256=screen.json_sha(cls.data['train']['source_ids']))
        cls.plan_path=cls.root/'plan.json'
        atomic_json(cls.plan_path,cls.plan)
        with contextlib.redirect_stdout(io.StringIO()):
            for task in tasks:
                binding=dict(schema=cls.plan['schema'],**cls.inputs,plan_sha256=sha(cls.plan_path),task=task,
                    source_sha256=cls.plan['source_sha256'],fold_plan_sha256=cls.plan['fold_plan_sha256'],
                    threads=1,torch_version=torch.__version__,numpy_version=np.__version__)
                context={**cls.data['train'],'binding':binding,'cache':cls.cache,'movable':torch.ones(3,dtype=torch.bool)}
                args=argparse.Namespace(**screen.CONFIG,**task,
                    output=Path(cls.plan['output'])/task['objective']/f"seed_{task['seed']}")
                outcomes=[]
                for fold in folds:
                    target=cls.data['validation'] if fold['name']=='refit' else cls.data['train']
                    outcomes.append(screen.fit(args,context,fold['name'],fold['train_indices'],fold['target_indices'],target))
                atomic_json(args.output/'summary.json',dict(status='completed',binding=binding,
                    objective=task['objective'],seed=task['seed'],outcomes=outcomes,**screen.summarize(outcomes)))
                atomic_json(args.output/'completed.json',dict(status='completed',binding=binding,
                    summary_sha256=sha(args.output/'summary.json'),backend_calls=0,actor_advanced=False))

    @classmethod
    def tearDownClass(cls):
        screen.CONFIG=cls.original_config
        screen.STOP=False
        cls.temporary.cleanup()

    def setUp(self):
        self.restore={}

    def tearDown(self):
        for path,value in self.restore.items():
            path.write_bytes(value)

    def mutate(self,path,value=None):
        path=Path(path)
        self.restore.setdefault(path,path.read_bytes())
        if value is None:
            path.unlink()
        else:
            path.write_bytes(value)

    def collect(self):
        with mock.patch.object(report,'checked_plan',return_value=(self.plan,self.data)):
            return report.collect(self.plan_path)

    def folder(self,objective='timing',seed=0,fit='cv_0'):
        return Path(self.plan['output'])/objective/f'seed_{seed}'/fit

    def test_all_36_real_fits_are_audited_without_inference_and_aggregated_by_seed(self):
        with mock.patch.object(screen,'scalar_reward',side_effect=AssertionError('No report inference allowed')):
            result=self.collect()
        self.assertEqual((result['status'],result['completed_tasks'],result['verified_fits']),('completed',9,36))
        self.assertEqual(result['integrity_errors'],0)
        self.assertEqual(len(result['seed_summaries']),18)
        for row in result['multi_seed']:
            if row['arm']=='critic' and row['statistic']=='mae':
                self.assertEqual(row['n'],3)
                values=[r['critic']['mae'] for r in result['seed_summaries']
                        if (r['objective'],r['scope'])==(row['objective'],row['scope'])]
                self.assertAlmostEqual(row['mean'],float(np.mean(values)))
                self.assertAlmostEqual(row['sd'],float(np.std(values,ddof=1)))
            if row['arm']=='constant' and row['statistic']=='mae':
                self.assertEqual(row['n'],1)
                self.assertIsNone(row['sd'])
                self.assertEqual(row['independent_training_runs'],0)
        for name in ['cv_0','cv_1','cv_2','refit']:
            fitted=[r for r in result['fits'] if r['fit']==name]
            for row in fitted[1:]:
                assert_common_scale(fitted[0]['utility_spec'],row['utility_spec'])
        self.assertTrue(all(r['target_definition_differs'] for r in result['paired_objective_summary']))
        self.assertFalse(result['actor_advanced'])

    def test_cv_summary_is_mean_of_fold_metrics_not_pooled_different_B_scales(self):
        result=self.collect()
        row=next(r for r in result['seed_summaries'] if
                 (r['objective'],r['seed'],r['scope'])==('timing',0,report.SCOPES[0]))
        folds=[r for r in result['fits'] if r['objective']=='timing' and r['seed']==0 and r['fit'].startswith('cv_')]
        self.assertAlmostEqual(row['critic']['mae'],float(np.mean([r['critic']['mae'] for r in folds])))
        self.assertEqual(row['validity']['rows'],12)
        self.assertEqual(row['native_fixed_pool_metrics']['rWL_um']['observed'],12)
        self.assertAlmostEqual(row['native_fixed_pool_metrics']['rWL_um']['success_mean'],self.data['train']['native'][:,0].mean())
        self.assertIn('NEVER pool',result['conventions']['cv'])
        self.assertEqual(len({tuple(r['utility_spec']['scales']) for r in folds}),3)

    def test_late_corrupt_fit_invalidates_all_earlier_evidence_of_that_task(self):
        folder=self.folder(fit='cv_2')
        self.mutate(folder/'predictions.npz',(folder/'predictions.npz').read_bytes()+b'tamper')
        result=self.collect()
        self.assertEqual((result['integrity_errors'],result['verified_fits']),(1,32))
        self.assertFalse(any(r['objective']=='timing' and r['seed']==0 for r in result['fits']))
        paired=[r for r in result['paired_objective_seed_deltas'] if r['first']=='wl' and r['second']=='timing']
        self.assertTrue(paired)
        self.assertEqual({r['seed'] for r in paired},{1,2})

    def test_missing_work_is_partial_never_assigned_failure_or_fabricated_score(self):
        folder=self.folder()
        self.mutate(folder/'completed.json')
        self.mutate(folder.parent/'completed.json')
        result=self.collect()
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['integrity_errors'],0)
        task=next(t for t in result['tasks'] if t['objective']=='timing' and t['seed']==0)
        self.assertEqual(task['missing_fits'],['cv_0'])
        self.assertFalse(any(r['objective']=='timing' and r['seed']==0 and r['scope']==report.SCOPES[0]
                             for r in result['seed_summaries']))
        self.assertEqual(result['verified_fits'],35)

    def test_native_labels_cannot_be_changed_even_with_recomputed_prediction_SHA(self):
        folder=self.folder()
        prediction=folder/'predictions.npz'
        done_path=folder/'completed.json'
        with np.load(prediction,allow_pickle=False) as source:
            arrays={k:source[k].copy() for k in source.files}
        arrays['native_metrics'][0,5]*=2
        self.mutate(prediction,prediction.read_bytes())
        np.savez_compressed(prediction,**arrays)
        done=json.loads(done_path.read_text())
        done['predictions_sha256']=sha(prediction)
        self.mutate(done_path,json.dumps(done).encode())
        result=self.collect()
        task=next(t for t in result['tasks'] if t['objective']=='timing' and t['seed']==0)
        self.assertEqual(task['status'],'integrity_error')
        self.assertIn('native_metrics',task['error'])

    def test_checkpoint_normalization_and_step_are_not_trusted_from_summary_alone(self):
        folder=self.folder()
        checkpoint=folder/'resume.pt'
        done_path=folder/'completed.json'
        saved=torch.load(checkpoint,weights_only=False)
        saved['step']+=1
        self.mutate(checkpoint,checkpoint.read_bytes())
        torch.save(saved,checkpoint)
        done=json.loads(done_path.read_text())
        done['checkpoint_sha256']=sha(checkpoint)
        self.mutate(done_path,json.dumps(done).encode())
        result=self.collect()
        task=next(t for t in result['tasks'] if t['objective']=='timing' and t['seed']==0)
        self.assertIn('Checkpoint budget',task['error'])
        self.assertEqual(task['status'],'integrity_error')

    def test_raw_native_summary_preserves_failure_missingness_and_signed_slacks(self):
        native=self.data['validation']['native'].copy()
        native[:2]=np.nan
        mask=np.isfinite(native)
        statuses=np.asarray(['physical_failure','infrastructure_failure','success','success'])
        result=report.native_summary(native,mask,statuses)
        self.assertEqual(result['rows'],4)
        self.assertEqual(result['status_counts']['physical_failure'],1)
        self.assertEqual(result['metrics']['WNS_ns']['missing'],2)
        self.assertLess(result['metrics']['WNS_ns']['success_mean'],0)
        self.assertEqual(result['metrics']['rWL_um']['observed'],2)

    def test_snapshot_is_immutable_and_latest_points_to_a_new_snapshot(self):
        output=self.root/'reports'
        argv=['report','--plan',str(self.plan_path),'--output',str(output)]
        with mock.patch('sys.argv',argv),mock.patch.object(report,'checked_plan',return_value=(self.plan,self.data)),\
             contextlib.redirect_stdout(io.StringIO()):
            report.main()
            first=json.loads((output/'latest.json').read_text())
            first_path=Path(first['snapshot'])
            hashes={p.name:sha(p) for p in first_path.iterdir() if p.is_file()}
            report.main()
        latest=json.loads((output/'latest.json').read_text())
        self.assertNotEqual(first['snapshot'],latest['snapshot'])
        self.assertEqual(hashes,{p.name:sha(p) for p in first_path.iterdir() if p.is_file()})
        self.assertEqual(first['report_sha256'],sha(first_path/'report.json'))
        self.assertEqual(sha(first_path/'plan.json'),sha(self.plan_path))
        self.assertEqual(sha(first_path/'reporter_source.py'),sha(Path(report.__file__)))


if __name__=='__main__':
    unittest.main()
