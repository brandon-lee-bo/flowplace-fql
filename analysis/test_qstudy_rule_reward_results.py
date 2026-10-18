"""Saved-only reporting tests using96 tiny real fits; CPU Slurm only."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from analysis import qstudy_rule_reward_results as report
from fql import qstudy_rule_reward_screen as screen
from fql.test_qstudy_rule_reward_screen import RuleRewardTests as Fixture
from fql.qstudy_data import sha


class RuleResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run reporter fixtures through CPU Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.fixture = Fixture('test_exact_eight_objectives_and_lambdas')
        cls.fixture.setUp()
        cls.root = cls.fixture.root
        cls.data = copy.deepcopy(cls.fixture.data)
        # Deliberately varying narrow but nonoverlapping channels. All proxy
        # variation is now actual geometry, not the screen fixture override.
        executed = np.zeros((12,3,2),np.float32)
        for i in range(12):
            executed[i,0] = [-.5+.005*i,-.35+.007*i]
            executed[i,1] = [executed[i,0,0]+.202+.0015*i,executed[i,0,1]]
            executed[i,2] = [.2+.01*i,.4-.009*i]
        for split,shift in [('train',0.),('validation',.003)]:
            source = cls.data[split]
            source['executed'] = (executed+shift).reshape(12,-1)
            source['x'] = torch.from_numpy(executed+shift+.01).float()
            if split == 'validation':
                source['families'] = source['families']+100
                source['source_ids'] = ['validation_'+x for x in source['source_ids']]
        screen.add_costs(cls.data,cls.fixture.cache)
        cls.context = cls.data['train']
        cls.config = dict(screen.CONFIG,steps=3)
        cls.patch_config = mock.patch.dict(screen.CONFIG,cls.config,clear=True)
        cls.patch_config.start()
        folds = screen.folds_for(cls.data)
        dataset = cls.root/'data'/'dataset.npz'
        dataset.parent.mkdir()
        dataset.write_bytes(b'synthetic identity only')
        cls.plan = dict(schema=screen.SCHEMA,output=str(cls.root/'study'),
            inputs=dict(dataset_path=str(dataset),dataset_sha256='a'*64),tasks=screen.tasks(),
            folds=folds,source_sha256={},fold_plan_sha256=screen.json_sha(folds))
        cls.plan_path = cls.root/'plan.json'
        cls.plan_path.write_text(json.dumps(cls.plan))
        with contextlib.redirect_stdout(io.StringIO()):
            for task in screen.tasks():
                context = copy.deepcopy(cls.context)
                context['binding'] = screen.run_binding(cls.plan,cls.plan_path,task)
                folder=Path(cls.plan['output'])/task['objective']/f"seed_{task['seed']}"
                folder.mkdir(parents=True)
                args=SimpleNamespace(**cls.config,**task,output=folder)
                outcomes=[]
                for fold in folds:
                    target=cls.data['validation'] if fold['name']=='refit' else context
                    outcomes.append(screen.fit(args,context,fold['name'],fold['train_indices'],fold['target_indices'],target))
                summary=dict(status='completed',binding=context['binding'],objective=task['objective'],seed=task['seed'],
                    outcomes=outcomes,**screen.summarize(outcomes))
                screen.bounded.atomic_json(folder/'summary.json',summary)
                screen.bounded.atomic_json(folder/'completed.json',dict(status='completed',binding=context['binding'],
                    summary_sha256=sha(folder/'summary.json'),backend_calls=0,actor_advanced=False))
        cls.loaded = (cls.plan,cls.data,cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.patch_config.stop()
        cls.fixture.tearDown()

    def collect(self):
        with mock.patch.object(screen,'checked_plan',return_value=self.loaded):
            return report.collect(self.plan_path)

    @contextlib.contextmanager
    def changed_file(self,path,value):
        path=Path(path);original=path.read_bytes()
        try:
            path.write_bytes(value)
            yield
        finally:path.write_bytes(original)

    def test_full96_saved_only_and_common_streams(self):
        with mock.patch.object(screen.MultiMetricCritic,'forward',side_effect=AssertionError('No forward')):
            result=self.collect()
        self.assertEqual((result['status'],result['completed_tasks'],result['verified_fits']),('completed',24,96))
        self.assertEqual(result['all_objectives_retained'],list(screen.OBJECTIVES))
        self.assertTrue(all(x['all_eight_available'] and x['identities_equal'] for x in result['paired_streams']))
        self.assertEqual((result['model_inference_calls'],result['backend_calls']), (0,0))
        self.assertEqual(len(result['train_only_scales']),3*4*6)
        self.assertEqual(len(result['seed_summaries']),8*3*2)

    def test_partial_task_withholds_all_four_fits_without_physical_failure(self):
        marker=Path(self.plan['output'])/'grt_wl/seed_0/completed.json'
        original=marker.read_bytes()
        marker.unlink()
        try:result=self.collect()
        finally:marker.write_bytes(original)
        self.assertEqual((result['status'],result['verified_fits']),('partial',92))
        self.assertEqual(result['tasks'][0]['status'],'missing')
        self.assertFalse(any(r['objective']=='grt_wl' and r['seed']==0 for r in result['fits']))

    def test_corrupt_late_fit_invalidates_entire_task(self):
        path=Path(self.plan['output'])/'grt_wl/seed_0/refit/predictions.npz'
        with self.changed_file(path,path.read_bytes()+b'changed'):
            result=self.collect()
        self.assertEqual(result['integrity_errors'],1)
        self.assertEqual(result['verified_fits'],92)
        self.assertEqual(result['tasks'][0]['status'],'integrity_error')

    def test_stored_statistic_tamper_is_recomputed(self):
        task=self.plan['tasks'][0];fold=self.plan['folds'][0]
        folder=Path(self.plan['output'])/'grt_wl/seed_0/cv_0'
        path=folder/'completed.json';value=report.read(path);value['critic']['mae']=100.
        with self.changed_file(path,json.dumps(value).encode()):
            with self.assertRaisesRegex(ValueError,'diagnostics disagree'):
                report.fit_evidence(folder,fold,task,self.plan,self.data,self.context)

    def test_paired_stream_mismatch_refused(self):
        a=dict(objective='grt_wl',seed=0,fit='cv_0',paired_stream={'same':1})
        b=dict(objective='macro_hpwl',seed=0,fit='cv_0',paired_stream={'same':2})
        with self.assertRaisesRegex(ValueError,'paired|Paired'):
            report.paired_streams([a,b])

    def test_constant_comparator_not_three_training_replications(self):
        result=self.collect()
        row=next(r for r in result['multi_seed'] if r['objective']=='grt_wl' and r['scope']==report.SCOPES[0]
                 and r['arm']=='constant' and r['statistic']=='mae')
        self.assertEqual((row['n'],row['independent_training_runs'],row['sd']),(1,0,None))
        self.assertEqual(row['verified_replica_seeds'],[0,1,2])

    def test_cv_aggregation_is_equal_fold_not_pooled_truth(self):
        fits=[]
        for i,value in enumerate([.1,.3,.8]):
            stat=dict(mae=value,spearman=value,kendall=value,family_pair_accuracy=value,top_regret=value,
                      family_pairs=i+1)
            fits.append(dict(objective='grt_wl',seed=0,fit=f'cv_{i}',scope=report.SCOPES[0],
                critic=stat,constant=stat,family_mean=stat,
                validity={k:1 for k in ['target_rows','valid_target_rows','invalid_target_rows','confirmed_failure']},
                prediction_range={k:0 for k in ['success_below_zero','success_at_or_above_two','any_below_zero','any_above_four']},
                input_gradients=dict(rows=2,nonzero_rows=2,rms_mean=1.,rms_maximum=2.)))
        row=report.seed_summaries(fits)[0]
        self.assertAlmostEqual(row['critic']['mae'],.4)
        self.assertAlmostEqual(row['critic']['within_fold_pair_accuracy'],(.1+.6+2.4)/6)

    def test_geometry_uses_actual_movable_mask_and_executed_costs(self):
        context=copy.deepcopy(self.context);context['movable']=torch.tensor([True,False,True])
        result=report.geometry_audit(self.data,context)
        self.assertEqual(result['movable_nodes'],2)
        self.assertEqual(result['changed_macros'],24)
        self.assertAlmostEqual(result['row_rms']['mean'],.01,places=6)
        self.assertEqual(result['group_saturation']['non_singleton_clusters'],1)
        altered=copy.deepcopy(self.data);altered['train']['cost_values'][0,2]+=.1
        with self.assertRaisesRegex(ValueError,'executed geometric'):
            report.geometry_audit(altered,context)

    def test_familymeans_keep_finite_sibling_and_constant_rank_undefined(self):
        value=report.family_mean_diagnostics(np.array([1.,np.nan,2.,3.]),np.array([1.,2.,2.,3.]),np.array([0,0,1,2]))
        self.assertEqual(value['n_families'],3)
        self.assertEqual(value['spearman'],1.)
        constant=report.diagnostics(np.ones(4),np.arange(4.),np.repeat([0,1],2))
        self.assertIsNone(constant['spearman'])
        self.assertEqual(constant['family_pair_accuracy'],0.)

    def test_snapshot_is_immutable_and_source_output_guard(self):
        with mock.patch.object(screen,'checked_plan',return_value=self.loaded):
            output=self.root/'reports'
            first=report.write_snapshot(self.plan_path,output)
            second=report.write_snapshot(self.plan_path,output)
            self.assertNotEqual(first['snapshot'],second['snapshot'])
            self.assertEqual(report.read(output/'latest.json')['report'],second['report'])
            self.assertEqual(sha(first['report']['path']),first['report']['sha256'])
            with self.assertRaisesRegex(ValueError,'separate'):
                report.write_snapshot(self.plan_path,Path(self.plan['output'])/'report')


if __name__=='__main__':unittest.main()
