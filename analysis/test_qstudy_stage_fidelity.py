"""Synthetic checks of native-stage pairing, ties, and fixed denominators."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis import qstudy_stage_fidelity as m


class FidelityTests(unittest.TestCase):
    def rows(self):
        result=[]
        for arm in ('bc','legacy'):
            for r in (0,2):
                for seed in (0,1,2):
                    for i in range(4):
                        v=100+seed*10+r+i+(arm=='legacy')
                        result.append(dict(arm=arm,round=r,seed=seed,noise_index=i,
                            noise_sha256='noise'+str(i),actual_macro_geometry_identity=f'{arm}-{r}-{seed}-{i}',
                            GRT={k:v for k in m.METRICS},DRT={k:2*v+1 for k in m.METRICS}))
        return result

    def test_tied_ranks_are_average_and_constant_undefined(self):
        np.testing.assert_array_equal(m.rank([3,1,1,9]),[3,1.5,1.5,4])
        self.assertIsNone(m.correlation([1,1],[2,3]))
        self.assertIsNone(m.correlation([],[]))

    def test_all48_layouts_and12_strata_no_policy_mean_rows(self):
        result=m.summarize(self.rows())
        self.assertEqual(result['completed_same_layout_pairs'],48)
        self.assertEqual(len(result['strata']),12)
        self.assertTrue(all(r['planned']==r['associations']['WL']['n']==4 for r in result['strata']))
        self.assertEqual(result['pooled_layout_level']['WL']['n'],48)
        self.assertAlmostEqual(result['pooled_layout_level']['WL']['spearman'],1)
        self.assertEqual(result['matched_contrasts_completed'],24)
        self.assertEqual(result['matched_policy_direction']['WL']['concordant'],24)

    def test_within_stratum_separates_between_group_reversal(self):
        rows=self.rows()
        for j,row in enumerate(rows):
            group=j//4
            for k in m.METRICS:
                row['GRT'][k]=100*group+row['noise_index']
                row['DRT'][k]=-100*group+row['noise_index']
        result=m.summarize(rows)
        self.assertLess(result['pooled_layout_level']['WL']['spearman'],0)
        self.assertAlmostEqual(result['within_stratum_centered_rank_correlation']['WL']['correlation'],1)

    def test_ties_not_automatically_agreement(self):
        result=m.direction_counts([1,1,0,1,0],[2,-1,1,0,0])
        self.assertEqual(result,dict(total=5,concordant=1,discordant=1,both_tied=1,
            GRT_tied_only=1,DRT_tied_only=1,non_tied_pairs=2,agreement_excluding_ties=.5))

    def test_missing_layout_stays_missing_denominator(self):
        rows=self.rows()[:-1]
        result=m.summarize(rows)
        self.assertFalse(result['complete'])
        self.assertEqual(result['planned_same_layout_pairs'],48)
        self.assertEqual(result['matched_contrasts_planned'],24)
        self.assertEqual(result['matched_contrasts_completed'],23)
        self.assertEqual(sum(r['complete'] for r in result['strata']),11)

    def test_policy_comparison_requires_actual_same_noise(self):
        rows=self.rows()
        rows[-1]['noise_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'actual noise'):
            m.summarize(rows)

    def pair_fixture(self,folder):
        g={spec[0]:10. for spec in m.METRICS.values()}
        d={spec[1]:11. for spec in m.METRICS.values()};d['DRC']=0
        (folder/'backend_macro_audit.json').write_text(json.dumps(dict(legal_no_halo=True,
            actual_macros={'A':{'bbox_dbu':[0,0,1,1]}},dbu_per_um=1000,core_dbu=[0,0,10,10])))
        task=dict(key='bc_s0_r0_n0',arm='bc',round=0,seed=0,noise_index=0)
        b=dict(source_outcome_sha256='source',GRT_metrics=g,native={'physical_run':str(folder)},
            noise_sha256='noise',raw_action_sha256='raw',action_sha256='executed',def_sha256='def')
        source=dict(state='resolved',binding=b,binding_sha256='binding')
        final=dict(state='completed',source_binding_sha256='binding',source_outcome_sha256='source',
            final_native_audit=dict(source_metrics=g,source_run=str(folder),
                actual_macro_DEF_reparsed=True,final_def_sha256='finaldef'),metrics=d,
            outcome_sha256='out',physical_run=str(folder/'drt'))
        return task,source,final

    def test_exact_geometry_source_and_completed_report_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,source,final=self.pair_fixture(Path(tmp))
            row=m.matched_layout(task,source,final,copy.deepcopy(final))
            self.assertEqual(row['GRT']['WL'],10)
            self.assertEqual(row['DRT']['WL'],11)
            self.assertEqual(row['action_sha256'],'executed')
            self.assertEqual(row['DRT_DRC'],0)

    def test_saved_report_metric_or_source_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,source,final=self.pair_fixture(Path(tmp))
            wrong=copy.deepcopy(final);wrong['metrics']['DRT_WL']=12
            with self.assertRaisesRegex(ValueError,'completed DRT report'):
                m.matched_layout(task,source,final,wrong)
            final['final_native_audit']['source_run']=str(Path(tmp)/'other')
            with self.assertRaisesRegex(ValueError,'executed geometry'):
                m.matched_layout(task,source,final,copy.deepcopy(final))

    def test_failure_has_no_numeric_stage_pair(self):
        failed=dict(state='upstream_physical_failure',metrics=None)
        self.assertIsNone(m.matched_layout({},dict(state='resolved'),failed,copy.deepcopy(failed)))

    def dataset_fixture(self,root):
        folder=root/'audited';folder.mkdir()
        path=folder/'dataset.npz';path.write_bytes(b'fixture')
        initial=root/'datasets/bp_fe_grt_v1/manifest.json';initial.parent.mkdir(parents=True)
        initial.write_text(json.dumps(dict(source_proxy_manifest={'provenance':{'checkpoint_sha256':'public'}})))
        common=root/'common.pt';common.write_bytes(b'no tensor loading')
        config=root/'config.json'
        config.write_text(json.dumps(dict(dataset_path=str(path),dataset_sha256=m.sha(path),
            common_checkpoints={'0':str(common)},common_checkpoint_sha256={'0':m.sha(common)})))
        data={'behavior_action':np.zeros((256,2))};rows=[]
        for family,split in enumerate(('train','validation')):
            raw=np.asarray([[.1,.2]],dtype=np.float32);executed=np.asarray([[.2,.3]],dtype=np.float32)
            metrics=np.ones((1,7))
            data[split]=dict(raw_action=raw,executed_action=executed,metrics=metrics,
                metric_mask=np.ones((1,7),dtype=bool),family=np.asarray([family]),variant=np.asarray([0]))
            rows.append(dict(split=split,raw_action_sha256=m.action_sha(raw[0]),
                action_sha256=m.action_sha(executed[0]),family=family,variant=0,
                metrics={k:1 for k in m.METRIC_NAMES},collection_kind='public',status='success',
                actual_backend_geometry={'legal':True}))
        rp=folder/'rows.jsonl';rp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=dict(dataset_sha256=m.sha(path),rows_sha256=m.sha(rp),
            source_bindings=[{'path':str(initial),'sha256':m.sha(initial)}],
            executed_action_semantics='external legalizer before native snapping')
        (folder/'manifest.json').write_text(json.dumps(manifest))
        return path,config,data,manifest

    def test_dataset_audit_distinguishes_measured_target_and_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config,data,manifest=self.dataset_fixture(root)
            with patch.object(m,'ROOT',root),patch.object(m,'load_dataset',return_value=(data,manifest)):
                result=m.audit_dataset(path,config)
            self.assertEqual(result['splits']['train']['rows'],1)
            self.assertEqual(result['splits']['train']['raw_executed_unequal_rows'],1)
            self.assertEqual(result['family_intersection'],[])
            self.assertEqual(result['separate_behavior_targets_count'],256)
            self.assertTrue(result['DRT_evaluation_not_merged'])

    def test_dataset_metric_row_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,config,data,manifest=self.dataset_fixture(root)
            data['train']['metrics'][0,0]=999
            with patch.object(m,'ROOT',root),patch.object(m,'load_dataset',return_value=(data,manifest)):
                with self.assertRaisesRegex(ValueError,'metric correspondence'):
                    m.audit_dataset(path,config)


if __name__=='__main__':
    unittest.main()
