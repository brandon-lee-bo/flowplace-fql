"""Finite synthetic utility/statistics tests; no fresh policy or physical call."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_fresh_report as report
from . import qstudy_fresh_eval as fresh
from . import test_qstudy_fresh_native as fixtures
from .qstudy_bounded_utility import make_spec


class FreshReportTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run numerical summaries with bounded CPU Slurm')
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.root=Path(self.temp.name)
        self.spec=make_spec([1,.15,.35,0,0],[100,1,10,3,4],dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64))
        self.streams=[dict(id='candidate',purpose='candidate',kind='one_step',seeds=[0,1,2]),
            dict(id='bc',purpose='bc',kind='one_step',seeds=[0,1,2]),
            dict(id='public',purpose='public_reference',kind='public_official20',seeds=[0])]

    def tearDown(self):self.temp.cleanup()

    def saved_record(self,stream,seed,offset=0):
        rows=[]
        for i in range(8):
            metrics=dict(zip(report.native.METRICS,[100+offset+seed+i,-.2,-3,1,2,.2,1000]))
            row=dict(index=i,state='success',metrics=metrics,noise_sha256=f'noise{i}',
                bounded={})
            row['bounded']={'timing':report.measured(row,self.spec)};rows.append(row)
        return dict(stream=stream['id'],kind=stream['kind'],purpose=stream['purpose'],seed=seed,
            planned=8,exported=8,not_exported=0,success=8,geometry_failure=0,physical_failure=0,missing=0,invalid=0,
            fully_accounted=True,conditional_PPA=False,
            actual_noise_by_index={str(i):f'noise{i}' for i in range(8)},rows=rows,
            successful_native_metric_means={k:report.native.mean([r['metrics'][k] for r in rows]) for k in report.native.METRICS},
            generation_noise_statistics_success_only={k:report.native.describe([r['metrics'][k] for r in rows]) for k in report.native.METRICS},
            measured_objectives={'timing':dict(B_all_request_mean=report.native.mean([r['bounded']['timing']['B'] for r in rows]),
                utility_all_request_mean=report.native.mean([r['bounded']['timing']['utility'] for r in rows]))})

    def test_explicit_locked_specs_common_reference_and_no_fitting(self):
        path=self.root/'utility.json'
        fresh.atomic_json(path,dict(version=report.UTILITY_VERSION,utility_specs={'timing':self.spec,'wl':dict(self.spec,weights=[1.,0,0,0,0])}))
        lock=dict(common_train_utility_reference=fresh.binding(path))
        self.assertEqual(set(report.utility_specs(lock)),{'timing','wl'})
        fresh.atomic_json(path,dict(TRAIN_scales=[100,1,10,3,4],weights=[1,0,0,0,0]))
        lock['common_train_utility_reference']=fresh.binding(path)
        with self.assertRaises(ValueError):report.utility_specs(lock)
        bad=copy.deepcopy(self.spec);bad['scales'][0]=200
        fresh.atomic_json(path,dict(version=report.UTILITY_VERSION,utility_specs={'a':self.spec,'b':bad}))
        lock['common_train_utility_reference']=fresh.binding(path)
        with self.assertRaises(ValueError):report.utility_specs(lock)

    def test_native_signed_slacks_same_utility_and_invalid_no_target(self):
        row=dict(state='success',metrics=dict(zip(report.native.METRICS,[100,-1,-10,20,30,.2,1000])))
        result=report.measured(row,self.spec)
        self.assertAlmostEqual(result['C'],1.5)
        self.assertAlmostEqual(result['B'],1.2)
        self.assertAlmostEqual(result['utility'],-1.2)
        self.assertEqual(report.measured(dict(state='physical_failure',metrics=None),self.spec)['B'],4)
        for state in ['missing','invalid']:
            self.assertEqual(report.measured(dict(state=state,metrics=None),self.spec),
                dict(valid=False,C=None,B=None,utility=None))

    def test_three_independent_seed_means_and_single_public_noise_SD_distinct(self):
        current=[self.saved_record(self.streams[0],s) for s in range(3)]
        result=report.aggregate(current,self.streams[0],{'timing':self.spec})
        self.assertTrue(result['aggregate_eligible']);self.assertEqual(result['independent_training_seed_n'],3)
        self.assertAlmostEqual(result['metrics']['rWL_um']['mean'],104.5)
        self.assertAlmostEqual(result['metrics']['rWL_um']['sd'],1.)
        public=report.aggregate([self.saved_record(self.streams[2],0)],self.streams[2],{'timing':self.spec})
        self.assertEqual(public['independent_training_seed_n'],0)
        self.assertEqual(public['public_frozen_checkpoint_n'],1)
        self.assertEqual(public['metrics']['rWL_um']['n'],8)
        self.assertIsNone(public['objectives']['timing']['training_seed_sd'])
        self.assertIn('noise outputs',public['uncertainty'])

    def test_missing_eighth_request_withholds_complete_three_seed_means(self):
        records=[self.saved_record(self.streams[0],s) for s in range(3)]
        records[1].update(fully_accounted=False,missing=1,success=7,conditional_PPA=True)
        result=report.aggregate(records,self.streams[0],{'timing':self.spec})
        self.assertFalse(result['aggregate_eligible']);self.assertIsNone(result['metrics'])
        self.assertIsNone(result['objectives']);self.assertEqual(result['request_accounting']['planned'],24)
        self.assertEqual(result['request_accounting']['missing'],1)

    def test_same_noise_paired_native_deltas_and_partial_mean_withholding(self):
        records=[]
        for stream in self.streams:
            for seed in stream['seeds']:
                records.append(self.saved_record(stream,seed,offset=-2 if stream['purpose']=='candidate' else 0))
        pairs,seeds,summary=report.paired(records,dict(streams=self.streams),{'timing':self.spec})
        matched=next(r for r in summary if r['reference']=='bc')
        self.assertAlmostEqual(matched['metric_deltas']['rWL_um']['mean'],-2.)
        self.assertTrue(matched['aggregate_eligible'])
        self.assertTrue(matched['metric_deltas_conditioned_on_both_success'])
        public=next(r for r in summary if r['reference']=='public')
        self.assertIn('ONE fixed public',public['interpretation'])
        records[0]['fully_accounted']=False
        summary=report.paired(records,dict(streams=self.streams),{'timing':self.spec})[2]
        self.assertFalse(next(r for r in summary if r['reference']=='bc')['aggregate_eligible'])
        records[0]['rows'][0]['noise_sha256']='anothernoise'
        with self.assertRaisesRegex(ValueError,'same actual saved noise'):
            report.paired(records,dict(streams=self.streams),{'timing':self.spec})

    def test_complete80_real_geometry_failures_preserve_denominator_no_fake_PPA(self):
        fixture=fixtures.FreshNativeTests('test_real_native_geometry_failure_has_B4_no_PPA_and_never_calls_EDA')
        fixture.setUp()
        try:
            for _,_,rows,tasks in fixture.exports:
                for row in rows:fixture.geometry_outcome(row,tasks)
            with patch.object(report.finish.backend,'measure',side_effect=AssertionError('EDA forbidden')), \
                 patch.object(fresh,'build_sampler',side_effect=AssertionError('policy forbidden')), \
                 patch.object(fresh,'declare',side_effect=AssertionError('declaration writer forbidden')):
                result=report.collect(fixture.f.path,fixture.f.digest)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['expected_requests'],80)
            self.assertEqual(sum(r['geometry_failure'] for r in result['streams']),80)
            self.assertTrue(all(r['fully_accounted'] for r in result['streams']))
            self.assertTrue(all(r['successful_PPA_denominator']==0 for r in result['streams']))
            self.assertTrue(all(r['measured_objectives']['wl']['B_all_request_mean']==4 for r in result['streams']))
            self.assertTrue(all(all(v is None for v in r['successful_native_metric_means'].values()) for r in result['streams']))
            self.assertEqual(result['call_accounting']['unique_physical_calls_with_manifest'],0)
            self.assertEqual(result['call_accounting']['logical_exported_requests'],80)
            self.assertEqual(result['backend_calls'],0)
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main()
