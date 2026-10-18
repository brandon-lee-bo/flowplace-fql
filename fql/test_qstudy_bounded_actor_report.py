"""Synthetic read-only reporter tests; CPU Slurm, no model/physical calls."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import qstudy_bounded_actor_report as report
from .qstudy_bounded_utility import make_spec


def specs():
    reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64)
    return {o:make_spec(w,[100.,1.,10.,2.,3.],reference) for o,w in report.adapter.screen.OBJECTIVES.items()}


def metrics(wl=100.):
    return dict(zip(report.native.METRICS,[wl,-.3,-50.,2.,3.,.2,70000.]))


def record(arm='wl',seed=0,failure=None):
    rows=[]
    ss=specs()
    for i in range(8):
        row=dict(role='evaluation',index=i,state=failure if i==0 and failure else 'success',
            noise_sha256='noise'+str(i),metrics=metrics(90.+i),raw_to_legalizer_rms=.01,
            request_sha256='request',outcome_sha256='outcome',raw_action_sha256='raw',action_sha256='executed')
        row['predicted_q']=report.measured(row,ss[arm if arm!='BC' else 'wl'])['utility']
        row['measured_utility']=-999. # Must never enter bounded utility.
        rows.append(row)
    return dict(seed=seed,experiment='/example/'+arm,rows=rows,errors=[],training=dict(completed=True))


def streams():
    return [report.evaluation_stream(record(arm,seed),arm,specs()) for arm in report.ARMS for seed in report.SEEDS]


class BoundedActorReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.folder=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_arms_under_all_specs_and_only_own_Q_calibration(self):
        ss=specs()
        for arm in report.OBJECTIVES:
            stream=report.evaluation_stream(record(arm),arm,ss)
            self.assertEqual(stream['own_objective_q_calibration']['mae'],0.)
            self.assertEqual(stream['own_objective_q_calibration']['spearman'],1.)
            self.assertEqual(stream['own_objective_q_calibration']['layout_pair_accuracy'],1.)
            row=stream['rows'][0]
            self.assertEqual(set(row['bounded']),set(report.OBJECTIVES))
            self.assertNotEqual(row['bounded']['wl']['B'],row['bounded']['timing']['B'])
            self.assertNotIn('measured_utility',row)
            for objective in report.OBJECTIVES:
                measured=row['bounded'][objective]
                self.assertAlmostEqual(measured['B'],2*measured['C']/(1+measured['C']))
        bc=report.evaluation_stream(record('BC'),'BC',ss)
        self.assertIsNone(bc['own_objective_q_calibration'])
        self.assertTrue(all(r['predicted_scalar_q'] is None for r in bc['rows']))

    def test_failure_B4_without_PPA_and_missing_never_failure(self):
        stream=report.evaluation_stream(record(failure='physical_failure'),'wl',specs())
        row=stream['rows'][0]
        self.assertTrue(stream['fully_accounted'])
        self.assertTrue(stream['conditional_PPA'])
        self.assertIsNone(row['metrics'])
        for v in row['bounded'].values():
            self.assertEqual(v,dict(valid=True,C=None,B=4.,utility=-4.))
        missing=report.evaluation_stream(record(failure='missing'),'wl',specs())
        self.assertFalse(missing['fully_accounted'])
        self.assertEqual(missing['physical_failure'],0)
        self.assertEqual(missing['missing'],1)
        self.assertIsNone(missing['measured_objectives']['wl']['B_all_request_mean'])

    def test_absent_export_is_partial_with_fixed_eight_denominator(self):
        raw=record()
        raw.update(rows=[],training=dict(completed=False))
        result=report.evaluation_stream(raw,'wl',specs())
        self.assertFalse(result['fully_accounted'])
        self.assertFalse(result['integrity_invalid'])
        self.assertEqual((result['planned'],result['not_exported'],result['missing']),(8,8,0))

    def test_duplicate_noise_index_and_bad_native_are_integrity_invalid(self):
        raw=record()
        raw['rows'][1]['index']=0
        raw['rows'][1]['noise_sha256']=raw['rows'][0]['noise_sha256']
        raw['rows'][2]['metrics']['Power_W']=float('nan')
        raw['rows'][3]['metrics']['horizontal_overflow_sum']=-1.
        stream=report.evaluation_stream(raw,'wl',specs())
        self.assertTrue(stream['integrity_invalid'])
        self.assertFalse(stream['fully_accounted'])
        self.assertEqual(stream['invalid'],2)
        self.assertIsNone(stream['rows'][2]['metrics'])

    def test_three_equal_seed_means_and_paired_BC_is_descriptive_only(self):
        values=streams()
        arms,pairs,contrasts=report.aggregate(values)
        self.assertEqual(len(arms),4)
        self.assertEqual(len(pairs),6*3*8)
        for arm in arms:
            self.assertTrue(arm['aggregate_eligible'])
            self.assertEqual(arm['request_accounting']['planned'],24)
            self.assertEqual(arm['native_metrics']['rWL_um'],dict(n=3,mean=93.5,sd=0.,eligible=True))
        for contrast in contrasts:
            self.assertTrue(contrast['paired_native_reference_eligible'])
            self.assertEqual(contrast['controlled_effective_update_eligible'],contrast['baseline']!='BC')
            for stat in contrast['utility_deltas'].values():
                self.assertEqual(stat['mean'],0.)
                self.assertEqual(stat['sd'],0.)

    def test_zero_success_seed_not_dropped_and_unresolved_withholds_aggregate(self):
        values=streams()
        raw=record('timing',0)
        for row in raw['rows']:
            row['state']='geometry_failure'
        values=[report.evaluation_stream(raw,'timing',specs()) if (s['arm'],s['seed'])==('timing',0) else s for s in values]
        arms,_,_=report.aggregate(values)
        timing=next(a for a in arms if a['arm']=='timing')
        self.assertTrue(timing['aggregate_eligible'])
        self.assertEqual(timing['native_metrics']['rWL_um'],dict(n=2,mean=None,sd=None,eligible=False))
        self.assertEqual(timing['measured_objectives']['wl']['utility_all_request_mean']['n'],3)
        values[-1]['fully_accounted']=False
        arms,_,contrasts=report.aggregate(values)
        self.assertIsNone(next(a for a in arms if a['arm']=='congestion')['native_metrics'])
        self.assertTrue(all(not c['aggregate_eligible'] for c in contrasts if c['target']=='congestion'))

    def test_same_actual_noise_required_across_all_seeds(self):
        values=streams()
        values[2]['actual_noise_by_index']['0']='changed'
        arms,_,contrasts=report.aggregate(values)
        self.assertFalse(next(a for a in arms if a['arm']=='BC')['aggregate_eligible'])
        self.assertTrue(all(not c['aggregate_eligible'] for c in contrasts if c['baseline']=='BC'))
        with self.assertRaisesRegex(ValueError,'four arms'):
            report.aggregate(values+values[:1])

    def test_frozen_native_guard_prevents_implicit_EDA(self):
        experiment=self.folder/'actor'
        folder=experiment/'seed_0/round_0/requests'
        folder.mkdir(parents=True)
        tasks=folder/'tasks.jsonl'
        tasks.write_text('bound-tasks')
        tasks.with_name('completed.json').write_text('{}')
        physical=self.folder/'physical'
        physical.mkdir()
        (physical/'manifest.json').write_text('{"success":true}')
        requests=[]
        for i in range(8):
            sample=folder/str(i);sample.mkdir()
            row=dict(seed=0,round=0,arm='bounded_wl',role='evaluation',index=i,graph='bp_fe',
                training_feedback_eligible=False,noise_seed=4600000,noise_sha256='noise'+str(i),
                raw_action=[[0.,0.]],executed_action=[[0.,0.]],predicted_q=[-1.],checkpoint_sha256='actor',sample_dir=str(sample))
            (sample/'request.json').write_text(json.dumps(row));(sample/'actions.npz').write_text('bound-array')
            if i==0:
                (sample/'outcome.json').write_text(json.dumps(dict(row,status='success',physical_run=str(physical))))
            requests.append(row)
        cfg=dict(arm='bounded_wl',design='bp_fe',evaluation_samples=8,online_rounds=0,evaluation_noise_seed=4600000,weights=[1,0,0,0,0],physical_failure_reward=-4)
        (experiment/'config.json').write_text(json.dumps(cfg))
        with patch.object(report.native,'curve',return_value=dict(available=False)), \
             patch.object(report.native,'check_training',return_value=dict(completed=True,checkpoint_sha256='actor')), \
             patch('fql.graph_o2o_backend.load_requests',return_value=(requests,tasks)), \
             patch('fql.qstudy_backend.checked_request',side_effect=lambda row,*_:Path(row['sample_dir'])), \
             patch('fql.qstudy_backend.checked_existing') as checker:
            raw=report.native.inspect_round(experiment,cfg,0,0,np.ones(5),({},None,{}))
        checker.assert_not_called()
        stream=report.evaluation_stream(raw,'wl',specs())
        self.assertEqual(stream['invalid'],1)
        self.assertIn('Missing saved native geometry audit',stream['rows'][0]['integrity_error'])

    def declarations(self):
        bc=self.folder/'bc';root=self.folder/'bounded';bc.mkdir();root.mkdir()
        native=np.tile(np.array([100.,-.3,-50.,2.,3.,.2,70000.]),(384,1))
        source=dict(raw_action=np.zeros((384,4),dtype=np.float32),executed_action=np.zeros((384,4)),metrics=native,
            metric_mask=np.ones((384,7),dtype=bool),status=np.repeat('success',384),family=np.repeat(np.arange(192),2),
            source_row=np.arange(384),variant=np.repeat('v',384))
        data=dict(train=source,behavior_action=np.zeros((256,4)),behavior_family=np.arange(256))
        dataset=self.folder/'dataset.npz';dataset.write_text('hash-bound-dataset')
        cfg=dict(design='bp_fe',arm='bc',q_weight=0,seeds=[0,1,2],evaluation_samples=8,offline_actor_steps=1000,
            online_flow_update=False,offline_critic_update=False,dataset_path=str(dataset),dataset_sha256=report.native.sha(dataset),
            common_checkpoints={},common_checkpoint_sha256={})
        for seed in report.SEEDS:
            p=self.folder/f'common{seed}.pt';p.write_text('common'+str(seed))
            cfg['common_checkpoints'][str(seed)]=str(p);cfg['common_checkpoint_sha256'][str(seed)]=report.native.sha(p)
        (bc/'config.json').write_text(json.dumps(cfg))
        context=report.adapter.screen.arrays(source,'train');context['binding']=dict(dataset_sha256=cfg['dataset_sha256'])
        configs={}
        for objective,(arm,_) in report.adapter.ARMS.items():
            path=root/arm;path.mkdir()
            spec,_=report.adapter.screen.specification(context,list(range(384)),objective)
            current=copy.deepcopy(cfg);current.update(arm=arm,q_weight=1,bounded_utility_spec=spec,
                bounded_behavior=report.adapter.behavior_identity(data),bounded_refits={},critic_checkpoints={},critic_checkpoint_sha256={})
            for seed in report.SEEDS:
                cp=self.folder/f'{objective}_{seed}.pt';cp.write_text('weights')
                completed=self.folder/f'{objective}_{seed}.json'
                completed.write_text(json.dumps(dict(utility_spec=spec,normalization={'train':384},binding=dict(fit='refit',train_indices=list(range(384))))))
                current['critic_checkpoints'][str(seed)]=str(cp);current['critic_checkpoint_sha256'][str(seed)]=report.native.sha(cp)
                current['bounded_refits'][str(seed)]=dict(checkpoint=report.bind(cp),completion=report.bind(completed),normalization={'train':384})
            configs[path]=current
            (path/'config.json').write_text(json.dumps(current))
        return root,bc,data,configs

    def test_declaration_scale_targets_and_refit_artifact_validation(self):
        root,bc,data,configs=self.declarations()
        with patch.object(report,'load_dataset',return_value=(data,{})), \
             patch.object(report.adapter,'validate_config',side_effect=lambda p,*_:configs[p]):
            plans,ss,evidence,reuse=report.plans_and_specs(root,bc)
            self.assertEqual(set(plans),set(report.ARMS))
            self.assertFalse(reuse['controlled_effective_update_eligible'])
            self.assertFalse(reuse['old_source_certificate_extended'])
            self.assertTrue(all(s['scales']==ss['wl']['scales'] for s in ss.values()))
            victim=next(iter(configs.values()))['bounded_refits']['0']['checkpoint']['path']
            Path(victim).write_text('tampered')
            with self.assertRaises(ValueError):
                report.plans_and_specs(root,bc)

    def test_changed_behavior_and_fullTRAIN_scales_refused(self):
        root,bc,data,configs=self.declarations()
        with patch.object(report,'load_dataset',return_value=(data,{})), \
             patch.object(report.adapter,'validate_config',side_effect=lambda p,*_:configs[p]):
            victim=next(iter(configs.values()))
            victim['bounded_behavior']['action_sha256']='x'*64
            with self.assertRaisesRegex(ValueError,'targets'):
                report.plans_and_specs(root,bc)
            victim['bounded_behavior']=report.adapter.behavior_identity(data)
            victim['bounded_utility_spec']['scales'][0]*=2
            # A re-signed declaration cannot replace the native TRAIN scale.
            for value in victim['bounded_refits'].values():
                p=Path(value['completion']['path']);done=json.loads(p.read_text())
                done['utility_spec']=victim['bounded_utility_spec'];p.write_text(json.dumps(done))
                value['completion']=report.bind(p)
            with self.assertRaisesRegex(ValueError,'full-TRAIN'):
                report.plans_and_specs(root,bc)

    def test_complete_readonly_snapshot_and_preexport_partial(self):
        root,bc,data,configs=self.declarations()
        plans={'BC':(bc,json.loads((bc/'config.json').read_text()))}
        for objective,(arm,_) in report.adapter.ARMS.items():
            plans[objective]=(root/arm,configs[root/arm])
        args=argparse.Namespace(bounded_root=root,bc=bc,output=self.folder/'reports')
        def inspect(experiment,cfg,seed,*_):
            arm=next(a for a,(p,_) in plans.items() if p==experiment)
            return record(arm,seed)
        with patch.object(report,'plans_and_specs',return_value=(plans,specs(),{},dict(controlled_effective_update_eligible=False))), \
             patch.object(report.backend,'context',return_value=({},None,dict(physical_flow_revision='fixed'))), \
             patch.object(report.native,'inspect_round',side_effect=inspect):
            path=report.run(args)
            result=json.loads((path/'report.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['backend_calls'],0)
            self.assertEqual(result['model_forward_calls'],0)
            self.assertTrue(result['same_actual_noise_plan_all_streams'])
            self.assertIsNone(result['selected_winner'])
            self.assertEqual(len(result['per_seed_streams']),12)
            marker=json.loads((path/'completed.json').read_text())
            self.assertEqual(marker['artifacts']['report.json'],report.native.sha(path/'report.json'))
        def partial(*args):
            return dict(seed=args[2],experiment=str(args[0]),rows=[],errors=[],training=dict(completed=False))
        with patch.object(report,'plans_and_specs',return_value=(plans,specs(),{},{})), \
             patch.object(report.backend,'context',return_value=({},None,dict(physical_flow_revision='fixed'))), \
             patch.object(report.native,'inspect_round',side_effect=partial):
            result=json.loads((report.run(args)/'report.json').read_text())
            self.assertEqual(result['status'],'partial')
            self.assertTrue(all(a['request_accounting']['not_exported']==24 for a in result['independent_seed_aggregates']))
            self.assertTrue(all(a['native_metrics'] is None for a in result['independent_seed_aggregates']))


if __name__=='__main__':
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run report tests through bounded CPU Slurm')
    unittest.main()
