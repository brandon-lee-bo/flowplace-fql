"""Bounded real-fit provenance and saved-array COM reporting tests; CPU Slurm."""
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

from . import qstudy_com_report as report
from . import qstudy_com_screen as com
from . import qstudy_bounded_screen as screen
from .graph_o2o_archive import sha


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run tiny numerical tests with bounded CPU Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='qstudy-com-report-',dir=os.environ['TMPDIR'])
        self.root=Path(self.temporary.name)
        cache=dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.2) for n in 'abc'},
            net_info={'n':{'nodes':{'a':{'x_offset':.2},'b':{},'c':{'y_offset':-.3}}}})
        x=torch.rand((12,3,2),generator=torch.Generator().manual_seed(19))*1.2-.6
        # One raw out-of-box row remains fully supervised but COM-ineligible.
        x[0,0,0]=1.2
        native=torch.stack((100+3*x[:,0,0],-(.4+.1*x[:,1,0]),-(25+x[:,2,1]),
            (3*x[:,0,1]).clamp_min(0),(4*x[:,1,1]).clamp_min(0),
            torch.ones(12)*.2,torch.ones(12)*1000),-1).double().numpy()
        train=dict(x=x,native=native,mask=np.isfinite(native),executed=(x*.9).numpy().reshape(12,-1),
            statuses=np.full(12,'success',dtype='<U32'),families=np.arange(6).repeat(2),
            source_ids=[f'family{i//2}_row{i}' for i in range(12)])
        self.data=dict(train=train,validation=copy.deepcopy(train))
        self.fold=screen.family_folds(train['families'])[0]
        self.task=dict(index=2,objective='wl',seed=2)
        inputs=dict(dataset_sha256='a'*64,manifest_sha256='b'*64,cache_sha256='c'*64)
        self.plan=dict(schema='qstudy_local_com_bounded_wl_v1',inputs=inputs,
            config=dict(com.CONFIG,steps=3,batch_size=8),source_sha256={'fixture':'d'*64},
            fold_plan_sha256='e'*64,supervised_control={'plan_sha256':'f'*64})
        self.plan_sha='1'*64
        binding=dict(schema=self.plan['schema'],**inputs,plan_sha256=self.plan_sha,task=self.task,
            source_sha256=self.plan['source_sha256'],fold_plan_sha256=self.plan['fold_plan_sha256'],
            supervised_control=self.plan['supervised_control'],threads=torch.get_num_threads(),
            torch_version=str(torch.__version__),numpy_version=np.__version__)
        self.context=dict(train,cache=cache,movable=torch.tensor([True,True,False]),binding=binding)
        self.args=SimpleNamespace(output=self.root/'com',objective='wl',seed=2,steps=3,
                                  batch_size=8,lr=3e-4,weight_decay=1e-4)
        oldargs=SimpleNamespace(**dict(vars(self.args),output=self.root/'old'))
        com.STOP=screen.STOP=False
        with contextlib.redirect_stdout(io.StringIO()):
            self.control=screen.fit(oldargs,self.context,self.fold['name'],self.fold['train_indices'],
                self.fold['target_indices'],train)
            self.done=com.fit(self.args,self.context,self.fold['name'],self.fold['train_indices'],
                self.fold['target_indices'],train)
        self.folder=self.args.output/self.fold['name']
        self.oldfolder=oldargs.output/self.fold['name']

    def tearDown(self):
        com.STOP=screen.STOP=False
        self.temporary.cleanup()

    def verify(self,done=None,data=None):
        return report.verify_fit(self.folder,self.done if done is None else done,self.task,self.fold,
            self.plan,self.plan_sha,self.data if data is None else data,self.oldfolder,self.control)

    def test_real_COM_checkpoint_native7_normalization_coverage_and_no_model_forward(self):
        with mock.patch.object(com.supervised.MultiMetricCritic,'forward',side_effect=AssertionError('model forward')):
            row=self.verify()
        self.assertEqual(len(row['native']['metrics']),7)
        self.assertEqual(row['com_audit']['supervised_admitted_rows'],len(self.fold['train_indices']))
        self.assertEqual(row['learning_curve']['coverage'],1.)
        self.assertEqual(row['learning_curve']['observed_steps'],[3])
        self.assertEqual(row['bindings']['checkpoint_sha256'],self.done['checkpoint_sha256'])
        self.assertEqual(row['critic'],self.done['critic'])
        self.assertTrue(row['input_gradients']['finite'])

    def test_loss_reference_and_control_inventory_not_stripped(self):
        for key in ['com_loss','com_official_reference','supervised_control']:
            changed=copy.deepcopy(self.done)
            changed['binding']['run'][key]={}
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'run/plan/source binding'):
                self.verify(changed)
        changed=copy.deepcopy(self.done)
        changed['com_audit']['label_or_family_based_eligibility']=True
        with self.assertRaisesRegex(ValueError,'eligibility audit'):
            self.verify(changed)

    def test_native_labels_actions_source_ids_and_masks_revalidated(self):
        for key in ['native','executed','mask','source_ids']:
            data=copy.deepcopy(self.data)
            idx=self.fold['target_indices'][0]
            if key=='source_ids':
                data['train'][key][idx]='changed_source'
            elif key=='mask':
                data['train'][key][idx,6]=False
            else:
                data['train'][key][idx,0]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.verify(data=data)

    def test_checkpoint_feature_normalizer_and_update_count_tamper_refused(self):
        path=self.folder/'resume.pt';original=path.read_bytes()
        saved=torch.load(path,map_location='cpu',weights_only=False)
        saved['model']['feature_mean'][0]+=1
        torch.save(saved,path)
        done=dict(self.done,checkpoint_sha256=sha(path))
        with self.assertRaisesRegex(ValueError,'paired control buffer'):
            self.verify(done)
        path.write_bytes(original)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        next(iter(saved['optimizer']['state'].values()))['step']-=1
        torch.save(saved,path)
        with self.assertRaisesRegex(ValueError,'optimizer update count'):
            self.verify(dict(self.done,checkpoint_sha256=sha(path)))

    def test_prediction_tamper_detected_even_if_completion_SHA_rebound(self):
        path=self.folder/'predictions.npz'
        with np.load(path,allow_pickle=False) as source:
            arrays={k:source[k].copy() for k in source.files}
        arrays['native_metrics'][0,6]+=2
        np.savez(path,**arrays)
        with self.assertRaisesRegex(ValueError,'native label mismatch'):
            self.verify(dict(self.done,predictions_sha256=sha(path)))

    def test_per100_curve_sign_weight_gradient_pressure_and_geometry(self):
        path=self.folder/'learning_curve.jsonl'
        record=json.loads(path.read_text())
        for change in ['loss','weight','ratio','radius','fixed','coverage']:
            row=copy.deepcopy(record)
            if change=='loss': row['loss']+=1
            if change=='weight': row['parts']['com_weighted_penalty']+=1
            if change=='ratio': row['parts']['com_gradient_ratio']+=1
            if change=='radius': row['parts']['com_radius_rms_max']=.03
            if change=='fixed': row['parts']['com_fixed_max_abs_change']=1e-3
            if change=='coverage': row['parts']['com_eligible_rows']+=1
            path.write_text(json.dumps(row)+'\n')
            with self.subTest(change=change),self.assertRaises(ValueError):
                report.curve_diagnostics(self.folder,3)

    def test_missing_log_diagnostics_explicit_and_committed_corruption_refused(self):
        path=self.folder/'learning_curve.jsonl';record=path.read_text()
        path.unlink()
        row=report.curve_diagnostics(self.folder,200)
        self.assertEqual(row['status'],'missing')
        self.assertEqual(row['missing_steps'],[100,200])
        path.write_text(record+'{"interrupted')
        row=report.curve_diagnostics(self.folder,200)
        self.assertEqual(row['status'],'partial')
        self.assertEqual(row['observed_steps'],[3])
        self.assertEqual(row['coverage'],0.)
        path.write_text('{"interior\n'+record)
        with self.assertRaisesRegex(ValueError,'Corrupt committed'):
            report.curve_diagnostics(self.folder,200)

    def test_duplicate_curve_steps_refused(self):
        path=self.folder/'learning_curve.jsonl';record=path.read_text()
        path.write_text(record+record)
        with self.assertRaisesRegex(ValueError,'Nonmonotone'):
            report.curve_diagnostics(self.folder,3)

    def test_paired_fold_then_seed_aggregation_and_historical_separation(self):
        verified=self.verify()
        fits=[];controls=[]
        for seed in range(3):
            for fold in ['cv_0','cv_1','cv_2','refit']:
                scope=report.SCOPES[1] if fold=='refit' else report.SCOPES[0]
                new=copy.deepcopy(verified);old=copy.deepcopy(verified)
                for row in [new,old]: row.update(seed=seed,fit=fold,scope=scope)
                old['critic']['mae']=10+seed
                new['critic']['mae']=10+seed+(1+seed if fold!='refit' else 100)
                fits.append(new);controls.append(old)
        pairs=report.paired_rows(fits,controls)
        seeds,summary=report.paired_summaries(pairs)
        cv=next(r for r in summary if r['scope']==report.SCOPES[0] and r['statistic']=='mae')
        self.assertTrue(cv['aggregate_eligible'])
        self.assertEqual(cv['paired_three_seed_delta']['mean'],2.)
        self.assertEqual(cv['paired_three_seed_delta']['sd'],1.)
        historical=next(r for r in summary if r['scope']==report.SCOPES[1] and r['statistic']=='mae')
        self.assertEqual(historical['paired_three_seed_delta']['mean'],100.)
        partial=report.paired_summaries([r for r in pairs if not (r['seed']==2 and r['fit']=='cv_1')])[1]
        cv=next(r for r in partial if r['scope']==report.SCOPES[0] and r['statistic']=='mae')
        self.assertFalse(cv['aggregate_eligible'])
        self.assertIsNone(cv['paired_three_seed_delta'])
        self.assertEqual(cv['observed_seeds'],[0,1])

    def test_pair_environment_or_physical_scale_difference_refused(self):
        row=self.verify();old=copy.deepcopy(row)
        old['bindings']['environment']['threads']=100
        with self.assertRaisesRegex(ValueError,'environments differ'):
            report.paired_rows([row],[old])
        old=copy.deepcopy(row);old['utility_spec']['scales'][0]*=2
        with self.assertRaises(ValueError):
            report.paired_rows([row],[old])

    def test_plan_SHA_and_source_inventory_loss_validation_before_load(self):
        path=self.root/'plan.json';path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'plan SHA changed'):
            report.checked_plan(path,'0'*64)
        minimal=dict(self.plan,tasks=[dict(index=s,objective='wl',seed=s) for s in range(3)],
            maximum_new_fits=12,design='bp_fe',objectives={'wl':[1,0,0,0,0]},
            official_reference=com.OFFICIAL,actor_advanced=False,selected_winner=None,
            prospective_new_native_calls=0,historical_validation_previously_inspected=True)
        path.write_text(json.dumps(minimal))
        with self.assertRaisesRegex(ValueError,'Undeclared COM plan'):
            report.checked_plan(path)


if __name__=='__main__':
    unittest.main()
