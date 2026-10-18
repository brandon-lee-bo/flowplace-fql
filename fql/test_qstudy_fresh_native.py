"""Synthetic locked-export/native contracts and scheduling; bounded CPU Slurm.

Manual fixture arrays are not the reserved CUDA noise stream. No policy or EDA
is executed. Native context/request/existing-geometry-failure validators remain
real; only scheduler/physical-call sites are mocked in scheduling tests.
"""
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

from . import qstudy_fresh_eval as fresh
from . import qstudy_fresh_native as finish
from . import qstudy_fresh_report as report
from .qstudy_bounded_utility import make_spec
from .test_qstudy_fresh_eval import Fixture


class FreshNativeTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run synthetic contracts with bounded CPU Slurm')
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.root=Path(self.temp.name);f=self.f=Fixture(self.root)
        self.patches=[patch.object(fresh,'RUNTIME',f.runtime),patch.object(fresh,'CHECKPOINT',f.public),
            patch.object(fresh,'PUBLIC_SHA',fresh.sha(f.public)),patch.object(fresh.backend,'REFERENCE',f.reference)]
        for p in self.patches:p.start()
        self.spec=make_spec([1,0,0,0,0],[100,1,10,3,4],dict(split='train',
            dataset_sha256=fresh.sha(f.data),train_rows_sha256='a'*64))
        utility=Path(f.lock['common_train_utility_reference']['path'])
        fresh.atomic_json(utility,dict(version=report.UTILITY_VERSION,utility_specs={'wl':self.spec}))
        f.lock['common_train_utility_reference']=fresh.binding(utility);f.save()
        self.lock,self.cfgs,self.phys,self.declaration=fresh.declare(f.path,f.digest)
        self.noises,self.noise_info=f.saved_noise()
        self.exports=[]
        for stream in self.lock['streams']:
            cfg=self.cfgs[stream['id']];physical=self.phys[stream['id']]
            experiment=Path(stream['evaluation_config']['path']).parent
            for seed in stream['seeds']:
                folder=experiment/f'seed_{seed}/round_0/requests';rows=[]
                for i in range(fresh.COUNT):
                    sample=folder/stream['id']/'evaluation'/str(i);sample.mkdir(parents=True)
                    raw=np.full((3,2),.01*i,dtype=np.float32);action=raw.copy()
                    np.savez(sample/'actions.npz',noise=self.noises[i],raw_action=raw,action=action)
                    row=dict(fresh.row_identity(stream,seed,i,cfg,physical,f.digest,self.declaration,self.noise_info),
                        sample_dir=str(sample),raw_action=raw.tolist(),executed_action=action.tolist(),
                        raw_action_sha256=fresh.action_sha(raw),action_sha256=fresh.action_sha(action),
                        geometry=dict(legal=False),pre_backend_failure='normalized_geometry')
                    if stream['kind']=='public_official20':
                        (sample/'trajectory.pt').write_bytes(b'synthetic trajectory, never passed to torch')
                        row['trajectory_sha256']=fresh.sha(sample/'trajectory.pt')
                    fresh.atomic_json(sample/'request.json',row);rows.append(row)
                tasks=folder/'tasks.jsonl';tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                fresh.atomic_json(folder/'completed.json',dict(status='completed',count=8,
                    config_sha256=physical['config_sha256'],tasks_sha256=fresh.sha(tasks),
                    selection_lock_sha256=f.digest,fresh_declaration_sha256=fresh.object_sha(self.declaration),
                    backend_calls=0,optimizer_updates=0,evaluation_never_replayed=True))
                self.exports.append((stream,seed,rows,tasks))
        self.args=argparse.Namespace(selection=f.path,selection_sha256=f.digest,output=self.root/'native_finish')

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def plan(self):
        finish.prepare(self.args)
        path=self.args.output/'plan.json'
        return path,finish.checked_plan(path,fresh.sha(path))

    def geometry_outcome(self,row,tasks):
        cfg,verification,physical=finish.backend.context(Path(row['sample_dir']).parents[5])
        # sample/stream/evaluation/index -> requests -> round -> seed -> experiment
        sample=Path(row['sample_dir'])
        value=dict(status='geometry_failure',metrics=None,reward=-4,stage='pre_backend',
            failure_class='normalized_geometry',terminal_mask=0,
            request_sha256=fresh.sha(sample/'request.json'),tasks_sha256=fresh.sha(tasks),
            config_sha256=physical['config_sha256'],dataset_sha256=physical['dataset_sha256'],
            **{k:row[k] for k in ['seed','round','arm','role','index','graph','training_feedback_eligible',
                'raw_action_sha256','action_sha256','noise_sha256','checkpoint_sha256']})
        fresh.atomic_json(sample/'outcome.json',value)
        return value,cfg,physical,verification

    def test_sealed_all80_exports_actual_noise_and_real_native_request_validators(self):
        with patch.object(fresh,'declare',side_effect=AssertionError('writer forbidden')), \
             patch.object(fresh,'shared_noise',side_effect=AssertionError('noise writer forbidden')), \
             patch.object(fresh,'build_sampler',side_effect=AssertionError('policy forbidden')), \
             patch.object(finish.backend,'measure',side_effect=AssertionError('EDA forbidden')):
            path,(plan,checked)=self.plan()
            self.assertEqual(len(plan['dispatch']),80)
            self.assertEqual(plan['expected_requests'],80)
            self.assertEqual(plan['noise']['shape'],[8,3,2])
            self.assertEqual(plan['drt_indices'],[0,1,2,3])
            self.assertFalse(plan['drt_execution_implemented'])
            self.assertTrue(all(not r['training_feedback_eligible'] and r['round']==0 for r in plan['dispatch']))
            before=path.read_bytes();finish.prepare(self.args)
            self.assertEqual(path.read_bytes(),before)

    def test_request_actual_noise_marker_and_public_trajectory_tamper_refused(self):
        stream,seed,rows,tasks=self.exports[-1]
        sample=Path(rows[0]['sample_dir']);trajectory=sample/'trajectory.pt'
        original=trajectory.read_bytes();trajectory.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'trajectory changed'):
            finish.checked_exports(self.f.path,self.f.digest)
        trajectory.write_bytes(original)
        marker=tasks.with_name('completed.json');value=fresh.read(marker)
        fresh.atomic_json(marker,dict(value,evaluation_never_replayed=False))
        with self.assertRaisesRegex(ValueError,'export marker'):
            finish.checked_exports(self.f.path,self.f.digest)
        fresh.atomic_json(marker,value)
        arrays_path=sample/'actions.npz'
        with np.load(arrays_path) as arrays: saved={k:arrays[k].copy() for k in arrays.files}
        saved['noise'][0,0]+=.1;np.savez(arrays_path,**saved)
        with self.assertRaisesRegex(ValueError,'Noise identity'):
            finish.checked_exports(self.f.path,self.f.digest)

    def test_missing_export_partial_report_denominator_and_dispatch_refusal(self):
        _,_,rows,tasks=self.exports[0]
        tasks.with_name('completed.json').unlink()
        with self.assertRaisesRegex(ValueError,'All locked fresh exports'):
            finish.prepare(self.args)
        result=report.collect(self.f.path,self.f.digest)
        self.assertEqual(result['status'],'partial')
        candidate=next(r for r in result['aggregates'] if r['purpose']=='candidate')
        self.assertFalse(candidate['aggregate_eligible'])
        self.assertIsNone(candidate['metrics'])
        self.assertEqual(candidate['request_accounting']['planned'],24)
        self.assertEqual(candidate['request_accounting']['not_exported'],8)
        self.assertIsNone(next(r for r in result['streams'] if r['purpose']=='candidate')['measured_objectives']['wl']['B_all_request_mean'])

    def test_real_native_geometry_failure_has_B4_no_PPA_and_never_calls_EDA(self):
        _,_,rows,tasks=self.exports[0];row=rows[0]
        out,cfg,physical,verification=self.geometry_outcome(row,tasks)
        with patch.object(finish.backend,'measure',side_effect=AssertionError('EDA forbidden')):
            result=finish.inspect_outcome(row,tasks,cfg,physical,verification)
        self.assertEqual(result['state'],'geometry_failure')
        self.assertIsNone(result['metrics'])
        value=report.measured(result,self.spec)
        self.assertEqual(value,dict(valid=True,C=None,B=4.,utility=-4.))
        out['failure_class']='not_a_native_failure';fresh.atomic_json(Path(row['sample_dir'])/'outcome.json',out)
        self.assertEqual(finish.inspect_outcome(row,tasks,cfg,physical,verification)['state'],'invalid')

    def test_missing_geometry_audit_blocks_before_native_verifier_can_call_OpenROAD(self):
        _,_,rows,tasks=self.exports[0];row=rows[0]
        out,cfg,physical,verification=self.geometry_outcome(row,tasks)
        run=self.root/'synthetic_run';run.mkdir();fresh.atomic_json(run/'manifest.json',dict(success=True))
        out.update(physical_run=str(run),status='success');fresh.atomic_json(Path(row['sample_dir'])/'outcome.json',out)
        with patch.object(finish.backend,'checked_existing',side_effect=AssertionError('must guard first')) as nativecheck:
            result=finish.inspect_outcome(row,tasks,cfg,physical,verification)
        self.assertEqual(result['state'],'invalid');self.assertIn('cannot run OpenROAD',result['integrity_error'])
        nativecheck.assert_not_called()

    def test_source_inventory_plan_and_request_tamper_before_worker(self):
        path,(plan,checked)=self.plan()
        changed=copy.deepcopy(plan);changed['source_sha256']['fql/qstudy_backend.py']='0'*64
        fresh.atomic_json(path,changed)
        with self.assertRaisesRegex(ValueError,'finish source changed'):
            finish.checked_plan(path,fresh.sha(path))
        fresh.atomic_json(path,plan)
        row=plan['dispatch'][0];request=Path(row['sample_dir'])/'request.json'
        value=fresh.read(request);value['training_feedback_eligible']=True;fresh.atomic_json(request,value)
        with self.assertRaises(ValueError):finish.checked_plan(path,fresh.sha(path))

    def test_worker_only_exact_sealed_row_unchanged_backend_and_allocation_guard(self):
        path,(plan,checked)=self.plan()
        args=argparse.Namespace(plan=path,plan_sha256=fresh.sha(path),index=9)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'2'}), self.assertRaisesRegex(RuntimeError,'exactly8'):
            finish.worker(args)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'8'}),patch.object(finish.backend,'measure') as measure:
            finish.worker(args)
        call=measure.call_args.args[0]
        self.assertEqual((call.experiment,call.seed,call.round,call.index),
            (plan['dispatch'][9]['experiment'],plan['dispatch'][9]['seed'],0,plan['dispatch'][9]['index']))
        self.assertEqual(call.storage,'scratch')

    def test_scheduling_max4_missing_only_and_genuine_preemption_states(self):
        path,(plan,checked)=self.plan()
        outcomes=[dict(state='missing') for _ in plan['dispatch']]
        outcomes[0]=dict(state='geometry_failure')
        state=dict(jobs={'1':['11'],'2':['12'],'3':['13'],'4':['14']})
        calls=[]
        def observer(job):calls.append(job);return ['PREEMPTED'] if job=='14' else ['RUNNING']
        actions=finish.next_actions(plan,state,outcomes,observer=observer)
        self.assertEqual(actions['busy'],3)
        self.assertEqual(actions['submit_indices'],[4])
        self.assertEqual(calls,['11','12','13','14'])
        self.assertNotIn(0,actions['missing'])
        for status in ['FAILED','TIMEOUT','CANCELLED','COMPLETED']:
            with self.subTest(status=status),self.assertRaisesRegex(RuntimeError,'needs diagnosis'):
                finish.disposition([[status]])
        self.assertEqual(finish.disposition([['PREEMPTED'],['NODE_FAIL']]),'retry')
        self.assertEqual(finish.disposition([['ACCOUNTING_PENDING']]),'wait')

    def test_completed_requests_not_resubmitted_and_foreign_attempt_refused(self):
        _,(plan,checked)=self.plan()
        outcomes=[dict(state='physical_failure') for _ in plan['dispatch']]
        state=dict(jobs={})
        actions=finish.next_actions(plan,state,outcomes,observer=lambda _:self.fail('must not observe complete jobs'))
        self.assertEqual(actions['missing'],[]);self.assertEqual(actions['submit_indices'],[])
        task=plan['dispatch'][0];attempt=Path(task['sample_dir'])/'backend_attempts/a';attempt.mkdir(parents=True)
        fresh.atomic_json(attempt/'attempt.json',dict(request_sha256='0'*64,tasks_sha256=task['tasks_sha256'],job_id='1'))
        outcomes[0]=dict(state='missing')
        with self.assertRaisesRegex(ValueError,'Foreign physical attempt'):
            finish.next_actions(plan,state,outcomes,observer=lambda _:['PREEMPTED'])

    def test_submission_clears_inherited_restart_and_adopts_only_own_unique_name(self):
        calls=[]
        def run(command,**kwargs):
            calls.append((command,kwargs))
            if command[0]=='squeue': return argparse.Namespace(stdout='900|foreign_job\n',returncode=0)
            if command[0]=='sacct': return argparse.Namespace(stdout='',returncode=0)
            return argparse.Namespace(stdout='1234;cluster\n',stderr='',returncode=0)
        with patch.dict(os.environ,{'SLURM_RESTART_COUNT':'5'}),patch.object(finish.subprocess,'run',side_effect=run):
            self.assertEqual(finish.submit_unique(['worker.slurm'],'qsf_unique'),'1234')
            self.assertEqual(os.environ['SLURM_RESTART_COUNT'],'5')
        self.assertTrue(all('SLURM_RESTART_COUNT' not in kwargs['env'] for _,kwargs in calls))
        self.assertEqual(calls[-1][0][:3],['sbatch','--parsable','--job-name=qsf_unique'])


if __name__=='__main__':unittest.main()
