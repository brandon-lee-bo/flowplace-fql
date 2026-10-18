"""Small CPU-only scheduler/identity/restart contracts; never launch EDA."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from . import qstudy_train_drt_recovery as m


def account(job,state='PREEMPTED',raw=None,name='original'):
    return dict(job_id=job,job_id_raw=raw or job.split('_')[0],state=state,exit_code='0:0',name=name)


def snapshot(rows=(),queue=()):
    return dict(accounting=list(rows),queue=list(queue),captured_unix=1.,sacct_stdout='saved exact observation',
                squeue_stdout='saved exact observation')


class FakeScheduler:
    def __init__(self,completed,preempt_generations=0):
        self.completed=completed;self.preempt_generations=preempt_generations
        self.submitted=[];self.lookups=[];self.pending_initial=0;self.error_count=0
        self.names={};self.first=None

    def snapshot(self,jobs):
        if self.error_count:
            self.error_count-=1
            raise subprocess.CalledProcessError(1,['sacct'])
        rows=[account(m.ORIGINAL+'_'+str(i)) for i in range(64)]
        rows += [account(m.DEV012+'_'+str(i)) for i in range(3)]
        for number,command in enumerate(self.submitted,1):
            indices=[int(i) for i in next(x for x in command if x.startswith('--array=')).split('=',1)[1].split('%')[0].split(',')]
            state='PREEMPTED' if number<=self.preempt_generations else 'COMPLETED'
            rows.extend(account(str(700+number)+'_'+str(i),state) for i in indices)
        if self.pending_initial:
            self.pending_initial-=1
            return snapshot(rows,[dict(job_id=m.DEV012+'_[0-2%2]',state='PENDING',name='prior_dev')])
        return snapshot(rows)

    def find_name(self,name,created_unix):
        self.lookups.append(name)
        return self.names.get(name),dict(squeue_stdout='',sacct_stdout='')

    def submit(self,command):
        self.submitted.append(command)
        number=len(self.submitted);job=str(700+number)
        name=next(x.split('=',1)[1] for x in command if x.startswith('--job-name='))
        self.names[name]=job
        if number>self.preempt_generations:
            indices=[int(i) for i in next(x for x in command if x.startswith('--array=')).split('=',1)[1].split('%')[0].split(',')]
            self.completed.update(indices)
        return job,dict(command=command,stdout=job+'\n',returncode=0)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'recovery_v1'
        self.plan_path=self.root/'plan.json';self.plan_path.write_text('{}')
        self.tasks=[dict(index=i,key='k'+str(i),family=i//2,variant=i%2,
            provenance=dict(split='train'),provenance_sha256='prov'+str(i),
            native=dict(run=str(self.root/'original'),files={'physical_manifest':{'sha256':'native'}}),
            output=str(self.root/'full_route'/('k'+str(i))),run_id='run'+str(i),method='method') for i in range(64)]
        self.plan=dict(output=str(self.root),tasks=self.tasks)
        self.pinned=dict(version=m.VERSION,plan_sha256=m.PLAN_SHA,source_sha256={})
        self.completed=set(range(64))-{8,9}
        m.STOP=False
        self.addCleanup(setattr,m,'STOP',False)

    def inspect(self,plan,task):
        done=task['index'] in self.completed
        return dict(index=task['index'],key=task['key'],state='completed' if done else 'missing',
                    metrics={'DRC':0} if done else None)

    def finish(self,path,output,digest):
        value=dict(status='complete' if len(self.completed)==64 else 'incomplete',requested=64,
                   counts=dict(completed=len(self.completed)))
        output.parent.mkdir(parents=True,exist_ok=True);m.atomic_json(output,value)
        return value

    def run_controller(self,scheduler,sleep=lambda seconds:None):
        args=type('Args',(),dict(plan=self.plan_path,plan_sha256=m.PLAN_SHA,output=self.output))()
        with mock.patch.dict(os.environ,SLURM_JOB_ID='900',USER='test'), \
             mock.patch.object(m,'binding',return_value=(self.plan,self.pinned)), \
             mock.patch.object(m.collector,'inspect_result',side_effect=self.inspect), \
             mock.patch.object(m,'attempts',return_value=[]), \
             mock.patch.object(m.collector,'report',side_effect=self.finish):
            return m.controller(args,scheduler=scheduler,sleep=sleep)

    def test_parse_array_raw_accounting_and_unknown_ref_backoff(self):
        rows=m.parse_accounting('1798635_8|987654|PREEMPTED|0:0|original|\n')
        s=snapshot(rows)
        self.assertEqual(m.observed(s,'1798635_8'),['PREEMPTED'])
        self.assertEqual(m.observed(s,'987654'),['PREEMPTED'])
        self.assertEqual(m.observed(s,'1798635',parent=True),['PREEMPTED'])
        self.assertEqual(m.observed(s,'unseen'),['ACCOUNTING_PENDING'])
        self.assertFalse(m.terminal(['COMPLETED','ACCOUNTING_PENDING']))
        with self.assertRaises(ValueError):m.parse_accounting('1798635.batch|123|FAILED|1:0|name|')

    def test_actual_compressed_pending_records_wait_parent_never_prove_child(self):
        text=('1798635_[17-63%2]|1798635|PENDING|0:0|qstudy_train_drt64_v1|\n'
              '1798748_[0-2%2]|1798748|PENDING|0:0|qstudy_train_drt64_recovery012|\n')
        s=snapshot(m.parse_accounting(text))
        self.assertEqual(m.observed(s,m.ORIGINAL,parent=True),['PENDING'])
        self.assertEqual(m.observed(s,m.DEV012,parent=True),['PENDING'])
        self.assertEqual(m.observed(s,m.ORIGINAL+'_17'),['ACCOUNTING_PENDING'])
        for malformed in ('1798635_[63-17%2]|1798635|PENDING|0:0|name|',
                          '1798635_[17-64%2]|1798635|PENDING|0:0|name|',
                          '1798748_[0-3%2]|1798748|PENDING|0:0|name|',
                          '1798635_[17-63%2]|1798635|COMPLETED|0:0|name|'):
            with self.assertRaises(ValueError):m.parse_accounting(malformed)

    def test_only_missing_confirmed_preemption_retried_completed_skipped(self):
        state=dict(generations=[])
        s=snapshot([account(m.ORIGINAL+'_'+str(i)) for i in range(64)]+[account(m.DEV012+'_'+str(i)) for i in range(3)])
        with mock.patch.object(m.collector,'inspect_result',side_effect=self.inspect),mock.patch.object(m,'attempts',return_value=[]):
            rows=m.scan(self.plan,state,s)
        self.assertEqual([r['index'] for r in rows if r['decision']=='retry'],[8,9])
        self.assertEqual(sum(r['decision']=='completed_skip' for r in rows),62)
        self.assertTrue(all(r['metrics'] is None for r in rows if r['decision']=='retry'))

    def test_failed_timeout_completed_without_outcome_never_retried(self):
        for failure in ('FAILED','TIMEOUT','OUT_OF_MEMORY','COMPLETED','UNKNOWN'):
            rows=[account(m.ORIGINAL+'_'+str(i),failure if i==8 else 'PREEMPTED') for i in range(64)]
            with mock.patch.object(m.collector,'inspect_result',side_effect=self.inspect),mock.patch.object(m,'attempts',return_value=[]):
                result=m.scan(self.plan,dict(generations=[]),snapshot(rows))
            self.assertEqual(result[8]['decision'],'diagnosis_required',failure)
            self.assertIsNone(result[8]['metrics'])

    def test_live_and_accounting_pending_never_retry(self):
        s=snapshot([account(m.ORIGINAL+'_8','RUNNING')])
        with mock.patch.object(m.collector,'inspect_result',side_effect=self.inspect),mock.patch.object(m,'attempts',return_value=[]):
            rows=m.scan(self.plan,dict(generations=[]),s)
        self.assertEqual(rows[8]['decision'],'wait');self.assertEqual(rows[9]['decision'],'wait')

    def test_closed_native_failure_requires_diagnosis_despite_preemption(self):
        s=snapshot([account(m.ORIGINAL+'_'+str(i)) for i in range(64)])
        with mock.patch.object(m.collector,'inspect_result',side_effect=self.inspect), \
             mock.patch.object(m,'attempts',return_value=[dict(slurm_job_id='9',native_diagnosis_required=True)]):
            rows=m.scan(self.plan,dict(generations=[]),s)
        self.assertEqual(rows[8]['decision'],'diagnosis_required')

    def test_exact_worker_resource_cap_and_scope(self):
        value=m.submission(self.output,self.plan_path,1,[8,9])
        cmd=value['command']
        for item in ('--array=8,9%2','--cpus-per-task=8','--mem=12G','--time=02:30:00',
                     '--partition=cpu-scavenger','--qos=cpu-scavenger','--exclude='+m.EXCLUDE,
                     'fql/qstudy_train_final_route.slurm',m.PLAN_SHA):self.assertIn(item,cmd)
        self.assertFalse(any(item.startswith('--dependency=') for item in cmd))
        self.assertNotIn('--end-iter',cmd) # Physical cap comes from the unchanged collector.
        for number,indices in ((4,[8]),(1,[8,8]),(1,[64]),(1,[-1]),(1,[9,8])):
            with self.assertRaises(ValueError):m.submission(self.output,self.plan_path,number,indices)

    def test_uncertain_submit_adoption_never_blind_resubmit(self):
        scheduler=FakeScheduler(self.completed)
        intent=m.submission(self.output,self.plan_path,1,[8,9])
        pending=m.adopt_or_submit(intent,scheduler,False)
        self.assertIsNone(pending['job']);self.assertEqual(len(scheduler.submitted),0)
        scheduler.names[intent['name']]='777'
        recovered=m.adopt_or_submit(intent,scheduler,False)
        self.assertEqual(recovered['job'],'777');self.assertEqual(len(scheduler.submitted),0)

    def test_duplicate_scheduler_name_refused_and_environment_cleared(self):
        fake=m.Scheduler()
        fake.command=mock.Mock(side_effect=[type('R',(),dict(stdout='11|stage\n12|stage\n'))(),
                                           type('R',(),dict(stdout=''))()])
        with mock.patch.dict(os.environ,USER='test'):
            with self.assertRaisesRegex(ValueError,'Ambiguous'):fake.find_name('stage',1.)
        with mock.patch.dict(os.environ,SLURM_RESTART_COUNT='7'), \
             mock.patch.object(m.subprocess,'run',return_value=type('R',(),dict(stdout='123;cluster\n',stderr='',returncode=0))()) as run:
            self.assertEqual(m.Scheduler().submit(['sbatch','argument'])[0],'123')
            self.assertNotIn('SLURM_RESTART_COUNT',run.call_args.kwargs['env'])

    def test_controller_waits_both_old_arrays_then_exact_missing_and_resume(self):
        scheduler=FakeScheduler(self.completed);scheduler.pending_initial=1
        sleeps=[]
        state=self.run_controller(scheduler,sleep=lambda seconds:sleeps.append(seconds))
        self.assertEqual(state['status'],'completed');self.assertEqual(len(scheduler.submitted),1)
        self.assertEqual(state['generations'][0]['indices'],[8,9]);self.assertEqual(sleeps,[30])
        self.assertEqual(state['collector_report_status'],'complete')
        before=copy.deepcopy(scheduler.submitted)
        again=self.run_controller(scheduler)
        self.assertEqual(again['status'],'completed');self.assertEqual(scheduler.submitted,before)
        self.assertEqual(len(again['generations']),1)

    def test_active_original_does_not_read_attempt_writers(self):
        scheduler=FakeScheduler(self.completed);scheduler.pending_initial=1
        args=type('Args',(),dict(plan=self.plan_path,plan_sha256=m.PLAN_SHA,output=self.output))()
        with mock.patch.dict(os.environ,SLURM_JOB_ID='900',USER='test'), \
             mock.patch.object(m,'binding',return_value=(self.plan,self.pinned)), \
             mock.patch.object(m.collector,'inspect_result',side_effect=self.inspect), \
             mock.patch.object(m,'attempts',return_value=[]) as attempt, \
             mock.patch.object(m.collector,'report',side_effect=self.finish):
            state=m.controller(args,scheduler=scheduler,sleep=lambda seconds:self.assertEqual(attempt.call_count,0))
        self.assertEqual(state['status'],'completed')

    def test_any_unsettled_exact_attempt_blocks_all_new_dispatch(self):
        scheduler=FakeScheduler(self.completed)
        original=scheduler.snapshot;first=[True]
        def delayed(jobs):
            result=original(jobs)
            if first[0]:
                first[0]=False
                result['accounting']=[r for r in result['accounting'] if r['job_id']!=m.ORIGINAL+'_9']
            return result
        scheduler.snapshot=delayed;sleeps=[]
        def slept(seconds):
            self.assertEqual(scheduler.submitted,[]);sleeps.append(seconds)
        state=self.run_controller(scheduler,sleep=slept)
        self.assertEqual(state['status'],'completed');self.assertEqual(sleeps,[30])
        self.assertEqual(len(scheduler.submitted),1)

    def test_interrupted_submit_and_orphan_intent_adopt_without_duplicate(self):
        for orphan in (False,True):
            self.completed=set(range(64))-{8,9}
            self.output=self.root/('restart_'+str(orphan))
            scheduler=FakeScheduler(self.completed)
            submit=scheduler.submit
            def interrupted(command):
                submit(command)
                raise KeyboardInterrupt('simulate loss after successful sbatch before journal')
            scheduler.submit=interrupted
            with self.assertRaises(KeyboardInterrupt):self.run_controller(scheduler)
            scheduler.submit=submit
            saved=m.read(self.output/'state.json')
            self.assertIsNone(saved['generations'][0]['job'])
            if orphan:
                # A prior durable atomic state prefix can precede the intent;
                # the separately immutable intent must still be adopted safely.
                saved['generations']=[];m.atomic_json(self.output/'state.json',saved)
            state=self.run_controller(scheduler)
            self.assertEqual(state['status'],'completed');self.assertEqual(len(scheduler.submitted),1)
            self.assertEqual(state['generations'][0]['job'],'701')

    def test_bounded_three_generations_no_failure_label_after_repeated_preemption(self):
        scheduler=FakeScheduler(self.completed,preempt_generations=4)
        state=self.run_controller(scheduler)
        self.assertEqual(state['status'],'unresolved');self.assertEqual(len(scheduler.submitted),3)
        self.assertEqual(state['collector_report_status'],'incomplete');self.assertEqual(len(self.completed),62)
        self.assertTrue(all(g['indices']==[8,9] for g in state['generations']))
        for row in state['records']:
            if row['index'] in (8,9):self.assertIsNone(row['metrics'])

    def test_scheduler_transient_error_backoff_and_source_state_tamper(self):
        scheduler=FakeScheduler(self.completed);scheduler.error_count=1;sleeps=[]
        state=self.run_controller(scheduler,sleep=lambda seconds:sleeps.append(seconds))
        self.assertEqual(state['status'],'completed');self.assertEqual(sleeps,[30])
        saved=m.read(self.output/'state.json');saved['binding']['version']='changed'
        m.atomic_json(self.output/'state.json',saved)
        with self.assertRaisesRegex(ValueError,'Recovery plan'):self.run_controller(scheduler)

    def test_immutable_intent_and_generation_tamper_refused(self):
        scheduler=FakeScheduler(self.completed);self.run_controller(scheduler)
        state=m.read(self.output/'state.json');state['generations'][0]['indices']=[8,9,10]
        with self.assertRaisesRegex(ValueError,'submission identity'):
            m.checked_state(state,self.pinned,self.output,self.plan_path)

    def test_family_denominator_and_plan_digest_refused(self):
        real_sources={'fql/test_source': 'hash'}
        for mode in ('family','variant','identity'):
            bad=copy.deepcopy(self.plan)
            if mode=='family':bad['tasks'][0]['family']=999
            if mode=='variant':bad['tasks'][0]['variant']=1
            if mode=='identity':bad['tasks'][0]['provenance']['split']='validation'
            with mock.patch.object(m.collector,'checked_plan',return_value=bad), \
                 mock.patch.object(m.collector,'native_source',return_value=self.tasks[0]['native']), \
                 mock.patch.object(m,'source_hashes',return_value=real_sources):
                with self.assertRaises(ValueError):m.binding(self.plan_path,m.PLAN_SHA,self.output)
        with self.assertRaisesRegex(ValueError,'original fixed64'):m.binding(self.plan_path,'0'*64,self.output)

    def test_native_attempt_worker_source_and_spec_are_pinned(self):
        task=self.tasks[8];folder=Path(task['output']);attempt=folder/'attempt_123'
        (attempt/'source').mkdir(parents=True)
        names=('graph_o2o_full_route.py','graph_o2o_full_route.slurm',
               'graph_o2o_finish.py','graph_o2o_finish.sh','graph_o2o_archive.py')
        hashes={}
        for name in names:
            path=attempt/'source'/name;path.write_text('frozen '+name)
            hashes['fql/'+name]=m.sha(path)
        spec=dict(source_run=task['native']['run'],source_manifest_sha256='native',
                  method=task['method'],route_end_iteration=20,run_id=task['run_id'],storage='scratch')
        m.atomic_json(folder/'spec.json',spec)
        value=dict(slurm_job_id='123',spec_sha256=m.sha(folder/'spec.json'),
                   physical_run=str(m.ROOT/'backend/results'/(task['run_id']+'_a123')),
                   source_sha256={Path(k).name:v for k,v in hashes.items()})
        path=attempt/'binding.json';m.atomic_json(path,value)
        evidence=[dict(binding=dict(path=str(path),sha256=m.sha(path)),slurm_job_id='123',
                       physical_run=value['physical_run'])]
        with mock.patch.object(m.collector,'attempt_evidence',return_value=evidence):
            self.assertEqual(m.attempts(task,hashes),evidence)
            changed=copy.deepcopy(value);changed['source_sha256']={}
            m.atomic_json(path,changed)
            with self.assertRaisesRegex(ValueError,'changed physical worker'):m.attempts(task,hashes)
            changed=copy.deepcopy(value);changed['physical_run']='/some_foreign_run';m.atomic_json(path,changed)
            with self.assertRaisesRegex(ValueError,'source or physical'):m.attempts(task,hashes)
            m.atomic_json(path,value);(attempt/'source'/names[0]).write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Changed TRAIN source'):m.attempts(task,hashes)


if __name__=='__main__':unittest.main()
