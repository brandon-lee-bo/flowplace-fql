"""Locked fresh DRT finish: exact development worker, no new layouts or feedback.

Only saved indices 0..3 are eligible. Unknown/tool failures remain unresolved
and do not stop other ready requests; only genuine infrastructure interruptions
can be retried. This adapter never repairs one method with a different budget.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from . import qstudy_fresh_eval as fresh
from . import qstudy_fresh_native as finish
from . import qstudy_actor_report as native
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_controller import states
from .graph_o2o_final_evidence import audit_final_outcome

ROOT=fresh.ROOT
VERSION='fresh_detailed_route_v1'
METRICS=('DRT_WL','WNS','TNS','Power','StdCellArea','DRC')
WORKER=dict(threads=8,memory='12G',time='02:30:00',partition='cpu-scavenger',
    qos='cpu-scavenger',exclude=finish.EXCLUDE)
SOURCE_NAMES=tuple(dict.fromkeys((*finish.SOURCE_NAMES,'fql/qstudy_fresh_route.py',
    'fql/qstudy_fresh_route.slurm','fql/qstudy_fresh_route_worker.slurm',
    'fql/graph_o2o_full_route.py','fql/graph_o2o_full_route.slurm',
    'fql/graph_o2o_finish.py','fql/graph_o2o_finish.sh','fql/graph_o2o_archive.py',
    'fql/graph_o2o_final_evidence.py','backend/scripts/canonical_table2_metrics.py',
    'backend/scripts/grt_metrics.py','backend/scripts/grt_provenance.py',
    'analysis/audit_table2_macro_preservation.py',
    'upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py',
    'backend/thirdparty/OpenROAD-flow-scripts/Makefile',
    'backend/thirdparty/OpenROAD-flow-scripts/scripts/detail_route.tcl',
    'backend/thirdparty/OpenROAD-flow-scripts/scripts/fillcell.tcl',
    'backend/thirdparty/OpenROAD-flow-scripts/scripts/final_report.tcl',
    'backend/thirdparty/OpenROAD-flow-scripts/scripts/report_metrics.tcl')))


def tasks_for(output,dispatch,selection_sha):
    result=[]
    for row in dispatch:
        if row['index'] not in fresh.DRT_INDICES:
            continue
        key=f"{row['stream']}_s{row['seed']}_n{row['index']}"
        result.append(dict(row,key=key,source_outcome=str(Path(row['sample_dir'])/'outcome.json'),
            output=str(output/'full_route'/key),method='fresh_'+row['stream'],
            run_id='go2o_fresh_'+selection_sha[:12]+'_'+key))
    if (len({r['key'] for r in result})!=len(result) or
        len(result)*2!=len(dispatch) or any(r['round']!=0 for r in result)):
        raise ValueError('Fresh DRT exact preselected denominator changed')
    return result


def prepare(args):
    selection=Path(args.selection).resolve()
    checked=finish.checked_exports(selection,args.selection_sha256)
    lock,_,_,_,noise,_,dispatch,inventory=checked
    output=Path(args.output).resolve()
    for stream in lock['streams']:
        protected=Path(stream['evaluation_config']['path']).parent
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('Fresh DRT output cannot overwrite a stream experiment')
    plan=dict(version=VERSION,selection=fresh.binding(selection),output=str(output),
        declaration=fresh.binding(Path(lock['output_root'])/'declaration.json'),noise=noise,
        export_inventory=inventory,tasks=tasks_for(output,dispatch,args.selection_sha256),
        source_sha256={n:sha(ROOT/n) for n in SOURCE_NAMES},concurrency=2,worker=WORKER,
        design=lock['design'],indices=fresh.DRT_INDICES,route_cap=20,
        backend_storage='scratch',training_updates=0,layout_generation_calls=0,
        feedback_eligible=False,selection_independent_of_GRT=True,
        route_settings='unchanged graph_o2o_full_route -> graph_o2o_finish; only droute_end_iter20')
    plan['requested']=len(plan['tasks'])
    output.mkdir(parents=True,exist_ok=True);fresh.immutable(output/'plan.json',plan)
    return plan


def checked_plan(path,expected_sha):
    path=Path(path).resolve()
    if sha(path)!=expected_sha:
        raise ValueError('Fresh DRT plan SHA changed')
    plan=fresh.read(path)
    if (plan['version']!=VERSION or plan['concurrency']!=2 or plan['worker']!=WORKER or
        plan['indices']!=fresh.DRT_INDICES or plan['route_cap']!=20 or
        plan['backend_storage']!='scratch' or plan['training_updates']!=0 or
        plan['layout_generation_calls']!=0 or plan['feedback_eligible'] or
        not plan['selection_independent_of_GRT'] or Path(plan['output'])!=path.parent or
        set(plan['source_sha256'])!=set(SOURCE_NAMES)):
        raise ValueError('Fresh DRT fixed worker/budget/namespace changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen fresh DRT source changed: '+name)
    selection=fresh.checked_file(plan['selection'])
    checked=finish.checked_exports(selection,plan['selection']['sha256'])
    lock,_,_,_,noise,_,dispatch,inventory=checked
    if (plan['export_inventory']!=inventory or plan['noise']!=noise or
        plan['declaration']!=fresh.binding(Path(lock['output_root'])/'declaration.json') or
        plan['design']!=lock['design'] or
        plan['tasks']!=tasks_for(path.parent,dispatch,plan['selection']['sha256']) or
        plan['requested']!=len(plan['tasks'])):
        raise ValueError('Sealed fresh DRT input/export inventory changed')
    return plan,checked


def inspect_sources(plan,checked,*,seal=False):
    """Validate GRT natively and bind saved continuation files before DRT."""
    lookup={(r['stream']['id'],r['seed']):r for r in checked[5]};contexts={};results=[]
    for task in plan['tasks']:
        record=lookup[(task['stream'],task['seed'])]
        row=record['rows'][task['index']]
        if task['experiment'] not in contexts:
            contexts[task['experiment']]=finish.backend.context(task['experiment'])
        cfg,verification,physical=contexts[task['experiment']]
        result=finish.inspect_outcome(row,record['tasks'],cfg,physical,verification)
        target=Path(plan['output'])/'source_bindings'/f"{task['key']}.json"
        if result['state'] in native.ADMITTED:
            try:
                binding=dict(version=VERSION,task=task,GRT=result,continuation_inputs={})
                if result['state']=='success':
                    run=Path(result['physical_run']);design=plan['design'];work=run/'orfs_work'
                    files=[run/'manifest.json',run/'metrics.json',run/'backend_macro_audit.json']
                    files += [work/'results/nangate45'/design/'eval'/n for n in
                        ('1_synth.v','1_synth.sdc','5_1_grt.odb','5_1_grt.sdc','route.guide')]
                    for group in ('logs','reports'):
                        directory=work/group/'nangate45'/design/'eval'
                        if not directory.is_dir():
                            raise ValueError('Measured GRT continuation directory missing: '+str(directory))
                        files.extend(p for p in directory.rglob('*') if p.is_file())
                    binding['continuation_inputs']={str(p):sha(p) for p in sorted(files)}
                if target.exists():
                    if fresh.read(target)!=binding:
                        raise ValueError('Previously sealed fresh GRT continuation input changed')
                elif seal:
                    target.parent.mkdir(parents=True,exist_ok=True);fresh.immutable(target,binding)
                result=dict(result,source_binding=binding,
                    source_binding_artifact=fresh.binding(target) if target.exists() else None)
            except Exception as error:
                result=dict(result,state='invalid',metrics=None,integrity_error=str(error))
        elif target.exists():
            result=dict(result,state='invalid',metrics=None,
                integrity_error='Previously sealed GRT outcome disappeared or became invalid')
        results.append(result)
    return results


def expected_spec(task,source):
    return dict(source_run=str(Path(source['physical_run']).resolve()),
        source_manifest_sha256=source['native']['physical_manifest_sha256'],
        source_outcome=task['source_outcome'],source_outcome_sha256=source['outcome']['sha256'],
        method=task['method'],run_id=task['run_id'],route_end_iteration=20,storage='scratch')


def invocation_binding(plan,task,source):
    if not source.get('source_binding_artifact'):
        raise ValueError('Fresh DRT requires the sealed native GRT continuation binding')
    return dict(plan=fresh.binding(Path(plan['output'])/'plan.json'),task=task,
        source_binding=source['source_binding_artifact'],source_outcome=source['outcome'],
        worker_sha256=plan['source_sha256']['fql/graph_o2o_full_route.py'],
        route_cap=20,threads=8,feedback_eligible=False)


def inspect_final(plan,task,source):
    result=dict(key=task['key'],stream=task['stream'],seed=task['seed'],index=task['index'],
        noise_sha256=task['noise_sha256'],state='missing_GRT',metrics=None,
        source_binding_artifact=source.get('source_binding_artifact'),GRT_state=source['state'])
    if source['state'] not in native.ADMITTED:
        return dict(result,state='invalid' if source['state']=='invalid' else 'missing_GRT',
            integrity_error=source.get('integrity_error'))
    path=Path(task['output'])/'outcome.json'
    if source['state']!='success':
        if path.exists():
            out=fresh.read(path)
            expected=dict(status='upstream_physical_failure',source_outcome=task['source_outcome'],
                source_outcome_sha256=source['outcome']['sha256'],
                failure_class=source['failure_class'],metrics=None)
            if any(out.get(k)!=v for k,v in expected.items()):
                raise ValueError('Fresh upstream failure sidecar changed')
        return dict(result,state='upstream_physical_failure',failure_class=source['failure_class'],
            outcome=fresh.binding(path) if path.exists() else None)
    spec=expected_spec(task,source);folder=path.parent;spec_path=folder/'spec.json'
    invocation=folder/'fresh_invocation.json'
    if invocation.exists() or spec_path.exists() or path.exists() or any(folder.glob('attempt_*/binding.json')):
        if not invocation.exists() or fresh.read(invocation)!=invocation_binding(plan,task,source):
            raise ValueError('DRT invocation differs from its sealed fresh input')
        if spec_path.exists() and fresh.read(spec_path)!=spec:
            raise ValueError('Final DRT does not bind the exact locked fresh source/budget')
    if not path.exists():
        return dict(result,state='missing_DRT')
    out=fresh.read(path)
    if (out.get('status')!='completed' or out.get('spec')!=spec or
        fresh.read(path.parent/'spec.json')!=spec or out['spec_sha256']!=sha(path.parent/'spec.json')):
        raise ValueError('Final DRT does not bind the exact locked fresh source/budget')
    audit=audit_final_outcome(out,design=plan['design'],route_cap=20)
    return dict(result,state='completed',metrics=out['metrics'],outcome=fresh.binding(path),
        physical_run=out['physical_run'],physical_manifest_sha256=out['physical_manifest_sha256'],
        metrics_sha256=out['metrics_sha256'],route_drc_clean=out['route_drc_clean'],
        physical_identity=audit['physical_identity'],final_native_audit=audit)


def inspect_finals(plan,sources):
    if len(sources)!=len(plan['tasks']):
        raise ValueError('Fresh GRT source denominator changed')
    rows=[]
    for task,source in zip(plan['tasks'],sources):
        try:row=inspect_final(plan,task,source)
        except Exception as error:
            row=dict(key=task['key'],stream=task['stream'],seed=task['seed'],index=task['index'],
                noise_sha256=task['noise_sha256'],state='invalid',metrics=None,integrity_error=str(error))
        rows.append(row)
    completed=[r for r in rows if r['state']=='completed']
    if len({fresh.object_sha(r['physical_identity']) for r in completed})>1:
        for row in completed:row.update(state='invalid',metrics=None,integrity_error='Mixed DRT physical settings')
    seen={}
    for row in completed:
        if row['physical_run'] in seen:
            for duplicate in (row,seen[row['physical_run']]):
                duplicate.update(state='invalid',metrics=None,integrity_error='Distinct fresh requests reuse one DRT run')
        seen[row['physical_run']]=row
    return rows


def next_actions(plan,state,rows,observer=states):
    ready=[];busy=0;attention={};observations={}
    if len(rows)!=len(plan['tasks']):raise ValueError('Fresh DRT result denominator changed')
    for i,(task,row) in enumerate(zip(plan['tasks'],rows)):
        if row['state'] in ('completed','upstream_physical_failure'):continue
        if row['state']=='invalid':attention[str(i)]=row.get('integrity_error');continue
        jobs=list(state['jobs'].get(str(i),[]))
        for path in sorted(Path(task['output']).glob('attempt_*/binding.json')):
            attempt=fresh.read(path)
            spec=Path(task['output'])/'spec.json'
            if not spec.exists() or attempt['spec_sha256']!=sha(spec):
                attention[str(i)]='Foreign/changed DRT attempt binding';break
            if attempt['slurm_job_id'] not in jobs:jobs.append(attempt['slurm_job_id'])
        if str(i) in attention:continue
        if row['state']=='missing_GRT':continue
        if not jobs:ready.append(i);continue
        observed=[observer(job) for job in jobs];observations[str(i)]=dict(jobs=jobs,states=observed)
        # Historical retryable attempts do not hide the latest real failure;
        # successful native outputs were already recovered above.
        try:decision=finish.disposition(observed)
        except RuntimeError as error:attention[str(i)]=str(error);continue
        if decision=='wait':busy+=1
        else:ready.append(i)
    return dict(submit_indices=ready[:max(0,plan['concurrency']-busy)],busy=busy,
        attention=attention,observations=observations,
        unresolved=[i for i,r in enumerate(rows) if r['state'] not in ('completed','upstream_physical_failure')])


def worker(args):
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=8:
        raise RuntimeError('Fresh DRT requires exactly8Slurm CPUs')
    plan,checked=checked_plan(args.plan,args.plan_sha256)
    if not 0<=args.index<len(plan['tasks']):raise IndexError('Outside sealed fresh DRT tasks')
    task=plan['tasks'][args.index];output=Path(task['output']);output.mkdir(parents=True,exist_ok=True)
    with (output/'fresh_worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=inspect_sources(dict(plan,tasks=[task]),checked,seal=True)[0]
        row=inspect_final(plan,task,source)
        if row['state'] in ('completed','upstream_physical_failure'):return
        if row['state']!='missing_DRT':raise ValueError('Fresh GRT input is not ready and verified')
        fresh.immutable(output/'fresh_invocation.json',invocation_binding(plan,task,source))
        # Exact frozen host worker; it also rechecks source manifest/geometry,
        # resumes completed physical attempts, and invokes unchanged finish.sh.
        subprocess.run([sys.executable,'-u','-m','fql.graph_o2o_full_route',
            '--source-outcome',task['source_outcome'],'--run-id',task['run_id'],
            '--method',task['method'],'--storage','scratch','--output',task['output'],
            '--end-iter','20'],cwd=ROOT,check=True)
        finished=inspect_final(plan,task,source)
        if finished['state']!='completed':
            raise RuntimeError('Frozen route worker returned without a validated final outcome: '+finished['state'])


def controller(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for fresh DRT controller')
    args.plan=Path(args.plan).resolve();plan,checked=checked_plan(args.plan,args.plan_sha256)
    output=Path(plan['output'])
    with (output/'controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'state.json';binding=dict(plan=fresh.binding(args.plan),selection=plan['selection'])
        state=fresh.read(path) if path.exists() else dict(binding=binding,jobs={},events=[])
        if state['binding']!=binding:raise ValueError('Fresh DRT controller binding changed')
        while True:
            plan,checked=checked_plan(args.plan,args.plan_sha256)
            sources=inspect_sources(plan,checked,seal=True);rows=inspect_finals(plan,sources)
            actions=next_actions(plan,state,rows)
            state.update(status='running',controller_job=os.environ['SLURM_JOB_ID'],
                last_update_unix=time.time(),**actions)
            if not actions['unresolved']:
                fresh.immutable(output/'completed.json',dict(status='all_requested_outcomes_accounted',
                    binding=binding,logical_requests=len(rows),training_updates=0,feedback_eligible=False,
                    outcomes=[dict(key=r['key'],state=r['state'],outcome=r.get('outcome')) for r in rows]))
                state['status']='completed';atomic_json(path,state);return
            for i in actions['submit_indices']:
                key=str(i);attempt=len(state['jobs'].get(key,[]))
                name=f"qsdrt_{args.plan_sha256[:12]}_i{i}_a{attempt}"
                job=finish.submit_unique(['fql/qstudy_fresh_route_worker.slurm','worker','--plan',str(args.plan),
                    '--plan-sha256',args.plan_sha256,'--index',key],name)
                state['jobs'].setdefault(key,[]).append(job)
                state['events'].append(dict(event='missing_only_submission',index=i,job=job,attempt=attempt))
                atomic_json(path,state)
            if actions['attention'] and not actions['busy'] and not actions['submit_indices']:
                state['status']='attention_required';atomic_json(path,state);return
            atomic_json(path,state);time.sleep(45)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--selection-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    for command in ('controller','worker'):
        p=commands.add_parser(command);p.add_argument('--plan',type=Path,required=True)
        p.add_argument('--plan-sha256',required=True)
        if command=='worker':p.add_argument('--index',type=int,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    result=globals()[args.command](args)
    if result is not None:print(result,flush=True)


if __name__=='__main__':main()
