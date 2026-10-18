"""Sealed fresh-evaluation GRT finish adapter; no training or layout generation.

Preparation requires an existing selection/declaration and ALL completed export
inventories. The unchanged eight-CPU worker measures their exact saved DEFs.
Missing-only recovery admits PREEMPTED/NODE_FAIL/BOOT_FAIL; genuine failures,
unknown accounting and ordinary code errors are never relabeled or overridden.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from . import qstudy_fresh_eval as fresh
from . import qstudy_backend as backend
from . import qstudy_actor_report as native
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_controller import LIVE, RESTARTABLE, states

ROOT=fresh.ROOT
VERSION='fresh_native_finish_v1'
EXCLUDE='natural-selection,enterprise,blue-space'
SOURCE_NAMES=tuple(dict.fromkeys((*fresh.SOURCE_NAMES,'fql/qstudy_fresh_native.py',
    'fql/qstudy_fresh_native.slurm','fql/qstudy_fresh_native_worker.slurm',
    'fql/qstudy_backend.slurm','fql/graph_o2o_controller.py','fql/qstudy_actor_report.py',
    'fql/graph_o2o_design_verify.py','fql/graph_o2o_label.sh','fql/graph_o2o_geometry.sh',
    'fql/graph_o2o_geometry.tcl','backend/scripts/run_grt_label.py','backend/scripts/result_storage.py')))


def declaration(lock,selection_path,selection_sha,physicals):
    """Reconstruct the frozen export declaration without invoking its writer."""
    return dict(version=fresh.VERSION,selection_lock=fresh.binding(selection_path),
        expected_noise_arrays=fresh.COUNT,role='evaluation',terminal_mask=0,
        training_feedback_eligible=False,offline_train_eligible=False,
        evaluation_purpose='fresh_confirmation',evaluation=lock['evaluation'],
        streams=[dict(id=s['id'],kind=s['kind'],seeds=s['seeds'],
            expected_requests=fresh.COUNT*len(s['seeds']),selected_original_round=s['selected_round'],
            source_config=s['source_config'],checkpoints=s['checkpoints'],source_sampler=fresh.SAMPLERS[s['kind']],
            evaluation_config=s['evaluation_config'],native_context=physicals[s['id']]) for s in lock['streams']],
        backend_calls=0,optimizer_updates=0,selection='all locked seeds and indices0..7; no Q/metric selection',
        independent_public_checkpoints=1,drt_selection_independent_of_GRT=True)


def read_noise(root,selection_sha):
    """Pure saved-byte reader; unlike shared_noise(), creates no files/locks."""
    folder=Path(root)/'shared_noise';path=folder/'noises.npz'
    with np.load(path,allow_pickle=False) as saved:
        arrays=saved['noise'].copy()
        if str(saved['selection_lock_sha256'].item())!=selection_sha:
            raise ValueError('Actual fresh noise belongs to another selection')
    if arrays.ndim!=3:
        raise ValueError('Invalid actual fresh noise shape')
    expected=dict(fresh.noise_manifest(arrays,selection_sha,arrays.shape[1]),artifact=fresh.binding(path))
    if fresh.read(folder/'manifest.json')!=expected:
        raise ValueError('Saved shared fresh noise manifest/bytes changed')
    return arrays,expected


def checked_exports(selection_path,selection_sha,*,require_complete=True):
    lock,configs,physicals=fresh.validate_lock(selection_path,selection_sha)
    root=Path(lock['output_root'])
    declared=declaration(lock,selection_path,selection_sha,physicals)
    if fresh.read(root/'declaration.json')!=declared:
        raise ValueError('Preexisting fresh declaration changed')
    noise_path=root/'shared_noise/noises.npz'
    noises,noise_info=read_noise(root,selection_sha) if noise_path.exists() else (None,None)
    records=[];dispatch=[];inventory={str(root/'declaration.json'):sha(root/'declaration.json')}
    if noise_info is not None:
        inventory.update({str(root/'shared_noise/noises.npz'):sha(noise_path),
            str(root/'shared_noise/manifest.json'):sha(root/'shared_noise/manifest.json')})
    for stream in lock['streams']:
        cfg,physical=configs[stream['id']],physicals[stream['id']]
        experiment=Path(stream['evaluation_config']['path']).parent
        for seed in stream['seeds']:
            folder=experiment/f'seed_{seed}/round_0/requests'
            record=dict(stream=stream,seed=seed,experiment=str(experiment),status='not_exported',rows=[],
                expected_requests=fresh.COUNT,errors=[])
            marker=folder/'completed.json'
            if marker.exists():
                if noises is None:
                    raise ValueError('Completed fresh export has no saved actual shared noise')
                fresh.verify_completion(folder,cfg,selection_sha,declared)
                rows,tasks=load_requests(experiment,seed,0)
                if len(rows)!=fresh.COUNT or [r['index'] for r in rows]!=list(range(fresh.COUNT)):
                    raise ValueError('Fresh eight-request ordered inventory changed')
                for index,row in enumerate(rows):
                    sample=fresh.verify_row(row,stream,seed,index,cfg,physical,selection_sha,declared,noises,noise_info)
                    expected_sample=folder/stream['id']/'evaluation'/str(index)
                    if sample.resolve()!=expected_sample.resolve() or not sample.is_relative_to(experiment):
                        raise ValueError('Fresh sample escaped its exact stream/seed/index namespace')
                    files=[sample/'request.json',sample/'actions.npz']
                    if row.get('def_path'):
                        files.extend([Path(row['def_path']),Path(row['def_path']).with_suffix('.audit.json')])
                    if stream['kind']=='public_official20':
                        files.append(sample/'trajectory.pt')
                    inventory.update({str(p):sha(p) for p in files})
                    dispatch.append(dict(experiment=str(experiment),stream=stream['id'],seed=seed,round=0,index=index,
                        sample_dir=str(sample),tasks_sha256=sha(tasks),request_sha256=sha(sample/'request.json'),
                        selection_lock_sha256=selection_sha,checkpoint_sha256=row['checkpoint_sha256'],
                        noise_sha256=row['noise_sha256'],action_sha256=row['action_sha256'],
                        raw_action_sha256=row['raw_action_sha256'],role='evaluation',
                        training_feedback_eligible=False,offline_train_eligible=False))
                inventory.update({str(tasks):sha(tasks),str(marker):sha(marker)})
                record.update(status='exported',rows=rows,tasks=str(tasks),tasks_sha256=sha(tasks),
                    export_completion_sha256=sha(marker))
            elif require_complete:
                raise ValueError('All locked fresh exports must complete before native dispatch')
            records.append(record)
    expected=sum(fresh.COUNT*len(s['seeds']) for s in lock['streams'])
    if require_complete and len(dispatch)!=expected:
        raise ValueError('Fresh native dispatch denominator changed')
    if len({r['sample_dir'] for r in dispatch})!=len(dispatch):
        raise ValueError('Duplicate native sample identity')
    if sha(selection_path)!=selection_sha:
        raise ValueError('Selection changed during fresh inventory read')
    return lock,configs,physicals,declared,noise_info,records,dispatch,inventory


def inspect_outcome(row,tasks,cfg,physical,verification):
    sample=Path(row['sample_dir']);path=sample/'outcome.json'
    result=dict(index=row['index'],seed=row['seed'],stream=row['arm'],role=row['role'],
        noise_sha256=row['noise_sha256'],raw_action_sha256=row['raw_action_sha256'],
        action_sha256=row['action_sha256'],checkpoint_sha256=row['checkpoint_sha256'],
        sample_dir=str(sample),state='missing',metrics=None,
        requested_geometry_legal=(row.get('geometry') or {}).get('legal'),
        rounded_geometry_legal=row.get('rounded_geometry'),
        fresh_drt_preselected=row['fresh_drt_preselected'],fresh_drt_repair_cap=row['fresh_drt_repair_cap'])
    if not path.exists():
        return result
    try:
        out=fresh.read(path)
        for key in ['seed','round','arm','role','index','graph','training_feedback_eligible']:
            if out.get(key)!=row.get(key):
                raise ValueError('Fresh native outcome changed request identity: '+key)
        if out.get('physical_run'):
            run=Path(out['physical_run'])
            manifest=fresh.read(run/'manifest.json')
            if manifest.get('success') and not (run/'backend_macro_audit.json').is_file():
                raise ValueError('Saved native geometry audit missing; read-only inspection cannot run OpenROAD')
        backend.checked_existing(out,row,sample,Path(tasks),physical,cfg,verification)
        if out['status'] not in native.ADMITTED:
            raise ValueError('Infrastructure/unresolved result is not a physical target')
        if out['status']=='geometry_failure' and (
            row.get('pre_backend_failure') not in ('normalized_geometry','rounded_DEF_geometry') or
            out.get('failure_class')!=row['pre_backend_failure'] or out.get('stage')!='pre_backend'):
            raise ValueError('Geometry failure has no matching pre-backend cause')
        values=out['metrics'] if out['status']=='success' else None
        if values is not None and (any(type(values.get(k)) not in (int,float) or not np.isfinite(values[k])
                for k in native.METRICS) or (native.costs(values)<0).any()):
            raise ValueError('Success requires seven finite native metrics and nonnegative physical costs')
        result.update(state=out['status'],metrics=values,outcome=fresh.binding(path),
            native=native.native_fingerprints(out),backend_macro_legal=out.get('backend_macro_legal'),
            failure_class=out.get('failure_class'),stage=out['stage'],detailed_route_DRC=None,
            physical_run=out.get('physical_run'))
    except Exception as error:
        result.update(state='invalid',integrity_error=str(error),metrics=None)
    return result


def prepare(args):
    selection=Path(args.selection).resolve()
    checked=checked_exports(selection,args.selection_sha256)
    lock,configs,physicals,declared,noise_info,records,dispatch,inventory=checked
    output=Path(args.output).resolve()
    protected=[Path(s['evaluation_config']['path']).parent for s in lock['streams']]
    if any(output==p or output in p.parents or p in output.parents for p in protected):
        raise ValueError('Native finish plan cannot overwrite any stream experiment')
    output.mkdir(parents=True,exist_ok=True)
    source={n:sha(ROOT/n) for n in SOURCE_NAMES}
    plan=dict(version=VERSION,selection=fresh.binding(selection),output=str(output),
        declaration=fresh.binding(Path(lock['output_root'])/'declaration.json'),
        noise=noise_info,export_inventory=inventory,dispatch=dispatch,
        expected_requests=len(dispatch),source_sha256=source,concurrency=4,
        worker=dict(threads=8,memory='24G',time='02:00:00',partition='cpu-scavenger',qos='cpu-scavenger',exclude=EXCLUDE),
        no_training_feedback=True,training_updates=0,layout_generation_calls=0,
        drt_indices=fresh.DRT_INDICES,drt_repair_cap=fresh.REPAIR_CAP,drt_execution_implemented=False)
    fresh.immutable(output/'plan.json',plan)
    return plan


def checked_plan(path,expected_sha):
    path=Path(path).resolve()
    if sha(path)!=expected_sha:
        raise ValueError('Fresh native finish plan SHA changed')
    plan=fresh.read(path)
    if (plan.get('version')!=VERSION or plan['concurrency']!=4 or not plan['no_training_feedback'] or
        plan['training_updates']!=0 or plan['layout_generation_calls']!=0 or
        plan['worker']!=dict(threads=8,memory='24G',time='02:00:00',partition='cpu-scavenger',qos='cpu-scavenger',exclude=EXCLUDE) or
        plan['drt_indices']!=fresh.DRT_INDICES or plan['drt_repair_cap']!=20 or plan['drt_execution_implemented'] or
        set(plan['source_sha256'])!=set(SOURCE_NAMES)):
        raise ValueError('Fresh finish budget/source/worker contract changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen fresh finish source changed: '+name)
    selection=fresh.checked_file(plan['selection'])
    result=checked_exports(selection,plan['selection']['sha256'])
    if (result[6]!=plan['dispatch'] or result[7]!=plan['export_inventory'] or result[4]!=plan['noise'] or
        fresh.binding(Path(result[0]['output_root'])/'declaration.json')!=plan['declaration'] or
        len(plan['dispatch'])!=plan['expected_requests'] or Path(plan['output'])!=path.parent):
        raise ValueError('Sealed fresh dispatch/export/noise inventory changed')
    return plan,result


def inspect_dispatch(plan,checked):
    lock,configs,physicals,_,_,records,_,_=checked
    lookup={(r['stream']['id'],r['seed']):r for r in records}
    results=[]
    for task in plan['dispatch']:
        record=lookup[(task['stream'],task['seed'])];row=record['rows'][task['index']]
        cfg,verification,physical=backend.context(task['experiment'])
        results.append(inspect_outcome(row,record['tasks'],cfg,physical,verification))
    return results


def submission_environment():
    # Requeue belongs to the new worker's own allocation, never its controller.
    environment=dict(os.environ)
    environment.pop('SLURM_RESTART_COUNT',None)
    return environment


def submit_unique(arguments,name):
    """Adopt one exact own-stage job after controller preemption/submission gap."""
    env=submission_environment()
    queued=subprocess.run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j'],
        text=True,capture_output=True,check=True,env=env)
    matches={line.split('|')[0].split('_')[0] for line in queued.stdout.splitlines()
             if '|' in line and line.split('|')[1].strip()==name}
    if not matches:
        accounted=subprocess.run(['sacct','-u',os.environ['USER'],'--name='+name,'-S','now-2days',
            '-X','-n','-P','-o','JobIDRaw,JobName%200'],text=True,capture_output=True,check=True,env=env)
        matches={line.split('|')[0].split('_')[0] for line in accounted.stdout.splitlines()
                 if '|' in line and line.split('|')[1].strip()==name}
    if len(matches)>1:
        raise RuntimeError('Duplicate jobs share a sealed fresh stage identity')
    if matches:
        return matches.pop()
    result=subprocess.run(['sbatch','--parsable','--job-name='+name,*arguments],cwd=ROOT,
                          text=True,capture_output=True,env=env)
    if result.returncode or not result.stdout.strip().split(';')[0].isdigit():
        raise RuntimeError('Fresh native submission failed: '+result.stderr)
    return result.stdout.strip().split(';')[0]


def disposition(observations):
    flat=[state for group in observations for state in group]
    if not flat or any(s in LIVE or s in {'ACCOUNTING_PENDING','UNKNOWN'} for s in flat):
        return 'wait'
    if all(s in RESTARTABLE for s in flat):
        return 'retry'
    raise RuntimeError('Missing fresh outcome needs diagnosis, not blind restart: '+str(flat))


def next_actions(plan,state,outcomes,observer=states):
    """Pure scheduling decision; completed/invalid outputs are never rerun."""
    missing=[];busy=0;ready=[];observations={}
    if len(outcomes)!=len(plan['dispatch']):
        raise ValueError('Fresh outcome denominator differs')
    for index,out in enumerate(outcomes):
        if out['state']=='invalid':
            raise ValueError('Invalid native evidence requires diagnosis: '+out.get('integrity_error',''))
        if out['state'] in native.ADMITTED:
            continue
        if out['state']!='missing':
            raise ValueError('Unknown fresh result state')
        missing.append(index)
        jobs=list(state['jobs'].get(str(index),[]))
        # Also account for exact request-native attempts, including a controller
        # crash before its own job journal publication.
        sample=Path(plan['dispatch'][index]['sample_dir'])
        for path in sorted((sample/'backend_attempts').glob('*/attempt.json')):
            attempt=fresh.read(path)
            if (attempt['request_sha256']!=plan['dispatch'][index]['request_sha256'] or
                attempt['tasks_sha256']!=plan['dispatch'][index]['tasks_sha256']):
                raise ValueError('Foreign physical attempt under fresh sample')
            if attempt['job_id'] not in jobs:
                jobs.append(attempt['job_id'])
        if not jobs:
            ready.append(index);continue
        observed=[observer(job) for job in jobs]
        observations[str(index)]=dict(jobs=jobs,states=observed)
        if disposition(observed)=='wait':
            busy+=1
        else:
            ready.append(index)
    slots=max(0,plan['concurrency']-busy)
    return dict(missing=missing,busy=busy,submit_indices=ready[:slots],observations=observations)


def worker(args):
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=8:
        raise RuntimeError('Fresh native worker requires exactly8Slurm CPUs')
    plan,checked=checked_plan(args.plan,args.plan_sha256)
    if not 0<=args.index<len(plan['dispatch']):
        raise IndexError('Fresh dispatch index outside sealed inventory')
    task=plan['dispatch'][args.index]
    # The only physical-call site; unchanged native code retains its lock,
    # completed-run recovery and original Slurm restart semantics.
    backend.measure(argparse.Namespace(experiment=task['experiment'],seed=task['seed'],round=0,
        index=task['index'],storage='scratch'))


def controller(args):
    args.plan=Path(args.plan).resolve()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run fresh finish controller through Slurm')
    plan,checked=checked_plan(args.plan,args.plan_sha256)
    output=Path(plan['output'])
    with (output/'controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'state.json'
        binding=dict(plan=fresh.binding(args.plan),selection=plan['selection'],source_sha256=plan['source_sha256'])
        state=fresh.read(path) if path.exists() else dict(binding=binding,status='running',jobs={},events=[])
        if state['binding']!=binding:
            raise ValueError('Fresh finish journal binding changed')
        while True:
            plan,checked=checked_plan(args.plan,args.plan_sha256)
            outcomes=inspect_dispatch(plan,checked)
            actions=next_actions(plan,state,outcomes)
            state.update(status='running',last_update_unix=time.time(),controller_job=os.environ['SLURM_JOB_ID'],
                missing=actions['missing'],observations=actions['observations'])
            if not actions['missing']:
                fresh.immutable(output/'completed.json',dict(status='completed',binding=binding,
                    outcomes=[dict(index=i,outcome=r['outcome']) for i,r in enumerate(outcomes)],
                    logical_requests=len(outcomes),training_updates=0,new_layouts=0,feedback_eligible=False))
                state['status']='completed';atomic_json(path,state);return
            for index in actions['submit_indices']:
                key=str(index);attempt=len(state['jobs'].get(key,[]))
                name=f"qsf_{args.plan_sha256[:12]}_i{index}_a{attempt}"
                job=submit_unique(['fql/qstudy_fresh_native_worker.slurm','worker','--plan',str(args.plan),
                    '--plan-sha256',args.plan_sha256,'--index',key],name)
                state['jobs'].setdefault(key,[]).append(job)
                state['events'].append(dict(event='missing_only_submission',index=index,job=job,attempt=attempt,
                    unix=time.time()))
                atomic_json(path,state)
            atomic_json(path,state)
            time.sleep(45)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--selection-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    for command in ['controller','worker']:
        p=commands.add_parser(command);p.add_argument('--plan',type=Path,required=True)
        p.add_argument('--plan-sha256',required=True)
        if command=='worker': p.add_argument('--index',type=int,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Fresh native preparation/controller/worker require Slurm')
    result=globals()[args.command](args)
    if result is not None: print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
