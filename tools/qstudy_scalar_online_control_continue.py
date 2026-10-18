"""Explicit continuation stages for the sealed four-arm scalar GRT control.

This addendum never edits the stage0 manifest or its source closure. Training,
export, replay admission and final routing use the original audited modules.
Only deterministic request construction and stage receipts are added here.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
BASE_SOURCE_SHA='9c30c121fa611c26a6368e1224ac6d3646273648c31e64ee0372a9781aed0535'
import hashlib
BASE_FILE=ROOT/'tools/qstudy_scalar_online_control.py'
if hashlib.sha256(BASE_FILE.read_bytes()).hexdigest()!=BASE_SOURCE_SHA:
    raise ValueError('The original reviewed stage0 helper changed')
spec=importlib.util.spec_from_file_location('_sealed_scalar_stage0',BASE_FILE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

VERSION='qstudy_scalar_online_control_continuation_v1'
PHASES=('train1','gate1','prepare1','export1','native1','train2','prepare2','export2','native2')
FUTURE_PHASES=('route_prepare','route')
HELPERS=('tools/qstudy_scalar_online_control_continue.py',
         'tools/qstudy_scalar_online_control_continue.slurm',
         'tools/qstudy_scalar_online_control_continue_worker.slurm')
ADDENDUM='docs/qstudy_scalar_online_control_continuation.md'
read,sha,bound,atomic,immutable=base.read,base.sha,base.bound,base.atomic,base.immutable


def resources(phase):
    if phase.startswith('train'):
        return ['--partition=gpu-scavenger','--qos=gpu-scavenger','--gres=gpu:nvidia_l20:1',
            '--cpus-per-task=2','--mem=20G','--time=00:20:00','--signal=B:USR1@90'],2
    if phase.startswith('export'):return base.resources('export0')
    if phase.startswith('native'):return base.resources('native0')
    if phase=='route':
        return ['--partition=cpu-scavenger','--qos=cpu-scavenger','--cpus-per-task=1',
            '--mem=3G','--time=24:00:00'],1
    return ['--partition=cpu-scavenger','--qos=cpu-scavenger','--cpus-per-task=2',
        '--mem=8G','--time=00:15:00'],2


def create(args):
    original=base.checked_manifest(args.base_manifest,args.base_manifest_sha256)
    output=Path(original['output'])/'continuation_v1'
    names=HELPERS+(ADDENDUM,'fql/graph_o2o_route_controller.py','fql/graph_o2o_route_controller.slurm')
    value=dict(version=VERSION,output=str(output),base_manifest=bound(args.base_manifest,args.base_manifest_sha256),
        implemented_phases=list(PHASES),future_phases=list(FUTURE_PHASES),max_generations=original['max_generations'],
        source_sha256={name:sha(ROOT/name) for name in names},
        stage_resources={p:dict(options=resources(p)[0],concurrency=resources(p)[1]) for p in PHASES},
        scope='Fixed original four-arm protocol only; no new samples, settings, rewards or scientific endpoints')
    output.mkdir(parents=True,exist_ok=True);(output/'logs').mkdir(exist_ok=True)
    for name,digest in value['source_sha256'].items():
        dest=output/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(ROOT/name,dest)
        bound(dest,digest)
    immutable(output/'manifest.json',value)
    print(json.dumps(dict(manifest=bound(output/'manifest.json'),submitted_jobs=0)))


def checked_manifest(path,digest):
    plan=read(bound(path,digest)['path'])
    original=base.checked_manifest(plan['base_manifest']['path'],plan['base_manifest']['sha256'])
    if (plan['version']!=VERSION or Path(plan['output'])!=Path(original['output'])/'continuation_v1' or
        Path(path).resolve()!=Path(plan['output'])/'manifest.json' or plan['implemented_phases']!=list(PHASES) or
        plan['future_phases']!=list(FUTURE_PHASES) or plan['max_generations']!=original['max_generations'] or
        plan['stage_resources']!={p:dict(options=resources(p)[0],concurrency=resources(p)[1]) for p in PHASES}):
        raise ValueError('Continuation addendum/resource identity changed')
    for name,digest in plan['source_sha256'].items():
        bound(ROOT/name,digest);bound(Path(plan['output'])/'sources'/name,digest)
    return plan


def original(plan):return base.checked_manifest(plan['base_manifest']['path'],plan['base_manifest']['sha256'])


def phase_tasks(plan,phase):
    if phase in ('train1','train2'):
        return [dict(arm=a,seed=s,cycle=int(phase[-1])) for a in base.ARMS for s in base.SEEDS]
    if phase=='gate1':return [dict(seed=s) for s in base.SEEDS]
    if phase.startswith('prepare'):return [dict(physical_key=a+'_t'+phase[-1]) for a in base.ARMS]
    if phase.startswith('export'):
        return [dict(physical_key=a+'_t'+phase[-1],seed=s) for a in base.ARMS for s in base.SEEDS]
    if phase.startswith('native'):
        return [dict(row=r) for r in original(plan)['physical_grt_requests'] if r['round']==int(phase[-1])]
    if phase in ('route_prepare','route'):return [dict(requested=72,indices=[0,1,2,3])]
    raise ValueError('Unknown fixed continuation phase')


def training_request(plan,arm,seed,cycle):
    old=original(plan);key=base.parent_key(arm,cycle);exp=base.experiment(old['output'],key)
    module=importlib.import_module(base.train_module(arm,cycle))
    from fql.graph_o2o_backend import load_requests
    _,tasks=load_requests(exp,seed,cycle-1)
    value=dict(version=module.VERSION,experiment=str(exp),config_sha256=sha(exp/'config.json'),
        declaration_sha256=sha(exp/'declaration.json'),seed=seed,tasks_sha256=sha(tasks),
        final_outcomes={},settings=old['arms'][arm],diagnostic_interval=old['diagnostic_interval'],
        output=str(Path(old['output'])/f'training/cycle_{cycle}'/arm/f'seed_{seed}'))
    if cycle==2:value['cycle']=2
    module.request_settings(value)
    path=Path(plan['output'])/'requests'/f'train{cycle}_{arm}_{seed}.json';immutable(path,value)
    return bound(path)


def parent_training(plan,arm,seed,cycle):
    request=training_request(plan,arm,seed,cycle)
    folder=Path(read(request['path'])['output'])
    return dict(request=request,checkpoint=bound(folder/'checkpoint.pt'))


def prepare_request(plan,key):
    old=original(plan);arm,stage=key.rsplit('_t',1);stage=int(stage)
    module=importlib.import_module(base.export_module(key))
    value=dict(version=module.VERSION,logical_round=stage,
        parent_training={str(s):parent_training(plan,arm,s,stage) for s in base.SEEDS},
        arm=base.physical_arm(key),run_prefix='soc1'+key.replace('_',''),
        streams=old['streams'][str(stage)],evaluation_purpose='development_validation',
        backend_storage='scratch',output=str(base.experiment(old['output'],key)))
    module.request_settings(value)
    path=Path(plan['output'])/'requests'/('export_'+key+'.json');immutable(path,value)
    return bound(path)


def checked_export(plan,key,seed):
    from fql.graph_o2o_backend import load_requests
    old=original(plan);exp=base.experiment(old['output'],key);module=importlib.import_module(base.export_module(key))
    cfg,decl,physical=module.settings(exp,sha(exp/'config.json'))
    rows=module.verify_export(exp,seed,cfg,decl,physical)
    saved,tasks=load_requests(exp,seed,int(key[-1]))
    if saved!=rows:raise ValueError('Verified export differs from native task inventory')
    return cfg,decl,physical,rows,tasks


def check_noise(plan,stage):
    old=original(plan);gate=read(Path(old['output'])/'gates/T0_actual_noise.json')
    common=gate['shared_evaluation_noise_sha256'];excluded=set(old['noise_history']['observed_noise_sha256'])|set(common)
    for row in gate['records']:excluded.update(r['noise_sha256'] for r in row['collect'])
    if stage==2:
        prior=read(Path(plan['output'])/'gates/T1_actual_noise.json')
        for row in prior['records']:excluded.update(row['collect'])
    paired={};records=[]
    for arm in base.ARMS:
        for seed in base.SEEDS:
            key=arm+'_t'+str(stage);_,_,_,rows,tasks=checked_export(plan,key,seed)
            evals=[r['noise_sha256'] for r in rows if r['role']=='evaluation']
            collects=[r['noise_sha256'] for r in rows if r['role']=='collect']
            if evals!=common or len(evals)!=8 or len(collects)!=(8 if stage==1 else 0):
                raise ValueError('Child changed shared eight evaluation arrays/role denominator')
            if collects:
                if len(set(collects))!=8 or set(collects)&excluded:
                    raise ValueError('New collect actual noise duplicates historical/previous/evaluation input')
                if seed in paired and paired[seed]!=collects:raise ValueError('Paired branch C1 noises differ')
                paired[seed]=collects
            records.append(dict(physical_key=key,seed=seed,tasks=bound(tasks),collect=collects,evaluation=evals))
    if paired and len(set().union(*(set(v) for v in paired.values())))!=24:
        raise ValueError('Cross-seed child collect actual noises overlap')
    path=Path(plan['output'])/'gates'/f'T{stage}_actual_noise.json'
    immutable(path,dict(status='pass',stage=stage,records=records,model_calls=0))
    return bound(path)


def gate1(plan,seed):
    from fql import qstudy_scalar_online_child_export as child, qstudy_scalar_online as core
    from fql.qstudy_scalar_online_bc import state_sha
    arms=('updated_q10','updated_q30')
    a,b=[child.checked_parent(parent_training(plan,arm,seed,1),seed) for arm in arms]
    for arm,result in zip(arms,(a,b)):
        payload=result['payload'];binding=result['identity']['continuation_binding'];audit=result['identity']['state_audit']
        if binding['settings']!=original(plan)['arms'][arm]:
            raise ValueError('T1 updated-Q branch settings changed')
        if (payload['step']!=600 or payload['parent_steps']!=1000 or
            sum(r['critic_updates'] for r in payload['history'])!=600 or
            any(r['teacher_updates'] for r in payload['history']) or
            audit['continued_actor_updates']!=600 or audit['continued_critic_updates']!=600):
            raise ValueError('T1 actual actor/critic/teacher update counts differ')
    core.equal_tree(a['source']['payload'],b['source']['payload'])
    core.equal_tree(a['identity']['continuation_binding']['replay'],b['identity']['continuation_binding']['replay'])
    pa,pb=a['payload'],b['payload']
    for key in ('critic','flow','rng','cuda_rng','generators'):core.equal_tree(pa[key],pb[key])
    for key in ('critic','flow'):core.equal_tree(pa['optimizers'][key],pb['optimizers'][key])
    if [r['qloss'] for r in pa['history']]!=[r['qloss'] for r in pb['history']]:
        raise ValueError('Shared-data alpha arms used different Q update trajectories')
    path=Path(plan['output'])/'gates'/f'T1_alpha_pair_seed_{seed}.json'
    value=dict(status='pass',seed=seed,parents=[r['identity'] for r in (a,b)],
        critic_state_sha256=state_sha(pa['critic']),critic_adam_sha256=state_sha(pa['optimizers']['critic']),
        q_generator_sha256=state_sha(pa['generators']['q']),actor_adam_steps=1600,critic_adam_steps=600,
        complete_critic_teacher_adam_rng_bitwise_equal=True,actor_and_actor_adam_allowed_to_differ=True,
        model_forward_calls=0,backend_calls=0)
    immutable(path,value);return bound(path)


def launch(module,args):
    """Forward batch preemption warnings to the unchanged checkpointing worker."""
    child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',module,*map(str,args)],cwd=ROOT)
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGUSR1,signal.SIGTERM)}
    def forward(sig,_):
        if child.poll() is None:child.send_signal(sig)
    for sig in previous:signal.signal(sig,forward)
    try:
        code=child.wait()
        if code:raise SystemExit(code)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def task_run(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Bounded Slurm allocation required')
    plan=checked_manifest(args.manifest,args.manifest_sha256);old=original(plan);phase=args.phase
    index=args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    tasks=phase_tasks(plan,phase)
    if not 0<=index<len(tasks):raise IndexError('Outside fixed phase task set')
    task=tasks[index];receipt=Path(plan['output'])/'receipts'/phase/(str(index)+'.json')
    if receipt.exists():dispatch['checked_receipt'](plan,phase,index);return
    if phase.startswith('train'):
        request=training_request(plan,task['arm'],task['seed'],task['cycle'])
        launch(base.train_module(task['arm'],task['cycle']),['--request',request['path'],'--request-sha256',request['sha256']])
        folder=Path(read(request['path'])['output'])
        result=[bound(folder/name) for name in ('checkpoint.pt','completed.json','binding.json','learning_curve.json')]+[request]
    elif phase=='gate1':result=[gate1(plan,task['seed'])]
    elif phase.startswith('prepare'):
        request=prepare_request(plan,task['physical_key'])
        launch(base.export_module(task['physical_key']),['prepare','--request',request['path'],
            '--request-sha256',request['sha256'],'--output',base.experiment(old['output'],task['physical_key'])])
        exp=base.experiment(old['output'],task['physical_key'])
        result=[bound(exp/'config.json'),bound(exp/'declaration.json'),request]
    elif phase.startswith('export'):
        exp=base.experiment(old['output'],task['physical_key'])
        launch(base.export_module(task['physical_key']),['export','--experiment',exp,
            '--config-sha256',sha(exp/'config.json'),'--seed',task['seed']])
        *_,path=checked_export(plan,task['physical_key'],task['seed'])
        result=[bound(path),bound(path.with_name('completed.json'))]
    elif phase.startswith('native'):
        row=task['row'];*_,rows,path=checked_export(plan,row['physical_key'],row['seed'])
        positions=[i for i,r in enumerate(rows) if (r['role'],r['index'])==(row['role'],row['index'])]
        if len(positions)!=1:raise ValueError('Missing/duplicate actual native role/index')
        launch('fql.qstudy_backend',['--experiment',row['experiment'],'--seed',row['seed'],
            '--round',row['round'],'--index',positions[0],'--storage','scratch'])
        _,_,_,sample,_=base.inspect_native(old,row);result=[bound(sample/'outcome.json')]
    else:
        raise ValueError('Route stages are declared; implementation requires separate source inspector review')
    immutable(receipt,dict(version=VERSION,manifest=bound(args.manifest,args.manifest_sha256),
        phase=phase,index=index,task=task,results=result,slurm_job_id=os.environ['SLURM_JOB_ID']))


def stage_submit(args):
    plan=checked_manifest(args.manifest,args.manifest_sha256);old=original(plan)
    if args.phase in ('route_prepare','route'):
        raise ValueError('DRT source adapter is prospectively declared, not implemented by this version')
    for phase in base.PHASES:
        if base.missing_indices(old,phase):raise ValueError('Original stage0 has unfinished '+phase)
        state,_=base.stage_state(old,phase)
        states,_=base.scheduler_states(state['attempts'][-1]['job'])
        if any(s in base.LIVE or s=='ACCOUNTING_PENDING' for s in states):
            raise ValueError('Original stage0 still live/accounting pending')
    if args.phase.startswith('native'):check_noise(plan,int(args.phase[-1]))
    dispatch['stage_submit'](args)


# Only clone the already-reviewed metadata bookkeeping functions into a private
# namespace. The original helper's globals, source bytes and manifest stay intact.
dispatch=dict(vars(base));dispatch.update(VERSION=VERSION,PHASES=PHASES,FUTURE_PHASES=FUTURE_PHASES,
    HELPERS=HELPERS,checked_manifest=checked_manifest,phase_tasks=phase_tasks,resources=resources)
for name in ('checked_receipt','stage_state','missing_indices','stage_submit','status'):
    function=getattr(base,name)
    dispatch[name]=FunctionType(function.__code__,dispatch,name,function.__defaults__,function.__closure__)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('manifest');p.add_argument('--base-manifest',required=True);p.add_argument('--base-manifest-sha256',required=True)
    for name in ('submit','status','task'):
        p=sub.add_parser(name);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
        if name in ('submit','task'):p.add_argument('--phase',choices=PHASES,required=True)
        if name=='task':p.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command=='manifest':create(args)
    elif args.command=='task':task_run(args)
    elif args.command=='submit':stage_submit(args)
    else:dispatch['status'](args)


if __name__=='__main__':main()
