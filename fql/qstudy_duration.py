"""Continue an immutable original BC/FQL step1000 prefix to actor step2000.

No original file changes. Only logical arm/run-prefix and offline_actor_steps
change in the persisted config. The unchanged round0 training bytecode restores
all optimizer/RNG/generator state and keeps flow/Q frozen. Export is evaluation
only: a private role-enumeration overlay prevents unused collection requests.
This is development noise4600000, never fresh confirmation or online learning.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
from types import FunctionType

import torch
import numpy as np

from . import qstudy_actor as frozen
from . import qstudy_branch as prefix
from .graph_o2o_archive import atomic_json, sha
from .qstudy_screen import atomic_torch

ROOT=frozen.ROOT
VERSION='actor_duration_prefix_v1'
FROM_STEP=1000
TO_STEP=2000
SEEDS=(0,1,2)
TRAIN_SOURCES=prefix.TRAIN_SOURCES
SOURCE_NAMES=tuple(dict.fromkeys((*TRAIN_SOURCES,'fql/qstudy_branch.py',
    'fql/qstudy_duration.py','fql/qstudy_duration.slurm','fql/qstudy_backend.py',
    'fql/graph_o2o_backend.py','fql/qstudy_data.py','fql/physical_ridge.py',
    'fql/graph_o2o_geometry.py','backend/scripts/export_macro_def.py')))
EXPORT_OVERLAY=dict(online_rounds=0)
COORDINATE_CONTRACT=dict(training_prediction='unclipped actor(noise, clip=False)',
    exported_raw_action='clipped actor(noise, clip=True), before external legalizer')
NAME=re.compile(r'[A-Za-z][A-Za-z0-9_]{0,31}\Z')


def read(path):return prefix.read(path)


def bind(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path))


def slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Checkpoint import/validation requires bounded Slurm')


def configurations(source,target):
    if (source.get('offline_actor_steps')!=FROM_STEP or target.get('offline_actor_steps')!=TO_STEP or
        source.get('seeds')!=list(SEEDS) or source.get('critic_variant')!='legacy' or
        source.get('weights')!=[1,0,0,0,0] or source.get('q_weight') not in (0.,1.) or
        source.get('alpha')!=10. or source.get('online_flow_update') is not False or
        source.get('offline_critic_update') is not False or source.get('flow_steps')!=10 or
        source.get('evaluation_samples')!=8 or source.get('evaluation_noise_seed')!=4600000):
        raise ValueError('Duration study requires original BC/FQL offline1000 and fixed teacher/Q/noises')
    for key in set(source)|set(target):
        if key not in {'arm','run_prefix','offline_actor_steps'} and (
            key not in source or key not in target or source[key]!=target[key]):
            raise ValueError('Unsupported duration configuration difference: '+key)
    for name in ('arm','run_prefix'):
        if not isinstance(target.get(name),str) or not NAME.fullmatch(target[name]) or target[name]==source.get(name):
            raise ValueError('Duration requires a distinct safe namespace: '+name)


def expected_binding(source,cfg,seed):
    return dict(config_sha256=sha(source/'config.json'),dataset_sha256=cfg['dataset_sha256'],
        common_sha256=cfg['common_checkpoint_sha256'][str(seed)],
        initial_critic_sha256=cfg['critic_checkpoint_sha256'][str(seed)],round=0,seed=seed,replay=[],parent=None,
        source_sha256={name:sha(ROOT/name) for name in TRAIN_SOURCES})


def source_noises(source,cfg,seed,checkpoint):
    """Bind saved development arrays, without model inference or native tools."""
    from .graph_o2o_backend import load_requests
    from .qstudy_backend import checked_request
    rows,tasks=load_requests(source,seed,0)
    marker=read(tasks.with_name('completed.json'))
    if marker.get('status')!='completed' or marker.get('evaluation_never_replayed') is not True:
        raise ValueError('Original evaluation export is incomplete or replay eligible')
    evaluation=[r for r in rows if r.get('role')=='evaluation']
    if len(evaluation)!=8 or [r['index'] for r in evaluation]!=list(range(8)):
        raise ValueError('Original source requires eight ordered development evaluation noises')
    files=[tasks,tasks.with_name('completed.json')];hashes=[];checkpoint_sha=sha(checkpoint)
    # checked_request only needs these identity fields; no physical context is
    # constructed here, and it cannot run EDA or infer/repair missing geometry.
    binding=dict(config_sha256=sha(source/'config.json'),dataset_sha256=cfg['dataset_sha256'])
    for row in evaluation:
        if (row.get('source_checkpoint')!=str(checkpoint) or row.get('checkpoint_sha256')!=checkpoint_sha or
            row.get('noise_seed')!=4600000 or row.get('training_feedback_eligible') is not False or
            row.get('arm')!=cfg['arm'] or row.get('evaluation_purpose')!=cfg.get('evaluation_purpose','development_validation')):
            raise ValueError('Original source evaluation identity/noise/eligibility changed')
        sample=checked_request(row,argparse.Namespace(seed=seed,round=0),cfg,binding)
        with np.load(sample/'actions.npz',allow_pickle=False) as arrays:
            noise=arrays['noise']
            if noise.dtype!=np.float32 or noise.ndim!=2 or noise.shape[1]!=2 or ((noise < -1)|(noise > 1)).any():
                raise ValueError('Original development noise is not a normalized float32 uniform sample')
        hashes.append(row['noise_sha256'])
        files.extend((sample/'request.json',sample/'actions.npz'))
        if row.get('def_path'):
            files.extend((Path(row['def_path']),Path(row['def_path']).with_suffix('.audit.json')))
    return hashes,files


def checked_source(source):
    """Read original r0 completion/init/source chains, never request native EDA."""
    source=Path(source).resolve();cfg=read(source/'config.json')
    test=dict(cfg,arm='DurationCheck',run_prefix='DurationCheck',offline_actor_steps=TO_STEP)
    configurations(cfg,test)
    files=[source/'config.json',Path(cfg['dataset_path'])]
    if sha(files[-1])!=cfg['dataset_sha256']:raise ValueError('Source TRAIN dataset changed')
    cache=Path(frozen.RUNTIME)/'benchmark_cache'/(cfg['design']+'.pt')
    if sha(cache)!=cfg['cache_sha256']:raise ValueError('Source graph cache changed')
    files.append(cache)
    seeds=[]
    for seed in SEEDS:
        folder=source/f'seed_{seed}/round_0';checkpoint=folder/'checkpoint.pt'
        done,binding=read(folder/'completed.json'),read(folder/'binding.json')
        expected=expected_binding(source,cfg,seed)
        if (done.get('status')!='completed' or done.get('binding')!=expected or binding!=expected or
            done.get('steps')!=FROM_STEP or done.get('checkpoint_sha256')!=sha(checkpoint) or
            done.get('offline_critic_frozen') is not True or done.get('online_flow_update') is not False):
            raise ValueError('Source step1000 completion/binding/initialization changed')
        for name in ('common','critic'):
            path=Path(cfg[name+'_checkpoints'][str(seed)])
            if sha(path)!=cfg[name+'_checkpoint_sha256'][str(seed)]:
                raise ValueError('Source frozen initialization changed: '+name)
            files.append(path)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['binding']!=expected or saved['step']!=FROM_STEP or 'prefix_import' in saved:
            raise ValueError('Source is not an original completed round0 prefix')
        fingerprints=prefix.state_fingerprints(saved)
        noises,noise_files=source_noises(source,cfg,seed,checkpoint)
        if seeds and noises!=seeds[0]['evaluation_noise_sha256']:
            raise ValueError('Original three seeds did not export the same actual eight development noises')
        seeds.append(dict(seed=seed,checkpoint=bind(checkpoint),binding=expected,payload=fingerprints,
            evaluation_noise_sha256=noises))
        del saved
        files.extend(folder/name for name in ('checkpoint.pt','binding.json','completed.json','learning_curve.jsonl'))
        files.extend(noise_files)
    return cfg,seeds,{str(p.resolve()):sha(p) for p in files}


def prepare(args):
    slurm();source=Path(args.source).resolve();output=Path(args.output).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Duration output must be separate from the original experiment')
    if sha(source/'config.json')!=args.source_sha256:raise ValueError('Source config SHA changed')
    cfg,seeds,inventory=checked_source(source)
    target=dict(cfg,arm=args.arm,run_prefix=args.run_prefix,offline_actor_steps=TO_STEP)
    configurations(cfg,target)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'duration_prepare.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (output/'config.json').exists():
            prefix.atomic_json(output/'config.json',target)
        if read(output/'config.json')!=target:raise ValueError('Immutable target duration config changed')
        declared=dict(version=VERSION,source=bind(source/'config.json'),target=bind(output/'config.json'),
            source_inventory=inventory,source_sha256={n:sha(ROOT/n) for n in SOURCE_NAMES},
            from_step=FROM_STEP,to_step=TO_STEP,seeds=list(SEEDS),export_role_overlay=EXPORT_OVERLAY,
            evaluation_noise_seed=4600000,evaluation_samples=8,online_updates=0,
            evaluation_noise_sha256=seeds[0]['evaluation_noise_sha256'],
            coordinate_contract=COORDINATE_CONTRACT,
            new_training_updates_per_seed=TO_STEP-FROM_STEP,new_backend_calls=0)
        markers=[]
        for spec in seeds:
            seed=spec['seed'];final=output/f'seed_{seed}';marker=final/'round_0/prefix_import.json'
            binding=dict(spec['binding'],config_sha256=sha(output/'config.json'))
            evidence=dict(version=VERSION,source_checkpoint=spec['checkpoint'],source_binding=spec['binding'],
                target_binding=binding,payload=spec['payload'],imported_step=FROM_STEP,
                original_completed_marker_not_copied=True,new_training_updates=0,new_backend_calls=0)
            if final.exists():
                old=read(marker)
                if any(old.get(k)!=v for k,v in evidence.items()):
                    raise ValueError('Previously imported duration prefix changed')
            else:
                staging=output/f'.duration_seed_{seed}.staging'
                if staging.exists():shutil.rmtree(staging)
                folder=staging/'round_0';folder.mkdir(parents=True)
                saved=torch.load(spec['checkpoint']['path'],map_location='cpu',weights_only=False)
                if prefix.state_fingerprints(saved)!=spec['payload']:
                    raise ValueError('Original duration prefix changed during import')
                saved['binding']=binding;atomic_torch(folder/'resume.pt',saved)
                rebuilt=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
                if prefix.state_fingerprints(rebuilt)!=spec['payload'] or rebuilt['binding']!=binding:
                    raise ValueError('Duration import changed model/optimizer/RNG/generator/step')
                del saved,rebuilt
                atomic_json(folder/'binding.json',binding)
                atomic_json(folder/'prefix_import.json',dict(evidence,initial_resume_sha256=sha(folder/'resume.pt')))
                staging.rename(final)
            markers.append(bind(marker))
        declared['prefix_imports']=markers
        gate=output/'duration_import.json'
        if gate.exists():
            if read(gate)!=declared:raise ValueError('Immutable duration import gate changed')
        else:atomic_json(gate,declared)
        # Detect a changed source during the CPU import transaction.
        prefix.checked_files(inventory)
    return declared


def checked_import(experiment,seed):
    slurm();exp=Path(experiment).resolve();gate=read(exp/'duration_import.json')
    if (gate['version']!=VERSION or gate['target']!=bind(exp/'config.json') or
        gate['from_step']!=FROM_STEP or gate['to_step']!=TO_STEP or gate['seeds']!=list(SEEDS) or
        gate['export_role_overlay']!=EXPORT_OVERLAY or gate['evaluation_noise_seed']!=4600000 or
        gate['evaluation_samples']!=8 or gate['online_updates']!=0 or seed not in SEEDS or
        gate.get('coordinate_contract')!=COORDINATE_CONTRACT or
        len(gate.get('evaluation_noise_sha256',[]))!=8 or
        gate['new_training_updates_per_seed']!=1000 or gate['new_backend_calls']!=0 or
        set(gate['source_sha256'])!=set(SOURCE_NAMES)):
        raise ValueError('Duration import/runtime contract changed')
    for name,digest in gate['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen duration source changed: '+name)
    prefix.checked_files(gate['source_inventory'])
    source=Path(gate['source']['path']);cfg=read(exp/'config.json')
    if bind(source)!=gate['source']:raise ValueError('Duration source config changed')
    configurations(read(source),cfg)
    expected=[exp/f'seed_{s}/round_0/prefix_import.json' for s in SEEDS]
    if gate['prefix_imports']!=[bind(p) for p in expected]:raise ValueError('Duration prefix import inventory changed')
    folder=exp/f'seed_{seed}/round_0';imported=read(folder/'prefix_import.json')
    binding=dict(imported['source_binding'],config_sha256=sha(exp/'config.json'))
    if read(folder/'binding.json')!=binding or imported['target_binding']!=binding:
        raise ValueError('Duration resumed training binding changed')
    checkpoint=folder/'resume.pt' if (folder/'resume.pt').exists() else folder/'checkpoint.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['binding']!=binding or not FROM_STEP<=saved['step']<=TO_STEP:
        raise ValueError('Duration current checkpoint binding/step differs')
    current=prefix.state_fingerprints(saved)
    if any(current[k]!=imported['payload'][k] for k in ('flow','critic')):
        raise ValueError('Duration experiment changed the frozen teacher or Q')
    if saved['step']==FROM_STEP and current!=imported['payload']:
        raise ValueError('Imported step1000 optimizer/RNG/model state differs')
    if (folder/'completed.json').exists():
        done=read(folder/'completed.json')
        if (done['status']!='completed' or done['steps']!=TO_STEP or done['binding']!=binding or
            done['checkpoint_sha256']!=sha(folder/'checkpoint.pt') or saved['step']!=TO_STEP):
            raise ValueError('Duration final completion differs')
    del saved
    return cfg,gate


def isolated_runtime(*,evaluation_only=False):
    namespace=dict(vars(frozen));namespace['STOP']=False
    if evaluation_only:
        original_setup=namespace['setup']
        def export_setup(experiment,seed):
            values=original_setup(experiment,seed)
            return (dict(values[0],**EXPORT_OVERLAY),*values[1:])
        namespace['setup']=export_setup
    for name in ('stop','train','export'):
        original=getattr(frozen,name)
        namespace[name]=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    return namespace


def verify_export(experiment,seed,cfg,gate):
    """Only request/hash checks; this function cannot execute native evaluation."""
    from .graph_o2o_backend import load_requests
    from .qstudy_backend import context,checked_request
    exp=Path(experiment);rows,tasks=load_requests(exp,seed,0);native_cfg,_,physical=context(exp)
    if len(rows)!=8 or [r['index'] for r in rows]!=list(range(8)):
        raise ValueError('Duration evaluation must preserve exactly8 ordered requests')
    completed=read(tasks.with_name('completed.json'))
    if completed.get('status')!='completed' or completed.get('evaluation_never_replayed') is not True:
        raise ValueError('Duration export completion or evaluation eligibility changed')
    for row in rows:
        if (row['role']!='evaluation' or row['training_feedback_eligible'] or
            row['noise_seed']!=4600000 or row['evaluation_purpose']!=cfg.get('evaluation_purpose','development_validation') or
            row['noise_sha256']!=gate['evaluation_noise_sha256'][row['index']] or row['arm']!=cfg['arm'] or
            row['source_checkpoint']!=str(exp/f'seed_{seed}/round_0/checkpoint.pt') or
            row['checkpoint_sha256']!=sha(exp/f'seed_{seed}/round_0/checkpoint.pt')):
            raise ValueError('Duration export cannot collect/replay or use a fresh noise stream')
        checked_request(row,argparse.Namespace(seed=seed,round=0),native_cfg,physical)
    return dict(tasks=bind(tasks),requests=8,role='evaluation',training_feedback_eligible=False)


def run(args):
    cfg,gate=checked_import(args.experiment,args.seed)
    args.experiment=Path(args.experiment).resolve();args.round=0
    stages=('train','export') if args.stage=='all' else (args.stage,)
    for stage in stages:
        if stage=='export' and not (args.experiment/f'seed_{args.seed}/round_0/completed.json').exists():
            raise ValueError('Duration evaluation requires completed step2000')
        isolated_runtime(evaluation_only=stage=='export')[stage](args)
        checked_import(args.experiment,args.seed)
        if stage=='export':verify_export(args.experiment,args.seed,cfg,gate)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    for name in ('source','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-sha256','arm','run-prefix'):p.add_argument('--'+name,required=True)
    p=commands.add_parser('run');p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--stage',choices=('train','export','all'),default='train')
    args=parser.parse_args();result=globals()[args.command](args)
    if result is not None:print(result,flush=True)


if __name__=='__main__':main()
