"""Evaluation-only export of genuine completed second BC continuations.

Logical round two refers to the real T2 actor/critic checkpoint. Root, T1,
C0+C1, optimizer/RNG and fixed scalar-schema proofs stay intact. The original BC actor prefix and separately attached frozen scalar refit stay
explicit. No FQL source, training, selection, third collection or physical call is supported.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np
import torch

from . import qstudy_scalar_online_bc_second_train as training
from . import qstudy_scalar_online_bc_export as original
from . import qstudy_scalar_online as core
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import RUNTIME, legalize, load_problem, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json
from .graph_o2o_backend import load_requests
from .graph_o2o_models import GraphPolicy
from .qstudy_backend import checked_request, context
from .qstudy_data import action_sha, bind, sha

ROOT=Path(__file__).resolve().parents[1]
VERSION='qstudy_scalar_online_bc_second_export_v1'
ROUND=2
SEEDS=(0,1,2)
SOURCE_NAMES=tuple(dict.fromkeys((*training.SOURCE_NAMES,
    'fql/qstudy_scalar_online_bc_second_export.py','fql/qstudy_scalar_online_bc_second_export.slurm')))
LEGALIZER=dict(w_reg=.1,w_dist=1.,grid_res=128,padding=.001)


def read(path):return json.loads(Path(path).read_text())


def request_settings(value):
    required={'version','logical_round','parent_training','arm','run_prefix','streams',
              'evaluation_purpose','backend_storage','output'}
    if (set(value)!=required or value['version']!=VERSION or type(value['logical_round']) is not int or
        value['logical_round']!=ROUND or not isinstance(value['arm'],str) or
        not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',value['arm']) or
        not isinstance(value['run_prefix'],str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9]{0,23}',value['run_prefix']) or
        value['evaluation_purpose']!='development_validation' or value['backend_storage']!='scratch'):
        raise ValueError('Explicit second-BC evaluation-only declaration required')
    output=Path(value['output'])
    if (not output.is_absolute() or str(output.resolve())!=str(output) or output==ROOT or
        output in ROOT.parents or ROOT in output.parents):
        raise ValueError('Canonical separate second-export output required')
    if set(value['parent_training'])!={str(s) for s in SEEDS} or set(value['streams'])!={str(s) for s in SEEDS}:
        raise ValueError('All three independently trained T2 parents and evaluation streams required')
    for seed in SEEDS:
        parent=value['parent_training'][str(seed)]
        if set(parent)!={'request','checkpoint'}:raise ValueError('Externally pinned T2 request and actual checkpoint required')
        for item in parent.values():
            if (set(item)!={'path','sha256'} or not Path(item['path']).is_absolute() or
                str(Path(item['path']).resolve())!=item['path'] or not re.fullmatch('[0-9a-f]{64}',item['sha256'])):
                raise ValueError('Canonical T2 parent path and SHA required')
        streams=value['streams'][str(seed)]
        if set(streams)!={'evaluation'}:raise ValueError('T2 export never creates collection/training data')
        stream=streams['evaluation']
        if (set(stream)!={'seed','count'} or type(stream['seed']) is not int or not 0<=stream['seed']<2**63 or
            type(stream['count']) is not int or stream['count']<1):
            raise ValueError('Explicit evaluation seed/count required')
    return value


def checked_files(value):
    """Rehash only explicitly captured bindings; no filesystem discovery."""
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            if bind(value['path'],value['sha256'])!={k:value[k] for k in ('path','sha256')}:
                raise ValueError('Captured file binding is no longer canonical')
        for child in value.values():checked_files(child)
    elif isinstance(value,(list,tuple)):
        for child in value:checked_files(child)


def parent_files(parent):
    request_binding=bind(parent['request']['path'],parent['request']['sha256'])
    request=training.request_settings(read(request_binding['path']))
    folder=Path(request['output'])
    if bind(folder/'checkpoint.pt',parent['checkpoint']['sha256'])!=parent['checkpoint']:
        raise ValueError('Source must be the genuine second training checkpoint path')
    return request,{name:bind(folder/name) for name in ('checkpoint.pt','completed.json','binding.json','learning_curve.json')}


def checked_parent(parent,seed):
    request,files=parent_files(parent)
    if request['seed']!=seed:raise ValueError('Second training seed differs from evaluation declaration')
    cfg,decl,source,replay,binding=training.load_inputs(request)
    folder=Path(request['output'])
    runner=dict(version=training.VERSION,cycle=2,request=parent['request'],config=bind(Path(request['experiment'])/'config.json'),
        declaration=bind(Path(request['experiment'])/'declaration.json'),continuation=binding,
        source_sha256={name:sha(ROOT/name) for name in training.SOURCE_NAMES},
        diagnostic_interval=request['diagnostic_interval'],backend_calls=0)
    if read(folder/'binding.json')!=runner:raise ValueError('T2 runner binding differs from reconstructed admission')
    for name,digest in runner['source_sha256'].items():bind(folder/'sources'/name,digest)
    payload=training.checked_resume(folder,runner,binding,source)
    training.checked_completion(folder,payload,runner,binding)
    counts=training.audit_state(payload,source,binding)
    if (source['binding'].get('provider')!=training.child.VERSION or
        source['root']['binding'].get('version')!=training.provider.VERSION or
        binding['settings']['q_weight']!=0. or binding['settings']['update_critic'] is not False or
        counts['critic_adam_steps']!=0 or counts['teacher_updates']!=0):
        raise ValueError('Only the genuine BC second-cycle actor with a frozen separate scalar critic is supported')
    if payload['history']!=read(folder/'learning_curve.json'):
        raise ValueError('Actual T2 checkpoint and complete saved history disagree')
    # audit_state has checked scalar weights/Adam and unused streams against the original BC dual source.
    source['refit']['model'].load_state_dict(payload['critic'],strict=True)
    collect_hashes=sorted({record['noise_sha256'] for record in replay['records']})
    old_identity=source['binding']['first_continuation']
    old_cfg=read(old_identity['parent_export_config']['path'])
    previous_collect_seeds=sorted({streams['collect']['seed'] for pcfg in (cfg,old_cfg) for streams in pcfg['streams'].values()})
    identity=dict(training_request=parent['request'],checkpoint=files['checkpoint.pt'],completed=files['completed.json'],
        runner_binding=files['binding.json'],learning_curve=files['learning_curve.json'],
        parent_export_config=runner['config'],parent_export_declaration=runner['declaration'],
        root_source=source['root']['binding'],first_continuation=source['binding'],continuation_binding=binding,
        cumulative_steps=counts,previous_collect_noise_sha256=collect_hashes,
        previous_evaluation_noise_sha256=replay['binding']['excluded_evaluation_noise_sha256'],
        previous_collect_seeds=previous_collect_seeds,state_cycle=2,
        state_identity='genuine completed BC T2 actor; unchanged separate scalar refit, teacher and diagnostic optimizer',
        bc_actor_prefix=source['binding']['bc_actor_prefix'],scalar_source=source['binding']['scalar_source'],
        scalar_critic_refit=source['binding']['scalar_critic_refit'],q_weight=0.,q_units=core.Q_UNITS)
    checked_files(identity)
    return dict(payload=payload,source=source,cfg=cfg,declaration=decl,identity=identity)


def config_value(request,request_binding,base,parents,files):
    cfg={k:base[k] for k in ('design','dataset_path','dataset_sha256','cache_sha256','physical_failure_reward','reward_scale')}
    cfg.update(scalar_online_bc_second_export_schema=VERSION,arm=request['arm'],run_prefix=request['run_prefix'],
        seeds=list(SEEDS),logical_round=ROUND,streams=request['streams'],backend_threads=8,backend_storage='scratch',
        terminal_mask=0,evaluation_purpose=request['evaluation_purpose'],scalar_schema=base['scalar_schema'],
        parent_training=request['parent_training'],parent_files=files,verified_seconds=parents,
        prior_collect_noise_sha256=sorted({h for p in parents.values() for h in p['previous_collect_noise_sha256']}),
        prospective_request=request_binding,q_units=core.Q_UNITS,checkpoint_state_key='actor',critic_state_key='critic',
        scope='Completed second genuine BC cycle evaluation only; separate frozen scalar critic; no FQL, training, third collection or selection',
        raw_action_semantics='clipped actual T2 actor output; training distillation used unclipped output',
        q_weight=0.,actor_source_kind='genuine_bc_second_continuation',
        critic_source_kind='separately_bound_frozen_scalar_refit')
    return cfg


def declaration_value(cfg,request_binding,physical,sources,output):
    return dict(version=VERSION,config=bind(output/'config.json'),request=request_binding,
        parents=cfg['verified_seconds'],parent_files=cfg['parent_files'],physical_context=physical,
        schema=cfg['scalar_schema'],streams=cfg['streams'],logical_round=ROUND,source_sha256=sources,
        checkpoint_mutations=0,actor_updates=0,critic_updates=0,teacher_updates=0,backend_calls=0,
        q_units=core.Q_UNITS,q_selection=False,evaluation_never_replayed=True,roles=['evaluation'],legalizer=LEGALIZER,
        q_weight=0.,actor_source_kind='genuine_bc_second_continuation',critic_source_kind='separately_bound_frozen_scalar_refit')


def prepare(request_path,request_sha256,output):
    core.slurm();request_binding=bind(request_path,request_sha256);request=request_settings(read(request_path))
    output=Path(output).resolve()
    if str(output)!=request['output']:raise ValueError('Output differs from externally pinned request')
    parents={};files={};base=None;sources={};recipe=None
    for seed in SEEDS:
        result=checked_parent(request['parent_training'][str(seed)],seed);cfg=result['cfg'];identity=result['identity']
        if base is not None and cfg!=base:raise ValueError('All T2 seeds must share the same fixed scalar source config')
        this_recipe=dict(settings=identity['continuation_binding']['settings'],cumulative_steps=identity['cumulative_steps'],
            first_settings=identity['first_continuation']['first_continuation']['continuation_binding']['settings'])
        if recipe is not None and this_recipe!=recipe:raise ValueError('T2 seeds changed matched update budgets or Q branch')
        recipe=this_recipe;base=cfg;parents[str(seed)]=identity
        if request['streams'][str(seed)]['evaluation']['seed'] in identity['previous_collect_seeds']:
            raise ValueError('A training collect stream cannot become held-out T2 evaluation')
        train_request,files[str(seed)]=parent_files(request['parent_training'][str(seed)])
        checked_files(identity)
        for pcfg_path in (identity['parent_export_config']['path'],identity['root_source']['config']['path'],
                          identity['scalar_source']['config']['path'],
                          identity['first_continuation']['first_continuation']['parent_export_config']['path']):
            old=Path(pcfg_path).parent
            if output==old or output in old.parents or old in output.parents:
                raise ValueError('T2 export overlaps a real source/physical namespace')
        for old in (Path(train_request['output']),Path(identity['first_continuation']['checkpoint']['path']).parent):
            if output==old or output in old.parents or old in output.parents:raise ValueError('T2 export overlaps a training prefix')
        for name,digest in result['declaration']['source_sha256'].items():
            if name in sources and sources[name]!=digest:raise ValueError('Three source closures disagree')
            bind(ROOT/name,digest);sources[name]=digest
    for name in SOURCE_NAMES:
        digest=sha(ROOT/name)
        if name in sources and sources[name]!=digest:raise ValueError('Captured training source changed')
        sources[name]=digest
    cfg=config_value(request,request_binding,base,parents,files)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (output/'declaration.json').exists() and any(p.name not in ('prepare.lock','config.json','sources') for p in output.iterdir()):
            raise ValueError('Second-export output contains unrelated/unsealed artifacts')
        original.immutable(output/'config.json',cfg)
        _,_,physical=context(output);_,_,previous=context(Path(parents['0']['parent_export_config']['path']).parent)
        if {k:v for k,v in physical.items() if k!='config_sha256'}!={k:v for k,v in previous.items() if k!='config_sha256'}:
            raise ValueError('T2 changed its physical evaluation context')
        for name,digest in sources.items():
            path=output/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():shutil.copyfile(ROOT/name,path)
            bind(path,digest)
        original.immutable(output/'declaration.json',declaration_value(cfg,request_binding,physical,sources,output))
    print(json.dumps(dict(status='prepared_T2_evaluation_only',experiment=str(output),
        config_sha256=sha(output/'config.json'),declaration_sha256=sha(output/'declaration.json'))),flush=True)


def settings(experiment,expected_config_sha256):
    experiment=Path(experiment).resolve();bind(experiment/'config.json',expected_config_sha256)
    cfg=read(experiment/'config.json');decl=read(experiment/'declaration.json');rb=cfg['prospective_request']
    request=request_settings(read(bind(rb['path'],rb['sha256'])['path']))
    base=read(cfg['verified_seconds']['0']['parent_export_config']['path'])
    if cfg!=config_value(request,rb,base,cfg['verified_seconds'],cfg['parent_files']) or str(experiment)!=request['output']:
        raise ValueError('T2 config differs from its sealed request/source')
    _,_,physical=context(experiment)
    if decl!=declaration_value(cfg,rb,physical,decl['source_sha256'],experiment):
        raise ValueError('T2 declaration changed source/schema/roles/context')
    core.validate_schema(cfg['scalar_schema'])
    for seed in SEEDS:
        _,files=parent_files(request['parent_training'][str(seed)])
        if files!=cfg['parent_files'][str(seed)]:raise ValueError('Original T2 artifacts changed')
        checked_files(cfg['verified_seconds'][str(seed)])
    for name,digest in decl['source_sha256'].items():bind(ROOT/name,digest);bind(experiment/'sources'/name,digest)
    if any(name not in decl['source_sha256'] for name in SOURCE_NAMES):raise ValueError('T2 source closure incomplete')
    return cfg,decl,physical


def check_row(row,cfg,decl,physical,experiment,seed,index,noise=None):
    identity=cfg['verified_seconds'][str(seed)];parent=cfg['parent_training'][str(seed)]['checkpoint']
    expected=dict(arm=cfg['arm'],graph=cfg['design'],seed=seed,round=ROUND,role='evaluation',index=index,
        source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],source_state_key='actor',critic_state_key='critic',
        source_config_sha256=identity['parent_export_config']['sha256'],config_sha256=physical['config_sha256'],
        dataset_sha256=cfg['dataset_sha256'],continuation_identity_sha256=core.json_sha(identity),
        training_request_sha256=identity['training_request']['sha256'],export_declaration_sha256=sha(experiment/'declaration.json'),
        scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],q_units=core.Q_UNITS,
        noise_seed=cfg['streams'][str(seed)]['evaluation']['seed'],terminal_mask=0,training_feedback_eligible=False,
        offline_train_eligible=False,evaluation_purpose=cfg['evaluation_purpose'],stage='global_route_only',
        run_id=f"go2o_qsbc2_{cfg['run_prefix']}_s{seed}_r2_evaluation_{index}",objective_weights=cfg['scalar_schema']['utility_spec']['weights'],
        q_weight=0.,actor_source_kind='genuine_bc_second_continuation',
        critic_source_kind='separately_bound_frozen_scalar_refit',
        original_bc_checkpoint=identity['bc_actor_prefix']['checkpoint'],scalar_critic_refit=identity['scalar_critic_refit'])
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('T2 request changed true source/round/evaluation identity')
    q=row.get('predicted_q')
    if not isinstance(q,list) or len(q)!=1 or type(q[0]) not in (int,float) or not math.isfinite(q[0]):
        raise ValueError('Finite diagnostic scalar Q required; selection remains disabled')
    sample=checked_request(row,argparse.Namespace(seed=seed,round=ROUND),cfg,physical)
    if sample.resolve()!=(experiment/f'seed_{seed}'/'round_2'/'requests'/cfg['arm']/'evaluation'/str(index)).resolve():
        raise ValueError('T2 evaluation escaped its declared namespace')
    if noise is not None and row['noise_sha256']!=action_sha(noise):raise ValueError('Resumed T2 export changed actual noise')
    if row['noise_sha256'] in cfg['prior_collect_noise_sha256']:raise ValueError('T2 evaluation reuses native training collect noise')
    return sample


def completion(cfg,experiment,seed,rows,tasks):
    return dict(status='completed',version=VERSION,count=len(rows),tasks_sha256=sha(tasks),
        config_sha256=sha(experiment/'config.json'),export_declaration_sha256=sha(experiment/'declaration.json'),
        source_checkpoint=cfg['parent_training'][str(seed)]['checkpoint'],logical_round=ROUND,
        continuation_identity_sha256=core.json_sha(cfg['verified_seconds'][str(seed)]),
        role_counts={'evaluation':cfg['streams'][str(seed)]['evaluation']['count']},evaluation_never_replayed=True,
        q_units=core.Q_UNITS,q_selection=False,checkpoint_mutations=0,actor_updates=0,critic_updates=0,teacher_updates=0,backend_calls=0,
        q_weight=0.,actor_source_kind='genuine_bc_second_continuation',
        scalar_critic_refit=cfg['verified_seconds'][str(seed)]['scalar_critic_refit'])


def verify_export(experiment,seed,cfg,decl,physical):
    rows,tasks=load_requests(experiment,seed,ROUND);count=cfg['streams'][str(seed)]['evaluation']['count']
    if [(r['role'],r['index']) for r in rows]!=[('evaluation',i) for i in range(count)]:
        raise ValueError('T2 omitted/reordered evaluation population or inserted collection')
    for row in rows:check_row(row,cfg,decl,physical,experiment,seed,row['index'])
    if len({r['noise_sha256'] for r in rows})!=count:raise ValueError('Repeated T2 evaluation noise')
    if read(tasks.with_name('completed.json'))!=completion(cfg,experiment,seed,rows,tasks):raise ValueError('T2 completion marker changed')
    return rows


def export(experiment,expected_config_sha256,seed):
    core.slurm();experiment=Path(experiment).resolve();cfg,decl,physical=settings(experiment,expected_config_sha256)
    if seed not in SEEDS:raise ValueError('Undeclared T2 seed')
    folder=experiment/f'seed_{seed}'/'round_2'/'requests';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'completed.json').exists():return verify_export(experiment,seed,cfg,decl,physical)
        if not torch.cuda.is_available():raise RuntimeError('New T2 graph inference requires allocated Slurm GPU')
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']));torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        verified=checked_parent(cfg['parent_training'][str(seed)],seed)
        if verified['identity']!=cfg['verified_seconds'][str(seed)]:raise ValueError('Actual T2 source changed')
        bind(RUNTIME/'benchmark_cache'/(cfg['design']+'.pt'),cfg['cache_sha256']);problem=load_problem(cfg['design'],'cuda')
        actor=GraphPolicy(problem)
        for name in ('movable','fixed'):core.equal_tree(getattr(actor,name),verified['payload']['actor'][name])
        actor.load_state_dict(verified['payload']['actor'],strict=True);actor.eval()
        critic=verified['source']['refit']['model']
        critic.load_state_dict(verified['payload']['critic'],strict=True);critic=critic.cuda().eval()
        metric=MacroHPWL(problem.benchmark_info);stream=cfg['streams'][str(seed)]['evaluation']
        generator=torch.Generator(device='cuda').manual_seed(stream['seed'])
        noises=2*torch.rand(stream['count'],len(problem.x),2,device='cuda',generator=generator)-1;rows=[]
        for index,noise in enumerate(noises):
            sample=folder/cfg['arm']/'evaluation'/str(index);sample.mkdir(parents=True,exist_ok=True);path=sample/'request.json'
            if path.exists():
                row=read(path);check_row(row,cfg,decl,physical,experiment,seed,index,noise.cpu().numpy());rows.append(row);continue
            with torch.no_grad():
                raw=actor(noise[None])[0];q=float(core.scalar_q(critic,raw[None])[0])
                if not math.isfinite(q):raise ValueError('Nonfinite T2 Q; no layout replaced or selected')
                action=legalize(raw,problem)
            a,r,n=(v.detach().cpu().numpy() for v in (action,raw,noise));geometry=legality(a,metric.sizes)
            identity=cfg['verified_seconds'][str(seed)];parent=cfg['parent_training'][str(seed)]['checkpoint']
            row=dict(arm=cfg['arm'],seed=seed,round=ROUND,role='evaluation',index=index,graph=cfg['design'],
                terminal_mask=0,training_feedback_eligible=False,offline_train_eligible=False,noise_seed=stream['seed'],
                raw_action=r.tolist(),executed_action=a.tolist(),predicted_q=[q],q_units=core.Q_UNITS,
                source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],source_state_key='actor',critic_state_key='critic',
                source_config_sha256=identity['parent_export_config']['sha256'],continuation_identity_sha256=core.json_sha(identity),
                training_request_sha256=identity['training_request']['sha256'],export_declaration_sha256=sha(experiment/'declaration.json'),
                scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],config_sha256=physical['config_sha256'],
                dataset_sha256=cfg['dataset_sha256'],sample_dir=str(sample),run_id=f"go2o_qsbc2_{cfg['run_prefix']}_s{seed}_r2_evaluation_{index}",
                action_sha256=action_sha(a),raw_action_sha256=action_sha(r),noise_sha256=action_sha(n),geometry=geometry,
                stage='global_route_only',macro_hpwl=float(metric(a)),evaluation_purpose=cfg['evaluation_purpose'],
                objective_weights=cfg['scalar_schema']['utility_spec']['weights'],
                q_weight=0.,actor_source_kind='genuine_bc_second_continuation',
                critic_source_kind='separately_bound_frozen_scalar_refit',
                original_bc_checkpoint=identity['bc_actor_prefix']['checkpoint'],scalar_critic_refit=identity['scalar_critic_refit'])
            np.savez_compressed(sample/'actions.npz',noise=n,raw_action=r,action=a)
            if geometry['legal']:
                save_sample(sample,action,noise,problem,row);def_path=sample/(cfg['design']+'.def')
                audit=convert(cfg['design'],sample/'macro_physical.json',def_path)
                row.update(def_path=str(def_path),def_sha256=sha(def_path),rounded_geometry=audit['legal_no_halo'])
                if not audit['legal_no_halo']:row['pre_backend_failure']='rounded_DEF_geometry'
            else:row['pre_backend_failure']='normalized_geometry'
            atomic_json(path,row);rows.append(row)
        tasks=folder/'tasks.jsonl';temp=tasks.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(tasks)
        original.immutable(folder/'completed.json',completion(cfg,experiment,seed,rows,tasks))
        return verify_export(experiment,seed,cfg,decl,physical)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare');p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('export');p.add_argument('--experiment',type=Path,required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.request,args.request_sha256,args.output)
    else:export(args.experiment,args.config_sha256,args.seed)


if __name__=='__main__':main()
