"""Export genuine completed first BC continuations as logical round one.

The actor/Adam/RNG parent is the real BC continuation checkpoint. Its teacher
and separately attached scalar refit remain frozen; the original two-source
assembly and measured C0 lineage stay explicit. This module only exports every
declared collect/evaluation noise. It does not train, measure, select layouts,
relabel an FQL source, or implement a subsequent BC continuation.
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

from . import qstudy_scalar_online_bc_export as parent_export
from . import qstudy_scalar_online_bc_train as training
from . import qstudy_scalar_online_train as common
from . import qstudy_scalar_online_child_export as state_validation
from . import qstudy_scalar_online_bc as provider
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
VERSION='qstudy_scalar_online_bc_child_export_v1'
SEEDS=parent_export.SEEDS
ROLES=parent_export.ROLES
ROUND=1
SOURCE_NAMES=tuple(dict.fromkeys((*parent_export.SOURCE_NAMES,*training.SOURCE_NAMES,
    'fql/qstudy_scalar_online_child_export.py',
    'fql/qstudy_scalar_online_bc_child_export.py','fql/qstudy_scalar_online_bc_child_export.slurm')))


def read(path):return json.loads(Path(path).read_text())


def request_settings(value):
    required={'version','logical_round','parent_training','arm','run_prefix','streams',
              'evaluation_purpose','backend_storage','output'}
    if (set(value)!=required or value['version']!=VERSION or type(value['logical_round']) is not int or
        value['logical_round']!=ROUND or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',value['arm']) or
        not re.fullmatch(r'[A-Za-z][A-Za-z0-9]{0,23}',value['run_prefix']) or
        value['evaluation_purpose']!='development_validation' or value['backend_storage']!='scratch'):
        raise ValueError('Explicit genuine first-BC-continuation child request required; later cycles are unsupported')
    output=Path(value['output'])
    if not output.is_absolute() or str(output.resolve())!=str(output) or output==ROOT or output in ROOT.parents or ROOT in output.parents:
        raise ValueError('Child output must be canonical absolute and outside the source checkout')
    if set(value['parent_training'])!={str(s) for s in SEEDS} or set(value['streams'])!={str(s) for s in SEEDS}:
        raise ValueError('All three continuation parents and explicit streams are required')
    collects=[];evaluations=set()
    for seed in SEEDS:
        parent=value['parent_training'][str(seed)]
        if set(parent)!={'request','checkpoint'}:
            raise ValueError('Parent requires separate externally pinned training request and checkpoint')
        for item in parent.values():
            if (set(item)!={'path','sha256'} or not Path(item['path']).is_absolute() or
                str(Path(item['path']).resolve())!=item['path'] or not re.fullmatch('[0-9a-f]{64}',item['sha256'])):
                raise ValueError('Canonical external parent file bindings required')
        streams=value['streams'][str(seed)]
        if set(streams)!=set(ROLES):raise ValueError('Both explicit collect/evaluation streams are required')
        for stream in streams.values():
            if (set(stream)!={'seed','count'} or type(stream['seed']) is not int or not 0<=stream['seed']<2**63 or
                type(stream['count']) is not int or stream['count']<1):
                raise ValueError('Explicit stream seed/count required')
        collects.append(streams['collect']['seed']);evaluations.add(streams['evaluation']['seed'])
    if len(set(collects))!=3 or set(collects)&evaluations:
        raise ValueError('Collect seeds must be independent and cannot alias evaluation streams')
    return value


def validate_payload(payload,source,binding):
    """Audit actual actor continuation and the bitwise unchanged dual-source Q."""
    if (source['binding'].get('version')!=provider.VERSION or
        binding['settings'].get('q_weight')!=0. or binding['settings'].get('update_critic') is not False):
        raise ValueError('Only a genuine zero-Q BC continuation is a BC child source')
    training.checked_frozen_source(payload,source)
    result=state_validation.validate_payload(payload,source,binding)
    result.update(actor_source_kind='genuine_bc_continuation',critic_frozen=True,
        critic_source='separate original scalar refit; unchanged weights, optimizer and q RNG')
    return result


def checked_parent(parent,seed):
    """Reconstruct the actual frozen train admission, including real native replay."""
    request_bind=bind(parent['request']['path'],parent['request']['sha256'])
    request=training.request_settings(read(request_bind['path']))
    if request['seed']!=seed:raise ValueError('Continuation training seed differs')
    folder=Path(request['output'])
    checkpoint=bind(folder/'checkpoint.pt',parent['checkpoint']['sha256'])
    if checkpoint!=parent['checkpoint']:raise ValueError('Child source must be the real TRAIN checkpoint path')
    experiment=Path(request['experiment'])
    cfg,decl,physical=parent_export.settings(experiment,request['config_sha256'])
    bind(experiment/'declaration.json',request['declaration_sha256'])
    rows=parent_export.verify_export(experiment,seed,cfg,decl,physical)
    old=cfg['parent_identities'][str(seed)]['checkpoint']
    source_request=read(cfg['prospective_request']['path'])
    source=parent_export.load_source(source_request,seed)
    if source['binding']!=cfg['verified_parents'][str(seed)] or source['schema']!=cfg['scalar_schema']:
        raise ValueError('Genuine BC actor prefix or its separate scalar source differs')
    if old!=source['binding']['checkpoint']:
        raise ValueError('Original physical source is not the genuine BC checkpoint')
    original_cfg=read(Path(source_request['bc_experiment'])/'config.json')
    if (request['settings']['alpha']!=cfg['alpha'] or
        request['settings']['flow_steps']!=original_cfg['flow_steps'] or
        request['settings']['batch_size']!=original_cfg['batch_size']):
        raise ValueError('BC continuation changed the matched original actor update settings')
    cache_path=RUNTIME/'benchmark_cache'/(cfg['design']+'.pt');bind(cache_path,cfg['cache_sha256'])
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    stream=cfg['streams'][str(seed)]['collect']
    replay=core.admit_collect(experiment,seed=seed,source_round=0,count=stream['count'],
        tasks_sha256=request['tasks_sha256'],config_sha256=request['config_sha256'],
        checkpoint_sha256=old['sha256'],collection_noise_seed=stream['seed'],
        evaluation_noise_sha256=[r['noise_sha256'] for r in rows if r['role']=='evaluation'],
        value=source['schema'],cache=cache,final_outcomes={int(k):v for k,v in request['final_outcomes'].items()})
    if source['schema']['kind']=='rule' and request['final_outcomes']:
        raise ValueError('Rule source cannot claim final-route labels')
    binding=core.continuation_binding(source,replay,request['settings'])
    runner=dict(version=training.VERSION,request=request_bind,config=bind(experiment/'config.json'),
        declaration=bind(experiment/'declaration.json'),continuation=binding,
        source_sha256={name:sha(ROOT/name) for name in training.SOURCE_NAMES},
        diagnostic_interval=request['diagnostic_interval'],backend_calls=0)
    if read(folder/'binding.json')!=runner:raise ValueError('Actual continuation runner binding differs')
    payload=common.checked_resume(folder,runner,binding)
    training.checked_frozen_source(payload,source)
    training.checked_completion(folder,payload,runner,binding)
    history=read(folder/'learning_curve.json')
    if payload['history']!=history:raise ValueError('Saved continued history differs from the full checkpoint')
    audit=validate_payload(payload,source,binding)
    identity=dict(training_request=request_bind,checkpoint=checkpoint,completed=bind(folder/'completed.json'),
        runner_binding=bind(folder/'binding.json'),learning_curve=bind(folder/'learning_curve.json'),
        parent_export_config=bind(experiment/'config.json'),parent_export_declaration=bind(experiment/'declaration.json'),
        offline_source=source['binding'],bc_actor_prefix=source['binding']['actor_prefix'],
        scalar_source=source['binding']['target_scalar_source'],scalar_critic_refit=source['binding']['critic_refit'],
        continuation_binding=binding,state_audit=audit,
        previous_noise_sha256={role:sorted({r['noise_sha256'] for r in rows if r['role']==role}) for role in ROLES})
    return dict(payload=payload,source=source,cfg=cfg,physical=physical,declaration=decl,identity=identity)


def _parent_files(parent):
    request=training.request_settings(read(bind(parent['request']['path'],parent['request']['sha256'])['path']))
    folder=Path(request['output'])
    if bind(folder/'checkpoint.pt',parent['checkpoint']['sha256'])!=parent['checkpoint']:
        raise ValueError('Continuation checkpoint path identity differs')
    return request,{name:bind(folder/name) for name in ('checkpoint.pt','completed.json','binding.json','learning_curve.json')}


def prepare(request_path,request_sha256,output):
    core.slurm();request_bind=bind(request_path,request_sha256);request=request_settings(read(request_path))
    output=Path(output).resolve()
    if str(output)!=request['output']:raise ValueError('Child output differs from its externally pinned declaration')
    children={};base=None;sources={};files={}
    for seed in SEEDS:
        result=checked_parent(request['parent_training'][str(seed)],seed)
        cfg=result['cfg'];identity=result['identity']
        if base is not None and (cfg!=base or result['source']['schema']!=base['scalar_schema']):
            raise ValueError('Three continuation seeds must share one original physical config and fixed utility')
        base=cfg;children[str(seed)]=identity
        sources.update(result['declaration']['source_sha256'])
        previous={v['seed'] for s in base['streams'].values() for v in s.values()}
        if request['streams'][str(seed)]['collect']['seed'] in previous:
            raise ValueError('New collect stream must not repeat any previous collect/evaluation stream')
        if request['streams'][str(seed)]['evaluation']['seed'] in {s['collect']['seed'] for s in base['streams'].values()}:
            raise ValueError('Old training collect stream cannot become held-out evaluation')
        train_req,files[str(seed)]=_parent_files(request['parent_training'][str(seed)])
        for oldpath in (Path(train_req['output']),Path(train_req['experiment']),Path(cfg['source_experiment']),Path(cfg['scalar_experiment'])):
            if output==oldpath or output in oldpath.parents or oldpath in output.parents:
                raise ValueError('Child namespace must be disjoint from every real source namespace')
    sources.update({name:sha(ROOT/name) for name in SOURCE_NAMES})
    cfg={k:base[k] for k in ('design','dataset_path','dataset_sha256','cache_sha256','physical_failure_reward','reward_scale')}
    cfg.update(scalar_online_bc_child_export_schema=VERSION,arm=request['arm'],run_prefix=request['run_prefix'],
        seeds=list(SEEDS),logical_round=ROUND,streams=request['streams'],backend_threads=8,backend_storage='scratch',
        terminal_mask=0,evaluation_purpose=request['evaluation_purpose'],scalar_schema=base['scalar_schema'],
        q_weight=0.,alpha=base['alpha'],actor_source_kind='genuine_bc_continuation',
        critic_source_kind='separately_bound_frozen_scalar_refit',
        parent_training=request['parent_training'],parent_files=files,verified_children=children,
        prospective_request=request_bind,q_units=core.Q_UNITS,checkpoint_state_key='actor',
        scope='Genuine completed first BC continuation; frozen separate scalar critic; no subsequent training cycle',
        raw_action_semantics='clipped actor output; training distillation used unclipped actor output')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (output/'declaration.json').exists() and any(p.name not in ('prepare.lock','config.json','sources') for p in output.iterdir()):
            raise ValueError('Child destination contains unrelated/unsealed artifacts')
        parent_export.immutable(output/'config.json',cfg)
        _,_,physical=context(output)
        previous=context(Path(children['0']['parent_export_config']['path']).parent)[2]
        if {k:v for k,v in physical.items() if k!='config_sha256'}!={k:v for k,v in previous.items() if k!='config_sha256'}:
            raise ValueError('Child namespace changed its native physical evaluator')
        for name,digest in sources.items():
            path=output/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():shutil.copyfile(ROOT/name,path)
            bind(path,digest)
        declaration=dict(version=VERSION,config=bind(output/'config.json'),request=request_bind,
            parents=children,parent_files=files,physical_context=physical,schema=cfg['scalar_schema'],
            streams=request['streams'],logical_round=ROUND,source_sha256=sources,
            checkpoint_mutations=0,actor_updates=0,critic_updates=0,teacher_updates=0,backend_calls=0,
            actor_source_kind='genuine_bc_continuation',critic_source_kind='separately_bound_frozen_scalar_refit',q_weight=0.,
            q_units=core.Q_UNITS,q_selection=False,evaluation_never_replayed=True,
            legalizer=dict(w_reg=.1,w_dist=1.,grid_res=128,padding=.001))
        parent_export.immutable(output/'declaration.json',declaration)
    print(json.dumps(dict(status='prepared_child_export_only',experiment=str(output),
        config_sha256=sha(output/'config.json'),declaration_sha256=sha(output/'declaration.json'))),flush=True)


def settings(experiment,expected_config_sha256):
    experiment=Path(experiment).resolve();bind(experiment/'config.json',expected_config_sha256)
    cfg=read(experiment/'config.json');decl=read(experiment/'declaration.json')
    request_bind=cfg['prospective_request'];request=request_settings(read(bind(request_bind['path'],request_bind['sha256'])['path']))
    _,_,physical=context(experiment)
    if (cfg.get('scalar_online_bc_child_export_schema')!=VERSION or cfg.get('logical_round')!=ROUND or
        cfg.get('arm')!=request['arm'] or cfg.get('run_prefix')!=request['run_prefix'] or
        cfg.get('parent_training')!=request['parent_training'] or cfg.get('streams')!=request['streams'] or
        str(experiment)!=request['output'] or cfg.get('q_units')!=core.Q_UNITS or cfg.get('q_weight')!=0. or
        cfg.get('actor_source_kind')!='genuine_bc_continuation' or
        cfg.get('critic_source_kind')!='separately_bound_frozen_scalar_refit' or cfg.get('seeds')!=list(SEEDS) or
        decl.get('version')!=VERSION or decl.get('config')!=bind(experiment/'config.json') or
        decl.get('request')!=request_bind or decl.get('parents')!=cfg['verified_children'] or
        decl.get('parent_files')!=cfg['parent_files'] or decl.get('schema')!=cfg['scalar_schema'] or
        decl.get('streams')!=cfg['streams'] or decl.get('physical_context')!=physical or decl.get('logical_round')!=ROUND or
        decl.get('q_units')!=core.Q_UNITS or decl.get('q_selection') is not False or decl.get('q_weight')!=0. or
        decl.get('actor_source_kind')!='genuine_bc_continuation' or
        decl.get('critic_source_kind')!='separately_bound_frozen_scalar_refit' or
        decl.get('evaluation_never_replayed') is not True or
        decl.get('legalizer')!=dict(w_reg=.1,w_dist=1.,grid_res=128,padding=.001) or
        any(decl.get(k)!=0 for k in ('checkpoint_mutations','actor_updates','critic_updates','teacher_updates','backend_calls'))):
        raise ValueError('Child config/declaration identity changed')
    core.validate_schema(cfg['scalar_schema'])
    for seed in SEEDS:
        parent_request,files=_parent_files(request['parent_training'][str(seed)])
        if files!=cfg['parent_files'][str(seed)]:raise ValueError('Original continuation artifacts changed')
        identity=cfg['verified_children'][str(seed)];source=identity['offline_source']
        if (identity['training_request']!=request['parent_training'][str(seed)]['request'] or
            identity['checkpoint']!=request['parent_training'][str(seed)]['checkpoint'] or
            source.get('version')!=provider.VERSION or identity['bc_actor_prefix']!=source['actor_prefix'] or
            identity['scalar_source']!=source['target_scalar_source'] or
            identity['scalar_critic_refit']!=source['critic_refit'] or
            identity['continuation_binding']['settings']!=parent_request['settings']):
            raise ValueError('BC child lost its genuine actor/scalar two-source lineage')
        for name in ('parent_export_config','parent_export_declaration'):
            item=identity[name];bind(item['path'],item['sha256'])
        original_cfg=read(identity['parent_export_config']['path'])
        if cfg['alpha']!=original_cfg['alpha'] or cfg['scalar_schema']!=original_cfg['scalar_schema']:
            raise ValueError('BC child changed the original fixed actor/scalar objective')
    for name,digest in decl['source_sha256'].items():bind(ROOT/name,digest);bind(experiment/'sources'/name,digest)
    if any(name not in decl['source_sha256'] for name in SOURCE_NAMES):raise ValueError('Child source closure incomplete')
    return cfg,decl,physical


def check_row(row,cfg,decl,physical,experiment,seed,role,index,noise=None):
    parent=cfg['parent_training'][str(seed)]['checkpoint'];identity=cfg['verified_children'][str(seed)]
    expected=dict(arm=cfg['arm'],graph=cfg['design'],seed=seed,round=ROUND,role=role,index=index,
        checkpoint_sha256=parent['sha256'],source_checkpoint=parent['path'],config_sha256=physical['config_sha256'],
        dataset_sha256=cfg['dataset_sha256'],source_config_sha256=identity['parent_export_config']['sha256'],
        continuation_identity_sha256=core.json_sha(identity),training_request_sha256=identity['training_request']['sha256'],
        export_declaration_sha256=sha(experiment/'declaration.json'),
        scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],q_units=core.Q_UNITS,
        noise_seed=cfg['streams'][str(seed)][role]['seed'],terminal_mask=0,training_feedback_eligible=role=='collect',
        evaluation_purpose=cfg['evaluation_purpose'],stage='global_route_only',
        run_id=f"go2o_qsbc_{cfg['run_prefix']}_s{seed}_r1_{role}_{index}",objective_weights=cfg['scalar_schema']['utility_spec']['weights'],
        actor_source_kind='genuine_bc_continuation',q_weight=0.,
        critic_source_kind='separately_bound_frozen_scalar_refit',
        original_bc_checkpoint=identity['bc_actor_prefix']['checkpoint'],scalar_critic_refit=identity['scalar_critic_refit'])
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Child request changed real continuation/round/role/noise/spec identity')
    q=row.get('predicted_q')
    if not isinstance(q,list) or len(q)!=1 or type(q[0]) not in (int,float) or not math.isfinite(q[0]):
        raise ValueError('Finite one-element child scalar Q required; no Q-selection')
    sample=checked_request(row,argparse.Namespace(seed=seed,round=ROUND),cfg,physical)
    if sample.resolve()!=(experiment/f'seed_{seed}'/'round_1'/'requests'/cfg['arm']/role/str(index)).resolve():
        raise ValueError('Child request escapes its declared logical round-one namespace')
    if noise is not None and row['noise_sha256']!=action_sha(noise):raise ValueError('Interrupted child export changed actual noise')
    old=identity['previous_noise_sha256']
    if row['noise_sha256'] in (set(old['collect'])|set(old['evaluation']) if role=='collect' else set(old['collect'])):
        raise ValueError('Child role reuses an observed training/evaluation noise illegally')
    return sample


def completion(cfg,experiment,seed,rows,tasks):
    return dict(status='completed',version=VERSION,count=len(rows),tasks_sha256=sha(tasks),
        config_sha256=sha(experiment/'config.json'),export_declaration_sha256=sha(experiment/'declaration.json'),
        source_checkpoint=cfg['parent_training'][str(seed)]['checkpoint'],logical_round=ROUND,
        continuation_identity_sha256=core.json_sha(cfg['verified_children'][str(seed)]),
        role_counts={r:cfg['streams'][str(seed)][r]['count'] for r in ROLES},evaluation_never_replayed=True,
        q_units=core.Q_UNITS,q_selection=False,q_weight=0.,actor_source_kind='genuine_bc_continuation',
        scalar_critic_refit=cfg['verified_children'][str(seed)]['scalar_critic_refit'],checkpoint_mutations=0,actor_updates=0,critic_updates=0,teacher_updates=0,backend_calls=0)


def verify_export(experiment,seed,cfg,decl,physical):
    rows,tasks=load_requests(experiment,seed,ROUND)
    if [(r['role'],r['index']) for r in rows]!=[(r,i) for r in ROLES for i in range(cfg['streams'][str(seed)][r]['count'])]:
        raise ValueError('Child export omitted/reordered declared role/index population')
    for row in rows:check_row(row,cfg,decl,physical,experiment,seed,row['role'],row['index'])
    noises={r:{x['noise_sha256'] for x in rows if x['role']==r} for r in ROLES}
    if any(len(noises[r])!=cfg['streams'][str(seed)][r]['count'] for r in ROLES) or noises['collect']&noises['evaluation']:
        raise ValueError('Duplicated/overlapping child actual noises')
    if read(tasks.with_name('completed.json'))!=completion(cfg,experiment,seed,rows,tasks):raise ValueError('Child completion changed')
    return rows


def export(experiment,expected_config_sha256,seed):
    core.slurm();experiment=Path(experiment).resolve();cfg,decl,physical=settings(experiment,expected_config_sha256)
    if seed not in SEEDS:raise ValueError('Undeclared child seed')
    folder=experiment/f'seed_{seed}'/'round_1'/'requests';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'completed.json').exists():return verify_export(experiment,seed,cfg,decl,physical)
        if not torch.cuda.is_available():raise RuntimeError('New child export requires an allocated Slurm GPU')
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']));torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        verified=checked_parent(cfg['parent_training'][str(seed)],seed)
        if verified['identity']!=cfg['verified_children'][str(seed)]:raise ValueError('Actual completed child source changed')
        bind(RUNTIME/'benchmark_cache'/(cfg['design']+'.pt'),cfg['cache_sha256']);problem=load_problem(cfg['design'],'cuda')
        actor=GraphPolicy(problem)
        for name in ('movable','fixed'):core.equal_tree(getattr(actor,name),verified['payload']['actor'][name])
        actor.load_state_dict(verified['payload']['actor'],strict=True);actor.eval()
        critic=verified['source']['refit']['model'].cuda().eval();metric=MacroHPWL(problem.benchmark_info);rows=[]
        for role in ROLES:
            stream=cfg['streams'][str(seed)][role];generator=torch.Generator(device='cuda').manual_seed(stream['seed'])
            noises=2*torch.rand(stream['count'],len(problem.x),2,device='cuda',generator=generator)-1
            for index,noise in enumerate(noises):
                sample=folder/cfg['arm']/role/str(index);sample.mkdir(parents=True,exist_ok=True);path=sample/'request.json'
                if path.exists():
                    row=read(path);check_row(row,cfg,decl,physical,experiment,seed,role,index,noise.cpu().numpy());rows.append(row);continue
                with torch.no_grad():
                    raw=actor(noise[None])[0];q=float(core.scalar_q(critic,raw[None])[0])
                    if not math.isfinite(q):raise ValueError('Nonfinite child Q; no layout selected/replaced')
                    action=legalize(raw,problem)
                a,r,n=(v.detach().cpu().numpy() for v in (action,raw,noise));geometry=legality(a,metric.sizes)
                identity=cfg['verified_children'][str(seed)];parent=cfg['parent_training'][str(seed)]['checkpoint']
                row=dict(arm=cfg['arm'],seed=seed,round=ROUND,role=role,index=index,graph=cfg['design'],
                    terminal_mask=0,training_feedback_eligible=role=='collect',noise_seed=stream['seed'],
                    raw_action=r.tolist(),executed_action=a.tolist(),predicted_q=[q],q_units=core.Q_UNITS,
                    actor_source_kind='genuine_bc_continuation',q_weight=0.,
                    critic_source_kind='separately_bound_frozen_scalar_refit',
                    original_bc_checkpoint=cfg['verified_children'][str(seed)]['bc_actor_prefix']['checkpoint'],
                    scalar_critic_refit=cfg['verified_children'][str(seed)]['scalar_critic_refit'],
                    source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],
                    source_config_sha256=identity['parent_export_config']['sha256'],
                    continuation_identity_sha256=core.json_sha(identity),training_request_sha256=identity['training_request']['sha256'],
                    export_declaration_sha256=sha(experiment/'declaration.json'),scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],
                    config_sha256=physical['config_sha256'],dataset_sha256=cfg['dataset_sha256'],sample_dir=str(sample),
                    run_id=f"go2o_qsbc_{cfg['run_prefix']}_s{seed}_r1_{role}_{index}",action_sha256=action_sha(a),
                    raw_action_sha256=action_sha(r),noise_sha256=action_sha(n),geometry=geometry,stage='global_route_only',
                    macro_hpwl=float(metric(a)),evaluation_purpose=cfg['evaluation_purpose'],objective_weights=cfg['scalar_schema']['utility_spec']['weights'])
                np.savez_compressed(sample/'actions.npz',noise=n,raw_action=r,action=a)
                if geometry['legal']:
                    save_sample(sample,action,noise,problem,row);def_path=sample/(cfg['design']+'.def')
                    audit=convert(cfg['design'],sample/'macro_physical.json',def_path)
                    row.update(def_path=str(def_path),def_sha256=sha(def_path),rounded_geometry=audit['legal_no_halo'])
                    if not audit['legal_no_halo']:row['pre_backend_failure']='rounded_DEF_geometry'
                else:row['pre_backend_failure']='normalized_geometry'
                atomic_json(path,row);rows.append(row)
        tasks=folder/'tasks.jsonl';temp=tasks.with_suffix('.jsonl.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(tasks)
        parent_export.immutable(folder/'completed.json',completion(cfg,experiment,seed,rows,tasks))
        return verify_export(experiment,seed,cfg,decl,physical)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare');p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('export');p.add_argument('--experiment',type=Path,required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.request,args.request_sha256,args.output)
    else:export(args.experiment,args.config_sha256,args.seed)


if __name__=='__main__':main()
