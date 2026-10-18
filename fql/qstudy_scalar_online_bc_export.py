"""Physical export of a genuine BC prefix with a separately bound scalar critic.

This adapter prepares/exports only. It neither updates BC nor claims the old
checkpoint was scalar-trained. Collection's real parent remains the original
BC checkpoint; the attached fixed scalar critic supplies a diagnostic only.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import torch

from . import qstudy_scalar_online_bc as provider
from . import qstudy_scalar_online as core
from . import qstudy_scalar_online_export as parent
from .qstudy_data import action_sha, bind, sha

ROOT = parent.ROOT
VERSION = 'qstudy_scalar_online_bc_export_v1'
SEEDS, ROLES = parent.SEEDS, parent.ROLES
SOURCE_NAMES = tuple(dict.fromkeys((*parent.SOURCE_NAMES,
    'fql/qstudy_scalar_online_bc.py', 'fql/qstudy_scalar_online_bc_export.py',
    'fql/qstudy_scalar_online_bc_export.slurm')))
read, immutable, context = parent.read, parent.immutable, parent.context


def request_settings(value):
    required = {'version', 'scalar_kind', 'bc_experiment', 'bc_config_sha256',
        'bc_parent_checkpoints', 'scalar_experiment', 'scalar_config_sha256',
        'scalar_parent_checkpoints', 'arm', 'run_prefix', 'streams',
        'evaluation_purpose', 'backend_storage', 'output'}
    if (set(value) != required or value['version'] != VERSION or
            value['scalar_kind'] not in ('rule', 'final') or
            not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', value['arm']) or
            not re.fullmatch(r'[A-Za-z][A-Za-z0-9]{0,23}', value['run_prefix'])):
        raise ValueError('Explicit BC plus rule/final scalar source declaration required')
    # Reuse the frozen stream/path validation, without relabeling its output.
    for kind in ('bc', 'scalar'):
        translated = dict(version=parent.VERSION,source_kind=value['scalar_kind'],
            source_experiment=value[kind+'_experiment'],
            source_config_sha256=value[kind+'_config_sha256'],
            parent_checkpoints=value[kind+'_parent_checkpoints'],
            **{k:value[k] for k in ('arm','run_prefix','streams','evaluation_purpose','backend_storage','output')})
        parent.request_settings(translated)
    if value['bc_experiment'] == value['scalar_experiment']:
        raise ValueError('Genuine BC and scalar target must remain separate sources')
    return value


def source_identities(request):
    identities = {}
    for kind in ('bc', 'scalar'):
        translated = dict(source_experiment=request[kind+'_experiment'],
            source_config_sha256=request[kind+'_config_sha256'],
            parent_checkpoints=request[kind+'_parent_checkpoints'])
        config, parents = parent.source_identities(translated)
        identities[kind] = dict(config=config, parents=parents)
    return identities


def load_source(request, seed):
    return provider.load_bc_source(request['scalar_kind'], request['bc_experiment'],
        request['scalar_experiment'], seed,
        expected_bc_config_sha256=request['bc_config_sha256'],
        expected_bc_checkpoint_sha256=request['bc_parent_checkpoints'][str(seed)]['sha256'],
        expected_scalar_config_sha256=request['scalar_config_sha256'],
        expected_scalar_checkpoint_sha256=request['scalar_parent_checkpoints'][str(seed)]['sha256'])


def source_files(request, identities):
    values = {name:sha(ROOT/name) for name in SOURCE_NAMES}
    scalar_cfg = read(identities['scalar']['config']['path'])
    prefix = 'rule' if request['scalar_kind'] == 'rule' else 'final_feedback'
    groups = [scalar_cfg[prefix+'_sources']]
    for identity in identities['bc']['parents'].values():
        groups.append(read(identity['training_binding']['path'])['source_sha256'])
    for group in groups:
        for name, digest in group.items():
            if name in values and values[name] != digest:
                raise ValueError('Two captured source closures disagree')
            bind(ROOT/name,digest)
            values[name] = digest
    return values


def config_value(request, request_binding, identities, verified, utility, reward_scale):
    old = read(identities['bc']['config']['path'])
    cfg = {k:old[k] for k in ('design','dataset_path','dataset_sha256','cache_sha256','physical_failure_reward')}
    cfg.update(scalar_online_bc_export_schema=VERSION, arm=request['arm'],run_prefix=request['run_prefix'],
        seeds=list(SEEDS),backend_threads=8,backend_storage='scratch',terminal_mask=0,
        q_weight=0.,alpha=10.,evaluation_purpose=request['evaluation_purpose'],
        source_kind='genuine_bc_with_separate_scalar_critic',scalar_kind=request['scalar_kind'],
        source_experiment=request['bc_experiment'],source_config=identities['bc']['config'],
        scalar_experiment=request['scalar_experiment'],scalar_config=identities['scalar']['config'],
        parent_identities=identities['bc']['parents'],scalar_parent_identities=identities['scalar']['parents'],
        verified_parents=verified,scalar_schema=utility,streams=request['streams'],
        prospective_request=request_binding,checkpoint_state_key='actor',reward_scale=reward_scale,
        online_training_implemented=False,q_units=core.Q_UNITS,
        raw_action_semantics='clipped original BC actor output; original distillation used unclipped output',
        scope='Original BC r0 physical export only, separate scalar diagnostic; no online updates/controller')
    return cfg


def declaration_value(cfg, request_binding, identities, physical, sources):
    return dict(version=VERSION,config=bind(Path(read(request_binding['path'])['output'])/'config.json'),
        request=request_binding,
        sources=identities,parents=cfg['verified_parents'],schema=cfg['scalar_schema'],
        physical_context=physical,streams=cfg['streams'],seeds=list(SEEDS),logical_round=0,
        old_training_checkpoint_round=0,source_checkpoint_unchanged=True,
        actor_source='genuine original BC r0 checkpoint; actor Adam and RNG not rewritten',
        critic_source='independently verified scalar refit; fresh empty critic optimizer; diagnostic only',
        q_weight=0.,alpha=10.,q_units=core.Q_UNITS,q_selection=False,
        role_semantics=dict(collect='training feedback eligible',evaluation='never replayed'),
        distribution='uniform [-1,1] generated on allocated CUDA device',
        legalizer=dict(w_reg=.1,w_dist=1.,grid_res=128,padding=.001),
        actor_updates=0,critic_updates=0,teacher_updates=0,backend_calls=0,source_sha256=sources)


def prepare(request_path, request_sha256, output):
    core.slurm()
    request_binding=bind(request_path,request_sha256)
    request=request_settings(read(request_path))
    output=Path(output).resolve()
    if str(output)!=request['output']:
        raise ValueError('Output differs from pinned prospective request')
    identities=source_identities(request)
    verified,utility={},None
    for seed in SEEDS:
        result=load_source(request,seed)
        if utility is not None and result['schema']!=utility:
            raise ValueError('All three scalar sources must share full-TRAIN utility')
        utility=result['schema'];verified[str(seed)]=result['binding']
    sources=source_files(request,identities)
    _,_,old_physical=context(Path(request['bc_experiment']))
    cfg=config_value(request,request_binding,identities,verified,utility,old_physical['reward_scale'])
    output.mkdir(parents=True,exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (output/'declaration.json').exists() and any(p.name not in ('prepare.lock','sources','config.json') for p in output.iterdir()):
            raise ValueError('Destination contains unrelated/unsealed artifacts')
        immutable(output/'config.json',cfg)
        _,_,physical=context(output)
        if {k:v for k,v in physical.items() if k!='config_sha256'}!={k:v for k,v in old_physical.items() if k!='config_sha256'}:
            raise ValueError('BC namespace changed original physical evaluator')
        for name,digest in sources.items():
            destination=output/'sources'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copyfile(ROOT/name,destination)
            bind(destination,digest)
        immutable(output/'declaration.json',declaration_value(cfg,request_binding,identities,physical,sources))
    print(json.dumps(dict(status='prepared_bc_export_only',experiment=str(output),
        config_sha256=sha(output/'config.json'),declaration_sha256=sha(output/'declaration.json'))),flush=True)


def settings(experiment, expected_config_sha256):
    experiment=Path(experiment).resolve();bind(experiment/'config.json',expected_config_sha256)
    cfg=read(experiment/'config.json');declaration=read(experiment/'declaration.json')
    request_binding=cfg['prospective_request'];bind(request_binding['path'],request_binding['sha256'])
    request=request_settings(read(request_binding['path']));identities=source_identities(request)
    _,_,old_physical=context(Path(request['bc_experiment']))
    backend_cfg,_,physical=context(experiment)
    expected=config_value(request,request_binding,identities,cfg['verified_parents'],cfg['scalar_schema'],old_physical['reward_scale'])
    sources=source_files(request,identities)
    if (str(experiment)!=request['output'] or cfg!=expected or backend_cfg!=cfg or
            {k:v for k,v in physical.items() if k!='config_sha256'}!={k:v for k,v in old_physical.items() if k!='config_sha256'} or
            declaration!=declaration_value(cfg,request_binding,identities,physical,sources)):
        raise ValueError('BC source/config/physical/declaration identity changed')
    core.validate_schema(cfg['scalar_schema'])
    for seed in SEEDS:
        bound=cfg['verified_parents'][str(seed)]
        if (bound['version']!=provider.VERSION or bound['schema']!=cfg['scalar_schema'] or
            bound['checkpoint']!=identities['bc']['parents'][str(seed)]['checkpoint'] or
            bound['config']!=identities['bc']['config'] or
            bound['completed']!=identities['bc']['parents'][str(seed)]['completed'] or
            bound['training_binding']!=identities['bc']['parents'][str(seed)]['training_binding'] or
            bound['target_scalar_source']['config']!=identities['scalar']['config'] or
            bound['target_scalar_source']['checkpoint']!=identities['scalar']['parents'][str(seed)]['checkpoint'] or
            bound['target_scalar_source']['completed']!=identities['scalar']['parents'][str(seed)]['completed'] or
            bound['target_scalar_source']['training_binding']!=identities['scalar']['parents'][str(seed)]['training_binding'] or
            bound['source_sha256']!=sha(provider.__file__)):
            raise ValueError('BC versus attached scalar source binding changed')
        bind(bound['critic_refit']['path'],bound['critic_refit']['sha256'])
    for name,digest in sources.items():bind(experiment/'sources'/name,digest)
    return cfg,declaration,physical


def check_row(row,cfg,declaration,physical,experiment,seed,role,index,expected_noise=None):
    parent.check_row(row,cfg,declaration,physical,experiment,seed,role,index,expected_noise)
    bound=cfg['verified_parents'][str(seed)]
    expected=dict(actor_source_kind='genuine_bc_prefix',q_weight=0.,
        scalar_source_config_sha256=cfg['scalar_config']['sha256'],
        scalar_source_checkpoint_sha256=cfg['scalar_parent_identities'][str(seed)]['checkpoint']['sha256'],
        scalar_critic_refit=bound['critic_refit'],assembled_payload_sha256=bound['assembled_payload_sha256'])
    if any(row.get(k)!=v for k,v in expected.items()):
        raise ValueError('Request confused genuine BC actor and attached scalar critic')


def completion(cfg,declaration,physical,experiment,seed,records,tasks):
    value=parent.completion(cfg,declaration,physical,experiment,seed,records,tasks)
    value.update(version=VERSION,q_weight=0.,scalar_source_config=cfg['scalar_config'],
        scalar_source_checkpoint=cfg['scalar_parent_identities'][str(seed)]['checkpoint'],
        assembled_payload_sha256=cfg['verified_parents'][str(seed)]['assembled_payload_sha256'])
    return value


def verify_export(experiment,seed,cfg,declaration,physical):
    records,tasks=parent.load_requests(experiment,seed,0)
    expected=[(role,i) for role in ROLES for i in range(cfg['streams'][str(seed)][role]['count'])]
    if [(r['role'],r['index']) for r in records]!=expected:
        raise ValueError('BC export omitted/reordered a declared noise')
    for row in records:check_row(row,cfg,declaration,physical,experiment,seed,row['role'],row['index'])
    noises={role:{r['noise_sha256'] for r in records if r['role']==role} for role in ROLES}
    if any(len(noises[role])!=cfg['streams'][str(seed)][role]['count'] for role in ROLES) or noises['collect']&noises['evaluation']:
        raise ValueError('BC collect/evaluation actual noises duplicate/overlap')
    if read(tasks.with_name('completed.json'))!=completion(cfg,declaration,physical,experiment,seed,records,tasks):
        raise ValueError('BC export completion differs from complete declared requests')
    return records


def export(experiment,expected_config_sha256,seed):
    core.slurm();experiment=Path(experiment).resolve()
    cfg,declaration,physical=settings(experiment,expected_config_sha256)
    if seed not in SEEDS:raise ValueError('Undeclared independent BC seed')
    folder=experiment/f'seed_{seed}'/'round_0'/'requests';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'completed.json').exists():return verify_export(experiment,seed,cfg,declaration,physical)
        if not torch.cuda.is_available():raise RuntimeError('BC layout export requires allocated Slurm GPU')
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        request=read(cfg['prospective_request']['path']);source=load_source(request,seed)
        if source['binding']!=cfg['verified_parents'][str(seed)] or source['schema']!=cfg['scalar_schema']:
            raise ValueError('BC assembled source changed before export')
        parent.bind(parent.RUNTIME/'benchmark_cache'/(cfg['design']+'.pt'),cfg['cache_sha256'])
        problem=parent.load_problem(cfg['design'],'cuda');actor=parent.GraphPolicy(problem)
        for name in ('movable','fixed'):
            if not torch.equal(getattr(actor,name).cpu(),source['payload']['actor'][name].cpu()):
                raise ValueError('BC fixed/movable identity differs from actual graph')
        actor.load_state_dict(source['payload']['actor'],strict=True);actor.eval()
        critic=source['refit']['model'].cuda().eval();metric=parent.MacroHPWL(problem.benchmark_info)
        records=[]
        for role in ROLES:
            stream=cfg['streams'][str(seed)][role]
            generator=torch.Generator(device='cuda').manual_seed(stream['seed'])
            noises=2*torch.rand(stream['count'],len(problem.x),2,device='cuda',generator=generator)-1
            for index,noise in enumerate(noises):
                sample=folder/cfg['arm']/role/str(index);sample.mkdir(parents=True,exist_ok=True)
                path=sample/'request.json'
                if path.exists():
                    row=read(path);check_row(row,cfg,declaration,physical,experiment,seed,role,index,noise.cpu().numpy())
                    records.append(row);continue
                with torch.no_grad():
                    raw=actor(noise[None])[0];predicted=float(core.scalar_q(critic,raw[None])[0])
                    if not np.isfinite(predicted):raise ValueError('Nonfinite diagnostic Q; no replacement/selection')
                    action=parent.legalize(raw,problem)
                a,r,n=(v.detach().cpu().numpy() for v in (action,raw,noise))
                geometry=parent.legality(a,metric.sizes);bound=cfg['verified_parents'][str(seed)]
                row=dict(arm=cfg['arm'],seed=seed,round=0,role=role,index=index,graph=cfg['design'],terminal_mask=0,
                    training_feedback_eligible=role=='collect',noise_seed=stream['seed'],raw_action=r.tolist(),executed_action=a.tolist(),
                    predicted_q=[predicted],q_units=core.Q_UNITS,actor_source_kind='genuine_bc_prefix',q_weight=0.,
                    source_checkpoint=bound['checkpoint']['path'],checkpoint_sha256=bound['checkpoint']['sha256'],
                    source_config_sha256=cfg['source_config']['sha256'],parent_identity_sha256=core.json_sha(bound),
                    export_declaration_sha256=sha(experiment/'declaration.json'),scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],
                    scalar_source_config_sha256=cfg['scalar_config']['sha256'],
                    scalar_source_checkpoint_sha256=cfg['scalar_parent_identities'][str(seed)]['checkpoint']['sha256'],
                    scalar_critic_refit=bound['critic_refit'],assembled_payload_sha256=bound['assembled_payload_sha256'],
                    config_sha256=physical['config_sha256'],dataset_sha256=cfg['dataset_sha256'],sample_dir=str(sample),
                    run_id=f"go2o_qs1_{cfg['run_prefix']}_s{seed}_r0_{role}_{index}",action_sha256=action_sha(a),
                    raw_action_sha256=action_sha(r),noise_sha256=action_sha(n),geometry=geometry,stage='global_route_only',
                    macro_hpwl=float(metric(a)),evaluation_purpose=cfg['evaluation_purpose'],
                    objective_weights=cfg['scalar_schema']['utility_spec']['weights'])
                np.savez_compressed(sample/'actions.npz',noise=n,raw_action=r,action=a)
                if geometry['legal']:
                    parent.save_sample(sample,action,noise,problem,row);def_path=sample/(cfg['design']+'.def')
                    audit=parent.convert(cfg['design'],sample/'macro_physical.json',def_path)
                    row.update(def_path=str(def_path),def_sha256=sha(def_path),rounded_geometry=audit['legal_no_halo'])
                    if not audit['legal_no_halo']:row['pre_backend_failure']='rounded_DEF_geometry'
                else:row['pre_backend_failure']='normalized_geometry'
                parent.atomic_json(path,row);records.append(row)
        tasks=folder/'tasks.jsonl';temp=tasks.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(r)+'\n' for r in records));temp.replace(tasks)
        immutable(folder/'completed.json',completion(cfg,declaration,physical,experiment,seed,records,tasks))
        return verify_export(experiment,seed,cfg,declaration,physical)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--request',type=Path,required=True)
    prep.add_argument('--request-sha256',required=True);prep.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('export');run.add_argument('--experiment',type=Path,required=True)
    run.add_argument('--config-sha256',required=True);run.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.request,args.request_sha256,args.output)
    else:export(args.experiment,args.config_sha256,args.seed)


if __name__=='__main__':main()
