"""Export a verified offline scalar actor into a new physical namespace.

CPU prepare binds genuine old r0 config/checkpoint/critic identities to a new
immutable physical config. GPU export preserves the old checkpoint verbatim,
generates only the explicitly declared collect/evaluation streams, and runs
the existing final legalizer/DEF conversion. No update/controller/BC source is
implemented. This is not an end-to-end online experiment or a reward choice.
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

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import RUNTIME, legalize, load_problem, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json
from .graph_o2o_backend import load_requests
from .graph_o2o_models import GraphPolicy
from .qstudy_backend import checked_request, context
from .qstudy_data import action_sha, bind, sha
from . import qstudy_scalar_online as core

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qstudy_scalar_online_parent_export_v1'
SEEDS = (0, 1, 2)
ROLES = ('collect', 'evaluation')
SOURCE_NAMES = ('fql/qstudy_scalar_online_export.py', 'fql/qstudy_scalar_online_export.slurm',
    'fql/qstudy_scalar_online.py', 'fql/qstudy_backend.py', 'fql/graph_o2o_backend.py',
    'fql/graph_o2o_models.py', 'baseline/api.py', 'backend/scripts/export_macro_def.py',
    'analysis/placement_metrics.py', 'upstream/flowplace/benchmark_utils/openroad_utils.py',
    *('upstream/flowplace/flow_matching/'+name for name in
      ('__init__.py','models.py','pos_encoding.py','schedulers.py','guidance.py',
       'hard_constraint_legalizer.py','configs/config_eval.yaml',
       'configs/model/att-gnn.yaml','configs/model/conv-layer/gat.yaml',
       'configs/model/size/large.yaml','configs/guidance/none.yaml',
       'configs/legalizer/none.yaml','configs/sc_placer/sgd.yaml')),
    # networks.__init__ imports these implementations, even for the fixed
    # AttGNN backbone. This is a small, explicit package scope, not a repo scan.
    *(str(p.relative_to(ROOT)) for p in sorted(
        (ROOT/'upstream/flowplace/flow_matching/networks').rglob('*.py'))))


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable parent-export artifact differs: '+str(path))
    else:
        atomic_json(path, value)


def request_settings(value):
    required = {'version', 'source_kind', 'source_experiment', 'source_config_sha256',
                'parent_checkpoints', 'arm', 'run_prefix', 'streams', 'evaluation_purpose',
                'backend_storage', 'output'}
    if (set(value) != required or value['version'] != VERSION or value['source_kind'] not in ('rule','final') or
            not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', value['arm']) or
            not re.fullmatch(r'[A-Za-z][A-Za-z0-9]{0,23}', value['run_prefix']) or
            value['evaluation_purpose'] != 'development_validation' or value['backend_storage'] != 'scratch'):
        raise ValueError('Unsupported or incomplete prospective parent-export declaration (BC is not supported)')
    for name in ('source_experiment','output'):
        path = Path(value[name])
        if not path.is_absolute() or str(path.resolve()) != value[name]:
            raise ValueError('Canonical absolute experiment/output paths required')
    if set(value['streams']) != {str(s) for s in SEEDS} or set(value['parent_checkpoints']) != {str(s) for s in SEEDS}:
        raise ValueError('All three independent offline source seeds must be explicit')
    # Evaluation may share one seed across actor seeds; collect must never draw
    # a prefix of any held-out evaluation stream, including another seed.
    collect_seeds, evaluation_seeds = [], set()
    for seed in SEEDS:
        stream = value['streams'][str(seed)]
        if set(stream) != set(ROLES):
            raise ValueError('Both collect and evaluation streams are required')
        for role in ROLES:
            item = stream[role]
            if (set(item) != {'seed','count'} or not isinstance(item['seed'],int) or isinstance(item['seed'],bool) or
                    not 0 <= item['seed'] < 2**63 or not isinstance(item['count'],int) or
                    isinstance(item['count'],bool) or item['count'] < 1):
                raise ValueError('Stream seed/count must be explicit nonnegative/positive integers')
        collect_seeds.append(stream['collect']['seed'])
        evaluation_seeds.add(stream['evaluation']['seed'])
    if len(set(collect_seeds)) != len(SEEDS) or set(collect_seeds) & evaluation_seeds:
        raise ValueError('Collect seeds must be independent and disjoint from every evaluation stream')
    source, output = Path(value['source_experiment']), Path(value['output'])
    if source == output or source in output.parents or output in source.parents or output == ROOT or output in ROOT.parents:
        raise ValueError('New physical namespace must be disjoint from the original actor experiment')
    return value


def source_identities(request):
    """Bind old files without copying or editing a training config/checkpoint."""
    source = Path(request['source_experiment'])
    config = bind(source/'config.json', request['source_config_sha256'])
    parents = {}
    for seed in SEEDS:
        folder = source/f'seed_{seed}'/'round_0'
        expected = request['parent_checkpoints'][str(seed)]
        if set(expected) != {'path','sha256'} or bind(folder/'checkpoint.pt',expected['sha256']) != expected:
            raise ValueError('Parent checkpoint must name the genuine original completed round0 file')
        parents[str(seed)] = dict(checkpoint=expected, completed=bind(folder/'completed.json'),
                                 training_binding=bind(folder/'binding.json'))
    return config, parents


def prepare(request_path, request_sha256, output):
    core.slurm()
    request_binding = bind(request_path, request_sha256)
    request = request_settings(read(request_path))
    output = Path(output).resolve()
    if str(output) != request['output']:
        raise ValueError('Output differs from its externally pinned prospective request')
    source_config, identities = source_identities(request)
    source_cfg = read(source_config['path'])
    parents, utility = {}, None
    for seed in SEEDS:
        result = core.load_offline_source(request['source_kind'],request['source_experiment'],seed,
                                         identities[str(seed)]['checkpoint']['sha256'])
        if utility is not None and result['schema'] != utility:
            raise ValueError('Source seeds do not share the fixed full-TRAIN scalar utility')
        utility = result['schema']
        parents[str(seed)] = result['binding']
    physical_keys = ('design','dataset_path','dataset_sha256','cache_sha256','physical_failure_reward')
    cfg = {key:source_cfg[key] for key in physical_keys}
    cfg.update(scalar_online_export_schema=VERSION, arm=request['arm'],run_prefix=request['run_prefix'],
        seeds=list(SEEDS),backend_threads=8,backend_storage='scratch',terminal_mask=0,
        evaluation_purpose=request['evaluation_purpose'],scalar_schema=utility,
        source_kind=request['source_kind'],source_experiment=request['source_experiment'],
        source_config=source_config,parent_identities=identities,verified_parents=parents,
        streams=request['streams'],prospective_request=request_binding,
        checkpoint_state_key='actor',online_training_implemented=False,
        scope='verified offline r0 actor export only; no BC source or online training/controller',
        q_units=core.Q_UNITS,raw_action_semantics='clipped actor output; distillation used unclipped actor output')
    # Keep the old native reward scale explicit even though the actor's Q uses
    # B units. These are different quantities and never overwrite one another.
    _,_,old_physical = context(Path(request['source_experiment']))
    cfg['reward_scale'] = old_physical['reward_scale']
    prefix = 'rule' if request['source_kind']=='rule' else 'final_feedback'
    source_sha = dict(source_cfg[prefix+'_sources'])
    source_sha.update({name:sha(ROOT/name) for name in SOURCE_NAMES})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (output/'declaration.json').exists() and any(p.name not in ('prepare.lock','sources','config.json') for p in output.iterdir()):
            raise ValueError('Destination contains unrelated or unsealed artifacts')
        immutable(output/'config.json',cfg)
        _,_,new_physical = context(output)
        if {k:v for k,v in old_physical.items() if k!='config_sha256'} != {
                k:v for k,v in new_physical.items() if k!='config_sha256'}:
            raise ValueError('New physical namespace changed the frozen native evaluator')
        for name,digest in source_sha.items():
            target=output/'sources'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                bind(target,digest)
            else:
                shutil.copyfile(ROOT/name,target);bind(target,digest)
        declaration=dict(version=VERSION,config=bind(output/'config.json'),request=request_binding,
            source_config=source_config,parents=parents,source_identities=identities,
            physical_context=new_physical,schema=utility,streams=request['streams'],seeds=list(SEEDS),
            role_semantics=dict(collect='training feedback eligible',evaluation='never replayed'),
            logical_round=0,old_training_checkpoint_round=0,source_checkpoint_unchanged=True,
            actor_updates=0,critic_updates=0,teacher_updates=0,backend_calls=0,
            q_units=core.Q_UNITS,q_selection=False,distribution='uniform [-1,1] generated on allocated CUDA device',
            legalizer=dict(w_reg=.1,w_dist=1.,grid_res=128,padding=.001),
            source_sha256=source_sha)
        immutable(output/'declaration.json',declaration)
    print(json.dumps(dict(status='prepared_export_only',experiment=str(output),
        config_sha256=sha(output/'config.json'),declaration_sha256=sha(output/'declaration.json'))),flush=True)


def settings(experiment, expected_config_sha256):
    experiment=Path(experiment).resolve()
    bind(experiment/'config.json',expected_config_sha256)
    cfg=read(experiment/'config.json'); declaration=read(experiment/'declaration.json')
    request_binding=cfg['prospective_request'];bind(request_binding['path'],request_binding['sha256'])
    request=request_settings(read(request_binding['path']))
    source_config,parents=source_identities(request)
    source_cfg=read(source_config['path'])
    backend_cfg,_,physical=context(experiment)
    _,_,old_physical=context(Path(request['source_experiment']))
    if (cfg!=backend_cfg or cfg.get('scalar_online_export_schema')!=VERSION or
        cfg.get('online_training_implemented') is not False or cfg.get('checkpoint_state_key')!='actor' or
        cfg.get('source_config')!=source_config or cfg.get('parent_identities')!=parents or
        cfg.get('streams')!=request['streams'] or str(experiment)!=request['output'] or
        cfg.get('arm')!=request['arm'] or cfg.get('run_prefix')!=request['run_prefix'] or
        cfg.get('source_kind')!=request['source_kind'] or cfg.get('source_experiment')!=request['source_experiment'] or
        cfg.get('q_units')!=core.Q_UNITS or declaration.get('version')!=VERSION or
        cfg.get('seeds')!=list(SEEDS) or cfg.get('backend_threads')!=8 or
        cfg.get('backend_storage')!='scratch' or cfg.get('terminal_mask')!=0 or
        cfg.get('evaluation_purpose')!=request['evaluation_purpose'] or
        any(cfg.get(k)!=source_cfg[k] for k in
            ('design','dataset_path','dataset_sha256','cache_sha256','physical_failure_reward')) or
        {k:v for k,v in physical.items() if k!='config_sha256'}!={
            k:v for k,v in old_physical.items() if k!='config_sha256'} or
        declaration.get('config')!=bind(experiment/'config.json') or declaration.get('request')!=request_binding or
        declaration.get('source_config')!=source_config or declaration.get('source_identities')!=parents or
        declaration.get('parents')!=cfg['verified_parents'] or declaration.get('schema')!=cfg['scalar_schema'] or
        declaration.get('streams')!=cfg['streams'] or declaration.get('physical_context')!=physical or
        declaration.get('logical_round')!=0 or declaration.get('source_checkpoint_unchanged') is not True or
        declaration.get('old_training_checkpoint_round')!=0 or declaration.get('seeds')!=list(SEEDS) or
        declaration.get('role_semantics')!=dict(collect='training feedback eligible',evaluation='never replayed') or
        declaration.get('q_units')!=core.Q_UNITS or
        declaration.get('distribution')!='uniform [-1,1] generated on allocated CUDA device' or
        declaration.get('legalizer')!=dict(w_reg=.1,w_dist=1.,grid_res=128,padding=.001) or
        declaration.get('q_selection') is not False or any(declaration.get(k)!=0 for k in
            ('actor_updates','critic_updates','teacher_updates','backend_calls'))):
        raise ValueError('Prepared bridge config/source/declaration identity changed')
    core.validate_schema(cfg['scalar_schema'])
    for name,digest in declaration['source_sha256'].items():
        bind(ROOT/name,digest);bind(experiment/'sources'/name,digest)
    if any(name not in declaration['source_sha256'] for name in SOURCE_NAMES):
        raise ValueError('Parent-export source closure is incomplete')
    prefix='rule' if request['source_kind']=='rule' else 'final_feedback'
    if any(declaration['source_sha256'].get(name)!=digest for name,digest in source_cfg[prefix+'_sources'].items()):
        raise ValueError('Source actor dependencies differ from their verified old closure')
    return cfg,declaration,physical


def check_row(row,cfg,declaration,physical,experiment,seed,role,index,expected_noise=None):
    stream=cfg['streams'][str(seed)][role]
    parent=cfg['parent_identities'][str(seed)]['checkpoint']
    expected=dict(arm=cfg['arm'],graph=cfg['design'],seed=seed,round=0,role=role,index=index,
        checkpoint_sha256=parent['sha256'],source_checkpoint=parent['path'],
        config_sha256=physical['config_sha256'],dataset_sha256=cfg['dataset_sha256'],
        source_config_sha256=cfg['source_config']['sha256'],
        parent_identity_sha256=core.json_sha(cfg['verified_parents'][str(seed)]),
        export_declaration_sha256=sha(experiment/'declaration.json'),
        scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],q_units=core.Q_UNITS,
        noise_seed=stream['seed'],terminal_mask=0,training_feedback_eligible=role=='collect',
        evaluation_purpose=cfg['evaluation_purpose'],stage='global_route_only',
        run_id=f"go2o_qs1_{cfg['run_prefix']}_s{seed}_r0_{role}_{index}",
        objective_weights=cfg['scalar_schema']['utility_spec']['weights'])
    if any(row.get(k)!=v for k,v in expected.items()):
        raise ValueError('Bridge request changed old-parent/new-config/role/noise/spec identity')
    predicted=row.get('predicted_q')
    if (not isinstance(predicted,list) or len(predicted)!=1 or isinstance(predicted[0],bool) or
            not isinstance(predicted[0],(int,float)) or not math.isfinite(predicted[0])):
        raise ValueError('Saved scalar Q must be a finite one-element diagnostic, never a selector')
    sample=checked_request(row,argparse.Namespace(seed=seed,round=0),cfg,physical)
    if sample.resolve()!=(experiment/f'seed_{seed}'/'round_0'/'requests'/cfg['arm']/role/str(index)).resolve():
        raise ValueError('Request sample escapes its declared seed/role/index')
    if expected_noise is not None and row['noise_sha256']!=action_sha(expected_noise):
        raise ValueError('Interrupted export changed its actual declared CUDA noise')
    return sample


def completion(cfg,declaration,physical,experiment,seed,records,tasks):
    return dict(status='completed',version=VERSION,count=len(records),tasks_sha256=sha(tasks),
        config_sha256=physical['config_sha256'],export_declaration_sha256=sha(experiment/'declaration.json'),
        source_checkpoint=cfg['parent_identities'][str(seed)]['checkpoint'],
        source_config=cfg['source_config'],parent_identity_sha256=core.json_sha(cfg['verified_parents'][str(seed)]),
        scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],
        role_counts={role:cfg['streams'][str(seed)][role]['count'] for role in ROLES},
        selection='all explicitly declared noises; no Q-selection',evaluation_never_replayed=True,
        checkpoint_mutations=0,actor_updates=0,critic_updates=0,teacher_updates=0,backend_calls=0)


def verify_export(experiment,seed,cfg,declaration,physical):
    records,tasks=load_requests(experiment,seed,0)
    expected=[(role,i) for role in ROLES for i in range(cfg['streams'][str(seed)][role]['count'])]
    if [(r['role'],r['index']) for r in records]!=expected:
        raise ValueError('Bridge export omitted/reordered a declared role or noise index')
    for row in records:
        check_row(row,cfg,declaration,physical,experiment,seed,row['role'],row['index'])
    noises={role:{r['noise_sha256'] for r in records if r['role']==role} for role in ROLES}
    if any(len(noises[role])!=cfg['streams'][str(seed)][role]['count'] for role in ROLES) or noises['collect'] & noises['evaluation']:
        raise ValueError('Collect/evaluation actual-noise streams are duplicated or overlap')
    if read(tasks.with_name('completed.json'))!=completion(cfg,declaration,physical,experiment,seed,records,tasks):
        raise ValueError('Bridge completion marker does not seal the complete declared export')
    return records


def export(experiment,expected_config_sha256,seed):
    core.slurm()
    experiment=Path(experiment).resolve()
    cfg,declaration,physical=settings(experiment,expected_config_sha256)
    if seed not in SEEDS:
        raise ValueError('Undeclared independent actor seed')
    folder=experiment/f'seed_{seed}'/'round_0'/'requests'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'completed.json').exists():
            return verify_export(experiment,seed,cfg,declaration,physical)
        if not torch.cuda.is_available():
            raise RuntimeError('New layout export requires an allocated Slurm GPU')
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        parent=cfg['parent_identities'][str(seed)]['checkpoint']
        source=core.load_offline_source(cfg['source_kind'],cfg['source_experiment'],seed,parent['sha256'])
        if source['binding']!=cfg['verified_parents'][str(seed)] or source['schema']!=cfg['scalar_schema']:
            raise ValueError('Verified source parent or fixed utility changed before export')
        bind(RUNTIME/'benchmark_cache'/(cfg['design']+'.pt'),cfg['cache_sha256'])
        problem=load_problem(cfg['design'],'cuda')
        actor=GraphPolicy(problem)
        for name in ('movable','fixed'):
            if not torch.equal(getattr(actor,name).cpu(),source['payload']['actor'][name].cpu()):
                raise ValueError('Old actor fixed/movable buffers differ from the actual graph')
        actor.load_state_dict(source['payload']['actor'],strict=True);actor.eval()
        critic=source['refit']['model'].cuda().eval()
        metric=MacroHPWL(problem.benchmark_info)
        records=[]
        for role in ROLES:
            stream=cfg['streams'][str(seed)][role]
            generator=torch.Generator(device='cuda').manual_seed(stream['seed'])
            noises=2*torch.rand(stream['count'],len(problem.x),2,device='cuda',generator=generator)-1
            for index,noise in enumerate(noises):
                sample=folder/cfg['arm']/role/str(index);sample.mkdir(parents=True,exist_ok=True)
                path=sample/'request.json'
                if path.exists():
                    row=read(path)
                    check_row(row,cfg,declaration,physical,experiment,seed,role,index,noise.cpu().numpy())
                    records.append(row);continue
                with torch.no_grad():
                    raw=actor(noise[None])[0]
                    predicted=float(core.scalar_q(critic,raw[None])[0])
                    if not math.isfinite(predicted):
                        raise ValueError('Nonfinite source Q; no layout or score is selected/replaced')
                    action=legalize(raw,problem)
                a,r,n=(v.detach().cpu().numpy() for v in (action,raw,noise))
                geometry=legality(a,metric.sizes)
                row=dict(arm=cfg['arm'],seed=seed,round=0,role=role,index=index,graph=cfg['design'],
                    terminal_mask=0,training_feedback_eligible=role=='collect',noise_seed=stream['seed'],
                    raw_action=r.tolist(),executed_action=a.tolist(),predicted_q=[predicted],q_units=core.Q_UNITS,
                    source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],
                    source_config_sha256=cfg['source_config']['sha256'],
                    parent_identity_sha256=core.json_sha(cfg['verified_parents'][str(seed)]),
                    export_declaration_sha256=sha(experiment/'declaration.json'),
                    scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],
                    config_sha256=physical['config_sha256'],dataset_sha256=cfg['dataset_sha256'],
                    sample_dir=str(sample),run_id=f"go2o_qs1_{cfg['run_prefix']}_s{seed}_r0_{role}_{index}",
                    action_sha256=action_sha(a),raw_action_sha256=action_sha(r),noise_sha256=action_sha(n),
                    geometry=geometry,stage='global_route_only',macro_hpwl=float(metric(a)),
                    evaluation_purpose=cfg['evaluation_purpose'],objective_weights=cfg['scalar_schema']['utility_spec']['weights'])
                np.savez_compressed(sample/'actions.npz',noise=n,raw_action=r,action=a)
                if geometry['legal']:
                    save_sample(sample,action,noise,problem,row)
                    def_path=sample/(cfg['design']+'.def')
                    audit=convert(cfg['design'],sample/'macro_physical.json',def_path)
                    row.update(def_path=str(def_path),def_sha256=sha(def_path),rounded_geometry=audit['legal_no_halo'])
                    if not audit['legal_no_halo']:row['pre_backend_failure']='rounded_DEF_geometry'
                else:row['pre_backend_failure']='normalized_geometry'
                atomic_json(path,row);records.append(row)
        tasks=folder/'tasks.jsonl'
        content=''.join(json.dumps(r)+'\n' for r in records)
        temp=tasks.with_suffix('.jsonl.tmp');temp.write_text(content);temp.replace(tasks)
        immutable(folder/'completed.json',completion(cfg,declaration,physical,experiment,seed,records,tasks))
        return verify_export(experiment,seed,cfg,declaration,physical)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--request',type=Path,required=True)
    prep.add_argument('--request-sha256',required=True);prep.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('export');run.add_argument('--experiment',type=Path,required=True)
    run.add_argument('--config-sha256',required=True);run.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.request,args.request_sha256,args.output)
    else:export(args.experiment,args.config_sha256,args.seed)


if __name__=='__main__':main()
