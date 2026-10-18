"""Reward-only offline FQL contrasts using the unchanged one-step actor.

Prepare only after the fixed rule-reward screen completes. No automatic job
submission, online replay, policy selection, or change to physical execution.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

# Resolve the local analysis package before importing the upstream adapter.
from . import qstudy_backend as backend
from . import qstudy_actor as frozen
from . import qstudy_rule_reward_screen as screen
from .qstudy_branch import atomic_json
from .qstudy_data import bind, load_dataset, sha

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'qstudy_rule_reward_offline_actor_v1'
OBJECTIVES = ('grt_wl', 'macro_hpwl', 'hpwl_edge', 'hpwl_group',
              'hpwl_dataflow', 'hpwl_channel', 'hpwl_expert_composite',
              'grt_expert_composite')
ARMS = {name: ('rule_'+name, 'rrv1'+str(i)) for i, name in enumerate(OBJECTIVES)}
FIXED = dict(seeds=[0,1,2], alpha=10., q_weight=1., uncertainty_beta=0.,
    online_flow_update=False, offline_critic_update=False, offline_actor_steps=1000,
    evaluation_samples=8, evaluation_noise_seed=4600000, batch_size=16,
    flow_steps=10, actor_lr=1e-5, flow_lr=1e-5, critic_lr=3e-4,
    terminal_mask=0, physical_failure_reward=-4, backend_threads=8)


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path = Path(path)
    if path.is_symlink() or (path.exists() and read(path) != value):
        raise ValueError('Changed immutable rule-actor artifact: '+str(path))
    if not path.exists():
        atomic_json(path, value)


def slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a bounded Slurm allocation')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))


def sources():
    names = tuple(dict.fromkeys((*screen.SOURCE_NAMES,
        'fql/qstudy_rule_reward_actor.py', 'fql/qstudy_rule_reward_actor.slurm',
        'fql/qstudy_actor.py', 'fql/qstudy_backend.py', 'fql/qstudy_branch.py',
        'fql/graph_o2o_geometry.py', 'backend/scripts/export_macro_def.py',
        'docs/qstudy_rule_reward_protocol.md')))
    return {name:sha(ROOT/name) for name in names}


def fixed(cfg):
    if any(cfg.get(k) != v for k,v in FIXED.items()):
        raise ValueError('Rule-only contrast changed policy budget/network/objective contract')


def behavior_identity(data):
    actions = np.asarray(data['behavior_action'])
    families = np.asarray(data['behavior_family'])
    if (actions.shape != (256,22) or families.shape != (256,) or
            not np.isfinite(actions).all()):
        raise ValueError('Expected the exact original 256 BC targets, distinct from measured384')
    from .qstudy_bounded_screen import json_sha
    return dict(shape=list(actions.shape), targets=256,
        actions_sha256=json_sha(actions.tolist()), families_sha256=json_sha(families.tolist()))


def common_sources(base):
    result = {}
    for seed in FIXED['seeds']:
        key = str(seed)
        path = Path(base['common_checkpoints'][key])
        checkpoint = bind(path, base['common_checkpoint_sha256'][key])
        metadata = path.with_name('training_status.json')
        status = read(metadata)
        if (status.get('status') != 'completed' or status.get('seed') != seed or
                status.get('round') != 0 or status.get('graph_cache_sha256') != base['cache_sha256'] or
                status.get('no_heldout_labels_used') is not True):
            raise ValueError('Common teacher/actor initialization differs')
        result[key] = dict(checkpoint=checkpoint, metadata=bind(metadata),
            original_config=bind(path.parents[2]/'config.json', status['config_sha256']))
    if len({v['checkpoint']['sha256'] for v in result.values()}) != 3:
        raise ValueError('The three policy initializations must be independent')
    return result


def prepare(args):
    slurm()
    plan_path, base_path, output = (Path(v).resolve() for v in
                                    (args.plan,args.base_config,args.output))
    base = read(base_path)
    fixed(base)
    if base['design'] != 'bp_fe' or base['backend_storage'] != 'scratch':
        raise ValueError('Only the existing scratch bp_fe physical flow is supported')
    original,_ = load_dataset(Path(base['dataset_path']))
    bind(base['dataset_path'],base['dataset_sha256'])
    behavior = behavior_identity(original)
    common = common_sources(base)
    source_hashes = sources()
    evidence = {str(base_path):sha(base_path),str(plan_path):sha(plan_path)}
    selected = {}
    loaded = screen.checked_plan(plan_path)
    if (loaded[0]['inputs']['dataset_path'] != base['dataset_path'] or
            loaded[0]['inputs']['dataset_sha256'] != base['dataset_sha256'] or
            loaded[0]['inputs']['cache_sha256'] != base['cache_sha256']):
        raise ValueError('Rule screen and common policy use different dataset/cache inputs')
    # Fixed full matrix, never select refits or weights by CV/development scores.
    for objective in OBJECTIVES:
        for seed in FIXED['seeds']:
            result = screen.checked_refit(plan_path,objective,seed,loaded=loaded)
            context = result['context']
            if (len(context['x']) != 384 or len(set(context['families'])) != 192 or
                    context['binding']['dataset_sha256'] != base['dataset_sha256']):
                raise ValueError('Rule critic TRAIN dataset/denominator differs')
            if result['normalization'] != result['model'].normalization_metadata():
                raise ValueError('Rule critic normalizer differs')
            evidence.update(result['evidence'])
            selected[(objective,seed)] = {k:result[k] for k in
                ('spec','checkpoint','completion','normalization')}
    for path in (Path(base['dataset_path']).parent,Path(base['common_checkpoints']['0']).parent,
                 plan_path.parent,base_path.parent):
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('New configs must be disjoint from original sources')
    configs = {}
    for objective in OBJECTIVES:
        arm,prefix = ARMS[objective]
        specs = [selected[(objective,s)]['spec'] for s in FIXED['seeds']]
        if any(spec != specs[0] for spec in specs):
            raise ValueError('Reward definition or TRAIN scales depend on model seed')
        cfg = copy.deepcopy(base)
        cfg.update(rule_actor_schema=SCHEMA,arm=arm,run_prefix=prefix,
            critic_variant='rule_scalar_huber',rule_objective=objective,
            weights=specs[0]['weights'],rule_utility_spec=specs[0],
            rule_plan=bind(plan_path),rule_sources=source_hashes,rule_common=common,
            rule_refits={str(s):selected[(objective,s)] for s in FIXED['seeds']},
            rule_behavior=behavior,rule_preparation_root=str(output),online_rounds=0,
            critic_checkpoints={str(s):selected[(objective,s)]['checkpoint']['path'] for s in FIXED['seeds']},
            critic_checkpoint_sha256={str(s):selected[(objective,s)]['checkpoint']['sha256'] for s in FIXED['seeds']},
            note='Fixed reward-only FQL contrast; same behavior flow, actor, streams and final legalizer; no online feedback.')
        configs[arm] = cfg
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,digest in evidence.items():
            bind(path,digest)
        for arm,cfg in configs.items():
            (output/arm).mkdir(exist_ok=True)
            immutable(output/arm/'config.json',cfg)
        proposal = dict(schema=SCHEMA,status='prepared_configs_only',plan=bind(plan_path),
            base=bind(base_path),source_sha256=source_hashes,evidence=evidence,
            common=common,behavior=behavior,objectives=list(OBJECTIVES),
            configs={arm:bind(output/arm/'config.json') for arm in configs},
            automatic_selection=False,online_implemented=False,backend_calls=0,actor_jobs_submitted=0)
        immutable(output/'preparation.json',proposal)
    print(json.dumps(dict(status=proposal['status'],preparation=str(output/'preparation.json'),
                         arms=8,training_seeds=3,backend_calls=0)),flush=True)


def validate_config(experiment,seed,round_number=0):
    if round_number != 0:
        raise ValueError('Expert-reward first contrast is offline only; online requires its own protocol')
    cfg = read(experiment/'config.json')
    fixed(cfg)
    if (cfg.get('rule_actor_schema') != SCHEMA or cfg.get('online_rounds') != 0 or
            cfg.get('critic_variant') != 'rule_scalar_huber' or seed not in FIXED['seeds'] or
            cfg.get('rule_sources') != sources()):
        raise ValueError('Rule actor source/schema/round changed')
    objective = cfg['rule_objective']
    if objective not in ARMS or (cfg['arm'],cfg['run_prefix']) != ARMS[objective]:
        raise ValueError('Unknown rule arm')
    if cfg['weights'] != cfg['rule_utility_spec']['weights']:
        raise ValueError('Exported objective weights differ from actual scalar reward')
    preparation = read(Path(cfg['rule_preparation_root'])/'preparation.json')
    if (preparation.get('schema') != SCHEMA or preparation.get('status') != 'prepared_configs_only' or
            preparation['source_sha256'] != cfg['rule_sources'] or
            preparation['configs'][cfg['arm']] != bind(experiment/'config.json') or
            preparation['online_implemented'] is not False):
        raise ValueError('Rule actor configuration publication gate changed')
    bind(cfg['rule_plan']['path'],cfg['rule_plan']['sha256'])
    for artifact in cfg['rule_common'][str(seed)].values():
        bind(artifact['path'],artifact['sha256'])
    return cfg


def setup(experiment,seed):
    slurm()
    if not torch.cuda.is_available():
        raise RuntimeError('Actor training/export needs a Slurm GPU')
    cfg = validate_config(experiment,seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    result = screen.checked_refit(Path(cfg['rule_plan']['path']),cfg['rule_objective'],seed)
    for key in ('spec','checkpoint','completion','normalization'):
        if result[key] != cfg['rule_refits'][str(seed)][key]:
            raise ValueError('Prepared full-TRAIN critic changed: '+key)
    if result['spec'] != cfg['rule_utility_spec']:
        raise ValueError('Runtime rule target/scales differ')
    context = result['context']
    original,_ = load_dataset(Path(cfg['dataset_path']))
    if behavior_identity(original) != cfg['rule_behavior']:
        raise ValueError('Behavior targets changed')
    from baseline.api import load_problem
    problem = load_problem(cfg['design'],'cuda')
    labels = screen.labels(context,list(range(384)),result['spec'])
    behavior = torch.as_tensor(original['behavior_action'],device='cuda').reshape(-1,len(problem.x),2)
    return (cfg,problem,context['x'].cuda(),labels['cost'].float().cuda()[:,None],
            labels['valid'].cuda()[:,None],behavior,result['model'].cuda(),
            torch.as_tensor(context['families'],device='cuda'))


def reward(critic,actions,_costs,_cfg):
    return screen.scalar_reward(critic,actions)


def isolated_runtime():
    namespace = dict(vars(frozen))
    namespace.update(setup=setup,reward=reward,STOP=False)
    for name in ('stop','train','export'):
        original = getattr(frozen,name)
        namespace[name] = FunctionType(original.__code__,namespace,original.__name__,
                                     original.__defaults__,original.__closure__)
    return namespace


def run(args):
    slurm()
    args.experiment = args.experiment.resolve()
    validate_config(args.experiment,args.seed,args.round)
    namespace = isolated_runtime()
    for stage in (['train','export'] if args.stage == 'all' else [args.stage]):
        namespace[stage](args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    for name in ('plan','base-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p = sub.add_parser('run')
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--round',type=int,default=0)
    p.add_argument('--stage',choices=['train','export','all'],default='all')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
