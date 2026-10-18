"""Offline-only bounded scalar-Q adapter for the frozen actor train/export.

No production job is submitted here. Preparation verifies all nine bounded
screen tasks / 36 fits and emits three fixed objective configs. Training and
export reuse the exact frozen function code in an isolated globals namespace;
only setup/reward change. Round>0 is deliberately unsupported.
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

# Import native validation before the upstream adapter adds analysis.py paths.
from . import qstudy_backend as backend
from . import qstudy_actor as frozen
from . import qstudy_bounded_screen as screen
from .qstudy_bounded_utility import assert_common_scale, scale_identity, utility_identity, validate_spec
from .qstudy_branch import atomic_json, recover_journal_temporaries
from .qstudy_data import bind, load_dataset, sha
from .qstudy_models import MultiMetricCritic


ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'bounded_scalar_huber'
SCHEMA = 'qstudy_bounded_offline_actor_v1'
ARMS = dict(wl=('bounded_wl', 'bndwl'), timing=('bounded_timing', 'bndtim'),
            congestion=('bounded_congestion', 'bndcon'))
SOURCE_NAMES = tuple(dict.fromkeys((
    'fql/qstudy_bounded_actor.py', 'fql/qstudy_bounded_actor.slurm',
    'fql/qstudy_actor.py', 'fql/qstudy_backend.py', 'fql/qstudy_branch.py',
    'fql/graph_o2o_backend.py', 'fql/graph_o2o_geometry.py',
    'fql/graph_o2o_design_verify.py', 'backend/scripts/export_macro_def.py',
    *screen.SOURCE_NAMES)))
FIXED = dict(seeds=[0, 1, 2], alpha=10., q_weight=1., uncertainty_beta=0.,
    online_flow_update=False, offline_critic_update=False, offline_actor_steps=1000,
    evaluation_samples=8, evaluation_noise_seed=4600000, batch_size=16, flow_steps=10,
    actor_lr=1e-5, flow_lr=1e-5, critic_lr=3e-4, terminal_mask=0,
    physical_failure_reward=-4, backend_threads=8)


def read(path):
    return json.loads(Path(path).read_text())


def _slurm():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Bounded checkpoint/config work requires Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))


def _fixed(cfg):
    if any(cfg.get(key) != value for key, value in FIXED.items()):
        raise ValueError('Bounded actor fixed seeds/budgets/flow/noise/objective settings differ')


def source_binding():
    return {name:sha(ROOT/name) for name in SOURCE_NAMES}


def behavior_identity(data):
    action, families = np.asarray(data['behavior_action']), np.asarray(data['behavior_family'])
    if len(action) != 256 or not np.isfinite(action).all() or len(families) != 256:
        raise ValueError('Expected exactly 256 finite original BC targets and families')
    return dict(targets=256, shape=list(action.shape),
        action_sha256=screen.json_sha(action.tolist()), family_sha256=screen.json_sha(families.tolist()))


def _model(context, spec, seed, indices):
    target = screen.labels(context, indices, spec)
    if not bool(target['valid'].all()):
        raise ValueError('Initial bounded TRAIN labels must all be valid')
    x = context['x'][indices]
    return MultiMetricCritic(context['cache'], x, target['cost'].float()[:, None], context['movable'],
        ['bounded_cost'], train_ids=[context['source_ids'][i] for i in indices],
        train_splits=['train']*len(x), width=128, depth=2, ensemble_size=2,
        hpwl_residual=False, seed=seed)


def checked_model(context, spec, seed, indices, checkpoint, done):
    """Reconstruct TRAIN buffers before loading parameters; never fit on val."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = _model(context, spec, seed, indices)
    normalization = model.normalization_metadata()
    if (saved['binding'] != done['binding'] or saved['step'] != screen.CONFIG['steps'] or
        saved['utility_spec'] != spec or done['utility_spec'] != spec or
        saved['normalization'] != normalization or done['normalization'] != normalization or
        saved['spec'] != model.spec or saved['actor_contract'] != dict(
            q_units='-mean predicted B; no mean(B) renormalization', beta=0., native_reward_unchanged=True)):
        raise ValueError('Bounded checkpoint/spec/full TRAIN normalization differs')
    for name, value in model.named_buffers():
        actual = saved['model'].get(name)
        if not isinstance(actual, torch.Tensor) or actual.dtype != value.dtype or not torch.equal(actual, value):
            raise ValueError('Bounded TRAIN/cache normalization buffer differs: '+name)
    if saved['model'].get('_extra_state') != model.get_extra_state():
        raise ValueError('Bounded checkpoint TRAIN identity/spec extra state differs')
    model.load_state_dict(saved['model'], strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()
           if isinstance(v,torch.Tensor) and v.is_floating_point()):
        raise ValueError('Nonfinite bounded checkpoint')
    return model


def _context(plan):
    data, manifest, inputs = screen.load_inputs(Path(plan['inputs']['dataset_path']))
    if (inputs != plan['inputs'] or manifest['design'] != 'bp_fe' or
        len(data['train']['x']) != 384 or len(data['validation']['x']) != 16 or
        not (data['train']['statuses'] == 'success').all() or
        not np.isfinite(data['train']['native']).all() or
        not bool(torch.isfinite(data['train']['x']).all()) or
        set(data['train']['families']) & set(data['validation']['families'])):
        raise ValueError('Bounded actor requires the unchanged finite bp_fe384 TRAIN /16 historical validation')
    from baseline.api import load_problem
    cache = torch.load(inputs['cache_path'], map_location='cpu', weights_only=False)
    movable = ~load_problem('bp_fe', 'cpu').cond.is_ports.bool()
    return data, dict(**data['train'], cache=cache, movable=movable,
                     binding=dict(dataset_sha256=inputs['dataset_sha256']))


def checked_study(plan_path):
    """Verify nine completed tasks and every original fold/checkpoint/prediction."""
    plan = read(plan_path)
    tasks = [dict(index=i, objective=o, seed=s) for i,(o,s) in
             enumerate((o,s) for o in screen.OBJECTIVES for s in range(3))]
    if (plan['schema'] != 'qstudy_bounded_scalar_screen_v1' or plan['tasks'] != tasks or
        plan['config'] != screen.CONFIG or plan['objectives'] != {k:list(v) for k,v in screen.OBJECTIVES.items()} or
        plan['source_sha256'] != {name:sha(ROOT/name) for name in screen.SOURCE_NAMES} or
        plan['native_reward_unchanged'] is not True or plan['actor_advanced'] is not False or
        plan['heldout_used'] is not False or plan['backend_calls'] != 0):
        raise ValueError('Bounded study declaration/source/config differs')
    evidence = {str(Path(plan_path).resolve()):sha(plan_path)}
    for name, digest in plan['source_sha256'].items():
        snapshot = Path(plan['output'])/'source_snapshot'/name
        bind(snapshot, digest)
        evidence[str(snapshot)] = digest
    data, context = _context(plan)
    folds = screen.family_folds(context['families'], 3)
    folds.append(dict(name='refit', train_indices=list(range(384)), target_indices=list(range(16))))
    if (plan['folds'] != folds or plan['fold_plan_sha256'] != screen.json_sha(folds) or
        plan['train_source_ids'] != context['source_ids'] or
        plan['train_source_ids_sha256'] != screen.json_sha(context['source_ids']) or
        plan['validation_source_ids'] != data['validation']['source_ids']):
        raise ValueError('Bounded grouped TRAIN/fold/source identities differ')
    selected, specs = {}, {}
    for task in tasks:
        folder = Path(plan['output'])/task['objective']/f"seed_{task['seed']}"
        done, summary = read(folder/'completed.json'), read(folder/'summary.json')
        run = done['binding']
        expected_run = dict(schema=plan['schema'], **plan['inputs'], plan_sha256=sha(plan_path), task=task,
            source_sha256=plan['source_sha256'], fold_plan_sha256=plan['fold_plan_sha256'],
            threads=run['threads'], torch_version=torch.__version__, numpy_version=np.__version__)
        if (run != expected_run or not isinstance(run['threads'], int) or run['threads'] < 1 or
            done['status'] != 'completed' or done['summary_sha256'] != sha(folder/'summary.json') or
            done['backend_calls'] != 0 or done['actor_advanced'] is not False or
            summary['status'] != 'completed' or summary['binding'] != run or
            summary['objective'] != task['objective'] or summary['seed'] != task['seed'] or len(summary['outcomes']) != 4):
            raise ValueError('Bounded task is not a complete bound four-fit screen')
        for path in (folder/'completed.json', folder/'summary.json'):
            evidence[str(path)] = sha(path)
        for j, fold in enumerate(folds):
            fit = folder/fold['name']
            completed = read(fit/'completed.json')
            spec, audit = screen.specification(context, fold['train_indices'], task['objective'])
            target_data = data['validation'] if fold['name'] == 'refit' else context
            binding = dict(run=run, fit=fold['name'], objective=task['objective'], utility_spec=spec,
                utility_scale_audit=audit, config={key:value for key,value in dict(seed=task['seed'], **screen.CONFIG).items()
                    if key in ['seed','steps','batch_size','lr','weight_decay']},
                train_indices=fold['train_indices'], admitted_train_indices=fold['train_indices'],
                target_indices=fold['target_indices'],
                train_source_ids=[context['source_ids'][i] for i in fold['train_indices']],
                target_source_ids=[target_data['source_ids'][i] for i in fold['target_indices']],
                q_units='-mean predicted B; no mean(B) renormalization', beta=0.)
            if (completed['binding'] != binding or completed != summary['outcomes'][j] or
                completed['status'] != 'completed' or completed['steps'] != screen.CONFIG['steps'] or
                completed['backend_calls'] != 0 or completed['actor_advanced'] is not False or
                completed['heldout_used'] is not False):
                raise ValueError('Bounded fit did not use the exact predeclared TRAIN/target rows')
            for name, key in [('resume.pt','checkpoint_sha256'), ('predictions.npz','predictions_sha256')]:
                bind(fit/name, completed[key])
                evidence[str(fit/name)] = completed[key]
            evidence[str(fit/'completed.json')] = sha(fit/'completed.json')
            model = checked_model(context, spec, task['seed'], fold['train_indices'], fit/'resume.pt', completed)
            with np.load(fit/'predictions.npz', allow_pickle=False) as prediction:
                truth = screen.labels(target_data, fold['target_indices'], spec)
                for key, value in dict(train_indices=fold['train_indices'], target_indices=fold['target_indices'],
                    train_source_ids=binding['train_source_ids'], target_source_ids=binding['target_source_ids'],
                    raw_action=target_data['x'][fold['target_indices']].numpy(),
                    native_metrics=target_data['native'][fold['target_indices']],
                    valid=truth['valid'].numpy(), truth=truth['cost'].numpy()).items():
                    if not np.array_equal(prediction[key], value, equal_nan=np.asarray(value).dtype.kind in 'fc'):
                        raise ValueError('Bounded saved target/row provenance differs: '+key)
            if fold['name'] == 'refit':
                specs[task['objective']] = spec
                selected[(task['objective'], task['seed'])] = dict(checkpoint=bind(fit/'resume.pt'),
                    completion=bind(fit/'completed.json'), normalization=model.normalization_metadata())
    for spec in specs.values():
        assert_common_scale(specs['wl'], spec)
    return plan, data, context, selected, specs, evidence


def prepare(args):
    _slurm()
    plan_path, base_path, output = (Path(p).resolve() for p in (args.plan, args.base_config, args.output))
    base = read(base_path)
    _fixed(base)
    plan, _, _, selected, specs, evidence = checked_study(plan_path)
    if base['design'] != plan['design'] or base['cache_sha256'] != plan['inputs']['cache_sha256']:
        raise ValueError('Base actor design/cache differs')
    original, _ = load_dataset(Path(base['dataset_path']))
    data, _ = load_dataset(Path(plan['inputs']['dataset_path']))
    bind(base['dataset_path'], base['dataset_sha256'])
    identity = behavior_identity(original)
    if identity != behavior_identity(data) or any(not np.array_equal(data[k], original[k])
            for k in ['behavior_action', 'behavior_family']):
        raise ValueError('Original 256 BC targets/families differ')
    common = {}
    for seed in base['seeds']:
        key = str(seed)
        path = Path(base['common_checkpoints'][key])
        bound = bind(path, base['common_checkpoint_sha256'][key])
        status = read(path.with_name('training_status.json'))
        if (status['status'] != 'completed' or status['seed'] != seed or status['round'] != 0 or
            status['graph_cache_sha256'] != base['cache_sha256'] or status['no_heldout_labels_used'] is not True):
            raise ValueError('Common checkpoint metadata/cache/seed differs')
        common[key] = dict(checkpoint=bound, metadata=bind(path.with_name('training_status.json')),
                          original_config=bind(path.parents[2]/'config.json', status['config_sha256']))
    if len({v['checkpoint']['sha256'] for v in common.values()}) != 3:
        raise ValueError('Common checkpoints must represent three independent initializations')
    sources = source_binding()
    configs = {}
    for objective, (arm, prefix) in ARMS.items():
        cfg = copy.deepcopy(base)
        cfg.update(bounded_actor_schema=SCHEMA, arm=arm, run_prefix=prefix, critic_variant=VARIANT,
            bounded_objective=objective, weights=list(screen.OBJECTIVES[objective]), online_rounds=0,
            dataset_path=plan['inputs']['dataset_path'], dataset_sha256=plan['inputs']['dataset_sha256'],
            bounded_utility_spec=specs[objective], bounded_utility_identity=utility_identity(specs[objective]),
            bounded_scale_identity=scale_identity(specs[objective]), bounded_source_sha256=sources,
            bounded_plan=bind(plan_path), bounded_behavior=identity, bounded_common=common,
            bounded_preparation_root=str(output),
            critic_checkpoints={str(s):selected[(objective,s)]['checkpoint']['path'] for s in base['seeds']},
            critic_checkpoint_sha256={str(s):selected[(objective,s)]['checkpoint']['sha256'] for s in base['seeds']},
            bounded_refits={str(s):selected[(objective,s)] for s in base['seeds']},
            note='Prospective bounded scalar Q=-mean(B), no second normalization; offline only; native WL reward unchanged.')
        configs[arm] = cfg
    protected = [Path(plan['output']), base_path.parent, Path(plan['inputs']['dataset_path']).parent,
        Path(plan['inputs']['cache_path']).parent, *[Path(v['checkpoint']['path']).parent for v in common.values()]]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Config destination must be disjoint from preserved source artifacts')
    manifest = dict(schema=SCHEMA, status='prepared_configs_only', bounded_plan=bind(plan_path),
        base_config=bind(base_path), source_sha256=sources, evidence=evidence, common=common, behavior=identity,
        tasks=9, critic_fits=36, actor_jobs_submitted=0, backend_calls=0, online_implemented=False,
        heldout_used=False, automatic_selection=False, native_reward_unchanged=True)
    proposal = dict(manifest=manifest, configs=configs)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        recover_journal_temporaries(output, ('preparation_plan.json', 'preparation.json'))
        allowed = set(configs) | {'.prepare.lock', 'preparation_plan.json', 'preparation.json'}
        if any(p.name not in allowed or p.is_symlink() for p in output.iterdir()):
            raise ValueError('Foreign/symlink artifact in bounded config destination')
        _immutable(output/'preparation_plan.json', proposal)
        for path,digest in evidence.items():
            bind(path,digest)
        for arm,cfg in configs.items():
            folder = output/arm
            folder.mkdir(exist_ok=True)
            recover_journal_temporaries(folder, ('config.json',))
            _immutable(folder/'config.json', cfg)
        manifest['configs'] = {arm:bind(output/arm/'config.json') for arm in configs}
        _immutable(output/'preparation.json', manifest)
    print(json.dumps(dict(status='prepared_configs_only', manifest=str(output/'preparation.json'),
                          actor_jobs_submitted=0, backend_calls=0)), flush=True)
    return manifest


def _immutable(path, value):
    if path.is_symlink() or (path.exists() and read(path) != value):
        raise ValueError('Immutable bounded actor artifact differs: '+str(path))
    if not path.exists():
        atomic_json(path, value)


def validate_config(experiment, seed, round_number=0):
    if round_number != 0:
        raise ValueError('Bounded actor online rounds are not implemented')
    cfg = read(experiment/'config.json')
    _fixed(cfg)
    if (cfg.get('bounded_actor_schema') != SCHEMA or cfg['critic_variant'] != VARIANT or
        cfg.get('online_rounds') != 0 or seed not in cfg['seeds'] or
        cfg['bounded_source_sha256'] != source_binding()):
        raise ValueError('Bounded actor schema/source/round/seed differs')
    objective = cfg['bounded_objective']
    if objective not in ARMS or (cfg['arm'], cfg['run_prefix']) != ARMS[objective]:
        raise ValueError('Unknown bounded objective/arm identity')
    spec = validate_spec(cfg['bounded_utility_spec'])
    if (spec['weights'] != cfg['weights'] or cfg['weights'] != list(screen.OBJECTIVES[objective]) or
        utility_identity(spec) != cfg['bounded_utility_identity'] or scale_identity(spec) != cfg['bounded_scale_identity']):
        raise ValueError('Bounded utility spec/identity differs')
    preparation = read(Path(cfg['bounded_preparation_root'])/'preparation.json')
    if (preparation['schema'] != SCHEMA or preparation['status'] != 'prepared_configs_only' or
        preparation['configs'][cfg['arm']] != bind(experiment/'config.json') or
        preparation['source_sha256'] != cfg['bounded_source_sha256'] or preparation['online_implemented'] is not False):
        raise ValueError('Bounded configuration publication gate differs')
    for name,bound in cfg['bounded_common'][str(seed)].items():
        bind(bound['path'],bound['sha256'])
    bind(cfg['bounded_plan']['path'], cfg['bounded_plan']['sha256'])
    return cfg


def setup(experiment, seed):
    _slurm()
    if not torch.cuda.is_available():
        raise RuntimeError('Bounded policy training/export requires a Slurm GPU')
    cfg = validate_config(experiment, seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    plan = read(cfg['bounded_plan']['path'])
    data, context = _context(plan)
    if cfg['dataset_path'] != plan['inputs']['dataset_path'] or cfg['dataset_sha256'] != plan['inputs']['dataset_sha256']:
        raise ValueError('Bounded runtime dataset differs from verified full TRAIN')
    spec,_ = screen.specification(context,list(range(384)),cfg['bounded_objective'])
    if spec != cfg['bounded_utility_spec']:
        raise ValueError('Bounded runtime common TRAIN scale/spec differs')
    path = Path(cfg['critic_checkpoints'][str(seed)])
    evidence = cfg['bounded_refits'][str(seed)]
    if bind(path,cfg['critic_checkpoint_sha256'][str(seed)]) != evidence['checkpoint']:
        raise ValueError('Bounded full TRAIN refit checkpoint differs')
    bind(evidence['completion']['path'], evidence['completion']['sha256'])
    done = read(evidence['completion']['path'])
    model = checked_model(context,spec,seed,list(range(384)),path,done)
    if model.normalization_metadata() != evidence['normalization']:
        raise ValueError('Bounded prepared normalization differs')
    original,_ = load_dataset(Path(cfg['dataset_path']))
    if behavior_identity(original) != cfg['bounded_behavior']:
        raise ValueError('Runtime original 256 BC targets changed')
    from baseline.api import load_problem
    problem = load_problem(cfg['design'],'cuda')
    x = context['x'].cuda()
    targets = screen.labels(context,list(range(384)),spec)
    costs = targets['cost'].float().cuda()[:,None]
    mask = targets['valid'].cuda()[:,None]
    behavior = torch.as_tensor(original['behavior_action'],device='cuda').reshape(-1,len(problem.x),2)
    families = torch.as_tensor(context['families'],device='cuda')
    return cfg,problem,x,costs,mask,behavior,model.cuda(),families


def reward(critic, actions, _costs, _cfg):
    """Q has fixed bounded utility units; never call model.reward()."""
    return -critic(actions).mean((1,2))


def isolated_runtime():
    """Keep audited function code and isolate STOP/setup/reward per process."""
    namespace = dict(vars(frozen))
    namespace.update(setup=setup,reward=reward,STOP=False)
    for name in ('stop','train','export'):
        original = getattr(frozen,name)
        namespace[name] = FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    return namespace


def run(args):
    _slurm()
    args.experiment = args.experiment.resolve()
    validate_config(args.experiment,args.seed,args.round)
    namespace = isolated_runtime()
    for name in (['train','export'] if args.stage == 'all' else [args.stage]):
        namespace[name](args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('prepare')
    for name in ('plan','base-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p = commands.add_parser('run')
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--round',type=int,default=0)
    p.add_argument('--stage',choices=['train','export','all'],default='all')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
