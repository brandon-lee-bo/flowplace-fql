"""Fixed eight-objective expert-rule critic screen; no actor or physical calls.

Rule labels describe the saved executed geometry; critic inputs are the raw
proposals. All cost scales and model buffers are fitted on fold TRAIN only.
Training reuses the frozen bounded-screen fit bytecode in private globals,
replacing only specification/label construction. This is not EXPlace's exact
sequential reward and does not interpret dataflow weights as STA criticality.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
from types import FunctionType

import numpy as np
import torch

# Resolve the local analysis namespace before any upstream loader mutates paths.
from analysis.placement_metrics import MacroHPWL
from analysis.qstudy_rule_audit import DATASET_SHA, RULES, rules, structure
from . import qstudy_bounded_screen as bounded
from .qstudy_data import METRIC_NAMES, action_sha, bind, sha
from .qstudy_models import MultiMetricCritic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = bounded.DEFAULT_DATA
SCHEMA = 'qstudy_rule_reward_screen_v1'
SPEC_VERSION = 'qstudy_expert_rule_bounded_utility_v1'
COST_NAMES = ('grt_wl', 'macro_hpwl', 'edge_clearance', 'cluster_bbox_fraction',
              'dataflow_distance', 'narrow_channel_risk')
LAMBDA = .25
OBJECTIVES = dict(
    grt_wl=(1., 0., 0., 0., 0., 0.),
    macro_hpwl=(0., 1., 0., 0., 0., 0.),
    hpwl_edge=(0., .8, .2, 0., 0., 0.),
    hpwl_group=(0., .8, 0., .2, 0., 0.),
    hpwl_dataflow=(0., .8, 0., 0., .2, 0.),
    hpwl_channel=(0., .8, 0., 0., 0., .2),
    hpwl_expert_composite=(0., .8, .05, .05, .05, .05),
    grt_expert_composite=(.8, 0., .05, .05, .05, .05))
CONFIG = dict(bounded.CONFIG)
SOURCE_NAMES = tuple(dict.fromkeys((
    'fql/qstudy_rule_reward_screen.py', 'fql/qstudy_rule_reward_screen.slurm',
    'docs/qstudy_rule_reward_protocol.md', 'analysis/qstudy_rule_audit.py',
    'analysis/placement_metrics.py', *bounded.SOURCE_NAMES)))
Q_UNITS = '-mean predicted B; no mean(B) renormalization'
ACTOR_CONTRACT = dict(q_units=Q_UNITS, beta=0., native_reward_unchanged=True)
STOP = False
_ACTIVE_FIT_GLOBALS = []
json_sha = bounded.json_sha
family_folds = bounded.family_folds
scalar_reward = bounded.scalar_reward
summarize = bounded.summarize


def read(path):
    return json.loads(Path(path).read_text())


def _slurm():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Rule reward numerical work requires CPU Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))


def _stop(*_):
    global STOP
    STOP = True
    for namespace in _ACTIVE_FIT_GLOBALS:
        namespace['STOP'] = True


def rule_definition():
    return dict(coordinates='executed_action in cache node order', critic_input='raw_action',
        cost_names=list(COST_NAMES), units=['um', RULES['macro_hpwl']['unit'],
            *[RULES[name]['unit'] for name in COST_NAMES[2:]]],
        rules={name:RULES[name] for name in COST_NAMES[1:]},
        narrow_channel_tau='2/127 of each [0,1] core axis; two 128-lattice intervals',
        excluded=['I/O obstruction', 'corner clearance', 'validation-derived weights',
                  'STA criticality', 'nonoverlap/bounds as soft rewards'],
        expert_lambda=LAMBDA, composite='equal mean of four individually TRAIN-mean-normalized expert costs',
        hard_constraints='Execution legality is checked separately; raw-to-executed map is not differentiated')


def add_costs(data, cache):
    """Compute fixed proxies on saved executed coordinates, never predictions."""
    ctx, _ = structure(cache)
    hpwl = MacroHPWL(cache)
    for split in ('train', 'validation'):
        source = data[split]
        n = len(source['x'])
        layouts = np.asarray(source['executed']).reshape(n, len(ctx['names']), 2)
        if not np.isfinite(layouts).all():
            raise ValueError('Nonfinite executed coordinates cannot supply rule labels')
        scores = [rules(layout, ctx, hpwl) for layout in layouts]
        values = np.full((n, len(COST_NAMES)), np.nan, dtype=np.float64)
        values[:, 0] = np.where(source['mask'][:, 0], source['native'][:, 0], np.nan)
        for k, name in enumerate(COST_NAMES[1:], 1):
            values[:, k] = [np.nan if score[name] is None else score[name] for score in scores]
        source['cost_values'] = values
        source['cost_mask'] = np.isfinite(values)
    return data


def load_inputs(dataset):
    """Bounded-screen-compatible arrays plus six executed/native cost columns."""
    dataset = Path(dataset).resolve()
    data, manifest, inputs = bounded.load_inputs(dataset)
    if (inputs['dataset_sha256'] != DATASET_SHA or manifest['design'] != 'bp_fe' or
        len(data['train']['x']) != 384 or len(data['validation']['x']) != 16 or
        set(data['train']['families']) & set(data['validation']['families'])):
        raise ValueError('Rule study requires the exact audited bp_fe384/16 dataset and families')
    records = [json.loads(line) for line in dataset.with_name('rows.jsonl').read_text().splitlines() if line]
    for split in ('train', 'validation'):
        rows = [row for row in records if row['split'] == split]
        source = data[split]
        if len(rows) != len(source['x']):
            raise ValueError('Rule study row provenance denominator changed')
        for i, row in enumerate(rows):
            native = np.asarray([np.nan if row['metrics'][name] is None else row['metrics'][name]
                                 for name in METRIC_NAMES], dtype=float)
            if (row['family'] != int(source['families'][i]) or row['status'] != source['statuses'][i] or
                row['raw_action_sha256'] != action_sha(source['x'][i].numpy()) or
                row['action_sha256'] != action_sha(source['executed'][i]) or
                not np.array_equal(native, source['native'][i], equal_nan=True)):
                raise ValueError('Rule raw/executed/native provenance correspondence changed')
    cache = torch.load(inputs['cache_path'], map_location='cpu', weights_only=False)
    add_costs(data, cache)
    # These are input identities, not fitted statistics or selected objectives.
    inputs.update(rule_definition_sha256=json_sha(rule_definition()),
        train_cost_values_sha256=json_sha(data['train']['cost_values'].tolist()),
        validation_cost_values_sha256=json_sha(data['validation']['cost_values'].tolist()))
    return data, manifest, inputs


def fit_scales(values, statuses, mask):
    """Success-only TRAIN means and variation, independent of the objective."""
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(mask, bool) & np.isfinite(values)
    if values.ndim != 2 or values.shape[1] != len(COST_NAMES) or observed.shape != values.shape:
        raise ValueError('Rule cost values/mask must be rows by six costs')
    observed &= (np.asarray(statuses) == 'success')[:, None]
    if ((values < 0) & observed).any():
        raise ValueError('Observed rule costs must be nonnegative')
    count = observed.sum(0)
    means = np.where(observed, values, 0.).sum(0) / np.maximum(count, 1)
    span = np.asarray([np.ptp(values[observed[:, k], k]) if count[k] else 0.
                       for k in range(len(COST_NAMES))])
    # Inactive zero columns retain a structural placeholder only. Required
    # unavailable, zero-mean or constant columns are refused by specification.
    scales = np.where(means > 1e-8, means, 1.)
    return scales.tolist(), dict(observed_counts=count.tolist(), means=means.tolist(),
        train_span=span.tolist(), inadmissible=( (count < 2) | (means <= 1e-8) | (span <= 1e-12)).tolist(),
        policy='fold TRAIN success-only mean; any active missing/zero-mean/constant term refuses the objective')


def validate_spec(spec):
    required = {'version', 'objective', 'cost_names', 'weights', 'scales', 'reference',
                'rule_definition_sha256', 'success_cap', 'failure_cost'}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError('Rule utility specification keys differ')
    objective = spec['objective']
    if (spec['version'] != SPEC_VERSION or objective not in OBJECTIVES or
        spec['cost_names'] != list(COST_NAMES) or spec['weights'] != list(OBJECTIVES[objective]) or
        spec['success_cap'] != 2. or spec['failure_cost'] != 4. or
        spec['rule_definition_sha256'] != json_sha(rule_definition())):
        raise ValueError('Fixed rule objective semantics differ')
    scales = np.asarray(spec['scales'], dtype=np.float64)
    if scales.shape != (len(COST_NAMES),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Rule scales must be six positive finite TRAIN means')
    reference = spec['reference']
    if not isinstance(reference, dict) or set(reference) != {'split', 'dataset_sha256', 'train_rows_sha256'}:
        raise ValueError('Rule utility requires exact TRAIN row reference')
    if reference['split'] != 'train' or any(not isinstance(reference[k], str) or len(reference[k]) != 64 or
        any(c not in '0123456789abcdef' for c in reference[k]) for k in ('dataset_sha256', 'train_rows_sha256')):
        raise ValueError('Rule utility reference is not explicit TRAIN SHA identity')
    return spec


def scale_identity(spec):
    spec = validate_spec(spec)
    return json_sha({k:spec[k] for k in ('version', 'cost_names', 'scales', 'reference', 'rule_definition_sha256')})


def utility_identity(spec):
    return json_sha(validate_spec(spec))


def specification(context, train_indices, objective):
    if objective not in OBJECTIVES:
        raise ValueError('Objective was not predeclared')
    indices = np.asarray(train_indices, dtype=int)
    if indices.ndim != 1 or len(set(indices.tolist())) != len(indices):
        raise ValueError('Scale rows must be distinct TRAIN indices')
    scales, audit = fit_scales(context['cost_values'][indices], context['statuses'][indices],
                               context['cost_mask'][indices])
    active = np.asarray(OBJECTIVES[objective]) > 0
    bad = [name for name, a, invalid in zip(COST_NAMES, active, audit['inadmissible']) if a and invalid]
    if bad:
        raise ValueError('Required rule objective terms unavailable or constant: '+','.join(bad))
    spec = dict(version=SPEC_VERSION, objective=objective, cost_names=list(COST_NAMES),
        weights=list(OBJECTIVES[objective]), scales=scales,
        reference=dict(split='train', dataset_sha256=context['binding']['dataset_sha256'],
            train_rows_sha256=json_sha([context['source_ids'][i] for i in indices])),
        rule_definition_sha256=json_sha(rule_definition()), success_cap=2., failure_cost=4.)
    return validate_spec(spec), audit


def labels(data, indices, spec):
    """Executed/native cost labels; unresolved rows stay invalid, never zero."""
    spec = validate_spec(spec)
    indices = np.asarray(indices, dtype=int)
    values = torch.as_tensor(data['cost_values'][indices], dtype=torch.float64)
    observed = torch.as_tensor(data['cost_mask'][indices], dtype=torch.bool) & torch.isfinite(values)
    weights = torch.tensor(spec['weights'], dtype=values.dtype)
    scales = torch.tensor(spec['scales'], dtype=values.dtype)
    active = weights > 0
    active_ok = (observed[:, active] & (values[:, active] >= 0)).all(1)
    status = data['statuses'][indices].tolist()
    success = torch.tensor([s == 'success' for s in status], dtype=torch.bool) & active_ok
    failure = torch.tensor([s in ('physical_failure', 'geometry_failure') for s in status], dtype=torch.bool)
    valid = success | failure
    result = torch.full((len(indices),), float('nan'), dtype=values.dtype)
    composite = result.clone()
    if bool(success.any()):
        c = (values[success][:, active] / scales[active] * weights[active]).sum(1)
        if not bool(torch.isfinite(c).all()):
            raise ValueError('Nonfinite normalized rule target')
        composite[success] = c
        result[success] = 2*c/(1+c)
    result[failure] = 4.
    return dict(cost=result, utility=-result, composite_cost=composite, valid=valid,
                success=success, failure=failure)


def fit(args, context, name, train_indices, target_indices, target_data, intercept=None):
    """Use unchanged bounded fit implementation with private label functions."""
    namespace = dict(bounded.fit.__globals__, specification=specification, labels=labels, STOP=STOP)
    private = FunctionType(bounded.fit.__code__, namespace, bounded.fit.__name__, bounded.fit.__defaults__)
    private.__kwdefaults__ = bounded.fit.__kwdefaults__
    def observed(batch):
        if intercept is not None:
            intercept(batch)
        namespace['STOP'] = STOP
    _ACTIVE_FIT_GLOBALS.append(namespace)
    try:
        return private(args, context, name, train_indices, target_indices, target_data, intercept=observed)
    finally:
        _ACTIVE_FIT_GLOBALS.remove(namespace)


def tasks():
    return [dict(index=i, objective=o, seed=s) for i, (o,s) in
            enumerate((o,s) for o in OBJECTIVES for s in range(3))]


def folds_for(data):
    folds = family_folds(data['train']['families'], 3)
    folds.append(dict(name='refit', train_indices=list(range(len(data['train']['x']))),
                      target_indices=list(range(len(data['validation']['x'])))))
    return folds


def source_binding():
    return {name:sha(ROOT/name) for name in SOURCE_NAMES}


def _finite_training(data):
    train = data['train']
    if (not (train['statuses'] == 'success').all() or not np.isfinite(train['native']).all() or
        not bool(torch.isfinite(train['x']).all()) or not np.isfinite(train['cost_values']).all()):
        raise ValueError('Declared original384 TRAIN must retain all finite native successes and required rules')


def prepare(args):
    _slurm()
    data, manifest, inputs = load_inputs(Path(args.dataset))
    _finite_training(data)
    folds = folds_for(data)
    context = dict(**data['train'], binding=dict(dataset_sha256=inputs['dataset_sha256']))
    # Admit or refuse the fixed whole experiment before training; do not
    # silently omit a constant objective and keep the rest as a selected grid.
    for fold in folds:
        for objective in OBJECTIVES:
            specification(context, fold['train_indices'], objective)
    plan = dict(schema=SCHEMA, output=str(Path(args.output).resolve()), design=manifest['design'],
        inputs=inputs, source_sha256=source_binding(), config=CONFIG,
        objectives={k:list(v) for k,v in OBJECTIVES.items()}, tasks=tasks(), folds=folds,
        fold_plan_sha256=json_sha(folds), metric_names=list(METRIC_NAMES), cost_names=list(COST_NAMES),
        rule_definition=rule_definition(), train_source_ids=data['train']['source_ids'],
        validation_source_ids=data['validation']['source_ids'],
        train_source_ids_sha256=json_sha(data['train']['source_ids']), q_units=Q_UNITS,
        scope='fixed expert reward development screen; historical16 validation previously inspected',
        native_reward_unchanged=True, backend_calls=0, actor_advanced=False, heldout_used=False,
        training_engine='private FunctionType of frozen bounded.fit exact bytecode; only specification/labels replaced')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = output/'plan.json'
        if path.exists() and read(path) != plan:
            raise ValueError('Immutable rule screen declaration differs')
        for name, digest in plan['source_sha256'].items():
            dest = output/'source_snapshot'/name
            if dest.exists():
                bind(dest, digest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_name(dest.name+'.tmp')
                shutil.copyfile(ROOT/name, temporary)
                bind(temporary, digest)
                temporary.replace(dest)
        if not path.exists():
            bounded.atomic_json(path, plan)
    print(json.dumps(dict(plan=str(path), plan_sha256=sha(path), tasks=len(tasks()), fits=len(tasks())*4)), flush=True)


def checked_plan(plan_path):
    """Return (plan, data, TRAIN context); no fit, selection or model forward."""
    _slurm()
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    sources = source_binding()
    if (plan['schema'] != SCHEMA or plan['config'] != CONFIG or plan['tasks'] != tasks() or
        plan['source_sha256'] != sources or plan['objectives'] != {k:list(v) for k,v in OBJECTIVES.items()} or
        plan['rule_definition'] != rule_definition() or plan['cost_names'] != list(COST_NAMES) or
        plan['q_units'] != Q_UNITS or plan['native_reward_unchanged'] is not True or
        plan['backend_calls'] != 0 or plan['actor_advanced'] is not False or plan['heldout_used'] is not False):
        raise ValueError('Rule screen fixed declaration/source/objectives differ')
    for name, digest in sources.items():
        bind(Path(plan['output'])/'source_snapshot'/name, digest)
    data, manifest, inputs = load_inputs(Path(plan['inputs']['dataset_path']))
    _finite_training(data)
    folds = folds_for(data)
    if (inputs != plan['inputs'] or manifest['design'] != plan['design'] or plan['folds'] != folds or
        plan['fold_plan_sha256'] != json_sha(folds) or plan['metric_names'] != list(METRIC_NAMES) or
        plan['train_source_ids'] != data['train']['source_ids'] or
        plan['validation_source_ids'] != data['validation']['source_ids'] or
        plan['train_source_ids_sha256'] != json_sha(data['train']['source_ids'])):
        raise ValueError('Rule screen input/family/fold/row binding changed')
    from baseline.api import load_problem
    cache = torch.load(inputs['cache_path'], map_location='cpu', weights_only=False)
    movable = ~load_problem(plan['design'], 'cpu').cond.is_ports.bool()
    context = dict(**data['train'], cache=cache, movable=movable,
                   binding=dict(dataset_sha256=inputs['dataset_sha256']))
    return plan, data, context


def run_binding(plan, plan_path, task):
    return dict(schema=plan['schema'], **plan['inputs'], plan_sha256=sha(plan_path), task=task,
        source_sha256=plan['source_sha256'], fold_plan_sha256=plan['fold_plan_sha256'],
        threads=torch.get_num_threads(), torch_version=torch.__version__, numpy_version=np.__version__)


def checked_model(context, spec, seed, indices, checkpoint, done, expected_steps=2000):
    """Reconstruct full fit-TRAIN model buffers before strict state loading."""
    checkpoint = Path(checkpoint)
    bind(checkpoint, done['checkpoint_sha256'])
    truth = labels(context, indices, spec)
    if not bool(truth['valid'].all()):
        raise ValueError('Declared rule checkpoint has invalid TRAIN targets')
    x = context['x'][indices]
    model = MultiMetricCritic(context['cache'], x, truth['cost'].float()[:, None], context['movable'],
        ['bounded_cost'], train_ids=[context['source_ids'][i] for i in indices],
        train_splits=['train']*len(x), width=128, depth=2, ensemble_size=2, hpwl_residual=False, seed=seed)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (done['status'] != 'completed' or done['steps'] != expected_steps or saved['step'] != expected_steps or
        done['utility_spec'] != spec or saved['utility_spec'] != spec or saved['binding'] != done['binding'] or
        saved['normalization'] != model.normalization_metadata() or
        done['normalization'] != model.normalization_metadata() or saved['spec'] != model.spec or
        saved['actor_contract'] != ACTOR_CONTRACT or
        saved['model'].get('_extra_state') != model.get_extra_state()):
        raise ValueError('Rule checkpoint fixed step/spec/TRAIN identity/normalization differs')
    for name, value in model.named_buffers():
        actual = saved['model'].get(name)
        if not isinstance(actual, torch.Tensor) or actual.dtype != value.dtype or not torch.equal(actual, value):
            raise ValueError('Rule checkpoint TRAIN/cache buffer changed: '+name)
    model.load_state_dict(saved['model'], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()
           if isinstance(value, torch.Tensor) and value.is_floating_point()):
        raise ValueError('Nonfinite rule checkpoint')
    return model


def checked_refit(plan_path, objective, seed, loaded=None):
    """Verify all four fits of this task; return fixed last-update refit model.

Actor preparation may share ``loaded=checked_plan(path)`` across24 tasks.
No CV score, historical label or best checkpoint is used for admission.
    """
    plan_path = Path(plan_path).resolve()
    plan, data, context = checked_plan(plan_path) if loaded is None else loaded
    task = next((t for t in plan['tasks'] if t['objective'] == objective and t['seed'] == seed), None)
    if task is None:
        raise ValueError('Undeclared rule refit task')
    folder = Path(plan['output'])/objective/f'seed_{seed}'
    done, summary = read(folder/'completed.json'), read(folder/'summary.json')
    run = done['binding']
    expected_run = run_binding(plan, plan_path, task)
    # CPU thread identity is an actual recorded run setting. Verification can
    # use a different allocation, but cannot rewrite that original setting.
    expected_run['threads'] = run.get('threads')
    if (run != expected_run or not isinstance(run['threads'], int) or run['threads'] < 1 or
        done['status'] != 'completed' or done['summary_sha256'] != sha(folder/'summary.json') or
        done['backend_calls'] != 0 or done['actor_advanced'] is not False or
        summary['status'] != 'completed' or summary['binding'] != run or
        summary['objective'] != objective or summary['seed'] != seed or len(summary['outcomes']) != 4):
        raise ValueError('Rule task is not a bound completed four-fit screen')
    evidence = {str(plan_path):sha(plan_path), str(folder/'completed.json'):sha(folder/'completed.json'),
                str(folder/'summary.json'):done['summary_sha256']}
    selected = None
    for j, fold in enumerate(plan['folds']):
        destination = data['validation'] if fold['name'] == 'refit' else context
        fit_folder = folder/fold['name']
        completed = read(fit_folder/'completed.json')
        spec, audit = specification(context, fold['train_indices'], objective)
        identity = dict(train_indices=fold['train_indices'], admitted_train_indices=fold['train_indices'],
            target_indices=fold['target_indices'],
            train_source_ids=[context['source_ids'][i] for i in fold['train_indices']],
            target_source_ids=[destination['source_ids'][i] for i in fold['target_indices']])
        expected = dict(run=run, fit=fold['name'], objective=objective, utility_spec=spec,
            utility_scale_audit=audit, config={k:dict(seed=seed, **CONFIG)[k]
                for k in ('seed', 'steps', 'batch_size', 'lr', 'weight_decay')},
            **identity, q_units=Q_UNITS, beta=0.)
        if (completed != summary['outcomes'][j] or completed['binding'] != expected or
            completed['status'] != 'completed' or completed['steps'] != CONFIG['steps'] or
            completed['backend_calls'] != 0 or completed['actor_advanced'] is not False or
            completed['heldout_used'] is not False):
            raise ValueError('Rule fit TRAIN/target/step/source identity differs')
        for filename, key in (('resume.pt', 'checkpoint_sha256'), ('predictions.npz', 'predictions_sha256')):
            evidence[str(fit_folder/filename)] = bind(fit_folder/filename, completed[key])['sha256']
        evidence[str(fit_folder/'completed.json')] = sha(fit_folder/'completed.json')
        target = labels(destination, fold['target_indices'], spec)
        with np.load(fit_folder/'predictions.npz', allow_pickle=False) as arrays:
            check = dict(train_indices=fold['train_indices'], target_indices=fold['target_indices'],
                raw_action=destination['x'][fold['target_indices']].numpy(),
                executed_action=destination['executed'][fold['target_indices']],
                native_metrics=destination['native'][fold['target_indices']],
                native_metric_mask=destination['mask'][fold['target_indices']],
                family=destination['families'][fold['target_indices']],
                status=destination['statuses'][fold['target_indices']],
                valid=target['valid'].numpy(), truth=target['cost'].numpy(), utility_truth=target['utility'].numpy(),
                train_source_ids=identity['train_source_ids'], target_source_ids=identity['target_source_ids'])
            for key, value in check.items():
                numeric = np.asarray(value).dtype.kind in 'fc'
                if not np.array_equal(arrays[key], value, equal_nan=numeric):
                    raise ValueError('Rule saved prediction provenance/labels differ: '+key)
            if not np.isfinite(arrays['prediction']).all():
                raise ValueError('Nonfinite saved rule predictions')
        model = checked_model(context, spec, seed, fold['train_indices'], fit_folder/'resume.pt', completed,
                              expected_steps=CONFIG['steps'])
        if fold['name'] == 'refit':
            selected = dict(model=model, spec=spec, scale_audit=audit, context=context,
                checkpoint=bind(fit_folder/'resume.pt'), completion=bind(fit_folder/'completed.json'),
                normalization=model.normalization_metadata(), done=completed)
    if selected is None:
        raise ValueError('Rule task lacks fixed full-TRAIN refit')
    return dict(**selected, plan=plan, data=data, evidence=evidence)


def run(args):
    _slurm()
    torch.use_deterministic_algorithms(True)
    plan, data, context = checked_plan(args.plan)
    if not 0 <= args.index < len(tasks()):
        raise ValueError('Rule screen task index out of declared range')
    task = plan['tasks'][args.index]
    fit_args = argparse.Namespace(**CONFIG, **task,
        output=Path(plan['output'])/task['objective']/f"seed_{task['seed']}")
    fit_args.output.mkdir(parents=True, exist_ok=True)
    context['binding'] = run_binding(plan, args.plan, task)
    with (fit_args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        outcomes = []
        for fold in plan['folds']:
            target = data['validation'] if fold['name'] == 'refit' else context
            outcomes.append(fit(fit_args, context, fold['name'], fold['train_indices'], fold['target_indices'], target))
        report = dict(status='completed', binding=context['binding'], objective=task['objective'], seed=task['seed'],
                      outcomes=outcomes, **summarize(outcomes))
        bounded.atomic_json(fit_args.output/'summary.json', report)
        bounded.atomic_json(fit_args.output/'completed.json', dict(status='completed', binding=context['binding'],
            summary_sha256=sha(fit_args.output/'summary.json'), backend_calls=0, actor_advanced=False))
    print(json.dumps(dict(status='completed', task=task, output=str(fit_args.output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    prepare_parser.add_argument('--output', type=Path, required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--plan', type=Path, required=True)
    run_parser.add_argument('--index', type=int)
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, _stop)
    if args.command == 'run' and args.index is None:
        if 'SLURM_ARRAY_TASK_ID' not in os.environ:
            parser.error('--index or SLURM_ARRAY_TASK_ID required')
        args.index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
