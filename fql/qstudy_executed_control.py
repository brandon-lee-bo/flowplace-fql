"""Fixed predictive diagnostic: raw proposals versus pre-DEF legalizer outputs.

Only the critic coordinate input changes. Dataset labels, failure accounting,
family folds, residual architecture, optimizer and paired minibatches remain
those of the completed raw-input screen. This is not an actor-ready surrogate:
better prediction after a nondifferentiable legalizer validates no actor gradient.
No placement generation, physical evaluation or dataset writes are performed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
from types import SimpleNamespace

import numpy as np
import torch

from .graph_o2o_archive import atomic_json, sha
from .qstudy_data import load_dataset
from .qstudy_models import measured_costs
from . import qstudy_screen as screen

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')
INPUT = 'legalizer_output_pre_DEF'
DATASETS = ('bp_fe384', 'bp_be64')
VARIANT = 'huber_residual'
FOLD_SEED = 41307
CONFIG = dict(steps=2000, folds=3, batch_size=64, lr=3e-4, weight_decay=1e-4)
SOURCE_NAMES = (
    'fql/qstudy_executed_control.py', 'fql/qstudy_executed_control.slurm',
    'fql/qstudy_screen.py', 'fql/qstudy_models.py', 'fql/qstudy_data.py',
    'fql/graph_o2o_models.py', 'fql/physical_ridge.py', 'fql/analytic.py',
    'fql/graph_o2o_archive.py', 'baseline/api.py',
    'upstream/flowplace/benchmark_utils/openroad_utils.py',
)


def require_slurm():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Declare and fit this diagnostic through Slurm')


def array_sha(value):
    a = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
    header = json.dumps(dict(shape=list(a.shape), dtype=a.dtype.str), sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + a.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_bindings():
    return {name: sha(ROOT/name) for name in SOURCE_NAMES}


def check_sources(expected):
    for name, digest in expected.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen source changed: '+name)


def assemble_arrays(data):
    """Match screen's successful-WL filter exactly, preserving every omission."""
    arrays, accounting = {}, {}
    for split in ('train', 'validation'):
        source = data[split]
        raw = torch.tensor(source['raw_action'], dtype=torch.float32)
        executed = torch.tensor(source['executed_action'], dtype=torch.float32)
        if raw.ndim != 2 or raw.shape != executed.shape or raw.shape[1] % 2:
            raise ValueError('Raw/executed coordinates must have identical flattened shape')
        if not bool(torch.isfinite(raw).all() & torch.isfinite(executed).all()):
            raise ValueError('Raw/executed coordinates must be finite')
        costs, mask = measured_costs(torch.tensor(source['metrics'][:, :5], dtype=torch.float32),
            screen.METRICS, torch.tensor(source['metric_mask'][:, :5]))
        keep = mask[:, 0] & (costs[:, 0] > 0)
        families = torch.tensor(source['family'], dtype=torch.long)
        rows = np.flatnonzero(keep.numpy())
        arrays[split] = dict(x=executed.reshape(len(raw), -1, 2)[keep],
            raw_x=raw.reshape(len(raw), -1, 2)[keep], costs=costs[keep], mask=mask[keep],
            families=families[keep], source_indices=rows)
        values = arrays[split]
        accounting[split] = dict(original_rows=len(raw), fitted_rows=int(keep.sum()),
            excluded_rows=int((~keep).sum()), excluded_source_indices=np.flatnonzero(~keep.numpy()).tolist(),
            status_counts={str(s): int(np.sum(np.asarray(source['status']) == s))
                           for s in np.unique(source['status'])},
            unique_executed_rows=len({array_sha(row) for row in values['x']}),
            array_sha256={k: array_sha(v) for k,v in values.items()})
    if set(arrays['train']['families'].tolist()) & set(arrays['validation']['families'].tolist()):
        raise ValueError('TRAIN/validation family leakage')
    return arrays, accounting


def folds(arrays, count=3):
    families = arrays['train']['families'].numpy()
    groups = np.unique(families)
    if count < 2 or len(groups) < count:
        raise ValueError('Insufficient TRAIN families for grouped CV')
    np.random.default_rng(FOLD_SEED).shuffle(groups)
    result = []
    for index, held in enumerate(np.array_split(groups, count)):
        result.append(dict(name=f'cv_{index}', split='train',
            train_indices=np.flatnonzero(~np.isin(families, held)).tolist(),
            target_indices=np.flatnonzero(np.isin(families, held)).tolist()))
    result.append(dict(name='refit', split='validation',
        train_indices=list(range(len(families))),
        target_indices=list(range(len(arrays['validation']['x'])))))
    return result


def verify_prediction(folder, done, binding, fold, arrays):
    """No model load: verify full completion, exact rows/labels and native units."""
    folder = Path(folder)
    train = np.asarray(fold['train_indices'], dtype=np.int64)
    target = np.asarray(fold['target_indices'], dtype=np.int64)
    source = arrays[fold['split']]
    if done['status'] != 'completed' or done['binding'] != binding:
        raise ValueError('Fit completion binding differs')
    if (done['variant'] != VARIANT or done['seed'] != binding['config']['seed'] or
            done['steps'] != binding['config']['steps'] or
            done['train_rows'] != len(train) or done['target_rows'] != len(target)):
        raise ValueError('Fit identity/budget/row count differs')
    if sha(folder/'resume.pt') != done['checkpoint_sha256']:
        raise ValueError('Completed checkpoint SHA differs')
    if fold['split'] == 'train' and set(arrays['train']['families'][train].tolist()) & set(source['families'][target].tolist()):
        raise ValueError('CV placement family leakage')
    with np.load(folder/'predictions.npz', allow_pickle=False) as p:
        for key, expected in dict(train_indices=train, target_indices=target,
            truth=source['costs'][target].numpy(), family=source['families'][target].numpy()).items():
            if not np.array_equal(p[key], expected, equal_nan=True):
                raise ValueError('Prediction rows/labels differ: '+key)
        if p['prediction'].shape != (len(target), len(screen.METRICS)) or not np.isfinite(p['prediction']).all():
            raise ValueError('Invalid native-unit target predictions')
        if p['prediction_train'].shape != (len(train), len(screen.METRICS)) or not np.isfinite(p['prediction_train']).all():
            raise ValueError('Invalid native-unit TRAIN predictions')
    # Outcome target statistics must use only this fold's TRAIN, not targets.
    costs, mask = arrays['train']['costs'][train], arrays['train']['mask'][train]
    from .qstudy_models import _masked_stats
    mean, std, scale, count = _masked_stats(costs, mask & torch.isfinite(costs))
    metadata = done['normalization']
    for key, expected in [('target_mean', mean), ('target_std', std), ('objective_scale', scale)]:
        if not np.allclose(metadata[key], expected.numpy(), rtol=1e-6, atol=1e-8):
            raise ValueError('TRAIN-only target normalization differs: '+key)
    return dict(completed_sha256=sha(folder/'completed.json'), checkpoint_sha256=sha(folder/'resume.pt'),
                predictions_sha256=sha(folder/'predictions.npz'))


def raw_comparisons(task, arrays, sources):
    folder = Path(task['raw_output'])
    overall = read_json(folder/'completed.json')
    old = read_json(folder/'binding.json')
    if overall['status'] != 'completed' or overall['binding'] != old:
        raise ValueError('Raw comparator is not completely fitted')
    for key in ('dataset_sha256', 'manifest_sha256', 'cache_sha256'):
        if old[key] != task[key]:
            raise ValueError('Raw comparator input binding differs: '+key)
    expected = dict(dataset=task['dataset'], output=task['raw_output'], variant=VARIANT,
                    seed=task['seed'], **task['config'])
    if old['config'] != expected or old.get('fold_seed') != FOLD_SEED or old.get('heldout_used') or old.get('backend_calls') != 0:
        raise ValueError('Raw comparator budget/fold/configuration differs')
    for name, digest in old['source_sha256'].items():
        if sources.get(name) != digest:
            raise ValueError('Raw comparator frozen source differs: '+name)
    if len(overall['outcomes']) != len(task['folds']):
        raise ValueError('Raw comparator four-fit accounting differs')
    result = dict(completed_sha256=sha(folder/'completed.json'), binding_sha256=sha(folder/'binding.json'), fits={})
    for fold, retained in zip(task['folds'], overall['outcomes']):
        done = read_json(folder/fold['name']/'completed.json')
        if done != retained:
            raise ValueError('Raw comparator retained outcome changed')
        result['fits'][fold['name']] = verify_prediction(folder/fold['name'], done, old, fold, arrays)
    return result


def make_task(dataset, seed, output, matrix, sources, config=None):
    path = Path(dataset).resolve()
    data, manifest = load_dataset(path)
    arrays, accounting = assemble_arrays(data)
    task = dict(dataset=str(path), dataset_sha256=sha(path),
        manifest_sha256=sha(path.with_name('manifest.json')), rows_sha256=sha(path.with_name('rows.jsonl')),
        cache_sha256=sha(ROOT/'baseline/runtime/benchmark_cache'/(manifest['design']+'.pt')),
        design=manifest['design'], seed=seed, variant=VARIANT, config=dict(config or CONFIG),
        input_representation=INPUT, folds=folds(arrays, (config or CONFIG)['folds']), accounting=accounting,
        output=str(Path(output).resolve()), raw_output=str(Path(matrix).resolve()/path.parent.name/VARIANT/f'seed_{seed}'))
    task['raw_comparison'] = raw_comparisons(task, arrays, sources)
    return task, arrays


def prepare(output, matrix=BASE/'screen_v1', data_root=BASE/'data_audit'):
    require_slurm()
    output, matrix = Path(output).resolve(), Path(matrix).resolve()
    data_root = Path(data_root).resolve()
    if any(output == p or p in output.parents or output in p.parents for p in (matrix,data_root)):
        raise ValueError('Diagnostic outputs must be separate from immutable raw fits and datasets')
    if output.exists():
        raise FileExistsError('Choose a new immutable diagnostic directory')
    sources = source_bindings()
    tasks = [make_task(Path(data_root)/dataset/'dataset.npz', seed,
        output/dataset/f'seed_{seed}', matrix, sources)[0] for dataset in DATASETS for seed in (0,1,2)]
    check_sources(sources)
    output.mkdir(parents=True)
    for name in sources:
        destination = output/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        if sha(destination) != sources[name]:
            raise ValueError('Source changed during declaration: '+name)
    atomic_json(output/'tasks.json', tasks)
    plan = dict(status='declared_before_executed_input_results', task_count=6,
        tasks_sha256=sha(output/'tasks.json'), source_sha256=sources,
        datasets=list(DATASETS), seeds=[0,1,2], variant=VARIANT, config=CONFIG,
        raw_matrix=str(matrix), data_root=str(data_root),
        input_representation=INPUT, primary_metrics=['WNS_ns','TNS_ns'],
        interpretation='Predictive preprocessing diagnostic; no legalizer or actor gradient validation',
        raw_and_executed_prior='All coordinate-derived physical features and TRAIN-fitted smooth-HPWL baseline use the selected representation',
        collisions='Original logical rows/family folds retained; repeated executed layouts disclosed, not independent evidence',
        historical_validation='Previously inspected development data; grouped TRAIN CV primary',
        layout_generation_calls=0, backend_calls=0, actor_advanced=False, dataset_mutations=0,
        critic_fits_per_task=4, planned_critic_fits=24)
    atomic_json(output/'plan.json', plan)
    return plan


def run(plan_path, index):
    require_slurm()
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if (plan['task_count'] != 6 or plan['datasets'] != list(DATASETS) or plan['seeds'] != [0,1,2]
            or plan['variant'] != VARIANT or plan['config'] != CONFIG or plan['input_representation'] != INPUT):
        raise ValueError('Only the predeclared six-task residual comparison is supported')
    check_sources(plan['source_sha256'])
    tasks_path = plan_path.with_name('tasks.json')
    if sha(tasks_path) != plan['tasks_sha256']:
        raise ValueError('Declared diagnostic tasks changed')
    tasks = read_json(tasks_path)
    if len(tasks) != plan['task_count'] or not 0 <= index < len(tasks):
        raise ValueError('Task index/count outside declared bounds')
    identities = [(Path(t['dataset']).parent.name,t['seed']) for t in tasks]
    if identities != [(dataset,seed) for dataset in DATASETS for seed in (0,1,2)]:
        raise ValueError('Six-task dataset/seed identities differ')
    if (len({t['output'] for t in tasks}) != 6 or
            any(t['config'] != CONFIG or t['variant'] != VARIANT for t in tasks)):
        raise ValueError('Task budgets/variant/output identities differ')
    for t, (dataset,seed) in zip(tasks,identities):
        if (Path(t['output']) != plan_path.parent/dataset/f'seed_{seed}' or
                Path(t['dataset']) != Path(plan['data_root'])/dataset/'dataset.npz' or
                Path(t['raw_output']) != Path(plan['raw_matrix'])/dataset/VARIANT/f'seed_{seed}'):
            raise ValueError('Task output/data/comparator path differs from declaration')
    task = tasks[index]
    if task['input_representation'] != INPUT:
        raise ValueError('Only pre-DEF legalizer coordinates are permitted')
    current, arrays = make_task(task['dataset'], task['seed'], task['output'],
        Path(task['raw_output']).parents[2], plan['source_sha256'], task['config'])
    if current != task:
        raise ValueError('Declared data/folds/labels/raw comparator changed')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    torch.use_deterministic_algorithms(True)
    screen.STOP = False
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, screen._stop)
    folder = Path(task['output'])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_task(plan_path, plan, task, arrays)


def _run_task(plan_path, plan, task, arrays):
    folder = Path(task['output'])
    binding = dict(dataset_sha256=task['dataset_sha256'], manifest_sha256=task['manifest_sha256'],
        cache_sha256=task['cache_sha256'], config=dict(output=str(folder), dataset=task['dataset'],
        variant=VARIANT, seed=task['seed'], **task['config']), source_sha256=plan['source_sha256'],
        fold_seed=FOLD_SEED, heldout_used=False, backend_calls=0,
        input_representation=INPUT, plan_sha256=sha(plan_path), task=task)
    if (folder/'binding.json').exists() and read_json(folder/'binding.json') != binding:
        raise ValueError('Executed-input run binding changed')
    if not (folder/'binding.json').exists():
        atomic_json(folder/'binding.json', binding)
    cache = torch.load(ROOT/'baseline/runtime/benchmark_cache'/(task['design']+'.pt'),
                       map_location='cpu', weights_only=False)
    from baseline.api import load_problem
    movable = ~load_problem(task['design'], 'cpu').cond.is_ports.bool()
    context = dict(**arrays['train'], cache=cache, movable=movable, binding=binding)
    args = SimpleNamespace(output=folder, variant=VARIANT, seed=task['seed'], **task['config'])
    results = []
    for fold in task['folds']:
        if screen.STOP:
            raise SystemExit(85)
        name = fold['name']
        inventory_path = folder/name/'verified.json'
        if inventory_path.exists():
            before = read_json(inventory_path)
            for key, filename in [('completed_sha256','completed.json'),('checkpoint_sha256','resume.pt'),('predictions_sha256','predictions.npz')]:
                if sha(folder/name/filename) != before[key]:
                    raise ValueError('Completed executed artifact changed: '+filename)
        done = screen.fit(args, context, name, np.asarray(fold['train_indices'], dtype=np.int64),
            np.asarray(fold['target_indices'], dtype=np.int64), arrays[fold['split']])
        inventory = verify_prediction(folder/name, done, binding, fold, arrays)
        train_hashes = {array_sha(row) for row in arrays['train']['x'][fold['train_indices']]}
        inventory.update(raw_comparison=task['raw_comparison']['fits'][name],
            executed_target_rows_also_in_train=sum(array_sha(row) in train_hashes
                for row in arrays[fold['split']]['x'][fold['target_indices']]))
        if inventory_path.exists() and read_json(inventory_path) != inventory:
            raise ValueError('Executed-input verified inventory changed')
        if not inventory_path.exists():
            atomic_json(inventory_path, inventory)
        results.append(dict(name=name, **inventory))
    check_sources(plan['source_sha256'])
    final = dict(status='completed', binding=binding, fits=results,
        layout_generation_calls=0, backend_calls=0, actor_advanced=False, critic_fits=4,
        note='Predictive comparison only; nondifferentiable legalization remains outside actor gradients')
    if (folder/'completed.json').exists() and read_json(folder/'completed.json') != final:
        raise ValueError('Executed-input task completion changed')
    if not (folder/'completed.json').exists():
        atomic_json(folder/'completed.json', final)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--matrix', type=Path, default=BASE/'screen_v1')
    q.add_argument('--data-root', type=Path, default=BASE/'data_audit')
    q = sub.add_parser('run')
    q.add_argument('--plan', type=Path, required=True)
    q.add_argument('--index', type=int)
    args = p.parse_args()
    if args.stage == 'prepare':
        print(json.dumps(prepare(args.output, args.matrix, args.data_root)), flush=True)
    else:
        run(args.plan, args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))


if __name__ == '__main__':
    main()
