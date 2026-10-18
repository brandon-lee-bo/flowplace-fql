"""Predeclared, measured-Q data-source controls; no EDA or actor promotion.

Stages: plan before new labels; assemble immutable six-arm datasets; train one
array task by reusing the frozen screen.fit; report matched completed evidence.
All stages run under Slurm. Original native rows/bindings remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal

import numpy as np

from .qstudy_data import METRIC_NAMES, action_sha, bind, load_dataset, sha, summarize

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')
DEFAULT_OLD = BASE/'data_audit/bp_fe64/dataset.npz'
DEFAULT_TEACHER = BASE/'teacher_source_v1/bp_fe_common0'
ARMS = ('O64', 'T64', 'M64', 'O32_base', 'T32_base', 'M32_base')
VARIANTS = ('legacy', 'huber_residual')
SEEDS = (0, 1, 2)
SOURCE_NAMES = ('old', 'teacher')
SELECTION_SEED = 74241
FOLD_SEED = 41307
STATS = ('mae', 'spearman', 'kendall', 'top_regret', 'family_pair_accuracy')
COST_DEFINITIONS = ('GRT wirelength um', 'max(-native WNS ns,0)', 'max(-native TNS ns,0)',
                    'horizontal positive demand-minus-capacity', 'vertical positive demand-minus-capacity')


def read_json(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def object_sha(value):
    text = json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def dataset_binding(path):
    path = Path(path).resolve()
    return {name: bind(path if name == 'dataset' else path.with_name(name+suffix))
            for name, suffix in [('dataset', ''), ('manifest', '.json'), ('rows', '.jsonl')]}


def read_rows(path):
    return [json.loads(line) for line in Path(path).with_name('rows.jsonl').read_text().splitlines() if line]


def declared_families(rows, split):
    selected = [r for r in rows if r['split'] == split]
    groups = defaultdict(list)
    for row in selected:
        groups[int(row['family'])].append(int(row['variant']))
    if any(sorted(variants) != [0, 1] for variants in groups.values()):
        raise ValueError('Source must retain exactly base/paired-variant rows per family')
    return sorted(groups)


def plan(args):
    old = args.old_dataset.resolve()
    teacher = args.teacher_experiment.resolve()
    bindings = dataset_binding(old)
    manifest = read_json(old.with_name('manifest.json'))
    rows = read_rows(old)
    old_families = declared_families(rows, 'train')
    if len(old_families) != 32 or len(declared_families(rows, 'validation')) != 8:
        raise ValueError('O64 requires the audited32TRAIN/8validation family source')
    cfg_binding = bind(teacher/'config.json')
    cfg = read_json(cfg_binding['path'])
    if (cfg['design'] != manifest['design'] or cfg['train_families'] != 32 or
            cfg['val_families'] != 8 or cfg['noise_seed'] != 5300000 or
            cfg.get('common_state_key', 'flow') != 'flow' or cfg.get('flow_steps', 10) != 10):
        raise ValueError('Unexpected predeclared current teacher configuration')
    teacher_families = list(range(cfg['noise_seed'], cfg['noise_seed']+32))
    rng = np.random.default_rng(SELECTION_SEED)
    old_order = np.asarray(old_families, dtype=np.int64)
    teacher_order = np.asarray(teacher_families, dtype=np.int64)
    rng.shuffle(old_order)
    rng.shuffle(teacher_order)
    selections = {}
    for arm in ARMS:
        if arm.startswith('O'):
            families = dict(old=old_families, teacher=[])
        elif arm.startswith('T'):
            families = dict(old=[], teacher=teacher_families)
        else:
            families = dict(old=sorted(old_order[:16].tolist()), teacher=sorted(teacher_order[:16].tolist()))
        selections[arm] = dict(families=families, variants=[0] if 'base' in arm else [0, 1],
                               train_requests=32 if 'base' in arm else 64)
    value = dict(version=1, design=cfg['design'], selection_seed=SELECTION_SEED,
        selection_rule='Shuffle sorted oldTRAIN families, then declared teacherTRAIN families with NumPy default_rng74241; take first16each for mixed; never inspect metrics for selection',
        selection_numpy_version=np.__version__, selections=selections,
        old_source=bindings, teacher_experiment=str(teacher), teacher_config=cfg_binding,
        expected_teacher_dataset=str(teacher/'dataset/dataset.npz'),
        teacher_common=bind(cfg['common_checkpoint'], cfg['common_checkpoint_sha256']),
        teacher_reference=bind(cfg['dataset_path'], cfg['dataset_sha256']),
        cache=bind(ROOT/'baseline/runtime/benchmark_cache'/(cfg['design']+'.pt'), cfg['cache_sha256']),
        variants=list(VARIANTS), seeds=list(SEEDS), steps=args.steps, folds=3,
        fold_seed=FOLD_SEED, batch_size=64, lr=3e-4, weight_decay=1e-4,
        validation='Every arm uses all old16+teacher16 rows; report source separately and base-only separately',
        cv='Deterministic source-stratified grouped3fold; same partition across criticseeds/variants within arm',
        behavior='Exact original audited256 targets unchanged across arms; Q labels alone vary',
        heldout_used=False, online_included=False, automatic_promotion=False,
        expanded448_included=False)
    if args.steps < 1:
        raise ValueError('Training steps must be positive')
    path = args.output/'plan.json'
    if path.exists():
        old_plan = read_json(path)
        if old_plan['declaration'] != value:
            raise ValueError('Immutable source plan differs; use a new output directory')
        return old_plan
    result = dict(declaration=value, declaration_sha256=object_sha(value),
        planned_at=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ['SLURM_JOB_ID'],
        planner_source_sha256=sha(__file__), labels_not_used_for_selection=True)
    atomic_json(path, result)
    print(json.dumps(dict(status='plan_frozen', plan=str(path), plan_sha256=sha(path),
                         declaration_sha256=result['declaration_sha256'], selections=selections)), flush=True)
    return result


def load_plan(output):
    path = output/'plan.json'
    plan = read_json(path)
    d = plan['declaration']
    if object_sha(d) != plan['declaration_sha256']:
        raise ValueError('Plan declaration fingerprint changed')
    for binding in list(d['old_source'].values())+[d['teacher_config'], d['teacher_common'], d['teacher_reference'], d['cache']]:
        bind(binding['path'], binding['sha256'])
    return plan


def check_binding_tree(value, checked):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            key = (value['path'], value['sha256'])
            if key not in checked:
                bind(*key)
                checked.add(key)
        for child in value.values():
            check_binding_tree(child, checked)
    elif isinstance(value, list):
        for child in value:
            check_binding_tree(child, checked)


def audited_source(path, source, checked):
    data, manifest = load_dataset(path)
    if manifest.get('stage') != 'global_route_only' or not manifest.get('not_final_table2'):
        raise ValueError('Only audited native GRT labels enter source controls')
    rows = read_rows(path)
    check_binding_tree(manifest.get('source_bindings'), checked)
    split_rows = {}
    for split in ('train', 'validation'):
        array = data[split]
        selected = [r for r in rows if r['split'] == split]
        if len(selected) != len(array['family']):
            raise ValueError('Native rows do not match split arrays')
        for index, row in enumerate(selected):
            if row['design'] != manifest['design'] or row['terminal_mask'] != 0:
                raise ValueError('Wrong design or nonterminal supervision')
            for key in ('family', 'variant', 'source_row', 'status'):
                if row[key] != array[key][index]:
                    raise ValueError('Native row/array differs: '+key)
            for key in ('raw_action', 'executed_action'):
                if not np.array_equal(np.asarray(row[key], dtype=np.float32), array[key][index]):
                    raise ValueError('Native coordinates changed')
            if (action_sha(row['raw_action']) != row['raw_action_sha256'] or
                    action_sha(row['executed_action']) != row['action_sha256']):
                raise ValueError('Native coordinate SHA changed')
            values = np.asarray([np.nan if row['metrics'][m] is None else row['metrics'][m]
                                 for m in METRIC_NAMES], dtype=np.float64)
            if not np.array_equal(values, array['metrics'][index], equal_nan=True):
                raise ValueError('Native metric/array mismatch')
            if row['status'] not in ('success', 'geometry_failure', 'physical_failure'):
                raise ValueError('Unresolved infrastructure request cannot become Q label')
            if row['status'] != 'success' and np.isfinite(values).any():
                raise ValueError('Failure PPA cannot be fabricated numeric labels')
            check_binding_tree(row['bindings'], checked)
            check_binding_tree(row.get('actual_backend_geometry'), checked)
            if row['status'] == 'success':
                native = read_json(row['bindings']['physical_metrics']['path'])
                if any(native.get(m) != row['metrics'][m] for m in METRIC_NAMES):
                    raise ValueError('Row changed measured native metric')
                if not np.isfinite(values[0]) or values[0] <= 0:
                    raise ValueError('Successful physical execution needs positive measured WL')
                if str(row['backend']['num_cores']) != '8':
                    raise ValueError('Source control must retain fixed8-thread backend')
            if source == 'teacher':
                if (row.get('source_kind') != 'current_fql_behavior_flow' or
                        row.get('source_sampler') != 'fql_bc_flow_euler_10_final_legalize' or
                        row.get('offline_train_eligible') != (split == 'train') or
                        row.get('online_training_feedback_eligible') is not False):
                    raise ValueError('Teacher source/eligibility differs from fixed collection')
                namespace = row['source_family_namespace']
                if object_sha(namespace) != row['source_family_sha256']:
                    raise ValueError('Teacher full family namespace changed')
        split_rows[split] = selected
    return data, manifest, split_rows


def implementation_binding():
    names = ['fql/qstudy_source.py', 'fql/qstudy_source.slurm', 'fql/qstudy_screen.py',
             'fql/qstudy_models.py', 'fql/qstudy_data.py', 'fql/graph_o2o_models.py',
             'fql/physical_ridge.py', 'fql/analytic.py']
    return {name: sha(ROOT/name) for name in names}


def arm_rows(sources, selection):
    result = []
    for split in ('train', 'validation'):
        for source in SOURCE_NAMES:
            for row in sources[source][split]:
                if split == 'train' and (row['family'] not in selection['families'][source] or
                                         row['variant'] not in selection['variants']):
                    continue
                value = copy.deepcopy(row)
                # Original native coordinates, metrics, family, and bindings stay unchanged.
                value['qstudy_source'] = source
                value['qstudy_family_namespace'] = dict(source=source, design=row['design'],
                    source_family=int(row['family']))
                result.append(value)
    return result


def source_counts(rows):
    counts = {}
    for split in ('train', 'validation'):
        for source in SOURCE_NAMES:
            selected = [r for r in rows if r['split'] == split and r['qstudy_source'] == source]
            counts[split+'_'+source] = dict(requests=len(selected), families=len({r['family'] for r in selected}),
                successful=sum(r['status'] == 'success' for r in selected),
                failures=sum(r['status'] != 'success' for r in selected),
                finite_metrics={m: sum(r['metrics'][m] is not None for r in selected) for m in METRIC_NAMES},
                unique_raw=len({r['raw_action_sha256'] for r in selected}),
                unique_executed=len({r['action_sha256'] for r in selected}))
    return counts


def export_arm(folder, arm, rows, reference, parent_bindings, selection, design):
    if folder.exists():
        _, manifest = load_dataset(folder)
        if manifest['source_bindings'] != parent_bindings or manifest['selection'] != selection:
            raise ValueError('Existing assembled arm changed its immutable parents or selection')
        return manifest
    groups = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation')}
    if len(groups['train']) != selection['train_requests'] or len(groups['validation']) != 32:
        raise ValueError('Source arm request denominator changed')
    collisions = {}
    for key in ('raw_action_sha256', 'action_sha256'):
        common = sorted({r[key] for r in groups['train']} & {r[key] for r in groups['validation']})
        collisions[key] = common
        if common:
            raise ValueError('Coordinate leakage across source TRAIN/validation: '+key)
    if {r['family'] for r in groups['train']} & {r['family'] for r in groups['validation']}:
        raise ValueError('Source family crosses TRAIN/validation')
    staging = folder.with_name(folder.name+'.building')
    staging.mkdir(parents=True, exist_ok=True)
    arrays, ordered = {}, []
    for split, selected in groups.items():
        values = np.asarray([[np.nan if r['metrics'][m] is None else r['metrics'][m]
                              for m in METRIC_NAMES] for r in selected], dtype=np.float64)
        fields = dict(raw_action=np.asarray([r['raw_action'] for r in selected], dtype=np.float32),
            executed_action=np.asarray([r['executed_action'] for r in selected], dtype=np.float32),
            metrics=values, metric_mask=np.isfinite(values),
            family=np.asarray([r['family'] for r in selected], dtype=np.int64),
            variant=np.asarray([r['variant'] for r in selected], dtype=np.int64),
            source_row=np.asarray([r['source_row'] for r in selected], dtype=np.int64),
            status=np.asarray([r['status'] for r in selected], dtype='U24'),
            source=np.asarray([r['qstudy_source'] for r in selected], dtype='U8'),
            provenance_index=np.arange(len(ordered), len(ordered)+len(selected), dtype=np.int64))
        arrays.update({split+'_'+key: value for key, value in fields.items()})
        ordered.extend(selected)
    arrays['behavior_action'] = reference['behavior_action']
    arrays['behavior_family'] = reference['behavior_family']
    np.savez_compressed(staging/'dataset.npz', **arrays)
    atomic_text(staging/'rows.jsonl', ''.join(json.dumps(r, allow_nan=False)+'\n' for r in ordered))
    revisions = defaultdict(int)
    for row in ordered:
        if row['backend']:
            revisions[row['backend']['physical_flow_revision']] += 1
    manifest = dict(version=1, design=design, source_arm=arm, selection=selection,
        metric_names=list(METRIC_NAMES), stage='global_route_only', not_final_table2=True,
        metric_units=['um', 'ns', 'ns', 'positive_demand_minus_capacity', 'positive_demand_minus_capacity', 'W', 'um2'],
        dataset_sha256=sha(staging/'dataset.npz'), rows_sha256=sha(staging/'rows.jsonl'),
        source_bindings=parent_bindings, split_counts={s: len(v) for s, v in groups.items()},
        behavior_targets=len(reference['behavior_action']), behavior_source='Unchanged exact reference256 TRAIN targets',
        no_test_or_heldout_exported=True, historical_online_included=False,
        validation_status='old16 historical tuning+teacher16 new designated tuning; never a final test',
        source_counts=source_counts(ordered), exact_cross_split_collisions=collisions,
        physical_flow_revisions=dict(revisions), summary={s: summarize(v) for s, v in groups.items()},
        executed_action_semantics='normalized final-legalizer proposal before DEF rounding/native snapping; actual backend macro geometry retained in native row evidence',
        original_native_fields_unchanged=True, missing_metrics='NaN plus explicit mask; failures retain no numeric PPA',
        assembly_source_sha256=sha(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(staging/'manifest.json', manifest)
    load_dataset(staging)
    staging.replace(folder)
    return manifest


def assemble(args):
    plan = load_plan(args.output)
    d = plan['declaration']
    checked = set()
    old, old_manifest, old_rows = audited_source(d['old_source']['dataset']['path'], 'old', checked)
    teacher_path = Path(d['expected_teacher_dataset'])
    teacher, teacher_manifest, teacher_rows = audited_source(teacher_path, 'teacher', checked)
    if (old_manifest['design'] != d['design'] or teacher_manifest['design'] != d['design'] or
            len(old['behavior_action']) != 256 or
            not np.array_equal(old['behavior_action'], teacher['behavior_action']) or
            not np.array_equal(old['behavior_family'], teacher['behavior_family'])):
        raise ValueError('Changing behavior targets would confound Q-label source')
    cfg = read_json(d['teacher_config']['path'])
    for split, count, start in [('train', 32, cfg['noise_seed']), ('validation', 8, cfg['noise_seed']+32)]:
        if declared_families(teacher_rows[split], split) != list(range(start, start+count)):
            raise ValueError('Teacher families differ from pre-label declaration')
        for row in teacher_rows[split]:
            if row['checkpoint_sha256'] != d['teacher_common']['sha256']:
                raise ValueError('Teacher checkpoint source changed')
    all_rows = old_rows['train']+old_rows['validation']+teacher_rows['train']+teacher_rows['validation']
    signatures = {object_sha({key: r['backend'][key] for key in
                  ['design', 'num_cores', 'design_config_sha256', 'openroad_sha256', 'physical_flow_revision']})
                  for r in all_rows if r['backend']}
    if len(signatures) != 1:
        raise ValueError('Source arms use incompatible physical backend/tool/config revisions')
    parents = dict(old=d['old_source'], teacher=dataset_binding(teacher_path),
                   declaration=bind(args.output/'plan.json'))
    tasks, arms = [], {}
    for arm in ARMS:
        selected = arm_rows(dict(old=old_rows, teacher=teacher_rows), d['selections'][arm])
        path = args.output/'datasets'/arm
        export_arm(path, arm, selected, old, parents, d['selections'][arm], d['design'])
        arms[arm] = dataset_binding(path/'dataset.npz')
        for variant in d['variants']:
            for seed in d['seeds']:
                tasks.append(dict(index=len(tasks), arm=arm, variant=variant, seed=seed,
                    dataset=str(path/'dataset.npz'), output=str(args.output/'runs'/arm/variant/('seed_'+str(seed)))))
    matrix = dict(version=1, plan=bind(args.output/'plan.json'), datasets=arms,
        implementation=implementation_binding(), tasks=tasks, tasks_sha256=object_sha(tasks),
        source_bindings=parents, native_file_bindings_verified=len(checked),
        no_backend_calls=True, automatic_promotion=False)
    path = args.output/'matrix.json'
    if path.exists() and read_json(path) != matrix:
        raise ValueError('Immutable source matrix already differs')
    atomic_json(path, matrix)
    tasks_path = args.output/'tasks.jsonl'
    text = ''.join(json.dumps(task, allow_nan=False)+'\n' for task in tasks)
    if tasks_path.exists() and tasks_path.read_text() != text:
        raise ValueError('Immutable source tasks already differ')
    atomic_text(tasks_path, text)
    print(json.dumps(dict(status='source_datasets_assembled', arms=list(ARMS), tasks=len(tasks),
                         matrix_sha256=sha(path), native_bindings_verified=len(checked))), flush=True)


def load_matrix(output, verify_implementation=True):
    load_plan(output)
    matrix = read_json(output/'matrix.json')
    bind(matrix['plan']['path'], matrix['plan']['sha256'])
    if object_sha(matrix['tasks']) != matrix['tasks_sha256']:
        raise ValueError('Source task list changed')
    for bindings in matrix['datasets'].values():
        for value in bindings.values():
            bind(value['path'], value['sha256'])
    if verify_implementation and implementation_binding() != matrix['implementation']:
        raise ValueError('Frozen source-study implementation changed; new experiment required')
    return matrix


def stratified_folds(families, sources, folds=3):
    result = [[] for _ in range(folds)]
    seen = set()
    for k, source in enumerate(SOURCE_NAMES):
        groups = np.unique(families[sources == source])
        if seen & set(groups):
            raise ValueError('Source namespace collision in integer family identifiers')
        seen.update(groups)
        np.random.default_rng(FOLD_SEED+1009*k).shuffle(groups)
        for fold, values in zip(result, np.array_split(groups, folds)):
            fold.extend(values.tolist())
    if any(not groups for groups in result):
        raise ValueError('Insufficient successful TRAIN families for grouped3fold')
    return [sorted(groups) for groups in result]


def check(args):
    """Native old-source audit and metadata-only future split checks; no model/EDA."""
    plan = load_plan(args.output)
    d = plan['declaration']
    checked = set()
    old, _, rows = audited_source(d['old_source']['dataset']['path'], 'old', checked)
    if len(old['behavior_action']) != 256:
        raise ValueError('Reference behavior denominator changed')
    outcomes = {}
    for arm, selection in d['selections'].items():
        families, sources = [], []
        for source in SOURCE_NAMES:
            for family in selection['families'][source]:
                for _ in selection['variants']:
                    families.append(family)
                    sources.append(source)
        families, sources = np.asarray(families), np.asarray(sources)
        folds = stratified_folds(families, sources)
        if folds != stratified_folds(families, sources) or len(families) != selection['train_requests']:
            raise ValueError('Predeclared source partition is not reproducible')
        if sorted(group for fold in folds for group in fold) != sorted(np.unique(families).tolist()):
            raise ValueError('Source CV family folds are not disjoint/exhaustive')
        balance = {}
        for source in SOURCE_NAMES:
            values = set(families[sources == source])
            counts = [len(values & set(fold)) for fold in folds]
            if max(counts)-min(counts) > 1:
                raise ValueError('Source CV is not family-stratified')
            balance[source] = counts
        outcomes[arm] = dict(train_requests=len(families), family_folds=folds, source_balance=balance)
    result = dict(status='passed', plan=bind(args.output/'plan.json'), arms=outcomes,
        old_native_bindings_verified=len(checked), teacher_metrics_loaded=False, backend_calls=0,
        reference_behavior_targets=len(old['behavior_action']), slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(args.output/'source_check.json', result)
    print(json.dumps(result), flush=True)


def train(args):
    import torch
    from . import qstudy_screen as screen
    from .qstudy_models import measured_costs
    from baseline.api import load_problem
    matrix = load_matrix(args.output)
    index = args.task_index if args.task_index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(matrix['tasks']):
        raise ValueError('Source matrix task index out of bounds')
    task = matrix['tasks'][index]
    d = load_plan(args.output)['declaration']
    local = argparse.Namespace(dataset=Path(task['dataset']), output=Path(task['output']),
        variant=task['variant'], seed=task['seed'], steps=d['steps'], folds=d['folds'],
        batch_size=d['batch_size'], lr=d['lr'], weight_decay=d['weight_decay'])
    local.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, screen._stop)
    data, manifest = load_dataset(local.dataset)
    arrays, source_indices = {}, {}
    for split in ('train', 'validation'):
        source = data[split]
        costs, mask = measured_costs(torch.tensor(source['metrics'][:, :5], dtype=torch.float32),
            screen.METRICS, torch.tensor(source['metric_mask'][:, :5]))
        successful = (mask[:, 0] & (costs[:, 0] > 0)).numpy()
        source_indices[split] = np.flatnonzero(successful).tolist()
        arrays[split] = dict(x=torch.tensor(source['raw_action'][successful], dtype=torch.float32).reshape(successful.sum(), -1, 2),
            costs=costs[successful], mask=mask[successful],
            families=torch.tensor(source['family'][successful], dtype=torch.long))
    train_source = data['train']['source'][source_indices['train']]
    groups = stratified_folds(arrays['train']['families'].numpy(), train_source, d['folds'])
    binding = dict(dataset_sha256=sha(local.dataset), manifest_sha256=sha(local.dataset.with_name('manifest.json')),
        matrix_sha256=sha(args.output/'matrix.json'), plan_sha256=sha(args.output/'plan.json'),
        cache_sha256=d['cache']['sha256'], source_sha256=matrix['implementation'],
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(local).items()},
        family_folds=groups, source_indices=source_indices, task=task,
        heldout_used=False, backend_calls=0, fold_seed=FOLD_SEED,
        training_implementation='unchanged qstudy_screen.fit; source-stratified grouped folds only')
    with (local.output/'.task.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = local.output/'binding.json'
        if path.exists() and read_json(path) != binding:
            raise ValueError('Resumed source run changed immutable identity')
        atomic_json(path, binding)
        cache = torch.load(d['cache']['path'], map_location='cpu', weights_only=False)
        movable = ~load_problem(d['design'], 'cpu').cond.is_ports.bool()
        context = {**arrays['train'], 'cache': cache, 'movable': movable, 'binding': binding}
        outcomes = []
        for fold, held in enumerate(groups):
            families = context['families'].numpy()
            target = np.flatnonzero(np.isin(families, held))
            training = np.flatnonzero(~np.isin(families, held))
            outcomes.append(screen.fit(local, context, 'cv_'+str(fold), training, target, arrays['train']))
        outcomes.append(screen.fit(local, context, 'refit', np.arange(len(context['x'])),
                                   np.arange(len(arrays['validation']['x'])), arrays['validation']))
        atomic_json(local.output/'completed.json', dict(status='completed', binding=binding, outcomes=outcomes))


def raw_costs(split):
    costs = split['metrics'][:, :5].copy()
    costs[:, 1:3] = np.maximum(-costs[:, 1:3], 0)
    keep = np.isfinite(costs[:, 0]) & (costs[:, 0] > 0)
    return costs[keep], np.flatnonzero(keep)


def checked_predictions(task, matrix, declaration):
    """Verify every fit before making any numeric rows from this task."""
    folder = Path(task['output'])
    completed = folder/'completed.json'
    if not completed.exists():
        count = sum((folder/name/'completed.json').exists() for name in ('cv_0', 'cv_1', 'cv_2', 'refit'))
        return None, dict(status='partial' if count else 'pending', completed_fits=count)
    binding = read_json(folder/'binding.json')
    done = read_json(completed)
    dataset, _ = load_dataset(task['dataset'])
    if (done.get('status') != 'completed' or done['binding'] != binding or
            binding['task'] != task or binding['source_sha256'] != matrix['implementation'] or
            binding['dataset_sha256'] != sha(task['dataset']) or
            binding['manifest_sha256'] != sha(Path(task['dataset']).with_name('manifest.json')) or
            binding['plan_sha256'] != matrix['plan']['sha256'] or
            binding['matrix_sha256'] != object_file_sha(matrix) or
            binding['cache_sha256'] != declaration['cache']['sha256']):
        raise ValueError('Task completion/immutable binding mismatch')
    config = binding['config']
    expected = dict(dataset=task['dataset'], output=task['output'], variant=task['variant'], seed=task['seed'],
        steps=declaration['steps'], folds=3, batch_size=declaration['batch_size'],
        lr=declaration['lr'], weight_decay=declaration['weight_decay'])
    if config != expected or binding['heldout_used'] or binding['backend_calls']:
        raise ValueError('Training setup differs from declared matched source experiment')
    data = {}
    for split in ('train', 'validation'):
        costs, indices = raw_costs(dataset[split])
        if indices.tolist() != binding['source_indices'][split]:
            raise ValueError('Successful/missing-label row filter changed')
        data[split] = dict(truth=costs, family=dataset[split]['family'][indices],
            source=dataset[split]['source'][indices], variant=dataset[split]['variant'][indices],
            source_row=dataset[split]['source_row'][indices], source_indices=indices)
    folds = stratified_folds(data['train']['family'], data['train']['source'])
    if folds != binding['family_folds']:
        raise ValueError('Source-stratified family partition differs')
    oof = np.full((len(data['train']['truth']), 1 if task['variant'] == 'legacy' else 5), np.nan)
    counts = np.zeros(len(oof), dtype=np.int64)
    validation = None
    outcomes = []
    for name in ('cv_0', 'cv_1', 'cv_2', 'refit'):
        fit = read_json(folder/name/'completed.json')
        if (fit['binding'] != binding or fit['status'] != 'completed' or
                fit['steps'] != declaration['steps'] or fit['variant'] != task['variant'] or
                fit['seed'] != task['seed'] or sha(folder/name/'resume.pt') != fit['checkpoint_sha256']):
            raise ValueError('Fit checkpoint/completion binding mismatch')
        outcomes.append(fit)
        with np.load(folder/name/'predictions.npz', allow_pickle=False) as archive:
            pred = {key: archive[key].copy() for key in archive.files}
        refit = name == 'refit'
        target_data = data['validation' if refit else 'train']
        target = (np.arange(len(target_data['truth'])) if refit else
                  np.flatnonzero(np.isin(data['train']['family'], folds[int(name[-1])])) )
        training = (np.arange(len(data['train']['truth'])) if refit else
                    np.flatnonzero(~np.isin(data['train']['family'], folds[int(name[-1])])))
        if (not np.array_equal(pred['target_indices'], target) or
                not np.array_equal(pred['train_indices'], training) or
                not np.array_equal(pred['family'], target_data['family'][target]) or
                not np.allclose(pred['truth'], target_data['truth'][target], rtol=2e-6, atol=1e-6, equal_nan=True) or
                pred['prediction'].shape != (len(target), oof.shape[1]) or
                not np.isfinite(pred['prediction']).all() or
                fit['train_rows'] != len(training) or fit['target_rows'] != len(target)):
            raise ValueError('Saved prediction shape/indices/physical truth mismatch')
        if refit:
            validation = pred['prediction']
        else:
            oof[target] = pred['prediction']
            counts[target] += 1
    if done['outcomes'] != outcomes or not np.all(counts == 1) or not np.isfinite(oof).all():
        raise ValueError('OOF rows not exactly disjoint/exhaustive or root completion differs')
    data['train']['prediction'] = oof
    data['validation']['prediction'] = validation
    return data, dict(status='completed', completed_fits=4, root_sha256=sha(completed))


def object_file_sha(matrix):
    # matrix is deliberately not mutated with a helper metadata field.
    plan_path = Path(matrix['plan']['path'])
    return sha(plan_path.with_name('matrix.json'))


def csv_text(rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(clean(v)) if isinstance(v, (list, tuple, dict)) else clean(v)
                         for k, v in row.items()})
    return output.getvalue()


def report(args):
    from itertools import combinations
    from .qstudy_report import describe, diagnostics
    matrix = load_matrix(args.output, verify_implementation=False)
    declaration = load_plan(args.output)['declaration']
    metrics, predictions, task_rows, pools, arm_audit = [], [], [], {}, {}
    for arm, bindings in matrix['datasets'].items():
        arm_audit[arm] = read_json(bindings['manifest']['path'])['source_counts']
    for task in matrix['tasks']:
        try:
            data, state = checked_predictions(task, matrix, declaration)
            task_rows.append({**task, **state})
        except (ValueError, KeyError, OSError, IndexError) as exc:
            task_rows.append({**task, 'status': 'integrity_error', 'error': str(exc)})
            continue
        if data is None:
            continue
        for split in ('train', 'validation'):
            source_data = data[split]
            stage = 'train_cv_oof' if split == 'train' else 'external_validation'
            for source in SOURCE_NAMES:
                selected = source_data['source'] == source
                for base_only in (False, True):
                    keep = selected & ((source_data['variant'] == 0) if base_only else True)
                    indices = np.flatnonzero(keep)
                    if not len(indices):
                        continue
                    scope = stage+'_'+source+('_base' if base_only else '')
                    p, y, families = [source_data[k][indices] for k in ('prediction', 'truth', 'family')]
                    key = (task['arm'], task['variant'], task['seed'], scope)
                    ids = [(source, int(source_data['family'][i]), int(source_data['variant'][i])) for i in indices]
                    if len(set(ids)) != len(ids):
                        raise ValueError('Source family/variant identity duplicates in report pool')
                    pools[key] = dict(ids=ids, prediction=p, truth=y, family=families)
                    for aggregation in ('rows', 'family_means'):
                        for k in range(p.shape[1]):
                            if aggregation == 'rows':
                                ap, ay, af = p[:, k], y[:, k], families
                            else:
                                valid = np.isfinite(p[:, k]) & np.isfinite(y[:, k])
                                af = np.unique(families[valid])
                                ap = np.asarray([p[valid & (families == f), k].mean() for f in af])
                                ay = np.asarray([y[valid & (families == f), k].mean() for f in af])
                            metrics.append(dict(arm=task['arm'], variant=task['variant'], seed=task['seed'],
                                scope=scope, aggregation=aggregation, metric=METRIC_NAMES[k],
                                cost_definition=COST_DEFINITIONS[k],
                                **diagnostics(ap, ay, af)))
                    if not base_only:
                        for index, identity in enumerate(ids):
                            predictions.append(dict(arm=task['arm'], variant=task['variant'], seed=task['seed'],
                                scope=scope, source=source, family=identity[1], coordinate_variant=identity[2],
                                prediction_costs=p[index].tolist(), measured_costs=y[index].tolist()))
    summary = []
    grouped = defaultdict(list)
    for row in metrics:
        grouped[tuple(row[k] for k in ('arm', 'variant', 'scope', 'aggregation', 'metric'))].append(row)
    for key, rows in sorted(grouped.items()):
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('Duplicate independent critic seed in source summary')
        result = dict(zip(('arm', 'variant', 'scope', 'aggregation', 'metric'), key),
                      seeds=sorted(r['seed'] for r in rows), n_seeds=len(rows))
        for statistic in STATS:
            result.update({statistic+'_'+k: v for k, v in describe([r.get(statistic) for r in rows]).items()})
        summary.append(result)
    paired = []
    comparisons = []
    for arms in (ARMS[:3], ARMS[3:]):
        for baseline, candidate in combinations(arms, 2):
            for variant in VARIANTS:
                comparisons.append(('data_source', baseline, variant, candidate, variant))
    for arm in ARMS:
        comparisons.append(('critic', arm, 'legacy', arm, 'huber_residual'))
    scopes = sorted({key[3] for key in pools})
    for kind, baseline, bv, candidate, cv in comparisons:
        for seed in SEEDS:
            for scope in scopes:
                a, b = pools.get((baseline, bv, seed, scope)), pools.get((candidate, cv, seed, scope))
                if a is None or b is None:
                    continue
                shared = sorted(set(a['ids']) & set(b['ids']))
                if not shared:
                    continue
                ai = np.asarray([a['ids'].index(key) for key in shared])
                bi = np.asarray([b['ids'].index(key) for key in shared])
                if not np.array_equal(a['truth'][ai], b['truth'][bi], equal_nan=True):
                    raise ValueError('Matched source comparison physical labels differ')
                family = a['family'][ai]
                for k in range(min(a['prediction'].shape[1], b['prediction'].shape[1])):
                    ad = diagnostics(a['prediction'][ai, k], a['truth'][ai, k], family)
                    bd = diagnostics(b['prediction'][bi, k], b['truth'][bi, k], family)
                    for statistic in STATS:
                        if ad.get(statistic) is None or bd.get(statistic) is None:
                            continue
                        paired.append(dict(comparison=kind, baseline=baseline, baseline_variant=bv,
                            candidate=candidate, candidate_variant=cv, seed=seed, scope=scope,
                            metric=METRIC_NAMES[k], statistic=statistic, n_shared_targets=len(shared),
                            baseline_value=ad[statistic], candidate_value=bd[statistic],
                            delta=bd[statistic]-ad[statistic], lower_delta_better=statistic in ('mae', 'top_regret')))
    paired_summary, grouped = [], defaultdict(list)
    pairkeys = ('comparison', 'baseline', 'baseline_variant', 'candidate', 'candidate_variant', 'scope', 'metric', 'statistic')
    for row in paired:
        grouped[tuple(row[k] for k in pairkeys)].append(row)
    for key, rows in sorted(grouped.items()):
        paired_summary.append(dict(zip(pairkeys, key), **describe([r['delta'] for r in rows]),
            seeds=[r['seed'] for r in rows], deltas=[r['delta'] for r in rows],
            shared_targets_per_seed=[r['n_shared_targets'] for r in rows],
            lower_delta_better=rows[0]['lower_delta_better']))
    complete = sum(r['status'] == 'completed' for r in task_rows)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid())
    folder = args.output/'report'/stamp
    folder.mkdir(parents=True, exist_ok=False)
    tables = dict(tasks=task_rows, per_seed_metrics=metrics, seed_summary=summary,
                  paired_seed_metrics=paired, paired_seed_summary=paired_summary, predictions=predictions)
    for name, rows in tables.items():
        atomic_json(folder/(name+'.json'), rows)
        atomic_text(folder/(name+'.csv'), csv_text(rows))
    result = dict(status='complete' if complete == len(matrix['tasks']) else 'partial_snapshot',
        completed_tasks=complete, planned_tasks=len(matrix['tasks']),
        integrity_errors=[r for r in task_rows if r['status'] == 'integrity_error'],
        matrix=bind(args.output/'matrix.json'), plan=matrix['plan'], source_counts=arm_audit,
        metric_names=list(METRIC_NAMES[:5]), prediction_semantics=[
            'native GRT WL_um', 'max(-nativeWNS_ns,0)', 'max(-nativeTNS_ns,0)', 'Hoverflow', 'Voverflow'],
        missing_metrics='Absent legacy timing/congestion predictions are NA, not zero',
        statistic_unit='Each criticseed supplies one estimate; sampleSD across matched completed criticseeds only',
        pair_definition='Candidate minus baseline on identical source+family+variant target intersections; OOF different-source targets are never treated as paired',
        no_pooled_fold_utility=True, uncertainty_not_calibrated=True,
        automatic_promotion=False, actor_training=False, backend_calls=0,
        interpretation='Predictive source-distribution diagnostics only; no gradient-validity or placement-gain claim',
        report_source_sha256=sha(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(folder/'summary.json', result)
    text = ('# Q data-source controls\n\n'
        f"Completed tasks: {complete}/{len(matrix['tasks'])}. Snapshot: {result['status']}.\n\n"
        'Every arm uses the same old16+teacher16 validation candidates, reported separately. '
        'Base-only sensitivity has32TRAIN labels; main arms have64. '
        'Failures remain counted but do not supply invented numericPPA.\n\n'
        'See seed_summary.csv for per-source physical-cost scores and paired_seed_summary.csv '
        'for identical-target, matched-seed deltas. Missing models/seeds are explicit in tasks.csv. '
        'TRAIN-CV source pools differ across arms; pairedOOF deltas use only shared original targets. '
        'No source or actor is automatically selected.\n')
    atomic_text(folder/'report.md', text)
    atomic_json(args.output/'report/latest.json', dict(path=str(folder), summary_sha256=sha(folder/'summary.json')))
    print(json.dumps(dict(status=result['status'], completed=complete, planned=len(matrix['tasks']),
                         integrity_errors=len(result['integrity_errors']), report=str(folder))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'check', 'assemble', 'train', 'report'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-dataset', type=Path, default=DEFAULT_OLD)
    parser.add_argument('--teacher-experiment', type=Path, default=DEFAULT_TEACHER)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--task-index', type=int)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Source planning/assembly/fitting/reporting must run through Slurm')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ('plan', 'assemble'):
        with (args.output/'.plan.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            plan(args) if args.stage == 'plan' else assemble(args)
    elif args.stage == 'train':
        train(args)
    elif args.stage == 'check':
        check(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
