"""Saved-only report of all eight fixed expert-rule critic objectives.

Each task is admitted only through screen.checked_refit (all four fits). No
neural forward, fitting, selection, actor advancement or backend invocation.
Different fold scales are never pooled, and different target definitions do
not provide a ranking of physical policy quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch

# Pin the local namespace before upstream loaders change sys.path.
from analysis.placement_metrics import MacroHPWL, legality
from analysis.qstudy_rule_audit import describe as geometry_describe, rules, structure
from fql import qstudy_rule_reward_screen as screen
from fql.graph_o2o_archive import atomic_json
from fql.qstudy_data import METRIC_NAMES, bind, sha
from fql.qstudy_report import clean, csv_text, describe, diagnostics


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'qstudy_rule_reward_results_v1'
SCOPES = ('train_family_cv', 'historical_validation_previously_inspected')
STATISTICS = ('mae', 'spearman', 'kendall', 'family_pair_accuracy', 'top_regret')
CHANGE_TOLERANCE = 1e-7
SOURCE_FILES = ('analysis/qstudy_rule_reward_results.py',
                'analysis/qstudy_rule_reward_results.slurm', 'fql/qstudy_report.py')


def read(path):
    return json.loads(Path(path).read_text())


def equal(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    numeric = actual.dtype.kind in 'fc' or expected.dtype.kind in 'fc'
    if not np.array_equal(actual, expected, equal_nan=numeric):
        raise ValueError('Saved result correspondence differs: '+name)


def state_sha(value):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError('Expected saved CPU RNG byte tensor')
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def family_mean_diagnostics(prediction, truth, families):
    rows = []
    for family in np.unique(families):
        keep = (families == family) & np.isfinite(prediction) & np.isfinite(truth)
        if keep.any():
            rows.append((family, prediction[keep].mean(), truth[keep].mean()))
    return diagnostics(np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows]),
                       np.asarray([r[0] for r in rows]))


def fit_evidence(folder, fold, task, plan, data, context):
    """Additional saved-array/statistic and paired-stream checks; never forward."""
    done = read(folder/'completed.json')
    expected_scope = SCOPES[1] if fold['name'] == 'refit' else SCOPES[0]
    if done['scope'] != expected_scope:
        raise ValueError('Saved diagnostic TRAIN/historical scope differs')
    train, target = np.asarray(fold['train_indices']), np.asarray(fold['target_indices'])
    destination = data['validation'] if fold['name'] == 'refit' else context
    spec, scale_audit = screen.specification(context, train, task['objective'])
    train_labels, target_labels = screen.labels(context, train, spec), screen.labels(destination, target, spec)
    train_valid = train_labels['valid'].numpy()
    admitted = train[train_valid]
    truth = target_labels['cost'].numpy()
    with np.load(folder/'predictions.npz', allow_pickle=False) as source:
        arrays = {k:source[k].copy() for k in source.files}
    expected = dict(admitted_train_indices=admitted, train_native_metrics=context['native'][train],
        train_native_metric_mask=context['mask'][train], train_status=context['statuses'][train],
        train_valid=train_valid, train_truth=train_labels['cost'][train_labels['valid']].float().numpy(),
        train_truth_all=train_labels['cost'].numpy())
    for name, value in expected.items():
        equal(arrays[name], value, name)
    for key, n in (('prediction', len(target)), ('train_prediction', len(admitted)),
                   ('train_prediction_all', len(train))):
        if arrays[key].shape != (n,) or not np.isfinite(arrays[key]).all():
            raise ValueError('Saved prediction shape/finite requirement differs: '+key)
    equal(arrays['train_prediction_all'][train_valid], arrays['train_prediction'], 'TRAIN admitted prediction')
    constant = float(train_labels['cost'][train_labels['valid']].float().mean())
    equal(arrays['constant_prediction'], np.full(len(target), constant), 'TRAIN constant comparator')
    metrics = diagnostics(arrays['prediction'], truth, arrays['family'])
    baseline = diagnostics(arrays['constant_prediction'], truth, arrays['family'])
    if done['critic'] != metrics or done['constant'] != baseline or done['constant_train_B'] != constant:
        raise ValueError('Stored diagnostics disagree with saved predictions')
    valid = target_labels['valid'].numpy()
    failure = target_labels['failure'].numpy()
    counts = dict(train_rows=len(train), admitted_train_rows=len(admitted),
        excluded_train_rows=int((~train_valid).sum()), target_rows=len(target),
        valid_target_rows=int(valid.sum()), invalid_target_rows=int((~valid).sum()))
    if any(done[k] != v for k, v in counts.items()):
        raise ValueError('Saved validity denominator differs')
    p, success = arrays['prediction'], target_labels['success'].numpy() & valid
    ranges = dict(minimum=float(p.min()), maximum=float(p.max()),
        success_below_zero=int(((p < 0) & success).sum()),
        success_at_or_above_two=int(((p >= 2) & success).sum()),
        any_below_zero=int((p < 0).sum()), any_above_four=int((p > 4).sum()))
    if any(done['prediction_range'][k] != v for k, v in ranges.items()):
        raise ValueError('Saved prediction range differs')
    gradient = done['input_gradients']
    if (gradient['finite'] is not True or gradient['rows'] != len(target) or
        not 0 <= gradient['nonzero_rows'] <= len(target) or
        any(not np.isfinite(gradient[k]) or gradient[k] < 0 for k in ('rms_mean', 'rms_maximum'))):
        raise ValueError('Saved raw-input gradient diagnostics invalid')
    checkpoint = torch.load(folder/'resume.pt', map_location='cpu', weights_only=False)
    run = done['binding']['run']
    paired = dict(seed=task['seed'], fit=fold['name'], step=checkpoint['step'],
        train_indices=fold['train_indices'], admitted_train_indices=admitted.tolist(),
        target_indices=fold['target_indices'], train_source_ids=done['binding']['train_source_ids'],
        target_source_ids=done['binding']['target_source_ids'],
        scale_identity_sha256=screen.scale_identity(spec),
        batch_rng_sha256=state_sha(checkpoint['batch_rng']), global_rng_sha256=state_sha(checkpoint['rng']),
        config=done['binding']['config'], environment={k:run[k] for k in ('threads','torch_version','numpy_version')})
    return dict(objective=task['objective'], seed=task['seed'], fit=fold['name'], scope=done['scope'],
        critic=metrics, constant=baseline, constant_train_B=constant,
        family_mean=family_mean_diagnostics(p, truth, arrays['family']),
        utility_spec=spec, scale_audit=scale_audit, normalization=done['normalization'],
        validity=dict(**counts, confirmed_failure=int(failure.sum())),
        prediction_range=ranges, input_gradients=gradient, paired_stream=paired,
        bindings=dict(completion=bind(folder/'completed.json'), checkpoint=bind(folder/'resume.pt'),
                      predictions=bind(folder/'predictions.npz')))


def seed_summaries(fits):
    result = []
    for objective in screen.OBJECTIVES:
        for seed in range(3):
            for scope in SCOPES:
                selected = [r for r in fits if (r['objective'], r['seed'], r['scope']) == (objective, seed, scope)]
                expected = {'cv_0', 'cv_1', 'cv_2'} if scope == SCOPES[0] else {'refit'}
                if {r['fit'] for r in selected} != expected:
                    continue
                item = dict(objective=objective, seed=seed, scope=scope, folds=len(selected),
                    aggregation='equal mean of within-fold statistics; never pooled B' if scope == SCOPES[0]
                                else 'single full-TRAIN refit; historical16 previously inspected')
                for arm in ('critic', 'constant', 'family_mean'):
                    item[arm] = {k:float(np.mean([r[arm][k] for r in selected]))
                        if all(r[arm][k] is not None for r in selected) else None for k in STATISTICS}
                    count = sum(r[arm]['family_pairs'] for r in selected)
                    item[arm]['within_fold_pairs'] = count
                    item[arm]['within_fold_pair_accuracy'] = (sum(r[arm]['family_pair_accuracy']*r[arm]['family_pairs']
                        for r in selected if r[arm]['family_pair_accuracy'] is not None)/count if count else None)
                item['validity'] = {k:sum(r['validity'][k] for r in selected)
                    for k in ('target_rows', 'valid_target_rows', 'invalid_target_rows', 'confirmed_failure')}
                item['prediction_range'] = {k:sum(r['prediction_range'][k] for r in selected)
                    for k in ('success_below_zero', 'success_at_or_above_two', 'any_below_zero', 'any_above_four')}
                item['input_gradients'] = dict(rows=sum(r['input_gradients']['rows'] for r in selected),
                    nonzero_rows=sum(r['input_gradients']['nonzero_rows'] for r in selected),
                    equal_fold_rms_mean=float(np.mean([r['input_gradients']['rms_mean'] for r in selected])),
                    maximum=max(r['input_gradients']['rms_maximum'] for r in selected))
                result.append(item)
    return result


def aggregate(seeds):
    rows = []
    for objective in screen.OBJECTIVES:
        for scope in SCOPES:
            matched = [r for r in seeds if r['objective'] == objective and r['scope'] == scope]
            for arm in ('critic', 'constant', 'family_mean'):
                for statistic in (*STATISTICS, 'within_fold_pair_accuracy'):
                    valid = [r for r in matched if r[arm][statistic] is not None]
                    values = [r[arm][statistic] for r in valid]
                    if arm == 'constant' and values:
                        if any(v != values[0] for v in values[1:]):
                            raise ValueError('Deterministic comparator changed between replica seeds')
                        values = values[:1]
                    rows.append(dict(objective=objective, scope=scope, arm=arm, statistic=statistic,
                        seeds=[r['seed'] for r in valid] if arm != 'constant' else [],
                        verified_replica_seeds=[r['seed'] for r in valid] if arm == 'constant' else None,
                        planned_seeds=[0,1,2], independent_training_runs=len(valid) if arm != 'constant' else 0,
                        all_three_seeds_available=len(valid) == 3, **describe(values),
                        uncertainty='one deterministic comparator; no seed SD' if arm == 'constant' else
                                    'sample SD across critic-training seeds on one fixed dataset/fold plan'))
    return rows


def paired_streams(fits):
    rows = []
    for seed in range(3):
        for fold in ('cv_0', 'cv_1', 'cv_2', 'refit'):
            selected = [r for r in fits if r['seed'] == seed and r['fit'] == fold]
            same = not selected or all(r['paired_stream'] == selected[0]['paired_stream'] for r in selected[1:])
            if not same:
                raise ValueError('Paired objective fold/scale/minibatch-RNG/environment identity differs')
            rows.append(dict(seed=seed, fit=fold, verified_objectives=[r['objective'] for r in selected],
                expected_objectives=list(screen.OBJECTIVES), all_eight_available=len(selected) == 8,
                identities_equal=same, identity=selected[0]['paired_stream'] if selected else None,
                interpretation='source-deterministic same recipe, admitted rows, budgets and saved RNG endpoints; full batch drawchain was not separately logged'))
    return rows


def geometry_audit(data, context):
    """TRAIN-only executed-label variation and paired map; no native associations."""
    train = data['train']
    raw = train['x'].detach().cpu().numpy()
    executed = np.asarray(train['executed']).reshape(raw.shape)
    movable = context['movable'].cpu().numpy().astype(bool)
    if not movable.any():
        raise ValueError('No movable macros for geometry audit')
    delta = (executed-raw)[:, movable]
    per_macro = np.sqrt((delta**2).mean(-1))
    per_row = np.sqrt((delta**2).mean((1,2)))
    ctx, metadata = structure(context['cache'])
    hpwl = MacroHPWL(context['cache'])
    raw_scores = [rules(layout, ctx, hpwl) for layout in raw]
    executed_scores = [rules(layout, ctx, hpwl) for layout in executed]
    raw_rules = np.asarray([[score[name] for name in screen.COST_NAMES[1:]] for score in raw_scores], float)
    executed_rules = np.asarray([[score[name] for name in screen.COST_NAMES[1:]] for score in executed_scores], float)
    equal(executed_rules, train['cost_values'][:,1:], 'executed geometric cost labels')
    group = executed_rules[:, list(screen.COST_NAMES[1:]).index('cluster_bbox_fraction')]
    unique, counts = np.unique(group, return_counts=True)
    mode = int(counts.argmax())
    family_rms = [float(per_row[train['families'] == family].mean()) for family in np.unique(train['families'])]
    changed = (np.abs(delta).max(-1) > CHANGE_TOLERANCE)
    row_records = [dict(index=i, source_id=train['source_ids'][i], family=int(train['families'][i]),
        movable_coordinate_rms=float(per_row[i]), changed_macros=int(changed[i].sum()),
        raw_costs=dict(zip(screen.COST_NAMES[1:], raw_rules[i].tolist())),
        executed_costs=dict(zip(screen.COST_NAMES[1:], executed_rules[i].tolist()))) for i in range(len(raw))]
    legal = {label:[legality(layout, hpwl.sizes) for layout in layouts]
             for label,layouts in [('raw',raw),('executed',executed)]}
    return dict(scope='TRAIN only; no validation or native association selects rules', rows=len(raw),
        families=len(np.unique(train['families'])), total_nodes=raw.shape[1], movable_nodes=int(movable.sum()),
        coordinate_unit='model normalized [-1,1] coordinates; RMS averages x and y of movable nodes',
        change_tolerance=CHANGE_TOLERANCE, row_rms=geometry_describe(per_row),
        family_mean_row_rms=geometry_describe(family_rms), macro_rms=geometry_describe(per_macro.reshape(-1)),
        changed_rows=int(changed.any(1).sum()), changed_macros=int(changed.sum()),
        raw_range=[float(raw.min()),float(raw.max())], executed_range=[float(executed.min()),float(executed.max())],
        legal_counts={name:dict(legal=sum(row['legal'] for row in values), rows=len(values)) for name,values in legal.items()},
        executed_costs={name:geometry_describe(train['cost_values'][:,k]) for k,name in enumerate(screen.COST_NAMES)},
        raw_minus_executed_rule_deltas={name:geometry_describe(raw_rules[:,k]-executed_rules[:,k])
            for k,name in enumerate(screen.COST_NAMES[1:])},
        group_saturation=dict(non_singleton_clusters=metadata['non_singleton_clusters'],
            clusters=metadata['clusters'], unique_values=len(unique),
            mode_value=float(unique[mode]), exact_mode_count=int(counts[mode]),
            exact_mode_fraction=float(counts[mode]/len(group)),
            quantiles={str(p):float(np.quantile(group,p)) for p in (0,.25,.5,.75,1)},
            interpretation='geometric range and repeated exact values; no threshold-based selection, derivative or PPA claim'),
        per_row=row_records, caveat='True GRT labels belong to executed/snapped layouts, critic inputs are raw; raw GRT was not measured and map is not differentiated')


def native_summary(data):
    result = {}
    for split in ('train', 'validation'):
        source = data[split]
        result[split] = dict(rows=len(source['x']), families=len(np.unique(source['families'])),
            status_counts={str(s):int((source['statuses'] == s).sum()) for s in np.unique(source['statuses'])},
            metrics={name:geometry_describe(np.where(source['mask'][:,k],source['native'][:,k],np.nan))
                for k,name in enumerate(METRIC_NAMES)})
    return result


def collect(plan_path):
    plan_path = Path(plan_path).resolve()
    loaded = screen.checked_plan(plan_path)
    plan, data, context = loaded
    fits, tasks, evidence = [], [], {str(plan_path):sha(plan_path)}
    for task in plan['tasks']:
        folder = Path(plan['output'])/task['objective']/f"seed_{task['seed']}"
        status = dict(**task, status='missing', missing_fits=[f['name'] for f in plan['folds']
            if not (folder/f['name']/'completed.json').exists()], error=None)
        if (folder/'completed.json').exists():
            try:
                verified = screen.checked_refit(plan_path,task['objective'],task['seed'],loaded=loaded)
                task_fits = [fit_evidence(folder/f['name'],f,task,plan,data,context) for f in plan['folds']]
                for path,digest in verified['evidence'].items():
                    bind(path,digest)
                evidence.update(verified['evidence'])
                fits.extend(task_fits)
                status.update(status='completed', missing_fits=[], evidence=verified['evidence'])
            except (ValueError,KeyError,FileNotFoundError,RuntimeError,OSError,EOFError,pickle.UnpicklingError) as exc:
                status.update(status='integrity_error', error=str(exc))
        tasks.append(status)
    pairs = paired_streams(fits)
    seeds = seed_summaries(fits)
    scales = []
    for pair in pairs:
        matched = [r for r in fits if r['seed'] == pair['seed'] and r['fit'] == pair['fit']]
        if not matched:
            continue
        row = matched[0]
        for k,name in enumerate(screen.COST_NAMES):
            scales.append(dict(seed=pair['seed'],fit=pair['fit'],cost=name,
                units=screen.rule_definition()['units'][k],scale=row['utility_spec']['scales'][k],
                observed_train_rows=row['scale_audit']['observed_counts'][k],
                train_span=row['scale_audit']['train_span'][k],
                reference=row['utility_spec']['reference'],
                shared_objectives=pair['verified_objectives'],all_eight_available=pair['all_eight_available']))
    return dict(schema=SCHEMA, status='integrity_error' if any(t['status']=='integrity_error' for t in tasks)
        else 'completed' if len(fits)==96 and all(t['status']=='completed' for t in tasks) else 'partial',
        plan=bind(plan_path), inputs=plan['inputs'], screen_source_sha256=plan['source_sha256'],
        expected_tasks=24, expected_fits=96, completed_tasks=sum(t['status']=='completed' for t in tasks),
        verified_fits=len(fits), integrity_errors=sum(t['status']=='integrity_error' for t in tasks),
        tasks=tasks, fits=fits, seed_summaries=seeds, multi_seed=aggregate(seeds),
        paired_streams=pairs, train_only_scales=scales,
        geometry=geometry_audit(data,context), native_fixed_pools=native_summary(data),
        evidence=evidence, all_objectives_retained=list(screen.OBJECTIVES),
        conventions=dict(cv='within-fold diagnostics then equal-fold mean per seed; B scales are NEVER pooled',
            families='base/perturb rows are dependent; family-mean rank diagnostics supplement row metrics',
            seed_sd='independent critic training seeds on one fixed dataset/fold plan, not independent designs or calibrated confidence',
            targets='eight B definitions differ; prediction metrics do not establish physical policy improvement or choose an objective',
            historical16='previously inspected validation, separate full-TRAIN refit diagnostic; never weight fitting or target selection',
            ties='true family ties excluded; predicted pair ties incorrect; constant ranking undefined',
            normalization='success-only means on each fold TRAIN, shared across objectives; no second B normalization',
            failure='confirmed physical/geometry failure B4; unresolved/infrastructure invalid; no invented PPA',
            gradients='saved raw-input differentiability only, not legalizer or backend directional accuracy',
            selection='no actor promotion, best checkpoint, CV objective filtering, fresh noise or backend calls'),
        model_inference_calls=0, backend_calls=0, actor_advanced=False, heldout_used=False)


def write_snapshot(plan_path, output):
    plan_path,output=Path(plan_path).resolve(),Path(output).resolve()
    plan=read(plan_path)
    for protected in (Path(plan['output']).resolve(),Path(plan['inputs']['dataset_path']).resolve().parent):
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError('Read-only report output must be separate from source fits/dataset')
    result=collect(plan_path)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    snapshot=output/(stamp+'_job'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True,exist_ok=False)
    sources={name:sha(ROOT/name) for name in SOURCE_FILES}
    result.update(snapshot=str(snapshot),slurm_job_id=os.environ['SLURM_JOB_ID'],reporter_sources=sources)
    for name,digest in sources.items():
        destination=snapshot/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
        bind(destination,digest)
    (snapshot/'plan.json').write_bytes(plan_path.read_bytes())
    bind(snapshot/'plan.json',result['plan']['sha256'])
    atomic_json(snapshot/'report.json',clean(result))
    for name in ('tasks','fits','seed_summaries','multi_seed','paired_streams','train_only_scales'):
        (snapshot/(name+'.csv')).write_text(csv_text(result[name]))
    (snapshot/'train_geometry_rows.csv').write_text(csv_text(result['geometry']['per_row']))
    files={str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*') if path.is_file()}
    atomic_json(snapshot/'completed.json',dict(status='completed',report_status=result['status'],files=files,
        plan=result['plan'],model_inference_calls=0,backend_calls=0,actor_advanced=False))
    atomic_json(output/'latest.json',dict(snapshot=str(snapshot),report=bind(snapshot/'report.json'),
        completion=bind(snapshot/'completed.json'),status=result['status']))
    return dict(status=result['status'],snapshot=str(snapshot),report=bind(snapshot/'report.json'),
        completed_tasks=result['completed_tasks'],verified_fits=result['verified_fits'],integrity_errors=result['integrity_errors'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Saved result aggregation requires CPU Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    print(json.dumps(write_snapshot(args.plan,args.output)),flush=True)


if __name__=='__main__':main()
