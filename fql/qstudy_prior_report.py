"""Read-only physical-prior controls for the completed fixed Q-screen matrix.

Only one-feature closed-form TRAIN calibrations are fitted. No neural training,
actor update, legalization, new backend evaluation, or heldout data is used.
The smooth control exactly reproduces the analytic component of huber_residual.
Run the report through Slurm; software tests use tiny synthetic fixtures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .physical_ridge import PhysicalRidgeCritic
from .qstudy_data import load_dataset as strict_load_dataset
from .qstudy_models import _masked_stats, smooth_macro_hpwl
from . import qstudy_report as report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/prior_report_v1')
PRIOR_NAMES = ('train_mean', 'smooth_hpwl_calibrated', 'exact_hpwl_calibrated',
               'smooth_hpwl_rank_only', 'exact_hpwl_rank_only')
NATIVE_PRIORS = frozenset(PRIOR_NAMES[:3])
STATISTICS = ('mae', 'normalized_mae', 'spearman', 'kendall', 'family_pair_accuracy', 'top_regret')


def calibrate(train_feature, train_cost, *, ridge=.001, feature_std_floor=1e-5,
              target_relative_floor=.03):
    """The existing residual critic's scalar TRAIN-only ridge, in float32.

    h is HPWL/cache_canvas.sum(); z=(h-mean(h))/std(h), with std=1 when
    population std<=1e-5. y=(cost-mean(cost))/max(std(cost),.03*mean(abs(cost))).
    b=sum((z-zbar)*(y-ybar))/(sum((z-zbar)^2)+.001); a=ybar-b*zbar.
    Prediction=target_mean+target_std*(a+b*z). No positive-slope constraint.
    """
    h = torch.as_tensor(train_feature, dtype=torch.float32).reshape(-1)
    c = torch.as_tensor(train_cost, dtype=torch.float32).reshape(-1)
    if len(h) != len(c) or len(h) < 2 or not torch.isfinite(h).all() or not torch.isfinite(c).all():
        raise ValueError('Calibration needs at least two finite matching TRAIN rows')
    if (c < 0).any() or ridge < 0 or feature_std_floor <= 0 or target_relative_floor <= 0:
        raise ValueError('Invalid TRAIN costs or calibration parameters')
    mean, std, objective_scale, _ = _masked_stats(c[:, None], torch.ones((len(c), 1), dtype=torch.bool),
                                               relative_floor=target_relative_floor)
    h_mean, h_std = h.mean(), h.std(unbiased=False)
    h_std = h_std if h_std > feature_std_floor else h_std.new_ones(())
    x, y = (h-h_mean)/h_std, (c-mean[0])/std[0]
    xc, yc = x-x.mean(), y-y.mean()
    slope = (xc*yc).sum()/(xc.square().sum()+ridge).clamp_min(1e-12)
    intercept = y.mean()-slope*x.mean()
    return dict(train_rows=len(c), hpwl_mean=float(h_mean), hpwl_std=float(h_std),
                target_mean=float(mean[0]), target_std=float(std[0]),
                objective_scale=float(objective_scale[0]), slope=float(slope), intercept=float(intercept),
                baseline_ridge=float(ridge), feature_std_floor=float(feature_std_floor),
                target_relative_floor=float(target_relative_floor))


def calibrated_prediction(model, feature):
    h = torch.as_tensor(feature, dtype=torch.float32)
    z = (h-model['hpwl_mean'])/model['hpwl_std']
    return (model['target_mean']+model['target_std']*(model['intercept']+model['slope']*z)).numpy()


def analytic_features(cache, raw_coordinates):
    """Use the very same raw-action/cache ordering as screen.fit, without repair."""
    zero = dict(mean=np.zeros(17), std=np.ones(17), coeff=np.zeros(17), intercept=0.)
    extractor = PhysicalRidgeCritic(cache, zero).float()
    actions = torch.as_tensor(raw_coordinates, dtype=torch.float32).reshape(len(raw_coordinates), -1, 2)
    if actions.shape[1] != len(extractor.hpwl.names) or not torch.isfinite(actions).all():
        raise ValueError('Invalid raw coordinates or cache node ordering dimension')
    with torch.no_grad():
        exact = extractor.hpwl(actions)
        smooth = smooth_macro_hpwl(extractor, actions, .05)
        denominator = extractor.hpwl.canvas.sum()
    return dict(exact_hpwl=exact.numpy(), smooth_hpwl=smooth.numpy(),
                exact_feature=(exact/denominator).numpy(), smooth_feature=(smooth/denominator).numpy()), \
           extractor.hpwl.metadata()


def prior_predictions(train_feature, validation_feature, train_cost, train_families, folds):
    """One deterministic OOF/refit experiment, not repeated per neural seed."""
    n, nv = len(train_cost), len(validation_feature['exact_feature'])
    predictions = {name: dict(train=np.full(n, np.nan), validation=np.full(nv, np.nan))
                   for name in PRIOR_NAMES}
    calibration_rows, fold_rows = [], []
    seen = np.zeros(n, dtype=np.int64)
    for fold in range(len(folds)+1):
        refit = fold == len(folds)
        held = [] if refit else folds[fold]
        ti = np.arange(n) if refit else np.flatnonzero(~np.isin(train_families, held))
        vi = np.arange(nv) if refit else np.flatnonzero(np.isin(train_families, held))
        if not len(ti) or not len(vi):
            raise ValueError('Empty analytic TRAIN/target fold')
        if not refit:
            if set(train_families[ti]) & set(train_families[vi]):
                raise ValueError('Analytic calibration family leakage')
            seen[vi] += 1
        split = 'validation' if refit else 'train'
        source = validation_feature if refit else train_feature
        constant = float(torch.as_tensor(train_cost[ti], dtype=torch.float32).mean())
        predictions['train_mean'][split][vi] = constant
        fold_rows.append(dict(fit='refit' if refit else f'cv_{fold}', train_indices=ti.tolist(),
                              target_indices=vi.tolist(), train_mean=constant))
        for kind in ('smooth', 'exact'):
            model = calibrate(train_feature[kind+'_feature'][ti], train_cost[ti])
            predictions[kind+'_hpwl_calibrated'][split][vi] = calibrated_prediction(model, source[kind+'_feature'][vi])
            predictions[kind+'_hpwl_rank_only'][split][vi] = source[kind+'_hpwl'][vi]
            calibration_rows.append(dict(fit='refit' if refit else f'cv_{fold}', prior=kind+'_hpwl_calibrated', **model))
    if not np.all(seen == 1) or any(not np.isfinite(x).all() for p in predictions.values() for x in p.values()):
        raise ValueError('Analytic OOF predictions are not disjoint/exhaustive and finite')
    return predictions, calibration_rows, fold_rows


def add_metrics(rows, *, dataset, model, seed, scope, prediction, truth, families,
                reporting_scale, native_prediction, fit=None):
    for aggregation in ('rows', 'family_means'):
        p, y, f = prediction, truth, families
        if aggregation == 'family_means':
            p, y, f = report.family_means(p, y, f)
        values = report.diagnostics(p, y, f)
        values['normalized_mae'] = values['mae']/reporting_scale if native_prediction else None
        if not native_prediction:
            values['mae'] = None  # Cache units cannot be compared to native routed um.
        rows.append(dict(dataset=dataset, model=model, seed=seed, scope=scope, fit=fit,
                         aggregation=aggregation, metric='rWL_um', native_prediction=native_prediction,
                         independent_training_runs=0 if seed is None else 1, **values))


def paired_comparisons(metrics):
    priors = {(r['dataset'], r['scope'], r['aggregation'], r['model']): r
              for r in metrics if r['seed'] is None and r['fit'] is None}
    paired = []
    for row in metrics:
        if row['seed'] is None or row['fit'] is not None:
            continue
        for baseline in ('train_mean', 'smooth_hpwl_calibrated', 'exact_hpwl_calibrated'):
            ref = priors[(row['dataset'], row['scope'], row['aggregation'], baseline)]
            for statistic in STATISTICS:
                a, b = row.get(statistic), ref.get(statistic)
                if a is None or b is None:
                    continue
                paired.append(dict(dataset=row['dataset'], model=row['model'], baseline=baseline,
                    seed=row['seed'], scope=row['scope'], aggregation=row['aggregation'], statistic=statistic,
                    learned=a, deterministic_prior=b, delta=a-b,
                    lower_delta_is_better=statistic in {'mae', 'normalized_mae', 'top_regret'}))
    groups = defaultdict(list)
    for row in paired:
        key = tuple(row[k] for k in ('dataset', 'model', 'baseline', 'scope', 'aggregation', 'statistic'))
        groups[key].append(row)
    summary = []
    for key, values in sorted(groups.items()):
        summary.append(dict(zip(('dataset', 'model', 'baseline', 'scope', 'aggregation', 'statistic'), key),
            **report.describe([r['delta'] for r in values]), seeds=[r['seed'] for r in values],
            lower_delta_is_better=values[0]['lower_delta_is_better'],
            uncertainty_unit='independent neural training seeds against one fixed deterministic prior'))
    return paired, summary


def model_summary(metrics):
    groups = defaultdict(list)
    for row in metrics:
        if row['fit'] is not None:
            continue
        key = tuple(row[k] for k in ('dataset', 'model', 'scope', 'aggregation'))
        groups[key].append(row)
    result = []
    for key, values in sorted(groups.items()):
        deterministic = values[0]['seed'] is None
        if deterministic and len(values) != 1:
            raise ValueError('A deterministic prior was duplicated as pseudo-replicates')
        if not deterministic and len({r['seed'] for r in values}) != len(values):
            raise ValueError('A learned seed has duplicated aggregate results')
        for statistic in STATISTICS:
            result.append(dict(zip(('dataset', 'model', 'scope', 'aggregation'), key), statistic=statistic,
                **report.describe([r.get(statistic) for r in values]),
                independent_neural_seeds=0 if deterministic else len(values),
                deterministic_protocols=1 if deterministic else 0,
                uncertainty_unit='none: one deterministic protocol' if deterministic else 'independent neural training seeds',
                seeds=[] if deterministic else [r['seed'] for r in values]))
    return result


def build(args):
    matrix, output = args.matrix.resolve(), args.output.resolve()
    plan = report.read_json(matrix/'plan.json')
    if report.sha(matrix/'tasks.jsonl') != plan['tasks_sha256']:
        raise ValueError('Frozen screen task matrix changed')
    tasks = [json.loads(line) for line in (matrix/'tasks.jsonl').read_text().splitlines() if line.strip()]
    if len(tasks) != plan['count']:
        raise ValueError('Frozen screen task count differs')
    for name, digest in plan['source_sha256'].items():
        if report.sha(ROOT/name) != digest:
            raise ValueError('Frozen source changed: '+name)
    protected = [matrix]+[Path(t['output']).resolve() for t in tasks]+[Path(t['dataset']).parent.resolve() for t in tasks]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError('Prior diagnostics cannot write into source datasets/matrix')
    metrics, calibration, folds_table, predictions_table, statuses, provenance = [], [], [], [], [], []
    datasets, prior_arrays = {}, {}
    for task in tasks:
        if task['dataset'] in datasets:
            continue
        data, native_manifest = strict_load_dataset(task['dataset'])
        audit = report.load_dataset(task)
        name = Path(task['dataset']).parent.name
        if task['folds'] != 3:
            raise ValueError('This control is bound to the original three-family-fold protocol')
        features = {}
        cache = torch.load(audit['cache_path'], map_location='cpu', weights_only=False)
        for split in ('train', 'validation'):
            raw = data[split]['raw_action'][audit[split]['source_indices']]
            features[split], hpwl_metadata = analytic_features(cache, raw)
        prior, calibrations, folds = prior_predictions(features['train'], features['validation'],
            audit['train']['truth'][:, 0], audit['train']['family'], audit['folds'])
        scale = float(np.mean(audit['train']['truth'][:, 0]))
        for row in calibrations:
            calibration.append(dict(dataset=name, **row))
        for row in folds:
            folds_table.append(dict(dataset=name, **row))
        for model, pred in prior.items():
            for split, scope in (('train', 'train_cv_oof'), ('validation', 'historical_validation')):
                truth, family = audit[split]['truth'][:, 0], audit[split]['family']
                add_metrics(metrics, dataset=name, model=model, seed=None, scope=scope,
                    prediction=pred[split], truth=truth, families=family, reporting_scale=scale,
                    native_prediction=model in NATIVE_PRIORS)
                for index, (p, y, f) in enumerate(zip(pred[split], truth, family)):
                    predictions_table.append(dict(dataset=name, model=model, seed=None, scope=scope,
                        target_index=index, original_dataset_index=int(audit[split]['source_indices'][index]),
                        family=int(f), prediction=float(p), truth_rWL_um=float(y),
                        prediction_units='um' if model in NATIVE_PRIORS else 'macro_cache_length_units'))
            for fold in folds:
                if fold['fit'] == 'refit':
                    continue
                vi = np.asarray(fold['target_indices'])
                add_metrics(metrics, dataset=name, model=model, seed=None, scope='train_cv_fold', fit=fold['fit'],
                    prediction=pred['train'][vi], truth=audit['train']['truth'][vi, 0],
                    families=audit['train']['family'][vi], reporting_scale=scale,
                    native_prediction=model in NATIVE_PRIORS)
        datasets[task['dataset']] = audit
        prior_arrays[name] = {f'{model}_{split}': values for model, pair in prior.items() for split, values in pair.items()}
        for split in ('train', 'validation'):
            prior_arrays[name].update({f'{split}_{key}': value for key, value in features[split].items()})
            prior_arrays[name][split+'_truth_rWL_um'] = audit[split]['truth'][:, 0]
            prior_arrays[name][split+'_family'] = audit[split]['family']
        provenance.append(dict(dataset=name, dataset_sha256=task['dataset_sha256'],
            manifest_sha256=audit['manifest_sha256'], rows_sha256=native_manifest['rows_sha256'],
            cache_sha256=audit['cache_sha256'], cache_path=str(audit['cache_path']),
            successful_train_rows=len(audit['train']['family']), successful_validation_rows=len(audit['validation']['family']),
            excluded_train=audit['train']['excluded_rows'], excluded_validation=audit['validation']['excluded_rows'],
            hpwl=hpwl_metadata))
    for task in tasks:
        data = datasets[task['dataset']]
        name, folder = Path(task['dataset']).parent.name, Path(task['output'])
        status = dict(dataset=name, model=task['variant'], seed=task['seed'], status='pending', error=None)
        task_metrics, task_predictions, fit_provenance = [], [], []
        try:
            if not (folder/'completed.json').exists():
                status['status'] = 'incomplete'
                statuses.append(status)
                continue
            root = report.read_json(folder/'completed.json')
            report.validate_binding(root['binding'], task, data, plan)
            if root['status'] != 'completed' or len(root['outcomes']) != task['folds']+1:
                raise ValueError('Screen root completion marker differs')
            parts = []
            for fit_name in [f'cv_{i}' for i in range(task['folds'])]+['refit']:
                fit = report.read_fit(task, folder, fit_name, data, plan, True)
                if fit is None:
                    raise ValueError('Root completed but fold/refit incomplete')
                arrays = fit['arrays']
                fit_provenance.append(dict(dataset=name, model=task['variant'], seed=task['seed'], fit=fit_name,
                    completed_sha256=fit['completed_sha256'], predictions_sha256=fit['predictions_sha256'],
                    checkpoint_sha256=fit['done']['checkpoint_sha256']))
                split = 'validation' if fit_name == 'refit' else 'train'
                scope = 'historical_validation' if fit_name == 'refit' else 'train_cv_fold'
                add_metrics(task_metrics, dataset=name, model=task['variant'], seed=task['seed'], scope=scope,
                    fit=None if fit_name == 'refit' else fit_name, prediction=arrays['prediction'][:, 0],
                    truth=data[split]['truth'][arrays['target_indices'], 0], families=arrays['family'],
                    reporting_scale=float(data['train']['truth'][:, 0].mean()), native_prediction=True)
                if fit_name != 'refit':
                    parts.append(arrays)
                else:
                    task_predictions.append((scope, arrays))
            indices = np.concatenate([a['target_indices'] for a in parts])
            if not np.array_equal(np.sort(indices), np.arange(len(data['train']['family']))):
                raise ValueError('Learned OOF target indices are not disjoint/exhaustive')
            pooled = {key: np.concatenate([a[key] for a in parts]) for key in ('prediction', 'truth', 'family', 'target_indices')}
            order = np.argsort(pooled['target_indices'])
            pooled = {key: value[order] for key, value in pooled.items()}
            add_metrics(task_metrics, dataset=name, model=task['variant'], seed=task['seed'], scope='train_cv_oof',
                prediction=pooled['prediction'][:, 0], truth=data['train']['truth'][:, 0], families=pooled['family'],
                reporting_scale=float(data['train']['truth'][:, 0].mean()), native_prediction=True)
            task_predictions.append(('train_cv_oof', pooled))
            for scope, arrays in task_predictions:
                split = 'train' if scope == 'train_cv_oof' else 'validation'
                for i, p, y, f in zip(arrays['target_indices'], arrays['prediction'][:, 0],
                                      data[split]['truth'][arrays['target_indices'], 0], arrays['family']):
                    predictions_table.append(dict(dataset=name, model=task['variant'], seed=task['seed'], scope=scope,
                        target_index=int(i), original_dataset_index=int(data[split]['source_indices'][i]), family=int(f),
                        prediction=float(p), truth_rWL_um=float(y), prediction_units='um'))
            metrics.extend(task_metrics)
            provenance.extend(fit_provenance)
            status['status'] = 'completed'
        except (OSError, ValueError, KeyError, IndexError) as error:
            status.update(status='integrity_error', error=str(error))
        statuses.append(status)
    paired, paired_summary = paired_comparisons(metrics)
    return dict(plan=plan, metrics=metrics, calibration=calibration, folds=folds_table,
                predictions=predictions_table, statuses=statuses, provenance=provenance,
                paired_deltas=paired, paired_summary=paired_summary, model_summary=model_summary(metrics),
                prior_arrays=prior_arrays)


def write(args, result):
    output = args.output.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+f"_job{os.environ['SLURM_JOB_ID']}_{time.time_ns()%1000000:06d}"
    destination = output/'snapshots'/stamp
    destination.mkdir(parents=True, exist_ok=False)
    counts = {key: sum(r['status'] == key for r in result['statuses'])
              for key in ('completed', 'incomplete', 'integrity_error')}
    complete = counts['completed'] == len(result['statuses'])
    for name in ('metrics', 'calibration', 'folds', 'predictions', 'statuses', 'paired_deltas', 'paired_summary', 'model_summary'):
        report.atomic_text(destination/(name+'.csv'), report.csv_text(result[name]))
    for dataset, arrays in result['prior_arrays'].items():
        np.savez_compressed(destination/(dataset+'_analytic_predictions.npz'), **arrays)
    summary = dict(status='completed' if complete else 'partial', slurm_job_id=os.environ['SLURM_JOB_ID'],
        generated_utc=datetime.now(timezone.utc).isoformat(), automatic_advancement=False,
        neural_training_updates=0, actor_updates=0, backend_calls=0, heldout_used=False,
        matrix=str(args.matrix.resolve()), plan_sha256=report.sha(args.matrix/'plan.json'),
        tasks_sha256=report.sha(args.matrix/'tasks.jsonl'), report_source_sha256=report.sha(__file__),
        counts=counts, provenance=result['provenance'], paired_summary=result['paired_summary'], model_summary=result['model_summary'],
        semantics=dict(
            primary_prior='smooth_hpwl_calibrated: exact analytic component of existing huber_residual, temperature .05, one TRAIN-fitted slope/intercept',
            secondary_prior='exact_hpwl_calibrated: same fixed calibration applied to nonsmooth macro HPWL',
            calibration='h=HPWL/canvas.sum(); z=(h-mean_TRAIN)/std_TRAIN (std<=1e-5 ->1); y=(WL-mean_TRAIN)/max(population_std_TRAIN,.03*mean_abs_TRAIN,1e-8); b=sum(center(z)*center(y))/(sum(center(z)^2)+.001); a=mean(y)-b*mean(z); prediction=WLmean+WLstd*(a+b*z). No slope constraint or hyperparameter selection.',
            units='Learned/calibrated WL predictions and their MAE/top-regret use native GRT um. Raw HPWL scores use cache units and have no reported MAE. Temperature .05 is in normalized-coordinate units.',
            constant='Each CV TRAIN subset has its own mean; pooled OOF constants can vary across folds and may show misleading pooled rank. Within-fold constants and refit constants have undefined correlation.',
            replication='Each analytic method is one deterministic protocol, with zero neural training runs; it is NOT three seeds. Paired summary SD comes only from learned training seeds.',
            ties='As frozen screen: truth |difference|<=1e-9 ignored; predicted ties incorrect for family-pair accuracy. Top-regret argmin ties select the first candidate in original target-index order.',
            metric='WL only. This physical prior provides no timing/congestion/power predictions.',
            rank='Raw scores lower is better. A positive affine slope preserves ordering, a negative fitted slope reverses it. Pooled OOF calibrations use different TRAIN-only scales; per-fold ranks are also provided.',
            historical_validation='Previously inspected old16 rows/eight families per design; tuning evidence, not fresh final evaluation.',
            limitations='Macro-cache net HPWL excludes standard-cell placement, explicit fixed I/O terminals absent from this cache graph, backend routing/congestion/timing, and the external hard legalizer. Coordinates are identical raw actor proposals; post-legalization measurements are labels. Rank accuracy does not establish gradient accuracy or actor improvement.',
            candidate_regret='Best predicted candidate true WL minus best true WL within this fixed candidate pool; not a policy or online improvement result.'),
        deterministic_results=[r for r in result['metrics'] if r['seed'] is None and r['fit'] is None])
    report.atomic_text(destination/'report.json', report.json_text(summary))
    lines = ['# Physical HPWL prior control', '',
        'TRAIN-family OOF and historical validation, raw coordinates, no backend or actor updates. Analytic controls are deterministic; neural comparisons use independent model-training seeds.', '',
        '| Dataset | Scope | Model | Seed | MAE (um) | Spearman | Local pair accuracy | Top regret (um) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    def fmt(value):
        return 'NA' if value is None else f'{value:.5g}'
    for row in result['metrics']:
        if row['fit'] is None and row['aggregation'] == 'rows':
            lines.append('| '+' | '.join([row['dataset'], row['scope'], row['model'],
                'deterministic' if row['seed'] is None else str(row['seed']),
                *[fmt(row[k]) for k in ('mae', 'spearman', 'family_pair_accuracy', 'top_regret')]])+' |')
    lines.extend(['', 'Constant-score top regret uses first-index tie breaking. Uncalibrated macro HPWL has no native-unit MAE. The calibrated smooth prior exactly matches the existing residual component; its one slope/intercept use only that fold TRAIN subset.', '',
                  'These are fixed-pool surrogate diagnostics. They do not prove actor improvement, correct Q gradients, final routed PPA, or new-design generalization. See report.json for units, formula, verification hashes and exclusions.'])
    report.atomic_text(destination/'report.md', '\n'.join(lines)+'\n')
    output.mkdir(parents=True, exist_ok=True)
    report.atomic_text(output/'latest.json', report.json_text(dict(snapshot=str(destination), **counts, complete=complete)))
    print(report.json_text(dict(snapshot=str(destination), **counts, complete=complete)), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=report.DEFAULT_MATRIX)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run physical-prior diagnostics through Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    torch.use_deterministic_algorithms(True)
    complete = write(args, build(args))
    if not complete and not args.allow_partial:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
