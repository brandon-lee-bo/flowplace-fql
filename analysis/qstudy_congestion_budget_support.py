"""TRAIN-only support for one prospective congestion-excess reward, no fitting.

The empirical 75th-percentile budgets are experimental tolerances, not foundry
constraints. This diagnostic neither selects an actor nor validates a policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from fql.qstudy_data import METRIC_NAMES, action_sha, bind, load_dataset, sha


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/bp_fe384/dataset.npz')
DATA_SHA = '0daecf483b64283af46e3aec101161912ebff2984a493d0778fba5b2ebba99e4'
PROTOCOL = dict(version='qstudy_congestion_budget_support_v1', quantile=.75,
    quantile_method='linear', fold_seed=41307, folds=3,
    weights=[.8, .1, .1], metrics=['rWL_um', 'horizontal_overflow_sum', 'vertical_overflow_sum'],
    scopes='Original TRAIN only; no training, actor generation or backend calls',
    budgets='fit-TRAIN empirical tolerances; not design constraints',
    scales='fit-TRAIN WL mean and strictly positive mean excess for each overflow',
    no_scale_fallback=True, all_metrics_retained=True)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), minimum=float(a.min()),
        maximum=float(a.max()), p75=float(np.quantile(a, .75)),
        positive=int((a > 0).sum()))


def support(native, budgets):
    wl, h, v = native[:, 0], native[:, 3], native[:, 4]
    eh, ev = np.maximum(h-budgets[0], 0), np.maximum(v-budgets[1], 0)
    ok = (eh == 0) & (ev == 0)
    return dict(rows=len(wl), within_both=int(ok.sum()), within_both_fraction=float(ok.mean()),
        horizontal_excess=summary(eh), vertical_excess=summary(ev),
        unconstrained_min_WL_um=float(wl.min()),
        within_budget_min_WL_um=float(wl[ok].min()) if ok.any() else None,
        within_budget_mean_WL_um=float(wl[ok].mean()) if ok.any() else None)


def inspect_fit(native, family, train, target, name):
    tr, te = native[train], native[target]
    budget = np.quantile(tr[:, [3, 4]], .75, axis=0, method='linear')
    excess = np.maximum(tr[:, [3, 4]]-budget, 0)
    scales = np.r_[tr[:, 0].mean(), excess.mean(0)]
    admissible = bool((scales > 1e-8).all() and (np.ptp(excess, axis=0) > 1e-12).all())
    result = dict(name=name, train_indices=train.tolist(), target_indices=target.tolist(),
        train_families=sorted(set(family[train].tolist())),
        target_families=sorted(set(family[target].tolist())), budgets_H_V=budget.tolist(),
        scales_WL_Hexcess_Vexcess=scales.tolist(), admissible_nonconstant_excess=admissible,
        train_support=support(tr, budget), target_support=support(te, budget))
    if not admissible:
        result['target_objective_diagnostic'] = None
        return result
    te_excess = np.maximum(te[:, [3, 4]]-budget, 0)
    hinge = .8*te[:, 0]/scales[0] + .1*(te_excess/scales[1:]).sum(1)
    simple_scale = tr[:, [0, 3, 4]].mean(0)
    if not (simple_scale > 1e-8).all():
        raise ValueError('No fallback permitted for the matched no-timing linear comparison')
    linear = (te[:, [0, 3, 4]]/simple_scale*np.array([.8, .1, .1])).sum(1)
    pair_counts = dict(families=0, strict_WL_pair=0, hinge_opposes_WL=0,
                       linear_opposes_WL=0, hinge_and_linear_strict_opposite=0,
                       WL_ties=0, hinge_ties=0, linear_ties=0, exactly_one_objective_tied=0)
    for f in np.unique(family[target]):
        rows = np.flatnonzero(family[target] == f)
        if len(rows) != 2:
            raise ValueError('Expected the two pre-existing variants per family')
        i, j = rows
        dw, dh, dl = te[i, 0]-te[j, 0], hinge[i]-hinge[j], linear[i]-linear[j]
        pair_counts['families'] += 1
        pair_counts['strict_WL_pair'] += int(dw != 0)
        pair_counts['hinge_opposes_WL'] += int(dw*dh < 0)
        pair_counts['linear_opposes_WL'] += int(dw*dl < 0)
        pair_counts['hinge_and_linear_strict_opposite'] += int(dh*dl < 0)
        pair_counts['WL_ties'] += int(dw == 0)
        pair_counts['hinge_ties'] += int(dh == 0)
        pair_counts['linear_ties'] += int(dl == 0)
        pair_counts['exactly_one_objective_tied'] += int((dh == 0) != (dl == 0))
    # These are empirical candidate-set descriptions, not policy test results.
    pareto = []
    costs = te[:, [0, 3, 4]]
    for i, c in enumerate(costs):
        dominated = np.any(np.all(costs <= c, axis=1) & np.any(costs < c, axis=1))
        if not dominated:
            pareto.append(int(target[i]))
    result['target_objective_diagnostic'] = dict(family_local_pair_counts=pair_counts,
        nondominated_WL_H_V_indices=pareto,
        linear_C=summary(linear), excess_C=summary(hinge),
        bounded_excess_B=summary(2*hinge/(1+hinge)),
        interpretation='Reward rankings only; no surrogate fit, policy improvement or causality claim')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical dataset diagnostics require Slurm')
    output = Path(args.output).resolve()
    if output.parent != Path('/scratch/bohanli/flowplace_graph/qstudy_congestion_budget_support_v1'):
        raise ValueError('Use a unique job output in the declared scratch namespace')
    output.mkdir(parents=True, exist_ok=False)
    dataset = bind(DATA, DATA_SHA)
    data, manifest = load_dataset(DATA)
    source = data['train']
    records = [json.loads(x) for x in DATA.with_name('rows.jsonl').read_text().splitlines()]
    records = [r for r in records if r['split'] == 'train']
    n = len(records)
    if n != 384 or len(source['family']) != n or len(set(source['family'])) != 192:
        raise ValueError('Fixed original TRAIN denominator differs')
    if not source['metric_mask'].all() or not np.all(source['status'] == 'success'):
        raise ValueError('This support diagnostic requires the complete original success population')
    for i, row in enumerate(records):
        if (row['family'] != int(source['family'][i]) or row['variant'] != int(source['variant'][i]) or
            row['raw_action_sha256'] != action_sha(source['raw_action'][i]) or
            row['action_sha256'] != action_sha(source['executed_action'][i]) or
            row['metrics'] != dict(zip(METRIC_NAMES, source['metrics'][i].tolist()))):
            raise ValueError('TRAIN row identity or native metrics differ')
    native, family = source['metrics'], source['family']
    if not (np.isfinite(native).all() and (native[:, 0] > 0).all() and
            (native[:, [3, 4]] >= 0).all()):
        raise ValueError('Nonfinite metrics or invalid wirelength/overflow cost')
    groups = np.unique(family)
    np.random.default_rng(PROTOCOL['fold_seed']).shuffle(groups)
    fits = []
    for k, held in enumerate(np.array_split(groups, 3)):
        train, target = np.flatnonzero(~np.isin(family, held)), np.flatnonzero(np.isin(family, held))
        if set(family[train]) & set(family[target]):
            raise ValueError('Family leakage')
        fits.append(inspect_fit(native, family, train, target, f'cv_{k}'))
    fits.append(inspect_fit(native, family, np.arange(n), np.arange(n), 'full_TRAIN_description'))
    source_names = ['analysis/qstudy_congestion_budget_support.py',
                    'analysis/qstudy_congestion_budget_support.slurm', 'fql/qstudy_data.py']
    capture = output/'source'; capture.mkdir()
    for name in source_names:
        shutil.copy2(ROOT/name, capture/Path(name).name)
    report = dict(protocol=PROTOCOL, status='TRAIN_support_diagnostic_complete',
        slurm_job_id=os.environ['SLURM_JOB_ID'], dataset=dataset,
        manifest=bind(DATA.with_name('manifest.json')), rows=bind(DATA.with_name('rows.jsonl')),
        source_sha256={name:sha(ROOT/name) for name in source_names},
        validation_metrics_in_statistics_or_scale_fitting=False,
        fresh_or_development_evaluation_inputs=False, model_calls=0, backend_calls=0,
        native_TRAIN_summary={name:summary(native[:, k]) for k, name in enumerate(METRIC_NAMES)},
        fits=fits, all_fits_excess_nonconstant=all(f['admissible_nonconstant_excess'] for f in fits),
        caveats=['75th percentile is an empirical tolerance, not an engineering constraint.',
                 'Full TRAIN summaries are descriptive; CV target labels never fit their budgets/scales.',
                 'GRT overflows are not final DRT DRC or Innovus rO-H/rO-V.',
                 'Linear versus excess objectives change both threshold shape and above-budget scale; no isolated hinge effect.',
                 'This does not implement COptiDICE or certify unseen-layout constraint satisfaction.',
                 'A support PASS does not automatically launch a new actor/reward experiment.'])
    write(output/'report.json', report)
    print(json.dumps(dict(report=str(output/'report.json'), sha256=sha(output/'report.json'),
                         all_fits_excess_nonconstant=report['all_fits_excess_nonconstant'])))


if __name__ == '__main__':
    main()
