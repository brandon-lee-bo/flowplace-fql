"""Saved-only GRT/DRT fidelity on the same 48 declared executed layouts.

No model, EDA, label/replay export, fitting, or candidate selection is performed.
The frozen route helpers revalidate source/noise/geometry/native provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

# Import the local analysis/native namespace before any upstream baseline paths.
from fql import qstudy_final_route as route
from fql.qstudy_data import METRIC_NAMES, action_sha, load_dataset

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    'WL': ('rWL_um', 'DRT_WL', 'um', -1),
    'WNS': ('WNS_ns', 'WNS', 'ns', 1),
    'TNS': ('TNS_ns', 'TNS', 'ns', 1),
    'Power': ('Power_W', 'Power', 'W', -1),
    'Area': ('StdCellArea_um2', 'StdCellArea', 'um^2', -1),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return result


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def direction_counts(x, y):
    """Compare signed differences; ties are explicit, never silently correct."""
    counts = Counter()
    for a, b in zip(x, y):
        sa, sb = int(np.sign(a)), int(np.sign(b))
        counts['both_tied' if sa == sb == 0 else
               'GRT_tied_only' if sa == 0 else 'DRT_tied_only' if sb == 0 else
               'concordant' if sa == sb else 'discordant'] += 1
    n = counts['concordant'] + counts['discordant']
    return dict(total=len(x), **{k: counts[k] for k in
        ('concordant', 'discordant', 'both_tied', 'GRT_tied_only', 'DRT_tied_only')},
        non_tied_pairs=n, agreement_excluding_ties=counts['concordant'] / n if n else None)


def associations(rows):
    result = {}
    for name, (_, _, unit, _) in METRICS.items():
        x = np.asarray([r['GRT'][name] for r in rows], dtype=float)
        y = np.asarray([r['DRT'][name] for r in rows], dtype=float)
        pairs = list(itertools.combinations(range(len(rows)), 2))
        result[name] = dict(n=len(rows), unit=unit, pearson=correlation(x, y),
            spearman=correlation(rank(x), rank(y)),
            GRT_mean=float(x.mean()) if len(x) else None,
            DRT_mean=float(y.mean()) if len(y) else None,
            DRT_minus_GRT_mean=float((y - x).mean()) if len(x) else None,
            mean_absolute_stage_difference=float(np.abs(y - x).mean()) if len(x) else None,
            pairwise_order=direction_counts([x[i] - x[j] for i, j in pairs],
                                           [y[i] - y[j] for i, j in pairs]))
    return result


def summarize(rows, expected=48):
    """All associations are layout-level; seed means are never correlation rows."""
    strata = []
    within = {name: ([], []) for name in METRICS}
    for arm, round_number, seed in itertools.product(('bc', 'legacy'), (0, 2), (0, 1, 2)):
        subset = [r for r in rows if (r['arm'], r['round'], r['seed']) == (arm, round_number, seed)]
        for name in METRICS:
            if len(subset) > 1:
                x = rank([r['GRT'][name] for r in subset])
                y = rank([r['DRT'][name] for r in subset])
                within[name][0].extend((x - x.mean()).tolist())
                within[name][1].extend((y - y.mean()).tolist())
        strata.append(dict(arm=arm, round=round_number, seed=seed, planned=4,
            complete=len(subset) == 4, indices=[r['noise_index'] for r in subset],
            associations=associations(subset)))
    lookup = {(r['arm'], r['round'], r['seed'], r['noise_index']): r for r in rows}
    contrasts = []
    for round_number, seed, index in itertools.product((0, 2), (0, 1, 2), range(4)):
        bc, fql = (lookup.get((arm, round_number, seed, index)) for arm in ('bc', 'legacy'))
        if bc is None or fql is None:
            continue
        if bc['noise_sha256'] != fql['noise_sha256']:
            raise ValueError('BC/FQL contrast changed actual noise')
        contrasts.append(dict(round=round_number, seed=seed, noise_index=index,
            noise_sha256=bc['noise_sha256'], metrics={name: dict(
                GRT_legacy_minus_BC=fql['GRT'][name] - bc['GRT'][name],
                DRT_legacy_minus_BC=fql['DRT'][name] - bc['DRT'][name],
                unit=spec[2]) for name, spec in METRICS.items()}))
    contrast_signs = {name: direction_counts(
        [r['metrics'][name]['GRT_legacy_minus_BC'] for r in contrasts],
        [r['metrics'][name]['DRT_legacy_minus_BC'] for r in contrasts]) for name in METRICS}
    geometry = Counter(r['actual_macro_geometry_identity'] for r in rows)
    return dict(planned_same_layout_pairs=expected, completed_same_layout_pairs=len(rows),
        complete=len(rows) == expected, pooled_layout_level=associations(rows), strata=strata,
        within_stratum_centered_rank_correlation={name: dict(n=len(x),
            correlation=correlation(x, y),
            definition='Pearson of ranks centered within each arm/round/seed stratum; no policy-mean rows')
            for name, (x, y) in within.items()},
        matched_policy_contrasts=contrasts, matched_contrasts_planned=24,
        matched_contrasts_completed=len(contrasts), matched_policy_direction=contrast_signs,
        unique_actual_macro_geometries=len(geometry),
        repeated_geometry_counts=[n for n in geometry.values() if n > 1])


def matched_layout(task, source, final, saved):
    if final != saved:
        raise ValueError('Immutable completed DRT report differs from native revalidation')
    if source['state'] != 'resolved' or final['state'] != 'completed':
        return None
    b, audit = source['binding'], final['final_native_audit']
    if (final['source_binding_sha256'] != source['binding_sha256'] or
            final['source_outcome_sha256'] != b['source_outcome_sha256'] or
            audit['source_metrics'] != b['GRT_metrics'] or
            Path(audit['source_run']).resolve() != Path(b['native']['physical_run']).resolve() or
            not audit['actual_macro_DEF_reparsed']):
        raise ValueError('GRT/DRT source, metrics, or executed geometry changed')
    geometry_path = Path(audit['source_run']) / 'backend_macro_audit.json'
    geometry = read(geometry_path)
    if not geometry['legal_no_halo']:
        raise ValueError('Measured source macro geometry is illegal')
    actual = dict(actual_macros=geometry['actual_macros'], dbu_per_um=geometry['dbu_per_um'],
                  core_dbu=geometry['core_dbu'])
    metrics = {}
    for name, (g, d, _, _) in METRICS.items():
        values = (b['GRT_metrics'][g], final['metrics'][d])
        if any(isinstance(v, bool) or not math.isfinite(float(v)) for v in values):
            raise ValueError('Incomplete/nonfinite same-layout metric')
        metrics[name] = values
    return dict(**{k: task[k] for k in ('key', 'arm', 'round', 'seed', 'noise_index')},
        noise_sha256=b['noise_sha256'], raw_action_sha256=b['raw_action_sha256'],
        action_sha256=b['action_sha256'], def_sha256=b['def_sha256'],
        source_binding_sha256=source['binding_sha256'],
        source_outcome_sha256=b['source_outcome_sha256'], outcome_sha256=final['outcome_sha256'],
        GRT_physical_run=audit['source_run'], DRT_physical_run=final['physical_run'],
        GRT_geometry_sha256=sha(geometry_path), final_DEF_sha256=audit['final_def_sha256'],
        actual_macro_geometry_identity=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
        GRT={name: values[0] for name, values in metrics.items()},
        DRT={name: values[1] for name, values in metrics.items()}, DRT_DRC=final['metrics']['DRC'])


def audit_dataset(path, config_path):
    """Audit only the exact pre-existing measured FM input; no new labels."""
    path, config_path = Path(path).resolve(), Path(config_path).resolve()
    data, manifest = load_dataset(path)
    cfg = read(config_path)
    if str(path) != cfg['dataset_path'] or manifest['dataset_sha256'] != cfg['dataset_sha256']:
        raise ValueError('FM input differs from declared common initialization context')
    records = [json.loads(line) for line in path.with_name('rows.jsonl').read_text().splitlines() if line]
    split_info = {}
    for split in ('train', 'validation'):
        rows = [r for r in records if r['split'] == split]
        arrays = data[split]
        if len(rows) != len(arrays['raw_action']):
            raise ValueError('Array/provenance row denominator differs')
        for i, row in enumerate(rows):
            if (action_sha(arrays['raw_action'][i]) != row['raw_action_sha256'] or
                    action_sha(arrays['executed_action'][i]) != row['action_sha256'] or
                    int(arrays['family'][i]) != row['family'] or int(arrays['variant'][i]) != row['variant']):
                raise ValueError('Raw/executed/family array provenance changed')
            truth = np.asarray([np.nan if row['metrics'][k] is None else row['metrics'][k]
                                for k in METRIC_NAMES], dtype=float)
            if not np.array_equal(truth, arrays['metrics'][i], equal_nan=True):
                raise ValueError('Array/native-row metric correspondence changed')
        split_info[split] = dict(rows=len(rows), families=len(set(arrays['family'].tolist())),
            family_variant_counts=dict(Counter(len([r for r in rows if r['family'] == f])
                for f in set(arrays['family'].tolist()))),
            source_kinds=dict(Counter(r['collection_kind'] for r in rows)),
            status_counts=dict(Counter(r['status'] for r in rows)),
            raw_shape=list(arrays['raw_action'].shape), executed_shape=list(arrays['executed_action'].shape),
            raw_coordinate_range=[float(arrays['raw_action'].min()),float(arrays['raw_action'].max())],
            executed_coordinate_range=[float(arrays['executed_action'].min()),float(arrays['executed_action'].max())],
            raw_to_executed_RMS_all_coordinates=dict(
                mean=float(np.sqrt(np.mean((arrays['raw_action']-arrays['executed_action'])**2,axis=1)).mean()),
                max=float(np.sqrt(np.mean((arrays['raw_action']-arrays['executed_action'])**2,axis=1)).max())),
            all_seven_metrics_observed=bool(arrays['metric_mask'].all()),
            raw_executed_unequal_rows=int(np.count_nonzero(np.any(arrays['raw_action'] != arrays['executed_action'], axis=1))),
            actual_backend_geometry_rows=sum(r['actual_backend_geometry'] is not None for r in rows))
    checkpoints = {}
    for seed, value in cfg['common_checkpoints'].items():
        digest = sha(value)
        if digest != cfg['common_checkpoint_sha256'][seed]:
            raise ValueError('Common behavior flow source changed')
        checkpoints[seed] = dict(path=value, sha256=digest, exists=True)
    # This metadata source is explicitly retained by the audited assembly.
    initial_path = ROOT / 'datasets/bp_fe_grt_v1/manifest.json'
    source = read(initial_path)
    matching = [b for b in manifest['source_bindings'] if b['path'] == str(initial_path)]
    if len(matching) != 1 or sha(initial_path) != matching[0]['sha256']:
        raise ValueError('Original measured collector manifest changed')
    provenance = source['source_proxy_manifest']['provenance']
    return dict(dataset=dict(path=str(path), sha256=manifest['dataset_sha256']),
        manifest=dict(path=str(path.with_name('manifest.json')), sha256=sha(path.with_name('manifest.json'))),
        rows=dict(path=str(path.with_name('rows.jsonl')), sha256=manifest['rows_sha256']),
        reference_config=dict(path=str(config_path), sha256=sha(config_path)), splits=split_info,
        family_intersection=sorted(set(data['train']['family']) & set(data['validation']['family'])),
        original_collector_provenance=provenance, common_initializations=checkpoints,
        source_bindings=manifest['source_bindings'], separate_behavior_targets_count=len(data['behavior_action']),
        executed_action_semantics=manifest['executed_action_semantics'],
        warning='behavior_action has a distinct mixed measured/unmeasured recipe; use only measured TRAIN raw_action+native labels. Labels evaluate legalize/DEF/native-snapped geometry, not unprocessed raw coordinates. Historical validation is not fresh confirmation.',
        no_new_native_labels=True, DRT_evaluation_not_merged=True)


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) > 2:
        raise RuntimeError('Saved-array/native analysis requires bounded CPU Slurm')
    path, experiment = Path(args.report).resolve(), Path(args.experiment).resolve()
    if sha(path) != args.report_sha256:
        raise ValueError('Immutable completed report SHA changed')
    saved = read(path)
    plan = route.validate_plan(experiment)
    if (saved['status'] != 'all_requested_accounted' or saved['requested'] != 48 or
            saved['plan_sha256'] != sha(experiment / 'full_route_plan.json') or
            len(saved['records']) != 48):
        raise ValueError('Requires the complete declared development report')
    by_key = {r['key']: r for r in saved['records']}
    if len(by_key) != 48 or set(by_key) != {t['key'] for t in plan['tasks']}:
        raise ValueError('Report task namespace differs from declared48')
    sources = route.inspect_sources(experiment, plan, seal=False)
    pairs, states, evidence = [], [], []
    for task, source in zip(plan['tasks'], sources):
        final = route.inspect_final(experiment, plan, task, source)
        row = matched_layout(task, source, final, by_key[task['key']])
        states.append(final['state'])
        evidence.append(dict(key=task['key'], state=final['state'], source=source))
        if row is not None:
            pairs.append(row)
    result = dict(version='qstudy_same_executed_layout_GRT_DRT_fidelity_v1',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                       'analysis/test_qstudy_stage_fidelity.py':sha(ROOT/'analysis/test_qstudy_stage_fidelity.py'),
                       'analysis/qstudy_stage_fidelity.slurm':sha(ROOT/'analysis/qstudy_stage_fidelity.slurm')},
        input_report=dict(path=str(path), sha256=args.report_sha256),
        plan=dict(path=str(experiment/'full_route_plan.json'), sha256=saved['plan_sha256']),
        counts=dict(Counter(states)), rows=pairs, evidence=evidence, **summarize(pairs),
        dataset_audit=audit_dataset(args.dataset, args.reference_config) if args.dataset else None,
        conventions=dict(metrics={k:dict(unit=v[2], higher_is_better=v[3] == 1) for k,v in METRICS.items()},
            denominator='Only the same first4 actual noise/layouts per arm/round/seed enter either stage; no8-vs4 means.',
            correlations='48 executed-layout measurements and12 four-layout strata; correlated development noises/rounds are not48 independent replications. Pooled associations may contain between-stratum effects.',
            signs='Within-layout ordering and legacy-minus-BC deltas compare same native units; exact ties are separate, not counted correct. WNS/TNS stay signed ns.',
            stages='DRT inherits each exact GRT database and preserves macro geometry; standard cells/buffering/routing may change. This is stage fidelity, not identical full-chip placements.',
            congestion='No DRT congestion is inferred. DRT DRC is reported separately.',
            usage='Inspected development evidence only; no DRT labels/rewards are merged into TRAIN/replay and no candidate is selected.'),
        slurm_job_id=os.environ['SLURM_JOB_ID'], new_model_or_EDA_calls=0)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_job'+os.environ['SLURM_JOB_ID']
    output=Path(args.output).resolve()/stamp
    output.mkdir(parents=True,exist_ok=False)
    target=output/'report.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'completed.json').write_text(json.dumps(dict(status='completed',report_sha256=sha(target),
        source_sha256=result['source_sha256'],new_model_or_EDA_calls=0),indent=2)+'\n')
    print(json.dumps(dict(report=str(target),sha256=sha(target),counts=result['counts'],
        same_layout_pairs=len(pairs),pooled=result['pooled_layout_level'],
        within=result['within_stratum_centered_rank_correlation'],
        matched_policy_direction=result['matched_policy_direction']),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--reference-config')
    args=parser.parse_args()
    if bool(args.dataset)!=bool(args.reference_config):
        parser.error('--dataset and --reference-config must be supplied together')
    run(args)


if __name__=='__main__':
    main()
