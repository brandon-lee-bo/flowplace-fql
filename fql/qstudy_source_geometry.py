"""Coordinate-only source/support audit; no labels, fitting, EDA, or sampling.

Compare the retained original FlowPlace sampler's audited coordinates with the
predeclared current FQL behavior-flow requests. Validation remains separate.
Distances use original macro identities and normalized movable coordinates;
they are not invariant to a permutation/rotation and are not PPA predictions.
Run the numerical audit through Slurm.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from analysis.placement_metrics import MacroHPWL, legality
from .qstudy_data import action_sha, sha

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')


def read_json(path):
    return json.loads(Path(path).read_text())


def binding(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Changed coordinate/source input: '+str(path))
    return dict(path=str(path), sha256=digest)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite diagnostic vector')
    return dict(n=len(values), mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                min=float(values.min()), p05=float(np.quantile(values, .05)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                p99=float(np.quantile(values, .99)), max=float(values.max()))


def coordinates(value, n_nodes=None):
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0 or array.size % 2 or not np.isfinite(array).all():
        raise ValueError('Invalid coordinate array')
    array = array.reshape(-1, 2)
    if n_nodes is not None and len(array) != n_nodes:
        raise ValueError('Coordinate macro count differs')
    return array


def load_old(path):
    """Select coordinate/identity arrays only, never decompress metric arrays."""
    path = Path(path).resolve()
    manifest_path, rows_path = path.with_name('manifest.json'), path.with_name('rows.jsonl')
    manifest = read_json(manifest_path)
    bindings = dict(dataset=binding(path, manifest['dataset_sha256']),
                    manifest=binding(manifest_path), rows=binding(rows_path, manifest['rows_sha256']))
    if (manifest['design'] != 'bp_fe' or not manifest['no_test_or_heldout_exported'] or
            manifest['historical_online_included']):
        raise ValueError('Unexpected old dataset scope/design')
    # Provenance lines contain labels; only coordinate, identity and source keys
    # are inspected. No metrics, masks, native outcomes, or backend files are used.
    all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    groups = {}
    with np.load(path, allow_pickle=False) as arrays:
        for split in ('train', 'validation'):
            fields = {key: arrays[split+'_'+key].copy() for key in
                      ('raw_action', 'executed_action', 'family', 'variant', 'source_row', 'provenance_index')}
            count = len(fields['family'])
            if count != manifest['split_counts'][split]:
                raise ValueError('Old coordinate denominator changed')
            rows = []
            for index in range(count):
                row = all_rows[int(fields['provenance_index'][index])]
                raw, executed = coordinates(fields['raw_action'][index]), coordinates(fields['executed_action'][index])
                expected = dict(design='bp_fe', split=split, family=int(fields['family'][index]),
                                variant=int(fields['variant'][index]), source_row=int(fields['source_row'][index]))
                if any(row[key] != value for key, value in expected.items()):
                    raise ValueError('Old coordinate/provenance identity mismatch')
                if (not np.array_equal(raw, coordinates(row['raw_action'])) or
                        not np.array_equal(executed, coordinates(row['executed_action'])) or
                        action_sha(raw) != row['raw_action_sha256'] or
                        action_sha(executed) != row['action_sha256']):
                    raise ValueError('Old raw/executed coordinate fingerprint mismatch')
                rows.append(dict(source='old_public_flowplace20_hard', split=split,
                    family=expected['family'], variant=expected['variant'], index=index,
                    source_row=expected['source_row'], collection_kind=row['collection_kind'],
                    raw=raw, executed=executed, raw_sha256=row['raw_action_sha256'],
                    executed_sha256=row['action_sha256']))
            groups['old_'+split] = rows
    if set(r['family'] for r in groups['old_train']) & set(r['family'] for r in groups['old_validation']):
        raise ValueError('Old split family overlap')
    bindings['retained_source_bindings'] = manifest['source_bindings']
    bindings['validation_status'] = manifest['validation_status']
    return groups, bindings


def load_teacher(experiment, expected_old_sha):
    """Validate the sealed export only; no outcome/native files are opened."""
    experiment = Path(experiment).resolve()
    config_path, source_path = experiment/'config.json', experiment/'teacher_binding.json'
    cfg, source = read_json(config_path), read_json(source_path)
    folder = experiment/'seed_0/round_0/requests'
    marker_path, tasks_path = folder/'completed.json', folder/'tasks.jsonl'
    marker = read_json(marker_path)
    bindings = dict(config=binding(config_path, marker['config_sha256']),
        teacher_binding=binding(source_path), tasks=binding(tasks_path, marker['tasks_sha256']),
        completion=binding(marker_path), checkpoint=binding(cfg['common_checkpoint'], cfg['common_checkpoint_sha256']))
    if (cfg['design'] != 'bp_fe' or cfg['dataset_sha256'] != expected_old_sha or
            source['source_kind'] != 'current_fql_behavior_flow' or
            source['state_key'] != 'flow' or source['integration_steps'] != 10 or
            source['intermediate_hard_constraints'] or marker['teacher_binding_sha256'] != object_sha(source) or
            source['config_sha256'] != bindings['config']['sha256']):
        raise ValueError('Changed predeclared current behavior-flow source')
    requests = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    if len(requests) != marker['count'] or len(requests) != 2*(cfg['train_families']+cfg['val_families']):
        raise ValueError('Teacher request denominator differs')
    groups = dict(teacher_train=[], teacher_validation=[])
    files = []
    for index, row in enumerate(requests):
        split = 'train' if row['family_index'] < cfg['train_families'] else 'validation'
        namespace = dict(source_kind=source['source_kind'], design=cfg['design'],
            common_checkpoint_sha256=cfg['common_checkpoint_sha256'], noise_seed=cfg['noise_seed'],
            family_index=row['family_index'])
        expected = dict(index=index, family_index=index//2, variant=index%2, split=split,
            role='data_'+split, offline_train_eligible=split == 'train', training_feedback_eligible=False,
            terminal_mask=0, graph='bp_fe', seed=0, round=0, source_kind=source['source_kind'],
            source_sampler=source['source_sampler'], source_state_key='flow', source_sampler_steps=10,
            intermediate_hard_constraints=False, family=cfg['noise_seed']+index//2,
            checkpoint_sha256=bindings['checkpoint']['sha256'], dataset_sha256=expected_old_sha,
            config_sha256=bindings['config']['sha256'], teacher_binding_sha256=object_sha(source),
            cache_sha256=cfg['cache_sha256'], source_family_namespace=namespace,
            source_family_sha256=object_sha(namespace))
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('Teacher request source/family/split changed')
        sample = Path(row['sample_dir']).resolve()
        if sample.parent.parent.parent != folder or read_json(sample/'request.json') != row:
            raise ValueError('Teacher request path or frozen sidecar mismatch')
        row_binding = dict(index=index, request=binding(sample/'request.json'), actions=binding(sample/'actions.npz'))
        raw, executed = coordinates(row['raw_action']), coordinates(row['executed_action'])
        with np.load(sample/'actions.npz', allow_pickle=False) as arrays:
            for key, expected_array, hash_key in [('raw_action', raw, 'raw_action_sha256'),
                    ('action', executed, 'action_sha256'), ('noise', None, 'noise_sha256'),
                    ('direction', None, 'direction_sha256'), ('base_raw_action', None, 'base_raw_action_sha256')]:
                value = arrays[key]
                if (action_sha(value) != row[hash_key] or not np.isfinite(value).all() or
                        value.shape != raw.shape or
                        (expected_array is not None and not np.array_equal(value, expected_array))):
                    raise ValueError('Teacher raw/executed/noise/direction identity changed')
        if 'def_path' in row:
            row_binding['def'] = binding(row['def_path'], row['def_sha256'])
        files.append(row_binding)
        groups['teacher_'+split].append(dict(source=source['source_kind'], split=split,
            family=row['family'], variant=row['variant'], index=index, source_row=index,
            collection_kind=row['variant_kind'], raw=raw, executed=executed,
            raw_sha256=row['raw_action_sha256'], executed_sha256=row['action_sha256']))
    bindings['requests'] = files
    bindings['sampler'] = {key: source[key] for key in ('source_kind', 'source_sampler', 'state_key',
        'integration_steps', 'intermediate_hard_constraints', 'final_coordinate_clipping', 'legalizer')}
    bindings['cache_sha256'] = cfg['cache_sha256']
    return groups, bindings


def nearest(query, reference, movable, excluded=None):
    """RMS over movable coordinate entries; exact same-family exclusion if given."""
    q = np.asarray(query, dtype=np.float64)[:, movable].reshape(len(query), -1)
    r = np.asarray(reference, dtype=np.float64)[:, movable].reshape(len(reference), -1)
    if not q.shape[1] or not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError('Empty/nonfinite movable coordinates')
    distances = np.sqrt(np.mean((q[:, None]-r[None, :])**2, axis=-1))
    if excluded is not None:
        if excluded.shape != distances.shape:
            raise ValueError('Nearest-neighbor exclusion shape mismatch')
        distances[excluded] = np.inf
    ids = distances.argmin(axis=1)
    values = distances[np.arange(len(query)), ids]
    if not np.isfinite(values).all():
        raise ValueError('No eligible distinct-family reference')
    return values, ids


def analyze(groups, movable, metric):
    rows, summaries = [], []
    for group, selected in groups.items():
        raw = np.stack([r['raw'] for r in selected])
        executed = np.stack([r['executed'] for r in selected])
        if raw.shape[1:] != metric.sizes.shape or movable.shape != (raw.shape[1],) or not movable.any():
            raise ValueError('Cache/node order/movable mask mismatch')
        jumps = np.sqrt(np.mean((executed[:, movable].astype(float)-raw[:, movable])**2, axis=(1, 2)))
        raw_hpwl, exec_hpwl = metric(raw), metric(executed)
        item_rows = []
        for index, source_row in enumerate(selected):
            row = {key: source_row[key] for key in ('source', 'split', 'family', 'variant', 'index',
                                                   'source_row', 'collection_kind', 'raw_sha256', 'executed_sha256')}
            row.update(group=group, raw_to_executed_rms=float(jumps[index]),
                raw_macro_hpwl=float(raw_hpwl[index]), executed_macro_hpwl=float(exec_hpwl[index]),
                executed_minus_raw_macro_hpwl=float(exec_hpwl[index]-raw_hpwl[index]))
            for name, centers in [('raw', raw[index]), ('executed', executed[index])]:
                geometry = legality(centers[movable], metric.sizes[movable])
                row.update({name+'_'+key: value for key, value in geometry.items()})
            rows.append(row)
            item_rows.append(row)
        summaries.append(dict(group=group, rows=len(selected), families=len({r['family'] for r in selected}),
            raw_to_executed_rms=stats(jumps), raw_macro_hpwl=stats(raw_hpwl),
            executed_macro_hpwl=stats(exec_hpwl), executed_minus_raw_macro_hpwl=stats(exec_hpwl-raw_hpwl),
            geometry={name: dict(illegal_fraction=float(np.mean([not r[name+'_legal'] for r in item_rows])),
                overlap_fraction=float(np.mean([r[name+'_overlap_pairs'] > 0 for r in item_rows])),
                out_of_bounds_fraction=float(np.mean([r[name+'_out_of_bounds'] > 0 for r in item_rows])),
                overlap_pairs=stats([r[name+'_overlap_pairs'] for r in item_rows]),
                out_of_bounds_macros=stats([r[name+'_out_of_bounds'] for r in item_rows])) for name in ('raw', 'executed')}))
    supports, nearest_rows = [], []
    for coordinate in ('raw', 'executed'):
        ref = np.stack([r[coordinate] for r in groups['old_train']])
        families = np.asarray([r['family'] for r in groups['old_train']])
        lfo, _ = nearest(ref, ref, movable, families[:, None] == families[None, :])
        thresholds = {key: float(np.quantile(lfo, value)) for key, value in [('p95', .95), ('p99', .99)]}
        comparisons = [('old_train', 'old_train', True), ('old_validation', 'old_train', False),
            ('teacher_train', 'old_train', False), ('teacher_validation', 'old_train', False),
            ('old_train', 'teacher_train', False), ('old_validation', 'teacher_train', False)]
        for query_name, reference_name, leave_family_out in comparisons:
            query, reference = groups[query_name], groups[reference_name]
            q, r = np.stack([x[coordinate] for x in query]), np.stack([x[coordinate] for x in reference])
            excluded = families[:, None] == families[None, :] if leave_family_out else None
            distances, ids = nearest(q, r, movable, excluded)
            supports.append(dict(query=query_name, reference=reference_name, coordinate=coordinate,
                leave_family_out=leave_family_out, nearest_rms=stats(distances),
                thresholds_fit_on='old_train leave-family-out only', old_train_lfo_thresholds=thresholds,
                fraction_above_old_train_lfo_p95=float(np.mean(distances > thresholds['p95'])),
                fraction_above_old_train_lfo_p99=float(np.mean(distances > thresholds['p99']))))
            for source_row, distance, neighbor in zip(query, distances, ids):
                nearest_rows.append(dict(query=query_name, reference=reference_name, coordinate=coordinate,
                    leave_family_out=leave_family_out, query_index=source_row['index'], query_family=source_row['family'],
                    query_variant=source_row['variant'], nearest_index=reference[neighbor]['index'],
                    nearest_family=reference[neighbor]['family'], nearest_variant=reference[neighbor]['variant'], rms=float(distance)))
    return rows, summaries, supports, nearest_rows


def write_csv(path, rows):
    stream = io.StringIO()
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def run(args):
    import torch
    from baseline.api import CHECKPOINT, RUNTIME, load_problem

    groups, old_bindings = load_old(args.old_dataset)
    teacher, teacher_bindings = load_teacher(args.teacher, old_bindings['dataset']['sha256'])
    groups.update(teacher)
    cache_path = RUNTIME/'benchmark_cache/bp_fe.pt'
    cache_binding = binding(cache_path, teacher_bindings['cache_sha256'])
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    problem = load_problem('bp_fe', 'cpu')
    movable = (~problem.cond.is_ports.bool()).cpu().numpy().reshape(-1)
    metric = MacroHPWL(cache)
    # Both readers retain source order. Confirm the graph's named ordering too.
    if list(problem.macro_names) != metric.names:
        raise ValueError('Problem/cache macro-name ordering differs')
    original_manifest_path = ROOT/'datasets/bp_fe_macro_deterministic_v1/manifest.json'
    original_manifest = read_json(original_manifest_path)
    old_bindings['original_collector_manifest'] = binding(original_manifest_path)
    old_bindings['original_checkpoint'] = binding(CHECKPOINT, original_manifest['provenance']['checkpoint_sha256'])
    if original_manifest['provenance']['cache_sha256'] != cache_binding['sha256']:
        raise ValueError('Old sampler and current teacher use different cache geometry')
    rows, summaries, supports, neighbors = analyze(groups, movable, metric)
    destination = args.output.resolve()/('snapshots/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+
        '_job'+os.environ['SLURM_JOB_ID']+'_'+str(time.time_ns()%1000000))
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination/'coordinates.csv', rows)
    write_csv(destination/'nearest_neighbors.csv', neighbors)
    report = dict(status='completed', slurm_job_id=os.environ['SLURM_JOB_ID'],
        generated_utc=datetime.now(timezone.utc).isoformat(), source=binding(__file__),
        dependencies={name: binding(ROOT/name) for name in ('analysis/placement_metrics.py', 'baseline/api.py',
            'fql/qstudy_data.py', 'fql/qstudy_cpu_stage.slurm')},
        inputs=dict(old=old_bindings, teacher=teacher_bindings, cache=cache_binding),
        node_order=metric.names, movable=movable.tolist(), movable_macros=int(movable.sum()),
        fixed_ports=int((~movable).sum()), summaries=summaries, support=supports,
        output_bindings=dict(coordinates=binding(destination/'coordinates.csv'),
                             nearest_neighbors=binding(destination/'nearest_neighbors.csv')),
        scope=dict(labels_used=False, metric_arrays_loaded=False, native_outcomes_opened=False,
            inference_calls=0, legalization_calls=0, backend_calls=0, actor_updates=0,
            validation_fit=False, automatic_advancement=False),
        semantics=dict(rms='sqrt(mean over all movable macro x/y normalized center-coordinate entries)); [-1,1] per axis, no data-fitted scaling',
            geometry='Computed from retained raw proposals and final legalized coordinates plus audited cache macro sizes; movable macros only, zero halo, tolerance 1e-6. Executed is before DEF rounding/native snapping, not saved-backend geometry or DRC.',
            hpwl=metric.metadata(), support='Node-identity-aware Euclidean coordinate support diagnostic. TRAIN reference thresholds use only old TRAIN leave-family-out nearest distances; validation is evaluated separately and never fitted.',
            replication='Rows include paired variants and are not independent seeds; summaries are descriptive, not inferential uncertainty.',
            confounding='Source differs in checkpoint/training, Euler10 vs released20 hard sampling, intermediate constraint handling, and perturbation fixed-RMS .02 vs older Gaussian sigma .04; no causal attribution to one factor.',
            limits='Coordinate/source mismatch can explain risk but cannot prove Q-label mismatch, gradient usefulness, or any PPA improvement. No heldout/test/native labels evaluated.'))
    (destination/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Coordinate-only source/support audit', '',
        'No backend labels, inference, new legalization, or EDA were used. Old historical validation and fresh teacher validation remain separate.', '',
        '| Group | Rows / families | Raw→executed RMS mean / p95 | Raw overlap fraction | Raw bounds fraction | Executed illegal fraction |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in summaries:
        lines.append(f"| {row['group']} | {row['rows']} / {row['families']} | {row['raw_to_executed_rms']['mean']:.5g} / {row['raw_to_executed_rms']['p95']:.5g} | {row['geometry']['raw']['overlap_fraction']:.3f} | {row['geometry']['raw']['out_of_bounds_fraction']:.3f} | {row['geometry']['executed']['illegal_fraction']:.3f} |")
    lines += ['', '| Coordinates | Query → reference | Nearest RMS mean / p95 | Above old TRAIN LFO p95 |', '| --- | --- | --- | --- |']
    for row in supports:
        lines.append(f"| {row['coordinate']} | {row['query']} → {row['reference']}{' (leave family out)' if row['leave_family_out'] else ''} | {row['nearest_rms']['mean']:.5g} / {row['nearest_rms']['p95']:.5g} | {row['fraction_above_old_train_lfo_p95']:.3f} |")
    lines += ['', 'Distances preserve macro identities and use movable normalized coordinates. Fractions are row descriptions, not independent-seed confidence estimates. Source factors and perturbation amplitudes differ. Macro HPWL is a cache-only geometric proxy, not routed WL. This audit does not establish performance gains or authorize model selection.']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed', snapshot=str(destination), rows=len(rows),
                          counts={key: len(value) for key, value in groups.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-dataset', type=Path, default=BULK/'data_audit/bp_fe384/dataset.npz')
    parser.add_argument('--teacher', type=Path, default=BULK/'teacher_source_v1/bp_fe_common0')
    parser.add_argument('--output', type=Path, default=BULK/'source_geometry_v1')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run coordinate diagnostics through Slurm')
    run(args)


if __name__ == '__main__':
    main()
