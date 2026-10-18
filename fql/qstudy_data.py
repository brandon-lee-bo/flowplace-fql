"""Strict, replayable assembly of measured Q-study TRAIN/validation datasets.

The assembly reads only predeclared TRAIN/validation labels, keeps historical
online feedback separate, and never exports heldout/test metrics. All metric
values are native GRT measurements; missing physical results remain NaN with
an explicit mask. The public load_dataset function checks the saved binding.
Run assembly/statistical inspection through Slurm, not on a login node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('/fact_data/bohanli/flowplace_fql')
SCRATCH = Path('/scratch/bohanli/flowplace_graph')
METRIC_NAMES = ('rWL_um', 'WNS_ns', 'TNS_ns', 'horizontal_overflow_sum',
                'vertical_overflow_sum', 'Power_W', 'StdCellArea_um2')
SPLIT_NAMES = ('train', 'validation')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def action_sha(action):
    return hashlib.sha256(np.asarray(action, dtype='<f4').reshape(-1).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def bind(path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    if expected is not None and value != expected:
        raise ValueError('Source fingerprint changed: ' + str(path))
    return {'path': str(path), 'sha256': value}


def checked_json(path, expected=None):
    return read_json(bind(path, expected)['path'])


def _native(record, design, raw, executed, split, family, variant, source_row,
            collection_kind, source_label_path, source_label_sha):
    """Normalize one label after checking original action/DEF/backend bindings."""
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    executed = np.asarray(executed, dtype=np.float32).reshape(-1)
    if raw.shape != executed.shape or not np.isfinite(raw).all() or not np.isfinite(executed).all():
        raise ValueError('Invalid raw/executed coordinate pair')
    if split not in SPLIT_NAMES and split != 'historical_online':
        raise ValueError('Heldout/test or undefined split cannot enter Q-study')
    expected_action_sha = record['action_sha256']
    if action_sha(executed) != expected_action_sha:
        raise ValueError('Executed action does not match measured label')
    status = record.get('status', 'success')
    if status not in ('success', 'geometry_failure', 'physical_failure'):
        raise ValueError('Unresolved infrastructure failures cannot become labels')
    outcome_binding = None
    outcome = record
    if record.get('outcome_path'):
        outcome_binding = bind(record['outcome_path'], record['outcome_sha256'])
        outcome = read_json(outcome_binding['path'])
        for key in ['status', 'metrics', 'action_sha256']:
            if outcome.get(key) != record.get(key):
                raise ValueError('Retained row differs from its native outcome: ' + key)
        for key, value in [('raw_action', raw), ('executed_action', executed)]:
            if not np.array_equal(np.asarray(outcome[key], dtype=np.float32).reshape(-1), value):
                raise ValueError('Outcome coordinates changed: ' + key)
        request_path = Path(outcome_binding['path']).with_name('request.json')
        request_binding = bind(request_path, outcome['request_sha256'])
        request = read_json(request_path)
        if request.get('action_sha256') != expected_action_sha:
            raise ValueError('Original request lost its action binding')
    else:
        request_binding = None
    metrics = record.get('metrics')
    bindings = {'label_file': {'path': str(source_label_path), 'sha256': source_label_sha},
                'outcome': outcome_binding, 'request': request_binding}
    backend = None
    geometry = None
    if status == 'success':
        if not metrics or metrics.get('stage') != 'global_route_only' or not metrics.get('not_final_table2'):
            raise ValueError('Only measured GRT labels may enter this assembly')
        run = Path(outcome['physical_run']) if 'physical_run' in outcome else Path(record['manifest_path']).parent
        manifest_binding = bind(run / 'manifest.json',
            outcome.get('physical_manifest_sha256', record.get('manifest_sha256')))
        native_metrics = checked_json(run / 'metrics.json',
            outcome.get('metrics_sha256', record.get('metrics_sha256')))
        manifest = read_json(manifest_binding['path'])
        if native_metrics != metrics or not manifest.get('success') or manifest.get('design') != design:
            raise ValueError('Wrong design, failed physical run or modified measured metrics')
        if manifest.get('stage') != 'global_route_only':
            raise ValueError('Mixed backend stage')
        def_binding = bind(manifest['def_path'], manifest['def_sha256'])
        bindings.update(physical_manifest=manifest_binding,
                        physical_metrics=bind(run / 'metrics.json'), input_def=def_binding)
        backend = {key: manifest.get(key) for key in [
            'design', 'num_cores', 'design_config', 'design_config_sha256', 'openroad_sha256',
            'flow_script_patch', 'physical_flow_revision', 'physical_flow_components',
            'evaluator_revision', 'executed_reporting_revision', 'parser_version']}
        gp = outcome.get('backend_macro_geometry_path')
        if gp:
            geometry = bind(gp, outcome['backend_macro_geometry_sha256'])
        elif (run / 'backend_macro_audit.json').exists():
            geometry = bind(run / 'backend_macro_audit.json')
        if geometry:
            g = read_json(geometry['path'])
            if g['bindings']['physical_manifest_sha256'] != manifest_binding['sha256']:
                raise ValueError('Actual backend geometry belongs to another run')
            geometry.update(legal_no_halo=g['legal_no_halo'],
                            input_to_actual_changes=g['input_to_actual_changes'],
                            actual_macros=g['actual_macros'], dbu_per_um=g['dbu_per_um'])
        for key in METRIC_NAMES:
            value = metrics.get(key)
            if value is not None and (isinstance(value, bool) or not np.isfinite(float(value))):
                raise ValueError('Nonfinite or invalid physical metric: ' + key)
        if metrics.get('rWL_um', 0) <= 0:
            raise ValueError('Successful route requires positive wirelength')
    elif metrics is not None:
        raise ValueError('Physical failures must have missing PPA, not numeric labels')
    return dict(design=design, split=split, family=int(family), variant=int(variant),
                source_row=int(source_row), collection_kind=collection_kind, status=status,
                raw_action=raw.tolist(), executed_action=executed.tolist(),
                raw_action_sha256=action_sha(raw), action_sha256=expected_action_sha,
                metrics={key: None if metrics is None else metrics.get(key) for key in METRIC_NAMES},
                backend=backend, actual_backend_geometry=geometry, bindings=bindings,
                failure_class=outcome.get('failure_class'), terminal_mask=0)


def _behavior_arrays(path, expected):
    binding = bind(path, expected)
    # No reward or backend label arrays are loaded from the synthetic collector.
    with np.load(path, allow_pickle=False) as a:
        fields = ['raw_action', 'action', 'critic_split', 'critic_parent_index', 'variant',
                  'teacher_action', 'split']
        arrays = {key: a[key].copy() for key in fields}
    if not np.array_equal(arrays['critic_split'], arrays['split'][arrays['critic_parent_index']]):
        raise ValueError('Behavior family/split correspondence changed')
    return arrays, binding


def initial_bp_fe():
    folder = ROOT / 'datasets/bp_fe_grt_v1'
    meta = read_json(folder / 'manifest.json')
    arrays, binding = _behavior_arrays(meta['source_dataset'], meta['source_dataset_sha256'])
    bind(folder / 'dataset.npz', meta['dataset_sha256'])
    labels = folder / 'backend_labels.jsonl'
    label_sha = sha(labels)
    rows = []
    excluded = 0
    for line in labels.read_text().splitlines():
        label = json.loads(line)
        split = int(label['critic_split'])
        if split == 2:
            excluded += 1
            continue  # Do not inspect/export heldout metrics or any native heldout files.
        index = int(label['row_index'])
        if split not in (0, 1) or arrays['critic_split'][index] != split:
            raise ValueError('Original label/behavior split mismatch')
        rows.append(_native(label, 'bp_fe', arrays['raw_action'][index], arrays['action'][index],
            SPLIT_NAMES[split], arrays['critic_parent_index'][index], arrays['variant'][index], index,
            'initial_offline_behavior', labels, label_sha))
    if len(rows) != 80 or excluded != 16:
        raise ValueError('Original fixed TRAIN/validation/heldout denominator changed')
    return rows, arrays, [binding, bind(folder / 'manifest.json'), bind(labels)], excluded


def expanded_bp_fe(arrays):
    folder = SCRATCH / 'graph_o2o_offline_expand_v1'
    plan_path = folder / 'selection_plan.json'
    plan = read_json(plan_path)
    if plan['requested'] != 320 or plan['validation_heldout_eligible']:
        raise ValueError('Expanded data selection is not TRAIN-only')
    labels = folder / 'report/offline_replay.json'
    replay = read_json(labels)
    if replay['validation_heldout_used'] or len(replay['rows']) != 320:
        raise ValueError('Expanded replay has unexpected scope')
    label_sha = sha(labels)
    rows = []
    for expected_index, label in zip(plan['selected_source_rows'], replay['rows']):
        index = label['source_row_index']
        if index != expected_index or arrays['critic_split'][index] != 0:
            raise ValueError('Expansion crossed its frozen TRAIN source boundary')
        if not np.array_equal(np.asarray(label['raw_action'], dtype=np.float32).reshape(-1), arrays['raw_action'][index]):
            raise ValueError('Expanded raw action differs from behavior source')
        if not np.array_equal(np.asarray(label['executed_action'], dtype=np.float32).reshape(-1), arrays['action'][index]):
            raise ValueError('Expanded executed action differs from behavior source')
        rows.append(_native(label, 'bp_fe', label['raw_action'], label['executed_action'],
            'train', arrays['critic_parent_index'][index], arrays['variant'][index], index,
            'additional_fixed_offline_behavior', labels, label_sha))
    if {r['family'] for r in rows} != set(range(32, 192)):
        raise ValueError('Expanded TRAIN families changed')
    return rows, [bind(labels), bind(plan_path)]


def initial_bp_be():
    plan_path = BULK / 'graph_o2o_second_design_v1/initial_label_plan.json'
    plan = read_json(plan_path)
    design = checked_json(plan['design_plan'], plan['design_plan_sha256'])
    arrays, binding = _behavior_arrays(Path(design['dataset']) / 'dataset.npz', plan['behavior_dataset_sha256'])
    folder = Path(plan['output']) / 'dataset'
    meta = read_json(folder / 'manifest.json')
    bind(folder / 'dataset.npz', meta['dataset_sha256'])
    labels = folder / 'backend_labels.json'
    label_binding = bind(labels, meta['backend_labels_sha256'])
    rows = []
    for label in read_json(labels):
        index = int(label['source_row_index'])
        split = int(label['critic_split'])
        if split not in (0, 1) or arrays['critic_split'][index] != split:
            raise ValueError('Second-design data crossed TRAIN/validation boundary')
        rows.append(_native(label, 'bp_be', arrays['raw_action'][index], arrays['action'][index],
            SPLIT_NAMES[split], arrays['critic_parent_index'][index], arrays['variant'][index], index,
            'initial_offline_behavior', labels, label_binding['sha256']))
    if len(rows) != 80:
        raise ValueError('Second-design initial denominator changed')
    return rows, arrays, [binding, bind(plan_path), bind(folder / 'manifest.json'), label_binding]


def summarize(rows):
    from scipy.stats import spearmanr
    values = np.asarray([[np.nan if r['metrics'][name] is None else r['metrics'][name]
                          for name in METRIC_NAMES] for r in rows], dtype=np.float64)
    stats = {}
    for k, name in enumerate(METRIC_NAMES):
        v = values[np.isfinite(values[:, k]), k]
        stats[name] = dict(observed=len(v), missing=len(rows)-len(v),
            zero_count=int(np.count_nonzero(v == 0)),
            mean=float(v.mean()) if len(v) else None,
            std=float(v.std(ddof=1)) if len(v)>1 else None,
            min=float(v.min()) if len(v) else None, max=float(v.max()) if len(v) else None,
            unique=len(np.unique(v)))
    correlations = {}
    for a in range(len(METRIC_NAMES)):
        for b in range(a+1, len(METRIC_NAMES)):
            mask = np.isfinite(values[:, a]) & np.isfinite(values[:, b])
            x, y = values[mask, a], values[mask, b]
            defined = len(x)>2 and len(np.unique(x))>1 and len(np.unique(y))>1
            correlations[METRIC_NAMES[a] + '__' + METRIC_NAMES[b]] = dict(
                n=len(x), pearson=float(np.corrcoef(x, y)[0, 1]) if defined else None,
                spearman=float(spearmanr(x, y).statistic) if defined else None)
    by_family = {}
    for row in rows:
        by_family.setdefault(row['family'], []).append(row)
    pair_deltas = {}
    for name in METRIC_NAMES:
        deltas = []
        for family in by_family.values():
            if len(family) == 2:
                a, b = sorted(family, key=lambda r:r['variant'])
                x, y = a['metrics'][name], b['metrics'][name]
                if x is not None and y is not None:
                    deltas.append(y-x)
        v = np.asarray(deltas)
        pair_deltas[name] = dict(pairs=len(v), mean=float(v.mean()) if len(v) else None,
            mean_abs=float(np.abs(v).mean()) if len(v) else None,
            positive=int(np.count_nonzero(v>0)), negative=int(np.count_nonzero(v<0)),
            equal=int(np.count_nonzero(v==0)))
    raw = np.asarray([r['raw_action'] for r in rows], dtype=np.float64)
    executed = np.asarray([r['executed_action'] for r in rows], dtype=np.float64)
    delta = np.sqrt(np.mean((raw-executed)**2, axis=1))
    return dict(count=len(rows), families=len(by_family), successful=sum(r['status']=='success' for r in rows),
        metrics=stats, correlations=correlations, within_family_variant1_minus_variant0=pair_deltas,
        unique_raw_actions=len({r['raw_action_sha256'] for r in rows}),
        unique_executed_actions=len({r['action_sha256'] for r in rows}),
        raw_to_legalized_rms=dict(mean=float(delta.mean()), max=float(delta.max())),
        actual_backend_geometry_available=sum(r['actual_backend_geometry'] is not None for r in rows))


def export_dataset(folder, design, rows, behavior_arrays, bindings, expected_train):
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {}
    groups = {name:[r for r in rows if r['split'] == name] for name in SPLIT_NAMES}
    if len(groups['train']) != expected_train or len(groups['validation']) != 16:
        raise ValueError('Unexpected dataset size')
    if {r['family'] for r in groups['train']} & {r['family'] for r in groups['validation']}:
        raise ValueError('TRAIN and validation share noise families')
    collisions = {}
    for key in ['raw_action_sha256', 'action_sha256']:
        common = {r[key] for r in groups['train']} & {r[key] for r in groups['validation']}
        collisions[key] = sorted(common)
        if common:
            raise ValueError('Exact coordinate leakage across TRAIN/validation: '+key)
    ordered = []
    for split, group in groups.items():
        values = np.asarray([[np.nan if r['metrics'][name] is None else r['metrics'][name]
                              for name in METRIC_NAMES] for r in group], dtype=np.float64)
        data = dict(raw_action=np.asarray([r['raw_action'] for r in group], dtype=np.float32),
            executed_action=np.asarray([r['executed_action'] for r in group], dtype=np.float32),
            metrics=values, metric_mask=np.isfinite(values),
            family=np.asarray([r['family'] for r in group], dtype=np.int64),
            variant=np.asarray([r['variant'] for r in group], dtype=np.int64),
            source_row=np.asarray([r['source_row'] for r in group], dtype=np.int64),
            status=np.asarray([r['status'] for r in group], dtype='U24'),
            provenance_index=np.arange(len(ordered), len(ordered)+len(group), dtype=np.int64))
        arrays.update({split+'_'+key:value for key,value in data.items()})
        ordered.extend(group)
    # Preserve the original256 behavior-target recipe even for384 critic labels.
    original_train = [r for r in groups['train'] if r['collection_kind']=='initial_offline_behavior']
    extra = behavior_arrays['teacher_action'][behavior_arrays['split'] == 0]
    arrays['behavior_action'] = np.concatenate([np.asarray([r['raw_action'] for r in original_train], dtype=np.float32), extra])
    arrays['behavior_family'] = np.concatenate([np.asarray([r['family'] for r in original_train]),
                                                np.flatnonzero(behavior_arrays['split']==0)]).astype(np.int64)
    np.savez_compressed(folder / 'dataset.npz', **arrays)
    (folder / 'rows.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in ordered))
    revisions = {}
    for r in ordered:
        if r['backend']:
            key = r['backend']['physical_flow_revision']
            revisions[key] = revisions.get(key, 0)+1
    manifest = dict(version=1, design=design, metric_names=list(METRIC_NAMES),
        metric_units=['um','ns','ns','positive_demand_minus_capacity','positive_demand_minus_capacity','W','um2'],
        stage='global_route_only', not_final_table2=True,
        executed_action_semantics='normalized legalizer output before DEF rounding/native snapping; actual backend macro boxes are separate row geometry evidence when available',
        dataset_sha256=sha(folder/'dataset.npz'), rows_sha256=sha(folder/'rows.jsonl'),
        source_bindings=bindings, split_counts={s:len(v) for s,v in groups.items()},
        behavior_targets=len(arrays['behavior_action']), behavior_source='64 initial TRAIN raw proposals +192 TRAIN teacher layouts',
        no_test_or_heldout_exported=True, historical_online_included=False,
        validation_status='historical tuning validation; previously inspected, not a fresh final test',
        physical_flow_revisions=revisions, exact_cross_split_collisions=collisions,
        summary={s:summarize(v) for s,v in groups.items()},
        slurm_job_id=os.environ['SLURM_JOB_ID'], assembly_source_sha256=sha(__file__))
    write_json(folder/'manifest.json', manifest)
    # Read-back verifies array dimensions, finite masks and split constraints.
    load_dataset(folder/'dataset.npz')
    return manifest


def load_dataset(path):
    """Return ({'train': arrays, 'validation': arrays, 'behavior_action': ...}, manifest).

    Coordinate matrices are flattened float32 (N,2*macros). Signed native
    metrics are float64 (N,7); metric_mask is boolean. Caller fits any cost or
    feature normalization on TRAIN only. This API never returns test data.
    """
    path = Path(path)
    if path.is_dir():
        path = path/'dataset.npz'
    manifest = read_json(path.with_name('manifest.json'))
    if sha(path) != manifest['dataset_sha256'] or list(METRIC_NAMES) != manifest['metric_names']:
        raise ValueError('Q-study dataset identity or metric order changed')
    if sha(path.with_name('rows.jsonl')) != manifest['rows_sha256']:
        raise ValueError('Q-study row provenance changed')
    if not manifest['no_test_or_heldout_exported'] or manifest['historical_online_included']:
        raise ValueError('Initial offline loader rejects heldout/online contamination')
    with np.load(path, allow_pickle=False) as source:
        data = {split:{key[len(split)+1:]:source[key].copy() for key in source.files
                        if key.startswith(split+'_')} for split in SPLIT_NAMES}
        data['behavior_action'] = source['behavior_action'].copy()
        data['behavior_family'] = source['behavior_family'].copy()
    for split in SPLIT_NAMES:
        v = data[split]
        if len(v['raw_action']) != manifest['split_counts'][split] or v['metrics'].shape != (len(v['raw_action']),len(METRIC_NAMES)):
            raise ValueError('Q-study row count or metric shape mismatch')
        if not np.array_equal(np.isfinite(v['metrics']),v['metric_mask']):
            raise ValueError('Missing metric mask inconsistent with labels')
        if not np.isfinite(v['raw_action']).all() or not np.isfinite(v['executed_action']).all():
            raise ValueError('Nonfinite policy input coordinates')
    if set(data['train']['family']) & set(data['validation']['family']):
        raise ValueError('Train/validation family leakage')
    return data, manifest


def historical_online(output):
    experiments = [(BULK/'graph_o2o_v1', [0], 'bp_fe'),
                   (SCRATCH/'graph_o2o_replication_v1/seed_1_experiment', [1], 'bp_fe'),
                   (SCRATCH/'graph_o2o_replication_v1/seed_2_experiment', [2], 'bp_fe'),
                   (SCRATCH/'graph_o2o_expanded_policy_v1', [0], 'bp_fe'),
                   (SCRATCH/'graph_o2o_second_design_policy_v1', [0], 'bp_be')]
    rows, missing = [], []
    for exp, seeds, design in experiments:
        for seed in seeds:
            path = exp/f'seed_{seed}/replay_round_2.json'
            if not path.exists():
                missing.append(str(path));continue
            replay = read_json(path)
            label_sha = sha(path)
            for arm in ['bc','fql']:
                if len(replay['arms'][arm]) != 16:
                    raise ValueError('Historical online replay lost feedback')
                for index,label in enumerate(replay['arms'][arm]):
                    request = read_json(Path(label['outcome_path']).with_name('request.json'))
                    if request['role']!='collect' or not request['training_feedback_eligible'] or request['arm']!=arm:
                        raise ValueError('Evaluation sample cannot become historical replay')
                    row = _native(label,design,label['raw_action'],label['executed_action'],
                        'historical_online',-1,-1,index,'historical_own_collect_not_initial_offline',path,label_sha)
                    row.update(experiment=str(exp),training_seed=seed,arm=arm,
                               collection_round=label['collection_round'],run_id=label['run_id'])
                    rows.append(row)
    folder = output/'historical_online'
    folder.mkdir(exist_ok=False)
    (folder/'rows.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    result = dict(rows=len(rows),missing_replays=missing, included_in_initial_offline=False,
        selection='own collect roles actually used in the historical two online updates only; all evaluation roles excluded',
        exposure='previously used/inspected historical replay, not newly collected online evidence for this goal',
        rows_sha256=sha(folder/'rows.jsonl'))
    write_json(folder/'manifest.json',result)
    return result


def larger_design_inventory():
    # Use the known fixed six-cache location, never recursively scan shared storage.
    import torch
    cache_dir = ROOT/'baseline/runtime/benchmark_cache'
    entries = []
    for name in ['bp_fe','bp_be','bp','swerv_wrapper','ariane133','ariane136']:
        path = cache_dir/(name+'.pt')
        cache = torch.load(path,map_location='cpu',weights_only=False)
        geom = cache.get('macro_pos',cache.get('x')) if isinstance(cache,dict) else None
        shapes = {k:list(v.shape) for k,v in cache.items() if hasattr(v,'shape')} if isinstance(cache,dict) else {}
        entries.append(dict(design=name,cache=bind(path),tensor_shapes=shapes,
            q_training_labels_available=name in ['bp_fe','bp_be'],
            eligibility='Q-study initial datasets' if name in ['bp_fe','bp_be'] else
                'Only sparse historical Table2 physical evaluations known; different stage/config and not interchangeable with this GRT TRAIN distribution'))
    return entries


def inspect_historical_routes(output):
    """Read-only native validation of old DRT labels; never merge into TRAIN."""
    paths = [BULK/'graph_o2o_v1/full_route/summary.json',
             SCRATCH/'graph_o2o_replication_v1/seed_1_experiment/full_route/summary.json',
             SCRATCH/'graph_o2o_replication_v1/seed_2_experiment/full_route/summary.json',
             SCRATCH/'graph_o2o_expanded_policy_v1/full_route/summary.json',
             SCRATCH/'graph_o2o_second_design_policy_v1/full_route/summary.json']
    inventory = []
    for path in paths:
        summary = read_json(path)
        if summary['status'] != 'all_requested_outcomes_accounted':
            raise ValueError('Historical route summary is not terminal')
        entries = []
        for row in summary['records']:
            result = row['result']
            if result['status'] != 'completed':
                raise ValueError('Nonterminal historical detailed route')
            run = Path(result['physical_run'])
            manifest = checked_json(run/'manifest.json', result['physical_manifest_sha256'])
            metrics = checked_json(run/'metrics.json', result['metrics_sha256'])
            if metrics != result['metrics']:
                raise ValueError('Historical route metrics differ from native result')
            entries.append(dict(design=manifest['design'], physical_run=str(run),
                metrics=metrics, source_run=result['spec']['source_run'],
                route_end_iteration=result['spec']['route_end_iteration'],
                manifest=bind(run/'manifest.json'), metrics_binding=bind(run/'metrics.json'),
                included_in_initial_offline=False, already_inspected=True))
        inventory.append(dict(summary_binding=bind(path), records=entries,
                              logical_requests=len(entries)))
    duplicates = {}
    for name in ['bp_fe64','bp_fe384','bp_be64']:
        data,manifest = load_dataset(output/name)
        records = [json.loads(line) for line in (output/name/'rows.jsonl').read_text().splitlines()]
        groups = {}
        for row in records:
            groups.setdefault((row['split'],row['action_sha256']),[]).append(
                dict(family=row['family'],variant=row['variant'],source_row=row['source_row']))
        duplicates[name] = [dict(split=key[0],action_sha256=key[1],rows=value,
                                distinct_families=len({r['family'] for r in value}))
                            for key,value in groups.items() if len(value)>1]
    report = dict(status='completed',slurm_job_id=os.environ['SLURM_JOB_ID'],
        route_inventory=inventory,logical_requests=sum(i['logical_requests'] for i in inventory),
        unique_physical_runs=len({r['physical_run'] for i in inventory for r in i['records']}),
        initial_offline_included=False,already_inspected_not_fresh_test=True,
        duplicate_executed_actions=duplicates,source=bind(__file__))
    write_json(output/'historical_diagnostics.json',report)
    print(json.dumps(dict(status='completed',historical_route_logical=report['logical_requests'],
        unique_physical_runs=report['unique_physical_runs'],
        duplicate_groups={k:len(v) for k,v in duplicates.items()})),flush=True)


def run(args):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Assembly/decompression/statistics must run through Slurm')
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.inspect_routes:
        inspect_historical_routes(output)
        return
    if (output/'audit.json').exists():
        raise FileExistsError('Immutable completed audit already exists; choose a new output')
    bpfe, fa, fb, excluded = initial_bp_fe()
    extra, eb = expanded_bp_fe(fa)
    bpbe, ba, bb = initial_bp_be()
    manifests = {}
    for name,design,rows,arrays,bindings,n in [
        ('bp_fe64','bp_fe',bpfe,fa,fb,64),
        ('bp_fe384','bp_fe',bpfe+extra,fa,fb+eb,384),
        ('bp_be64','bp_be',bpbe,ba,bb,64)]:
        manifests[name] = export_dataset(output/name,design,rows,arrays,bindings,n)
    online = historical_online(output)
    audit = dict(status='completed',slurm_job_id=os.environ['SLURM_JOB_ID'],
        metric_names=list(METRIC_NAMES),datasets={k:dict(path=str(output/k/'dataset.npz'),
            manifest_sha256=sha(output/k/'manifest.json'),summary=v['summary']) for k,v in manifests.items()},
        excluded_bp_fe_heldout_rows=excluded,heldout_native_files_opened=False,
        historical_online=online,larger_design_inventory=larger_design_inventory(),
        contamination_notes=[
            'Historical16 validation rows per design have already guided prior diagnostics; use as tuning validation only.',
            'Historical fixed900000 evaluation noises,40-proposal critic diagnostics,gradient probes and published route summaries have been inspected; do not rebrand them as fresh confirmation.',
            'Same-noise perturbation variants remain in one family; split by family for cross-validation.',
            'Initial TRAIN, expanded TRAIN, historical own-online replay and evaluation must retain separate provenance and budgets.',
            'Physical parameters fixed per design; do not mix prior Table2 repair-cap4/exact-import runs with current GRT or cap20 final evaluation.',
            'Native signed WNS/TNS retained. Any objective scaling or transformed costs must be fitted on TRAIN only.',
            'Zero H/V overflow is a measured routing-edge statistic, not evidence of DRC0 or completed signoff.'],
        assembly_source=bind(__file__))
    write_json(output/'audit.json',audit)
    print(json.dumps(dict(status='completed',output=str(output),slurm_job_id=os.environ['SLURM_JOB_ID'],
        datasets={k:v['split_counts'] for k,v in manifests.items()},historical_online_rows=online['rows'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(BULK/'qstudy_v1/data_audit'))
    parser.add_argument('--inspect-routes',action='store_true')
    run(parser.parse_args())
