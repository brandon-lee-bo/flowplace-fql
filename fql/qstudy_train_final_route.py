"""Predeclare TRAIN-only detailed-route labels; never select evaluation layouts.

Thirty-two original TRAIN families (both variants) are chosen before DRT.
The unchanged graph_o2o_full_route worker continues their measured GRT states.
This module never fits a model, generates a layout, or reads old policy DRT.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_controller import LIVE, RESTARTABLE, states
from .graph_o2o_final_evidence import audit_final_outcome
from .qstudy_backend import PHYSICAL_REVISIONS
from .qstudy_data import METRIC_NAMES, action_sha, load_dataset

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path('/scratch/bohanli/flowplace_graph')
DATASET = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/bp_fe384/dataset.npz')
DATASET_SHA = '0daecf483b64283af46e3aec101161912ebff2984a493d0778fba5b2ebba99e4'
CONTAINER = Path('/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif')
VERSION = 'qstudy_measured_TRAIN_DRT64_v1'
SELECTION_SALT = 'qstudy_train_drt_v1_41307'
RESULT_ROOT = 'orfs_work/results/nangate45/bp_fe/eval/'
CHECKPOINT_NAMES = ('1_synth.v', '1_synth.sdc', '5_1_grt.odb', '5_1_grt.sdc', 'route.guide')
FINAL_METRICS = ('DRT_WL', 'WNS', 'TNS', 'Power', 'StdCellArea', 'DRC')
SOURCE_NAMES = ('fql/qstudy_train_final_route.py', 'fql/qstudy_train_final_route.slurm',
    'fql/qstudy_data.py', 'fql/graph_o2o_full_route.py', 'fql/graph_o2o_full_route.slurm',
    'fql/graph_o2o_finish.py', 'fql/graph_o2o_finish.sh', 'fql/graph_o2o_archive.py',
    'fql/graph_o2o_final_evidence.py', 'fql/graph_o2o_controller.py',
    'fql/qstudy_backend.py', 'fql/graph_o2o_backend.py',
    'backend/scripts/result_storage.py', 'backend/scripts/grt_provenance.py',
    'backend/scripts/grt_metrics.py', 'backend/scripts/canonical_table2_metrics.py',
    'analysis/audit_table2_macro_preservation.py',
    'upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py')


def read(path):
    return json.loads(Path(path).read_text())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind(path, expected=None):
    # Retain the original declared pathname, including recorded project aliases.
    path = Path(path).absolute()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Changed TRAIN source: ' + str(path))
    return dict(path=str(path), sha256=digest)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable TRAIN-DRT artifact changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a bounded Slurm CPU allocation')


def sources():
    return {name: sha(ROOT/name) for name in SOURCE_NAMES}


def load_train(dataset):
    dataset = Path(dataset).absolute()
    if dataset != DATASET or sha(dataset) != DATASET_SHA:
        raise ValueError('Only the exact audited bp_fe384 dataset is authorized')
    data, manifest = load_dataset(dataset)
    rows = [json.loads(line) for line in dataset.with_name('rows.jsonl').read_text().splitlines() if line.strip()]
    selected = [r for r in rows if r['split'] == 'train']
    train = data['train']
    if (len(selected) != 384 or len(train['raw_action']) != 384 or
            len({r['family'] for r in selected}) != 192):
        raise ValueError('Changed original TRAIN denominator')
    for i, row in enumerate(selected):
        if (row['design'] != 'bp_fe' or row['status'] != 'success' or
                row['family'] != int(train['family'][i]) or row['variant'] != int(train['variant'][i]) or
                row['source_row'] != int(train['source_row'][i]) or
                action_sha(train['raw_action'][i]) != row['raw_action_sha256'] or
                action_sha(train['executed_action'][i]) != row['action_sha256'] or
                any(row['metrics'][key] != train['metrics'][i, k] for k, key in enumerate(METRIC_NAMES))):
            raise ValueError('Audited TRAIN row/coordinate/native-label identity differs')
    return selected, manifest


def select_families(rows):
    """Four equal WL quartiles; SHA selection, then balanced family CV folds.

    Family WL is the arithmetic mean of its TWO native GRT WL labels. Sort
    (mean_WL, integer family) for deterministic ties. Each quartile contains48
    families; take the first8 by SHA(salt:family), assigning fold j modulo4.
    No DRT or old evaluation information is accepted by this function.
    """
    groups = {}
    for row in rows:
        if row['split'] != 'train':
            raise ValueError('Selection accepts original TRAIN rows only')
        groups.setdefault(int(row['family']), []).append(row)
    if len(groups) != 192:
        raise ValueError('Selection requires192 original TRAIN families')
    means = {}
    for family, pair in groups.items():
        if len(pair) != 2 or sorted(r['variant'] for r in pair) != [0, 1]:
            raise ValueError('Each selected family must retain both original variants')
        values = [r['metrics']['rWL_um'] for r in pair]
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Invalid TRAIN-only GRT wirelength')
        means[family] = sum(values)/2
    ordered = sorted(groups, key=lambda f: (means[f], f))
    result = []
    for quartile in range(4):
        candidates = ordered[48*quartile:48*(quartile+1)]
        ranked = sorted(candidates, key=lambda f: (hashlib.sha256(f'{SELECTION_SALT}:{f}'.encode()).hexdigest(), f))
        for j, family in enumerate(ranked[:8]):
            result.append(dict(family=family, quartile=quartile, fold=j % 4,
                family_mean_GRT_WL_um=means[family],
                selection_sha256=hashlib.sha256(f'{SELECTION_SALT}:{family}'.encode()).hexdigest()))
    return result


def native_source(row):
    """Validate measured TRAIN provenance only; never creates a geometry audit."""
    b = row['bindings']
    files = {key: bind(value['path'], value['sha256']) for key, value in b.items() if value is not None}
    run = Path(files['physical_manifest']['path']).parent
    manifest, metrics = read(run/'manifest.json'), read(run/'metrics.json')
    geometry_binding = bind(row['actual_backend_geometry']['path'], row['actual_backend_geometry']['sha256'])
    geometry = read(geometry_binding['path'])
    if (not manifest.get('success') or not manifest.get('time_end') or
            manifest.get('design') != 'bp_fe' or manifest.get('stage') != 'global_route_only' or
            int(manifest['num_cores']) != 8 or
            manifest['physical_flow_revision'] != PHYSICAL_REVISIONS['bp_fe'] or
            manifest['def_sha256'] != files['input_def']['sha256'] or
            Path(manifest['def_path']).absolute() != Path(files['input_def']['path']) or
            geometry['bindings']['physical_manifest_sha256'] != files['physical_manifest']['sha256'] or
            geometry.get('legal_no_halo') is not True or
            any(metrics.get(key) != row['metrics'][key] for key in METRIC_NAMES)):
        raise ValueError('TRAIN row does not bind a completed macro-legal eight-thread GRT source')
    if not isinstance(manifest.get('tool_image_sha256'), str) or len(manifest['tool_image_sha256']) != 64:
        raise ValueError('TRAIN source lacks its recorded physical container identity')
    files['backend_geometry'] = geometry_binding
    return dict(run=str(run), files=files, physical_flow_revision=manifest['physical_flow_revision'],
                tool_image_sha256=manifest['tool_image_sha256'])


def archive_binding(run, manifest_sha):
    """Support the original archive and the verified lossless legacy repack.

    A missing scratch cache is not missing scientific evidence. Compact
    archives are admitted only through their unchanged original archive record,
    exact decoded inventory, and completed legacy repack metadata.
    """
    run = Path(run)
    record_path, inventory_path = run/'grt_artifact_archive.json', run/'grt_intermediates_inventory.json'
    if not record_path.is_file() or not inventory_path.is_file():
        return None
    record, inventory = read(record_path), read(inventory_path)
    if (record.get('physical_manifest_sha256') != manifest_sha or
            record.get('decoded_full_content_verified') is not True or len(inventory) != record['file_count']):
        raise ValueError('Archive does not bind the exact original TRAIN physical source')
    archive = Path(record['archive_path'])
    extra = None
    legacy_path = run/'legacy_archive_repack.json'
    # Prefer the durable compact file when a completed repack exists. Its
    # original-byte scratch cache can disappear after declaration without
    # invalidating the verified decoded source inventory.
    if legacy_path.is_file():
        legacy = read(legacy_path)
        if (legacy.get('status') != 'durable_compact_archive_and_exact_original_cache' or
                legacy.get('original_archive_sha256') != record['archive_sha256'] or
                legacy.get('physical_manifest_sha256') != manifest_sha or
                legacy.get('inventory_sha256') != sha(inventory_path) or
                legacy.get('archive_record_sha256') != sha(record_path)):
            raise ValueError('Missing cache has no completed exact legacy restoration binding')
        archive = Path(legacy['compact_path'])
        expected = legacy['compact_sha256']
        extra = bind(legacy_path)
    else:
        expected = record['archive_sha256']
    return dict(archive=bind(archive, expected), archive_record=bind(record_path),
                inventory=bind(inventory_path), legacy_repack=extra, decoded_inventory=inventory)


def checkpoint_binding(native):
    run = Path(native['run'])
    archive = archive_binding(run, native['files']['physical_manifest']['sha256'])
    inventory = archive['decoded_inventory'] if archive else {}
    required, missing = {}, []
    for name in CHECKPOINT_NAMES:
        relative = RESULT_ROOT+name
        path = run/relative
        if path.is_file():
            item = dict(bytes=path.stat().st_size, sha256=sha(path))
            if relative in inventory and item != inventory[relative]:
                raise ValueError('Live checkpoint differs from its decoded archive inventory')
        elif relative in inventory:
            item = inventory[relative]
            missing.append(relative)
        else:
            raise FileNotFoundError('No original checkpoint or verified archive member: '+str(path))
        required[relative] = item
    for kind in ('logs', 'reports'):
        folder = run/f'orfs_work/{kind}/nangate45/bp_fe/eval'
        if not folder.is_dir():
            raise FileNotFoundError('Required original source '+kind+' missing')
        paths = sorted(p for p in folder.rglob('*') if p.is_file())
        if not paths:
            raise ValueError('Empty original source '+kind)
        for path in paths:
            required[str(path.relative_to(run))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return dict(files=required, missing_at_declaration=missing, restore_required=bool(missing),
                archive=archive, source_checkpoint_bytes=sum(v['bytes'] for v in required.values()))


def prepare(dataset, output):
    rows, manifest = load_train(dataset)
    selected = select_families(rows)
    output = Path(output).absolute()
    if output != output.resolve() or output == SCRATCH_ROOT or not output.is_relative_to(SCRATCH_ROOT):
        raise ValueError('Use a new canonical project scratch output directory')
    tasks = []
    for family in selected:
        pair = sorted((r for r in rows if r['family'] == family['family']), key=lambda r: r['variant'])
        for row in pair:
            native = native_source(row)
            key = f"f{row['family']:03d}_v{row['variant']}"
            tasks.append(dict(index=len(tasks), key=key, **family, variant=row['variant'],
                provenance=row, provenance_sha256=object_sha(row), native=native,
                checkpoint=checkpoint_binding(native), output=str(output/'full_route'/key),
                run_id='go2o_train_drt64_'+key, method='measured_original_TRAIN_DRT64'))
    image_sha = tasks[0]['native']['tool_image_sha256']
    if any(t['native']['tool_image_sha256'] != image_sha for t in tasks):
        raise ValueError('Selected measured TRAIN sources do not share the exact physical image')
    container = bind(CONTAINER, image_sha)
    plan = dict(version=VERSION, output=str(output), dataset=bind(dataset, DATASET_SHA),
        dataset_manifest=bind(Path(dataset).with_name('manifest.json')),
        rows=bind(Path(dataset).with_name('rows.jsonl'), manifest['rows_sha256']),
        selection=dict(families=32, variants_per_family=2, requested=64, quartiles=4,
            families_per_quartile=8, salt=SELECTION_SALT, family_statistic='arithmetic_mean_of_two_GRT_WL_um',
            ties='mean_WL_then_integer_family; SHA_then_family', folds=4,
            fold_rule='within_quartile_selected_SHA_order_modulo4', selected_families=selected),
        protocol=dict(backend_threads=8, route_end_iteration=20, storage='scratch', max_concurrency=2,
            role='original_TRAIN_label_extension', old_policy_DRT_labels_used=False,
            old_validation_or_eval_labels_used=False, policy_layout_selection=False,
            failure_replacement=False, model_updates=0, layout_generation_calls=0),
        software=dict(python=sys.version, numpy=np.__version__, container=container), source_sha256=sources(), tasks=tasks,
        job_ids=dict(prepare=(read(output/'plan.json')['job_ids']['prepare']
            if (output/'plan.json').is_file() else os.environ['SLURM_JOB_ID'])),
        availability=dict(requested=64, sources_ready=sum(not t['checkpoint']['restore_required'] for t in tasks),
            sources_needing_verified_restore=sum(t['checkpoint']['restore_required'] for t in tasks)))
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'plan.json').exists() and any(output.glob('full_route/*')):
        raise ValueError('Cannot declare after any DRT output exists')
    for name, digest in plan['source_sha256'].items():
        path = output/'sources'/name
        if path.exists():
            bind(path, digest)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
            bind(path, digest)
    freeze(output/'plan.json', plan)
    return plan


def checked_plan(path, expected_sha256):
    path = Path(path).absolute()
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in expected_sha256) or sha(path) != expected_sha256):
        raise ValueError('Changed or unpinned TRAIN-DRT plan SHA')
    plan = read(path)
    output = Path(plan['output'])
    if (plan.get('version') != VERSION or path != output/'plan.json' or
            output != output.resolve() or output == SCRATCH_ROOT or not output.is_relative_to(SCRATCH_ROOT) or
            plan['source_sha256'] != sources() or len(plan['tasks']) != 64 or
            plan['protocol'] != dict(backend_threads=8, route_end_iteration=20, storage='scratch', max_concurrency=2,
                role='original_TRAIN_label_extension', old_policy_DRT_labels_used=False,
                old_validation_or_eval_labels_used=False, policy_layout_selection=False,
                failure_replacement=False, model_updates=0, layout_generation_calls=0)):
        raise ValueError('Changed TRAIN-only64 plan/source/protocol')
    for value in (plan['dataset'], plan['dataset_manifest'], plan['rows']):
        bind(value['path'], value['sha256'])
    image = plan['software']['container']
    if (Path(image['path']) != CONTAINER or
            any(t['native']['tool_image_sha256'] != image['sha256'] for t in plan['tasks'])):
        raise ValueError('TRAIN source/continuation physical image binding differs')
    bind(image['path'], image['sha256'])
    for name, digest in plan['source_sha256'].items():
        bind(Path(plan['output'])/'sources'/name, digest)
    rows, _ = load_train(plan['dataset']['path'])
    families = select_families(rows)
    expected_selection = dict(families=32, variants_per_family=2, requested=64, quartiles=4,
        families_per_quartile=8, salt=SELECTION_SALT, family_statistic='arithmetic_mean_of_two_GRT_WL_um',
        ties='mean_WL_then_integer_family; SHA_then_family', folds=4,
        fold_rule='within_quartile_selected_SHA_order_modulo4', selected_families=families)
    if expected_selection != plan['selection']:
        raise ValueError('TRAIN-only family selection or folds changed')
    expected = [r for f in families for r in sorted((r for r in rows if r['family'] == f['family']), key=lambda r:r['variant'])]
    if len({t['run_id'] for t in plan['tasks']}) != 64 or len({t['output'] for t in plan['tasks']}) != 64:
        raise ValueError('Duplicate output or native request identity')
    for i, (task, row) in enumerate(zip(plan['tasks'], expected)):
        family = families[i//2]
        key = f"f{row['family']:03d}_v{row['variant']}"
        if (task['index'] != i or task['key'] != key or task['provenance'] != row or
                task['provenance_sha256'] != object_sha(row) or task['variant'] != row['variant'] or
                any(task[k] != v for k,v in family.items()) or
                task['output'] != str(Path(plan['output'])/'full_route'/key) or
                task['run_id'] != 'go2o_train_drt64_'+key or task['method'] != 'measured_original_TRAIN_DRT64'):
            raise ValueError('TRAIN task identity/coordinate/family/fold binding changed')
    return plan


def copy_checked(source, destination, expected):
    if source.stat().st_size != expected['bytes'] or sha(source) != expected['sha256']:
        raise ValueError('Source checkpoint changed before copying')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.stat().st_size != expected['bytes'] or sha(destination) != expected['sha256']:
        raise ValueError('Restored/copied checkpoint differs')


def decode_required(archive, inventory, wanted, target):
    """Verify EVERY decoded archive member; copy only required regular files."""
    bind(archive['path'], archive['sha256'])
    process = subprocess.Popen(['zstd', '-dc', '--long=28', archive['path']], stdout=subprocess.PIPE)
    seen = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
            for member in stream:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or str(path) != member.name:
                    raise ValueError('Unsafe archive member')
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in inventory or member.name in seen:
                    raise ValueError('Unexpected/nonregular/duplicate archive member')
                digest = hashlib.sha256()
                destination = target/member.name
                handle = None
                if member.name in wanted:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    handle = destination.open('xb')
                try:
                    with stream.extractfile(member) as raw:
                        for chunk in iter(lambda:raw.read(1024**2), b''):
                            digest.update(chunk)
                            if handle:
                                handle.write(chunk)
                finally:
                    if handle:
                        handle.close()
                seen[member.name] = dict(bytes=member.size, sha256=digest.hexdigest())
        if process.wait() or seen != inventory or not set(wanted).issubset(seen):
            raise ValueError('Full decoded TRAIN archive inventory differs')
    finally:
        if process.poll() is None:
            process.kill()
        process.stdout.close()
        process.wait()


def ready_source(plan, task):
    """Never modify original sources; missing caches restore to a private copy."""
    actual = native_source(task['provenance'])
    if actual != task['native']:
        raise ValueError('Original TRAIN native evidence changed')
    original, cp = Path(actual['run']), task['checkpoint']
    missing = []
    for relative, item in cp['files'].items():
        path = original/relative
        if path.is_file():
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('Original measured GRT checkpoint changed')
        else:
            missing.append(relative)
    if not missing:
        return original
    archive = cp['archive']
    if not archive or any(name not in archive['decoded_inventory'] for name in missing):
        raise FileNotFoundError('Missing original source not recoverable from its bound archive')
    for key in ('archive_record', 'inventory', 'legacy_repack'):
        if archive[key]:
            bind(archive[key]['path'], archive[key]['sha256'])
    target = Path(plan['output'])/'restored_sources'/task['key']
    identity = dict(plan_sha256=sha(Path(plan['output'])/'plan.json'), key=task['key'],
        original_run=str(original), native=actual, checkpoint_sha256=object_sha(cp),
        new_GRT_measurements=0, old_source_mutated=False)
    if target.exists():
        if read(target/'restore_completed.json') != identity:
            raise ValueError('Foreign or partial source restoration')
    else:
        stage = target.with_name(target.name+'.staging_'+os.environ['SLURM_JOB_ID'])
        stage.mkdir(parents=True, exist_ok=False)
        for name, key in (('manifest.json','physical_manifest'), ('metrics.json','physical_metrics'),
                          ('backend_macro_audit.json','backend_geometry')):
            source = Path(actual['files'][key]['path'])
            copy_checked(source, stage/name, dict(bytes=source.stat().st_size, sha256=actual['files'][key]['sha256']))
        decode_required(archive['archive'], archive['decoded_inventory'], missing, stage)
        for relative, item in cp['files'].items():
            if relative not in missing:
                copy_checked(original/relative, stage/relative, item)
        freeze(stage/'restore_completed.json', identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(target)
    for relative, item in cp['files'].items():
        path = target/relative
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Private restored checkpoint changed')
    for name, key in (('manifest.json','physical_manifest'), ('metrics.json','physical_metrics'),
                      ('backend_macro_audit.json','backend_geometry')):
        bind(target/name, actual['files'][key]['sha256'])
    return target


def inspect_result(plan, task):
    path = Path(task['output'])/'outcome.json'
    if not path.is_file():
        return dict(index=task['index'], key=task['key'], state='missing', metrics=None)
    out = read(path)
    audited = audit_final_outcome(out, design='bp_fe', route_cap=20)
    source = Path(audited['source_run'])
    original = Path(task['native']['run']).resolve()
    restored = Path(plan['output'])/'restored_sources'/task['key']
    if (source not in (original, restored.resolve()) or
            out['spec']['source_manifest_sha256'] != task['native']['files']['physical_manifest']['sha256'] or
            out['spec']['run_id'] != task['run_id'] or out['spec']['method'] != task['method'] or
            out['spec'].get('storage') != 'scratch'):
        raise ValueError('DRT result did not continue its declared TRAIN source/spec')
    return dict(index=task['index'], key=task['key'], family=task['family'], fold=task['fold'],
        variant=task['variant'], state='completed', outcome=bind(path),
        metrics=out['metrics'], DRC_zero=out['metrics']['DRC'] == 0,
        actual_macro_preserved=True, new_DRT_label=True, training_eligible=True,
        raw_action_sha256=task['provenance']['raw_action_sha256'],
        action_sha256=task['provenance']['action_sha256'])


def attempt_evidence(folder):
    """Retain failures without turning an unclassified tool error into a label."""
    result = []
    for path in sorted(Path(folder).glob('attempt_*/binding.json')):
        binding = read(path)
        record = dict(binding=bind(path), slurm_job_id=binding['slurm_job_id'],
            physical_run=binding['physical_run'])
        for name, key in (('manifest.json', 'physical_manifest'), ('metrics.json', 'physical_metrics')):
            source = Path(binding['physical_run'])/name
            if source.is_file():
                record[key] = bind(source)
                if key == 'physical_manifest':
                    physical = read(source)
                    record.update(native_success=physical.get('success'), time_end=physical.get('time_end'),
                        native_error=physical.get('error'))
        exit_path = path.with_name('exit.json')
        if exit_path.is_file():
            record.update(process_exit=bind(exit_path), returncode=read(exit_path)['returncode'])
        result.append(record)
    return result


def prior_attempts_safe(folder):
    """Only ordinary infrastructure preemptions retry automatically."""
    for path in sorted(folder.glob('attempt_*/binding.json')):
        binding = read(path)
        manifest = Path(binding['physical_run'])/'manifest.json'
        if manifest.is_file() and read(manifest).get('success') and read(manifest).get('time_end'):
            continue  # Exact old worker recovers reporting/storage without rerouting.
        observed = states(binding['slurm_job_id'])
        if not observed or any(s in LIVE or s == 'ACCOUNTING_PENDING' for s in observed) or not all(s in RESTARTABLE for s in observed):
            raise RuntimeError('Prior DRT attempt requires diagnosis; genuine states='+repr(observed))


def worker(plan_path, index, plan_sha256):
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 8:
        raise RuntimeError('Use exactly8 allocated physical-worker CPUs')
    plan = checked_plan(plan_path, plan_sha256)
    if not 0 <= index < 64:
        raise IndexError('TRAIN DRT index outside the fixed64')
    task = plan['tasks'][index]
    folder = Path(task['output'])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.train_drt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/'outcome.json').exists():
            return inspect_result(plan, task)
        prior_attempts_safe(folder)
        source = ready_source(plan, task)
        command = [sys.executable, '-u', '-m', 'fql.graph_o2o_full_route', '--source-run', str(source),
            '--run-id', task['run_id'], '--method', task['method'], '--output', str(folder),
            '--end-iter', '20', '--storage', 'scratch']
        freeze(folder/('worker_'+os.environ['SLURM_JOB_ID']+'.json'), dict(plan=bind(plan_path),
            task=index, command=command, source_run=str(source), sources=sources(),
            slurm_job_id=os.environ['SLURM_JOB_ID'], no_model_or_layout_generation=True))
        env = dict(os.environ)
        env.pop('SLURM_RESTART_COUNT', None)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        result = inspect_result(plan, task)
        if result['state'] != 'completed':
            raise RuntimeError('Route worker exited without a validated native DRT label')
        return result


def report(plan_path, output, plan_sha256):
    plan = checked_plan(plan_path, plan_sha256)
    rows = []
    for task in plan['tasks']:
        try:
            row = inspect_result(plan, task)
        except Exception as error:
            row = dict(index=task['index'], key=task['key'], state='invalid', metrics=None, reason=str(error))
        row['attempts'] = attempt_evidence(task['output'])
        if row['state'] == 'missing' and any(a.get('returncode', 0) != 0 for a in row['attempts']):
            row.update(state='unresolved_attempt_failure',
                reason='Native/tool or infrastructure failure requires diagnosis; no fictitious DRT PPA')
        rows.append(row)
    counts = dict(Counter(r['state'] for r in rows))
    value = dict(version=VERSION, plan=bind(plan_path), source_sha256=sources(),
        requested=64, counts=counts, status='complete' if counts.get('completed') == 64 else 'incomplete',
        DRC_zero=sum(r.get('DRC_zero', False) for r in rows), records=rows,
        source_split='original384TRAIN_only', policy_evaluation_or_old_DRT_used=False,
        family_folds=plan['selection']['selected_families'],
        note='New measured TRAIN labels only; no model fit, reward choice, actor/PPA efficacy claim or winner selection',
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    output = Path(output).absolute()
    if output.exists() or output.is_relative_to(Path(plan['output'])/'full_route'):
        raise ValueError('Write a new report outside immutable native outputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'worker', 'report'))
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256', help='Externally pinned immutable64-request declaration SHA')
    parser.add_argument('--index', type=int, default=os.environ.get('SLURM_ARRAY_TASK_ID'))
    args = parser.parse_args()
    slurm()
    if args.stage == 'prepare':
        if args.output is None:
            parser.error('prepare requires --output')
        result = prepare(args.dataset, args.output)
    elif args.stage == 'worker':
        if args.plan is None or args.index is None or args.plan_sha256 is None:
            parser.error('worker requires --plan, --plan-sha256 and --index')
        result = worker(args.plan, args.index, args.plan_sha256)
    else:
        if args.plan is None or args.output is None or args.plan_sha256 is None:
            parser.error('report requires --plan, --plan-sha256 and a new --output file')
        result = report(args.plan, args.output, args.plan_sha256)
    print(json.dumps(dict(stage=args.stage, status=result.get('status', result.get('state', 'prepared')),
                         slurm_job_id=os.environ['SLURM_JOB_ID']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
