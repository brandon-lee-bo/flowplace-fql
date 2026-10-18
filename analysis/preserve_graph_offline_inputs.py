"""Retain completed offline datasets and their newly collected native evidence.

This closes the scratch-only input-data gap. It neither relabels data nor runs
training/EDA. Initial bp_fe arrays include separately marked unused heldout rows;
copying their bytes does not make them eligible for training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from backend.scripts.result_storage import MIN_HEADROOM, storage_status
from fql.graph_o2o_archive import atomic_json, sha
from fql.graph_o2o_checkpoint_archive import verify

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = (ROOT, Path('/fact_data/bohanli'), Path('/scratch/bohanli'))


def preserve(plan_path):
    plan = json.loads(plan_path.read_text())
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Registered implementation changed: ' + name)
    files, aliases, audits = {}, {}, []

    def add(path, expected=None):
        original = Path(path).absolute()
        source = original.resolve(strict=True)
        if not source.is_file() or not any(source.is_relative_to(p) for p in ALLOWED):
            raise ValueError('Invalid source evidence: ' + str(original))
        if expected is not None and sha(source) != expected:
            raise ValueError('Bound input changed: ' + str(original))
        aliases[str(original)] = str(source)
        files['files/' + str(source).lstrip('/')] = source

    def read(path, expected=None):
        add(path, expected)
        return json.loads(Path(path).read_text())

    for binding in plan['input_files']:
        add(binding['path'], binding['sha256'])
    for study in plan['measured_collections']:
        data = read(study['labels'], study['labels_sha256'])
        rows = data['rows'] if study['format'] == 'expanded_replay' else data
        if len(rows) != study['requested']:
            raise ValueError('Offline label denominator changed')
        if study['format'] == 'expanded_replay' and (
                data['status'] != 'all_requested_accounted'
                or data['collection_kind'] != 'additional_offline_TRAIN_labels_not_on_policy'
                or data['new_requested_labels'] != 320
                or data['validation_heldout_used']
                or data['reward_scale'] != study['reward_scale']):
            raise ValueError('Expanded labels changed their offline TRAIN scope')
        if study['format'] == 'initial_labels':
            initial = read(study['dataset_manifest'], study['dataset_manifest_sha256'])
            if (initial['requested'] != len(rows) or initial['successful'] != len(rows)
                    or initial['TRAIN_requested'] != 64 or initial['validation_requested'] != 16
                    or initial['backend_labels_sha256'] != study['labels_sha256']
                    or initial['reward_scale_train_only'] != study['reward_scale']):
                raise ValueError('Initial offline dataset/normalization scope changed')
        seen = set()
        for row in rows:
            outcome_path = Path(row['outcome_path'])
            if str(outcome_path) in seen:
                raise ValueError('Duplicate logical offline observation')
            seen.add(str(outcome_path))
            out = read(outcome_path, row['outcome_sha256'])
            request = read(outcome_path.parent / 'request.json', out['request_sha256'])
            # All400 declared observations succeeded in the completed source
            # audits. A newly changed failure must not silently disappear.
            if out['status'] != row['status'] or out['status'] != 'success':
                raise ValueError('Successful completed offline scope changed')
            for key in ['raw_action', 'executed_action', 'action_sha256', 'terminal_mask', 'metrics']:
                if row[key] != out[key]:
                    raise ValueError('Input data differs from actual outcome: ' + key)
            for key in ['raw_action', 'executed_action', 'action_sha256']:
                if request[key] != out[key]:
                    raise ValueError('Offline request/action binding differs: ' + key)
            if out['terminal_mask'] != 0:
                raise ValueError('Offline termination context changed')
            if study['format'] == 'expanded_replay':
                if out['reward'] != row['reward'] or out['reward_scale'] != study['reward_scale']:
                    raise ValueError('Expanded offline reward context changed')
                reward_source = 'measured_outcome_with_preexisting_frozen_scale'
            else:
                # Initial measurements intentionally store no normalized reward:
                # the TRAIN mean is frozen only after all80 have been audited.
                if out['reward'] is not None or out['reward_scale'] is not None or out['policy_feedback']:
                    raise ValueError('Initial measurement/derived-label semantics changed')
                reward_source = 'derived_label_with_subsequently_frozen_initial_TRAIN_scale'
            run = Path(out['physical_run'])
            manifest = read(run / 'manifest.json', out['physical_manifest_sha256'])
            metrics = read(run / 'metrics.json', out['metrics_sha256'])
            if metrics != out['metrics'] or not math.isclose(
                    row['reward'], -metrics['rWL_um'] / study['reward_scale'], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Offline reward differs from the measured GRT result')
            if (manifest['stage'] != 'global_route_only' or not manifest['success']
                    or manifest['design'] != study['design'] or int(manifest['num_cores']) != 8
                    or manifest['tool_image_sha256'] != study['tool_image_sha256']
                    or manifest['physical_flow_components'] != study['physical_flow_components']
                    or manifest['def_sha256'] != request['def_sha256']):
                raise ValueError('Actual offline execution uses a different physical context')
            add(request['def_path'], request['def_sha256'])
            add(manifest['def_path'], manifest['def_sha256'])
            geometry_key = ('backend_macro_geometry_path' if 'backend_macro_geometry_path' in out
                            else 'backend_geometry_path')
            geometry_hash = geometry_key.replace('_path', '_sha256')
            read(out[geometry_key], out[geometry_hash])
            if not out['backend_macro_legal']:
                raise ValueError('Declared successful offline geometry is illegal')
            for name in ['actions.npz', 'sample.json', 'macro_physical.json']:
                path = outcome_path.parent / name
                if path.is_file():
                    add(path)
            for source in run.iterdir():
                if source.is_file() and source.suffix in {'.json', '.log'}:
                    add(source)
            for kind in ['logs', 'reports']:
                folder = run / 'orfs_work' / kind / 'nangate45' / study['design'] / 'eval'
                retained = 0
                for source in folder.iterdir():
                    if source.is_file() and source.suffix in {'.json', '.log', '.rpt'}:
                        add(source)
                        retained += 1
                if not retained:
                    raise ValueError('Successful offline measurement lacks native ' + kind)
            audits.append(dict(study=study['id'], outcome_path=str(outcome_path),
                outcome_sha256=row['outcome_sha256'], source_row_index=row['source_row_index'],
                status=out['status'], reward=row['reward'], reward_source=reward_source,
                action_sha256=out['action_sha256'],
                physical_run=str(run), physical_manifest_sha256=out['physical_manifest_sha256']))
    if len(audits) != plan['requested_native_observations']:
        raise ValueError('Complete offline preservation scope changed')
    for name in plan['source_sha256']:
        add(ROOT / name)
    add(plan_path)
    inventory = {member: dict(bytes=source.stat().st_size, sha256=sha(source))
                 for member, source in sorted(files.items())}
    scratch = Path(os.environ['TMPDIR'])
    tree = scratch / 'tree'
    tree.mkdir(parents=True, exist_ok=True)
    for member, source in files.items():
        target = tree / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
        elif target.resolve() != source:
            raise ValueError('Interrupted staging link differs')
    names = scratch / 'files.null'
    names.write_bytes(b''.join(member.encode() + b'\0' for member in sorted(inventory)))
    staged = scratch / 'offline_inputs.tar.zst'
    subprocess.run(['tar', '--dereference', '-I', 'zstd -T2 -3 --long=28', '-cf', str(staged),
                    '-C', str(tree), '--null', '-T', str(names)], check=True)
    verify(staged, inventory)
    for member, source in files.items():
        if sha(source) != inventory[member]['sha256']:
            raise ValueError('Original evidence changed during preservation')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'offline_inputs.tar.zst'
    storage = storage_status(Path('/fact_data/bohanli'), output)
    if not archive.exists():
        for key in ['filesystem_free_bytes', 'quota_remaining_bytes']:
            if storage.get(key) is not None and storage[key] - staged.stat().st_size < MIN_HEADROOM:
                raise RuntimeError('Preservation would breach8GiB reserve')
        pending = output / ('pending_job_' + os.environ['SLURM_JOB_ID'] + '.tar.zst')
        with pending.open('xb') as dest, staged.open('rb') as src:
            shutil.copyfileobj(src, dest, length=4 * 1024**2)
            dest.flush()
            os.fsync(dest.fileno())
        if sha(pending) != sha(staged):
            raise ValueError('Durable copy differs')
        pending.rename(archive)
    verify(archive, inventory)
    atomic_json(output / 'inventory.json', inventory)
    atomic_json(output / 'path_aliases.json', aliases)
    atomic_json(output / 'offline_observation_audits.json', audits)
    record = dict(status='offline_inputs_and_native_evidence_preserved_and_fully_decoded',
        plan_sha256=sha(plan_path), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        file_count=len(inventory), source_bytes=sum(r['bytes'] for r in inventory.values()),
        native_observations=len(audits), collection_counts={s['id']: s['requested'] for s in plan['measured_collections']},
        inventory_sha256=sha(output / 'inventory.json'), aliases_sha256=sha(output / 'path_aliases.json'),
        audits_sha256=sha(output / 'offline_observation_audits.json'), source_files_unchanged=True,
        new_training_or_EDA=False, copying_heldout_bytes_does_not_authorize_training=True,
        EDA_or_training_resume_archive=False,
        limitations='Input arrays, labels, actions, import DEFs and native GRT logs/reports; full OpenDB/SPEF and optimizer state remain at original locations.',
        restoration='Decode into an empty directory; files/ mirrors resolved original paths. Verify inventory before deliberate restoration; never overwrite live experiments.',
        storage_before=storage, slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output / 'manifest.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for native offline auditing and compression')
    preserve(args.plan.resolve())
