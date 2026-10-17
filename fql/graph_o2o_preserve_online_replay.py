"""Preserve the exact measured feedback actually used by online graph policies.

Archive input DEFs, request/action/outcome bindings and native GRT reports.
Full OpenDB/SPEF and optimizer checkpoints stay at their original locations.
This preserves completed observations; it does not run EDA or train models.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from backend.scripts.result_storage import MIN_HEADROOM, storage_status
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_checkpoint_archive import verify
from .graph_o2o_collect import audit_outcome

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = (ROOT, Path('/fact_data/bohanli'), Path('/scratch/bohanli'))


def preserve(plan_path):
    plan = json.loads(plan_path.read_text())
    if len({r['id'] for r in plan['replays']}) != len(plan['replays']):
        raise ValueError('Duplicate study preservation IDs')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Preservation source changed: ' + name)
    files, aliases, audits = {}, {}, []

    def add(path):
        original = Path(path).absolute()
        source = original.resolve(strict=True)
        if not source.is_file() or not any(source.is_relative_to(p) for p in ALLOWED):
            raise ValueError('Invalid evidence file: ' + str(original))
        aliases[str(original)] = str(source)
        files['files/' + str(source).lstrip('/')] = source

    def physical_files(run, design, required=False):
        run = Path(run)
        for source in run.iterdir():
            if source.is_file() and source.suffix in {'.json', '.log'}:
                add(source)
        for kind in ['logs', 'reports']:
            folder = run / 'orfs_work' / kind / 'nangate45' / design / 'eval'
            if required and not folder.is_dir():
                raise ValueError('Successful GRT result is missing native ' + kind)
            if folder.is_dir():
                retained = 0
                for source in folder.iterdir():
                    if source.is_file() and source.suffix in {'.json', '.log', '.rpt'}:
                        add(source)
                        retained += 1
                if required and not retained:
                    raise ValueError('Successful GRT result has no native ' + kind)

    for study in plan['replays']:
        replay_path, config_path = Path(study['replay']), Path(study['config'])
        if sha(replay_path) != study['replay_sha256'] or sha(config_path) != study['config_sha256']:
            raise ValueError('Frozen replay/config changed')
        replay, cfg = json.loads(replay_path.read_text()), json.loads(config_path.read_text())
        if (replay['seed'] != study['seed'] or replay['for_training_round'] != 2
                or replay['includes_collection_rounds'] != [0, 1]
                or replay['config_sha256'] != study['config_sha256']
                or set(replay['arms']) != {'bc', 'fql'}):
            raise ValueError('Replay identity or actual training scope changed')
        add(replay_path)
        add(config_path)
        training_path = Path(study['training_status'])
        if sha(training_path) != study['training_status_sha256']:
            raise ValueError('Training status binding changed')
        training = json.loads(training_path.read_text())
        if (training['status'] != 'completed' or training['seed'] != study['seed']
                or training['round'] != 2 or training['config_sha256'] != study['config_sha256']
                or training['arms'] != {'bc': 'completed', 'fql': 'completed'}
                or training['reward_scale'] != replay['reward_scale']
                or training['terminal_mask'] != 0 or not training['no_heldout_labels_used']):
            raise ValueError('Preserve feedback only after both actual training arms finish')
        add(training_path)
        if set(study['completions']) != {'bc', 'fql'}:
            raise ValueError('Both completed training arms must be bound')
        for arm, binding in study['completions'].items():
            completion_path = Path(binding['path'])
            if sha(completion_path) != binding['sha256']:
                raise ValueError('Training completion changed')
            done = json.loads(completion_path.read_text())
            if (done['status'] != 'completed' or done['seed'] != study['seed']
                    or done['round'] != 2 or done['arm'] != arm
                    or done['online_training_rows'] != len(replay['arms'][arm])
                    or done['slurm_job_id'] != training['slurm_job_id']):
                raise ValueError('Completed training does not bind the requested replay scope')
            add(completion_path)
        used = set()
        for arm, rows in replay['arms'].items():
            if len(rows) != 16:
                raise ValueError('Incomplete two-round feedback denominator')
            coordinates = []
            for row in rows:
                outcome_path = Path(row['outcome_path'])
                if sha(outcome_path) != row['outcome_sha256']:
                    raise ValueError('Measured replay outcome changed')
                request_path = outcome_path.parent / 'request.json'
                request = json.loads(request_path.read_text())
                actual = audit_outcome(request, cfg, replay['reward_scale'])
                if (actual['seed'] != study['seed'] or actual['arm'] != arm
                        or actual['role'] != 'collect' or actual['round'] not in [0, 1]
                        or not actual['training_feedback_eligible']
                        or actual['run_id'] in used or request['graph'] != study['design']):
                    raise ValueError('Repeated, cross-arm or evaluation feedback')
                for key in ['raw_action', 'executed_action', 'reward', 'status',
                            'terminal_mask', 'run_id', 'action_sha256', 'metrics']:
                    if row[key] != actual[key]:
                        raise ValueError('Replay/action/measurement differs: ' + key)
                if row['collection_round'] != actual['round']:
                    raise ValueError('Replay collection round changed')
                coordinates.append((actual['round'], actual['index']))
                used.add(actual['run_id'])
                add(request_path)
                add(outcome_path)
                for name in ['actions.npz', 'sample.json', 'macro_physical.json']:
                    path = outcome_path.parent / name
                    if path.is_file():
                        add(path)
                if request.get('def_path'):
                    if sha(request['def_path']) != request['def_sha256']:
                        raise ValueError('Requested macro DEF changed')
                    add(request['def_path'])
                if actual.get('backend_macro_geometry_path'):
                    add(actual['backend_macro_geometry_path'])
                if actual['status'] == 'success':
                    run = Path(actual['physical_run'])
                    manifest = json.loads((run / 'manifest.json').read_text())
                    if (manifest['design'] != study['design'] or int(manifest['num_cores']) != 8
                            or manifest['tool_image_sha256'] != study['tool_image_sha256']
                            or manifest['physical_flow_components'] != study['physical_flow_components']
                            or manifest['physical_flow_revision'] != actual['physical_flow_revision']
                            or not manifest['success'] or manifest['stage'] != 'global_route_only'
                            or manifest['def_sha256'] != request['def_sha256']
                            or sha(manifest['def_path']) != manifest['def_sha256']):
                        raise ValueError('Actual input/tool/physical context differs')
                    add(manifest['def_path'])
                    physical_files(run, study['design'], required=True)
                elif actual.get('physical_run') and Path(actual['physical_run']).is_dir():
                    physical_files(actual['physical_run'], study['design'])
                audits.append(dict(study=study['id'], arm=arm, collection_round=actual['round'],
                    index=actual['index'], status=actual['status'], reward=actual['reward'],
                    outcome_sha256=actual['outcome_sha256'], action_sha256=actual['action_sha256'],
                    input_checkpoint_sha256=actual['checkpoint_sha256'],
                    physical_run=actual.get('physical_run'), failure_class=actual.get('failure_class')))
            if sorted(coordinates) != [(r, i) for r in range(2) for i in range(8)]:
                raise ValueError('Missing or duplicated logical collection observation')
    if len(audits) != plan['requested_used_observations']:
        raise ValueError('Requested used-feedback scope changed')
    for name in plan['source_sha256']:
        add(ROOT / name)
    add(plan_path)

    inventory = {member: dict(bytes=path.stat().st_size, sha256=sha(path))
                 for member, path in sorted(files.items())}
    scratch = Path(os.environ['TMPDIR'])
    tree = scratch / 'tree'
    tree.mkdir(parents=True, exist_ok=True)
    for member, source in files.items():
        target = tree / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
        elif target.resolve() != source:
            raise ValueError('Interrupted staging link changed')
    names = scratch / 'files.null'
    names.write_bytes(b''.join(member.encode() + b'\0' for member in sorted(inventory)))
    staged = scratch / 'online_feedback.tar.zst'
    subprocess.run(['tar', '--dereference', '-I', 'zstd -T2 -3 --long=28',
                    '-cf', str(staged), '-C', str(tree), '--null', '-T', str(names)], check=True)
    verify(staged, inventory)
    for member, source in files.items():
        if sha(source) != inventory[member]['sha256']:
            raise ValueError('Source evidence changed while archiving')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'online_feedback.tar.zst'
    storage = storage_status(Path('/fact_data/bohanli'), output)
    if not archive.exists():
        for key in ['filesystem_free_bytes', 'quota_remaining_bytes']:
            if storage.get(key) is not None and storage[key] - staged.stat().st_size < MIN_HEADROOM:
                raise RuntimeError('Preservation would breach 8 GiB reserve')
        temporary = output / ('pending_job_' + os.environ['SLURM_JOB_ID'] + '.tar.zst')
        with temporary.open('xb') as dest, staged.open('rb') as src:
            shutil.copyfileobj(src, dest, length=4 * 1024**2)
            dest.flush()
            os.fsync(dest.fileno())
        if sha(temporary) != sha(staged):
            raise ValueError('Durable archive copy differs')
        temporary.rename(archive)
    verify(archive, inventory)
    atomic_json(output / 'inventory.json', inventory)
    atomic_json(output / 'path_aliases.json', aliases)
    atomic_json(output / 'feedback_audits.json', audits)
    record = dict(status='used_online_feedback_durably_preserved_and_fully_decoded',
        plan_sha256=sha(plan_path), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        file_count=len(inventory), source_bytes=sum(r['bytes'] for r in inventory.values()),
        used_observations=len(audits), successful_observations=sum(r['status'] == 'success' for r in audits),
        studies=[s['id'] for s in plan['replays']], source_files_unchanged=True,
        inventory_sha256=sha(output / 'inventory.json'), audits_sha256=sha(output / 'feedback_audits.json'),
        aliases_sha256=sha(output / 'path_aliases.json'), evaluation_feedback_in_training=False,
        new_training_or_EDA=False, EDA_or_training_resume_archive=False,
        limitations='Input DEFs, actions, GRT reports and replay are retained; OpenDB/SPEF, complete intermediate archives and optimizer checkpoints remain at original locations.',
        restoration='Decode into an empty directory; files/ mirrors resolved absolute paths. Verify inventory hashes before any deliberate restore; never overwrite live experiments.',
        storage_before=storage, slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output / 'manifest.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for native observation auditing and compression')
    preserve(args.plan.resolve())
