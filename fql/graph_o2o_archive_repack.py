"""Shrink closed legacy archives while preserving their exact original bytes.

The durable replacement stores the identical tar stream with a longer window.
Before replacing the old file by a verified cache link, reconstruct its original
compressed bytes and check the original SHA. No outcome or physical manifest is
edited. The restore command also reconstructs the old, hash-bound archive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .graph_o2o_archive import ROOT, atomic_json, sha, verify_archive

PILOT = Path('/fact_data/bohanli/flowplace_fql/graph_o2o_v1')
CACHE = Path('/scratch/bohanli/flowplace_graph/legacy_archive_cache')


def transcode(source, target, long_window):
    decoder = subprocess.Popen(['zstd', '-dc', '--long=28', str(source)], stdout=subprocess.PIPE)
    command = ['zstd', '-q', '-T2', '-3'] + (['--long=28'] if long_window else [])
    try:
        subprocess.run(command + ['-o', str(target)], stdin=decoder.stdout, check=True)
        if decoder.wait():
            raise ValueError('Archive decoder failed')
    finally:
        if decoder.poll() is None:
            decoder.kill()
        decoder.wait()


def make_plan(output):
    probe = output/'probe/report.json'
    check = json.loads(probe.read_text())
    if not check['original_archive_reproduction']['original_bytes_reproduced']:
        raise ValueError('A successful exact-byte regeneration probe is required')
    paths = []
    for number in range(3):
        for line in (PILOT/f'seed_0/round_{number}/requests/tasks.jsonl').read_text().splitlines():
            paths.append(Path(json.loads(line)['sample_dir'])/'outcome.json')
    route_plan = json.loads((PILOT/'full_route_plan.json').read_text())
    paths.append(PILOT/'full_route/best_train/outcome.json')
    paths += [PILOT/'full_route'/f"{group['arm']}_r{group['round']}_n{i}"/'outcome.json'
              for group in route_plan['groups'] for i in group['indices']]
    items, seen = [], set()
    for path in paths:
        if not path.exists():
            continue
        outcome = json.loads(path.read_text())
        if outcome['status'] not in ['success', 'completed']:
            continue
        run = Path(outcome['physical_run']).resolve()
        if run.parent != ROOT or not run.name.startswith('go2o_'):
            raise ValueError('Physical run is outside this experiment')
        record_path = run/'grt_artifact_archive.json'
        record = json.loads(record_path.read_text())
        if '--long=28' in record.get('compression', '') or run in seen:
            continue
        archive = run/'grt_intermediates.tar.zst'
        manifest_path = run/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (not manifest['success'] or not manifest.get('time_end') or archive.is_symlink() or
                sha(manifest_path) != outcome['physical_manifest_sha256'] or
                sha(archive) != record['archive_sha256'] or
                record['archive_sha256'] != outcome['archive']['archive_sha256']):
            raise ValueError('Legacy source is not a closed, unchanged regular archive')
        seen.add(run)
        items.append(dict(run=str(run), original_archive_sha256=record['archive_sha256'],
            original_bytes=archive.stat().st_size, archive_record_sha256=sha(record_path),
            physical_manifest_sha256=sha(manifest_path),
            inventory_sha256=sha(run/'grt_intermediates_inventory.json'),
            outcome_path=str(path), outcome_sha256=sha(path),
            original_job_id=str(manifest['slurm_job_id'])))
    jobs = sorted({row['original_job_id'] for row in items})
    accounting = subprocess.check_output(['sacct', '-j', ','.join(jobs), '-n', '-X', '-P',
        '--format=JobIDRaw,State,ExitCode'], text=True)
    states = {row.split('|')[0]:row.split('|')[1:3] for row in accounting.splitlines()}
    for row in items:
        if states.get(row['original_job_id']) != ['COMPLETED', '0:0']:
            raise ValueError('A source worker is not confirmed COMPLETED0:0: '+row['original_job_id'])
    plan = dict(version=1, scope='Closed main-pilot legacy archives only', items=items,
        probe_sha256=sha(probe), encoder=check['original_archive_reproduction'],
        source_job_accounting=states, total_source_bytes=sum(row['original_bytes'] for row in items),
        original_outcomes_and_manifests_unchanged=True, created_unix=time.time())
    atomic_json(output/'plan.json', plan)
    return plan


def check_tool(plan):
    if sha(shutil.which('zstd')) != plan['encoder']['encoder_binary_sha256']:
        raise ValueError('Original zstd encoder differs from the verified reconstruction tool')


def durable_copy(source, target):
    pending = target.with_name(target.name+'.pending_'+os.environ['SLURM_JOB_ID'])
    shutil.copyfile(source, pending)
    with pending.open('rb') as handle:
        os.fsync(handle.fileno())
    if sha(pending) != sha(source):
        raise ValueError('Durable archive copy differs')
    pending.replace(target)


def repack(row, plan_digest):
    run = Path(row['run'])
    original = run/'grt_intermediates.tar.zst'
    compact = run/'grt_intermediates.long28.tar.zst'
    metadata = run/'legacy_archive_repack.json'
    if metadata.exists():
        record = json.loads(metadata.read_text())
        if record['status'] == 'durable_compact_archive_and_exact_original_cache':
            if sha(original) != row['original_archive_sha256'] or sha(compact) != record['compact_sha256']:
                raise ValueError('Previously repacked archive changed')
            return record
    for path, key in [(run/'manifest.json','physical_manifest_sha256'),
                      (run/'grt_artifact_archive.json','archive_record_sha256'),
                      (run/'grt_intermediates_inventory.json','inventory_sha256'),
                      (Path(row['outcome_path']),'outcome_sha256'),
                      (original,'original_archive_sha256')]:
        if sha(path) != row[key]:
            raise ValueError('Bound legacy source changed: '+str(path))
    folder = CACHE/run.name/('job_'+os.environ['SLURM_JOB_ID'])
    folder.mkdir(parents=True, exist_ok=True)
    staged = folder/'compact.tar.zst'
    regenerated = folder/'original.tar.zst'
    for candidate in [staged, regenerated]:
        if candidate.exists():
            candidate.unlink()
    transcode(original, staged, True)
    verify_archive(staged, json.loads((run/'grt_intermediates_inventory.json').read_text()))
    transcode(staged, regenerated, False)
    if sha(regenerated) != row['original_archive_sha256']:
        raise ValueError('Regenerated original bytes differ; no source replacement permitted')
    if staged.stat().st_size >= row['original_bytes']:
        raise ValueError('No storage saving; original left unchanged')
    durable_copy(staged, compact)
    record = dict(status='verified_before_original_cache_swap', plan_sha256=plan_digest,
        **row, compact_path=str(compact), compact_sha256=sha(compact),
        compact_bytes=compact.stat().st_size, original_cache_path=str(regenerated),
        reclaimed_bytes=row['original_bytes']-compact.stat().st_size,
        full_decoded_file_hashes_verified=True, original_archive_bytes_reproduced=True,
        restoration='Run this module with --stage restore --record THIS_RECORD under Slurm; it restores the exact original archive SHA, not merely equivalent files.',
        slurm_job_id=os.environ['SLURM_JOB_ID'], created_unix=time.time())
    atomic_json(metadata, record)
    pending = original.with_name(original.name+'.cache_link_'+os.environ['SLURM_JOB_ID'])
    pending.symlink_to(regenerated)
    pending.replace(original)
    if sha(original) != row['original_archive_sha256']:
        raise ValueError('Published original archive cache differs')
    record['status'] = 'durable_compact_archive_and_exact_original_cache'
    atomic_json(metadata, record)
    staged.unlink()
    return record


def restore(record_path, output):
    row = json.loads(record_path.read_text())
    plan = json.loads((output/'plan.json').read_text())
    check_tool(plan)
    compact = Path(row['compact_path'])
    if sha(compact) != row['compact_sha256']:
        raise ValueError('Durable compact archive changed')
    destination = Path(row['original_cache_path'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != row['original_archive_sha256']:
            raise ValueError('Existing original archive cache differs')
        return dict(status='already_present_and_verified',original_sha256=sha(destination),
                    destination=str(destination),record_sha256=sha(record_path))
    staged = destination.with_name('restore_job_'+os.environ['SLURM_JOB_ID']+'.tar.zst')
    transcode(compact, staged, False)
    if sha(staged) != row['original_archive_sha256']:
        raise ValueError('Restoration did not reproduce original archive bytes')
    staged.replace(destination)
    return dict(status='restored_original_archive_bytes',original_sha256=sha(destination),
        destination=str(destination),record_sha256=sha(record_path),
        compact_sha256=sha(compact),slurm_job_id=os.environ['SLURM_JOB_ID'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['plan','commit','restore'], required=True)
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for archive verification/recompression')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, output/('source_job_'+os.environ['SLURM_JOB_ID']+'.py'))
    if args.stage == 'plan':
        if (output/'plan.json').exists():
            raise ValueError('Do not overwrite a frozen storage plan')
        plan = make_plan(output)
        print(json.dumps(dict(status='planned',archives=len(plan['items']),bytes=plan['total_source_bytes'])),flush=True)
        return
    if args.stage == 'restore':
        if args.record is None:
            raise ValueError('A specific legacy_archive_repack.json is required')
        result=restore(args.record, output)
        evidence=output/'restores';evidence.mkdir(exist_ok=True)
        atomic_json(evidence/('job_'+os.environ['SLURM_JOB_ID']+'.json'),result)
        print(json.dumps(result),flush=True)
        return
    plan_path = output/'plan.json'
    plan = json.loads(plan_path.read_text())
    check_tool(plan)
    records = []
    for row in plan['items']:
        record = repack(row, sha(plan_path))
        records.append(record)
        atomic_json(output/'manifest.json', dict(status='in_progress',plan_sha256=sha(plan_path),
            completed=len(records),requested=len(plan['items']),records=records,
            reclaimed_bytes=sum(row['reclaimed_bytes'] for row in records)))
        print(json.dumps(dict(run=row['run'],reclaimed_bytes=record['reclaimed_bytes'])),flush=True)
    atomic_json(output/'manifest.json', dict(status='completed',plan_sha256=sha(plan_path),
        completed=len(records),requested=len(plan['items']),records=records,
        reclaimed_bytes=sum(row['reclaimed_bytes'] for row in records),
        slurm_job_id=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    main()
