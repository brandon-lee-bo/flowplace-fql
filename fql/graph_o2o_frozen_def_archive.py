"""Losslessly compact explicitly pinned, completed DREAMPlace DEF snapshots.

This only changes storage: every original byte has a verified durable archive
and a verified scratch cache before a DEF is atomically replaced by a link.
The original manifests, geometry audits and physical results stay untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_checkpoint_archive import verify

BASE = Path('/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2')
ALLOWED = {
    'dmp_public41_h20_abacusfix_bp_v1',
    'dmp_public41_h20_bp_frozen_v1',
    'dmp_public41_h20_abacusfix_ariane136_v1',
    'dmp_public41_h20_abacusfix_ariane133_v1',
}
REMAINING = {
    'dmp_public41_h20_abacusfix_swerv_wrapper_v1',
    'dmp_public41_h20_swerv_wrapper_frozen_v1',
    'dmp_public41_h20_ariane136_frozen_v1',
    'dmp_public41_h20_abacusfix_bp_be_v1',
    'dmp_public41_h20_bp_be_frozen_v1',
}


def check_closed(plan):
    allowed = REMAINING if plan.get('cohort') == 'remaining_five' else ALLOWED
    if {row['name'] for row in plan['runs']} != allowed:
        raise ValueError('Only explicitly pinned frozen DREAMPlace runs are eligible')
    # squeue -j OLD_ID exits nonzero after Slurm forgets a completed job.
    # Query the live user queue, then require the pinned terminal accounting.
    live = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%A|%T'],
                          capture_output=True, text=True, check=True)
    live_ids = {line.split('|')[0] for line in live.stdout.splitlines() if '|' in line}
    bindings = {}
    for row in plan['runs']:
        root = BASE / row['name']
        path = root / 'manifest.json'
        record = json.loads(path.read_text())
        if (sha(path) != row['manifest_sha256'] or not record.get('success')
                or not record.get('time_end') or str(record['slurm_job_id']) != row['job_id']):
            raise ValueError('A selected DREAMPlace run is not the pinned closed run')
        if row['job_id'] in live_ids:
            raise ValueError('Do not compact a live source job')
        accounting = subprocess.run(['sacct', '-j', row['job_id'], '-X', '-n', '-P',
                                     '-o', 'JobID,State,ExitCode'],
                                    capture_output=True, text=True, check=True)
        actual = [line.split('|')[1:3] for line in accounting.stdout.splitlines()
                  if line.split('|')[0] == row['job_id']]
        # One old multi-design job failed after closing this successful design.
        # Archival needs a terminal process and the closed per-design manifest;
        # it neither upgrades the batch status nor certifies downstream PPA.
        if (row['job_state'] not in ['COMPLETED', 'FAILED'] or
                actual != [[row['job_state'], row['job_exit_code']]]):
            raise ValueError('Source job differs from the pinned terminal accounting')
        bindings[str(path)] = row['manifest_sha256']
    return bindings


def run(plan_path):
    folder = plan_path.parent
    plan = json.loads(plan_path.read_text())
    bindings = check_closed(plan)
    cache = Path('/scratch/bohanli/flowplace_graph/frozen_dmp_def_v1')
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    archive = folder / 'def_snapshots.tar.zst'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if sha(plan_path) != manifest['plan_sha256'] or sha(archive) != manifest['archive_sha256']:
            raise ValueError('Existing archive or immutable plan changed')
        expected = manifest['files']
        verify(archive, expected)
    else:
        paths = []
        # Only known large DEF snapshots, not a broad filesystem scan.
        for row in plan['runs']:
            root = BASE / row['name']
            candidates = [*root.glob('*.def'), *root.glob('macro_legalizer_*/after.def')]
            paths += [path for path in candidates if path.stat().st_size > 1024**2]
        paths.sort()
        if len(paths) != plan['expected_def_files'] or any(p.is_symlink() for p in paths):
            raise ValueError('Unexpected frozen DEF inventory; leave sources untouched')
        expected = {str(path.relative_to(BASE)): dict(bytes=path.stat().st_size, sha256=sha(path))
                    for path in paths}
        names = cache / 'archive_files.null'
        names.write_bytes(b''.join(name.encode() + b'\0' for name in expected))
        staged = cache / ('archive_job_' + os.environ['SLURM_JOB_ID'] + '.tar.zst')
        subprocess.run(['tar', '-I', 'zstd -T2 -3 --long=28', '-cf', str(staged),
                        '-C', str(BASE), '--null', '-T', str(names)], check=True)
        verify(staged, expected)
        if check_closed(plan) != bindings:
            raise ValueError('Frozen source bindings changed during compression')
        for name, item in expected.items():
            if sha(BASE / name) != item['sha256']:
                raise ValueError('Source DEF changed during compression')
        quota = Path('/fact_data/bohanli')
        remaining = int(os.getxattr(quota, 'ceph.quota.max_bytes')) - int(os.getxattr(quota, 'ceph.dir.rbytes'))
        if remaining < staged.stat().st_size + 8 * 1024**3:
            raise RuntimeError('Durable archive copy would breach the 8 GiB reserve')
        temporary = archive.with_suffix('.pending_' + os.environ['SLURM_JOB_ID'])
        shutil.copyfile(staged, temporary)
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        if sha(temporary) != sha(staged):
            raise ValueError('Durable archive does not match the verified staged bytes')
        temporary.replace(archive)
        manifest = dict(status='verified_durable_before_source_links', plan_sha256=sha(plan_path),
                        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                        files=expected, source_bytes=sum(item['bytes'] for item in expected.values()),
                        closed_source_bindings=bindings, decoded_every_file_sha_verified=True,
                        archive_source_sha256=sha(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'],
                        restoration='zstd -dc --long=28 def_snapshots.tar.zst | tar -xf - -C NEW_EMPTY_DIRECTORY; verify all manifest file hashes',
                        no_physical_or_training_result_changed=True)
        atomic_json(manifest_path, manifest)
        staged.unlink()
    for name, item in expected.items():
        source, target = BASE / name, cache / name
        if source.is_symlink() and source.resolve() != target:
            raise ValueError('Unexpected DEF cache link')
        if not source.exists() or sha(source) != item['sha256']:
            raise ValueError('Source/cache missing or changed; restore from the durable archive first')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
            raise ValueError('Scratch DEF does not match the durable original')
        if not source.is_symlink():
            link = source.with_name(source.name + '.verified_link_' + os.environ['SLURM_JOB_ID'])
            link.symlink_to(target)
            link.replace(source)
    if check_closed(plan) != bindings:
        raise ValueError('Closed run metadata changed')
    manifest.update(status='durable_archive_and_verified_scratch_links', finished_unix=time.time(),
                    reclaimed_bytes=manifest['source_bytes'] - manifest['archive_bytes'])
    atomic_json(manifest_path, manifest)
    shutil.copyfile(__file__, folder / ('source_job_' + os.environ['SLURM_JOB_ID'] + '.py'))
    shutil.copyfile(Path(__file__).with_suffix('.slurm'), folder / 'archive_wrapper.slurm')
    print(json.dumps({key: manifest[key] for key in
                      ['status', 'source_bytes', 'archive_bytes', 'reclaimed_bytes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    run(args.plan.resolve())
