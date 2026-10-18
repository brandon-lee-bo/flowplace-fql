#!/usr/bin/env python3
"""Relocate only this project's re-downloadable container cache to scratch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

SOURCE = Path('/fact_data/bohanli/flowplace_fql_backend/singularity-cache')
DEST = Path('/scratch/bohanli/flowplace_table2/container_cache_relocated_v1')
JOURNAL = Path('/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/plans/table2_background_v1/container_cache_relocation_v1.json')


def inventory(root):
    result = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(parent) / name
            info = path.lstat()
            assert not path.is_symlink(), 'Unexpected cache link'
            assert stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
            record = {'mode': stat.S_IMODE(info.st_mode), 'type': 'dir' if path.is_dir() else 'file'}
            if record['type'] == 'file':
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                record.update(size=info.st_size, mtime_ns=info.st_mtime_ns, sha256=digest.hexdigest())
                assert path.stat().st_mtime_ns == info.st_mtime_ns
            result[path.relative_to(root).as_posix()] = record
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert SOURCE.is_dir() and not SOURCE.is_symlink() and not DEST.exists() and not JOURNAL.exists()
    # The build cache predates this reproduction. Active wrappers use per-job scratch TMPDIR;
    # published SIF, frozen binaries, source trees and every result remain untouched.
    assert SOURCE.stat().st_mtime < time.time() - 12 * 3600
    before = inventory(SOURCE)
    assert sum(r.get('size', 0) for r in before.values()) < 5 * 2**30
    assert shutil.disk_usage(DEST.parent).free > 20 * 2**30
    shutil.copytree(SOURCE, DEST, copy_function=shutil.copy2)
    assert inventory(DEST) == before == inventory(SOURCE), 'Cache copy/source verification failed'
    subprocess.run(['sync', '-f', str(DEST)], check=True)
    backup = SOURCE.with_name('singularity-cache.verified_relocation_backup_v1')
    assert not backup.exists()
    record = {'status': 'verified_before_commit', 'source': str(SOURCE), 'destination': str(DEST),
              'backup': str(backup), 'job': os.environ['SLURM_JOB_ID'], 'inventory': before,
              'scope': 'Re-downloadable container cache only; important SIF, binaries and results untouched.',
              'bytes': sum(r.get('size', 0) for r in before.values())}
    JOURNAL.write_text(json.dumps(record, indent=2) + '\n')
    SOURCE.rename(backup)
    SOURCE.symlink_to(DEST, target_is_directory=True)
    assert inventory(SOURCE) == before == inventory(backup)
    shutil.rmtree(backup)  # Only the verified redundant copy of this re-downloadable cache.
    record['status'] = 'completed'
    JOURNAL.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'inventory'}))


if __name__ == '__main__':
    main()
