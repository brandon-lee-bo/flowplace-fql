#!/usr/bin/env python3
"""Archive completed Table 2 results/objects without changing physical design.

create/verify never move or remove source artifacts. commit is a separate,
explicit operation for newly authorized formal runs; it retains the durable
archive, verifies a scratch copy, then replaces only results/objects with links.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import time

from result_storage import MIN_HEADROOM, storage_status

RUN_ROOT = Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')
ARCHIVE_ROOT = Path('/fact_data/bohanli/flowplace_fql_backend/table2_archives')
CACHE_ROOT = Path('/scratch/bohanli/flowplace_table2/artifact_cache')
TREES = ('orfs_work/results', 'orfs_work/objects')
CONTROLS = ('manifest.json', 'metrics.json', 'fullflow_audit.json')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as source:
        return stream_sha(source)


def stream_sha(source):
    value = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def direct_child(path, root):
    path = path.absolute()
    require(path.parent == root and path.resolve() == path,
            'Expected a direct, real child of ' + str(root))
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', path.name), 'Unsafe name')
    return path


def completed_run(run):
    run = direct_child(run, RUN_ROOT)
    require(run.is_dir(), 'Run is missing')
    manifest = json.loads((run / 'manifest.json').read_text())
    require(manifest.get('run_id') == run.name and manifest.get('status') == 'completed'
            and manifest.get('success') is True and manifest.get('evaluation_complete') is True
            and manifest.get('verification_complete') is True and manifest.get('time_end'),
            'Only completed, verified runs are eligible')
    audit = json.loads((run / 'fullflow_audit.json').read_text())
    require(audit.get('all_checks_pass') is True, 'Fullflow audit did not pass')
    array_job, array_task = manifest.get('slurm_array_job_id'), manifest.get('slurm_array_task_id')
    job = (str(array_job) + '_' + str(array_task) if array_job is not None and array_task is not None
           else str(manifest.get('slurm_job_id', '')))
    require(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', job), 'Missing source scheduler ID')
    # aftercorr can release this job slightly before accounting catches up.
    deadline = time.monotonic() + 30
    while True:
        probe = subprocess.run(['sacct', '-n', '-P', '-j', job, '--format=JobID%64,State%30,ExitCode'],
                               capture_output=True, text=True, timeout=10)
        rows = [line.strip().split('|') for line in probe.stdout.splitlines() if line.strip()]
        exact = [row for row in rows if row[0] == job]
        if probe.returncode == 0 and len(exact) == 1 and exact[0][1:3] == ['COMPLETED', '0:0']:
            break
        terminal_failure = any(row[1].split()[0] in
                               {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
                               for row in exact)
        require(not terminal_failure and time.monotonic() < deadline,
                'Exact source task is not COMPLETED/0:0: ' + job + ': ' + probe.stdout + probe.stderr)
        time.sleep(2)
    return manifest


def inventory(run):
    """Read bounded trees; reject links, devices, sockets, and external hardlinks."""
    result = {}
    for relative in TREES:
        base = run / relative
        require(base.is_dir() and base.resolve() == base, 'Missing real tree or symlink ancestor: ' + str(base))
        paths = [base]
        for parent, dirs, files in os.walk(base, followlinks=False):
            paths.extend(Path(parent) / name for name in sorted(dirs + files))
        for path in sorted(paths):
            info = path.lstat()
            name = path.relative_to(run).as_posix()
            require(not path.is_symlink() and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)),
                    'Unsupported link or special file: ' + name)
            row = {'type': 'dir' if path.is_dir() else 'file', 'mode': stat.S_IMODE(info.st_mode)}
            if row['type'] == 'file':
                require(info.st_nlink == 1, 'Hardlinked files need a separate policy: ' + name)
                row.update(size=info.st_size, mtime_ns=info.st_mtime_ns,
                           allocated_bytes=info.st_blocks * 512, sha256=sha(path))
                after = path.stat()
                require((info.st_ino, info.st_size, info.st_mtime_ns) ==
                        (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed while hashing: ' + name)
            result[name] = row
    return result


def content_identity(entries):
    # Different filesystems may allocate different block counts. Copy/restore
    # identity is names, types, permissions, file lengths, and complete contents.
    return {name: {k: v for k, v in row.items() if k not in ('allocated_bytes', 'mtime_ns')}
            for name, row in entries.items()}


def verify_tar(archive, expected):
    for name in expected:
        parts = Path(name).parts
        require(not Path(name).is_absolute() and '..' not in parts and '.' not in parts
                and len(parts) >= 2 and parts[0] == 'orfs_work' and parts[1] in ('results', 'objects')
                and Path(name).as_posix() == name, 'Unsafe inventory member path: ' + name)
    seen = set()
    with subprocess.Popen(['zstd', '-dc', '--', str(archive)], stdout=subprocess.PIPE) as decoder:
        try:
            with tarfile.open(fileobj=decoder.stdout, mode='r|') as reader:
                for member in reader:
                    name = member.name.rstrip('/')
                    require(name in expected and name not in seen, 'Unexpected/duplicate archive member: ' + name)
                    seen.add(name)
                    row = expected[name]
                    require(member.mode == row['mode'], 'Permission mismatch: ' + name)
                    if row['type'] == 'dir':
                        require(member.isdir(), 'Expected directory: ' + name)
                    else:
                        require(member.isfile() and member.size == row['size'], 'File type/size mismatch: ' + name)
                        require(stream_sha(reader.extractfile(member)) == row['sha256'], 'File hash mismatch: ' + name)
            # Read the entire zstd frame, including its checksum, even if tar
            # stopped at end-of-archive padding.
            while decoder.stdout.read(1024 * 1024):
                pass
            require(decoder.wait() == 0, 'zstd decompression/checksum failed')
        except BaseException:
            decoder.kill()
            decoder.wait()
            raise
    require(seen == set(expected), 'Archive omitted source files/directories')


def create(run, destination, threads):
    completed_run(run)
    require(destination.is_relative_to(ARCHIVE_ROOT) and destination != ARCHIVE_ROOT,
            'Archive must be under durable archive root')
    require(destination.resolve() == destination, 'Archive destination must not traverse links')
    destination.mkdir(parents=True, exist_ok=False)
    before = inventory(run)
    total = sum(row.get('size', 0) for row in before.values())
    storage = storage_status(Path('/fact_data/bohanli'), ARCHIVE_ROOT)
    # Compression can expand incompressible data. Retain 8 GiB even if that
    # happens; run this serially with other bulk-storage producers.
    reserve = total + max(64 * 1024**2, total // 100) + MIN_HEADROOM
    for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
        require(storage.get(key) is None or storage[key] >= reserve, 'Insufficient staging headroom: ' + key)
    controls = {name: sha(run / name) for name in CONTROLS}
    save(destination / 'source_inventory.json', before)
    for name in ('archive_table2_artifacts.py', 'archive_table2_artifacts.slurm'):
        shutil.copyfile(Path(__file__).with_name(name), destination / name)
    started = time.time()
    pending = destination / 'artifacts.tar.zst.partial'
    with pending.open('xb') as out, (destination / 'compression.log').open('w') as log:
        with subprocess.Popen(['zstd', '-3', '-T' + str(threads), '-c'],
                              stdin=subprocess.PIPE, stdout=out, stderr=log) as compressor:
            try:
                with tarfile.open(fileobj=compressor.stdin, mode='w|', format=tarfile.PAX_FORMAT) as writer:
                    for name in sorted(before):
                        writer.add(run / name, arcname=name, recursive=False)
                compressor.stdin.close()
                require(compressor.wait() == 0, 'Compression failed; partial archive retained')
            except BaseException:
                compressor.kill()
                compressor.wait()
                raise
        out.flush()
        os.fsync(out.fileno())
    verify_tar(pending, before)
    require(before == inventory(run), 'Source files changed during archive creation')
    require(controls == {name: sha(run / name) for name in CONTROLS}, 'Run controls changed')
    artifact = destination / 'artifacts.tar.zst'
    pending.rename(artifact)
    sync_directory(destination)
    record = {'schema': 1, 'status': 'verified', 'run': str(run), 'run_id': run.name,
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'created_unix': time.time(),
              'elapsed_seconds': time.time() - started, 'source_trees': list(TREES),
              'source_control_sha256': controls, 'source_inventory_sha256': sha(destination / 'source_inventory.json'),
              'archive': artifact.name, 'archive_sha256': sha(artifact), 'archive_bytes': artifact.stat().st_size,
              'source_bytes': total, 'source_allocated_bytes': sum(row.get('allocated_bytes', 0) for row in before.values()),
              'archive_to_source_ratio': artifact.stat().st_size / total if total else None,
              'potential_bulk_saving_bytes': total - artifact.stat().st_size,
              'actual_source_bytes_removed': 0, 'source_directories_unchanged': True,
              'compression': ['zstd', '-3', '-T' + str(threads)], 'storage_before': storage,
              'helper_sha256': sha(destination / 'archive_table2_artifacts.py'),
              'wrapper_sha256': sha(destination / 'archive_table2_artifacts.slurm')}
    save(destination / 'archive_manifest.json', record)
    print(json.dumps(record, indent=2), flush=True)


def load_verified(destination):
    require(destination.resolve() == destination and destination.is_relative_to(ARCHIVE_ROOT),
            'Expected a real durable archive directory')
    record = json.loads((destination / 'archive_manifest.json').read_text())
    require(record['status'] == 'verified' and record['source_trees'] == list(TREES), 'Archive is not verified')
    require(record['archive'] == 'artifacts.tar.zst', 'Unexpected archive filename')
    require(all(not (destination / name).is_symlink() for name in
                ('archive_manifest.json', 'source_inventory.json', record['archive'])), 'Archive file is a symlink')
    require(sha(destination / 'source_inventory.json') == record['source_inventory_sha256'], 'Inventory hash changed')
    require(sha(destination / record['archive']) == record['archive_sha256'], 'Archive hash changed')
    entries = json.loads((destination / 'source_inventory.json').read_text())
    verify_tar(destination / record['archive'], entries)
    return record, entries


def commit(destination, scratch, confirmation):
    require(confirmation, 'commit requires --confirm-remove-bulk-backup; create/verify are non-destructive')
    record, entries = load_verified(destination)
    run = Path(record['run'])
    completed_run(run)
    require(record['source_control_sha256'] == {name: sha(run / name) for name in CONTROLS}, 'Run controls changed')
    require(content_identity(inventory(run)) == content_identity(entries), 'Source differs from verified archive')
    require(scratch.is_relative_to(CACHE_ROOT) and scratch != CACHE_ROOT and scratch.resolve() == scratch,
            'Scratch destination must be a new real path under ' + str(CACHE_ROOT))
    state_path = destination / 'commit_state.json'
    require(not state_path.exists(), 'Prior commit attempt exists; inspect its recovery state')
    scratch_storage = storage_status(Path('/scratch/bohanli'), CACHE_ROOT)
    for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
        require(scratch_storage.get(key) is None or scratch_storage[key] > record['source_bytes'] + MIN_HEADROOM,
                'Insufficient scratch staging headroom: ' + key)
    scratch.mkdir(parents=True, exist_ok=False)
    state = {'status': 'copying', 'run': str(run), 'scratch': str(scratch),
             'archive_manifest_sha256': sha(destination / 'archive_manifest.json'),
             'job_id': os.environ['SLURM_JOB_ID'], 'backup_paths': {}, 'scratch_storage_before': scratch_storage,
             'commit_helper_sha256': sha(Path(__file__)),
             'note': 'Durable archive is authoritative; scratch links are only a convenience cache.'}
    save(state_path, state)
    for relative in TREES:
        target = scratch / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run / relative, target, symlinks=False)
    require(content_identity(inventory(scratch)) == content_identity(entries), 'Scratch copy failed full hash verification')
    # Flush copied files before publishing links. The durable archive has already
    # been flushed and fully decoded/hashed; scratch is not the sole retained copy.
    for name, row in entries.items():
        if row['type'] == 'file':
            with (scratch / name).open('rb') as source:
                os.fsync(source.fileno())
    state['status'] = 'copy_verified'
    save(state_path, state)
    require(content_identity(inventory(run)) == content_identity(entries), 'Original changed during scratch copy')
    swapped = []
    try:
        for relative in TREES:
            original = run / relative
            backup = original.with_name(original.name + '.archive-backup-' + os.environ['SLURM_JOB_ID'])
            require(not os.path.lexists(backup), 'Backup already exists')
            state['backup_paths'][relative] = str(backup)
            save(state_path, state)
            original.rename(backup)
            swapped.append((original, backup))
            original.symlink_to(scratch / relative, target_is_directory=True)
            sync_directory(original.parent)
        require(content_identity(inventory(scratch)) == content_identity(entries), 'Published scratch cache changed')
        for original, _ in swapped:
            require(original.is_symlink() and original.resolve() == scratch / original.relative_to(run),
                    'Published symlink target mismatch')
        require(record['source_control_sha256'] == {name: sha(run / name) for name in CONTROLS}, 'Run controls changed')
    except BaseException:
        # No bulk backup is deleted before the entire link publication verifies.
        for original, backup in reversed(swapped):
            if original.is_symlink():
                original.unlink()
            if not os.path.lexists(original) and backup.exists():
                backup.rename(original)
        state['status'] = 'rolled_back_or_needs_inspection'
        save(state_path, state)
        raise
    state['status'] = 'links_verified_backups_pending_removal'
    save(state_path, state)
    # This is the only deletion phase, gated by explicit commit confirmation.
    # Interruption here leaves a usable link, durable archive, and any residual
    # backup recorded above. Never repeat commit blindly after interruption.
    for original, backup in swapped:
        shutil.rmtree(backup)
        sync_directory(original.parent)
    state.update(status='committed', completed_unix=time.time(), bulk_source_bytes_removed=record['source_bytes'])
    save(state_path, state)
    print(json.dumps(state, indent=2), flush=True)


def acquire_lock(lock):
    deadline = time.monotonic() + 120
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            require(time.monotonic() < deadline, 'Archive lock remained busy for 120 seconds')
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'verify', 'commit'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--archive-dir', required=True, type=Path)
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--confirm-remove-bulk-backup', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'Run archive I/O under Slurm')
    require(1 <= args.threads <= int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), 'Threads exceed allocation')
    destination = args.archive_dir.absolute()
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    # Serialize archive producers and commits, including headroom checks.
    with (ARCHIVE_ROOT / '.archive.lock').open('a') as lock:
        acquire_lock(lock)
        if args.action == 'create':
            require(args.run is not None, 'create needs --run')
            create(args.run.absolute(), destination, args.threads)
        elif args.action == 'verify':
            record, _ = load_verified(destination)
            print(json.dumps({'verified': True, 'archive_sha256': record['archive_sha256']}), flush=True)
        else:
            require(args.scratch_dir is not None, 'commit needs --scratch-dir')
            commit(destination, args.scratch_dir.absolute(), args.confirm_remove_bulk_backup)


if __name__ == '__main__':
    main()
