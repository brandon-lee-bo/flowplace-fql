#!/usr/bin/env python3
"""Preserve superseded infrastructure-interrupted Table 2 artifacts without declaring success.

create/verify never move or remove source artifacts. commit is a separate
storage operation for hash-bound interrupted attempts; it retains the durable
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
CONTROLS = ('manifest.json',)
RETRY_BINDING = None
RETRY_BINDING_SHA256 = None


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


def interrupted_run(run):
    """Preserve a hash-bound, superseded infrastructure attempt; never mark it successful."""
    run = direct_child(run, RUN_ROOT)
    require(RETRY_BINDING is not None and RETRY_BINDING_SHA256 is not None,
            'Interrupted attempts require an explicit frozen retry binding')
    require(sha(RETRY_BINDING) == RETRY_BINDING_SHA256, 'Retry binding changed')
    binding = json.loads(RETRY_BINDING.read_text())
    require(binding['old_run_id'] == run.name, 'Binding refers to a different interrupted run')
    old_path, new_path = Path(binding['old_tasks']), Path(binding['new_tasks'])
    require(sha(old_path) == binding['old_tasks_sha256']
            and sha(new_path) == binding['new_tasks_sha256'], 'Frozen task files changed')
    old = json.loads(old_path.read_text().splitlines()[binding['old_task_index']])
    new = json.loads(new_path.read_text().splitlines()[binding['new_task_index']])
    require(old['run_id'] == run.name and new['run_id'] == binding['new_run_id']
            and old['run_id'] != new['run_id'], 'Invalid source/replacement run IDs')
    require({k:v for k,v in old.items() if k != 'run_id'} ==
            {k:v for k,v in new.items() if k != 'run_id'}, 'Replacement changed experimental settings')
    manifest = json.loads((run / 'manifest.json').read_text())
    require(manifest.get('run_id') == run.name and manifest.get('task') == old
            and manifest.get('task_manifest_sha256') == binding['old_tasks_sha256']
            and manifest.get('task_index') == binding['old_task_index'], 'Source run differs from frozen task')
    require(manifest.get('status') != 'completed' and manifest.get('success') is not True,
            'This helper is only for interrupted attempts, not successful runs')
    require(not any((run / name).exists() for name in
                    ('metrics.json', 'author_metrics.json', 'fullflow_audit.json'))
            and not any((run / 'orfs_work/logs').glob('*/*/*/6_report.json')),
            'Final metric evidence exists; do not reclassify or archive through interrupted-attempt policy')
    job = str(manifest['slurm_array_job_id']) + '_' + str(manifest['slurm_array_task_id'])
    require(job == binding['old_job'] and re.fullmatch(r'[0-9]+_[0-9]+', job), 'Source scheduler ID differs')
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',job,
                                   '-o','JobID%64,State%30,ExitCode'],text=True,timeout=20)
    exact = [line.strip().split('|') for line in raw.splitlines()
             if line.strip() and line.strip().split('|')[0] == job]
    require(len(exact) == 1 and exact[0][1] == 'TIMEOUT'
            and binding.get('reason_kind') == 'runtime_capacity_no_final_metrics'
            and 0 < binding.get('old_elapsed_seconds', 0) < binding.get('new_walltime_seconds', 0),
            'Source must be an explicitly bound terminal no-metric capacity timeout; all other failures are excluded')
    return {'classification':'superseded_no_metric_time_limit_not_a_success',
            'retry_binding':str(RETRY_BINDING.resolve()),'retry_binding_sha256':RETRY_BINDING_SHA256,
            'source_scheduler':{'job_id':job,'state':exact[0][1],'exit_code':exact[0][2]},
            'source_tasks_sha256':binding['old_tasks_sha256'],
            'replacement_tasks_sha256':binding['new_tasks_sha256'],
            'source_task':old,'replacement_task':new}



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
    eligibility = interrupted_run(run)
    require(destination.is_relative_to(ARCHIVE_ROOT) and destination != ARCHIVE_ROOT,
            'Archive must be under durable archive root')
    require(destination.resolve() == destination, 'Archive destination must not traverse links')
    destination.mkdir(parents=True, exist_ok=False)
    before = inventory(run)
    total = sum(row.get('size', 0) for row in before.values())
    storage = storage_status(Path('/fact_data/bohanli'), ARCHIVE_ROOT)
    # Keep the same 8 GiB durable reserve; compression is temporary scratch IO.
    # Compress in scratch first. Durable admission uses the verified compressed
    # size rather than assuming an uncompressed duplicate in /fact_data.
    reserve = MIN_HEADROOM + 64 * 1024**2
    for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
        require(storage.get(key) is None or storage[key] >= reserve, 'Insufficient durable headroom: ' + key)
    scratch_storage = storage_status(Path('/scratch/bohanli'), CACHE_ROOT)
    for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
        require(scratch_storage.get(key) is None or scratch_storage[key] >= total + MIN_HEADROOM,
                'Insufficient scratch compression headroom: ' + key)
    staging = CACHE_ROOT / '.archive_staging' / os.environ['SLURM_JOB_ID']
    require(staging.resolve() == staging, 'Staging path must not traverse links')
    staging.mkdir(parents=True, exist_ok=False)
    controls = {name: sha(run / name) for name in CONTROLS}
    save(destination / 'source_inventory.json', before)
    for name in ('archive_table2_runtime_interrupted.py', 'archive_table2_runtime_interrupted.slurm'):
        shutil.copyfile(Path(__file__).with_name(name), destination / name)
    started = time.time()
    pending = staging / 'artifacts.tar.zst.partial'
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
    compressed_sha256 = sha(pending)
    compressed_bytes = pending.stat().st_size
    durable_storage = storage_status(Path('/fact_data/bohanli'), ARCHIVE_ROOT)
    for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
        require(durable_storage.get(key) is None or durable_storage[key] >= compressed_bytes + MIN_HEADROOM,
                'Insufficient durable headroom for verified compressed archive: ' + key)
    durable_pending = destination / 'artifacts.tar.zst.partial'
    with pending.open('rb') as source, durable_pending.open('xb') as out:
        shutil.copyfileobj(source, out, length=1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    require(sha(durable_pending) == compressed_sha256, 'Durable copy hash differs from verified archive')
    require(before == inventory(run), 'Source changed while copying durable archive')
    require(controls == {name: sha(run / name) for name in CONTROLS}, 'Run controls changed')
    artifact = destination / 'artifacts.tar.zst'
    durable_pending.rename(artifact)
    sync_directory(destination)
    pending.unlink()  # The fully verified, flushed durable copy is retained.
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
              'archive_policy': 'superseded_infrastructure_attempt_scratch_durable_v1',
              'source_eligibility': eligibility, 'source_completion_not_implied': True,
              'base_helper_sha256': '7fe87e6e021d8da8f4ad680515640d0af716f731d7fce18650dbf4a63f50fcbf',
              'scratch_staging': str(staging), 'scratch_storage_before': scratch_storage,
              'durable_storage_before_copy': durable_storage,
              'helper_sha256': sha(destination / 'archive_table2_runtime_interrupted.py'),
              'wrapper_sha256': sha(destination / 'archive_table2_runtime_interrupted.slurm')}
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
    eligibility = interrupted_run(run)
    require(record.get('source_eligibility') == eligibility, 'Interrupted source eligibility changed')
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
    parser.add_argument('--retry-binding', type=Path)
    parser.add_argument('--retry-binding-sha256')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--archive-dir', required=True, type=Path)
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--confirm-remove-bulk-backup', action='store_true')
    args = parser.parse_args()
    global RETRY_BINDING, RETRY_BINDING_SHA256
    RETRY_BINDING, RETRY_BINDING_SHA256 = args.retry_binding, args.retry_binding_sha256
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
