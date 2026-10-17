"""Verify and archive this experiment's completed GRT intermediates.

Logs/reports/metrics remain in bulk. Only results and objects are replaced with
scratch cache links, after a complete decoded archive and per-file SHA audit.
This is a storage operation and never changes a physical result or manifest.
Explicit scratch runs retain their archive on shared scratch and record that
this does not provide bulk-storage redundancy.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

ROOT = Path('/fact_data/bohanli/flowplace_fql_backend/grt_results')
SCRATCH_ROOT = Path('/scratch/bohanli/flowplace_graph/physical_results')
CACHE = Path('/scratch/bohanli/flowplace_graph/grt_archive_cache')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def inventory(directory, prefix):
    result = {}
    for parent, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs:
            if (Path(parent)/name).is_symlink():
                raise ValueError('Unexpected intermediate directory symlink')
        for name in files:
            path = Path(parent)/name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Archive accepts regular intermediate files only')
            key = str(Path(prefix)/path.relative_to(directory))
            result[key] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def verify_archive(archive, expected):
    process = subprocess.Popen(['zstd', '-dc', '--long=28', str(archive)], stdout=subprocess.PIPE)
    seen = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
            for member in stream:
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected:
                    raise ValueError('Unexpected archive member: ' + member.name)
                digest = hashlib.sha256()
                handle = stream.extractfile(member)
                for part in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(part)
                if member.name in seen:
                    raise ValueError('Duplicate archive member')
                seen[member.name] = dict(bytes=member.size, sha256=digest.hexdigest())
        if process.wait() or seen != expected:
            raise ValueError('Decoded archive content differs from source inventory')
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def archive_run(path, *, prior_manifest_sha256=None):
    run = Path(path).resolve()
    prior = (prior_manifest_sha256 is not None and
             re.fullmatch(r'fqlphys_1788587_s[012]_(?:bc_continued|fql_physical_alpha_1)_i19[23](?:_preempt_r1)?', run.name)
             and sha(run/'manifest.json') == prior_manifest_sha256)
    if run.parent not in [ROOT, SCRATCH_ROOT] or not (run.name.startswith('go2o_') or prior):
        raise ValueError('Archive scope is this experiment or explicitly hash-bound prior FQL GRT measurements')
    scratch_storage = run.parent == SCRATCH_ROOT
    manifest = json.loads((run/'manifest.json').read_text())
    if not manifest.get('time_end'):
        raise ValueError('Never archive a live/incomplete physical process')
    record_path = run/'grt_artifact_archive.json'
    archive = run/'grt_intermediates.tar.zst'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if sha(archive) != record['archive_sha256'] or sha(run/'manifest.json') != record['physical_manifest_sha256']:
            raise ValueError('Existing archive or physical manifest changed')
        expected = json.loads((run/'grt_intermediates_inventory.json').read_text())
    else:
        names = [str(Path('orfs_work')/name) for name in ['results', 'objects']
                 if (run/'orfs_work'/name).is_dir()]
        if not names:
            return dict(status='no_intermediate_directories', run=str(run))
        expected = {}
        for name in names:
            source = run/name
            if source.is_symlink():
                raise ValueError('Unrecorded intermediate symlink')
            expected.update(inventory(source, name))
        temporary = CACHE/run.name/('archive_job_' + os.environ['SLURM_JOB_ID'])
        temporary.mkdir(parents=True, exist_ok=True)
        staged = temporary/'intermediates.tar.zst'
        subprocess.run(['tar', '--sort=name', '-I', 'zstd -T2 -3 --long=28', '-cf', str(staged),
                        '-C', str(run), *names], check=True)
        verify_archive(staged, expected)
        for name in names:
            actual = inventory(run/name, name)
            subset = {k: v for k, v in expected.items() if k.startswith(name+'/')}
            if actual != subset:
                raise ValueError('Physical intermediates changed during compression')
        quota = Path('/scratch/bohanli' if scratch_storage else '/fact_data/bohanli')
        remaining = int(os.getxattr(quota, 'ceph.quota.max_bytes')) - int(os.getxattr(quota, 'ceph.dir.rbytes'))
        if remaining < staged.stat().st_size + 8 * 1024**3:
            raise RuntimeError('Compressed archive does not fit durable quota plus 8 GiB reserve')
        pending = run/('grt_intermediates.tar.zst.pending_' + os.environ['SLURM_JOB_ID'])
        shutil.copyfile(staged, pending)
        with pending.open('rb') as handle:
            os.fsync(handle.fileno())
        digest = sha(staged)
        if sha(pending) != digest:
            raise ValueError('Durable archive SHA differs from verified staged archive')
        pending.replace(archive)
        atomic_json(run/'grt_intermediates_inventory.json', expected)
        record = dict(status='archive_verified_before_cache_swap', archive_path=str(archive),
                      archive_sha256=digest, archive_bytes=archive.stat().st_size,
                      source_bytes=sum(v['bytes'] for v in expected.values()),
                      file_count=len(expected), decoded_full_content_verified=True,
                      physical_manifest_sha256=sha(run/'manifest.json'),
                      slurm_job_id=os.environ['SLURM_JOB_ID'], created_unix=time.time(), roots=names,
                      compression='zstd -T2 -3 --long=28; lossless, all intermediate files retained',
                      restoration='zstd -dc --long=28 grt_intermediates.tar.zst | tar -xf - -C NEW_EMPTY_DIRECTORY; verify every inventory SHA before use.')
        if scratch_storage:
            record.update(storage_policy='scratch',
                storage_notice='Verified archive in shared scratch; not a claim of bulk redundancy')
        atomic_json(record_path, record)
        staged.unlink()
    # Publish links only after the durable, verified archive exists.
    for name in record['roots']:
        source = run/name
        subset = {k: v for k, v in expected.items() if k.startswith(name+'/')}
        held = source.with_name(source.name+'.bulk_prearchive')
        if source.is_symlink():
            if inventory(source.resolve(), name) != subset:
                raise ValueError('Existing scratch cache no longer matches durable archive')
        else:
            cache = CACHE/run.name/('cache_job_' + os.environ['SLURM_JOB_ID'])/name
            cache.parent.mkdir(parents=True, exist_ok=True)
            original = source if source.exists() else held
            if not cache.exists():
                shutil.copytree(original, cache)
            if inventory(cache, name) != subset or inventory(original, name) != subset:
                raise ValueError('Cache/source content mismatch before commit')
            if source.exists():
                source.rename(held)
            source.symlink_to(cache, target_is_directory=True)
        if held.exists():
            if inventory(held, name) != subset:
                raise ValueError('Retained original changed; do not remove it')
            shutil.rmtree(held)
    record['status'] = ('verified_scratch_archive_and_cache' if scratch_storage
                        else 'durable_archive_and_verified_scratch_cache')
    record['committed_unix'] = time.time()
    atomic_json(record_path, record)
    return record
