#!/usr/bin/env python3
"""Release frozen Table 2 jobs conservatively as verified archives free quota."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'backend/plans/table2_exact_import_v1'
RUNS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')
GIB = 1024**3
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'}


def save(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def bulk_size(path):
    """Only this run's two known output trees; never follow archived cache links."""
    total = 0
    for name in ('results', 'reports'):
        start = path / 'orfs_work' / name
        if start.is_symlink() or not start.is_dir():
            continue
        for directory, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
            for name in files:
                try:
                    entry = Path(directory) / name
                    if not entry.is_symlink():
                        total += entry.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    os.chdir(ROOT)
    journal = BASE / 'storage_admission_controller_v1.json'
    held_path = BASE / 'storage_admission_hold_v1.json'
    with (BASE / '.storage_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(journal.read_text()) if journal.exists() else {
            'schema_version': 1, 'releases': [], 'started_unix': time.time(),
            'reserve_bytes': 8 * GIB, 'archive_margin_bytes': GIB,
            'expected_bulk_bytes': {'bp': 10 * GIB, 'ariane133': 6 * GIB,
                                    'ariane136': 6 * GIB},
            'policy': 'One released large compatibility job at a time, including archival; original jobs continue. Never change inputs, seeds or optimization parameters.',
            'controllers': []}
        state['controllers'].append({'job': os.environ['SLURM_JOB_ID'],
            'time': time.time(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        while True:
            try:
                held = json.loads(held_path.read_text())
                original = json.loads((ROOT / 'backend/plans/table2_background_v1/state.json').read_text())
                exact = json.loads((BASE / 'state.json').read_text())
                items = list(original['selected'].values()) + list(exact['selected'].values())
                parents = {item['job'].split('_')[0] for item in items}
                raw = subprocess.check_output(['sacct', '-X', '-P', '-j', ','.join(sorted(parents)),
                    '-o', 'JobID%64,State%30,ExitCode'], text=True)
                accounts = {r['JobID']: r for r in csv.DictReader(raw.splitlines(), delimiter='|')}
                queue = subprocess.check_output(['squeue', '-h', '-u', 'bohanli', '-r',
                    '-o', '%i|%T|%r'], text=True)
                queued = {r.split('|')[0]: r.split('|')[1:] for r in queue.splitlines()}
                held_ids = {row['job'] for row in held['held']}
                still_held = {job for job in held_ids if queued.get(job, ['', ''])[1] == 'JobHeldUser'}
                if not still_held:
                    state.update(status='all_frozen_jobs_released', completed_unix=time.time())
                    save(journal, state)
                    return
                remaining = int(os.getxattr('/fact_data/bohanli', 'ceph.quota.max_bytes')) - int(os.getxattr('/fact_data/bohanli', 'ceph.dir.rbytes'))
                future = 0
                active_details = []
                for item in items:
                    job = item['job']
                    account = accounts.get(job)
                    assert account, 'Missing scheduler accounting for ' + job
                    status = account['State'].split()[0].rstrip('+')
                    if job in still_held or status not in ACTIVE:
                        continue
                    design = item['task']['design']
                    if design not in state['expected_bulk_bytes']:
                        continue
                    run = RUNS / item['task']['run_id']
                    current = bulk_size(run)
                    growth = max(0, state['expected_bulk_bytes'][design] - current)
                    future += growth
                    active_details.append({'job': job, 'run': str(run), 'bulk_bytes': current,
                                           'future_reservation_bytes': growth})
                # Wait for a previously admitted compatibility job's lossless archive commit.
                outstanding = []
                for item in exact['selected'].values():
                    if item['task']['design'] not in state['expected_bulk_bytes'] or item['job'] in still_held:
                        continue
                    run = RUNS / item['task']['run_id']
                    if bulk_size(run) or accounts[item['job']]['State'].split()[0] in ACTIVE:
                        outstanding.append(item['job'])
                sample = {'time': time.time(), 'remaining_bytes': remaining,
                    'future_growth_bytes': future, 'active': active_details,
                    'waiting_for_archive_or_execution': outstanding, 'held': sorted(still_held)}
                state.update(status='waiting_for_safe_storage', latest=sample)
                save(journal, state)
                if not outstanding:
                    # Prefer larger jobs when safe; smaller jobs can proceed first.
                    for job in ('1788317_3', '1788319_3', '1788317_4', '1788319_4', '1788317_5', '1788319_5'):
                        if job not in still_held:
                            continue
                        item = next(x for x in exact['selected'].values() if x['job'] == job)
                        design = item['task']['design']
                        required = future + state['expected_bulk_bytes'][design] + state['reserve_bytes'] + state['archive_margin_bytes']
                        if remaining < required:
                            continue
                        event = {'job': job, 'design': design, 'time': time.time(),
                            'quota_snapshot': sample, 'required_bytes': required, 'status': 'releasing'}
                        state['releases'].append(event)
                        save(journal, state)
                        result = subprocess.run(['scontrol', 'release', job], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        event.update(status='released' if result.returncode == 0 else 'failed',
                                     returncode=result.returncode, output=result.stdout)
                        save(journal, state)
                        assert result.returncode == 0, result.stdout
                        held['release_events'].append(event)
                        save(held_path, held)
                        print(json.dumps(event), flush=True)
                        break
                state.pop('last_error', None)
                save(journal, state)
            except Exception as exc:
                state['last_error'] = repr(exc)
                save(journal, state)
                print(repr(exc), flush=True)
            time.sleep(45)


if __name__ == '__main__':
    main()
