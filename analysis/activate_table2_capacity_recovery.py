#!/usr/bin/env python3
"""Switch control jobs only after a documented, no-final-metric runtime timeout."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'backend/plans/table2_background_v1'
PRES = ROOT / 'backend/plans/table2_preservation_v1'
RUNS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    tmp = p.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)

def accounting(job):
    raw = subprocess.check_output(['sacct', '-X', '-P', '-j', job, '-o',
        'JobID%64,State%30,ExitCode,ElapsedRaw'], text=True, timeout=20)
    rows = list(csv.DictReader(raw.splitlines(), delimiter='|'))
    return next(row for row in rows if row['JobID'] == job)

def submit(args):
    command = ['sbatch', '--parsable'] + args
    result = subprocess.check_output(command, cwd=ROOT, text=True, timeout=30).strip().split(';')[0]
    return {'job': result, 'command': command, 'time': time.time()}

def main():
    os.chdir(ROOT)
    journal = BASE / 'capacity_transition_v1.json'
    assert not journal.exists(), 'Inspect existing transition before recovery'
    old = json.loads((BASE / 'state.json').read_text())
    entry = old['selected']['DREAMPlace/ariane136']
    evidence = accounting(entry['job'])
    assert evidence['State'] == 'TIMEOUT', evidence
    run = RUNS / entry['task']['run_id']
    m = json.loads((run / 'manifest.json').read_text())
    assert not any(m.get(k) for k in ['success', 'evaluation_complete', 'verification_complete'])
    assert not any((run / n).exists() for n in ['metrics.json', 'author_metrics.json', 'fullflow_audit.json'])
    assert not any((run / 'orfs_work/logs').glob('*/*/*/6_report.json'))
    transition = {'status': 'validated_timeout', 'evidence': evidence, 'run': str(run),
        'physical_manifest_sha256': sha(run / 'manifest.json'),
        'before_control_state': old, 'time': time.time(), 'submissions': {}}
    save(journal, transition)
    # These are controller or pending postprocessor jobs, never physical experiment jobs.
    for job in ['1788095', '1788403', '1788398']:
        a = accounting(job)
        if a['State'].split()[0] in ('RUNNING', 'PENDING', 'CONFIGURING'):
            subprocess.check_call(['scancel', job])
    for job in ['1788095', '1788403', '1788398']:
        for _ in range(30):
            if accounting(job)['State'].split()[0] not in ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Controller cancellation did not terminate: ' + job)
    with (BASE / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((BASE / 'state.json').read_text())
        transition['stopped_control_state'] = state
        assert state['selected']['DREAMPlace/ariane136']['task'] == entry['task']
        if state['status'] == 'audit_submitted':
            transition['superseded_incomplete_final_audit'] = state.get('final_audit_job')
        state['status'] = 'watching'
        for name in ['analysis/watch_table2_batch_capacity.py',
                     'analysis/summarize_table2_methods_capacity.py',
                     'analysis/run_table2_capacity_audit.py']:
            state['source_sha256'][name] = sha(ROOT / name)
        state['events'].append({'kind': 'capacity_controller_transition',
            'evidence': evidence, 'time': time.time(), 'journal': str(journal)})
        save(BASE / 'state.json', state)
    cfg = json.loads((BASE / 'capaware_finish_config_v1.json').read_text())
    cfg['prefix'] = 'source_era_capacity_completed_v1'
    for name in ['analysis/finish_table2_capaware.py', 'analysis/summarize_table2_methods_capaware.py',
                 'analysis/run_table2_capaware_audit.py']:
        cfg['source_sha256'].pop(name, None)
    for name in ['analysis/finish_table2_capacity.py', 'analysis/summarize_table2_methods_capacity.py',
                 'analysis/run_table2_capacity_audit.py']:
        cfg['source_sha256'][name] = sha(ROOT / name)
    cfgpath = BASE / 'capacity_finish_config_v1.json'
    assert not cfgpath.exists(); save(cfgpath, cfg)
    hashes = json.loads((PRES / 'source_sha256.json').read_text())
    for name in ['analysis/watch_table2_preservation_capacity.py', 'backend/scripts/archive_table2_runtime_interrupted.py']:
        hashes[name] = sha(ROOT / name)
    save(PRES / 'source_sha256_capacity.json', hashes)
    watcher = submit(['analysis/watch_table2_batch_capacity.slurm'])
    transition['submissions']['watcher'] = watcher; save(journal, transition)
    finish = submit(['--dependency=afterok:' + watcher['job'], 'analysis/finish_table2_capacity.slurm',
        sha(ROOT / 'analysis/finish_table2_capacity.py'), '--config', str(cfgpath), '--config-sha256', sha(cfgpath)])
    transition['submissions']['finish'] = finish; save(journal, transition)
    preservation = submit(['analysis/watch_table2_preservation_capacity.slurm',
        sha(ROOT / 'analysis/watch_table2_preservation_capacity.py')])
    transition['submissions']['preservation'] = preservation
    transition.update(status='controllers_submitted', completed_unix=time.time())
    save(journal, transition)
    print(json.dumps({'evidence': evidence, 'submissions': transition['submissions']}))

if __name__ == '__main__':
    main()
