#!/usr/bin/env python3
"""Persistent Slurm supervisor: identical-input infrastructure retries and final audit."""
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
RUNS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE'}
INFRA = {'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def submit_once(name, command):
    """Recover a submission interrupted between sbatch and journal persistence."""
    journal = BASE / ('submission_' + name + '.json')
    if journal.exists():
        return json.loads(journal.read_text())['job']
    probe = subprocess.check_output(['sacct', '-X', '-n', '-P', '-S', '2026-10-17',
                                     '--name', name, '-o', 'JobID%64,JobName%100'], text=True)
    found = {line.split('|')[0].split('_')[0] for line in probe.splitlines()
             if len(line.split('|')) > 1 and line.split('|')[1] == name}
    if len(found) > 1:
        raise RuntimeError('Ambiguous prior supervisor submission: ' + name)
    if found:
        job = found.pop()
    else:
        job = subprocess.check_output(['sbatch', '--parsable', '--job-name=' + name] + command,
                                      cwd=ROOT, text=True).strip().split(';')[0]
    save(journal, {'job': job, 'command': command, 'name': name, 'time': time.time()})
    return job


def scheduler(jobs):
    raw = subprocess.check_output(['sacct', '-X', '-P', '-j', ','.join(sorted(jobs)),
                                   '-o', 'JobID%64,State%30,ExitCode,ElapsedRaw'], text=True)
    return {row['JobID']: row for row in csv.DictReader(raw.splitlines(), delimiter='|')}


def retry(state, item, accounting):
    old = item['task']
    number = item.get('auto_retry_count', 0) + 1
    prefix = 't2auto_' + old['method'].lower() + '_' + old['design'] + '_' + str(number)
    directory = BASE / prefix
    directory.mkdir(exist_ok=True)
    new = dict(old, run_id=prefix)
    taskfile = directory / 'tasks.jsonl'
    raw = (json.dumps(new, sort_keys=True) + '\n').encode()
    if taskfile.exists():
        assert taskfile.read_bytes() == raw
    else:
        taskfile.write_bytes(raw)
    binding = directory / 'retry_binding.json'
    if not binding.exists():
        save(binding, {'schema_version': 1, 'old_tasks': item['plan']['path'],
            'old_tasks_sha256': item['plan']['sha256'], 'old_task_index': item['index'],
            'old_run_id': old['run_id'], 'old_job': item['job'],
            'new_tasks': str(taskfile), 'new_tasks_sha256': sha(taskfile),
            'new_task_index': 0, 'new_run_id': new['run_id'],
            'reason': 'Scheduler infrastructure interruption; identical task except run_id',
            'scheduler_evidence': accounting, 'created_unix': time.time()})
    job = submit_once(prefix, ['--array=0', '--nodelist=enterprise', '--mem=16G',
        '--time=02:30:00', 'backend/scripts/table2_fresh.slurm', str(taskfile), sha(taskfile)])
    plan = {'path': str(taskfile), 'sha256': sha(taskfile), 'array_job': job}
    archive = '/fact_data/bohanli/flowplace_fql_backend/table2_archives/' + new['run_id'] + '/automatic'
    create = submit_once(prefix + '_arc', ['--dependency=afterok:' + job,
        'backend/scripts/archive_table2_artifacts.slurm', 'create', '--run', str(RUNS / new['run_id']),
        '--archive-dir', archive])
    commit = submit_once(prefix + '_commit', ['--dependency=afterok:' + create,
        'backend/scripts/archive_table2_artifacts.slurm', 'commit', '--archive-dir', archive,
        '--scratch-dir', '/scratch/bohanli/flowplace_table2/artifact_cache/' + new['run_id'] + '/automatic',
        '--confirm-remove-bulk-backup'])
    state['plans'].append(plan)
    state['infrastructure_retries'].append({'path': str(binding), 'sha256': sha(binding)})
    state['additional_accounting_jobs'] += [create, commit]
    state['events'].append({'kind': 'infrastructure_retry', 'old': old['run_id'],
                            'new': new['run_id'], 'job': job, 'time': time.time()})
    return {'task': new, 'plan': plan, 'index': 0, 'job': job + '_0', 'auto_retry_count': number}


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    os.chdir(ROOT)
    BASE.mkdir(exist_ok=True)
    with (BASE / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = BASE / 'state.json'
        if path.exists():
            state = json.loads(path.read_text())
        else:
            config = ROOT / 'backend/plans/table2_source_era_dreamplace_v1/final_audit_invocation.json'
            state = json.loads(config.read_text())
            state.update(status='watching', events=[], selected={}, created_unix=time.time())
            superseded = {json.loads(Path(b['path']).read_text())['old_run_id']
                          for b in state['infrastructure_retries']}
            for plan in state['plans']:
                assert sha(plan['path']) == plan['sha256']
                for index, line in enumerate(Path(plan['path']).read_text().splitlines()):
                    task = json.loads(line)
                    if task['run_id'] not in superseded:
                        key = task['method'] + '/' + task['design']
                        assert key not in state['selected']
                        state['selected'][key] = {'task': task, 'plan': plan, 'index': index,
                                                  'job': str(plan['array_job']) + '_' + str(index)}
            assert len(state['selected']) == 12
            save(path, state)
        if state['status'] == 'audit_submitted':
            return
        while True:
            try:
                records = scheduler({str(p['array_job']) for p in state['plans']})
                done = True
                for key, item in list(state['selected'].items()):
                    account = records.get(item['job'], {})
                    status = account.get('State', '').split()[0].rstrip('+') if account.get('State') else ''
                    if status not in TERMINAL:
                        done = False
                        continue
                    run = RUNS / item['task']['run_id']
                    unfinished = not any((run / n).exists() for n in
                                         ['metrics.json', 'author_metrics.json', 'fullflow_audit.json'])
                    unfinished &= not any((run / 'orfs_work/logs').glob('*/*/*/6_report.json'))
                    if status in INFRA and unfinished and (run / 'manifest.json').exists() and item.get('auto_retry_count', 0) < 4:
                        remaining = int(os.getxattr('/fact_data/bohanli', 'ceph.quota.max_bytes')) \
                                    - int(os.getxattr('/fact_data/bohanli', 'ceph.dir.rbytes'))
                        if remaining >= 12 * 1024**3:
                            state['selected'][key] = retry(state, item, account)
                            save(path, state)
                            done = False
                        else:
                            state['events'].append({'kind': 'retry_not_launched_storage_headroom',
                                'run': item['task']['run_id'], 'remaining_bytes': remaining,
                                'time': time.time()})
                state['last_accounting'] = records
                state['last_checked_unix'] = time.time()
                save(path, state)
                if done:
                    invocation = {key: state[key] for key in
                        ['plans', 'infrastructure_retries', 'source_sha256', 'additional_accounting_jobs']}
                    invocation['output'] = '/fact_data/bohanli/flowplace_fql_backend/table2_reports/background_completed_v1'
                    final = BASE / 'final_invocation.json'
                    save(final, invocation)
                    job = submit_once('t2auto_final_audit', ['analysis/run_table2_method_audit.slurm',
                        '--invocation', str(final), '--invocation-sha256', sha(final)])
                    state.update(status='audit_submitted', final_audit_job=job, completed_unix=time.time())
                    save(path, state)
                    return
            except Exception as exc:
                state['last_supervisor_error'] = repr(exc)
                save(path, state)
                print(repr(exc), flush=True)
            time.sleep(45)


if __name__ == '__main__':
    main()
