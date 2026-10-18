"""Bind a completed same-input retry to a predeclared preemption plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    plan = json.loads(a.plan.read_text())
    tasks = Path(plan['source_tasks_path'])
    if sha(tasks) != plan['source_tasks_sha256']:
        raise ValueError('Frozen tasks changed')
    data = {k: plan[k] for k in ['schema_version', 'kind', 'source_tasks_path', 'source_tasks_sha256']}
    data['predeclared_plan_sha256'] = sha(a.plan)
    data['retries'] = {}
    for run_id, item in plan['retries'].items():
        for key in ['source_manifest', 'preemption']:
            if sha(item[key + '_path']) != item[key + '_sha256']:
                raise ValueError('Original interruption evidence changed')
        new = json.loads(Path(item['retry_manifest_path']).read_text())
        if new.get('success') is not True or not new.get('time_end'):
            raise ValueError('Retry incomplete or failed; preserve without inventing a result')
        job = str(new['slurm_job_id'])
        records = subprocess.check_output([
            'sacct', '-j', job, '--parsable2', '--noheader',
            '--format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList'], text=True)
        if not any(line.split('|')[:3] == [job, 'COMPLETED', '0:0'] for line in records.splitlines()):
            raise ValueError('Successful Slurm accounting required')
        data['retries'][run_id] = dict(item,
            retry_manifest_sha256=sha(item['retry_manifest_path']),
            retry_metrics_sha256=sha(item['retry_metrics_path']),
            retry_slurm_job_id=job, retry_sacct_record=records)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
