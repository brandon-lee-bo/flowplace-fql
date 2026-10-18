#!/usr/bin/env python3
"""Capture live Slurm evidence and audit one immutable Table 2 invocation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--invocation', required=True, type=Path)
    parser.add_argument('--invocation-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm')
    require(sha(args.invocation) == args.invocation_sha256, 'Audit invocation changed')
    config = json.loads(args.invocation.read_text())
    for path, digest in config['source_sha256'].items():
        require(sha(ROOT / path) == digest, 'Audit source changed: ' + path)
    destination = Path(config['output']).absolute()
    require(destination.is_relative_to(REPORTS) and destination != REPORTS
            and destination.resolve() == destination, 'Report must have a real path below the report root')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.invocation, destination / 'invocation.json')
    shutil.copyfile(__file__, destination / Path(__file__).name)
    jobs = sorted(set(str(plan['array_job']) for plan in config['plans'])
                  | set(str(job) for job in config.get('additional_accounting_jobs', [])))
    require(jobs and all(re.fullmatch(r'[0-9]+', job) for job in jobs), 'Unsafe scheduler ID')
    command = ['sacct', '-X', '-P', '-j', ','.join(jobs), '-o',
               'JobID%64,State%30,ExitCode,Elapsed,ElapsedRaw,AllocCPUS,CPUTimeRAW,NodeList%60,Start,End']
    accounting = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    (destination / 'sacct.txt').write_text(accounting.stdout)
    snapshot = {'by_job_id': {}, 'created_unix': time.time(), 'command': command,
                'sacct_sha256': sha(destination / 'sacct.txt')}
    for row in csv.DictReader(accounting.stdout.splitlines(), delimiter='|'):
        if row.get('JobID'):
            snapshot['by_job_id'][row['JobID']] = {
                'state': row['State'], 'exit_code': row['ExitCode'], 'elapsed': row['Elapsed'],
                'elapsed_seconds': row['ElapsedRaw'], 'allocated_cpus': row['AllocCPUS'],
                'cpu_time_seconds': row['CPUTimeRAW'], 'node_list': row['NodeList'],
                'start': row['Start'], 'end': row['End']}
    scheduler = destination / 'scheduler.json'
    scheduler.write_text(json.dumps(snapshot, indent=2) + '\n')
    cmd = [sys.executable, str(ROOT / 'analysis/summarize_table2_methods_capaware.py')]
    for plan in config['plans']:
        cmd += ['--tasks', plan['path'], '--tasks-sha256', plan['sha256'], '--array-job', str(plan['array_job'])]
    for binding in config.get('infrastructure_retries', []):
        cmd += ['--infrastructure-retry', binding['path'], '--infrastructure-retry-sha256', binding['sha256']]
    cmd += ['--scheduler-snapshot', str(scheduler), '--output', str(destination / 'report')]
    result = subprocess.run(cmd, text=True, capture_output=True)
    (destination / 'audit.stdout').write_text(result.stdout)
    (destination / 'audit.stderr').write_text(result.stderr)
    (destination / 'audit_execution.json').write_text(json.dumps({
        'command': cmd, 'returncode': result.returncode, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'invocation_sha256': args.invocation_sha256, 'driver_sha256': sha(__file__),
        'source_sha256': config['source_sha256']}, indent=2) + '\n')
    print(result.stdout, end='')
    print(result.stderr, file=sys.stderr, end='')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
