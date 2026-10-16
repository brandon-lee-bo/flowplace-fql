"""Snapshot reserved Slurm resources for this experiment, including failures.

Accounting here reports allocated CPU/GPU time, not measured utilization, energy,
or a monetary cost. Pending array ranges have zero allocated resources. Never
count batch/extern steps a second time.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--extra-jobs', default='')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run accounting aggregation through Slurm')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / 'docs/run_ledger.md']
    ids = set(re.findall(r'(?<!\d)(\d{7})(?!\d)', sources[0].read_text()))
    log_paths = list((ROOT / 'logs').glob('*'))
    for path in log_paths:
        ids.update(re.findall(r'(?<!\d)(\d{7})(?!\d)', path.name))
    ids.update(value for value in args.extra_jobs.split(',') if value)
    if not ids or any(not value.isdigit() for value in ids):
        raise ValueError('Invalid job IDs')
    fields = ['JobIDRaw', 'JobID', 'JobName', 'State', 'ExitCode', 'ElapsedRaw',
              'AllocCPUS', 'AllocTRES', 'CPUTimeRAW', 'NodeList', 'Submit', 'Start', 'End']
    command = ['sacct', '-X', '-P', '--noheader', '-j', ','.join(sorted(ids)),
               '--format=' + ','.join(fields)]
    raw = subprocess.check_output(command, text=True)
    (out / 'sacct.txt').write_text(raw)
    rows = []
    seen = set()
    for line in raw.splitlines():
        values = line.split('|')
        if len(values) != len(fields):
            raise ValueError(f'Unexpected accounting row: {line}')
        row = dict(zip(fields, values))
        identity = row['JobIDRaw']
        if '.' in identity:
            raise ValueError('Allocation-only query unexpectedly included steps')
        if identity in seen:
            continue  # Array parent and explicitly listed child can overlap.
        seen.add(identity)
        elapsed, cpu_time = int(row['ElapsedRaw']), int(row['CPUTimeRAW'])
        tres = dict(item.split('=', 1) for item in row['AllocTRES'].split(',') if '=' in item)
        gpu_count = int(tres.get('gres/gpu', '0'))
        row.update(reserved_cpu_hours=cpu_time / 3600,
                   reserved_gpu_hours=gpu_count * elapsed / 3600,
                   allocated_gpu_count=gpu_count,
                   allocated_wall_seconds=elapsed)
        rows.append(row)
    by_state, by_name = {}, {}
    for row in rows:
        for destination, key in [(by_state, row['State']), (by_name, row['JobName'])]:
            group = destination.setdefault(key, {'allocation_rows': 0, 'reserved_cpu_hours': 0,
                                                  'reserved_gpu_hours': 0})
            group['allocation_rows'] += 1
            group['reserved_cpu_hours'] += row['reserved_cpu_hours']
            group['reserved_gpu_hours'] += row['reserved_gpu_hours']
    report = {
        'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'query': command,
        'requested_job_ids': sorted(ids), 'job_id_sources': [str(p) for p in sources],
        'log_filename_sources': [str(p) for p in log_paths],
        'coverage': 'Jobs named in the experiment ledger or experiment-local log filenames, plus explicit extras',
        'resource_semantics': 'Reserved allocation time, including failed jobs; running costs are partial snapshots',
        'actual_utilization': 'Not inferred. Cluster accounting returned zero TotalCPU and no MaxRSS in inspected jobs.',
        'pending_arrays': 'One compressed pending-range row can represent multiple tasks; allocation_rows is not task count',
        'wall_time_warning': 'Parallel task elapsed times cannot be summed into experiment wall-clock duration',
        'reserved_cpu_hours': sum(r['reserved_cpu_hours'] for r in rows),
        'reserved_gpu_hours': sum(r['reserved_gpu_hours'] for r in rows),
        'by_state': by_state, 'by_job_name': by_name, 'allocations': rows,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'allocations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ['captured_utc', 'reserved_cpu_hours', 'reserved_gpu_hours', 'resource_semantics']}))


if __name__ == '__main__':
    main()
