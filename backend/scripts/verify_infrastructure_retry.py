#!/usr/bin/env python3
"""Validate frozen GRT retries against raw reports and exercise rejection cases."""
import argparse
import copy
import json
from pathlib import Path
from grt_metrics import atomic_json, parse_grt_metrics
from infrastructure_retry import load_retry_map, resolve_successful_retry

p = argparse.ArgumentParser()
p.add_argument('tasks', type=Path)
p.add_argument('retry_map', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = [json.loads(line) for line in a.tasks.read_text().splitlines() if line.strip()]
mapping = load_retry_map(a.retry_map, a.tasks, rows)
base = Path(__file__).resolve().parents[1]
report = {'retry_map_path': mapping['path'], 'retry_map_sha256': mapping['sha256'],
          'verified': [], 'negative_checks': [], 'all_checks_pass': False}
for task in rows:
    if task['run_id'] not in mapping['data']['retries']:
        continue
    original = base / 'results' / task['run_id']
    run, proof = resolve_successful_retry(task, a.tasks, original, mapping)
    metrics = json.loads((run / 'metrics.json').read_text())
    parsed = parse_grt_metrics(run, metrics['runtime_seconds'])
    for key in ['rWL_um', 'WNS_ns', 'TNS_ns', 'Power_W', 'StdCellArea_um2',
                'horizontal_overflow_sum', 'vertical_overflow_sum',
                'horizontal_overflow_edges', 'vertical_overflow_edges',
                'horizontal_overflow_max', 'vertical_overflow_max']:
        if metrics[key] != parsed[key]:
            raise ValueError(f'Retry raw metric mismatch: {key}')
    report['verified'].append(proof)
    for key in ['source_manifest_sha256', 'preemption_sha256', 'retry_manifest_sha256',
                'retry_metrics_sha256', 'retry_slurm_job_id']:
        changed = copy.deepcopy(mapping)
        changed['data']['retries'][task['run_id']][key] = 'invalid-for-rejection-test'
        try:
            resolve_successful_retry(task, a.tasks, original, changed)
        except ValueError:
            report['negative_checks'].append({'task': task['run_id'], 'field': key, 'rejected': True})
        else:
            raise AssertionError(f'Tampered {key} unexpectedly accepted')
    changed_task = copy.deepcopy(task)
    changed_task['def_sha256'] = 'invalid-for-rejection-test'
    try:
        resolve_successful_retry(changed_task, a.tasks, original, mapping)
    except ValueError:
        report['negative_checks'].append({'task': task['run_id'], 'field': 'requested_def_sha256', 'rejected': True})
    else:
        raise AssertionError('Tampered requested DEF unexpectedly accepted')
report['all_checks_pass'] = len(report['verified']) == len(mapping['data']['retries'])
atomic_json(a.output, report)
print(json.dumps(report, indent=2))
