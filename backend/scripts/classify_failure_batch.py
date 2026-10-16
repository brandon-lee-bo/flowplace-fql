#!/usr/bin/env python3
"""Account for fixed tasks; never turn unknown/infra/parser failures into physical labels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from grt_metrics import atomic_json
from infrastructure_retry import load_retry_map, resolve_successful_retry

p = argparse.ArgumentParser()
p.add_argument('tasks', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--infrastructure-retry-map', type=Path)
a = p.parse_args()
base = Path(__file__).resolve().parents[1]
tasks = a.tasks.resolve()
rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
retry_map = load_retry_map(a.infrastructure_retry_map, tasks, rows)
result = {'tasks_path': str(tasks), 'tasks_sha256': hashlib.sha256(tasks.read_bytes()).hexdigest(),
          'requested': len(rows), 'success_manifests': [], 'verified_physical_failures': [],
          'unknown_or_incomplete': [],
          'infrastructure_retries': [],
          'note': 'Success manifests are counted here; numerical metrics remain subject to independent audit/aggregation.'}
for row in rows:
    run = base / 'results' / row['run_id']
    try:
        run, retry_proof = resolve_successful_retry(row, tasks, run, retry_map)
    except (OSError, ValueError, KeyError) as exc:
        result['unknown_or_incomplete'].append({'run_id': row['run_id'], 'reason': f'invalid_infrastructure_retry: {exc}'})
        continue
    if retry_proof:
        result['infrastructure_retries'].append(retry_proof)
    manifest_path = run / 'manifest.json'
    if not manifest_path.exists():
        result['unknown_or_incomplete'].append({'run_id': row['run_id'], 'reason': 'missing_manifest'})
        continue
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('success') and manifest.get('time_end'):
        result['success_manifests'].append(row['run_id'])
        continue
    if not manifest.get('time_end'):
        result['unknown_or_incomplete'].append({'run_id': row['run_id'], 'reason': 'not_terminal_manifest'})
        continue
    proc = subprocess.run([sys.executable, str(base / 'scripts/classify_physical_failure.py'),
                           str(run), '--tasks', str(tasks)], capture_output=True, text=True)
    if proc.returncode:
        result['unknown_or_incomplete'].append({'run_id': row['run_id'],
                                               'reason': proc.stderr.strip() or proc.stdout.strip()})
        continue
    sidecar = run / 'physical_failure.json'
    result['verified_physical_failures'].append({'run_id': row['run_id'],
        'sidecar_path': str(sidecar), 'sidecar_sha256': hashlib.sha256(sidecar.read_bytes()).hexdigest()})
result['physical_success_count'] = len(result['success_manifests'])
result['verified_physical_failure_count'] = len(result['verified_physical_failures'])
result['outcome_completion_count'] = result['physical_success_count'] + result['verified_physical_failure_count']
result['all_outcomes_classified'] = result['outcome_completion_count'] == len(rows) and not result['unknown_or_incomplete']
atomic_json(a.output, result)
print(json.dumps(result, indent=2))
if not result['all_outcomes_classified']:
    raise SystemExit(1)
