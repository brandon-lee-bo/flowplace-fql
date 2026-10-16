#!/usr/bin/env python3
"""Hash-bind a narrowly recognized terminal tool failure without changing its manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import struct
import time
from grt_metrics import atomic_json

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('--tasks', type=Path, required=True)
a = p.parse_args()
run = a.run.resolve()
manifest_path = run / 'manifest.json'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_hash = sha(manifest_path)
m = json.loads(manifest_path.read_text())
log_path = run / 'floorplan_place_cts.log'
text = log_path.read_text()
error = '[ERROR RSZ-0060] Max buffer count reached.'
if m.get('success') or not m.get('time_end'):
    raise SystemExit('Manifest must record a terminal failed run')
if 'floorplan_place_cts exited 2' not in m.get('failure', '') or error not in text:
    raise SystemExit('Not the recognized terminal CTS hold-repair failure')
if 'do-4_1_cts] Error 1' not in text or 'do-cts] Error 2' not in text:
    raise SystemExit('Missing evidence of propagated CTS stage failure')
if (run / 'metrics.json').exists():
    raise SystemExit('Unexpected metrics file on physical failure')
def_path = Path(m['def_path'])
if sha(def_path) != m['def_sha256']:
    raise SystemExit('Input DEF no longer matches manifest')
tasks_path = a.tasks.resolve()
task_rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
matches = [row for row in task_rows if row['run_id'] == run.name]
if len(matches) != 1:
    raise SystemExit('Expected exactly one matching requested task')
task = matches[0]
if task['def_sha256'] != m['def_sha256'] or Path(task['def_path']).resolve() != def_path.resolve():
    raise SystemExit('Requested/evaluated DEF mismatch')
task_bindings = {'task_manifest_path': str(tasks_path), 'task_manifest_sha256': sha(tasks_path),
                 'task_id': task.get('task_id', task['run_id']), 'action_sha256': task['action_sha256']}
for path_key, hash_key in [('def_audit_path', 'def_audit_sha256'), ('action_path', 'action_file_sha256')]:
    if path_key in task:
        if sha(Path(task[path_key])) != task[hash_key]:
            raise SystemExit(f'Requested {path_key} hash mismatch')
        task_bindings[path_key] = task[path_key]
        task_bindings[hash_key] = task[hash_key]
for key in ['action_key', 'action_row', 'method', 'index', 'dataset_sha256']:
    if key in task:
        task_bindings[key] = task[key]
if 'normalized_action_flat' in task:
    values = task['normalized_action_flat']
    inline_hash = hashlib.sha256(struct.pack('<' + 'f' * len(values), *values)).hexdigest()
    if inline_hash != task['action_sha256']:
        raise SystemExit('Requested inline action hash mismatch')
    task_bindings['inline_action_hash_verified'] = True
proc = subprocess.run(['sacct', '-j', str(m['slurm_job_id']), '-X', '-n', '-P',
                       '-o', 'JobIDRaw,State,ExitCode,End'], check=True, text=True, capture_output=True)
rows = [row.split('|') for row in proc.stdout.strip().splitlines() if row.strip()]
if len(rows) != 1 or rows[0][1] != 'FAILED' or rows[0][2] != '1:0':
    raise SystemExit(f'Unexpected Slurm terminal record: {rows}')
result = {
    'schema_version': 1, 'status': 'physical_failure',
    'failure_class': 'cts_hold_repair_max_buffer_count',
    'error_code': 'RSZ-0060', 'error_line': error,
    'error_line_number': text[:text.index(error)].count('\n') + 1,
    'stage': 'floorplan_place_cts', 'substage': 'clock_tree_synthesis_hold_repair',
    'make_exit_code': 2, 'openroad_exit_code': 1,
    'slurm_job_id': m['slurm_job_id'], 'slurm_state': rows[0][1],
    'slurm_exit_code': rows[0][2], 'slurm_end': rows[0][3],
    'time_end': m['time_end'], 'classified_at_unix': time.time(),
    'run_id': run.name, 'run_path': str(run),
    'manifest_path': str(manifest_path), 'manifest_sha256': manifest_hash,
    'def_path': str(def_path), 'def_sha256': sha(def_path),
    'log_path': str(log_path), 'log_sha256': sha(log_path),
    'physical_flow_revision': m['physical_flow_revision'],
    'original_manifest_unchanged': True, 'grt_metrics_available': False,
    'not_parser_failure': True, 'not_preemption': True,
    'interpretation': 'Candidate failed the fixed evaluator during CTS; retain as a feasibility outcome, not a numerical PPA label.',
    **task_bindings,
}
if sha(manifest_path) != manifest_hash:
    raise SystemExit('Manifest changed while classifying')
sidecar = run / 'physical_failure.json'
if sidecar.exists():
    old = json.loads(sidecar.read_text())
    stable = lambda record: {k: v for k, v in record.items() if k != 'classified_at_unix'}
    if stable(old) == stable(result):
        result = old
    else:
        atomic_json(sidecar, result)
else:
    atomic_json(sidecar, result)
print(json.dumps(result, indent=2))
