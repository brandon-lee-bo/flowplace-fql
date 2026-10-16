"""Resolve an explicitly bound successful retry of a preempted fixed GRT task."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_retry_map(path, tasks_path, rows):
    if path is None:
        return None
    path, tasks_path = Path(path).resolve(), Path(tasks_path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or data.get('kind') != 'same_input_infrastructure_retries':
        raise ValueError('Unknown infrastructure retry-map schema')
    if Path(data['source_tasks_path']).resolve() != tasks_path or data['source_tasks_sha256'] != sha(tasks_path):
        raise ValueError('Retry map does not bind the unchanged requested task manifest')
    requested = {row['run_id'] for row in rows}
    if not data['retries'] or not set(data['retries']).issubset(requested):
        raise ValueError('Retry map contains unrequested or empty replacements')
    return {'path': str(path), 'sha256': sha(path), 'data': data}


def resolve_successful_retry(task, tasks_path, original_run, retry_map):
    """Keep the logical task fixed; admit only a completed same-input retry."""
    key = task['run_id']
    if retry_map is None or key not in retry_map['data']['retries']:
        return original_run, None
    item = retry_map['data']['retries'][key]
    original_run = Path(original_run).resolve()
    original_manifest = original_run / 'manifest.json'
    preemption = original_run / 'preemption.json'
    for field, path in [('source_manifest', original_manifest), ('preemption', preemption)]:
        if Path(item[field + '_path']).resolve() != path or item[field + '_sha256'] != sha(path):
            raise ValueError(f'Retry source {field} path/hash mismatch')
    old = json.loads(original_manifest.read_text())
    interrupted = json.loads(preemption.read_text())
    if old.get('success') is not False or old.get('time_end') or (original_run / 'metrics.json').exists():
        raise ValueError('Retry source must remain an incomplete original attempt without metrics')
    if interrupted.get('kind') != 'slurm_preemption' or interrupted.get('state') != 'PREEMPTED':
        raise ValueError('Retry is not backed by a Slurm preemption')
    if interrupted['manifest_sha256'] != sha(original_manifest) or str(interrupted['slurm_job_id']) != str(old['slurm_job_id']):
        raise ValueError('Preemption does not bind the original manifest/job')
    records = [line.split('|') for line in interrupted['sacct_record'].splitlines() if line.strip()]
    # Recorded columns: JobIDRaw,JobID,State,ExitCode,Start,End,Elapsed,NodeList.
    if not any(len(row) >= 8 and row[0] == str(old['slurm_job_id']) and row[2] == 'PREEMPTED' for row in records):
        raise ValueError('No PREEMPTED raw Slurm record for the original attempt')
    new_manifest = Path(item['retry_manifest_path']).resolve()
    new_run = new_manifest.parent
    if new_run == original_run or new_run.name != item['retry_run_id'] or new_manifest.name != 'manifest.json':
        raise ValueError('Retry must use its distinct declared result directory')
    new_metrics = new_run / 'metrics.json'
    if Path(item['retry_metrics_path']).resolve() != new_metrics:
        raise ValueError('Retry metrics must belong to the declared retry directory')
    if sha(new_manifest) != item['retry_manifest_sha256'] or sha(new_metrics) != item['retry_metrics_sha256']:
        raise ValueError('Frozen retry manifest/metrics changed')
    new = json.loads(new_manifest.read_text())
    if new.get('success') is not True or not new.get('time_end') or str(new.get('slurm_job_id')) != str(item['retry_slurm_job_id']):
        raise ValueError('Retry must have a completed successful manifest for its declared job')
    for field in ['design', 'def_sha256', 'design_config_sha256', 'openroad_sha256',
                  'tool_image_sha256', 'platform_config_sha256', 'physical_flow_revision',
                  'physical_flow_components', 'stage', 'not_final_table2', 'num_cores']:
        if field not in old or field not in new or old[field] != new[field]:
            raise ValueError(f'Retry changes input/physical-flow identity: {field}')
    if new['stage'] != 'global_route_only' or new['not_final_table2'] is not True:
        raise ValueError('Only the fixed GRT evaluator may be retried')
    if hashlib.sha256(json.dumps(new['physical_flow_components'], sort_keys=True).encode()).hexdigest() != new['physical_flow_revision']:
        raise ValueError('Physical-flow component hash is inconsistent')
    def_path = Path(task['def_path']).resolve()
    if any(Path(m['def_path']).resolve() != def_path for m in [old, new]) or sha(def_path) != task['def_sha256'] or new['def_sha256'] != task['def_sha256']:
        raise ValueError('Requested/original/retried actual DEF differs')
    proof = {'requested_run_id': key, 'actual_run_id': new_run.name,
             'retry_map_path': retry_map['path'], 'retry_map_sha256': retry_map['sha256'],
             'original_manifest_path': str(original_manifest), 'original_manifest_sha256': sha(original_manifest),
             'preemption_path': str(preemption), 'preemption_sha256': sha(preemption),
             'actual_manifest_path': str(new_manifest), 'actual_manifest_sha256': sha(new_manifest),
             'actual_metrics_sha256': sha(new_metrics), 'original_slurm_job_id': old['slurm_job_id'],
             'actual_slurm_job_id': new['slurm_job_id'], 'physical_flow_revision': new['physical_flow_revision'],
             'same_input_and_physical_flow_verified': True,
             'interpretation': 'Same requested candidate retried after infrastructure preemption; original attempt remains intact.'}
    return new_run, proof
