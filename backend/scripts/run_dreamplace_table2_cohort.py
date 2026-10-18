#!/usr/bin/env python3
"""Run a frozen six-design DREAMPlace generation plan serially on one GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BULK = Path('/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2')
DESIGNS = ['bp_fe', 'bp', 'ariane133', 'ariane136', 'bp_be', 'swerv_wrapper']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_result(task, installed):
    work = BULK / task['run_id']
    manifest_path = work / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest['success'] or manifest['design'] != task['design'] or manifest['run_id'] != task['run_id']:
        raise RuntimeError('Missing successful matching generation result.')
    if manifest['runner_sha256'] != task['runner_sha256'] or manifest['installation_root'] != str(installed):
        raise RuntimeError('Generation result used a different runner or installation.')
    if manifest['execution_environment']['gpu_name'] != 'NVIDIA H20' or manifest['requested_device'] != 'cuda':
        raise RuntimeError('Generation result is outside the frozen H20 GPU cohort.')
    if not manifest['actual_placement_tensor_device'].startswith('cuda'):
        raise RuntimeError('Generation did not actually allocate its placement tensor on CUDA.')
    if manifest['execution_environment']['gpu_capability'] != [9, 0]:
        raise RuntimeError('Generation result capability is outside the frozen GPU cohort.')
    if manifest['installation_file_hashes_sha256'] != task['runner_installation_manifest_sha256']:
        raise RuntimeError('Generation installation file hashes differ from the predeclared smoke.')
    if task.get('operator_patch_validation'):
        patch = manifest.get('operator_patch', {})
        if patch.get('path') != task['operator_patch_validation'] or patch.get('sha256') != task['operator_patch_validation_sha256']:
            raise RuntimeError('Generation operator correction differs from the frozen patch validation.')
    for field, digest_field in [('source_def_path', 'source_def_sha256'), ('author_config_path', 'author_config_sha256')]:
        if manifest['input_sha256'].get(task[field]) != task[digest_field]:
            raise RuntimeError('Generation input differs from frozen task: ' + field)
    for path, digest in task['library_and_sdc_sha256'].items():
        if manifest['input_sha256'].get(path) != digest:
            raise RuntimeError('Generation library/SDC differs from frozen task: ' + path)
    config = json.loads((work / 'config.json').read_text())
    if config['random_seed'] != 1000 or any(config[name] != task[name] for name in ['macro_halo_x', 'macro_halo_y']):
        raise RuntimeError('Generation seed/halo differs from frozen protocol.')
    handoff = Path(manifest['handoff_def'])
    audit_path = Path(manifest['handoff_audit'])
    audit = json.loads(audit_path.read_text())
    if sha(handoff) != manifest['handoff_sha256'] or sha(audit_path) != manifest['handoff_audit_sha256']:
        raise RuntimeError('Generation handoff or audit changed.')
    if audit['design'] != task['design'] or audit['output_sha256'] != manifest['handoff_sha256'] or not audit['legal_no_halo']:
        raise RuntimeError('Generation geometry audit is not bound to the handoff.')
    return {'method': 'DREAMPlace', 'design': task['design'], 'generation_run_id': task['run_id'],
            'def_path': str(handoff), 'def_sha256': manifest['handoff_sha256'],
            'generation_manifest': str(manifest_path), 'generation_manifest_sha256': sha(manifest_path),
            'geometry_audit': str(audit_path), 'geometry_audit_sha256': manifest['handoff_audit_sha256'],
            'source_runner_sha256': manifest['runner_sha256'],
            'source_commit': manifest['source_commit'],
            'installation_file_hashes_sha256': manifest['installation_file_hashes_sha256'],
            'ppa_evaluated': False, 'status': 'generation_passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--tasks-sha256', required=True)
    parser.add_argument('--install', type=Path, required=True)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        parser.error('Run through the single-GPU Slurm wrapper.')
    if sha(args.tasks) != args.tasks_sha256:
        parser.error('Frozen task file hash mismatch.')
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if [task['design'] for task in tasks] != DESIGNS:
        parser.error('Expected the complete ordered six-design cohort.')
    installed = args.install.resolve()
    work = BULK / ('cohort_' + job)
    work.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.tasks, work / 'tasks.jsonl')
    shutil.copyfile(__file__, work / Path(__file__).name)
    report = {'slurm_job_id': job, 'tasks_sha256': args.tasks_sha256, 'start_time': time.time(),
              'driver_sha256': sha(__file__), 'success': False, 'ppa_evaluated': False, 'rows': []}
    save(work / 'cohort_manifest.json', report)
    runner = ROOT / 'backend/scripts/run_dreamplace_table2.py'
    for task in tasks:
        row = {'design': task['design'], 'run_id': task['run_id'], 'success': False, 'start_time': time.time()}
        try:
            if sha(runner) != task['runner_sha256'] or str(installed) != task['installation_root']:
                raise RuntimeError('Frozen runner or installation changed before design execution.')
            if task.get('operator_patch_validation') and sha(task['operator_patch_validation']) != task['operator_patch_validation_sha256']:
                raise RuntimeError('Frozen operator-patch validation changed before design execution.')
            for field, digest_field in [('source_def_path', 'source_def_sha256'), ('author_config_path', 'author_config_sha256')]:
                if sha(task[field]) != task[digest_field]:
                    raise RuntimeError('Frozen input changed before design execution: ' + field)
            for path, digest in task['library_and_sdc_sha256'].items():
                if sha(path) != digest:
                    raise RuntimeError('Frozen library/SDC changed before design execution: ' + path)
            if task.get('reuse_successful_smoke'):
                # Reuse only the explicitly predeclared bp_fe result, never
                # search through candidate seeds or previously generated runs.
                if task['design'] != 'bp_fe':
                    raise RuntimeError('Only the predeclared bp_fe smoke may be reused.')
                if sha(BULK / task['run_id'] / 'manifest.json') != task['reuse_manifest_sha256']:
                    raise RuntimeError('The frozen successful smoke manifest changed.')
                row['reused'] = True
            else:
                env = dict(os.environ)
                temporary = Path(env['TMPDIR']) / task['design']
                temporary.mkdir(exist_ok=False)
                env['TMPDIR'] = str(temporary)
                command = [sys.executable, str(runner), '--gpu', '--install', str(installed),
                           '--design', task['design'], '--run-id', task['run_id']]
                row['command'] = command
                with (work / (task['design'] + '.log')).open('w') as log:
                    row['exit_code'] = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if row['exit_code'] != 0:
                    raise RuntimeError('Generation subprocess failed; see retained per-design log/manifest.')
            row['handoff'] = validate_result(task, installed)
            row['success'] = True
        except Exception as exc:
            row['failure'] = repr(exc)
            row['traceback'] = traceback.format_exc()
        finally:
            row['end_time'] = time.time()
            report['rows'].append(row)
            save(work / 'cohort_manifest.json', report)
            print(json.dumps(row), flush=True)
    report.update(end_time=time.time(), success=all(row['success'] for row in report['rows']))
    save(work / 'cohort_manifest.json', report)
    # Keep all six entries, including explicit failures. This is a handoff
    # inventory; root must separately freeze the identical downstream tool flow.
    (work / 'handoffs.jsonl').write_text(''.join(json.dumps(row.get('handoff', {
        'method': 'DREAMPlace', 'design': row['design'], 'generation_run_id': row['run_id'],
        'status': 'generation_failed', 'failure': row.get('failure'), 'ppa_evaluated': False})) + '\n'
        for row in report['rows']))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
