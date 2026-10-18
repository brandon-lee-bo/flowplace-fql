#!/usr/bin/env python3
"""Archive one completed formal task selected by its frozen six-design manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import archive_table2_staged as archive

DESIGNS = {'ariane133', 'ariane136', 'bp', 'bp_be', 'bp_fe', 'swerv_wrapper'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True, type=Path)
    parser.add_argument('--tasks-sha256', required=True)
    parser.add_argument('--helper-sha256', required=True)
    parser.add_argument('--driver-sha256', required=True)
    parser.add_argument('--commit-verified-artifacts', action='store_true')
    args = parser.parse_args()
    archive.require(args.commit_verified_artifacts, 'Explicit formal-run commit authorization is required')
    job = os.environ.get('SLURM_ARRAY_JOB_ID', '')
    index_text = os.environ.get('SLURM_ARRAY_TASK_ID', '')
    archive.require(re.fullmatch(r'[0-9]+', job) and re.fullmatch(r'[0-9]+', index_text)
                    and os.environ.get('SLURM_JOB_ID'), 'Submit as a Slurm array')
    index = int(index_text)
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    archive.require(1 <= threads <= 2, 'Storage job allocation is capped at two CPUs')
    archive.require(archive.sha(Path(archive.__file__)) == args.helper_sha256, 'Archive helper differs from frozen source')
    archive.require(archive.sha(Path(__file__)) == args.driver_sha256, 'Archive driver differs from frozen source')
    raw = args.tasks.read_bytes()
    archive.require(hashlib.sha256(raw).hexdigest() == args.tasks_sha256, 'Task manifest changed')
    tasks = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    archive.require(len(tasks) == 6 and {row['design'] for row in tasks} == DESIGNS
                    and len({row['run_id'] for row in tasks}) == 6, 'Require all six distinct frozen designs')
    archive.require(0 <= index < len(tasks), 'Array index is outside the frozen manifest')
    archive.require(all(row.get('method') in ('FlowPlace', 'DREAMPlace') and not row.get('diagnostic')
                        and not row.get('make_overrides') for row in tasks), 'Only explicit formal method tasks are eligible')
    archive.require(len({row['method'] for row in tasks}) == 1, 'One method per archive array')
    task = tasks[index]
    archive.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', task['run_id']), 'Unsafe run ID')
    run = archive.RUN_ROOT / task['run_id']
    manifest = json.loads((run / 'manifest.json').read_text())
    archive.require(manifest.get('task_manifest_sha256') == args.tasks_sha256
                    and manifest.get('task_index') == index and manifest.get('task') == task,
                    'Run does not match this frozen task and array index')
    archive.require(manifest.get('method') == task['method'] and not manifest.get('diagnostic')
                    and not manifest.get('not_release_recipe'), 'Run is not an eligible formal method run')
    attempt = 'array_' + job + '_' + index_text
    destination = archive.ARCHIVE_ROOT / run.name / attempt
    scratch = archive.CACHE_ROOT / run.name / attempt
    archive.ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    with (archive.ARCHIVE_ROOT / '.archive.lock').open('a') as lock:
        archive.acquire_lock(lock)
        archive.create(run, destination, threads)
        (destination / 'tasks.jsonl').write_bytes(raw)
        for name in ('archive_table2_staged_task.py', 'archive_table2_staged_task.slurm'):
            shutil.copyfile(Path(__file__).with_name(name), destination / name)
        archive.save(destination / 'task_binding.json', {
            'tasks_path': str(args.tasks.resolve()), 'tasks_sha256': args.tasks_sha256,
            'index': index, 'task': task, 'archive_array_job_id': job,
            'driver_sha256': args.driver_sha256, 'helper_sha256': args.helper_sha256,
            'wrapper_sha256': archive.sha(destination / 'archive_table2_staged_task.slurm'),
            'source_exact_job': str(manifest.get('slurm_array_job_id')) + '_' + str(manifest.get('slurm_array_task_id'))})
        archive.commit(destination, scratch, confirmation=True)


if __name__ == '__main__':
    main()
