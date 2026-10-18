#!/usr/bin/env python3
"""Predeclared fixed-author-layout controls; never replace baseline rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'backend/plans/table2_source_pin_diagnostic_v1'
REV = 'cd519bb5e861ec4d5d00931c85650c179a864e34'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-job', default='1788546')
    args = parser.parse_args()
    assert args.build_job.isdecimal(), 'Expected a Slurm build job ID'
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    os.chdir(ROOT)
    BASE.mkdir(exist_ok=True)
    assert not (BASE / 'submission.json').exists(), 'Inspect prior submission before recovery'
    candidates = []
    build_root = Path('/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1') / ('build_' + args.build_job)
    for path in build_root.glob('attempt_*/manifest.json'):
        if (path.parent / 'exit_code.txt').read_text().strip() == '0':
            candidates.append(path)
    assert len(candidates) == 1, 'Require one successful, published source build'
    build_path = candidates[0]
    build = json.loads(build_path.read_text())
    assert build['revision'] == REV and sha(build['binary']) == build['binary_sha256']
    old = json.loads((ROOT / 'backend/plans/table2_legacy_tool_bp_fe_v1/tasks.jsonl').read_text())
    assert sha(old['openroad_exe']) == old['openroad_sha256']
    tasks = []
    for tool, design in [('dec2024', 'bp'), ('cd519', 'bp_fe'), ('cd519', 'bp')]:
        path = ROOT / 'backend/releases/flowplace_DEFS/openroad' / (design + '.def')
        task = dict(old, design=design, def_path=str(path), def_sha256=sha(path),
            run_id='t2_fixed_author_' + tool + '_' + design + '_v1',
            recipe_label='fixed_author_layout_source_version_diagnostic_v1',
            source='Author released DEF; tool source and generation-vs-backend controls',
            diagnostic=True, not_primary_table2_result=True, make_overrides={})
        if tool == 'cd519':
            task.update(openroad_exe=build['binary'], openroad_sha256=build['binary_sha256'],
                tool_version=build['reported_version'], source_build_manifest=str(build_path),
                source_build_manifest_sha256=sha(build_path), source_revision=REV)
        tasks.append(task)
    taskfile = BASE / 'tasks.jsonl'
    assert not taskfile.exists()
    taskfile.write_text(''.join(json.dumps(t, sort_keys=True) + '\n' for t in tasks))
    names = ['backend/scripts/table2_fresh.slurm', 'backend/scripts/run_table2.py',
        'backend/scripts/run_table2_source_control.slurm', 'backend/scripts/archive_table2_staged.py',
        'backend/scripts/compare_table2_method.py', 'analysis/summarize_table2_generated.py']
    sources = BASE / 'source_sha256.json'
    sources.write_text(json.dumps({p: sha(ROOT / p) for p in names}, indent=2) + '\n')
    command = ['sbatch', '--parsable', '--array=0-2%1',
        'backend/scripts/run_table2_source_control.slurm', str(taskfile), sha(taskfile),
        str(sources), sha(sources)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    (BASE / 'submission.json').write_text(json.dumps({'time':time.time(), 'job':job,
        'command':command, 'task_sha256':sha(taskfile), 'source_build_manifest':str(build_path),
        'source_build_manifest_sha256':sha(build_path),
        'scope':'Three fixed-author-DEF diagnostics, serial through lossless archival, released GPL0/0 and all other physical parameters fixed. Does not identify the author binary; no baseline-row selection.'}, indent=2)+'\n')
    print(job, flush=True)


if __name__ == '__main__':
    main()
