"""Complete the fixed second-design label manifest; recover preemption only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_controller import wait_job, submit_unique, RESTARTABLE
from .graph_o2o_design_labels import inputs, requests, report


def run(args):
    values = inputs(args.plan)
    plan_path, plan, original, verification, arrays, selected = values
    output = Path(plan['output'])
    path = output / 'measurement_controller.json'
    journal = json.loads(path.read_text()) if path.exists() else dict(
        label_plan_sha256=sha(plan_path), backend_jobs=[], started_unix=time.time())
    if journal['label_plan_sha256'] != sha(plan_path):
        raise ValueError('Second-design controller changed label plan')
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    key = os.environ['SLURM_JOB_ID'] + (f'_restart_{restart}' if restart else '')
    snapshot = output / 'controller_sources' / key
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ['graph_o2o_design_label_controller.py', 'graph_o2o_design_labels.py',
                 'graph_o2o_design_verify.py', 'graph_o2o_controller.py']:
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    journal['controller_job_id'] = os.environ['SLURM_JOB_ID']
    journal.setdefault('startups', {})[key] = dict(path=str(snapshot), restart_count=restart)
    save = lambda: atomic_json(path, journal)
    save()
    try:
        rows, task_path = requests(plan_path, plan, selected, arrays)
        batches = journal['backend_jobs']
        if batches:
            observed = wait_job(batches[-1]['job'], journal, save)
            if not all(state == 'COMPLETED' or state in RESTARTABLE for state in observed):
                raise RuntimeError('Previous label worker requires evidence-based repair: ' + str(observed))
        while True:
            missing = [index for index, row in enumerate(rows)
                       if not (Path(row['sample_dir']) / 'outcome.json').exists()]
            journal['completed_outcomes'] = len(rows) - len(missing)
            save()
            if not missing:
                break
            indices = missing[:1] if not batches else missing
            job = submit_unique('go2olabel_' + sha(plan_path)[:10] + '_b' + str(len(batches)),
                ['--array=' + ','.join(map(str, indices)) + '%' + str(plan['backend_concurrency']),
                 '--exclude=' + ','.join(plan['backend_exclude'])],
                [str(ROOT / 'fql/graph_o2o_design_labels.slurm'), '--plan', str(plan_path), '--stage', 'measure'])
            batches.append(dict(job=job, indices=indices, tasks_sha256=sha(task_path)))
            save()
            observed = wait_job(job, journal, save)
            if not all(state == 'COMPLETED' or state in RESTARTABLE for state in observed):
                raise RuntimeError('Label worker requires evidence-based repair: ' + str(observed))
        journal['stage'] = 'auditing_all_80_physical_outcomes'
        save()
        report(*values)
        journal['stage'] = 'all_80_real_labels_audited_reward_scale_frozen'
        journal['finished_unix'] = time.time()
        save()
    except BaseException as error:
        journal['last_error'] = repr(error)
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use a Slurm controller')
    run(args)
