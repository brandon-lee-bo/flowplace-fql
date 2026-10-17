"""Measure the eight fixed feedback-channel diagnostic layouts with recovery."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_controller import RESTARTABLE, submit_unique, wait_job


def run(args):
    exp=Path(args.experiment).resolve();path=exp/'measurement_controller.json'
    journal=json.loads(path.read_text()) if path.exists() else dict(config_sha256=sha(exp/'config.json'),
        training_jobs=[args.initial_training_job],backend_jobs=[],concurrency=2,started_unix=time.time())
    if journal['config_sha256']!=sha(exp/'config.json'):raise ValueError('Diagnostic controller config changed')
    journal['controller_job_id']=os.environ['SLURM_JOB_ID']
    save=lambda:atomic_json(path,journal)
    save()
    try:
        while True:
            status=wait_job(journal['training_jobs'][-1],journal,save)
            if status==['COMPLETED']:break
            if not all(s in RESTARTABLE for s in status):raise RuntimeError('Training needs evidence-based repair: '+str(status))
            job=submit_unique('go2o_channels_train_recovery_'+str(len(journal['training_jobs'])),[],
                [str(ROOT/'fql/graph_o2o_feedback_channels.slurm'),'--experiment',str(exp)])
            journal['training_jobs'].append(job);save()
        rows,tasks=load_requests(exp,0,0)
        if len(rows)!=8:raise ValueError('Expected all eight predeclared layouts')
        if journal.get('tasks_sha256',sha(tasks))!=sha(tasks):raise ValueError('Diagnostic tasks changed')
        journal['tasks_sha256']=sha(tasks);save()
        if journal['backend_jobs']:
            status=wait_job(journal['backend_jobs'][-1]['job'],journal,save)
            if not all(s=='COMPLETED' or s in RESTARTABLE for s in status):raise RuntimeError('Existing array needs repair: '+str(status))
        while True:
            missing=[i for i,row in enumerate(rows)if not(Path(row['sample_dir'])/'outcome.json').exists()]
            journal['completed_outcomes']=8-len(missing);save()
            if not missing:break
            job=submit_unique('go2o_channels_grt_'+str(len(journal['backend_jobs'])),
                ['--array='+','.join(map(str,missing))+'%2'],
                [str(ROOT/'fql/graph_o2o_backend.slurm'),'--experiment',str(exp),'--seed','0','--round','0','--storage','scratch'])
            journal['backend_jobs'].append(dict(job=job,indices=missing));save()
            status=wait_job(job,journal,save)
            if not all(s=='COMPLETED' or s in RESTARTABLE for s in status):raise RuntimeError('Physical array needs repair: '+str(status))
        subprocess.run([sys.executable,'-m','fql.graph_o2o_feedback_channels','--experiment',str(exp),'--stage','report'],check=True)
        journal.update(stage='all_eight_grt_outcomes_audited',finished_unix=time.time());save()
    except BaseException as error:
        journal.update(last_error=repr(error),attention_required=True);save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    p.add_argument('--initial-training-job',required=True);args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Use Slurm')
    run(args)
