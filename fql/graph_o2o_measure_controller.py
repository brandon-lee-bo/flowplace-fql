"""Complete one fixed physical-label manifest with preemption-only recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .graph_o2o_archive import atomic_json,sha
from .graph_o2o_backend import ROOT,load_requests
from .graph_o2o_controller import RESTARTABLE,submit_unique,wait_job


def run(args):
    exp=Path(args.experiment).resolve();state_path=exp/'measurement_controller.json'
    journal=json.loads(state_path.read_text())if state_path.exists()else dict(
        version=1,config_sha256=sha(exp/'config.json'),seed=args.seed,round=args.round,
        export_job=args.initial_export_job,backend_storage=args.storage,concurrency=args.concurrency,
        backend_jobs=[],started_unix=time.time())
    for key,value in [('config_sha256',sha(exp/'config.json')),('seed',args.seed),
                      ('round',args.round),('backend_storage',args.storage),('concurrency',args.concurrency)]:
        if journal[key]!=value:raise ValueError('Measurement controller context changed: '+key)
    restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
    source_key=os.environ['SLURM_JOB_ID']+(f'_restart_{restart}'if restart else'')
    snapshot=exp/'measurement_controller_sources'/source_key
    snapshot.mkdir(parents=True,exist_ok=False)
    for name in ['graph_o2o_measure_controller.py','graph_o2o_measure_controller.slurm',
                 'graph_o2o_controller.py','graph_o2o_backend.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    journal['controller_job_id']=os.environ['SLURM_JOB_ID']
    journal.setdefault('source_snapshots',{})[source_key]=dict(
        path=str(snapshot),sha256={p.name:sha(p)for p in snapshot.iterdir()if p.is_file()})
    save=lambda:atomic_json(state_path,journal)
    save()
    try:
        terminal=wait_job(journal['export_job'],journal,save)
        if terminal!=['COMPLETED']:
            raise RuntimeError('The exporter is terminal without completion: '+str(terminal))
        rows,manifest=load_requests(exp,args.seed,args.round)
        if journal.get('tasks_sha256',sha(manifest))!=sha(manifest):
            raise ValueError('Fixed physical-label manifest changed')
        journal['tasks_sha256']=sha(manifest);journal['requested']=len(rows);save()
        prefix='go2omeasure_'+sha(exp/'config.json')[:10]
        batches=journal['backend_jobs']
        if batches:
            terminal=wait_job(batches[-1]['job'],journal,save)
            if not all(s=='COMPLETED'or s in RESTARTABLE for s in terminal):
                raise RuntimeError('Previous measurement array needs evidence-based repair: '+str(terminal))
        while True:
            missing=[i for i,row in enumerate(rows)if not(Path(row['sample_dir'])/'outcome.json').exists()]
            journal['completed_outcomes']=len(rows)-len(missing);save()
            if not missing:break
            indices=missing[:1]if not batches else missing
            job=submit_unique(prefix+'_batch_'+str(len(batches)),
                ['--array='+','.join(map(str,indices))+'%'+str(args.concurrency)],
                [str(ROOT/'fql/graph_o2o_backend.slurm'),'--experiment',str(exp),
                 '--seed',str(args.seed),'--round',str(args.round),'--storage',args.storage])
            batches.append(dict(job=job,indices=indices,tasks_sha256=sha(manifest)));save()
            terminal=wait_job(job,journal,save)
            if not all(s=='COMPLETED'or s in RESTARTABLE for s in terminal):
                raise RuntimeError('Measurement array needs evidence-based repair: '+str(terminal))
        journal['stage']='all_physical_outcomes_present_auditing';save()
        # This Slurm controller is lightweight CPU work; the physical workers
        # have separate explicit eight-thread allocations.
        subprocess.run([sys.executable,'-m','fql.graph_o2o_offline_expand',
            '--experiment',str(exp),'--stage','report'],check=True)
        journal['stage']='all_requested_physical_outcomes_audited'
        journal['finished_unix']=time.time();save()
    except BaseException as error:
        journal['last_error']=repr(error);journal['attention_required']=True;save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True);p.add_argument('--initial-export-job',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--round',type=int,default=0)
    p.add_argument('--storage',choices=['bulk','scratch'],default='scratch')
    p.add_argument('--concurrency',type=int,default=4)
    args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ or args.concurrency<1:raise RuntimeError('Use Slurm and positive concurrency')
    run(args)
