"""Restartable Slurm controller for one offline-to-online graph FQL pilot.

Only this experiment's jobs are observed/submitted. Failed code is surfaced for
repair; confirmed preemption can resume checkpoints. Old Table 2 stays paused.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_design_collect import collect
from .graph_o2o_design_policy_data import measured_context

LIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
RESTARTABLE = {'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL'}


def states(job):
    # squeue is authoritative for a live allocation; accounting may lag.
    p = subprocess.run(['squeue', '-j', str(job), '-h', '-o', '%i|%T'],
                       capture_output=True, text=True)
    queued = [s.split('|')[1].strip().split()[0] for s in p.stdout.splitlines() if '|' in s]
    if any(s in LIVE for s in queued):
        return queued
    p = subprocess.run(['sacct', '-j', str(job), '-X', '-n', '-P', '-o', 'JobID,State,ExitCode'],
                       capture_output=True, text=True, check=True)
    result = [line.split('|')[1].split()[0].rstrip('+') for line in p.stdout.splitlines() if '|' in line]
    return result or ['ACCOUNTING_PENDING']


def wait_job(job, journal, save):
    while True:
        try:
            observed = states(job)
        except (OSError, subprocess.SubprocessError) as error:
            journal['last_observation_error'] = repr(error)
            save()
            time.sleep(15)
            continue
        journal['last_observation'] = dict(job=job, states=observed, unix=time.time())
        save()
        if any(s in LIVE or s=='ACCOUNTING_PENDING' for s in observed):
            time.sleep(15)
            continue
        return observed


def submit_unique(name, options, command):
    p = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i|%j'],
                       capture_output=True, text=True, check=True)
    matches = [line.split('|')[0] for line in p.stdout.splitlines() if line.split('|')[-1]==name]
    # Recover a submission if the controller was preempted before journaling it.
    if matches:
        roots = {value.split('_')[0] for value in matches}
        if len(roots) != 1:
            raise ValueError('Multiple live jobs with this unique stage name')
        return roots.pop()
    p = subprocess.run(['sbatch', '--parsable', '--job-name='+name, *options, *command],
                       capture_output=True, text=True, check=True)
    result = p.stdout.strip().split(';')[0]
    if not result.isdigit():
        raise ValueError('Unexpected sbatch response: '+p.stdout)
    print(json.dumps(dict(event='submitted', name=name, job=result)), flush=True)
    return result


def run(args):
    experiment = Path(args.experiment).resolve()
    cfg_path = experiment/'config.json'
    cfg, binding, _ = measured_context(experiment)
    path = experiment/f'controller_seed_{args.seed}.json'
    journal = json.loads(path.read_text()) if path.exists() else dict(version=1, seed=args.seed,
        config_sha256=sha(cfg_path), training_jobs={'0': [str(args.initial_training_job)]},
        backend_storage=args.backend_storage,
        backend_jobs={}, started_utc=datetime.now(timezone.utc).isoformat(), old_table2_resumed=False)
    if journal['config_sha256'] != sha(cfg_path):
        raise ValueError('Configuration differs from the journal')
    if journal.get('backend_storage', 'bulk') != args.backend_storage:
        raise ValueError('Backend storage policy differs from the journal')
    journal['backend_storage'] = args.backend_storage
    journal['controller_job_id'] = os.environ['SLURM_JOB_ID']
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    source_key = os.environ['SLURM_JOB_ID'] + (f'_restart_{restart}' if restart else '')
    snapshot = experiment/'controller_sources'/source_key
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ['graph_o2o_design_controller.py', 'graph_o2o_design_collect.py', 'graph_o2o_design_backend.py', 'graph_o2o_design_policy_data.py', 'graph_o2o_archive.py']:
        shutil.copyfile(Path(__file__).with_name(name), snapshot/name)
    journal.setdefault('controller_startups', {})[source_key] = dict(
        source_directory=str(snapshot), restart_count=restart,
        files={p.name: sha(p) for p in snapshot.iterdir() if p.is_file()})
    save = lambda: atomic_json(path, journal)
    save()
    prefix = 'go2o_'+sha(cfg_path)[:8]+'_s'+str(args.seed)
    try:
        for number in range(cfg['online_rounds']+1):
            journal['stage'] = 'training_round_'+str(number)
            jobs = journal['training_jobs'].setdefault(str(number), [])
            while True:
                if not jobs:
                    job = submit_unique(prefix+f'_r{number}_train_{len(jobs)}',
                        ['--partition=gpu-scavenger', '--qos=gpu-scavenger'],
                        [str(ROOT/'fql/graph_o2o_design_train.slurm'), '--experiment', str(experiment),
                         '--seed', str(args.seed), '--round', str(number)])
                    jobs.append(job); save()
                terminal = wait_job(jobs[-1], journal, save)
                if all(state=='COMPLETED' for state in terminal):
                    break
                if all(state in RESTARTABLE or state=='COMPLETED' for state in terminal):
                    job = submit_unique(prefix+f'_r{number}_train_{len(jobs)}',
                        ['--partition=gpu-scavenger', '--qos=gpu-scavenger'],
                        [str(ROOT/'fql/graph_o2o_design_train.slurm'), '--experiment', str(experiment),
                         '--seed', str(args.seed), '--round', str(number)])
                    jobs.append(job); save()
                else:
                    raise RuntimeError(f'Training {jobs[-1]} terminal {terminal}; inspect and repair, preserve checkpoint')
            rows, task_path = load_requests(experiment, args.seed, number)
            journal['stage'] = 'physical_collection_round_'+str(number)
            batches = journal['backend_jobs'].setdefault(str(number), [])
            save()
            if batches:
                terminal = wait_job(batches[-1]['job'], journal, save)
                if not all(state=='COMPLETED' or state in RESTARTABLE for state in terminal):
                    raise RuntimeError(f'Backend {batches[-1]["job"]} terminal {terminal}; inspect missing outcomes')
            while True:
                missing = [i for i, row in enumerate(rows) if not (Path(row['sample_dir'])/'outcome.json').exists()]
                if not missing:
                    break
                # Validate the real evaluator and archive on one candidate first.
                indices = [missing[0]] if not batches and number==0 else missing
                option = '--array='+','.join(map(str, indices))+'%'+str(cfg['backend_concurrency'])
                job = submit_unique(prefix+f'_r{number}_grt_{len(batches)}', [option],
                    [str(ROOT/'fql/graph_o2o_design_backend.slurm'), '--experiment', str(experiment),
                     '--seed', str(args.seed), '--round', str(number), '--storage', args.backend_storage])
                batches.append(dict(job=job, indices=indices, tasks_sha256=sha(task_path))); save()
                terminal = wait_job(job, journal, save)
                if not all(state=='COMPLETED' or state in RESTARTABLE for state in terminal):
                    raise RuntimeError(f'Backend {job} terminal {terminal}; inspect and repair before retry')
            journal['stage'] = 'replay_audit_round_'+str(number)
            save()
            report = collect(experiment, args.seed, number)
            journal.setdefault('round_reports', {})[str(number)] = report['paired']
            save()
        journal['stage'] = 'pilot_completed_requires_full_route_and_replication'
        journal['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(dict(status=journal['stage'], reports=journal['round_reports'])), flush=True)
    except BaseException as error:
        journal['last_error'] = repr(error)
        journal['attention_required'] = True
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--initial-training-job', required=True)
    parser.add_argument('--backend-storage', choices=['scratch'], default='scratch')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Submit the controller through Slurm')
    run(args)


if __name__ == '__main__':
    main()
