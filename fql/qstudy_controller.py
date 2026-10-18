"""Bounded Slurm orchestration for matched actor runs and real online feedback.

Only declared experiments are touched. Completed outputs are never repeated;
preempted jobs resume checkpoints. Ordinary failed jobs leave diagnostic state
for the active research agent, rather than being relabeled physical failures.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_controller import states, LIVE, RESTARTABLE

ROOT = Path(__file__).resolve().parents[1]


def sbatch(arguments, unique_name):
    # Recover a submission whose controller was preempted before journaling its ID.
    queued = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i|%j'],
                            text=True, capture_output=True, check=True)
    matches = {line.split('|')[0].split('_')[0] for line in queued.stdout.splitlines()
               if '|' in line and line.split('|')[1].strip() == unique_name}
    if len(matches) > 1:
        raise RuntimeError('Duplicate live jobs have the same stage identity')
    if matches:
        return matches.pop()
    accounted = subprocess.run(['sacct', '-u', os.environ['USER'], '--name='+unique_name,
        '-S', 'now-2days', '-X', '-n', '-P', '-o', 'JobIDRaw,JobName%200'],
        text=True, capture_output=True, check=True)
    matches = {line.split('|')[0].split('_')[0] for line in accounted.stdout.splitlines()
               if '|' in line and line.split('|')[1].strip() == unique_name}
    if len(matches) > 1:
        raise RuntimeError('Multiple historical jobs share one submission identity')
    if matches:
        return matches.pop()
    result = subprocess.run(['sbatch', '--parsable', '--job-name='+unique_name, *arguments], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Submission refused: '+result.stderr)
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdecimal():
        raise RuntimeError('Submission returned a nonnumeric job identity')
    return job_id


def atomic_matrix(path, text):
    """Publish a complete immutable dispatch matrix, surviving interrupted writes."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Combined backend task identities changed')
        return
    temporary = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_round_outcomes(tasks):
    """Validate native evidence before admitting feedback into the next round.

    The controller may read existing geometry audits, but must never launch an
    eight-thread OpenROAD geometry audit inside its one-CPU allocation.
    """
    from . import qstudy_backend as backend
    missing, contexts, requests = [], {}, {}
    for index, task in enumerate(tasks):
        experiment, seed, round_number = task['experiment'], task['seed'], task['round']
        if experiment not in contexts:
            contexts[experiment] = backend.context(experiment)
        key = (experiment, seed, round_number)
        if key not in requests:
            requests[key] = load_requests(experiment, seed, round_number)
        rows, original = requests[key]
        if (sha(original) != task['tasks_sha256'] or not 0 <= task['index'] < len(rows)
                or rows[task['index']]['sample_dir'] != task['sample_dir']):
            raise ValueError('Round outcome no longer binds its dispatch identity')
        cfg, verification, binding = contexts[experiment]
        row = rows[task['index']]
        args = argparse.Namespace(seed=seed, round=round_number)
        sample = backend.checked_request(row, args, cfg, binding)
        outcome = sample/'outcome.json'
        if not outcome.exists():
            missing.append(index)
            continue
        out = json.loads(outcome.read_text())
        for name in ['seed', 'round', 'arm', 'role', 'index', 'graph', 'training_feedback_eligible']:
            if out.get(name) != row.get(name):
                raise ValueError('Round outcome changed request role/identity: '+name)
        if out.get('physical_run'):
            run = Path(out['physical_run'])
            manifest = json.loads((run/'manifest.json').read_text())
            if manifest.get('success') and not (run/'backend_macro_audit.json').is_file():
                raise ValueError('Native geometry audit missing; use the eight-CPU backend worker')
        backend.checked_existing(out, row, sample, original, binding, cfg, verification)
    return missing


def state_of(job):
    result = states(str(job))
    return result or ['UNKNOWN']


def recoverable(job):
    status = state_of(job)
    if any(s in LIVE or s == 'ACCOUNTING_PENDING' for s in status):
        return False
    if all(s in RESTARTABLE | {'COMPLETED', 'TIMEOUT'} for s in status) and any(s != 'COMPLETED' for s in status):
        return True
    raise RuntimeError('Worker needs diagnosis before restart: '+str(job)+' '+str(status))


def controller(args):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run even the controller through Slurm')
    if args.rounds < 0 or args.gpu_concurrency < 1 or args.backend_concurrency < 1:
        raise ValueError('Rounds must be nonnegative and worker concurrency positive')
    resolved = [Path(p).resolve() for p in args.experiments]
    if len(set(resolved)) != len(resolved) or len({p.name for p in resolved}) != len(resolved):
        raise ValueError('Experiment paths and stream basenames must be unique')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plans = [(Path(p).resolve(), json.loads((Path(p)/'config.json').read_text())) for p in args.experiments]
        for _, cfg in plans:
            seeds = cfg['seeds']
            if (not seeds or any(type(s) is not int or s < 0 for s in seeds)
                    or len(set(seeds)) != len(seeds)):
                raise ValueError('Training seeds must be distinct nonnegative integers')
            if cfg.get('online_rounds', args.rounds) != args.rounds:
                raise ValueError('Controller rounds differ from immutable actor configuration')
        binding = dict(experiments={str(p):sha(p/'config.json') for p,_ in plans},
            rounds=args.rounds, gpu_concurrency=args.gpu_concurrency, backend_concurrency=args.backend_concurrency,
            source_sha256={n:sha(ROOT/n) for n in ['fql/qstudy_actor.py', 'fql/qstudy_models.py',
                'fql/qstudy_backend.py', 'fql/qstudy_controller.py', 'fql/qstudy_actor.slurm',
                'fql/qstudy_backend_batch.slurm']})
        path = output/'state.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['binding'] != binding:
                raise ValueError('Controller source or declared experiment changed')
        else:
            state = dict(binding=binding, gpu_jobs={}, backend_jobs={}, status='running', event_log=[])
        def save(event=None):
            state['controller_job_id'] = os.environ['SLURM_JOB_ID']
            state['last_update_unix'] = time.time()
            if event:
                state['event_log'].append(dict(time=time.time(), **event))
            atomic_json(path, state)
        save()
        study_id = hashlib.sha256(str(output).encode()).hexdigest()[:8]
        for round_number in range(args.rounds+1):
            streams = [(p, seed) for p,cfg in plans for seed in cfg['seeds']]
            while True:
                missing = [(p,s) for p,s in streams if not
                    (p/f'seed_{s}'/f'round_{round_number}'/'requests/completed.json').exists()]
                if not missing:
                    break
                live = sum(any(v in LIVE or v == 'ACCOUNTING_PENDING' for v in state_of(jobs[-1]))
                           for jobs in state['gpu_jobs'].values() if jobs)
                for p,seed in missing:
                    key = f'{p.name}_s{seed}_r{round_number}'
                    jobs = state['gpu_jobs'].setdefault(key, [])
                    if jobs and not recoverable(jobs[-1]):
                        continue
                    if live >= args.gpu_concurrency:
                        break
                    job = sbatch([str(ROOT/'fql/qstudy_actor.slurm'),
                        '--experiment',str(p),'--seed',str(seed),'--round',str(round_number),'--stage','all'],
                        f'qs_{study_id}_{key}_a{len(jobs)}')
                    jobs.append(job); live += 1
                    save(dict(event='submitted_actor',key=key,job_id=job))
                time.sleep(20)
            matrix = output/f'backend_round_{round_number}.jsonl'
            tasks = []
            for p,seed in streams:
                requests, original = load_requests(p,seed,round_number)
                for index,row in enumerate(requests):
                    tasks.append(dict(experiment=str(p),seed=seed,round=round_number,index=index,
                        sample_dir=row['sample_dir'],tasks_sha256=sha(original)))
            text = ''.join(json.dumps(t)+'\n' for t in tasks)
            atomic_matrix(matrix, text)
            while True:
                missing = [i for i,t in enumerate(tasks) if not (Path(t['sample_dir'])/'outcome.json').exists()]
                if not missing:
                    if validate_round_outcomes(tasks):
                        raise RuntimeError('Backend outcomes disappeared during native validation')
                    break
                key = str(round_number)
                jobs = state['backend_jobs'].setdefault(key, [])
                if not jobs or recoverable(jobs[-1]):
                    indices = ','.join(map(str,missing))+f'%{args.backend_concurrency}'
                    job = sbatch(['--array='+indices,str(ROOT/'fql/qstudy_backend_batch.slurm'),
                                  '--dispatch',str(matrix),'--dispatch-sha256',sha(matrix)],
                                  f'qs_{study_id}_backend_r{round_number}_a{len(jobs)}')
                    jobs.append(job)
                    save(dict(event='submitted_backend',round=round_number,job_id=job,missing=len(missing)))
                time.sleep(20)
            save(dict(event='round_completed',round=round_number,count=len(tasks)))
        state['status'] = 'completed'
        save(dict(event='all_rounds_completed'))


def dispatch(args):
    if 'SLURM_JOB_ID' not in os.environ or sha(args.dispatch) != args.dispatch_sha256:
        raise ValueError('Run the immutable backend matrix via Slurm')
    tasks = [json.loads(line) for line in args.dispatch.read_text().splitlines()]
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(tasks):
        raise IndexError('Slurm task is outside the immutable dispatch matrix')
    row = tasks[index]
    requests, original = load_requests(row['experiment'],row['seed'],row['round'])
    if (sha(original) != row['tasks_sha256'] or not 0 <= row['index'] < len(requests)
            or requests[row['index']]['sample_dir'] != row['sample_dir']):
        raise ValueError('Combined/backend task identities changed')
    from .qstudy_backend import measure
    measure(argparse.Namespace(experiment=row['experiment'],seed=row['seed'],round=row['round'],
                              index=row['index'],storage='scratch'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments', nargs='+')
    p.add_argument('--output', type=Path)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--gpu-concurrency', type=int, default=2)
    p.add_argument('--backend-concurrency', type=int, default=4)
    p.add_argument('--dispatch', type=Path)
    p.add_argument('--dispatch-sha256')
    args = p.parse_args()
    if args.dispatch:
        dispatch(args)
    else:
        if not args.experiments or args.output is None:
            p.error('--experiments and --output are required for controller')
        controller(args)


if __name__ == '__main__':
    main()
