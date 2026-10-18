"""Recover only interrupted, missing native labels from one sealed export.

The original array and declared external attempts finish before dispatch.
Every retry uses the pinned eight-thread backend and unchanged request/DEF.
Completed native measurements are recovered by that backend, never blindly
repeated. Ordinary code failures and unknown accounting states are not labels.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import time

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_controller import LIVE, states
from .qstudy_controller import sbatch, validate_round_outcomes

ROOT = Path(__file__).resolve().parents[1]
RETRY = {'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'TIMEOUT'}
EXCLUDE = 'natural-selection,enterprise,blue-space'
SAFE_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,24}\Z')


def disposition(observations):
    """All known attempts must be terminal, interrupted, and unambiguous."""
    flat = [state for observed in observations for state in observed]
    if not flat or any(state in LIVE or state in {'ACCOUNTING_PENDING', 'UNKNOWN'} for state in flat):
        return 'wait'
    if all(state in RETRY for state in flat):
        return 'retry'
    raise RuntimeError('Missing outcome requires diagnosis, never blind repeat: '+str(flat))


def parse_known(values, count):
    result = {}
    for item in values:
        index, job = item.split('=', 1)
        if not index.isdecimal() or not job.isdecimal() or not 0 <= int(index) < count:
            raise ValueError('Known attempt must be request_index=numeric_job_id')
        result.setdefault(str(int(index)), []).append(job)
    return result


def source_binding(args, rows, tasks, known):
    cfg = json.loads((args.experiment/'config.json').read_text())
    if cfg['design'] not in {'bp_fe', 'bp_be'} or len(rows) not in {52, 80}:
        raise ValueError('This recovery scope is the sealed teacher80/probe52 exports only')
    if any(row['role'] not in {'diagnostic', 'data_train', 'data_validation'} or
           row['training_feedback_eligible'] or row['seed'] != 0 or row['round'] != 0 for row in rows):
        raise ValueError('Recovery cannot operate on actor/controller or online replay requests')
    return dict(experiment=str(args.experiment), original_array=args.original_array,
        known_attempts=known, config_sha256=sha(args.experiment/'config.json'),
        tasks_sha256=sha(tasks), completion_sha256=sha(tasks.with_name('completed.json')),
        requests=len(rows), run_ids=[row['run_id'] for row in rows],
        name=args.name, concurrency=args.concurrency, exclude=EXCLUDE,
        physical_threads=8, worker_memory='24G', worker_time='02:00:00',
        partition='cpu-scavenger', qos='cpu-scavenger',
        source_sha256={name: sha(ROOT/name) for name in ('fql/qstudy_recover.py', 'fql/qstudy_recover.slurm',
            'fql/qstudy_backend.py', 'fql/qstudy_backend.slurm', 'fql/qstudy_controller.py',
            'fql/graph_o2o_controller.py')})


def guarded_validate(rows, tasks, experiment, validator=validate_round_outcomes):
    """Use native validators without allowing missing-audit OpenROAD execution."""
    dispatch = []
    for index, row in enumerate(rows):
        sample = Path(row['sample_dir'])
        outcome = sample/'outcome.json'
        if not outcome.exists():
            raise ValueError('Native completion requires all predeclared outcomes')
        out = json.loads(outcome.read_text())
        if out.get('physical_run'):
            run = Path(out['physical_run'])
            manifest = json.loads((run/'manifest.json').read_text())
            if manifest.get('success') and not (run/'backend_macro_audit.json').is_file():
                raise ValueError('Saved native geometry audit missing; never run OpenROAD in recovery controller')
        dispatch.append(dict(experiment=str(experiment), seed=0, round=0, index=index,
                             sample_dir=str(sample), tasks_sha256=sha(tasks)))
    if validator(dispatch):
        raise ValueError('Native validator still reports missing outcomes')
    return [dict(index=index, path=str(Path(row['sample_dir'])/'outcome.json'),
                 sha256=sha(Path(row['sample_dir'])/'outcome.json')) for index, row in enumerate(rows)]


def controller(args):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run native recovery orchestration through Slurm')
    if not SAFE_NAME.fullmatch(args.name) or not args.original_array.isdecimal() or not 1 <= args.concurrency <= 2:
        raise ValueError('Invalid bounded recovery identity/concurrency')
    args.experiment = args.experiment.resolve()
    args.output = args.output.resolve()
    if args.output == args.experiment or args.output in args.experiment.parents:
        raise ValueError('Recovery journal cannot overwrite the source experiment')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, tasks = load_requests(args.experiment, 0, 0)
        known = parse_known(args.known_attempt, len(rows))
        binding = source_binding(args, rows, tasks, known)
        path = args.output/'state.json'
        state = json.loads(path.read_text()) if path.exists() else dict(
            binding=binding, status='running', jobs={}, events=[], started_unix=time.time())
        if state['binding'] != binding:
            raise ValueError('Recovery inputs, source, or scheduling contract changed')

        def save(event=None):
            state['controller_job_id'] = os.environ['SLURM_JOB_ID']
            state['last_update_unix'] = time.time()
            if event is not None:
                event['unix'] = time.time()
                state['events'].append(event)
                print(json.dumps(event), flush=True)
            atomic_json(path, state)

        try:
            while True:
                # The original array is authoritative; don't race its live work.
                original = states(args.original_array)
                state['original_states'] = original
                if any(value in LIVE or value == 'ACCOUNTING_PENDING' for value in original):
                    save()
                    time.sleep(45)
                    continue
                missing = [index for index, row in enumerate(rows)
                           if not (Path(row['sample_dir'])/'outcome.json').exists()]
                state['missing_indices'] = missing
                state['completed_outcomes'] = len(rows)-len(missing)
                if not missing:
                    outcomes = guarded_validate(rows, tasks, args.experiment)
                    completed = dict(status='completed', binding=binding, outcomes=outcomes,
                        jobs=state['jobs'], slurm_job_id=os.environ['SLURM_JOB_ID'],
                        finished_unix=time.time(), actor_jobs_touched=False,
                        numerical_settings_changed=False, heldout_used=False)
                    atomic_json(args.output/'completed.json', completed)
                    state['status'] = 'completed'
                    save(dict(event='all_native_outcomes_verified', count=len(outcomes)))
                    return
                busy, eligible = 0, []
                observations = {}
                for index in missing:
                    key = str(index)
                    jobs = [args.original_array+'_'+key]+known.get(key, [])+state['jobs'].get(key, [])
                    observed = [states(job) for job in jobs]
                    observations[key] = dict(jobs=jobs, states=observed)
                    decision = disposition(observed)
                    if decision == 'wait':
                        busy += 1
                    else:
                        # TIMEOUT is explicitly recognized but the pinned worker
                        # admits PREEMPTED/NODE_FAIL/BOOT_FAIL only. Surface that
                        # real limitation without silently editing frozen code.
                        if any('TIMEOUT' in value for value in observed):
                            raise RuntimeError('TIMEOUT needs an audited worker extension; frozen backend excludes it')
                        eligible.append(index)
                state['observations'] = observations
                slots = max(0, args.concurrency-busy)
                for index in eligible[:slots]:
                    key = str(index)
                    attempt = len(state['jobs'].get(key, []))
                    job = sbatch(['--exclude='+EXCLUDE, 'fql/qstudy_backend.slurm',
                        '--experiment', str(args.experiment), '--seed', '0', '--round', '0', '--index', key],
                        f'qsr_{args.name}_i{index}_a{attempt}')
                    state['jobs'].setdefault(key, []).append(job)
                    save(dict(event='submitted_missing_only', index=index, job=job, attempt=attempt,
                              prior_observation=observations[key]))
                save()
                time.sleep(45)
        except Exception as error:
            state['status'] = 'diagnosis_required'
            state['error'] = repr(error)
            save(dict(event='diagnosis_required', error=repr(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--original-array', required=True)
    parser.add_argument('--known-attempt', action='append', default=[])
    parser.add_argument('--concurrency', type=int, default=2)
    controller(parser.parse_args())


if __name__ == '__main__':
    main()
