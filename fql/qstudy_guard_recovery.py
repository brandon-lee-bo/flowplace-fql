"""Narrow recovery of the 25 independently diagnosed 1796007 guard failures.

The original evaluator and attempt records remain immutable.  A private copy of
its exact measure bytecode admits FAILED only for the original, SHA-bound jobs
whose interrupted attempts were audited in job1796838.  Real scheduler states
are retained; every other FAILED attempt still requires diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import FunctionType

# Import the local backend before upstream baseline paths can shadow analysis.
from . import qstudy_backend as backend
from .graph_o2o_archive import atomic_json, sha
from .qstudy_controller import validate_round_outcomes

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/main_native_requeue_diagnosis_v1/job1796838/report.json')
DIAGNOSIS_SHA = 'cdee1b354bf26364b02950fb0244ecd778ac345e9cad92a3a7f2faf9d5d6a642'
BACKEND_SHA = 'c88bbbfa21a0e816e1f645c91ed4f66deabe6d1f112f9f820786afeb93a43265'
INDICES = (0, 1, 10, 11, 13, 14, 15, 17, 18, 19, 20, 23, 24, 25, 28, 29,
           37, 41, 49, 59, 67, 70, 72, 78, 79)
SOURCE_FILES = ('fql/qstudy_guard_recovery.py', 'fql/qstudy_guard_recovery.slurm',
                'fql/qstudy_backend.py', 'fql/qstudy_controller.py')


def read(path):
    return json.loads(Path(path).read_text())


def bound(path, expected):
    if sha(path) != expected:
        raise ValueError('Recovery evidence fingerprint changed: ' + str(path))
    return Path(path)


def diagnosis():
    value = read(bound(DIAGNOSIS, DIAGNOSIS_SHA))
    bound(ROOT/'fql/qstudy_backend.py', BACKEND_SHA)
    if (value['schema'] != 'qstudy_main_native_requeue_diagnosis_v1'
            or value['slurm_job_id'] != '1796838'
            or value['declared'] != 96 or value['strictly_verified_existing'] != 71
            or tuple(value['missing_indices']) != INDICES
            or not value['all_failed_stdout_are_guard_only']
            or value['physical_tools_invoked'] != 0 or value['source_changed']):
        raise ValueError('Not the independently audited 25 guard-only failures')
    matrix = bound(value['matrix']['path'], value['matrix']['sha256'])
    tasks = [json.loads(line) for line in matrix.read_text().splitlines()]
    if len(tasks) != 96 or tuple(r['flat_index'] for r in value['failed_rows']) != INDICES:
        raise ValueError('Recovery declaration changed')
    for row in value['failed_rows']:
        if tasks[row['flat_index']] != row['task']:
            raise ValueError('Task differs from immutable original dispatch')
        validate_original(row)
    return value, tasks


def validate_original(row):
    """Small immutable evidence checks, repeated inside the original flock."""
    if not row['guard_only_failure'] or len(row['attempts']) != 1:
        raise ValueError('Recovery must have exactly one diagnosed interrupted attempt')
    log = bound(row['failed_stdout_path'], row['failed_stdout_sha256'])
    if 'Do not duplicate an unresolved/live physical attempt' not in log.read_text():
        raise ValueError('Original failure was not the diagnosed duplicate guard')
    sample = Path(row['task']['sample_dir'])
    task = row['task']
    stream = (Path(task['experiment'])/f"seed_{task['seed']}"/
              f"round_{task['round']}"/'requests/tasks.jsonl')
    bound(stream, task['tasks_sha256'])
    bound(sample/'request.json', row['request_sha256'])
    old = row['attempts'][0]
    meta = read(bound(old['attempt_path'], old['attempt_sha256']))
    if (meta != old['metadata'] or meta['array_job_id'] != '1796007'
            or int(meta['array_task_id']) != row['flat_index']
            or meta['restart_count'] != 1 or not str(meta['job_id']).isdecimal()
            or meta['request_sha256'] != row['request_sha256']
            or old['manifest_time_end'] or old['process_exit_exists']):
        raise ValueError('Original attempt is outside the narrow failure diagnosis')
    physical = Path(meta['physical_run'])
    native = read(bound(physical/'manifest.json', old['manifest_sha256']))
    if native.get('time_end'):
        raise ValueError('Original attempt unexpectedly became terminal; re-audit before recovery')
    if (Path(old['attempt_path']).parent/'process_exit.json').exists():
        raise ValueError('Original attempt exit evidence changed')
    if old['runner_log_sha256']:
        bound(Path(old['attempt_path']).parent/'runner.log', old['runner_log_sha256'])
    return meta


def checked_plan(path):
    value = read(path)
    if value.get('schema') != 'qstudy_guard_recovery_plan_v1':
        raise ValueError('Unknown recovery plan schema')
    if value['diagnosis'] != {'path': str(DIAGNOSIS), 'sha256': DIAGNOSIS_SHA}:
        raise ValueError('Wrong recovery diagnosis binding')
    if tuple(value['flat_indices']) != INDICES or value['original_array'] != '1796007':
        raise ValueError('Recovery is not restricted to the exact 25 original indices')
    if set(value['source_sha256']) != set(SOURCE_FILES):
        raise ValueError('Incomplete adapter source binding')
    for relative, digest in value['source_sha256'].items():
        bound(ROOT/relative, digest)
    evidence, tasks = diagnosis()
    if value['matrix'] != evidence['matrix']:
        raise ValueError('Recovery matrix changed')
    return value, evidence, tasks


def plan(args):
    evidence, tasks = diagnosis()
    if validate_round_outcomes(tasks) != list(INDICES):
        raise ValueError('Native missing set changed; do not redeclare or duplicate work')
    accounting = subprocess.run(['sacct', '-j', '1796007', '-X', '-n', '-P',
        '-o', 'JobIDRaw,State,ExitCode,Restarts'], text=True, capture_output=True, check=True).stdout
    records = {line.split('|')[0]: line.split('|') for line in accounting.splitlines() if line}
    for row in evidence['failed_rows']:
        job = row['attempts'][0]['metadata']['job_id']
        if records.get(job, [None, None, None, None])[1:4] != ['FAILED', '1:0', '1']:
            raise ValueError('Old job no longer has the diagnosed FAILED/restart1 accounting: '+job)
        if backend.states(job) != ['FAILED']:
            raise ValueError('Old job must be authoritatively FAILED with no live duplicate')
    output = args.plan.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        checked_plan(output)
        print(json.dumps({'status': 'existing_immutable_plan', 'plan': str(output), 'sha256': sha(output)}))
        return
    value = dict(schema='qstudy_guard_recovery_plan_v1', diagnosis={'path': str(DIAGNOSIS), 'sha256': DIAGNOSIS_SHA},
        matrix=evidence['matrix'], original_array='1796007', flat_indices=list(INDICES),
        source_sha256={relative: sha(ROOT/relative) for relative in SOURCE_FILES},
        accounting=accounting, created_job_id=os.environ['SLURM_JOB_ID'], created_unix=time.time(),
        physical_settings_unchanged=True, real_failed_scheduler_states_preserved=True,
        maximum_new_native_calls=25, concurrency=4, cpus=8, memory='24G',
        note='Only diagnosis-bound original FAILED jobs are admitted; new FAILED attempts remain fatal.')
    atomic_json(output, value)
    print(json.dumps({'status': 'planned', 'plan': str(output), 'sha256': sha(output), 'indices': list(INDICES)}))


def private_measure(original, state_guard, request_guard=None, loader_guard=None):
    """Isolate the single allow-list extension without mutating backend globals."""
    namespace = dict(original.__globals__)
    namespace['states'] = state_guard
    namespace['RESTARTABLE'] = frozenset(namespace['RESTARTABLE']) | {'FAILED'}
    if request_guard is not None:
        namespace['checked_request'] = request_guard
    if loader_guard is not None:
        namespace['load_requests'] = loader_guard
    result = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def target_guards(row, real_load=backend.load_requests, real_check=backend.checked_request):
    """Bind the actual loaded stream before even the no-attempt/early-return paths."""
    task = row['task']
    expected_sample = Path(task['sample_dir']).resolve()

    def load(experiment, seed, round_number):
        if (Path(experiment).resolve() != Path(task['experiment']).resolve()
                or seed != task['seed'] or round_number != task['round']):
            raise ValueError('Loaded recovery stream differs from its original dispatch')
        rows, path = real_load(experiment, seed, round_number)
        bound(path, task['tasks_sha256'])
        if (not 0 <= task['index'] < len(rows)
                or Path(rows[task['index']]['sample_dir']).resolve() != expected_sample):
            raise ValueError('Original recovery stream index now resolves to another sample')
        bound(expected_sample/'request.json', row['request_sha256'])
        return rows, path

    def check(actual_row, args, cfg, binding):
        sample = real_check(actual_row, args, cfg, binding)
        if (Path(sample).resolve() != expected_sample
                or args.seed != task['seed'] or args.round != task['round']):
            raise ValueError('Actual evaluator selected an unaudited recovery request')
        stream = (Path(task['experiment'])/f"seed_{task['seed']}"/
                  f"round_{task['round']}"/'requests/tasks.jsonl')
        bound(stream, task['tasks_sha256'])
        bound(expected_sample/'request.json', row['request_sha256'])
        return sample

    return load, check


def guarded_states(row, plan_path, real_states=backend.states, plan_sha256=None):
    """Called by the unmodified measure bytecode under its existing sample flock."""
    original_job = row['attempts'][0]['metadata']['job_id']

    def check(job):
        observed = real_states(job)
        if 'FAILED' not in observed:
            return observed  # Unchanged original LIVE/UNKNOWN/RESTARTABLE handling.
        if str(job) != original_job or observed != ['FAILED']:
            raise RuntimeError('Undiagnosed FAILED attempt is never restartable: '+str(job)+' '+str(observed))
        if plan_sha256 is not None:
            bound(plan_path, plan_sha256)
        value, evidence, tasks = checked_plan(plan_path)
        found = next(r for r in evidence['failed_rows'] if r['flat_index'] == row['flat_index'])
        if found != row:
            raise ValueError('Diagnosis row changed inside the physical-attempt lock')
        meta = validate_original(row)
        admission = Path(row['task']['sample_dir'])/'guard_failure_recovery'
        admission.mkdir(exist_ok=True)
        identity = os.environ['SLURM_JOB_ID']+'_restart_'+os.environ.get('SLURM_RESTART_COUNT', '0')
        dest = admission/(identity+'.json')
        record = dict(schema='qstudy_guard_failed_admission_v1', plan_path=str(plan_path), plan_sha256=sha(plan_path),
            flat_index=row['flat_index'], previous_job_id=original_job, actual_observed_scheduler_states=observed,
            previous_attempt_sha256=row['attempts'][0]['attempt_sha256'], previous_restart_count=meta['restart_count'],
            adapter_source_sha256=sha(Path(__file__)), current_job_id=os.environ['SLURM_JOB_ID'],
            admitted_under_original_sample_flock=True, created_unix=time.time())
        if dest.exists():
            prior = read(dest)
            for key in record.keys()-{'created_unix'}:
                if prior.get(key) != record[key]:
                    raise ValueError('Existing recovery admission changed')
        else:
            atomic_json(dest, record)
        return observed  # FAILED remains FAILED; only the private set admits it.

    return check


def worker(args):
    if not args.plan_sha256:
        raise ValueError('Dispatch requires the immutable recovery plan SHA')
    bound(args.plan, args.plan_sha256)
    value, evidence, tasks = checked_plan(args.plan)
    index = args.flat_index if args.flat_index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if index not in INDICES:
        raise IndexError('Recovery can only dispatch the 25 diagnosed flat indices')
    task = tasks[index]
    row = next(r for r in evidence['failed_rows'] if r['flat_index'] == index)
    arguments = argparse.Namespace(experiment=task['experiment'], seed=task['seed'], round=task['round'],
        index=task['index'], storage='scratch')
    load, check = target_guards(row)
    measure = private_measure(backend.measure, guarded_states(row, args.plan, plan_sha256=args.plan_sha256),
        request_guard=check, loader_guard=load)
    if measure.__code__ is not backend.measure.__code__:
        raise AssertionError('Frozen physical evaluator bytecode changed')
    measure(arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'worker', 'validate'], required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--flat-index', type=int)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run recovery and validation through Slurm')
    if args.stage == 'plan':
        plan(args)
    elif args.stage == 'validate':
        if not args.plan_sha256:
            raise ValueError('Validation requires the immutable recovery plan SHA')
        bound(args.plan, args.plan_sha256)
        _, _, tasks = checked_plan(args.plan)
        print(json.dumps({'missing_indices': validate_round_outcomes(tasks), 'plan_sha256': sha(args.plan)}))
    else:
        if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 8:
            raise RuntimeError('The unchanged physical evaluator requires exactly eight CPUs')
        worker(args)


if __name__ == '__main__':
    main()
