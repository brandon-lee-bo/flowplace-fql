"""Bounded recovery of the existing64 TRAIN detailed-route requests only.

The two previously declared arrays finish first. Only a missing label with
confirmed infrastructure interruption may enter at most three new arrays.
The original collector and physical worker remain unchanged; no failure labels
or replacement layouts are manufactured by this scheduling adapter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from . import qstudy_train_final_route as collector
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_controller import LIVE, RESTARTABLE

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = '18b109579d8076ee7f65afb84481b7c270a7edc101fdbb67d28216802c7c101f'
ORIGINAL = '1798635'
DEV012 = '1798748'
VERSION = 'qstudy_train_drt64_infrastructure_recovery_v1'
EXCLUDE = 'natural-selection,enterprise,blue-space'
TERMINAL = RESTARTABLE | {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'DEADLINE', 'REVOKED'}
SOURCE_NAMES = ('fql/qstudy_train_drt_recovery.py', 'fql/qstudy_train_drt_recovery.slurm',
                *collector.SOURCE_NAMES)
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {name: sha(ROOT/name) for name in SOURCE_NAMES}


def normalized_state(value):
    return value.strip().split()[0].rstrip('+')


def parse_accounting(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split('|')
        if len(fields) < 5 or not fields[0] or not fields[1].isdigit():
            raise ValueError('Malformed exact-job Slurm accounting row')
        job, raw, state, exit_code, name = fields[:5]
        state = normalized_state(state)
        compressed = re.fullmatch(r'(\d+)_\[((?:\d+(?:-\d+)?)(?:,\d+(?:-\d+)?)*)?(?:%(\d+))?\]',job)
        if compressed:
            root, ranges, throttle = compressed.groups()
            maximum = 2 if root == DEV012 else 63
            if (not ranges or raw != root or state not in LIVE or
                    (throttle is not None and not 1 <= int(throttle) <= 2)):
                raise ValueError('Ambiguous compressed array accounting record')
            for part in ranges.split(','):
                endpoints = [int(x) for x in part.split('-')]
                if not 0 <= endpoints[0] <= endpoints[-1] <= maximum:
                    raise ValueError('Compressed array indices outside the fixed request scope')
        elif not re.fullmatch(r'\d+(?:_\d+)?', job):
            raise ValueError('Unexpected allocation/step identifier from sacct -X')
        rows.append(dict(job_id=job, job_id_raw=raw, state=state,
                         exit_code=exit_code, name=name))
    return rows


def matches(row, reference, parent=False):
    return (row['job_id'] == reference or row['job_id_raw'] == reference or
            (parent and row['job_id'].split('_', 1)[0] == reference))


def observed(snapshot, reference, parent=False):
    queue = [r for r in snapshot['queue'] if r['job_id'] == reference or
             (parent and r['job_id'].split('_', 1)[0] == reference)]
    if queue:
        return [r['state'] for r in queue]
    return [r['state'] for r in snapshot['accounting'] if matches(r, reference, parent)] or ['ACCOUNTING_PENDING']


def terminal(values):
    return bool(values) and all(value in TERMINAL for value in values)


class Scheduler:
    def command(self, command):
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)

    def snapshot(self, jobs):
        jobs = sorted(set(str(job) for job in jobs))
        if not jobs or any(not re.fullmatch(r'\d+(?:_\d+)?', job) for job in jobs):
            raise ValueError('Only exact declared job IDs may be queried')
        joined = ','.join(jobs)
        queued = self.command(['squeue', '-j', joined, '-h', '-o', '%i|%T|%j'])
        queue = []
        for line in queued.stdout.splitlines():
            fields = line.split('|')
            if len(fields) != 3:
                raise ValueError('Malformed squeue observation')
            queue.append(dict(job_id=fields[0], state=normalized_state(fields[1]), name=fields[2]))
        accounted = self.command(['sacct', '-j', joined, '-X', '--duplicates', '-n', '-P',
            '-o', 'JobID%64,JobIDRaw,State,ExitCode,JobName%200'])
        return dict(jobs=jobs, queue=queue, accounting=parse_accounting(accounted.stdout),
                    captured_unix=time.time(), squeue_stdout=queued.stdout, sacct_stdout=accounted.stdout)

    def find_name(self, name, created_unix):
        queued = self.command(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i|%j'])
        roots = {line.split('|')[0].split('_', 1)[0] for line in queued.stdout.splitlines()
                 if '|' in line and line.split('|')[1].strip() == name}
        start = datetime.fromtimestamp(created_unix-60,timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        accounted = self.command(['sacct', '-u', os.environ['USER'], '--name='+name, '--starttime='+start, '-X',
            '--duplicates', '-n', '-P', '-o', 'JobID%64,JobIDRaw,State,ExitCode,JobName%200'])
        rows = parse_accounting(accounted.stdout)
        roots.update(r['job_id'].split('_', 1)[0] for r in rows if r['name'] == name)
        if len(roots) > 1 or any(not root.isdigit() for root in roots):
            raise ValueError('Ambiguous unique recovery job name; refuse duplicate dispatch')
        return next(iter(roots), None), dict(squeue_stdout=queued.stdout, sacct_stdout=accounted.stdout)

    def submit(self, command):
        env = dict(os.environ)
        env.pop('SLURM_RESTART_COUNT', None)
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Ambiguous sbatch response; inspect unique name before retry')
        return job, dict(command=command, stdout=result.stdout, stderr=result.stderr,
                         returncode=result.returncode, inherited_restart_count_removed=True)


def binding(plan_path, expected, output):
    if expected != PLAN_SHA:
        raise ValueError('Only the original fixed64 declaration is authorized')
    plan = collector.checked_plan(plan_path, expected)
    output = Path(output).resolve()
    if output != Path(plan['output'])/'recovery_v1':
        raise ValueError('Use the single declared recovery_v1 journal to prevent competing coordinators')
    # Explicit independent denominator guard, beyond the frozen collector check.
    families = {}
    for index, task in enumerate(plan['tasks']):
        if task['index'] != index or task['provenance']['split'] != 'train':
            raise ValueError('Recovery cannot change TRAIN identity or request order')
        families.setdefault(task['family'], []).append(task['variant'])
        if collector.native_source(task['provenance']) != task['native']:
            raise ValueError('Original measured TRAIN source changed')
    if len(plan['tasks']) != 64 or len(families) != 32 or any(sorted(v) != [0, 1] for v in families.values()):
        raise ValueError('Recovery must retain32 families and both original variants')
    return plan, dict(version=VERSION, plan=collector.bind(plan_path, expected), output=str(output),
        original_array=ORIGINAL, existing_dev012_array=DEV012,
        concurrency=2, max_generations=3, partition='cpu-scavenger', qos='cpu-scavenger',
        worker_cpus=8, worker_memory='12G', worker_time='02:30:00', route_cap=20,
        exclusions=EXCLUDE, source_sha256=source_hashes(),
        source_identity_sha256=collector.object_sha([dict(index=t['index'], key=t['key'],
            family=t['family'], variant=t['variant'], provenance_sha256=t['provenance_sha256'],
            native=t['native']) for t in plan['tasks']]),
        new_layouts=0, failure_replacement=False, old_policy_evaluation_used=False)


def attempts(task, source_sha256=None):
    folder = Path(task['output'])
    result = collector.attempt_evidence(folder)
    spec = folder/'spec.json'
    source_names = ('graph_o2o_full_route.py', 'graph_o2o_full_route.slurm',
                    'graph_o2o_finish.py', 'graph_o2o_finish.sh', 'graph_o2o_archive.py')
    source_sha256 = source_sha256 or collector.sources()
    expected = {name: source_sha256['fql/'+name] for name in source_names}
    for row in result:
        path = Path(row['binding']['path'])
        value = read(path)
        if (not row['slurm_job_id'].isdigit() or path.parent.name != 'attempt_'+row['slurm_job_id'] or
                not spec.is_file() or value['spec_sha256'] != sha(spec)):
            raise ValueError('Unbound native attempt identity/specification')
        declaration = read(spec)
        allowed = (Path(task['native']['run']).resolve(),
                   (folder.parents[1]/'restored_sources'/task['key']).resolve())
        if (Path(declaration['source_run']).resolve() not in allowed or
                declaration != dict(source_run=declaration['source_run'],
                    source_manifest_sha256=task['native']['files']['physical_manifest']['sha256'],
                    method=task['method'],route_end_iteration=20,run_id=task['run_id'],storage='scratch') or
                Path(value['physical_run']).absolute() != ROOT/'backend/results'/(task['run_id']+'_a'+row['slurm_job_id'])):
            raise ValueError('Native attempt changed the declared TRAIN source or physical specification')
        if value['source_sha256'] != expected:
            raise ValueError('Native attempt used a changed physical worker')
        for name, digest in expected.items():
            collector.bind(path.parent/'source'/name, digest)
        if row.get('time_end') and row.get('native_success') is False:
            row['native_diagnosis_required'] = True
    return result


def checked_state(state, pinned, output, plan_path):
    if state['binding'] != pinned or len(state['generations']) > 3:
        raise ValueError('Recovery plan, sources, scheduling contract or generation limit changed')
    jobs = []
    for number, generation in enumerate(state['generations'], 1):
        expected = submission(output,plan_path,number,generation['indices'])
        for key in ('generation','indices','name','command'):
            if generation[key] != expected[key]:
                raise ValueError('Recovery submission identity or exact indices changed')
        if generation.get('job'):
            if not generation['job'].isdigit():
                raise ValueError('Invalid saved recovery allocation ID')
            jobs.append(generation['job'])
        if generation['generation'] != number:
            raise ValueError('Recovery generations must form an exact prefix')
        collector.bind(output/'generations'/('intent_'+str(number)+'.json'),generation['intent_sha256'])
        original = read(output/'generations'/('intent_'+str(number)+'.json'))
        if any(generation[key] != original[key] for key in ('generation','indices','name','command','created_unix','admission')):
            raise ValueError('Recovery journal differs from its immutable dispatch intent')
        admission = generation['admission']
        if (admission['plan_sha256'] != PLAN_SHA or
                admission['source_sha256'] != pinned['source_sha256'] or
                generation['indices'] != [r['index'] for r in admission['missing_records'] if r['decision']=='retry']):
            raise ValueError('Recovery indices differ from the frozen eligibility evidence')
    if len(jobs) != len(set(jobs)):
        raise ValueError('Two recovery generations cannot adopt one allocation')


def scan(plan, state, snapshot):
    records = []
    for task in plan['tasks']:
        row = collector.inspect_result(plan, task)
        evidence = attempts(task,plan.get('source_sha256'))
        row['attempts'] = evidence
        if row['state'] == 'completed':
            row['decision'] = 'completed_skip'
        else:
            index = task['index']
            refs = [ORIGINAL+'_'+str(index)]
            if index < 3:
                refs.append(DEV012+'_'+str(index))
            for generation in state['generations']:
                if generation.get('job') and index in generation['indices']:
                    refs.append(generation['job']+'_'+str(index))
            refs.extend(a['slurm_job_id'] for a in evidence)
            refs = sorted(set(refs))
            observations = {ref: observed(snapshot, ref) for ref in refs}
            flat = [s for values in observations.values() for s in values]
            if any(a.get('native_diagnosis_required') for a in evidence):
                decision = 'diagnosis_required'
            elif any(s in LIVE or s == 'ACCOUNTING_PENDING' for s in flat):
                decision = 'wait'
            elif flat and all(s in RESTARTABLE for s in flat):
                decision = 'retry'
            else:
                decision = 'diagnosis_required'
            row.update(decision=decision, scheduler=observations,
                reason=None if decision == 'retry' else 'Missing label: retain real scheduler/native evidence; never infer PPA')
        records.append(row)
    return records


def submission(output, plan_path, generation, indices):
    if (type(generation) is not int or not 1 <= generation <= 3 or not indices or
            indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < 64 for i in indices)):
        raise ValueError('Invalid bounded missing-only recovery generation')
    identity = collector.object_sha(dict(plan_sha256=PLAN_SHA, output=str(output)))[:12]
    name = 'qtdr_'+identity+'_g'+str(generation)
    command = ['sbatch', '--parsable', '--job-name='+name, '--partition=cpu-scavenger',
        '--qos=cpu-scavenger', '--exclude='+EXCLUDE, '--cpus-per-task=8', '--mem=12G',
        '--time=02:30:00',
        '--array='+','.join(map(str, indices))+'%2',
        'fql/qstudy_train_final_route.slurm', '--stage', 'worker', '--plan', str(plan_path),
        '--plan-sha256', PLAN_SHA]
    return dict(generation=generation, indices=indices, name=name, command=command,
                status='submission_intent', created_unix=time.time(), job=None)


def adopt_or_submit(intent, scheduler, fresh):
    found, evidence = scheduler.find_name(intent['name'],intent['created_unix'])
    if found:
        return dict(intent, job=found, status='submitted', adoption=evidence)
    if not fresh:
        # A prior sbatch might have succeeded just before interruption. An empty
        # accounting query is not evidence that reissuing it is safe.
        return dict(intent, status='submission_accounting_pending', adoption=evidence)
    job, evidence = scheduler.submit(intent['command'])
    return dict(intent, job=job, status='submitted', submission=evidence)


def stop(_signum, _frame):
    global STOP
    STOP = True


def controller(args, scheduler=None, sleep=time.sleep):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this coordinator through bounded CPU Slurm')
    scheduler = scheduler or Scheduler()
    plan_path, output = Path(args.plan).resolve(), Path(args.output).resolve()
    plan, pinned = binding(plan_path, args.plan_sha256, output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output/'state.json'
        state = read(path) if path.exists() else dict(binding=pinned, generations=[], events=[], status='running')
        # Publication order is immutable intent -> journal -> sbatch. Recover
        # the harmless first gap without treating an uncertain submission as new.
        orphan = output/'generations'/('intent_'+str(len(state['generations'])+1)+'.json')
        if orphan.is_file():
            recovered = read(orphan)
            recovered['intent_sha256'] = sha(orphan)
            state['generations'].append(recovered)
        checked_state(state,pinned,output,plan_path)
        for name, digest in pinned['source_sha256'].items():
            target = output/'sources'/name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, target)
            collector.bind(target, digest)

        def save(event=None):
            if event:
                state['events'].append(dict(event, unix=time.time()))
            state['controller_job_id'] = os.environ['SLURM_JOB_ID']
            state['updated_unix'] = time.time()
            atomic_json(path, state)

        save(dict(event='controller_start', restart_count=os.environ.get('SLURM_RESTART_COUNT', '0')))
        try:
            while True:
                jobs = [ORIGINAL, DEV012]+[g['job'] for g in state['generations'] if g.get('job')]
                try:
                    snapshot = scheduler.snapshot(jobs)
                except (OSError, subprocess.SubprocessError) as error:
                    state['scheduler_error'] = repr(error);save()
                    if STOP:
                        break
                    sleep(30);continue
                state['last_scheduler_snapshot'] = snapshot
                gates = {job:observed(snapshot, job, parent=True) for job in (ORIGINAL, DEV012)}
                state['prior_array_gate'] = gates
                if not all(terminal(value) for value in gates.values()):
                    save()
                    if STOP:
                        break
                    sleep(30);continue
                # Recover an interrupted submit before inspecting new work.
                pending = next((g for g in state['generations'] if not g.get('job')), None)
                if pending:
                    new = adopt_or_submit(pending, scheduler, fresh=False)
                    state['generations'][pending['generation']-1] = new;save()
                    if not new.get('job'):
                        if time.time()-pending['created_unix'] > 900:
                            state['diagnosis'] = 'Ambiguous earlier sbatch; unique name not yet accounted; no duplicate submission'
                            break
                        if STOP:
                            break
                        sleep(30)
                    continue
                active = {g['job']:observed(snapshot,g['job'],parent=True) for g in state['generations']}
                state['recovery_array_gate'] = active
                if not all(terminal(value) for value in active.values()):
                    save()
                    if STOP:
                        break
                    sleep(30);continue
                # The arrays have stopped writing. Only now read attempt/native
                # metadata and query their exact allocated numeric job IDs.
                attempt_jobs = [a['slurm_job_id'] for task in plan['tasks']
                                for a in attempts(task,plan.get('source_sha256'))]
                if set(attempt_jobs)-set(jobs):
                    try:
                        snapshot = scheduler.snapshot(jobs+attempt_jobs)
                    except (OSError,subprocess.SubprocessError) as error:
                        state['scheduler_error']=repr(error);save()
                        if STOP:
                            break
                        sleep(30);continue
                    state['last_scheduler_snapshot'] = snapshot
                rows = scan(plan, state, snapshot)
                state['records'] = rows
                state['counts'] = {key:sum(r['decision']==key for r in rows)
                                   for key in ('completed_skip','wait','retry','diagnosis_required')}
                if state['counts']['completed_skip'] == 64:
                    state['status'] = 'completed';break
                eligible = [r['index'] for r in rows if r['decision']=='retry']
                if STOP:
                    break
                if state['counts']['wait']:
                    # An exact native attempt not yet settled in accounting
                    # could still occupy a worker slot. Never add a %2 array
                    # alongside it merely because other rows are retryable.
                    save();sleep(30);continue
                if not eligible or len(state['generations']) == 3:
                    state['status'] = 'unresolved';break
                intent = submission(output, plan_path, len(state['generations'])+1, eligible)
                intent['admission'] = dict(plan_sha256=args.plan_sha256,source_sha256=pinned['source_sha256'],scheduler=snapshot,
                    completed_skip=[r['index'] for r in rows if r['decision']=='completed_skip'],
                    missing_records=[r for r in rows if r['decision']!='completed_skip'])
                intent_path = output/'generations'/('intent_'+str(intent['generation'])+'.json')
                collector.freeze(intent_path,intent)
                intent['intent_sha256'] = sha(intent_path)
                state['generations'].append(intent);save(dict(event='missing_only_submission_intent',indices=eligible))
                state['generations'][-1] = adopt_or_submit(intent, scheduler, fresh=True)
                save(dict(event='recovery_generation_adopted_or_submitted',generation=intent['generation']))
        except Exception as error:
            state.update(status='unresolved',diagnosis=repr(error))
            save(dict(event='diagnosis_required',error=repr(error)))
        if STOP:
            state['status'] = 'interrupted'
        elif state['status'] == 'running':
            state['status'] = 'unresolved'
        report_path = output/'reports'/('collector_'+os.environ['SLURM_JOB_ID']+'_'+str(time.time_ns())+'.json')
        report = collector.report(plan_path,report_path,args.plan_sha256)
        state['collector_report'] = collector.bind(report_path)
        state['collector_report_status'] = report['status']
        save(dict(event='coordinator_finished',status=state['status'],collector_report=state['collector_report']))
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGUSR1,stop)
    signal.signal(signal.SIGTERM,stop)
    result = controller(args)
    print(json.dumps(dict(status=result['status'],collector_report=result['collector_report'],
                          generations=len(result['generations']))),flush=True)


if __name__ == '__main__':
    main()
