"""Short, metadata-only callbacks for the already sealed scalar-control stages."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path('/scratch/bohanli/flowplace_graph/qstudy_scalar_online_control_v1')
OUTPUT = STUDY/'advancement_v1'
VERSION = 'qstudy_scalar_control_advance_v1'
MANIFESTS = (
    ('base', STUDY/'manifest.json', 'd86e233577c600b095729eff34f466c725eab7fbe02ba60166fd7957667e9016',
     'tools/qstudy_scalar_online_control.py', '9c30c121fa611c26a6368e1224ac6d3646273648c31e64ee0372a9781aed0535'),
    ('continuation', STUDY/'continuation_v1/manifest.json', '89404d7256ce5d78b7c63999f3728d32ec923dd95c2b4fca1e5a36c48346087f',
     'tools/qstudy_scalar_online_control_continue.py', 'f43459392136c3c60876053a5a0c01cb6ba99536a48354f64ffd2b708ecb17b3'),
    ('route', STUDY/'route/manifest.json', '50d44c3a49221f3a929031801605dfbbefcc75953d73a788ce036d2ce96c4b82',
     'tools/qstudy_scalar_online_control_route.py', '6c0fb0bbeab681258306493dd14bf8ddbc8f73cd15d91f9278742ae9215985a8'),
)
SOURCES = ('tools/qstudy_scalar_control_advance.py', 'tools/qstudy_scalar_control_advance.slurm',
           'docs/qstudy_scalar_control_advance.md')
MAX_DELAYED = 6
MAX_CALLBACKS = 128
ANCHOR = '1800443'
ANCHOR_FILES = (
    (STUDY/'continuation_v1/launchers/train1_after_native0_request.json',
     'ca03267018083875f385c37e43cd122f3f7824523e096f73e0b32d88c74733a1'),
    (STUDY/'continuation_v1/launchers/train1_after_native0_submission.json',
     'c1134ca71efc9721c275df96bf8c2b4a8e4dadd639a91de45ab54085a78d3b35'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Changed advancement input: '+str(path))
    return dict(path=str(path), sha256=digest)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp_'+str(os.getpid()))
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Immutable advancement metadata changed')
    else:
        atomic(path, value)


def drivers():
    result = []
    for name, path, digest, source, source_sha in MANIFESTS:
        bound(ROOT/source, source_sha)
        spec = importlib.util.spec_from_file_location('_advance_'+name, ROOT/source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        plan = module.checked_manifest(path, digest)
        dispatch = vars(module) if name == 'base' else module.dispatch
        result.append((name, module, plan, dispatch, path, digest))
    raw_base = result[0][1]
    scheduler = lambda job: scheduler_states(job, raw_base.scheduler_states)
    private = []
    for name, module, plan, _, path, digest in result:
        proxy = private_driver(module, scheduler, private[0][1] if private else None,
                               private[1][1] if len(private) > 1 else None)
        dispatch = vars(proxy) if name == 'base' else proxy.dispatch
        private.append((name, proxy, plan, dispatch, path, digest))
    return private


def scheduler_states(job, original):
    """Only a Slurmctld-aged-out job may fall back to sacct directly."""
    try:
        return original(job)
    except subprocess.CalledProcessError as error:
        if (error.cmd[:3] != ['squeue', '-j', str(job)]
                or (error.stderr or '').strip() != 'slurm_load_jobs error: Invalid job id specified'):
            raise
    result = subprocess.run(['sacct', '-j', str(job), '-X', '--duplicates', '-n', '-P',
        '-o', 'JobID%64,JobIDRaw,State,ExitCode,JobName%200'], text=True, capture_output=True, check=True)
    states, children = [], {}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) < 5: continue
        state = fields[2].split()[0].rstrip('+'); states.append(state)
        child = re.fullmatch(re.escape(str(job))+r'_(\d+)', fields[0])
        if child: children.setdefault(child[1], []).append(state)
    return states or ['ACCOUNTING_PENDING'], dict(squeue_invalid_id=True, sacct=result.stdout, child_states=children)


def private_driver(module, scheduler, base=None, continuation=None):
    """Supply the scheduler compatibility fix only in isolated metadata globals."""
    namespace = dict(getattr(module, 'dispatch', vars(module)), scheduler_states=scheduler)
    for name in ('checked_receipt', 'stage_state', 'missing_indices', 'stage_submit', 'status'):
        function = namespace[name]
        namespace[name] = FunctionType(function.__code__, namespace, name, function.__defaults__, function.__closure__)
    attributes = dict(vars(module), scheduler_states=scheduler)
    if hasattr(module, 'dispatch'):
        outer = dict(vars(module), base=base, dispatch=namespace)
        if continuation is not None: outer['continuation_module'] = lambda: continuation
        function = module.stage_submit
        attributes.update(dispatch=namespace, stage_submit=FunctionType(function.__code__, outer,
            function.__name__, function.__defaults__, function.__closure__))
    else:
        attributes.update({name: namespace[name] for name in
            ('checked_receipt', 'stage_state', 'missing_indices', 'stage_submit', 'status')})
    return SimpleNamespace(**attributes)


def anchor_bindings():
    bindings = [bound(path, digest) for path, digest in ANCHOR_FILES]
    request, submitted = [read(item['path']) for item in bindings]
    expected_tail = [str(ROOT/'tools/qstudy_scalar_online_control_continue.slurm'), 'submit',
        '--manifest', str(MANIFESTS[1][1]), '--manifest-sha256', MANIFESTS[1][2], '--phase', 'train1']
    if (request['command'][-len(expected_tail):] != expected_tail or
            '--dependency=afterok:1800396' not in request['command'] or
            request['dependency_parent'] != '1800396' or submitted['request_sha256'] != bindings[0]['sha256'] or
            submitted['returncode'] != 0 or submitted['stdout'].strip() != ANCHOR):
        raise ValueError('Original train1 metadata submitter differs from its exact declaration')
    return bindings


def admission_value():
    return dict(version=VERSION, output=str(OUTPUT), anchor_submitter=ANCHOR,
        anchor_declaration=anchor_bindings(),
        manifests={name: bound(path, digest) for name, path, digest, _, _ in MANIFESTS},
        sources={name: sha(ROOT/name) for name in SOURCES},
        max_consecutive_delays=MAX_DELAYED, max_callbacks=MAX_CALLBACKS,
        stages='original base3, continuation9, route2; only original stage_submit',
        callback_resources=dict(cpus=1, memory='3G', time='00:10:00'),
        training_calls=0, model_forwards=0, backend_calls=0)


def prepare():
    drivers()
    value = admission_value()
    for name, digest in value['sources'].items():
        dest = OUTPUT/'sources'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(ROOT/name, dest)
        bound(dest, digest)
    immutable(OUTPUT/'admission.json', value)
    print(json.dumps(dict(admission=bound(OUTPUT/'admission.json'), submitted_jobs=0)))


def checked_admission(digest):
    path = OUTPUT/'admission.json'; value = read(bound(path, digest)['path'])
    if value != admission_value():
        raise ValueError('Advancement admission/source/bounds changed')
    for name, expected in value['sources'].items():
        bound(OUTPUT/'sources'/name, expected)
    return value


def phase_view(item, phase, scheduler):
    name, _, plan, dispatch, _, _ = item
    missing = dispatch['missing_indices'](plan, phase)
    state, _ = dispatch['stage_state'](plan, phase)
    last = state['attempts'][-1] if state['attempts'] else None
    states, evidence = scheduler(last['job']) if last and last.get('job') else ([], {})
    return dict(stage=name+':'+phase, missing=missing, last=last, states=states, scheduler=evidence)


def is_wait(view, base):
    return any(s in base.LIVE or s == 'ACCOUNTING_PENDING' for s in view['states'])


def anchor_view(base):
    declared = anchor_bindings()
    states, evidence = base.scheduler_states(ANCHOR)
    if not any(s in base.LIVE for s in states):
        return dict(states=states, evidence=evidence, obsolete=False)
    show = subprocess.run(['scontrol', '-o', 'show', 'job', ANCHOR], text=True,
                          capture_output=True, check=True)
    text = show.stdout
    exact = all(s in text for s in (
        'JobId='+ANCHOR+' ', 'JobName=soc_continue_after_native0 ',
        'UserId='+os.environ['USER']+'(', 'JobState=PENDING ',
        'Reason=DependencyNeverSatisfied ', 'Dependency=afterok:1800396',
        'Command='+str(ROOT/'tools/qstudy_scalar_online_control_continue.slurm')))
    obsolete = False
    if exact:
        original, original_evidence = base.scheduler_states('1800396')
        evidence['original_native0'] = dict(states=original, evidence=original_evidence)
        obsolete = (bool(original) and all(s not in base.LIVE and s != 'ACCOUNTING_PENDING' for s in original)
                    and any(s != 'COMPLETED' for s in original))
    evidence['scontrol'] = text
    evidence['declaration'] = declared
    # The stale afterok submitter cannot start. Do not cancel or alter it; the
    # original train1 submission.lock also excludes a simultaneous late submitter.
    return dict(states=states, evidence=evidence, obsolete=obsolete)


def advance_once(items, state, save, anchor=anchor_view):
    """Inspect/submit one phase; no waiting, model load or physical call here."""
    base = items[0][1]
    for item in items:
        name, module, plan, _, path, digest = item
        if name == 'continuation' and not state.get('anchor_closed'):
            observed = anchor(base); state['anchor'] = observed
            if not observed['obsolete'] and any(s in base.LIVE or s == 'ACCOUNTING_PENDING' for s in observed['states']):
                save(); return dict(kind='wait', job=ANCHOR, accounting='ACCOUNTING_PENDING' in observed['states'], stage='anchor')
            state['anchor_closed'] = True; save()
        for phase in module.PHASES:
            view = phase_view(item, phase, base.scheduler_states)
            if is_wait(view, base):
                state['current'] = view; save()
                return dict(kind='wait', job=view['last']['job'], accounting='ACCOUNTING_PENDING' in view['states'], stage=view['stage'])
            if not view['missing']:
                if not view['last'] or not view['last'].get('job'):
                    raise ValueError('Complete phase lacks an allocated submission journal')
                continue
            state['current'] = view; save()
            args = SimpleNamespace(manifest=path, manifest_sha256=digest, phase=phase)
            try:
                module.stage_submit(args)
            except (BlockingIOError, RuntimeError) as error:
                after = phase_view(item, phase, base.scheduler_states)
                # An interrupted original sbatch intent is never submitted twice.
                if isinstance(error, BlockingIOError) or (after['last'] and not after['last'].get('job')):
                    state['current'] = after; state['last_error'] = str(error); save()
                    return dict(kind='wait', job=None, accounting=True, stage=after['stage'])
                raise
            after = phase_view(item, phase, base.scheduler_states)
            state['current'] = after; save()
            if not after['last'] or not after['last'].get('job'):
                raise RuntimeError('Original stage did not provide an allocated worker handle')
            return dict(kind='wait', job=after['last']['job'],
                        accounting=not any(s in base.LIVE for s in after['states']), stage=after['stage'])
    return dict(kind='complete', stage='all14')


def callback_intent(number, admission_sha, decision):
    dependency = decision.get('job') if not decision.get('accounting') else None
    name = 'soca_'+admission_sha[:10]+'_'+str(number)
    command = ['sbatch', '--parsable', '--job-name='+name,
        '--output='+str(OUTPUT/'logs'/('%j.log'))]
    if dependency:
        command += ['--dependency=afterany:'+dependency, '--kill-on-invalid-dep=yes']
    else:
        command += ['--begin=now+2minutes']
    command += [str(ROOT/SOURCES[1]), 'run', '--admission-sha256', admission_sha]
    return dict(number=number, name=name, decision=decision, command=command, job=None)


def schedule(state, admission_sha, decision, save, base):
    callbacks = state['callbacks']
    if len(callbacks) >= MAX_CALLBACKS:
        raise RuntimeError('Bounded callback inventory exhausted')
    if decision.get('accounting'):
        state['consecutive_delays'] += 1
    else:
        state['consecutive_delays'] = 0
    if state['consecutive_delays'] > MAX_DELAYED:
        raise RuntimeError('Scheduler accounting/submission remained unresolved across six delayed callbacks')
    intent = callback_intent(len(callbacks), admission_sha, decision)
    path = OUTPUT/'intents'/(str(intent['number'])+'.json')
    immutable(path, intent); callbacks.append(dict(intent, intent=bound(path))); save()
    current = callbacks[-1]
    env = dict(os.environ); env.pop('SLURM_RESTART_COUNT', None)
    result = subprocess.run(intent['command'], text=True, capture_output=True, env=env)
    current.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr); save()
    if result.returncode:
        current['rejected'] = True; save()
        # No allocation exists on a rejected dependency race; retry without a
        # stale dependency, after a bounded delay and full runtime revalidation.
        if decision.get('job') and ('dependency' in result.stderr.lower()):
            return schedule(state, admission_sha, dict(decision, job=None, accounting=True), save, base)
        raise RuntimeError('Callback submission rejected; inspect saved stderr')
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unknown callback sbatch result; unique intent retained for adoption')
    current['job'] = job; save()
    return job


def run(digest):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Advancement requires a short Slurm allocation')
    checked_admission(digest); items = drivers(); base = items[0][1]
    (OUTPUT/'logs').mkdir(exist_ok=True)
    with (OUTPUT/'advance.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='another_advancer_holds_lock'))); return
        path = OUTPUT/'state.json'
        state = read(path) if path.exists() else dict(version=VERSION, admission=bound(OUTPUT/'admission.json', digest),
            callbacks=[], consecutive_delays=0, visits=[])
        if state['version'] != VERSION or state['admission'] != bound(OUTPUT/'admission.json', digest):
            raise ValueError('Advancement journal changed identity')
        def save(): atomic(path, state)
        # Recover the only possible crash window: immutable intent before journal.
        orphan = OUTPUT/'intents'/(str(len(state['callbacks']))+'.json')
        if orphan.exists():
            intent = read(orphan)
            if intent != callback_intent(len(state['callbacks']), digest, intent['decision']):
                raise ValueError('Orphan callback intent changed')
            state['callbacks'].append(dict(intent, intent=bound(orphan))); save()
        for callback in state['callbacks']:
            original = read(bound(callback['intent']['path'], callback['intent']['sha256'])['path'])
            if {k: v for k, v in original.items() if k != 'job'} != {k: callback[k] for k in original if k != 'job'}:
                raise ValueError('Callback journal differs from immutable intent')
        current_job = os.environ['SLURM_JOB_ID']
        if state['callbacks']:
            last = state['callbacks'][-1]
            if not last.get('job') and not last.get('rejected'):
                found, evidence = base.adopt(last['name']); last['adoption'] = evidence
                if not found:
                    state['status'] = 'unresolved_callback_submission'; save()
                    raise RuntimeError('Ambiguous prior callback submission; no duplicate issued')
                last['job'] = found; save()
            if last.get('job') and last['job'] != current_job:
                states, evidence = base.scheduler_states(last['job'])
                if any(s in base.LIVE or s == 'ACCOUNTING_PENDING' for s in states):
                    state['status'] = 'existing_callback_live'; save(); return
        state['visits'].append(dict(job=current_job)); save()
        try:
            decision = advance_once(items, state, save)
            state['visits'][-1]['decision'] = decision
            if decision['kind'] == 'complete':
                state['status'] = 'completed'; save(); print(json.dumps(dict(status='completed'))); return
            state['status'] = 'callback_queued'
            job = schedule(state, digest, decision, save, base)
            print(json.dumps(dict(status=state['status'], callback_job=job, decision=decision)))
        except Exception as error:
            state['status'] = 'unresolved'; state['diagnosis'] = str(error); save(); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    p = sub.add_parser('run'); p.add_argument('--admission-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    else: run(args.admission_sha256)


if __name__ == '__main__':
    main()
