"""Fixed72 detailed-route adapter for the sealed scalar-online control.

No model, physical flow or sampling change. Metadata creation is login-safe;
source sealing and every original native measurement require bounded Slurm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE_SHA = '9c30c121fa611c26a6368e1224ac6d3646273648c31e64ee0372a9781aed0535'
BASE_PATH = ROOT/'tools/qstudy_scalar_online_control.py'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('Original scalar stage0 helper changed')
spec = importlib.util.spec_from_file_location('_scalar_route_base', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

VERSION = 'qstudy_scalar_online_control_route_v1'
PHASES = ('prepare', 'route')
HELPERS = ('tools/qstudy_scalar_online_control_route.py',
           'tools/qstudy_scalar_online_control_route.slurm',
           'tools/qstudy_scalar_online_control_route_worker.slurm')
PROTOCOL = 'docs/qstudy_scalar_online_control_route.md'
RUN_ID_TRANSLATION = "run_id='go2o_'+original.run_id; every other task field unchanged"
read, sha, bound, immutable = base.read, base.sha, base.bound, base.immutable


def continuation_module():
    path = ROOT/'tools/qstudy_scalar_online_control_continue.py'
    spec = importlib.util.spec_from_file_location('_scalar_route_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resources(phase):
    if phase == 'prepare':
        return ['--partition=cpu-scavenger', '--qos=cpu-scavenger',
                '--cpus-per-task=2', '--mem=8G', '--time=00:20:00'], 1
    if phase == 'route':
        return ['--partition=cpu-scavenger', '--qos=cpu-scavenger',
                '--cpus-per-task=8', '--mem=12G', '--time=02:30:00'], 2
    raise ValueError('Unknown route phase')


def create(args):
    original = base.checked_manifest(args.base_manifest, args.base_manifest_sha256)
    continuation = bound(args.continuation_manifest, args.continuation_manifest_sha256)
    continued = continuation_module().checked_manifest(continuation['path'], continuation['sha256'])
    if continued['base_manifest'] != bound(args.base_manifest, args.base_manifest_sha256):
        raise ValueError('Continuation belongs to another control')
    names = (*HELPERS, PROTOCOL, 'fql/qstudy_final_route.py',
             'fql/qstudy_actor_report.py', 'fql/graph_o2o_full_route.py',
             'fql/graph_o2o_full_route.slurm', 'fql/graph_o2o_final_evidence.py',
             'fql/graph_o2o_archive.py')
    output = Path(original['output'])/'route'
    value = dict(version=VERSION, output=str(output),
                 base_manifest=bound(args.base_manifest, args.base_manifest_sha256),
                 continuation_manifest=continuation, implemented_phases=list(PHASES),
                 future_phases=[], max_generations=original['max_generations'],
                 source_sha256={name: sha(ROOT/name) for name in names},
                 stage_resources={p: dict(options=resources(p)[0], concurrency=resources(p)[1]) for p in PHASES},
                 requested=72, selection='original fixed T0/T2 indices0..3; no PPA/Q selection',
                 run_id_translation=RUN_ID_TRANSLATION,
                 original_tasks_sha256=task_inventory_sha256(original))
    output.mkdir(parents=True, exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    for name, digest in value['source_sha256'].items():
        dest = output/'sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(ROOT/name, dest)
        bound(dest, digest)
    immutable(output/'manifest.json', value)
    print(json.dumps(dict(manifest=bound(output/'manifest.json'), submitted_jobs=0)))


def checked_manifest(path, digest):
    value = read(bound(path, digest)['path'])
    original = base.checked_manifest(value['base_manifest']['path'], value['base_manifest']['sha256'])
    continued = continuation_module().checked_manifest(
        value['continuation_manifest']['path'], value['continuation_manifest']['sha256'])
    if (value['version'] != VERSION or value['implemented_phases'] != list(PHASES)
            or value['future_phases'] or value['requested'] != 72
            or value['max_generations'] != original['max_generations']
            or value['run_id_translation'] != RUN_ID_TRANSLATION
            or value['original_tasks_sha256'] != task_inventory_sha256(original)
            or Path(value['output']) != Path(original['output'])/'route'
            or Path(path).resolve() != Path(value['output'])/'manifest.json'
            or continued['base_manifest'] != value['base_manifest']
            or value['stage_resources'] != {p: dict(options=resources(p)[0], concurrency=resources(p)[1]) for p in PHASES}):
        raise ValueError('Route adapter identity, budget or resources changed')
    for name, expected in value['source_sha256'].items():
        bound(ROOT/name, expected)
        bound(Path(value['output'])/'sources'/name, expected)
    return value


def original(plan):
    return base.checked_manifest(plan['base_manifest']['path'], plan['base_manifest']['sha256'])


def task_inventory_sha256(old):
    return hashlib.sha256(json.dumps(old['physical_drt_requests'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def route_tasks(old):
    # The original frozen identities lacked the prefix required by the frozen
    # archiver. Declare this execution-only correction before any route call.
    return [dict(t, run_id='go2o_'+t['run_id']) for t in old['physical_drt_requests']]


def phase_tasks(plan, phase):
    if phase == 'prepare':
        return [dict(requested=72, indices=[0, 1, 2, 3])]
    if phase == 'route':
        return [dict(task=t) for t in route_tasks(original(plan))]
    raise ValueError('Unknown route phase')


def route_plan(plan):
    old = original(plan)
    return dict(version=1, schema=VERSION, study=VERSION, design='bp_fe',
                purpose='development_validation', fresh_confirmation=False,
                historical_table2=False, training_feedback_eligible=False,
                requested=72, actor_seeds=[0, 1, 2], rounds=[0, 2],
                evaluation_indices=[0, 1, 2, 3], original_evaluation_samples=8,
                detailed_route_end_iteration=20, backend_threads=8,
                backend_storage='scratch', concurrency=2,
                base_manifest=plan['base_manifest'], route_manifest=bound(Path(plan['output'])/'manifest.json'),
                tasks=route_tasks(old), original_tasks_sha256=task_inventory_sha256(old),
                run_id_translation=RUN_ID_TRANSLATION,
                final_metrics=['DRT_WL', 'WNS', 'TNS', 'Power', 'StdCellArea', 'DRC'],
                metric_units=dict(DRT_WL='um', WNS='ns', TNS='ns', Power='W', StdCellArea='um^2', DRC='count'),
                policy_estimand='within-seed fixed noise mean, then three training seeds; T0 two roots only',
                selection=plan['selection'],
                power_interpretation='common tool-default activity, not measured workload or silicon power')


def inspect_source(plan, task, cache):
    """Native read-only source barrier, compatible with inspect_final."""
    import numpy as np
    from fql.qstudy_actor_report import native_fingerprints

    old = base.checked_manifest(plan['base_manifest']['path'], plan['base_manifest']['sha256'])
    matches = [t for t in route_tasks(old) if t['key'] == task['key']]
    if len(matches) != 1 or matches[0] != task:
        raise ValueError('Route task is not the exact predeclared physical request')
    key = task['group']
    context_key = (key, task['seed'])
    if context_key not in cache:
        if task['round'] == 0:
            cache[context_key] = base.checked_export(old, key, task['seed'])
        else:
            adapter = read(bound(plan['route_manifest']['path'], plan['route_manifest']['sha256'])['path'])
            continued = continuation_module().checked_manifest(
                adapter['continuation_manifest']['path'], adapter['continuation_manifest']['sha256'])
            cache[context_key] = continuation_module().checked_export(continued, key, task['seed'])
    cfg, _, _, _, _ = cache[context_key]
    grt = [r for r in old['physical_grt_requests'] if r['physical_key'] == key
           and r['seed'] == task['seed'] and r['role'] == 'evaluation' and r['index'] == task['noise_index']]
    if len(grt) != 1 or grt[0]['round'] != task['round'] or grt[0]['arm'] != task['arm']:
        raise ValueError('DRT task changed its native evaluation source')
    request, outcome, tasks, sample, physical = base.inspect_native(old, grt[0])
    if (str((sample/'outcome.json').resolve()) != task['source_outcome']
            or request['evaluation_purpose'] != 'development_validation'
            or request['training_feedback_eligible'] is not False):
        raise ValueError('Source is not the declared evaluation layout')
    # Export request rows carry a path string; the completion record carries a
    # bound-artifact object. Canonicalize the actual producing row here.
    checkpoint = bound(request['source_checkpoint'], request['checkpoint_sha256'])
    with np.load(sample/'actions.npz', allow_pickle=False) as arrays:
        values = {name: arrays[name].tolist() for name in ('noise', 'raw_action', 'action')}
    return dict(version=1, key=task['key'], arm=task['arm'], seed=task['seed'], round=task['round'],
                noise_index=task['noise_index'], status=outcome['status'],
                source_outcome=task['source_outcome'], source_outcome_sha256=sha(sample/'outcome.json'),
                request_sha256=sha(sample/'request.json'), tasks_sha256=sha(tasks),
                export_completion_sha256=sha(tasks.with_name('completed.json')),
                actions_npz_sha256=sha(sample/'actions.npz'), config_sha256=physical['config_sha256'],
                dataset_sha256=physical['dataset_sha256'],
                actor=dict(checkpoint=checkpoint, export_config=bound(Path(grt[0]['experiment'])/'config.json'),
                           declaration=bound(Path(grt[0]['experiment'])/'declaration.json'),
                           source_parent=cfg.get('parent_training', cfg.get('parent_identities'))),
                raw_action_sha256=request['raw_action_sha256'], action_sha256=request['action_sha256'],
                noise_sha256=request['noise_sha256'], noise_seed=request['noise_seed'],
                exact_coordinates_and_noise=values, def_sha256=request.get('def_sha256'),
                native=native_fingerprints(outcome), physical_flow_revision=physical['physical_flow_revision'],
                failure_class=outcome.get('failure_class'), GRT_metrics=outcome['metrics'], final_metrics=None)


def inspect_sources(experiment, plan, *, seal=False):
    from fql import qstudy_final_route as native
    namespace = dict(vars(native), inspect_source=inspect_source)
    function = native.inspect_sources
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)(
        Path(experiment), plan, seal=seal)


def task_run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Source sealing and native routing require Slurm')
    plan = checked_manifest(args.manifest, args.manifest_sha256)
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    tasks = phase_tasks(plan, args.phase)
    if not 0 <= index < len(tasks):
        raise IndexError('Outside fixed route inventory')
    receipt = Path(plan['output'])/'receipts'/args.phase/(str(index)+'.json')
    if receipt.exists():
        dispatch['checked_receipt'](plan, args.phase, index)
        return
    from fql import qstudy_final_route as native
    expected = route_plan(plan)
    path = Path(plan['output'])/'full_route_plan.json'
    if args.phase == 'prepare':
        immutable(path, expected)
        rows = inspect_sources(plan['output'], expected, seal=True)
        if len(rows) != 72 or any(r['state'] != 'resolved' for r in rows):
            raise ValueError('All72 native sources must resolve before routing: '+str([(r['key'], r['state'], r.get('reason')) for r in rows if r['state'] != 'resolved']))
        gate = Path(plan['output'])/'source_resolution.json'
        immutable(gate, dict(status='pass', requested=72, records=rows, backend_calls=0))
        result = [bound(path), bound(gate)]
    else:
        if read(path) != expected:
            raise ValueError('Saved route plan changed')
        task = tasks[index]['task']
        binding = inspect_source(expected, task, {})
        sealed = Path(plan['output'])/'source_bindings'/(task['key']+'.json')
        if read(sealed) != binding:
            raise ValueError('Sealed GRT source changed before detailed route')
        subprocess.run([str(ROOT/'.venv/bin/python'), '-u', '-m', 'fql.graph_o2o_full_route',
                        '--source-outcome', task['source_outcome'], '--run-id', task['run_id'],
                        '--method', task['method'], '--storage', 'scratch', '--output', task['output'],
                        '--end-iter', '20'], cwd=ROOT, check=True)
        final = native.inspect_final(Path(plan['output']), expected, task,
            dict(state='resolved', binding=binding, binding_sha256=sha(sealed)))
        if final['state'] not in ('completed', 'upstream_physical_failure'):
            raise ValueError('No actual admitted final outcome')
        result = [bound(Path(task['output'])/'outcome.json'), bound(sealed)]
    immutable(receipt, dict(version=VERSION, manifest=bound(args.manifest, args.manifest_sha256),
                           phase=args.phase, index=index, task=tasks[index], results=result,
                           slurm_job_id=os.environ['SLURM_JOB_ID']))


def stage_submit(args):
    plan = checked_manifest(args.manifest, args.manifest_sha256)
    continued = continuation_module().checked_manifest(
        plan['continuation_manifest']['path'], plan['continuation_manifest']['sha256'])
    module = continuation_module()
    for phase in module.PHASES:
        if module.dispatch['missing_indices'](continued, phase):
            raise ValueError('Continuation has unfinished '+phase)
        state, _ = module.dispatch['stage_state'](continued, phase)
        states, _ = base.scheduler_states(state['attempts'][-1]['job'])
        if any(s in base.LIVE or s == 'ACCOUNTING_PENDING' for s in states):
            raise ValueError('Continuation still live/accounting pending')
    dispatch['stage_submit'](args)


dispatch = dict(vars(base))
dispatch.update(VERSION=VERSION, PHASES=PHASES, FUTURE_PHASES=(), HELPERS=HELPERS,
                checked_manifest=checked_manifest, phase_tasks=phase_tasks, resources=resources)
for name in ('checked_receipt', 'stage_state', 'missing_indices', 'stage_submit', 'status'):
    function = getattr(base, name)
    dispatch[name] = FunctionType(function.__code__, dispatch, name, function.__defaults__, function.__closure__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('manifest')
    for name in ('base-manifest', 'base-manifest-sha256', 'continuation-manifest', 'continuation-manifest-sha256'):
        p.add_argument('--'+name, required=True)
    for name in ('submit', 'status', 'task'):
        p = sub.add_parser(name)
        p.add_argument('--manifest', required=True, type=Path)
        p.add_argument('--manifest-sha256', required=True)
        if name != 'status':
            p.add_argument('--phase', required=True, choices=PHASES)
        if name == 'task':
            p.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.command == 'manifest':
        create(args)
    elif args.command == 'task':
        task_run(args)
    elif args.command == 'submit':
        stage_submit(args)
    else:
        dispatch['status'](args)


if __name__ == '__main__':
    main()
