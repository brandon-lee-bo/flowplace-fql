"""Second measured actor-only continuation of a genuine dual-source BC.

The unchanged first BC continuation supplies actor, Adam and all RNG states.
C0 and C1 are admitted under their own real collection checkpoints. The
original BC teacher and separately attached scalar refit are frozen throughout;
the latter is diagnostic, never represented as the old BC's learned critic.
This runner does not collect, export, schedule EDA or implement later cycles.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

# Keep local analysis imports ahead of upstream baseline's search-path changes.
from . import qstudy_scalar_online_bc_child_export as child
from . import qstudy_scalar_online_bc_export as original_export
from . import qstudy_scalar_online_bc_train as first_bc
from . import qstudy_scalar_online_bc as provider
from . import qstudy_scalar_online_second_train as second
from . import qstudy_scalar_online_train as common
from . import qstudy_scalar_online as core
from baseline.api import RUNTIME, load_problem
from .graph_o2o_archive import atomic_json
from .graph_o2o_models import GraphPolicy
from .qstudy_data import bind, sha
from .qstudy_screen import atomic_torch

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qstudy_scalar_online_bc_second_train_v1'
SOURCE_NAMES = tuple(dict.fromkeys((*child.SOURCE_NAMES, *second.SOURCE_NAMES,
    'fql/qstudy_scalar_online_bc_second_train.py',
    'fql/qstudy_scalar_online_bc_second_train.slurm')))
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def validate_settings(settings):
    second.validate_settings(settings)
    if settings['q_weight'] != 0. or settings['update_critic'] is not False:
        raise ValueError('Second genuine BC cycle requires zero Q and a frozen critic')
    return settings


def request_settings(value):
    if value.get('version') != VERSION:
        raise ValueError('Explicit genuine BC second-cycle request required')
    second.request_settings(dict(value, version=second.VERSION))
    validate_settings(value['settings'])
    return value


def source_from_verified(cfg, verified, seed):
    """Preserve T1 bytes/state and explicitly retain the original two sources."""
    identity, root, payload = (verified[k] for k in ('identity', 'source', 'payload'))
    if (identity != cfg['verified_children'][str(seed)] or
            root['schema'] != cfg['scalar_schema'] or
            root['binding'].get('version') != provider.VERSION or
            root['payload']['binding'].get('seed') != seed or
            identity.get('offline_source') != root['binding'] or
            identity.get('bc_actor_prefix') != root['binding']['actor_prefix'] or
            identity.get('scalar_source') != root['binding']['target_scalar_source'] or
            identity.get('scalar_critic_refit') != root['binding']['critic_refit']):
        raise ValueError('Child is not the exact genuine BC and separate scalar-source lineage')
    child.validate_payload(payload, root, identity['continuation_binding'])
    for name in ('checkpoint', 'completed', 'runner_binding', 'learning_curve', 'training_request'):
        item = identity[name]
        bind(item['path'], item['sha256'])
    root_actor = second.adam_steps(root['payload']['optimizers']['actor'], required=True)
    root_q = second.adam_steps(root['payload']['optimizers']['critic'])
    first_settings = identity['continuation_binding']['settings']
    validate_settings(first_settings)
    first_updates = first_settings['total_updates']
    if (root_actor != 1000 or root['payload']['step'] != root_actor or root_q != 0 or
            second.adam_steps(payload['optimizers']['actor'], required=True) != root_actor + first_updates or
            second.adam_steps(payload['optimizers']['critic']) != 0):
        raise ValueError('Actual BC root/T1 Adam counters differ from the genuine prefix')
    binding = dict(version=VERSION, provider=child.VERSION, seed=seed, source_cycle=1,
        schema=copy.deepcopy(root['schema']), checkpoint=copy.deepcopy(identity['checkpoint']),
        first_continuation=copy.deepcopy(identity), root_source=copy.deepcopy(root['binding']),
        bc_actor_prefix=copy.deepcopy(root['binding']['actor_prefix']),
        scalar_source=copy.deepcopy(root['binding']['target_scalar_source']),
        scalar_critic_refit=copy.deepcopy(root['binding']['critic_refit']),
        prefix_counts=dict(root_actor_updates=root_actor, root_critic_updates=0,
            first_actor_updates=first_updates, first_critic_updates=0,
            actor_adam_steps=root_actor+first_updates, critic_adam_steps=0),
        state_identity='unmodified completed genuine BC T1; not FQL or an offline r0 checkpoint',
        critic_source='separately attached original scalar refit; frozen, not an original BC learned scalar critic')
    return dict(payload=payload, binding=binding, schema=root['schema'], root=root, refit=root['refit'])


def combine_replay(components, source, evaluation_noise_sha256):
    # This helper's complete native identity/role checks are shared with FQL;
    # the new binding truthfully labels the actor-only BC consumption semantics.
    replay = second.combine_replay(components, source, evaluation_noise_sha256)
    replay['binding']['version'] = VERSION
    replay['binding']['sampling'] = ('complete admitted own C0+C1 population retained for matched collection '
        'accounting; zero-Q BC does not sample or fit the critic')
    return replay


def second_binding(source, replay, settings):
    validate_settings(settings)
    if (source['binding'].get('provider') != child.VERSION or
            source['binding'].get('source_cycle') != 1 or
            source['binding']['checkpoint'] != source['binding']['first_continuation']['checkpoint'] or
            not replay.get('ready') or replay['binding']['schema'] != source['schema'] or
            replay['binding']['seed'] != source['binding']['seed'] or
            len(replay['binding']['components']) != 2 or replay['binding'].get('version') != VERSION):
        raise ValueError('Verified genuine BC T1 and complete C0+C1 replay are required')
    first_settings = source['binding']['first_continuation']['continuation_binding']['settings']
    for name in ('batch_size', 'alpha', 'flow_steps', 'grad_clip'):
        if settings[name] != first_settings[name]:
            raise ValueError('Second BC changed matched actor update settings: '+name)
    return dict(version=VERSION, cycle=2, parent=copy.deepcopy(source['binding']),
        replay=copy.deepcopy(replay['binding']), settings=copy.deepcopy(settings),
        source_sha256=sha(core.__file__), root_anchor=copy.deepcopy(source['root']['binding']),
        limitations='explicit second genuine BC cycle only; zero Q, frozen separate critic/teacher; no collection/export')


def cumulative_counts(source, settings, local_step):
    validate_settings(settings)
    return second.cumulative_counts(source, settings, local_step)


def audit_state(payload, source, binding):
    validate_settings(binding['settings'])
    counts = second.audit_state(payload, source, binding)
    first_bc.checked_frozen_source(payload, source['root'])
    if counts['critic_adam_steps'] != 0 or counts['second_critic_updates'] != 0:
        raise ValueError('BC must never update its separate scalar critic')
    return counts


def load_inputs(request):
    experiment, seed = Path(request['experiment']), request['seed']
    cfg, declaration, physical = child.settings(experiment, request['config_sha256'])
    bind(experiment/'declaration.json', request['declaration_sha256'])
    rows = child.verify_export(experiment, seed, cfg, declaration, physical)
    verified = child.checked_parent(cfg['parent_training'][str(seed)], seed)
    source = source_from_verified(cfg, verified, seed)
    first_request = first_bc.request_settings(read(cfg['parent_training'][str(seed)]['request']['path']))
    older_cfg, older_experiment = verified['cfg'], Path(first_request['experiment'])
    old_rows = original_export.verify_export(older_experiment, seed, older_cfg,
        verified['declaration'], verified['physical'])
    cache_path = RUNTIME/'benchmark_cache'/(cfg['design']+'.pt')
    bind(cache_path, cfg['cache_sha256'])
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    excluded = sorted({r['noise_sha256'] for r in (*old_rows, *rows) if r['role']=='evaluation'})
    components = []
    for round_index, physical_exp, pcfg, previous in (
            (0, older_experiment, older_cfg, first_request), (1, experiment, cfg, request)):
        parent = source['root']['binding']['checkpoint'] if round_index==0 else source['binding']['checkpoint']
        collect = pcfg['streams'][str(seed)]['collect']
        if source['schema']['kind']=='rule' and previous['final_outcomes']:
            raise ValueError('Rule BC cannot claim final-route admission')
        replay = core.admit_collect(physical_exp, seed=seed, source_round=round_index,
            count=collect['count'], tasks_sha256=previous['tasks_sha256'],
            config_sha256=previous['config_sha256'], checkpoint_sha256=parent['sha256'],
            collection_noise_seed=collect['seed'], evaluation_noise_sha256=excluded,
            value=source['schema'], cache=cache,
            final_outcomes={int(k):v for k,v in previous['final_outcomes'].items()})
        components.append(dict(replay=replay, experiment=str(physical_exp), config=pcfg, source_round=round_index))
    replay = combine_replay(components, source, excluded)
    binding = second_binding(source, replay, request['settings'])
    output = Path(request['output'])
    for old in (experiment, older_experiment, Path(first_request['output']),
                Path(older_cfg['source_experiment']), Path(older_cfg['scalar_experiment'])):
        if output==old or output in old.parents or old in output.parents:
            raise ValueError('Second BC output overlaps a real ancestor or either original source')
    return cfg, declaration, source, replay, binding


def save_state(folder, continuation, source, runner_binding, history):
    payload = continuation.state()
    counts = audit_state(payload, source, continuation.binding)
    payload.update(runner_binding=copy.deepcopy(runner_binding), history=copy.deepcopy(history),
        cumulative_steps=counts,
        step_semantics='segment-local step/parent_steps; cumulative_steps records actual BC actor Adam totals')
    atomic_torch(folder/'resume.pt', payload)
    atomic_json(folder/'learning_curve.json', history)
    return payload


def checked_resume(folder, runner_binding, binding, source):
    payload = common.checked_resume(folder, runner_binding, binding)
    if payload is not None:
        counts = audit_state(payload, source, binding)
        if payload.get('cumulative_steps') != counts:
            raise ValueError('Saved cumulative BC counters differ from actual Adam')
        for row in payload['history']:
            if (row.get('cycle') != 2 or row.get('critic_updates') != 0 or
                    row.get('cumulative_steps') != cumulative_counts(source, binding['settings'], row['step'])):
                raise ValueError('Second BC history/cumulative counters changed')
    return payload


def completion_record(folder, binding, runner_binding, payload):
    return dict(status='completed', version=VERSION, cycle=2, steps=payload['step'],
        core_parent_segment_steps=payload['parent_steps'], cumulative_steps=payload['cumulative_steps'],
        checkpoint=bind(folder/'checkpoint.pt'), binding=runner_binding, continuation_binding=binding,
        parent_checkpoint=binding['parent']['checkpoint'], root_checkpoint=binding['root_anchor']['checkpoint'],
        bc_actor_prefix=binding['parent']['bc_actor_prefix'],
        separate_scalar_refit=binding['parent']['scalar_critic_refit'],
        replay_counts=[v['count'] for v in binding['replay']['components']],
        actor_updates=payload['step'], critic_updates=0, teacher_updates=0,
        q_weight=0., q_units=core.Q_UNITS, backend_calls=0,
        scope='second measured genuine BC cycle; separately attached scalar critic unchanged; no child export')


def checked_completion(folder, payload, runner_binding, binding):
    if (payload is None or payload['step'] != binding['settings']['total_updates'] or
            read(folder/'completed.json') != completion_record(folder, binding, runner_binding, payload)):
        raise ValueError('Second BC completion differs from its true full checkpoint')


def audit_live_frozen(continuation, source):
    original = source['root']['payload']
    for name in ('flow', 'critic'):
        core.equal_tree(continuation.models[name].state_dict(), original[name])
        core.equal_tree(continuation.opts[name].state_dict(), original['optimizers'][name])
    for name in ('q', 'flow'):
        core.equal_tree(continuation.generators[name].get_state(), original['generators'][name])


def run_loop(folder, continuation, source, offline, replay, runner_binding, history=(), *, diagnostic_interval):
    validate_settings(continuation.binding['settings'])
    audit_live_frozen(continuation, source)
    folder.mkdir(parents=True, exist_ok=True)
    history, settings = list(history), continuation.binding['settings']
    updates = {k:v for k,v in settings.items() if k!='total_updates'}
    while continuation.step < settings['total_updates']:
        row = continuation.update(offline, replay, **updates)
        row.update(cycle=2, cumulative_steps=cumulative_counts(source, settings, row['step']))
        if row['critic_updates'] != 0:
            raise ValueError('Zero-Q BC unexpectedly updated the critic')
        if row['step']==1 or row['step']%diagnostic_interval==0 or row['step']==settings['total_updates']:
            row['gradient_diagnostics'] = common.gradient_probe(continuation)
        audit_live_frozen(continuation, source)
        history.append(row)
        if row['step']%50==0 or row['step']==settings['total_updates'] or STOP:
            save_state(folder, continuation, source, runner_binding, history)
            print(json.dumps(row, allow_nan=False), flush=True)
        if STOP:
            raise SystemExit(85)
    payload = save_state(folder, continuation, source, runner_binding, history)
    (folder/'resume.pt').replace(folder/'checkpoint.pt')
    atomic_json(folder/'completed.json', completion_record(folder, continuation.binding, runner_binding, payload))


def run(request_path, request_sha256):
    core.slurm()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual graph BC continuation requires allocated Slurm GPU')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    request_binding = bind(request_path, request_sha256)
    request = request_settings(read(request_path))
    cfg, declaration, source, replay, binding = load_inputs(request)
    output = Path(request['output'])
    original_cfg = read(source['root']['binding']['config']['path'])
    runner_binding = dict(version=VERSION, cycle=2, request=request_binding,
        config=bind(Path(request['experiment'])/'config.json'),
        declaration=bind(Path(request['experiment'])/'declaration.json'), continuation=binding,
        source_sha256={name:sha(ROOT/name) for name in SOURCE_NAMES},
        diagnostic_interval=request['diagnostic_interval'], backend_calls=0)
    offline = common.offline_labels(source['root'])
    problem = load_problem(cfg['design'], 'cuda')
    for data in (offline, replay):
        data['x'] = data['x'].reshape(-1, len(problem.x), 2).cuda()
        data['derived_B'] = data['derived_B'].float().cuda()
        data['mask'] = data['mask'].cuda()
    models = dict(flow=GraphPolicy(problem), actor=GraphPolicy(problem), critic=source['refit']['model'].cuda())
    optimizers = dict(flow=torch.optim.Adam(models['flow'].parameters(), lr=original_cfg['flow_lr']),
        actor=torch.optim.Adam(models['actor'].parameters(), lr=original_cfg['actor_lr']),
        critic=torch.optim.AdamW(models['critic'].parameters(), lr=original_cfg['critic_lr'], weight_decay=1e-4))
    generators = {name:torch.Generator(device='cuda') for name in core.GENERATOR_KEYS}
    output.mkdir(parents=True, exist_ok=True)
    with (output/'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = output/'binding.json'
        if path.exists() and read(path)!=runner_binding:
            raise ValueError('Second BC output belongs to different sources/settings')
        atomic_json(path, runner_binding)
        for name,digest in runner_binding['source_sha256'].items():
            snapshot = output/'sources'/name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                shutil.copyfile(ROOT/name, snapshot)
            bind(snapshot, digest)
        payload = checked_resume(output, runner_binding, binding, source)
        if (output/'completed.json').exists():
            checked_completion(output, payload, runner_binding, binding)
            return
        continuation = core.Continuation(models, optimizers, generators,
            source['payload'] if payload is None else payload, binding,
            frozen_reference=source, resume=payload is not None)
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, stop)
        run_loop(output, continuation, source, offline, replay, runner_binding,
            () if payload is None else payload['history'], diagnostic_interval=request['diagnostic_interval'])


def stop(*_):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    run(args.request.resolve(), args.request_sha256)


if __name__ == '__main__':
    main()
