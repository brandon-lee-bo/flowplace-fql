"""First measured BC continuation from the genuine two-source BC exporter.

This is a zero-Q matched control, not an FQL source disguised as BC. Its actor,
actor Adam, frozen teacher and RNG prefix are the actual old BC states; the
separately verified scalar critic remains diagnostic and entirely unchanged.
All collect outcomes must nevertheless be measured and admitted, preserving
the explicit collection budget and preventing evaluation replay.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

# Resolve local analysis before upstream FlowPlace adds its analysis.py.
from . import qstudy_scalar_online_bc_export as exporter
from . import qstudy_scalar_online_train as common
from . import qstudy_scalar_online as core
from baseline.api import RUNTIME, load_problem
from .graph_o2o_archive import atomic_json
from .graph_o2o_models import GraphPolicy
from .qstudy_data import bind, sha

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qstudy_scalar_online_bc_train_v1'
SOURCE_NAMES = tuple(dict.fromkeys((*common.SOURCE_NAMES,
    'fql/qstudy_scalar_online_bc.py', 'fql/qstudy_scalar_online_bc_export.py',
    'fql/qstudy_scalar_online_bc_train.py', 'fql/qstudy_scalar_online_bc_train.slurm')))
STOP = False


def request_settings(value):
    if value.get('version') != VERSION:
        raise ValueError('Explicit BC-only continuation request required')
    # The original pure validator checks the common request body. This local
    # translation neither changes the bound request nor any saved source.
    common.request_settings(dict(value, version=common.VERSION))
    settings = value['settings']
    if settings.get('q_weight') != 0. or settings.get('update_critic') is not False:
        raise ValueError('Genuine BC control requires zero Q and a frozen critic')
    return value


def completion_record(folder, binding, runner_binding, step, parent_steps):
    return dict(status='completed', version=VERSION, steps=step,
        parent_steps=parent_steps, binding=runner_binding,
        continuation_binding=binding, checkpoint=bind(folder/'checkpoint.pt'),
        teacher_updates=0, actor_updates=step, critic_updates=0,
        q_units=core.Q_UNITS, backend_calls=0,
        scope='first zero-Q continuation of a genuine BC actor; separate scalar critic is unchanged')


def checked_completion(folder, payload, runner_binding, binding):
    if (payload is None or payload['step'] != binding['settings']['total_updates'] or
            common.read(folder/'completed.json') != completion_record(folder,
                binding, runner_binding, payload['step'], payload['parent_steps'])):
        raise ValueError('BC completed marker/checkpoint changed')


def checked_frozen_source(payload, source):
    """BC may resume its actor, but never a modified diagnostic Q or teacher."""
    if payload is None:
        return
    for name in ('flow', 'critic'):
        core.equal_tree(payload[name], source['payload'][name])
        core.equal_tree(payload['optimizers'][name], source['payload']['optimizers'][name])
    for name in ('q', 'flow'):
        core.equal_tree(payload['generators'][name], source['payload']['generators'][name])


def run_loop(folder, continuation, offline, replay, runner_binding, history=(), *, diagnostic_interval):
    if (continuation.binding['settings']['q_weight'] != 0. or
            continuation.binding['settings']['update_critic'] is not False):
        raise ValueError('BC loop must not update or optimize its diagnostic critic')
    folder.mkdir(parents=True, exist_ok=True)
    history = list(history)
    settings = continuation.binding['settings']
    updates = {k:v for k,v in settings.items() if k != 'total_updates'}
    original_critic = copy.deepcopy(continuation.models['critic'].state_dict())
    original_critic_optimizer = copy.deepcopy(continuation.opts['critic'].state_dict())
    while continuation.step < settings['total_updates']:
        row = continuation.update(offline, replay, **updates)
        if row['step'] == 1 or row['step'] % diagnostic_interval == 0 or row['step'] == settings['total_updates']:
            row['gradient_diagnostics'] = common.gradient_probe(continuation)
        if row['critic_updates'] != 0:
            raise ValueError('Zero-Q BC unexpectedly updated its critic')
        core.equal_tree(original_critic, continuation.models['critic'].state_dict())
        core.equal_tree(original_critic_optimizer, continuation.opts['critic'].state_dict())
        history.append(row)
        if row['step'] % 50 == 0 or row['step'] == settings['total_updates'] or STOP:
            common.save_state(folder, continuation, runner_binding, history)
            print(json.dumps(row, allow_nan=False), flush=True)
        if STOP:
            raise SystemExit(85)
    common.save_state(folder, continuation, runner_binding, history)
    (folder/'resume.pt').replace(folder/'checkpoint.pt')
    atomic_json(folder/'completed.json', completion_record(folder, continuation.binding,
        runner_binding, continuation.step, continuation.parent_steps))


def run(request_path, request_sha256):
    core.slurm()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual graph BC continuation requires an allocated Slurm GPU')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    request_binding = bind(request_path, request_sha256)
    request = request_settings(common.read(request_path))
    experiment, output = (Path(request[k]) for k in ('experiment', 'output'))
    cfg, declaration, physical = exporter.settings(experiment, request['config_sha256'])
    bind(experiment/'declaration.json', request['declaration_sha256'])
    seed = request['seed']
    rows = exporter.verify_export(experiment, seed, cfg, declaration, physical)
    source_request = common.read(cfg['prospective_request']['path'])
    source = exporter.load_source(source_request, seed)
    original_cfg = common.read(Path(source_request['bc_experiment'])/'config.json')
    if (source['binding'] != cfg['verified_parents'][str(seed)] or
            source['schema'] != cfg['scalar_schema']):
        raise ValueError('BC collector and the two independently verified source prefixes differ')
    for key in ('bc_experiment', 'scalar_experiment'):
        old = Path(source_request[key])
        if output == old or output in old.parents or old in output.parents:
            raise ValueError('BC training output may not overwrite or nest either original source')
    if (request['settings']['alpha'] != cfg['alpha'] or
            request['settings']['flow_steps'] != original_cfg['flow_steps'] or
            request['settings']['batch_size'] != original_cfg['batch_size']):
        raise ValueError('Matched BC control must retain its declared offline actor update settings')
    cache_path = RUNTIME/'benchmark_cache'/(cfg['design']+'.pt')
    bind(cache_path, cfg['cache_sha256'])
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    stream = cfg['streams'][str(seed)]['collect']
    replay = core.admit_collect(experiment, seed=seed, source_round=0,
        count=stream['count'], tasks_sha256=request['tasks_sha256'],
        config_sha256=request['config_sha256'],
        checkpoint_sha256=cfg['parent_identities'][str(seed)]['checkpoint']['sha256'],
        collection_noise_seed=stream['seed'],
        evaluation_noise_sha256=[r['noise_sha256'] for r in rows if r['role']=='evaluation'],
        value=source['schema'], cache=cache,
        final_outcomes={int(k):v for k,v in request['final_outcomes'].items()})
    if source['schema']['kind']=='rule' and request['final_outcomes']:
        raise ValueError('Rule BC control must not claim final-route evidence')
    binding = core.continuation_binding(source, replay, request['settings'])
    runner_binding = dict(version=VERSION, request=request_binding,
        config=bind(experiment/'config.json'), declaration=bind(experiment/'declaration.json'),
        continuation=binding, source_sha256={name:sha(ROOT/name) for name in SOURCE_NAMES},
        diagnostic_interval=request['diagnostic_interval'], backend_calls=0)
    offline = common.offline_labels(source)
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
        if path.exists() and common.read(path) != runner_binding:
            raise ValueError('BC output already belongs to different sources/settings')
        atomic_json(path, runner_binding)
        payload = common.checked_resume(output, runner_binding, binding)
        checked_frozen_source(payload, source)
        if (output/'completed.json').exists():
            checked_completion(output, payload, runner_binding, binding)
            return
        continuation = core.Continuation(models, optimizers, generators,
            source['payload'] if payload is None else payload, binding,
            frozen_reference=source, resume=payload is not None)
        for name in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(name, stop)
        run_loop(output, continuation, offline, replay, runner_binding,
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
