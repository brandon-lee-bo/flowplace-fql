"""One measured scalar-reward continuation, with atomic state and diagnostics.

This independent runner accepts a separately pinned, explicit request after
collect export and backend measurements. It selects no reward, budget, noise,
checkpoint or layout. Only a genuine rule/final offline FQL source is supported;
the BC two-source provider and later online cycles require separate adapters.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

# Resolve our local analysis package before baseline adds upstream import paths.
from . import qstudy_scalar_online_export as exporter
from baseline.api import RUNTIME, load_problem
from .graph_o2o_archive import atomic_json
from .graph_o2o_models import GraphPolicy
from .qstudy_data import bind, sha
from .qstudy_models import frozen_critic
from .qstudy_screen import atomic_torch
from . import qstudy_scalar_online as core

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qstudy_scalar_online_train_v1'
SOURCE_NAMES = ('fql/qstudy_scalar_online_train.py', 'fql/qstudy_scalar_online_train.slurm',
    'fql/qstudy_scalar_online.py', 'fql/qstudy_scalar_online_export.py',
    'fql/qstudy_models.py', 'fql/graph_o2o_models.py', 'fql/qstudy_screen.py')
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def request_settings(value):
    required = {'version', 'experiment', 'config_sha256', 'declaration_sha256',
        'seed', 'tasks_sha256', 'final_outcomes', 'settings', 'diagnostic_interval', 'output'}
    if (set(value) != required or value['version'] != VERSION or
            type(value['seed']) is not int or value['seed'] not in exporter.SEEDS or
            type(value['diagnostic_interval']) is not int or value['diagnostic_interval'] < 1):
        raise ValueError('Explicit complete first-continuation request required')
    for key in ('experiment', 'output'):
        if not Path(value[key]).is_absolute() or str(Path(value[key]).resolve()) != value[key]:
            raise ValueError('Canonical absolute input/output paths required')
    source, output = Path(value['experiment']), Path(value['output'])
    if (source == output or source in output.parents or output in source.parents or
            output == ROOT or output in ROOT.parents or ROOT in output.parents):
        raise ValueError('Training output must be disjoint from physical inputs and source checkout')
    if not isinstance(value['final_outcomes'], dict):
        raise ValueError('Externally bound collect-only final outcomes must be explicit')
    for key, identity in value['final_outcomes'].items():
        if (not key.isdecimal() or str(int(key)) != key or
                set(identity) != {'path', 'sha256'} or not Path(identity['path']).is_absolute()):
            raise ValueError('Final outcome keys/bindings differ from explicit collect indices')
    return value


def offline_labels(source):
    context = source['refit']['context']
    labels = core.targets(source['schema'], context['cost_values'], context['statuses'], context['cost_mask'])
    if not bool(labels['valid'].all()):
        raise ValueError('The original full TRAIN population is not wholly admitted')
    return dict(x=context['x'].clone(), **labels)


def gradient_probe(continuation):
    """Peek at the next actor noise without consuming any training RNG state.

    Diagnostics are evaluated after the saved update, not represented as that
    update's exact gradients. The cosine compares grad(alpha*D) and grad(+Q);
    actor descent on the total loss follows -grad(alpha*D) + grad(+Q).
    """
    actor, flow, critic = (continuation.models[k] for k in ('actor', 'flow', 'critic'))
    device = next(actor.parameters()).device
    settings = continuation.binding['settings']
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    generator = torch.Generator(device=device)
    generator.set_state(continuation.generators['actor'].get_state())
    with torch.random.fork_rng(devices=devices):
        noise = 2*torch.rand(settings['batch_size'], len(actor.fixed), 2,
            device=device, generator=generator)-1
        with torch.no_grad():
            teacher = flow.sample_flow(noise, settings['flow_steps'])
        prediction = actor(noise, clip=False)
        distill = (prediction[:,actor.movable]-teacher[:,actor.movable]).square().mean()
        with frozen_critic(critic):
            q = core.scalar_q(critic, actor.restore(prediction, clip=True)).mean()
        first, second = settings['alpha']*distill, settings['q_weight']*q
        tensors = tuple(actor.parameters())
        a = torch.autograd.grad(first, tensors, retain_graph=True, allow_unused=True)
        b = torch.autograd.grad(second, tensors, allow_unused=True)
        aa = sum(float(v.detach().square().sum()) for v in a if v is not None)
        bb = sum(float(v.detach().square().sum()) for v in b if v is not None)
        ab = sum(float(u.detach().mul(v.detach()).sum()) for u,v in zip(a,b) if u is not None and v is not None)
        norms = (aa**.5, bb**.5)
        values = dict(distill=float(distill.detach()), q=float(q.detach()),
            weighted_distill=float(first.detach()), weighted_q=float(second.detach()),
            distill_grad_norm=norms[0], q_ascent_grad_norm=norms[1],
            q_to_distill_grad_ratio=norms[1]/max(norms[0],1e-20),
            gradient_cosine=ab/max((aa*bb)**.5,1e-20))
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite scalar actor gradient diagnostics')
    return dict(phase='after_update_peek_next_noise', consumes_training_rng=False,
        q_units=core.Q_UNITS, **values)


def save_state(folder, continuation, runner_binding, history):
    payload = continuation.state()
    payload.update(runner_binding=copy.deepcopy(runner_binding), history=copy.deepcopy(history))
    atomic_torch(folder/'resume.pt', payload)
    # This view is not authoritative; a restart always regenerates it from pt.
    atomic_json(folder/'learning_curve.json', history)
    return payload


def checked_resume(folder, runner_binding, continuation_binding):
    path = folder/'checkpoint.pt' if (folder/'checkpoint.pt').exists() else folder/'resume.pt'
    if not path.exists():
        return None
    payload = torch.load(path, map_location='cpu', weights_only=False)
    history = payload.get('history')
    if (payload.get('runner_binding') != runner_binding or payload.get('binding') != continuation_binding or
            not isinstance(history,list) or [v['step'] for v in history] != list(range(1,payload['step']+1)) or
            not all(v.get('q_units') == core.Q_UNITS for v in history)):
        raise ValueError('Continuation checkpoint changed its inputs, settings or complete step history')
    return payload


def completion_record(folder, binding, runner_binding, step, parent_steps):
    settings=binding['settings']
    return dict(status='completed',version=VERSION,steps=step,parent_steps=parent_steps,
        binding=runner_binding,continuation_binding=binding,checkpoint=bind(folder/'checkpoint.pt'),
        teacher_updates=0,actor_updates=step,
        critic_updates=step if settings['update_critic'] and settings['q_weight']>0 else 0,
        q_units=core.Q_UNITS,backend_calls=0,
        scope='one measured offline-parent continuation; no child export, next cycle or BC provider')


def checked_completion(folder, payload, runner_binding, binding):
    if (payload is None or payload['step']!=binding['settings']['total_updates'] or
            read(folder/'completed.json')!=completion_record(folder,binding,runner_binding,
                payload['step'],payload['parent_steps'])):
        raise ValueError('Completed continuation marker/checkpoint changed')


def run_loop(folder, continuation, offline, replay, runner_binding, history=(), *, diagnostic_interval):
    """Recoverable exact core updates; useful independently in bounded fixtures."""
    folder.mkdir(parents=True,exist_ok=True)
    history = list(history)
    settings = continuation.binding['settings']
    updates = {k:v for k,v in settings.items() if k != 'total_updates'}
    while continuation.step < settings['total_updates']:
        row = continuation.update(offline,replay,**updates)
        if row['step'] == 1 or row['step'] % diagnostic_interval == 0 or row['step'] == settings['total_updates']:
            row['gradient_diagnostics'] = gradient_probe(continuation)
        history.append(row)
        if row['step'] % 50 == 0 or row['step'] == settings['total_updates'] or STOP:
            save_state(folder,continuation,runner_binding,history)
            print(json.dumps(row,allow_nan=False),flush=True)
        if STOP:
            raise SystemExit(85)
    # Includes a resume at the exact last update before publication.
    save_state(folder,continuation,runner_binding,history)
    (folder/'resume.pt').replace(folder/'checkpoint.pt')
    atomic_json(folder/'completed.json',completion_record(folder,continuation.binding,
        runner_binding,continuation.step,continuation.parent_steps))


def run(request_path, request_sha256):
    core.slurm()
    if not torch.cuda.is_available():
        raise RuntimeError('Actual graph continuation requires an allocated Slurm GPU')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    request_binding = bind(request_path,request_sha256)
    request = request_settings(read(request_path))
    experiment, output = (Path(request[k]) for k in ('experiment','output'))
    cfg,declaration,_ = exporter.settings(experiment,request['config_sha256'])
    bind(experiment/'declaration.json',request['declaration_sha256'])
    seed = request['seed']
    rows = exporter.verify_export(experiment,seed,cfg,declaration,
        exporter.context(experiment)[2])
    parent = cfg['parent_identities'][str(seed)]['checkpoint']
    source = core.load_offline_source(cfg['source_kind'],cfg['source_experiment'],seed,parent['sha256'])
    if source['binding'] != cfg['verified_parents'][str(seed)] or source['schema'] != cfg['scalar_schema']:
        raise ValueError('Exported collector and loaded offline scalar parent differ')
    old_experiment = Path(cfg['source_experiment'])
    if output == old_experiment or output in old_experiment.parents or old_experiment in output.parents:
        raise ValueError('Training output may not overwrite/nest the original source experiment')
    original_cfg = read(old_experiment/'config.json')
    cache_path = RUNTIME/'benchmark_cache'/(cfg['design']+'.pt')
    bind(cache_path,cfg['cache_sha256'])
    cache = torch.load(cache_path,map_location='cpu',weights_only=False)
    collect = cfg['streams'][str(seed)]['collect']
    replay = core.admit_collect(experiment,seed=seed,source_round=0,count=collect['count'],
        tasks_sha256=request['tasks_sha256'],config_sha256=request['config_sha256'],
        checkpoint_sha256=parent['sha256'],collection_noise_seed=collect['seed'],
        evaluation_noise_sha256=[r['noise_sha256'] for r in rows if r['role']=='evaluation'],
        value=source['schema'],cache=cache,
        final_outcomes={int(k):v for k,v in request['final_outcomes'].items()})
    if source['schema']['kind']=='rule' and request['final_outcomes']:
        raise ValueError('Rule-only continuation must not claim final-route label admission')
    binding = core.continuation_binding(source,replay,request['settings'])
    runner_binding = dict(version=VERSION,request=request_binding,config=bind(experiment/'config.json'),
        declaration=bind(experiment/'declaration.json'),continuation=binding,
        source_sha256={name:sha(ROOT/name) for name in SOURCE_NAMES},
        diagnostic_interval=request['diagnostic_interval'],backend_calls=0)
    offline = offline_labels(source)
    problem = load_problem(cfg['design'],'cuda')
    for data in (offline,replay):
        data['x'] = data['x'].reshape(-1,len(problem.x),2).cuda()
        data['derived_B'] = data['derived_B'].float().cuda()
        data['mask'] = data['mask'].cuda()
    models = dict(flow=GraphPolicy(problem),actor=GraphPolicy(problem),critic=source['refit']['model'].cuda())
    optimizers = dict(flow=torch.optim.Adam(models['flow'].parameters(),lr=original_cfg['flow_lr']),
        actor=torch.optim.Adam(models['actor'].parameters(),lr=original_cfg['actor_lr']),
        critic=torch.optim.AdamW(models['critic'].parameters(),lr=original_cfg['critic_lr'],weight_decay=1e-4))
    generators = {name:torch.Generator(device='cuda') for name in core.GENERATOR_KEYS}
    output.mkdir(parents=True,exist_ok=True)
    with (output/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding_path=output/'binding.json'
        if binding_path.exists() and read(binding_path)!=runner_binding:
            raise ValueError('Training output already belongs to different inputs/settings')
        atomic_json(binding_path,runner_binding)
        payload = checked_resume(output,runner_binding,binding)
        if (output/'completed.json').exists():
            checked_completion(output,payload,runner_binding,binding)
            return
        continuation = core.Continuation(models,optimizers,generators,
            source['payload'] if payload is None else payload,binding,frozen_reference=source,resume=payload is not None)
        for name in (signal.SIGTERM,signal.SIGUSR1):
            signal.signal(name,stop)
        run_loop(output,continuation,offline,replay,runner_binding,
            () if payload is None else payload['history'],diagnostic_interval=request['diagnostic_interval'])


def stop(*_):
    global STOP
    STOP=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args()
    run(args.request.resolve(),args.request_sha256)


if __name__=='__main__':
    main()
