"""One-step FQL actor study with fixed offline critics and measured online replay.

The flow and actor start from matching, already audited common warmups. A
screened terminal critic is fixed during offline actor fitting; online updates
refit it using own measured feedback. Teacher updates are an explicit factor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import RUNTIME, CHECKPOINT, load_problem, load_model, generate, legalize, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_models import GraphPolicy, FeatureCritics
from .qstudy_data import load_dataset, METRIC_NAMES
from .qstudy_models import MultiMetricCritic, measured_costs, frozen_critic
from .qstudy_screen import VARIANTS, make_model, atomic_torch

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def action_hash(x):
    return hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).hexdigest()


def setup(experiment, seed):
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Policy training and inference require a Slurm GPU')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    cfg = json.loads((experiment/'config.json').read_text())
    if seed not in cfg['seeds']:
        raise ValueError('Undeclared training seed')
    dataset = Path(cfg['dataset_path'])
    if sha(dataset) != cfg['dataset_sha256']:
        raise ValueError('Audited TRAIN data changed')
    data, manifest = load_dataset(dataset)
    problem = load_problem(cfg['design'], 'cuda')
    cache_path = RUNTIME/'benchmark_cache'/(cfg['design']+'.pt')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    if manifest['design'] != cfg['design'] or sha(cache_path) != cfg['cache_sha256']:
        raise ValueError('Graph context changed')
    shape = (-1, len(problem.x), 2)
    x = torch.tensor(data['train']['raw_action'], device='cuda').reshape(shape)
    costs, mask = measured_costs(torch.tensor(data['train']['metrics'][:, :5], dtype=torch.float32, device='cuda'),
        METRIC_NAMES[:5], torch.tensor(data['train']['metric_mask'][:, :5], device='cuda'))
    behavior = torch.tensor(data['behavior_action'], device='cuda').reshape(shape)
    families = torch.tensor(data['train']['family'], device='cuda')
    critic, _ = make_model(cache, x, costs, mask, ~problem.cond.is_ports.bool(),
        cfg['critic_variant'], families, seed)
    critic = critic.cuda()
    return cfg, problem, x, costs, mask, behavior, critic, families


def reward(critic, action, costs, cfg):
    if cfg['critic_variant'] == 'legacy':
        if cfg['weights'] != [1, 0, 0, 0, 0]:
            raise ValueError('Scalar WL legacy critic cannot optimize other objectives')
        value = critic(action)
        return value.mean(1)-cfg.get('uncertainty_beta', 0.)*value.std(1, unbiased=False)
    return critic.reward(action, cfg['weights'], cfg.get('uncertainty_beta', 0.))


def read_replay(experiment, seed, round_number, cfg, shape):
    from types import SimpleNamespace
    from .qstudy_backend import context, checked_request, checked_existing
    backend_cfg, verification, backend_binding = context(experiment)
    rows = []
    for r in range(round_number):
        tasks = experiment/f'seed_{seed}'/f'round_{r}'/'requests/tasks.jsonl'
        marker = json.loads(tasks.with_name('completed.json').read_text())
        if sha(tasks) != marker['tasks_sha256'] or marker['config_sha256'] != sha(experiment/'config.json'):
            raise ValueError('Online request manifest changed')
        requests = [json.loads(line) for line in tasks.read_text().splitlines()]
        selected = [q for q in requests if q['role'] == 'collect']
        if len(selected) != cfg['online_samples']:
            raise ValueError('Online collection budget differs')
        for q in selected:
            sample = checked_request(q, SimpleNamespace(seed=seed, round=r), backend_cfg, backend_binding)
            out_path = Path(q['sample_dir'])/'outcome.json'
            out = json.loads(out_path.read_text())
            checked_existing(out, q, sample, tasks, backend_binding, backend_cfg, verification)
            if (sha(out_path.with_name('request.json')) != out['request_sha256'] or
                out['action_sha256'] != q['action_sha256'] or out['raw_action'] != q['raw_action'] or
                not q['training_feedback_eligible'] or out['arm'] != cfg['arm'] or
                out['seed'] != seed or out['round'] != r or out['terminal_mask'] != 0 or
                any(out.get(k) != q[k] for k in ['role', 'index', 'training_feedback_eligible'])):
                raise ValueError('Foreign/changed/evaluation feedback cannot enter online replay')
            if out['status'] not in ['success', 'geometry_failure', 'physical_failure']:
                raise ValueError('Infrastructure failure has no reward label')
            if out['status'] == 'success':
                run = Path(out['physical_run'])
                if sha(run/'manifest.json') != out['physical_manifest_sha256'] or sha(run/'metrics.json') != out['metrics_sha256']:
                    raise ValueError('Native feedback evidence changed')
            rows.append(dict(out, outcome_path=str(out_path), outcome_sha256=sha(out_path)))
    if not rows:
        return None
    x = torch.tensor([r['raw_action'] for r in rows], dtype=torch.float32, device='cuda').reshape(shape)
    metrics = [[r['metrics'].get(k, float('nan')) if r['metrics'] is not None else float('nan')
                for k in METRIC_NAMES[:5]] for r in rows]
    costs, mask = measured_costs(torch.tensor(metrics, dtype=torch.float32, device='cuda'), METRIC_NAMES[:5])
    success = torch.tensor([r['status'] == 'success' for r in rows], device='cuda')
    return dict(x=x, costs=costs, mask=mask, success=success, bindings=[
        dict(path=r['outcome_path'], sha256=r['outcome_sha256'], status=r['status']) for r in rows])


def gradient_norms(first, second, tensors):
    a = torch.autograd.grad(first, tensors, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(second, tensors, retain_graph=True, allow_unused=True)
    aa = sum(float(t.detach().square().sum()) for t in a if t is not None)
    bb = sum(float(t.detach().square().sum()) for t in b if t is not None)
    ab = sum(float(u.detach().mul(v.detach()).sum()) for u,v in zip(a,b) if u is not None and v is not None)
    return dict(distill_grad_norm=aa**.5, q_grad_norm=bb**.5,
                gradient_cosine=ab/max((aa*bb)**.5, 1e-20))


def train(args):
    exp = args.experiment.resolve()
    cfg, problem, x, costs, mask, behavior, critic, families = setup(exp, args.seed)
    folder = exp/f'seed_{args.seed}'/f'round_{args.round}'
    folder.mkdir(parents=True, exist_ok=True)
    flow, actor = GraphPolicy(problem), GraphPolicy(problem)
    opts = dict(actor=torch.optim.Adam(actor.parameters(), lr=cfg.get('actor_lr', 1e-5)),
        flow=torch.optim.Adam(flow.parameters(), lr=cfg.get('flow_lr', 1e-5)),
        critic=torch.optim.AdamW(critic.parameters(), lr=cfg.get('critic_lr', 3e-4), weight_decay=1e-4))
    replay = read_replay(exp, args.seed, args.round, cfg, (-1, len(problem.x), 2))
    replay_binding = replay['bindings'] if replay is not None else []
    common = Path(cfg['common_checkpoints'][str(args.seed)])
    qpath = Path(cfg['critic_checkpoints'][str(args.seed)])
    for label, path in [('common', common), ('critic', qpath)]:
        if sha(path) != cfg[label+'_checkpoint_sha256'][str(args.seed)]:
            raise ValueError('Frozen initialization changed: '+label)
    parent = exp/f'seed_{args.seed}'/f'round_{args.round-1}'/'checkpoint.pt'
    parent_binding = None
    if args.round:
        parent_done = json.loads(parent.with_name('completed.json').read_text())
        if parent_done['checkpoint_sha256'] != sha(parent):
            raise ValueError('Completed online parent checkpoint changed')
        parent_binding = dict(checkpoint_sha256=sha(parent),
            completed_sha256=sha(parent.with_name('completed.json')),
            binding_sha256=sha(parent.with_name('binding.json')))
    binding = dict(config_sha256=sha(exp/'config.json'), dataset_sha256=sha(cfg['dataset_path']),
        common_sha256=sha(common), initial_critic_sha256=sha(qpath), round=args.round, seed=args.seed,
        replay=replay_binding, parent=parent_binding, source_sha256={n: sha(ROOT/n) for n in [
            'fql/qstudy_actor.py', 'fql/qstudy_models.py', 'fql/qstudy_screen.py',
            'fql/graph_o2o_models.py', 'baseline/api.py']})
    binding_path = folder/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Policy training binding changed')
    atomic_json(binding_path, binding)
    if (folder/'completed.json').exists():
        if json.loads((folder/'completed.json').read_text())['checkpoint_sha256'] != sha(folder/'checkpoint.pt'):
            raise ValueError('Completed policy checkpoint changed')
        return
    resume = folder/'resume.pt'
    recovery = resume if resume.exists() else folder/'checkpoint.pt'
    saved = None
    begin = 0
    if recovery.exists():
        saved = torch.load(recovery, map_location='cuda', weights_only=False)
        if saved['binding'] != binding:
            raise ValueError('Resume binding changed')
        begin = saved['step']
    elif args.round:
        saved = torch.load(parent, map_location='cuda', weights_only=False)
        if saved['binding'] != parent_done['binding'] or saved['binding']['round'] != args.round-1 or saved['binding']['seed'] != args.seed:
            raise ValueError('Online parent identity changed')
    else:
        source = torch.load(common, map_location='cuda', weights_only=False)
        flow.load_state_dict(source['flow'])
        actor.load_state_dict(source['actor'])
        critic.load_state_dict(torch.load(qpath, map_location='cuda', weights_only=False)['model'])
        del source
    if saved:
        for name, model in [('flow', flow), ('actor', actor), ('critic', critic)]:
            model.load_state_dict(saved[name])
            opts[name].load_state_dict(saved['optimizers'][name])
    # Paired update stream independent of critic architecture and init RNG.
    torch.manual_seed(530000+args.seed*100+args.round)
    generators = {name: torch.Generator(device=x.device).manual_seed(530000+args.seed*100+args.round+offset)
                  for name, offset in [('q', 1000000), ('actor', 2000000), ('flow', 3000000)]}
    if recovery.exists():
        torch.set_rng_state(saved['rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
        for name, generator in generators.items():
            generator.set_state(saved['generators'][name].cpu())
    total = cfg['offline_actor_steps'] if args.round == 0 else cfg['online_updates']
    started = time.monotonic()
    for sig in [signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(sig, stop)
    for step in range(begin, total):
        batch = cfg.get('batch_size', 16)
        # Real Q updates in online phase only; initial Q is independently fitted.
        qloss = x.new_zeros(())
        if args.round and cfg.get('q_weight', 1.) > 0:
            indices = torch.randint(len(x), (max(batch//2, 1),), device='cuda', generator=generators['q'])
            online_indices = torch.randint(len(replay['x']), (max(batch//2, 1),), device='cuda', generator=generators['q'])
            bx = torch.cat((x[indices], replay['x'][online_indices]))
            by = torch.cat((costs[indices], replay['costs'][online_indices]))
            bm = torch.cat((mask[indices], replay['mask'][online_indices]))
            valid = bm[:, 0]
            if cfg['critic_variant'] == 'legacy':
                target = -by[:, 0]/costs[:, 0].mean()
                target = torch.where(valid, target, target.new_full(target.shape, cfg['physical_failure_reward']))
                qloss = (critic(bx)-target[:, None]).square().mean()
            else:
                vcfg = VARIANTS[cfg['critic_variant']]
                pairs = None
                if vcfg.get('difference_weight'):
                    pair_family = torch.cat((families[indices], -1-online_indices))
                    ii, jj = torch.triu_indices(len(bx), len(bx), offset=1, device='cuda')
                    keep = (pair_family[ii] == pair_family[jj]) & ((bx[ii]-bx[jj]).abs().flatten(1).sum(1) > 0)
                    # Convert original batch indices to the success-only regression batch.
                    lookup = torch.full((len(bx),), -1, dtype=torch.long, device='cuda')
                    lookup[valid] = torch.arange(int(valid.sum()), device='cuda')
                    keep &= valid[ii] & valid[jj]
                    pairs = torch.stack((lookup[ii[keep]], lookup[jj[keep]]), 1)
                qloss, _ = critic.loss(bx[valid], by[valid], bm[valid], regression=vcfg.get('regression', 'mse'),
                    ranking_weight=vcfg.get('ranking_weight', 0.),
                    difference_weight=vcfg.get('difference_weight', 0.), pairs=pairs)
                if (~valid).any():
                    # Utility regression for confirmed physical failures; no fake PPA vector.
                    qloss = qloss+(critic.reward(bx[~valid], cfg['weights'])-cfg['physical_failure_reward']).square().mean()
            if not torch.isfinite(qloss):
                raise ValueError('Nonfinite online critic loss')
            opts['critic'].zero_grad(set_to_none=True)
            qloss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            opts['critic'].step()
        noise = 2*torch.rand(batch, len(problem.x), 2, device='cuda', generator=generators['actor'])-1
        with torch.no_grad():
            teacher = flow.sample_flow(noise, cfg.get('flow_steps', 10))
        prediction = actor(noise, clip=False)
        distill = (prediction[:, actor.movable]-teacher[:, actor.movable]).square().mean()
        with frozen_critic(critic):
            q = reward(critic, actor.restore(prediction, clip=True), costs, cfg).mean()
        distill_term = cfg.get('alpha', 10.)*distill
        qterm = cfg.get('q_weight', 1.)*q
        loss = distill_term-qterm
        diagnostics = {}
        if step % 100 == 0:
            diagnostics['action_gradients'] = gradient_norms(distill_term, qterm, (prediction,))
            diagnostics['parameter_gradients'] = gradient_norms(distill_term, qterm, tuple(actor.parameters()))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite actor objective')
        opts['actor'].zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        opts['actor'].step()
        flowloss = x.new_zeros(())
        if args.round and cfg.get('online_flow_update', False):
            successful = replay['x'][replay['success']]
            own = successful[torch.randint(len(successful), (batch//2,), device='cuda', generator=generators['flow'])] if len(successful) else behavior[:0]
            targets = torch.cat((behavior[torch.randint(len(behavior), (batch-len(own),), device='cuda', generator=generators['flow'])], own))
            t = torch.rand(batch, device='cuda', generator=generators['flow'])
            flowloss = flow.flow_loss(targets, noise, t)
            opts['flow'].zero_grad(set_to_none=True)
            flowloss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1., error_if_nonfinite=True)
            opts['flow'].step()
        if step % 50 == 0 or step+1 == total or STOP:
            row = dict(step=step+1, loss=float(loss), distill=float(distill), q=float(q), qloss=float(qloss),
                flowloss=float(flowloss), actor_grad=float(grad), slurm_job_id=os.environ['SLURM_JOB_ID'], **diagnostics)
            with (folder/'learning_curve.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            value = dict(flow=flow.state_dict(), actor=actor.state_dict(), critic=critic.state_dict(),
                optimizers={k:v.state_dict() for k,v in opts.items()}, rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), generators={k:g.get_state() for k,g in generators.items()},
                binding=binding, step=step+1)
            atomic_torch(resume, value)
        if STOP:
            raise SystemExit(85)
    if resume.exists():
        resume.replace(folder/'checkpoint.pt')
    atomic_json(folder/'completed.json', dict(binding=binding, status='completed', steps=total,
        offline_critic_frozen=args.round == 0, online_flow_update=cfg.get('online_flow_update', False),
        elapsed_this_attempt_s=time.monotonic()-started, checkpoint_sha256=sha(folder/'checkpoint.pt')))


def export(args):
    exp = args.experiment.resolve()
    cfg, problem, x, costs, mask, behavior, critic, families = setup(exp, args.seed)
    folder = exp/f'seed_{args.seed}'/f'round_{args.round}'
    dest = folder/'requests'
    if (dest/'completed.json').exists():
        from .graph_o2o_backend import load_requests
        from .qstudy_backend import context, checked_request
        backend_cfg, _, backend_binding = context(exp)
        records, _ = load_requests(exp, args.seed, args.round)
        for record in records:
            checked_request(record, args, backend_cfg, backend_binding)
        return
    checkpoint = folder/'checkpoint.pt'
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    completion = json.loads((folder/'completed.json').read_text())
    if completion['checkpoint_sha256'] != sha(checkpoint) or saved['binding'] != completion['binding']:
        raise ValueError('Export checkpoint is not the completed policy')
    actor = GraphPolicy(problem)
    actor.load_state_dict(saved['actor'])
    critic.load_state_dict(saved['critic'])
    actor.eval(); critic.eval()
    metric = MacroHPWL(problem.benchmark_info)
    records = []
    dest.mkdir(exist_ok=True)
    roles = ['collect', 'evaluation'] if args.round < cfg.get('online_rounds', 2) else ['evaluation']
    for role in roles:
        count = cfg['online_samples'] if role == 'collect' else cfg['evaluation_samples']
        noise_seed = (cfg['collection_noise_seed']+args.seed*100+args.round*1000 if role == 'collect' else cfg['evaluation_noise_seed'])
        generator = torch.Generator(device='cuda').manual_seed(noise_seed)
        noises = 2*torch.rand(count, len(problem.x), 2, device='cuda', generator=generator)-1
        for i, noise in enumerate(noises):
            sample = dest/cfg['arm']/role/str(i)
            sample.mkdir(parents=True, exist_ok=True)
            path = sample/'request.json'
            if path.exists():
                from .qstudy_backend import context, checked_request
                backend_cfg, _, backend_binding = context(exp)
                old = json.loads(path.read_text())
                checked_request(old, args, backend_cfg, backend_binding)
                if old['checkpoint_sha256'] != sha(checkpoint) or old['noise_sha256'] != action_hash(noise.cpu().numpy()):
                    raise ValueError('Interrupted export changed its checkpoint/noise')
                records.append(old);continue
            with torch.no_grad():
                raw = actor(noise[None])[0]
                predicted = float(reward(critic, raw[None], costs, cfg)[0])
                action = legalize(raw, problem)
            a = action.detach().cpu().numpy()
            n = noise.cpu().numpy()
            r = raw.cpu().numpy()
            geometry = legality(a, metric.sizes)
            record = dict(arm=cfg['arm'], seed=args.seed, round=args.round, role=role, index=i,
                graph=cfg['design'], terminal_mask=0, raw_action=r.tolist(), executed_action=a.tolist(),
                predicted_q=[predicted], source_checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                geometry=geometry, sample_dir=str(sample), config_sha256=sha(exp/'config.json'),
                run_id=f"go2o_qs1_{cfg['run_prefix']}_s{args.seed}_r{args.round}_{role}_{i}",
                action_sha256=action_hash(a), raw_action_sha256=action_hash(r), noise_sha256=action_hash(n),
                noise_seed=noise_seed, stage='global_route_only', training_feedback_eligible=role == 'collect',
                macro_hpwl=float(metric(a)), objective_weights=cfg['weights'],
                evaluation_purpose=cfg.get('evaluation_purpose', 'development_validation'))
            np.savez_compressed(sample/'actions.npz', noise=n, raw_action=r, action=a)
            if geometry['legal']:
                save_sample(sample, action, noise, problem, record)
                def_path = sample/(cfg['design']+'_job'+os.environ['SLURM_JOB_ID']+'.def')
                audit = convert(cfg['design'], sample/'macro_physical.json', def_path)
                record.update(def_path=str(def_path), def_sha256=sha(def_path), rounded_geometry=audit['legal_no_halo'])
                if not audit['legal_no_halo']:
                    record['pre_backend_failure'] = 'rounded_DEF_geometry'
            else:
                record['pre_backend_failure'] = 'normalized_geometry'
            atomic_json(path, record)
            records.append(record)
    path = dest/'tasks.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    atomic_json(dest/'completed.json', dict(status='completed', count=len(records), tasks_sha256=sha(path),
        config_sha256=sha(exp/'config.json'), selection='all predetermined noises; no Q-selection',
        evaluation_never_replayed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--stage', choices=['train', 'export', 'all'], default='train')
    args = parser.parse_args()
    if args.stage in ['train', 'all']:
        train(args)
    if args.stage in ['export', 'all']:
        export(args)


if __name__ == '__main__':
    main()
