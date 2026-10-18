"""Bounded three-network FQL-style terminal-placement diagnostic.

Uses fresh learned behavior-flow targets, unlike the earlier cached hard
FlowPlace distillation. This is one circuit and macro HPWL, not routed PPA or
a reproduction of the original multi-step FQL benchmark.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from .core import OneStepActor, Critics, critic_diagnostics, write_json
from .fresh_flow import BehaviorFlow, student_loss
from .analytic import load_hpwl
from .provenance import capture_source_provenance
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import RUNTIME, load_problem, legalize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 10.)
    optimizer.step()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/bp_fe_macro_deterministic_v1/dataset.npz')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--flow-steps', type=int, default=10)
    p.add_argument('--eval-count', type=int, default=32)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Submit through Slurm with one GPU')
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = capture_source_provenance(out / 'source_snapshot', extra_paths={
        name:root/name for name in ['fql/fresh_flow.py','fql/run_fresh_flow.py',
            'fql/fresh_flow_pilot.slurm','baseline/api.py','analysis/placement_metrics.py',
            'upstream/fql/agents/fql.py']})
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    dataset = Path(args.dataset).resolve()
    metadata = json.loads(dataset.with_name('manifest.json').read_text())
    cache = RUNTIME / 'benchmark_cache' / (metadata['args']['design'] + '.pt')
    if sha(cache) != metadata['provenance']['cache_sha256']:
        raise ValueError('Cache differs from source dataset')
    with np.load(dataset, allow_pickle=False) as a:
        train_idx = np.flatnonzero(a['critic_split'] == 0)
        val_idx = np.flatnonzero(a['split'] == 1)[:args.eval_count]
        critic_val_idx = np.flatnonzero(a['critic_split'] == 1)
        if len(val_idx) != args.eval_count or not len(train_idx):
            raise ValueError('Invalid diagnostic split budget')
        assert not set(a['critic_group_id'][train_idx]) & set(a['critic_group_id'][critic_val_idx])
        train_raw = a['raw_action'][train_idx].copy()
        train_y = a['reward'][train_idx].copy()
        val_raw = a['raw_action'][critic_val_idx].copy()
        val_y = a['reward'][critic_val_idx].copy()
        uniform_z = a['noise'][val_idx].copy()
        official_actions = a['teacher_action'][val_idx].copy()
        groups = a['critic_group_id'][train_idx].copy()
        mask = a['movable_mask'].copy() if 'movable_mask' in a.files else np.ones(train_raw.shape[1], bool)
        fixed = a['fixed_action'].copy() if 'fixed_action' in a.files else np.zeros(train_raw.shape[1])
        best_idx = train_idx[np.argmax(train_y)]
        best_train_action = a['action'][best_idx].copy()
    x = torch.tensor(train_raw.reshape(len(train_raw), -1), device=device)
    y = torch.tensor(train_y, device=device)
    vx = torch.tensor(val_raw.reshape(len(val_raw), -1), device=device)
    # Same quantiles couple the original uniform prior and FQL Gaussian prior.
    uz = torch.tensor(uniform_z.reshape(len(uniform_z), -1), device=device)
    if torch.any(uz.abs() >= 1):
        raise ValueError('Gaussian quantile transform requires open uniform support')
    z_eval = torch.erfinv(uz) * np.sqrt(2.)
    problem = load_problem(metadata['args']['design'], 'cuda')
    metric = MacroHPWL(problem.benchmark_info)
    hpwl, hpwl_meta = load_hpwl(cache, device)
    scale = float(metadata['reward_scale_train_only'])
    variants = {'distill_only': (10., 0.), 'fql_alpha_1': (1., 1.),
                'fql_alpha_10': (10., 1.), 'fql_alpha_100': (100., 1.)}
    report = dict(status='running', args=vars(args), slurm_job_id=os.environ['SLURM_JOB_ID'],
                  dataset=str(dataset), dataset_sha256=sha(dataset), cache_sha256=sha(cache),
                  source_provenance=source, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  purpose='Exploratory macro-only fixed-circuit FQL integration diagnostic',
                  prior='standard Gaussian; coupled by inverse Gaussian CDF to existing validation uniform noise',
                  train_rows=len(x), train_families=len(np.unique(groups)), validation_noise_rows=val_idx.tolist(),
                  heldout_evaluated=False, reward_scale=scale, seeds=[],
                  variants=variants, hpwl_metadata=hpwl_meta,
                  differences_from_original=['constant circuit context, terminal reward has no bootstrap',
                      '128x128x128 GELU networks and norm-10 gradient clipping',
                      'physical legality is external and all results are macro-only',
                      'one shared behavior flow and terminal critic across paired actor ablations'],
                  final_selection='None; fixed final step, all alphas retained; validation is exploratory')
    write_json(out / 'report.json', report)

    def evaluate(name, proposals, critic=None, gaussian_noise=None):
        raw = np.asarray(proposals, dtype=np.float32).reshape(len(proposals), -1)
        actions, legal, displacements = [], [], []
        for row in raw:
            executed = legalize(torch.tensor(row.reshape(-1, 2), device=device), problem).cpu().numpy()
            legal.append(legality(executed, metric.sizes)['legal'])
            displacements.append(float(np.square(executed.reshape(-1) - row).mean()))
            actions.append(executed)
        actions = np.stack(actions)
        costs = metric(actions)
        rewards = -costs / scale - (~np.asarray(legal)) * 10.
        result = dict(method=name, num_candidates=len(raw), legal_count=sum(legal),
                      mean_macro_hpwl=float(costs.mean()) if all(legal) else None,
                      mean_reward=float(rewards.mean()), mean_legalizer_mse=float(np.mean(displacements)))
        predicted = None
        if critic is not None:
            with torch.no_grad(): predicted = critic(torch.tensor(raw, device=device)).cpu().numpy()
            result['actor_selected_critic'] = critic_diagnostics(predicted, rewards)
        arrays = dict(raw_action=raw, action=actions, macro_hpwl=costs, reward=rewards, legal=np.asarray(legal))
        if predicted is not None: arrays['predicted_q'] = predicted
        if gaussian_noise is not None: arrays['gaussian_noise'] = gaussian_noise
        np.savez_compressed(out / (name.replace('/', '_') + '.npz'), **arrays)
        return result

    report['official_teacher'] = evaluate('official_teacher_relegalized', official_actions)
    # Original outputs have no additional projection; keep distinct from relegalized teacher.
    original_legal = [legality(row.reshape(-1, 2), metric.sizes)['legal'] for row in official_actions]
    report['official_teacher_original'] = dict(legal_count=sum(original_legal), count=len(original_legal),
        mean_macro_hpwl=float(metric(official_actions.reshape(len(official_actions), -1, 2)).mean()) if all(original_legal) else None)
    report['best_observed_train'] = dict(source_row=int(best_idx), candidates=1,
        macro_hpwl=float(metric(best_train_action.reshape(-1, 2))),
        note='One measured TRAIN layout, not replicated as independent noises')
    widths = [args.width] * 3
    for seed in map(int, args.seeds.split(',')):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        flow = BehaviorFlow(x.shape[1], widths, mask, fixed).to(device)
        critic = Critics(x.shape[1], widths).to(device)
        initial = OneStepActor(x.shape[1], widths, mask, fixed).to(device)
        actors = {k: copy.deepcopy(initial) for k in variants}
        optim_f = torch.optim.Adam(flow.parameters(), lr=3e-4)
        optim_q = torch.optim.Adam(critic.parameters(), lr=3e-4)
        optim_a = {k: torch.optim.Adam(v.parameters(), lr=3e-4) for k, v in actors.items()}
        generator = torch.Generator(device=device).manual_seed(120000 + seed)
        trace = []; started = time.perf_counter()
        for step in range(args.steps):
            index = torch.randint(len(x), (128,), generator=generator, device=device)
            bc_z = torch.randn(128, x.shape[1], generator=generator, device=device)
            t = torch.rand(128, 1, generator=generator, device=device)
            # Targets are evaluated before updating networks, as in joint original FQL loss.
            z = torch.randn(128, x.shape[1], generator=generator, device=device)
            target = flow.sample(z, args.flow_steps)
            loss_f = flow.bc_loss(x[index], bc_z, t)
            loss_q = (critic(x[index]) - y[index, None]).square().mean()
            actor_log = {}
            optim_q.zero_grad(set_to_none=True)
            for name, (alpha, weight) in variants.items():
                loss_a, distill, q = student_loss(actors[name], z, target, critic, alpha, weight)
                update(loss_a, optim_a[name], actors[name].parameters())
                actor_log[name] = dict(distill=float(distill.detach()), predicted_q=float(q.detach()))
            update(loss_f, optim_f, flow.parameters())
            update(loss_q, optim_q, critic.parameters())
            if step == 0 or (step + 1) % 500 == 0:
                row = dict(step=step + 1, flow_loss=float(loss_f.detach()),
                           critic_loss=float(loss_q.detach()), actors=actor_log,
                           seconds=time.perf_counter() - started)
                trace.append(row); print(json.dumps(dict(seed=seed, **row)), flush=True)
        folder = out / f'seed_{seed}'; folder.mkdir()
        torch.save(dict(flow_state=flow.state_dict(), critic_state=critic.state_dict(),
                        actor_states={k:v.state_dict() for k,v in actors.items()}, args=vars(args),
                        mask=mask, fixed=fixed, dataset_sha256=sha(dataset)), folder / 'checkpoint.pt')
        with torch.no_grad():
            teacher = flow.sample(z_eval, args.flow_steps)
            pred_train = critic(x).cpu().numpy(); pred_val = critic(vx).cpu().numpy()
            predictions = {k:v(z_eval, clip=True).cpu().numpy() for k,v in actors.items()}
            mse = {k:float((v(z_eval, clip=False)[:, mask] - teacher[:, mask]).square().mean()) for k,v in actors.items()}
        record = dict(seed=seed, seconds=time.perf_counter() - started, trace=trace,
                      critic_train=critic_diagnostics(pred_train, train_y),
                      critic_validation=critic_diagnostics(pred_val, val_y), results=[])
        teacher_np = teacher.cpu().numpy()
        record['results'].append(evaluate(f'seed_{seed}/learned_behavior_flow', teacher_np, critic, z_eval.cpu().numpy()))
        for name, prediction in predictions.items():
            r = evaluate(f'seed_{seed}/{name}', prediction, critic, z_eval.cpu().numpy())
            r['validation_fresh_teacher_distillation_mse'] = mse[name]
            record['results'].append(r)
        report['seeds'].append(record); write_json(out / 'report.json', report)
    report.update(status='completed', limitations=[
        'Single design and previously inspected validation families; no unseen-circuit or final heldout claim',
        'Reward is exact macro-only HPWL after legalization, not timing/power/routed wirelength',
        'A learned Q is unnecessary to calculate HPWL; this tests FQL mechanics, not its practical advantage',
        'Gaussian/behavior-flow/network changes form a combined adaptation; not an isolated single-factor ablation',
        'Reliable cached validation Q does not establish reliability on actor-selected layouts'])
    write_json(out / 'report.json', report)
    print(json.dumps(dict(status='completed', output=str(out))), flush=True)


if __name__ == '__main__':
    main()
