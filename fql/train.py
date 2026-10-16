"""Train cached-teacher distillation and FQL-style terminal-action variants.

Run on an allocated Slurm node. Outputs are RAW proposals requiring the external
FlowPlace legalizer and evaluator. Predicted Q values are not experiment rewards.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

# Must be present before the first CUDA context or cuBLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

try:
    from .core import (Critics, OneStepActor, PlacementData, critic_diagnostics,
                       distillation_loss, policy_loss, write_json)
    from .analytic import load_hpwl
    from .latency import benchmark_actor
except ImportError:
    from core import (Critics, OneStepActor, PlacementData, critic_diagnostics,
                      distillation_loss, policy_loss, write_json)
    from analytic import load_hpwl
    from latency import benchmark_actor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--alphas", default="1,10,100")
    parser.add_argument("--widths", default="256,256,256")
    parser.add_argument("--critic-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--actor-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=20261017)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize-q", action="store_true")
    parser.add_argument("--analytic-hpwl-cache", help="Optional real-circuit benchmark cache for analytic actor control")
    parser.add_argument("--analytic-alphas", default="", help="Optional same-student analytic HPWL alpha sweep")
    parser.add_argument("--reward-scale", type=float, help="Training-only HPWL normalization; defaults to dataset manifest")
    parser.add_argument("--latency-batches", default="1,32,128")
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=50, help="0 disables actor-only timing")
    parser.add_argument("--allow-random-split", action="store_true")
    parser.add_argument("--allow-local-smoke", action="store_true",
                        help="Only for <=20-step, <=1024-record synthetic unit smoke checks")
    return parser.parse_args()


def batch_indices(indices, batch_size, rng, device):
    chosen = rng.choice(indices, size=min(batch_size, len(indices)), replace=True)
    return torch.as_tensor(chosen, device=device, dtype=torch.long)


def log_line(stream, record):
    stream.write(json.dumps(record, allow_nan=False) + "\n")
    stream.flush()


def evaluate_critic(critic, actions, rewards, split, chunk=256):
    output = {}
    for value, name in [(0, "train"), (1, "validation"), (2, "heldout")]:
        idx = np.flatnonzero(split == value)
        with torch.no_grad():
            predictions = torch.cat([critic(actions[batch]) for batch in np.array_split(idx, max(1, (len(idx)+chunk-1)//chunk))])
        output[name] = critic_diagnostics(predictions.cpu().numpy(), rewards[idx].cpu().numpy())
    return output


def train_one_seed(args, data, seed, widths, alphas, device, root, analytic_hpwl=None, analytic_alphas=()):
    start = time.monotonic()
    out = root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    noise = torch.as_tensor(data.noise, device=device)
    target = torch.as_tensor(data.teacher, device=device)
    actions = torch.as_tensor(data.raw_action, device=device)
    rewards = torch.as_tensor(data.reward, device=device)
    teacher_train = np.flatnonzero(data.split == 0)
    critic_train = np.flatnonzero(data.critic_split == 0)
    teacher_val = np.flatnonzero(data.split == 1)
    dim = data.noise.shape[1]
    actor = OneStepActor(dim, widths, data.movable_mask, data.fixed_action).to(device)
    critic = Critics(dim, widths).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed + 10000)
    with (out / "training.jsonl").open("w") as logs:
        critic.train()
        for step in range(args.critic_steps):
            idx = batch_indices(critic_train, args.batch_size, rng, device)
            pred = critic(actions[idx])
            loss = (pred - rewards[idx, None]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Critic loss became nonfinite")
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            critic_optimizer.step()
            if (step + 1) % args.log_every == 0 or step + 1 == args.critic_steps:
                record = {"phase": "critic", "step": step + 1, "loss": float(loss.detach()),
                          "gradient_norm_before_clip": float(grad_norm)}
                log_line(logs, record)
                print(json.dumps({"seed": seed, **record}), flush=True)
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        diagnostics = evaluate_critic(critic, actions, rewards, data.critic_split)
        write_json(out / "critic_diagnostics.json", diagnostics)

        # Shared warmup controls initialization and distillation history across variants.
        warmup_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
        rng = np.random.default_rng(seed + 20000)
        actor.train()
        for step in range(args.warmup_steps):
            idx = batch_indices(teacher_train, args.batch_size, rng, device)
            loss = distillation_loss(actor(noise[idx]), target[idx], actor.movable_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Distillation loss became nonfinite")
            warmup_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            warmup_optimizer.step()
            if (step + 1) % args.log_every == 0 or step + 1 == args.warmup_steps:
                log_line(logs, {"phase": "warmup", "step": step + 1,
                               "distill_loss": float(loss.detach()), "gradient_norm_before_clip": float(grad_norm)})
        shared_state = copy.deepcopy(actor.state_dict())
        methods = [("distill_only", 1.0, 0.0, "distillation")] + [
            (f"fql_alpha_{alpha:g}", alpha, 1.0, "learned_critic") for alpha in alphas] + [
            (f"analytic_hpwl_alpha_{alpha:g}", alpha, 0.0, "analytic_prelegal_hpwl") for alpha in analytic_alphas]
        results = {}
        for method, alpha, q_weight, objective in methods:
            method_start = time.monotonic()
            actor.load_state_dict(shared_state)
            actor.train()
            # Reset Adam for every variant, including BC, for matched optimization budget.
            optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
            rng = np.random.default_rng(seed + 30000)
            for step in range(args.actor_steps):
                idx = batch_indices(teacher_train, args.batch_size, rng, device)
                raw = actor(noise[idx])
                clipped = torch.where(actor.movable_mask, raw.clamp(-1, 1), actor.fixed_action)
                qs = critic(clipped)
                loss, distill, q_loss = policy_loss(raw, target[idx], qs, actor.movable_mask,
                                                   alpha, q_weight, args.normalize_q)
                analytic_cost = None
                if objective == "analytic_prelegal_hpwl":
                    analytic_cost = (analytic_hpwl(clipped) / args.reward_scale).mean()
                    loss = alpha * distill + analytic_cost
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Actor loss became nonfinite for {method}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                optimizer.step()
                if (step + 1) % args.log_every == 0 or step + 1 == args.actor_steps:
                    with torch.no_grad():
                        val_raw = actor(noise[teacher_val])
                        val_mse = distillation_loss(val_raw, target[teacher_val], actor.movable_mask)
                    record = {"phase": method, "step": step + 1, "loss": float(loss.detach()),
                              "distill_loss": float(distill.detach()), "q_loss": float(q_loss.detach()),
                              "predicted_q": float(qs.detach().mean()), "validation_distill_mse": float(val_mse),
                              "gradient_norm_before_clip": float(grad_norm)}
                    if analytic_cost is not None:
                        record["analytic_prelegal_hpwl_normalized"] = float(analytic_cost.detach())
                    log_line(logs, record)
                    print(json.dumps({"seed": seed, **record}), flush=True)
            actor_training_seconds = time.monotonic() - method_start
            actor.eval()
            method_dir = out / method
            method_dir.mkdir()
            with torch.no_grad():
                raw = torch.cat([actor(part) for part in noise.split(256)])
                clipped = torch.where(actor.movable_mask, raw.clamp(-1, 1), actor.fixed_action)
                prediction = torch.cat([critic(part) for part in clipped.split(256)])
            raw_np, clipped_np, q_np = raw.cpu().numpy(), clipped.cpu().numpy(), prediction.cpu().numpy()
            np.savez_compressed(method_dir / "proposals.npz", noise=data.noise,
                                network_action=raw_np, raw_action=clipped_np,
                                teacher_action=data.teacher, predicted_q=q_np, split=data.split,
                                movable_mask=data.movable_mask, fixed_action=data.fixed_action)
            scores = {}
            for value, label in [(0, "train"), (1, "validation"), (2, "heldout")]:
                mask = data.split == value
                scores[label] = {
                    "count": int(mask.sum()),
                    "distill_mse": float(np.square(raw_np[mask][:, data.movable_mask] - data.teacher[mask][:, data.movable_mask]).mean()),
                    "clip_fraction": float((np.abs(raw_np[mask][:, data.movable_mask]) > 1).mean()),
                    "predicted_q": float(q_np[mask].mean()),
                    "ensemble_disagreement": float(q_np[mask].std(axis=-1).mean()),
                    "student_coordinate_std": float(clipped_np[mask][:, data.movable_mask].std(axis=0).mean()),
                    "teacher_coordinate_std": float(data.teacher[mask][:, data.movable_mask].std(axis=0).mean()),
                    "evaluated_reward": None,
                }
            latency = None
            if args.latency_repeats:
                latency = benchmark_actor(actor, noise[teacher_val],
                                          [int(x) for x in args.latency_batches.split(",")],
                                          args.latency_warmup, args.latency_repeats)
                write_json(method_dir / "actor_latency.json", latency)
            elapsed = time.monotonic() - method_start
            results[method] = {"alpha": alpha, "q_weight": q_weight, "seconds": elapsed, "splits": scores,
                               "objective": objective, "actor_training_seconds": actor_training_seconds,
                               "actor_latency": latency,
                               "selection": "fixed final iteration; no heldout metric used for selection"}
            write_json(method_dir / "diagnostics.json", results[method])
            torch.save({"actor_state": actor.state_dict(), "critic_state": critic.state_dict(),
                        "widths": widths, "dim": dim, "movable_mask": data.movable_mask.tolist(),
                        "fixed_action": data.fixed_action.tolist(), "coordinate_shape": data.coordinate_shape,
                        "seed": seed, "method": method, "objective": objective, "args": vars(args), "data": data.metadata},
                       method_dir / "checkpoint.pt")
        result = {"seed": seed, "seconds": time.monotonic() - start, "critic": diagnostics, "methods": results}
        write_json(out / "summary.json", result)
    return result


def main():
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_local_smoke:
        raise RuntimeError("Use a Slurm allocation/job for training; --allow-local-smoke is only a bounded synthetic check")
    widths = [int(x) for x in args.widths.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    alphas = [float(x) for x in args.alphas.split(",") if x]
    analytic_alphas = [float(x) for x in args.analytic_alphas.split(",") if x]
    if any(alpha <= 0 for alpha in alphas) or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be positive and unique")
    if any(alpha <= 0 for alpha in analytic_alphas) or len(set(analytic_alphas)) != len(analytic_alphas):
        raise ValueError("analytic alphas must be positive and unique")
    if analytic_alphas and not args.analytic_hpwl_cache:
        raise ValueError("Analytic control requires --analytic-hpwl-cache")
    if args.critic_steps < 1 or min(args.warmup_steps, args.actor_steps) < 0:
        raise ValueError("critic needs >=1 step; actor/warmup steps cannot be negative")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(args.threads)
    data = PlacementData.load(args.dataset, args.split_seed, args.allow_random_split)
    if args.allow_local_smoke and not os.environ.get("SLURM_JOB_ID"):
        if max(args.critic_steps, args.warmup_steps, args.actor_steps) > 20 or len(data.noise) > 1024 or len(data.reward) > 1024 or max(widths) > 64 or data.noise.shape[1] > 64 or len(seeds) > 1:
            raise RuntimeError("Local smoke budget exceeded; use Slurm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    analytic_hpwl, analytic_metadata = None, None
    if analytic_alphas:
        analytic_hpwl, analytic_metadata = load_hpwl(args.analytic_hpwl_cache, device)
        if len(analytic_hpwl.names) * 2 != data.noise.shape[1]:
            raise ValueError("Analytic cache macro count does not match dataset coordinates")
        if args.reward_scale is None:
            manifest = json.loads(Path(args.dataset).with_name("manifest.json").read_text())
            args.reward_scale = float(manifest["reward_scale_train_only"])
        if not np.isfinite(args.reward_scale) or args.reward_scale <= 0:
            raise ValueError("Analytic reward scale must be finite and positive")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "run.json").exists():
        raise FileExistsError("Output contains a run.json; choose a new directory rather than overwrite evidence")
    run = {"kind": "FQL-style fixed-circuit contextual-bandit adaptation", "args": vars(args),
           "data": data.metadata, "torch_version": torch.__version__, "numpy_version": np.__version__,
           "python": platform.python_version(), "device": str(device), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "determinism": {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                           "scope": "Repeated run with same seed, software and hardware; not cross-version identity"},
           "source_objective": "https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py",
           "changes_from_original": ["Frozen cached FlowPlace teacher", "Fixed circuit", "Terminal reward regression, no bootstrap",
                                     "Critic pretraining then frozen actor optimization", "Uniform source supplied by dataset",
                                     "Gradient-norm cap 10", "External non-differentiable legalizer"],
           "analytic_control": analytic_metadata,
           "evaluation_warning": "proposals.npz requires external legalization and measured reward; predicted Q is not quality evidence"}
    write_json(root / "run.json", run)
    try:
        summaries = [train_one_seed(args, data, seed, widths, alphas, device, root, analytic_hpwl, analytic_alphas) for seed in seeds]
        write_json(root / "summary.json", {"status": "training_completed_external_evaluation_required", "seeds": summaries})
    except Exception as exc:
        write_json(root / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
