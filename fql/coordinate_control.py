"""Optimize paired teacher layouts with exact pre-legalization HPWL gradients.

The saved proposals require the same external final legalizer as every student.
This method performs test-time optimization and pays for teacher generation plus
all gradient evaluations. It is not a one-step learned placement policy.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .analytic import load_hpwl, optimize_coordinates
    from .core import Critics, PlacementData, write_json
    from .latency import synchronize
except ImportError:
    from analytic import load_hpwl, optimize_coordinates
    from core import Critics, PlacementData, write_json
    from latency import synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reference-checkpoint", required=True,
                        help="Critic used only for diagnostic predicted_q; never for optimization")
    parser.add_argument("--training-dir", required=True, help="Existing root training output, for shared evaluator discovery")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--anchor-weights", default="0,1,10")
    parser.add_argument("--reward-scale", type=float, help="Defaults to dataset manifest's training-only scale")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run coordinate optimization in Slurm")
    torch.set_num_threads(args.threads)
    data = PlacementData.load(args.dataset)
    hpwl, hpwl_metadata = load_hpwl(args.cache, args.device)
    if len(hpwl.names) * 2 != data.noise.shape[1]:
        raise ValueError("Cache and dataset macro dimensions differ")
    if args.reward_scale is None:
        manifest = json.loads(Path(args.dataset).with_name("manifest.json").read_text())
        args.reward_scale = float(manifest["reward_scale_train_only"])
    if not np.isfinite(args.reward_scale) or args.reward_scale <= 0:
        raise ValueError("Reward scale must be positive and finite")
    weights = [float(x) for x in args.anchor_weights.split(",")]
    if any(x < 0 or not np.isfinite(x) for x in weights) or len(set(weights)) != len(weights):
        raise ValueError("Anchor weights must be finite, nonnegative and unique")
    checkpoint = torch.load(args.reference_checkpoint, map_location=args.device, weights_only=False)
    if checkpoint["data"]["sha256"] != data.metadata["sha256"]:
        raise ValueError("Diagnostic critic checkpoint belongs to a different dataset")
    critic = Critics(checkpoint["dim"], checkpoint.get("critic_widths", checkpoint["widths"])).to(args.device).eval()
    critic.load_state_dict(checkpoint["critic_state"])
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    target = torch.as_tensor(data.teacher, device=args.device)
    parent = Path(args.training_dir) / "seed_control"
    parent.mkdir(parents=True, exist_ok=True)
    for weight in weights:
        method = f"coordinate_hpwl_steps_{args.steps}_anchor_{weight:g}_lr_{args.lr:g}"
        output = parent / method
        output.mkdir(exist_ok=False)
        proposals, timing = [], []
        log = output / "optimization.jsonl"
        with log.open("w") as stream:
            for begin in range(0, len(target), args.batch_size):
                batch = target[begin:begin + args.batch_size]
                synchronize(batch.device)
                start = time.perf_counter()
                proposal, trace = optimize_coordinates(batch, hpwl, args.reward_scale,
                                                      data.movable_mask, data.fixed_action,
                                                      args.steps, args.lr, weight)
                synchronize(batch.device)
                elapsed = time.perf_counter() - start
                proposals.append(proposal)
                timing.append({"first_index": begin, "count": len(batch), "seconds": elapsed})
                stream.write(json.dumps({"first_index": begin, "count": len(batch), "seconds": elapsed,
                                         "trace": trace}, allow_nan=False) + "\n")
                stream.flush()
        raw = torch.cat(proposals)
        with torch.no_grad():
            prediction = torch.cat([critic(part) for part in raw.split(256)]).cpu().numpy()
            before = hpwl(target).cpu().numpy()
            after = hpwl(raw).cpu().numpy()
        np.savez_compressed(output / "proposals.npz", noise=data.noise, teacher_action=data.teacher,
                            raw_action=raw.cpu().numpy(), network_action=raw.cpu().numpy(),
                            predicted_q=prediction, split=data.split, movable_mask=data.movable_mask,
                            fixed_action=data.fixed_action,
                            prelegal_hpwl_initial=before, prelegal_hpwl_final=after)
        summary = {"method": method, "kind": "per-layout test-time analytic optimization control",
                   "dataset": data.metadata, "hpwl": hpwl_metadata, "args": vars(args),
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "torch_version": str(torch.__version__),
                   "initialization": "Exact teacher_action paired to the same recorded noise; no regenerated teacher",
                   "objective": "prelegal_HPWL / train_scale + anchor_weight * mean_movable_coordinate_squared_displacement",
                   "legalization": "Not performed here; shared evaluate_candidates.py must apply final official legalizer",
                   "diagnostic_critic": str(Path(args.reference_checkpoint).resolve()),
                   "critic_role": "predicted_q diagnostic only; contributes zero gradients and makes no choices",
                   "test_time_gradient_evaluations_per_layout": args.steps,
                   "test_time_teacher_generations_per_layout": 1,
                   "timing": timing, "optimization_total_seconds": sum(x["seconds"] for x in timing),
                   "timing_warning": "Batched optimization throughput, not isolated per-layout latency; add teacher and final legalization costs",
                   "validation": {}, "heldout": {}}
        for value, label in [(1, "validation"), (2, "heldout")]:
            select = data.split == value
            summary[label] = {"count": int(select.sum()), "initial_prelegal_hpwl_mean": float(before[select].mean()),
                              "final_prelegal_hpwl_mean": float(after[select].mean()),
                              "evaluated_legalized_reward": None}
        write_json(output / "diagnostics.json", summary)
        print(json.dumps({"output": str(output), "optimization_seconds": summary["optimization_total_seconds"],
                          "warning": "Prelegal HPWL only; external legalization and actual reward evaluation required"}), flush=True)


if __name__ == "__main__":
    main()
