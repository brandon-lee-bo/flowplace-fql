"""Measure actual actor forward latency; do not include legalizer or I/O."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .core import OneStepActor, write_json
except ImportError:
    from core import OneStepActor, write_json


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_actor(actor, noise, batch_sizes=(1, 32), warmup=10, repeats=50):
    """Host wall latency with completed GPU work, not asynchronous launch time.

    Inputs are already resident on device. Clipping and fixed-position restoration
    are included, matching deployed policy. Batch throughput is amortized and must
    not be described as per-request latency. No model compilation is performed.
    """
    if warmup < 0 or repeats < 1 or len(noise) < 1:
        raise ValueError("Need nonempty noise, nonnegative warmup, and >=1 repeat")
    if any(size < 1 for size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    device = noise.device
    was_training = actor.training
    actor.eval()
    records = {}
    try:
        with torch.inference_mode():
            for size in sorted(set(batch_sizes)):
                # Cycle only to populate a batch if requested batch > available N.
                # All slicing and allocations happen before timing.
                batches = [noise[(torch.arange(size, device=device) + i * size) % len(noise)].contiguous()
                           for i in range(max(repeats, warmup))]
                for batch in batches[:warmup]:
                    actor(batch, clip=True)
                synchronize(device)
                durations = []
                for batch in batches[:repeats]:
                    synchronize(device)
                    start = time.perf_counter()
                    result = actor(batch, clip=True)
                    synchronize(device)
                    durations.append(time.perf_counter() - start)
                values = np.asarray(durations)
                records[str(size)] = {
                    "batch_size": size, "repeats": repeats, "warmup_calls": warmup,
                    "mean_call_ms": float(values.mean() * 1000),
                    "median_call_ms": float(np.median(values) * 1000),
                    "p95_call_ms": float(np.quantile(values, 0.95) * 1000),
                    "min_call_ms": float(values.min() * 1000),
                    "std_call_ms": float(values.std(ddof=1) * 1000) if repeats > 1 else 0.0,
                    "amortized_ms_per_layout": float(values.mean() * 1000 / size),
                    "layouts_per_second": float(size / values.mean()),
                    "call_seconds": durations,
                    "output_shape": list(result.shape),
                }
    finally:
        actor.train(was_training)
    return {
        "scope": "actor_forward_including_clip_and_fixed_coordinate_restore",
        "excluded": ["noise sampling", "host/device transfer", "checkpoint loading", "legalization", "metric evaluation"],
        "clock": "perf_counter host wall with explicit CUDA synchronize before and after every timed call",
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
        "precision": str(noise.dtype), "warmup_excluded": True,
        "note": "Batch=1 is warm single-layout latency. Larger batches report throughput; not single-request latency.",
        "batches": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--noise", required=True, help="NPZ containing noise or NPY array")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", default="1,32,128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run latency measurements in a Slurm allocation")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    actor = OneStepActor(checkpoint["dim"], checkpoint["widths"], checkpoint["movable_mask"], checkpoint["fixed_action"])
    actor.load_state_dict(checkpoint["actor_state"])
    actor.to(args.device).eval()
    archive = np.load(args.noise, allow_pickle=False)
    raw = archive["noise"] if isinstance(archive, np.lib.npyio.NpzFile) else archive
    noise = torch.as_tensor(np.asarray(raw, dtype=np.float32).reshape(len(raw), -1), device=args.device)
    result = benchmark_actor(actor, noise, [int(x) for x in args.batch_sizes.split(",")], args.warmup, args.repeats)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), slurm_job_id=os.environ["SLURM_JOB_ID"])
    write_json(args.output, result)
    print(json.dumps({"output": args.output, "batches": {k: v["mean_call_ms"] for k, v in result["batches"].items()}}))


if __name__ == "__main__":
    main()
