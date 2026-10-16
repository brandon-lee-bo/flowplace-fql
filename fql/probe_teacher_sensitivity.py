"""Paired local-noise sensitivity of hard-projected and unprojected FlowPlace.

This diagnostic uses TRAIN noises only, does not train any model, and does not
measure placement quality. Finite differences are not a proof of continuity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from baseline.api import CHECKPOINT, RUNTIME, generate, load_model, load_problem


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def differences(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    norms = np.linalg.norm(delta.reshape(-1, 2), axis=1)
    return {"mse": float(np.square(delta).mean()),
            "rms": float(np.sqrt(np.square(delta).mean())),
            "max_abs_coordinate": float(np.abs(delta).max()),
            "max_macro_displacement": float(norms.max()),
            "mean_macro_displacement": float(norms.mean()),
            "macros_displaced_over_1e_3": int((norms > 1e-3).sum()),
            "exact_equal": bool(np.array_equal(a, b))}


def per_step_differences(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"mse": np.square(delta).mean(axis=(1, 2)).tolist(),
            "max_abs_coordinate": np.abs(delta).max(axis=(1, 2)).tolist()}


def sample(model, problem, z):
    torch.cuda.synchronize()
    start = time.perf_counter()
    x, used_z, trajectory = generate(model, problem, noise=torch.as_tensor(z, device="cuda"), intermediate_every=1)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    output = x.cpu().numpy()
    used = used_z.cpu().numpy()
    path = torch.stack([part.squeeze(0).cpu() for part in trajectory]).numpy()
    if not np.isfinite(output).all() or not np.isfinite(path).all():
        raise FloatingPointError("Nonfinite teacher output/trajectory")
    np.testing.assert_array_equal(used, z)
    return output, path, seconds


def summarize(records):
    summaries = []
    for mode in ("hard_projection", "no_hard_projection"):
        for eps in sorted(set(row["epsilon"] for row in records)):
            rows = [row for row in records if row["mode"] == mode and row["epsilon"] == eps]
            output = {"mode": mode, "epsilon": eps, "count": len(rows),
                      "exactly_unchanged_count": sum(row["output"]["exact_equal"] for row in rows),
                      "max_coordinate_change_over_1e_2_count": sum(row["output"]["max_abs_coordinate"] > 1e-2 for row in rows)}
            for metric in ["mse", "max_abs_coordinate", "max_macro_displacement"]:
                values = np.asarray([row["output"][metric] for row in rows])
                output["output_" + metric] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                             "min": float(values.min()), "max": float(values.max())}
            gains = np.asarray([row["rms_output_over_actual_input_rms"] for row in rows])
            output["rms_gain"] = {"median": float(np.median(gains)), "max": float(gains.max())}
            summaries.append(output)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "datasets/bp_fe_macro_deterministic_v1/dataset.npz"))
    parser.add_argument("--output", default=str(ROOT / "runs/teacher_sensitivity_v1"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--epsilons", default="1e-5,1e-3,1e-2")
    parser.add_argument("--direction-seed", type=int, default=271828)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Submit the sensitivity probe through Slurm with one GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires the requested single-GPU allocation")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    # Match collector arithmetic; report rather than silently change its TF32 defaults.
    torch.backends.cudnn.benchmark = False
    epsilons = [float(value) for value in args.epsilons.split(",")]
    if len(set(epsilons)) != len(epsilons) or any(not np.isfinite(x) or x <= 0 for x in epsilons):
        raise ValueError("Epsilons must be finite, positive and unique")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = Path(args.dataset).resolve()
    source = json.loads(dataset.with_name("manifest.json").read_text())
    source_args = source["args"]
    design = source_args["design"]
    report = {"status": "running", "purpose": "Local sensitivity diagnostic, not placement-quality evidence",
              "args": vars(args), "dataset": str(dataset), "dataset_sha256": sha256_file(dataset),
              "source_manifest": source, "checkpoint_sha256": sha256_file(CHECKPOINT),
              "cache_sha256": sha256_file(RUNTIME / "benchmark_cache" / f"{design}.pt"),
              "script_sha256": sha256_file(__file__), "adapter_sha256": sha256_file(ROOT / "baseline/api.py"),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "python_version": platform.python_version(), "gpu": torch.cuda.get_device_name(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "repeats": [], "cached_teacher_comparisons": [], "records": []}
    write_json(output / "report.json", report)
    start = time.perf_counter()
    try:
        for key in ["checkpoint_sha256", "cache_sha256", "adapter_sha256", "torch_version"]:
            if report[key] != source["provenance"][key]:
                raise ValueError(f"Probe and cached dataset disagree on {key}")
        with np.load(dataset, allow_pickle=False) as archive:
            train_rows = np.flatnonzero(archive["split"] == 0)
            if args.count < 1 or args.count > len(train_rows):
                raise ValueError("Invalid number of training noise points")
            selected = train_rows[:args.count]
            z = archive["noise"][selected].reshape(args.count, -1, 2).copy()
            cached_teacher = archive["teacher_action"][selected].reshape(z.shape).copy()
        problem = load_problem(design, "cuda")
        torch.manual_seed(source_args["model_init_seed"])
        model, config = load_model(problem, "cuda", steps=20, hard_constraint=True)
        mask = problem.cond.is_ports.detach().cpu().numpy().reshape(-1)
        movable = ~mask
        if not movable.any():
            raise ValueError("No movable coordinates")
        rng = np.random.default_rng(args.direction_seed)
        directions = rng.normal(size=z.shape)
        directions[:, mask] = 0
        directions /= np.sqrt(np.square(directions[:, movable]).mean(axis=(1, 2), keepdims=True))
        directions = directions.astype(np.float32)
        candidates, candidate_meta = [], []
        for local, row in enumerate(selected):
            for epsilon in epsilons:
                for sign in (-1, 1):
                    candidate = np.clip(z[local] + sign * epsilon * directions[local], -1, 1).astype(np.float32)
                    candidate[mask] = z[local, mask]
                    difference = differences(candidate, z[local])
                    if difference["mse"] == 0:
                        raise ValueError("Perturbation vanished at selected floating-point precision")
                    candidates.append(candidate)
                    candidate_meta.append({"local_index": local, "source_train_row": int(row), "epsilon": epsilon,
                                           "sign": sign, "input": difference,
                                           "boundary_clipped_coordinates": int(np.sum(np.abs(z[local] + sign * epsilon * directions[local]) > 1))})
        arrays = {"source_train_rows": selected, "base_noise": z, "directions": directions,
                  "perturbed_noise": np.stack(candidates), "cached_teacher": cached_teacher}
        report.update(selected_train_rows=selected.tolist(), steps=20, compile_disabled=False,
                      perturbation="One Gaussian direction per train noise, unit RMS over movable coordinates; both signs; clamp to prior support [-1,1]",
                      comparison="Same model instance and checkpoint, only use_hard_constraint_in_sampling changes; same current-x interpolation and [-2,2] clamp",
                      final_external_legalization=False)
        for mode, hard in [("hard_projection", True), ("no_hard_projection", False)]:
            model.use_hard_constraint_in_sampling = hard
            base_outputs, base_paths = [], []
            for local, row in enumerate(selected):
                baseline, trajectory, seconds = sample(model, problem, z[local])
                repeated, repeated_path, repeat_seconds = sample(model, problem, z[local])
                repeat_diff = differences(baseline, repeated)
                report["repeats"].append({"mode": mode, "source_train_row": int(row), **repeat_diff,
                                          "trajectory_exact_equal": bool(np.array_equal(trajectory, repeated_path)),
                                          "first_seconds": seconds, "repeat_seconds": repeat_seconds})
                if not repeat_diff["exact_equal"] or not np.array_equal(trajectory, repeated_path):
                    raise RuntimeError("Repeated identical noise is not exactly deterministic; sensitivity is confounded")
                if hard:
                    cache_diff = differences(baseline, cached_teacher[local])
                    report["cached_teacher_comparisons"].append({"source_train_row": int(row), **cache_diff})
                    if not cache_diff["exact_equal"]:
                        raise RuntimeError("Recomputed hard teacher differs from cached source; investigate before using sensitivity")
                base_outputs.append(baseline)
                base_paths.append(trajectory)
            changed_outputs, changed_paths = [], []
            for candidate, meta in zip(candidates, candidate_meta):
                current, trajectory, seconds = sample(model, problem, candidate)
                difference = differences(current, base_outputs[meta["local_index"]])
                record = {**meta, "mode": mode, "output": difference,
                          "rms_output_over_actual_input_rms": difference["rms"] / meta["input"]["rms"],
                          "trajectory": per_step_differences(trajectory, base_paths[meta["local_index"]]),
                          "sample_seconds": seconds}
                report["records"].append(record)
                changed_outputs.append(current)
                changed_paths.append(trajectory)
                write_json(output / "report.json", report)
                print(json.dumps({key: record[key] for key in ["mode", "source_train_row", "epsilon", "sign", "output"]}), flush=True)
            arrays[mode + "_base_output"] = np.stack(base_outputs)
            arrays[mode + "_base_trajectory"] = np.stack(base_paths)
            arrays[mode + "_perturbed_output"] = np.stack(changed_outputs)
            arrays[mode + "_perturbed_trajectory"] = np.stack(changed_paths)
            np.savez_compressed(output / "arrays.npz", **arrays)
        report.update(status="completed", seconds=time.perf_counter() - start, summary=summarize(report["records"]),
                      limitations=["Only eight training points and one direction each; not a global or statistical continuity proof",
                                   "Hard/no-hard changes the whole sampling trajectory; cannot isolate which projection iteration caused a jump without trajectory analysis",
                                   "No-hard still contains neural computation, finite precision and continuous clipping; not an unconstrained exact ODE solver",
                                   "No-hard outputs are not asserted legal and no quality metric is compared",
                                   "Sensitivity may help explain poor finite-data distillation but does not establish its only cause; capacity and sample coverage remain confounders",
                                   "Rows and epsilons are paired; signs and sizes are not independent training seeds"])
        write_json(output / "report.json", report)
        print(json.dumps({"status": report["status"], "output": str(output), "summary": report["summary"]}, indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", seconds=time.perf_counter() - start,
                      failure={"type": type(exc).__name__, "message": str(exc)})
        write_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
