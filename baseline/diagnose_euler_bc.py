"""Exploratory matched BC learnability diagnostic; no Q, backend or PPA claim.

Experimental Euler RAW clipped targets differ from the official hard teacher's
already-projected targets. Shared noise/splits and BC optimization isolate an
empirical learnability comparison, not a claim of equivalent task semantics.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from analysis.placement_metrics import MacroHPWL, summarize
from baseline.api import load_model, load_problem, legalize, CHECKPOINT, RUNTIME, UPSTREAM
from baseline.diagnose_euler_sampler import manual_sampler
from baseline.diagnose_final_projection import sha, write, timed, stage_report
from fql.core import OneStepActor, distillation_loss


CONFIG = dict(seeds=[0, 1, 2], widths=[256, 256, 256], warmup_steps=1000,
              actor_steps=2000, batch_size=128, lr=3e-4, gradient_norm_cap=10.,
              warmup_batch_seed_offset=20000, actor_batch_seed_offset=30000,
              reset_adam_between_phases=True, teacher_steps=20)
SPLITS = [(0, "train"), (1, "validation"), (2, "heldout_exploratory")]


def snapshot(output):
    paths = [Path(__file__), ROOT/"baseline/api.py", ROOT/"baseline/diagnose_euler_sampler.py",
             ROOT/"baseline/diagnose_final_projection.py", ROOT/"analysis/placement_metrics.py",
             ROOT/"fql/core.py", ROOT/"fql/train.py"]
    paths += list((UPSTREAM/"flow_matching").rglob("*.py"))
    paths += list((UPSTREAM/"flow_matching/configs").rglob("*.yaml"))
    paths += list((UPSTREAM/"benchmark_utils").rglob("*.py"))
    records = []
    for source in sorted(set(paths)):
        dest = output/"source_snapshot"/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        records.append({"original_path": str(source), "snapshot_path": str(dest), "sha256": sha(dest)})
    write(output/"source_snapshot.json", {"event": "startup_before_generation_or_training", "files": records})
    return records


def train_bc(noise, target, mask, fixed, split, seed, output, provenance):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    actor = OneStepActor(noise.shape[1], CONFIG["widths"], mask, fixed).cuda()
    x = torch.as_tensor(noise, device="cuda")
    y = torch.as_tensor(target, device="cuda")
    train_ids = np.flatnonzero(split == 0)
    logs = []
    started = time.perf_counter()
    # Same order as original train.py: actor initialization precedes critic.
    # No dropout/stochastic actor layer; separate NumPy batch RNG gives same batches.
    initial_state = copy.deepcopy(actor.state_dict())
    for phase, steps, offset in [("warmup", 1000, 20000), ("actor_bc", 2000, 30000)]:
        optimizer = torch.optim.Adam(actor.parameters(), lr=CONFIG["lr"])
        rng = np.random.default_rng(seed+offset)
        for step in range(steps):
            index = torch.as_tensor(rng.choice(train_ids, size=min(CONFIG["batch_size"], len(train_ids)), replace=True), device="cuda")
            loss = distillation_loss(actor(x[index]), y[index], actor.movable_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("BC loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), CONFIG["gradient_norm_cap"])
            optimizer.step()
            if (step+1) % 250 == 0:
                logs.append({"phase": phase, "step": step+1, "training_minibatch_mse": float(loss.detach()),
                             "gradient_norm_before_clip": float(grad_norm)})
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-started
    actor.eval()
    with torch.no_grad():
        raw = torch.cat([actor(part) for part in x.split(256)]).cpu().numpy()
    clipped = np.where(mask, np.clip(raw, -1, 1), fixed).astype(np.float32)
    folder = output/f"seed_{seed}"
    folder.mkdir()
    np.savez_compressed(folder/"proposals.npz", noise=noise, split=split, teacher_action=target,
                        network_action=raw, raw_action=clipped, movable_mask=mask, fixed_action=fixed)
    torch.save({"actor_state": actor.state_dict(), "initial_actor_state": initial_state,
                "config": CONFIG, "seed": seed, "source_provenance": provenance,
                "target_semantics": "experimental_standard_euler_raw_clipped_minus1_plus1",
                "training_seconds": training_seconds}, folder/"checkpoint.pt")
    write(folder/"training_log.json", logs)
    return raw, clipped, training_seconds


def error_report(prediction, target, train_mean, train_variance, mask, split):
    result = {}
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    for value, label in SPLITS:
        rows = split == value
        mse = float(np.square(prediction[rows][:, mask]-target[rows][:, mask]).mean())
        mean_mse = float(np.square(target[rows][:, mask]-train_mean[mask]).mean())
        result[label] = {"count": int(rows.sum()), "mse": mse,
                         "mse_over_train_target_variance": mse/train_variance,
                         "mse_over_same_split_train_mean_predictor": mse/mean_mse,
                         "same_split_train_mean_predictor_mse": mean_mse}
    return result


def final_projection(raw, problem):
    results = []
    for row in raw:
        results.append(legalize(torch.as_tensor(row.reshape(-1, 2), device="cuda"), problem).cpu().numpy())
    return np.stack(results)


def render(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for pos, target_name in enumerate(["hard", "euler_raw_clipped"]):
        data = report["targets"][target_name]
        values = [data["methods"][f"bc_seed_{seed}"]["raw_error"]["heldout_exploratory"]["mse_over_same_split_train_mean_predictor"] for seed in CONFIG["seeds"]]
        axes[0].scatter(np.full(3, pos), values, marker="o", label=target_name)
        residual = data["methods"]["identity_plus_train_mean_residual"]["raw_error"]["heldout_exploratory"]["mse_over_same_split_train_mean_predictor"]
        axes[0].scatter([pos], [residual], marker="x", color="black", s=70)
        axes[1].scatter(np.full(3, pos), [data["methods"][f"bc_seed_{seed}"]["final"]["heldout_exploratory"]["macro_hpwl_legal_only"]["mean"]/1000 for seed in CONFIG["seeds"]])
        axes[1].scatter([pos], [data["methods"]["teacher"]["final"]["heldout_exploratory"]["macro_hpwl_legal_only"]["mean"]/1000], marker="x", color="black", s=70)
    axes[0].axhline(1, color="gray", linestyle="--")
    axes[0].set(yscale="log", ylabel="Raw MSE / TRAIN-mean predictor MSE", title="Exploratory heldout; circles: 3 BC seeds\ncross: identity + TRAIN residual")
    axes[1].set(ylabel="Mean final macro HPWL (thousand cache units)", title="Circles: 3 BC seeds; cross: teacher + final L\nMacro proxy only, not PPA")
    for ax in axes:
        ax.set(xticks=[0, 1], xticklabels=["Official hard targets", "Experimental Euler raw targets"])
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output/f"learnability_comparison.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--hard-training-dir", type=Path, required=True)
    p.add_argument("--euler-diagnostic", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("One allocated Slurm GPU required")
    args.output.mkdir(parents=True, exist_ok=False)
    source_snapshot = snapshot(args.output)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # Original BC explicitly disables cuDNN TF32. Euler's AttGNN only uses
    # matmul/linear/scatter; no convolutions. Verify first16 prior outputs exactly.
    torch.backends.cudnn.allow_tf32 = False
    meta = json.loads((args.dataset_dir/"manifest.json").read_text())
    old_run = json.loads((args.hard_training_dir/"run.json").read_text())
    with np.load(args.dataset_dir/"dataset.npz", allow_pickle=False) as data:
        noise, hard, split = data["noise"].copy(), data["teacher_action"].copy(), data["split"].copy()
        # Match PlacementData.load: this bp_fe dataset omits global arrays
        # because every coordinate is movable. Validate against the graph below.
        mask = (data["movable_mask"].copy() if "movable_mask" in data else np.ones(noise.shape[1], dtype=bool)).reshape(-1)
        fixed = (data["fixed_action"].copy() if "fixed_action" in data else np.zeros(noise.shape[1], dtype=np.float32)).reshape(-1)
    if len(noise) != 288 or [int((split == i).sum()) for i in range(3)] != [192, 48, 48]:
        raise ValueError("Expected original 288 fixed noise families/splits")
    for key in ["warmup_steps", "actor_steps", "batch_size", "lr"]:
        if old_run["args"][key] != CONFIG[key]:
            raise ValueError("Unmatched original BC budget: "+key)
    if old_run["data"]["sha256"] != sha(args.dataset_dir/"dataset.npz"):
        raise ValueError("Original hard BC used another dataset")
    provenance = {"config": CONFIG, "dataset_sha256": sha(args.dataset_dir/"dataset.npz"),
                  "original_hard_run_sha256": sha(args.hard_training_dir/"run.json"),
                  "checkpoint_sha256": sha(CHECKPOINT), "checkpoint_path": str(CHECKPOINT),
                  "cache_sha256": sha(RUNTIME/"benchmark_cache"/f"{meta['args']['design']}.pt"),
                  "source_snapshot": source_snapshot, "torch_version": torch.__version__,
                  "numpy_version": np.__version__, "python_version": platform.python_version(),
                  "gpu": torch.cuda.get_device_name(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                  "matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    report = {"status": "running", "scope": "Exploratory matched BC learnability; same288 noises, no Q training, no backend labels",
              "warning": "Different target semantics: hard already-projected vs Euler raw clipped. Macro HPWL is not PPA; no hyperparameter selection.",
              "provenance": provenance, "targets": {}, "original_output_checks": []}
    write(args.output/"report.json", report)
    write(args.output/"experiment_config.json", {"args": {k: str(v) for k, v in vars(args).items()}, **CONFIG})
    started = time.perf_counter()
    try:
        problem = load_problem(meta["args"]["design"], "cuda")
        np.testing.assert_array_equal(mask, np.repeat(~problem.cond.is_ports.cpu().numpy(), 2))
        metric = MacroHPWL(problem.benchmark_info)
        torch.manual_seed(meta["args"]["model_init_seed"])
        model, cfg = load_model(problem, "cuda", steps=20, hard_constraint=False)
        OmegaConf.save(cfg, args.output/"teacher_config.yaml")
        all_raw, paths, generation_times = [], [], []
        for i, z in enumerate(noise):
            (x, trajectory, _, _), seconds = timed(lambda: manual_sampler(model, problem, torch.as_tensor(z.reshape(-1, 2), device="cuda"), "experimental_standard_euler"))
            all_raw.append(x.cpu().numpy().reshape(-1)); paths.append(trajectory.cpu().numpy()); generation_times.append(seconds)
            if i % 48 == 0:
                print(json.dumps({"phase": "teacher_generation", "completed": i+1}), flush=True)
        euler_unclipped = np.stack(all_raw)
        euler = np.where(mask, np.clip(euler_unclipped, -1, 1), fixed).astype(np.float32)
        with np.load(args.euler_diagnostic/"arrays.npz", allow_pickle=False) as previous:
            previous_ids = previous["source_train_indices"]
            np.testing.assert_array_equal(euler_unclipped[previous_ids].reshape(16, -1, 2), previous["experimental_standard_euler"])
        report["prior_first16_euler_exact_match"] = True
        arrays = {"noise": noise, "split": split, "hard_target": hard, "euler_raw_unclipped": euler_unclipped,
                  "euler_target_clipped": euler, "euler_trajectory": np.stack(paths), "movable_mask": mask,
                  "fixed_action": fixed, "generation_seconds": np.asarray(generation_times), "sizes": metric.sizes}
        np.savez_compressed(args.output/"teacher_arrays.npz", **arrays)
        report["teacher_arrays_sha256"] = sha(args.output/"teacher_arrays.npz")
        report["generation_seconds"] = summarize(generation_times)
        report["euler_clipping_fraction"] = float((np.abs(euler_unclipped[:, mask]) > 1).mean())
        report["metric"] = metric.metadata()
        del model
        torch.cuda.empty_cache()
        predictions = {}
        for seed in CONFIG["seeds"]:
            raw, clipped, seconds = train_bc(noise, euler, mask, fixed, split, seed, args.output, provenance)
            predictions[("euler_raw_clipped", seed)] = (raw, clipped)
            report.setdefault("bc_training_seconds", {})[str(seed)] = seconds
            path = args.hard_training_dir/f"seed_{seed}/distill_only/proposals.npz"
            with np.load(path, allow_pickle=False) as saved:
                for key, expected in [("noise", noise), ("split", split), ("teacher_action", hard)]:
                    np.testing.assert_array_equal(saved[key], expected)
                predictions[("hard", seed)] = (saved["network_action"].copy(), saved["raw_action"].copy())
            report["original_output_checks"].append({"seed": seed, "source": str(path), "sha256": sha(path), "noise_split_target_match": True})
            write(args.output/"report.json", report)
            print(json.dumps({"phase": "BC_training", "seed": seed, "seconds": seconds}), flush=True)
        evaluated = {}
        for name, target in [("hard", hard), ("euler_raw_clipped", euler)]:
            mean = target[split == 0].astype(np.float64).mean(axis=0)
            residual = (target[split == 0].astype(np.float64)-noise[split == 0]).mean(axis=0)
            train_variance = float(np.square(target[split == 0][:, mask]-mean[mask]).mean())
            methods = {"teacher": (target, target),
                       "train_mean": (np.broadcast_to(mean, target.shape).copy().astype(np.float32), None),
                       "identity_plus_train_mean_residual": ((noise+residual).astype(np.float32), None)}
            for seed in CONFIG["seeds"]:
                methods[f"bc_seed_{seed}"] = predictions[(name, seed)]
            target_report = {"train_target_variance": train_variance, "train_mean": mean.tolist(),
                             "train_mean_target_minus_noise": residual.tolist(), "methods": {}}
            for method, (raw, clipped) in methods.items():
                raw = np.where(mask, raw, fixed).astype(np.float32)
                if clipped is None:
                    clipped = np.where(mask, np.clip(raw, -1, 1), fixed).astype(np.float32)
                final, seconds = timed(lambda: final_projection(clipped, problem))
                method_report = {"raw_error": error_report(raw, target, mean, train_variance, mask, split),
                                 "clipped_error": error_report(clipped, target, mean, train_variance, mask, split),
                                 "legalization_seconds_total": seconds, "raw": {}, "final": {}}
                for value, label in SPLITS:
                    rows = split == value
                    method_report["raw"][label], _ = stage_report(clipped[rows].reshape(-1, len(metric.names), 2), metric)
                    method_report["final"][label], _ = stage_report(final[rows], metric)
                if name == "hard" and method.startswith("bc_seed_"):
                    seed = int(method.rsplit("_", 1)[1])
                    for phase, value in [("validation", 1), ("heldout", 2)]:
                        path = args.hard_training_dir/f"measured_{phase}/seed_{seed}/distill_only/evaluated.npz"
                        with np.load(path, allow_pickle=False) as saved:
                            np.testing.assert_array_equal(final[saved["indices"]], saved["action"])
                        report["original_output_checks"].append({"seed": seed, "phase": phase, "final_exact_match": True,
                                                                 "source": str(path), "sha256": sha(path)})
                target_report["methods"][method] = method_report
                evaluated[f"{name}__{method}__raw"] = raw
                evaluated[f"{name}__{method}__clipped"] = clipped
                evaluated[f"{name}__{method}__final"] = final
                print(json.dumps({"phase": "measured", "target": name, "method": method,
                                  "heldout_raw_error": method_report["raw_error"]["heldout_exploratory"]}), flush=True)
            report["targets"][name] = target_report
            write(args.output/"report.json", report)
        np.savez_compressed(args.output/"evaluated_arrays.npz", noise=noise, split=split, **evaluated)
        report["evaluated_arrays_sha256"] = sha(args.output/"evaluated_arrays.npz")
        report["seed_variation"] = {}
        for name, target_report in report["targets"].items():
            report["seed_variation"][name] = {}
            for _, label in SPLITS:
                report["seed_variation"][name][label] = {
                    "bc_seed_mean_raw_mse": summarize([target_report["methods"][f"bc_seed_{s}"]["raw_error"][label]["mse"] for s in CONFIG["seeds"]]),
                    "bc_seed_mean_final_macro_hpwl": summarize([target_report["methods"][f"bc_seed_{s}"]["final"][label]["macro_hpwl_legal_only"]["mean"] for s in CONFIG["seeds"]]),
                    "note": "Three training seeds; within-seed noise variation reported separately, never 144 independent training replicates"}
        # Check startup copies against source at end; later source edits cannot alter the preserved snapshot.
        report["source_changed_during_run"] = [r["original_path"] for r in source_snapshot if sha(r["original_path"]) != r["sha256"]]
        report.update(status="completed", elapsed_seconds=time.perf_counter()-started)
        write(args.output/"report.json", report)
        render(report, args.output)
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}, elapsed_seconds=time.perf_counter()-started)
        write(args.output/"report.json", report)
        raise


if __name__ == "__main__":
    main()
