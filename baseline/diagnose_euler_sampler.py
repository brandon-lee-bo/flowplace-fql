"""Experimental standard-Euler ablation; never patches the released baseline.

Uses the first 16 TRAIN noises, the same model instance/forward precision, time
grid, fixed-node mask and per-step [-2,2] clamp. No training or PPA evaluation.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL, summarize
from baseline.diagnose_final_projection import sha, write, timed, stage_report
from baseline.api import load_problem, load_model, generate, legalize, CHECKPOINT, RUNTIME, UPSTREAM
from omegaconf import OmegaConf


@torch.no_grad()
def manual_sampler(model, problem, noise, mode):
    """Calls the untouched model(x,cond,t), including its own backbone autocast."""
    if mode not in ("released_clone", "experimental_standard_euler"):
        raise ValueError(mode)
    x_in = problem.x.unsqueeze(0)
    x = noise.unsqueeze(0).clone()
    mask = model.get_mask(x_in, problem.cond)
    x = torch.where(mask, x_in, x) if mask is not None else x
    model._scheduler.set_timesteps(model.max_diffusion_steps)
    timesteps = model._scheduler.timesteps
    trajectory, velocities, precision = [x.clone()], [], None
    for i, t in enumerate(timesteps[:-1]):
        t_next = timesteps[i+1]
        # Match the released construction, dtype and device of time tensors.
        t_vec = torch.tensor(t, device=model.device).expand(1)
        t_next_vec = torch.tensor(t_next, device=model.device).expand(1)
        v_pred = model(x, problem.cond, t_vec)
        if precision is None:
            precision = {"input_dtype": str(x.dtype), "time_dtype": str(t_vec.dtype),
                         "velocity_dtype": str(v_pred.dtype),
                         "outer_autocast_enabled": torch.is_autocast_enabled(),
                         "forward": "Untouched FlowMatchingModel.__call__ -> AttGNN.forward with original internal torch.autocast"}
        velocities.append(v_pred.clone())
        if mode == "released_clone":
            # Preserve the exact operations; do not algebraically simplify.
            u1 = x+(1-t_vec.view(1, 1, 1))*v_pred
            x = (1-t_next_vec.view(1, 1, 1))*x+t_next_vec.view(1, 1, 1)*u1
        else:
            x = x+(t_next_vec-t_vec).view(1, 1, 1)*v_pred
        x = torch.clamp(x, -2, 2)
        x = torch.where(mask, x_in, x)
        trajectory.append(x.clone())
    return x.squeeze(0), torch.stack(trajectory).squeeze(1), torch.stack(velocities).squeeze(1), precision


def sample(mode, model, problem, noise):
    if mode == "released_nohard":
        x, _, trajectory = generate(model, problem, noise=noise, intermediate_every=1)
        return x, torch.stack(trajectory).squeeze(1)
    x, path, _, _ = manual_sampler(model, problem, noise, mode)
    return x, path


def difference(a, b):
    delta = np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
    return {"exact_equal": bool(np.array_equal(a, b)),
            "rms": float(np.sqrt(np.mean(delta**2))), "max_abs": float(np.abs(delta).max())}


def render(arrays, report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    modes = ["released_nohard", "experimental_standard_euler"]
    labels = ["Released no-hard", "Experimental standard Euler"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, label in zip(modes, labels):
        trajectory = arrays[mode+"_trajectory"]
        sd = np.sqrt(np.square(trajectory.std(axis=0, ddof=1)).mean(axis=(1, 2)))
        axes[0].plot(arrays["time_grid"], sd, marker=".", label=label)
    axes[0].set(xlabel="Nominal model time", ylabel="Across-noise coordinate std RMS",
                title="Same 16 TRAIN noises; same weights and forward precision")
    axes[0].legend(fontsize=8)
    values, labels2 = [], []
    for mode, label in zip(modes, labels):
        values.append([r["macro_hpwl"]/1000 for r in report["stages"][mode+"_final"]["per_noise"] if r["legal"]])
        labels2.append(label+"\n+ final projection")
    axes[1].boxplot(values, labels=labels2)
    axes[1].set(ylabel="Macro HPWL (thousand cache units)", title="Legal layouts only; macro proxy, not PPA")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out/f"sampler_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--previous-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Use one Slurm GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.backends.cudnn.benchmark = False
    source = json.loads((args.dataset_dir/"manifest.json").read_text())
    previous_report = json.loads((args.previous_diagnostic/"report.json").read_text())
    with np.load(args.dataset_dir/"dataset.npz", allow_pickle=False) as data:
        indices = np.flatnonzero(data["split"] == 0)[:16]
        noise = data["noise"][indices].reshape(16, -1, 2).copy()
    with np.load(args.previous_diagnostic/"arrays.npz", allow_pickle=False) as saved:
        previous = {k: saved[k] for k in saved.files}
    np.testing.assert_array_equal(indices, previous["source_train_indices"])
    np.testing.assert_array_equal(noise, previous["noise"])
    design = source["args"]["design"]
    provenance = {"dataset_sha256": sha(args.dataset_dir/"dataset.npz"),
        "previous_arrays_sha256": sha(args.previous_diagnostic/"arrays.npz"),
        "checkpoint_sha256": sha(CHECKPOINT), "cache_sha256": sha(RUNTIME/"benchmark_cache"/f"{design}.pt"),
        "adapter_sha256": sha(ROOT/"baseline/api.py"), "script_sha256": sha(__file__),
        "metric_helper_sha256": sha(ROOT/"baseline/diagnose_final_projection.py"),
        "upstream_models_sha256": sha(UPSTREAM/"flow_matching/models.py"),
        "upstream_forward_sha256": sha(UPSTREAM/"flow_matching/networks/gnn.py"),
        "torch_version": torch.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    for key in ("checkpoint_sha256", "cache_sha256", "adapter_sha256", "torch_version"):
        if provenance[key] != source["provenance"][key]:
            raise ValueError("Source provenance changed: "+key)
    if provenance["previous_arrays_sha256"] != previous_report["arrays_sha256"]:
        raise ValueError("Previous diagnostic arrays changed")
    arrays = {"source_train_indices": indices, "noise": noise,
              "hard_teacher_reference": previous["hard_original"]}
    report = {"status": "running", "provenance": provenance,
        "scope": "Same first16 TRAIN noises, 20 steps, no training or heldout selection",
        "experimental_status": "Standard Euler is a separately labeled sampler ablation, not a bugfix or claimed official FlowPlace result",
        "warning": "Macro-only HPWL excludes standard cells and fixed I/O; no PPA claim",
        "sampler_definitions": {
            "released_nohard": "u1=x+(1-t)*v; x_next=(1-t_next)*x+t_next*u1; preserves exact released arithmetic",
            "experimental_standard_euler": "x_next=x+(t_next-t)*v",
            "shared": "Same model object, forward call/internal autocast, time-grid, mask, [-2,2] per-step clamp; one identical external legalizer"},
        "stages": {}, "timings": {}, "repeatability": [], "previous_nohard_checks": [], "local_probes": []}
    write(args.output/"report.json", report)
    started = time.perf_counter()
    try:
        problem = load_problem(design, "cuda")
        metric = MacroHPWL(problem.benchmark_info)
        arrays["sizes"] = metric.sizes
        report["metric"] = metric.metadata()
        torch.manual_seed(source["args"]["model_init_seed"])
        model, cfg = load_model(problem, "cuda", steps=20, hard_constraint=False)
        OmegaConf.save(cfg, args.output/"config.yaml")
        first_z = torch.as_tensor(noise[0], device="cuda")
        official_x, official_path = sample("released_nohard", model, problem, first_z)
        clone_x, clone_path, clone_v, clone_precision = manual_sampler(model, problem, first_z, "released_clone")
        euler_x, euler_path, euler_v, euler_precision = manual_sampler(model, problem, first_z, "experimental_standard_euler")
        checks = {"released_clone_final_exact": torch.equal(official_x, clone_x),
                  "released_clone_all_steps_exact": torch.equal(official_path, clone_path),
                  "same_first_forward_velocity_exact": torch.equal(clone_v[0], euler_v[0]),
                  "same_forward_precision": clone_precision == euler_precision}
        report["forward_arithmetic_checks"] = checks
        report["precision"] = clone_precision
        if not all(checks.values()):
            raise AssertionError("Cannot attribute changes to sampler update: forward/control validation failed")
        arrays["released_clone_first_trajectory"] = clone_path.cpu().numpy()
        arrays["released_clone_first_velocities"] = clone_v.cpu().numpy()
        arrays["euler_first_velocities"] = euler_v.cpu().numpy()
        arrays["time_grid"] = model._scheduler.timesteps.numpy().copy()
        times = arrays["time_grid"].astype(np.float64)
        released_coeff = times[1:]*(1-times[:-1]); euler_coeff = np.diff(times)
        report["velocity_coefficients"] = {"released_algebraic_coefficients": released_coeff.tolist(),
            "standard_euler_dt": euler_coeff.tolist(), "released_sum": float(released_coeff.sum()),
            "euler_sum": float(euler_coeff.sum()), "maximum_released_over_dt": float(np.max(released_coeff/euler_coeff)),
            "interpretation": "Algebraic coefficients only; v is state/time dependent. Their sum is not a proof of equivalent integration time or the sole cause of contraction."}
        for mode in ("released_nohard", "experimental_standard_euler"):
            outputs, paths, costs, finals, legal_costs = [], [], [], [], []
            for local, z in enumerate(noise):
                z_gpu = torch.as_tensor(z, device="cuda")
                (x, path), seconds = timed(lambda: sample(mode, model, problem, z_gpu))
                outputs.append(x.cpu().numpy()); paths.append(path.cpu().numpy()); costs.append(seconds)
                if mode == "released_nohard":
                    same = np.array_equal(outputs[-1], previous["nohard_raw"][local])
                    report["previous_nohard_checks"].append({"source_train_index": int(indices[local]), "exact_equal": same})
                    if not same:
                        raise RuntimeError("Released no-hard differs from previous exact-noise diagnostic")
                if local < 4:
                    (repeat, repeat_path), _ = timed(lambda: sample(mode, model, problem, z_gpu))
                    check = {"mode": mode, "source_train_index": int(indices[local]),
                             "final_exact": torch.equal(repeat, x), "trajectory_exact": torch.equal(repeat_path, path)}
                    report["repeatability"].append(check)
                    if not check["final_exact"] or not check["trajectory_exact"]:
                        raise RuntimeError("Nondeterministic base output invalidates 1e-5 probes")
                final, seconds_l = timed(lambda: legalize(x, problem))
                finals.append(final.cpu().numpy()); legal_costs.append(seconds_l)
            arrays[mode] = np.stack(outputs); arrays[mode+"_trajectory"] = np.stack(paths)
            arrays[mode+"_final"] = np.stack(finals)
            arrays[mode+"_generation_seconds"] = np.array(costs)
            arrays[mode+"_legalizer_seconds"] = np.array(legal_costs)
            for stage in (mode, mode+"_final"):
                report["stages"][stage], arrays[stage+"_pairwise_rms"] = stage_report(arrays[stage], metric)
            report["timings"][mode] = {"generation_after_untimed_warmup": summarize(costs),
                "legalizer_seconds": summarize(legal_costs), "total_seconds": summarize(np.array(costs)+legal_costs)}
            np.savez_compressed(args.output/"arrays.npz", **arrays)
            print(json.dumps({"mode": mode, "raw_diversity": report["stages"][mode]["diversity"],
                "final_hpwl": report["stages"][mode+"_final"]["macro_hpwl_legal_only"]}), flush=True)
        # Four TRAIN noises, one unit-RMS direction each, both signs, epsilon=1e-5.
        mask = problem.cond.is_ports.cpu().numpy().reshape(-1)
        directions = np.random.default_rng(271828).normal(size=noise[:4].shape)
        directions[:, mask] = 0
        directions /= np.sqrt(np.square(directions[:, ~mask]).mean(axis=(1, 2), keepdims=True))
        arrays["probe_directions"] = directions.astype(np.float32)
        probe_noise, metadata = [], []
        for local in range(4):
            for sign in (-1, 1):
                z = np.clip(noise[local]+sign*1e-5*arrays["probe_directions"][local], -1, 1).astype(np.float32)
                z[mask] = noise[local, mask]
                probe_noise.append(z); metadata.append((local, sign))
        arrays["probe_noise"] = np.stack(probe_noise)
        for mode in ("released_nohard", "experimental_standard_euler"):
            outputs, finals, paths = [], [], []
            for z, (local, sign) in zip(probe_noise, metadata):
                x, path = sample(mode, model, problem, torch.as_tensor(z, device="cuda"))
                final = legalize(x, problem)
                outputs.append(x.cpu().numpy()); finals.append(final.cpu().numpy()); paths.append(path.cpu().numpy())
                dz = difference(z, noise[local]); raw = difference(outputs[-1], arrays[mode][local])
                executed = difference(finals[-1], arrays[mode+"_final"][local])
                report["local_probes"].append({"mode": mode, "source_train_index": int(indices[local]), "sign": sign,
                    "requested_epsilon_rms": 1e-5, "input_difference": dz, "raw_output_difference": raw,
                    "final_projected_difference": executed, "raw_rms_gain": raw["rms"]/dz["rms"],
                    "projected_rms_gain": executed["rms"]/dz["rms"]})
            arrays[mode+"_probe_raw"] = np.stack(outputs)
            arrays[mode+"_probe_final"] = np.stack(finals)
            arrays[mode+"_probe_trajectory"] = np.stack(paths)
        report["probe_summary"] = {}
        for mode in ("released_nohard", "experimental_standard_euler"):
            rows = [r for r in report["local_probes"] if r["mode"] == mode]
            report["probe_summary"][mode] = {"count": len(rows),
                "raw_rms_gain": summarize([r["raw_rms_gain"] for r in rows]),
                "projected_rms_gain": summarize([r["projected_rms_gain"] for r in rows]),
                "projected_unchanged_count": sum(r["final_projected_difference"]["exact_equal"] for r in rows),
                "projected_max_coordinate_change": summarize([r["final_projected_difference"]["max_abs"] for r in rows]),
                "note": "Four directions, two signs each; finite precision and final discrete projection remain. Not a global smoothness proof."}
        np.savez_compressed(args.output/"arrays.npz", **arrays)
        report.update(status="numerical_complete", wall_seconds=time.perf_counter()-started,
                      arrays_sha256=sha(args.output/"arrays.npz"))
        write(args.output/"report.json", report)
        render(arrays, report, args.output)
        report.update(status="completed", wall_seconds=time.perf_counter()-started)
        write(args.output/"report.json", report)
        print(json.dumps({"status": report["status"], "forward_checks": checks,
                          "probe_summary": report["probe_summary"]}, indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        write(args.output/"report.json", report)
        np.savez_compressed(args.output/"arrays_partial.npz", **arrays)
        raise


if __name__ == "__main__":
    main()
