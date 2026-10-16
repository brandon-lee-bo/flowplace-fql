"""Bounded train-only no-hard-constraint teacher diagnostic, not PPA evidence."""
import argparse
import hashlib
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
from omegaconf import OmegaConf
from analysis.placement_metrics import MacroHPWL, legality, summarize
from baseline.api import load_problem, load_model, generate, legalize, CHECKPOINT, RUNTIME


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter()-start


def diversity(x):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    centered = x-x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    pairs = np.sqrt(np.square(x[:, None]-x[None, :]).mean(axis=-1))
    upper = pairs[np.triu_indices(len(x), 1)]
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if energy.sum() > 0:
        p = energy[energy>0]/energy.sum()
        rank = float(np.exp(-np.sum(p*np.log(p))))
    else:
        rank = 0.
    return {"coordinate_std_mean": float(sd.mean()),
            "coordinate_std_rms": float(np.sqrt(np.mean(sd**2))),
            "coordinate_std_max": float(sd.max()),
            "pairwise_layout_rms_distance": summarize(upper),
            "unique_exact": len(np.unique(x, axis=0)),
            "unique_round_6": len(np.unique(x.round(6), axis=0)),
            "unique_round_3": len(np.unique(x.round(3), axis=0)),
            "centered_singular_values": singular.tolist(),
            "effective_rank_entropy": rank,
            "note": "Pair distances share inputs; they are not independent statistical replicates."}, pairs


def stage_report(x, metric):
    reports = [legality(p, metric.sizes) for p in x]
    if not all(r["finite"] for r in reports):
        raise FloatingPointError("Nonfinite coordinates")
    legal = np.array([r["legal"] for r in reports])
    hpwl = metric(x)
    spread, distances = diversity(x)
    return {"center_min": float(x.min()), "center_max": float(x.max()),
            "center_fraction_outside_minus1_plus1": float((np.abs(x)>1).mean()),
            "center_fraction_at_internal_clamp_minus2_plus2": float((np.abs(x)>=2-1e-7).mean()),
            "legal_fraction": float(legal.mean()), "legal_count": int(legal.sum()),
            "overlap_pairs": summarize([r["overlap_pairs"] for r in reports]),
            "out_of_bounds_macros": summarize([r["out_of_bounds"] for r in reports]),
            "macro_hpwl_all_geometric_proxy": summarize(hpwl),
            "macro_hpwl_legal_only": summarize(hpwl[legal]), "diversity": spread,
            "per_noise": [{"local_index": i, **r, "macro_hpwl": float(h)}
                          for i, (r, h) in enumerate(zip(reports, hpwl))]}, distances


def plot(arrays, report, metric, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    names = ["hard_original", "nohard_raw", "nohard_clipped", "nohard_raw_final", "nohard_clipped_final"]
    labels = ["Hard teacher", "No-hard raw", "No-hard clipped", "Raw + final projection", "Clipped + final projection"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    distances = [arrays[n+"_pairwise_rms"][np.triu_indices(len(arrays[n]), 1)] for n in names]
    axes[0].boxplot(distances, labels=labels)
    axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[0].set(ylabel="Between-noise layout RMS distance", title="Diversity across the same 16 TRAIN noises")
    for i, name in enumerate(names):
        rows = report["stages"][name]["per_noise"]
        valid = [r["macro_hpwl"]/1000 for r in rows if r["legal"]]
        if valid:
            axes[1].boxplot([valid], positions=[i+1], widths=.5)
        axes[1].text(i+1, .96, f"{len(valid)}/16 legal", transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=8)
    axes[1].margins(y=.18)
    axes[1].set(xticks=np.arange(1, len(names)+1), xticklabels=labels,
                ylabel="Macro HPWL (thousand cache units)", title="Legal layouts only; no routed PPA")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"diversity_and_macro_proxy.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, name, label in zip(axes, ["hard_original", "nohard_raw_final", "nohard_clipped_final"],
                              ["Hard teacher", "No-hard raw + final projection", "No-hard clipped + final projection"]):
        for j, positions in enumerate(arrays[name]):
            for i, (center, size) in enumerate(zip(positions, metric.sizes)):
                ax.add_patch(Rectangle(center-size/2, *size, facecolor="none", edgecolor=plt.get_cmap("tab20")(i),
                                       alpha=.13, linewidth=.8))
        ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal", title=label,
               xlabel="Normalized x", ylabel="Normalized y")
    fig.suptitle("Overlay of all 16 training-noise layouts; overlapping rectangles are different samples")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"layout_diversity_overlay.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render-existing", action="store_true", help="CPU-only plot recovery; never regenerate samples")
    args = p.parse_args()
    if args.render_existing:
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Use a Slurm CPU allocation for rendering")
        report = json.loads((args.output/"report.json").read_text())
        if len(report["stages"]) != 5 or len(report["hard_cache_comparisons"]) != 16:
            raise ValueError("Missing completed numerical diagnostic")
        if not (args.output/"report_before_plot_recovery.json").exists():
            write(args.output/"report_before_plot_recovery.json", report)
        with np.load(args.output/"arrays.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        design = report["provenance"]["source_manifest"]["args"]["design"]
        cache = torch.load(RUNTIME/"benchmark_cache"/f"{design}.pt", map_location="cpu", weights_only=False)
        started = time.perf_counter()
        plot(arrays, report, MacroHPWL(cache), args.output)
        if "render_recovery" in report:
            report.setdefault("render_history", []).append(report["render_recovery"])
        report["render_recovery"] = {"previous_failure": report.pop("failure", None),
            "render_slurm_job_id": os.environ["SLURM_JOB_ID"], "render_script_sha256": sha(__file__),
            "seconds": time.perf_counter()-started, "regenerated_samples": False}
        report.update(status="completed", arrays_sha256=sha(args.output/"arrays.npz"))
        write(args.output/"report.json", report)
        print(json.dumps({"status": "completed", "render_existing_only": True,
            "spread_relative_to_hard_teacher": report["spread_relative_to_hard_teacher"]}), flush=True)
        return
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Submit this bounded diagnostic with one Slurm GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.backends.cudnn.benchmark = False
    source = json.loads((args.dataset_dir/"manifest.json").read_text())
    design = source["args"]["design"]
    cache_path = RUNTIME/"benchmark_cache"/f"{design}.pt"
    provenance = {"dataset_sha256": sha(args.dataset_dir/"dataset.npz"),
        "checkpoint_sha256": sha(CHECKPOINT), "cache_sha256": sha(cache_path),
        "adapter_sha256": sha(ROOT/"baseline/api.py"), "script_sha256": sha(__file__),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "gpu": torch.cuda.get_device_name(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "model_init_seed": source["args"]["model_init_seed"],
        "source_manifest": source}
    for key in ("checkpoint_sha256", "cache_sha256", "adapter_sha256", "torch_version"):
        if provenance[key] != source["provenance"][key]:
            raise ValueError(f"Source provenance mismatch for {key}")
    report = {"status": "running", "provenance": provenance,
        "purpose": "Determine whether removing intermediate projections preserves useful teacher diversity",
        "scope": "First 16 TRAIN noise families only; same model instance and weights; no training or heldout tuning",
        "caveat": "Macro-only geometric HPWL excludes standard cells and I/O. It is not PPA; illegal raw HPWL is not feasible quality.",
        "changes": ["Only use_hard_constraint_in_sampling toggled; original current-x interpolation and internal [-2,2] clamp retained",
                    "No-hard raw is optionally clipped to [-1,1]; final legalizer applied once to raw and clipped in separate diagnostic branches",
                    "External legalizer is the same eager grid128, padding0.001, w_reg0.1,w_dist1 used by all student evaluations"],
        "stages": {}, "timings": {}, "repeat_checks": [], "hard_cache_comparisons": []}
    write(args.output/"report.json", report)
    started = time.perf_counter()
    arrays = {}
    try:
        with np.load(args.dataset_dir/"dataset.npz", allow_pickle=False) as d:
            indices = np.flatnonzero(d["split"] == 0)[:16]
            if len(indices) != 16:
                raise ValueError("Require 16 training families")
            arrays["source_train_indices"] = indices
            arrays["noise"] = d["noise"][indices].reshape(16, -1, 2).copy()
            arrays["cached_hard_teacher"] = d["teacher_action"][indices].reshape(arrays["noise"].shape).copy()
        problem = load_problem(design, "cuda")
        metric = MacroHPWL(problem.benchmark_info)
        arrays["sizes"] = metric.sizes
        arrays["macro_names"] = np.asarray(metric.names)
        report["metric"] = metric.metadata()
        report["source_train_indices"] = indices.tolist()
        torch.manual_seed(provenance["model_init_seed"])
        model, cfg = load_model(problem, "cuda", steps=20, hard_constraint=True)
        OmegaConf.save(cfg, args.output/"hard_config.yaml")
        for mode, hard in (("hard_original", True), ("nohard_raw", False)):
            model.use_hard_constraint_in_sampling = hard
            outputs, paths, seconds = [], [], []
            for local, z in enumerate(arrays["noise"]):
                noise = torch.as_tensor(z, device="cuda")
                (x, _, trajectory), elapsed = timed(lambda: generate(model, problem, noise=noise, intermediate_every=1))
                outputs.append(x.cpu().numpy()); paths.append(torch.stack(trajectory).cpu().numpy()); seconds.append(elapsed)
                if hard:
                    same = np.array_equal(outputs[-1], arrays["cached_hard_teacher"][local])
                    report["hard_cache_comparisons"].append({"source_train_index": int(indices[local]), "exact_equal": same,
                        "max_absolute_difference": float(np.abs(outputs[-1]-arrays["cached_hard_teacher"][local]).max())})
                    if not same:
                        raise RuntimeError("Hard teacher did not reproduce source cache exactly")
                if local == 0:
                    (repeat, _, _), repeat_sec = timed(lambda: generate(model, problem, noise=noise, intermediate_every=1))
                    same = torch.equal(repeat, x)
                    report["repeat_checks"].append({"mode": mode, "exact_equal": same, "repeat_seconds": repeat_sec})
                    if not same:
                        raise RuntimeError("Same-noise sampling is not deterministic")
            arrays[mode] = np.stack(outputs)
            arrays[mode+"_trajectory"] = np.stack(paths)
            arrays[mode+"_generation_seconds"] = np.asarray(seconds)
            report["timings"][mode] = {"first_generation_including_setup_seconds": seconds[0],
                "remaining_15_generation_seconds": summarize(seconds[1:]), "all_generation_seconds": summarize(seconds)}
            np.savez_compressed(args.output/"arrays.npz", **arrays)
            print(json.dumps({"mode": mode, "generation_seconds": seconds}), flush=True)
        arrays["nohard_clipped"] = np.clip(arrays["nohard_raw"], -1, 1)
        for name in ("nohard_raw", "nohard_clipped"):
            outputs, seconds = [], []
            for x in arrays[name]:
                (out, elapsed) = timed(lambda: legalize(torch.as_tensor(x, device="cuda"), problem))
                outputs.append(out.cpu().numpy()); seconds.append(elapsed)
            stage = name+"_final"
            arrays[stage] = np.stack(outputs)
            arrays[stage+"_legalizer_seconds"] = np.asarray(seconds)
            arrays[stage+"_total_seconds"] = arrays["nohard_raw_generation_seconds"]+np.asarray(seconds)
            report["timings"][stage] = {"final_legalizer_seconds": summarize(seconds),
                "total_generation_plus_legalizer_seconds": summarize(arrays[stage+"_total_seconds"]),
                "remaining_15_total_seconds": summarize(arrays[stage+"_total_seconds"][1:])}
        for name in ("hard_original", "nohard_raw", "nohard_clipped", "nohard_raw_final", "nohard_clipped_final"):
            report["stages"][name], arrays[name+"_pairwise_rms"] = stage_report(arrays[name], metric)
        report["noise_diversity"], arrays["noise_pairwise_rms"] = diversity(arrays["noise"])
        report["clipping"] = {"coordinate_fraction_changed": float((arrays["nohard_raw"] != arrays["nohard_clipped"]).mean()),
            "raw_vs_clipped_final_exact_equal": bool(np.array_equal(arrays["nohard_raw_final"], arrays["nohard_clipped_final"])),
            "raw_vs_clipped_final_rms_per_family": summarize(np.sqrt(np.square(arrays["nohard_raw_final"]-arrays["nohard_clipped_final"]).mean(axis=(1, 2))))}
        hard_sd = report["stages"]["hard_original"]["diversity"]["coordinate_std_rms"]
        report["spread_relative_to_hard_teacher"] = {name: stage["diversity"]["coordinate_std_rms"]/hard_sd
            for name, stage in report["stages"].items()}
        np.savez_compressed(args.output/"arrays.npz", **arrays)
        plot(arrays, report, metric, args.output)
        report.update(status="completed", wall_seconds=time.perf_counter()-started,
                      arrays_sha256=sha(args.output/"arrays.npz"))
        write(args.output/"report.json", report)
        print(json.dumps({"output": str(args.output), "status": report["status"],
                          "spread_relative_to_hard_teacher": report["spread_relative_to_hard_teacher"],
                          "legal_fractions": {k: v["legal_fraction"] for k,v in report["stages"].items()}}, indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", wall_seconds=time.perf_counter()-started,
                      failure={"type": type(exc).__name__, "message": str(exc)})
        write(args.output/"report.json", report)
        if arrays:
            np.savez_compressed(args.output/"arrays_partial.npz", **arrays)
        raise


if __name__ == "__main__":
    main()
