"""Two fixed anchors + one identical final legalizer; no learned generation."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import load_problem, legalize, save_sample, RUNTIME
from baseline.diagnose_final_projection import sha, write, timed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Use a Slurm GPU, matching prior legalizer arithmetic")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    source = json.loads((args.dataset_dir/"manifest.json").read_text())
    with np.load(args.dataset_dir/"dataset.npz", allow_pickle=False) as data:
        train = np.flatnonzero(data["split"] == 0)
        if len(train) != 192:
            raise ValueError("Expected all 192 TRAIN hard-teacher layouts")
        mean = data["teacher_action"][train].astype(np.float64).mean(axis=0).astype(np.float32).reshape(-1, 2)
    design = source["args"]["design"]
    problem = load_problem(design, "cuda")
    metric = MacroHPWL(problem.benchmark_info)
    mask = problem.cond.is_ports.cpu().numpy().reshape(-1)
    fixed = problem.x.cpu().numpy()
    zero = np.zeros_like(fixed)
    zero[mask] = fixed[mask]; mean[mask] = fixed[mask]
    arrays = {"training_family_indices": train, "fixed_mask": mask, "fixed_coordinates": fixed,
              "sizes": metric.sizes, "zero_centers_raw": zero, "train_mean_raw": mean}
    provenance = {"dataset_sha256": sha(args.dataset_dir/"dataset.npz"),
        "cache_sha256": sha(RUNTIME/"benchmark_cache"/f"{design}.pt"),
        "adapter_sha256": sha(ROOT/"baseline/api.py"), "script_sha256": sha(__file__),
        "torch_version": torch.__version__, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    if provenance["cache_sha256"] != source["provenance"]["cache_sha256"]:
        raise ValueError("Cache mismatch")
    report = {"status": "running", "provenance": provenance, "metric": metric.metadata(),
        "purpose": "Test whether central fixed anchors plus legalizer explain small macro-only HPWL",
        "warning": "Two one-off fixed-layout controls. No training, noise sampling or routed PPA evaluation.",
        "mean_construction": "Float64 coordinate mean of all192 TRAIN original hard teachers, cast to float32; fixed-node positions restored",
        "legalizer": "Exactly one baseline.api.legalize per anchor: same GPU, grid128, padding0.001, w_reg0.1, w_dist1",
        "controls": {}}
    for name in ("zero_centers", "train_mean"):
        raw = arrays[name+"_raw"]
        final, seconds = timed(lambda: legalize(torch.as_tensor(raw, device="cuda"), problem))
        position = final.cpu().numpy()
        arrays[name+"_final"] = position
        report["controls"][name] = {"raw_geometry": legality(raw, metric.sizes),
            "final_geometry": legality(position, metric.sizes),
            "raw_macro_hpwl_geometric_only": metric(raw), "final_macro_hpwl": metric(position),
            "projection_coordinate_mse": float(np.square(position-raw).mean()),
            "legalizer_seconds_first_call_not_a_latency_benchmark": seconds,
            "noise_conditioned": False, "generated_layout_count": 1,
            "unique_layouts_by_construction": 1, "fixed_macro_count": int(mask.sum())}
        # Noise is an explicit zero placeholder, not a sampled conditioning input.
        save_sample(args.output/name, final, torch.zeros_like(final), problem,
            {"method": "fixed_anchor_"+name, "noise_placeholder_only": True, "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    np.savez_compressed(args.output/"arrays.npz", **arrays)
    report.update(status="completed", arrays_sha256=sha(args.output/"arrays.npz"))
    write(args.output/"report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
