"""Warm device-noise -> legal-coordinate latency, not overall EDA speedup.

The original hard teacher already emits its final layout: no extra projection.
Each saved one-step BC/FQL actor includes its normal clipping and final legalizer.
Loading, first calls, and five warmup calls are recorded separately. No training,
method selection, random-noise generation, host transfer, DEF export or backend
is inside the timed deployment path.
"""
import argparse
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
from analysis.placement_metrics import legality, summarize
from baseline.api import load_problem, load_model, generate, legalize, CHECKPOINT, RUNTIME, UPSTREAM
from baseline.diagnose_final_projection import sha, write
from fql.core import OneStepActor


def timed(call):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter()-started


def differences(actual, expected):
    delta = actual.astype(np.float64)-expected.astype(np.float64)
    return {"exact_equal": bool(np.array_equal(actual, expected)),
            "max_abs": float(np.abs(delta).max()), "mse": float(np.square(delta).mean()),
            "allclose_rtol_1e5_atol_1e6": bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6))}


def snapshot(output):
    paths = [Path(__file__), ROOT/"baseline/benchmark_deployed_latency.slurm", ROOT/"baseline/api.py",
             ROOT/"baseline/diagnose_final_projection.py", ROOT/"analysis/placement_metrics.py", ROOT/"fql/core.py"]
    paths += list((UPSTREAM/"flow_matching").rglob("*.py"))
    paths += list((UPSTREAM/"flow_matching/configs").rglob("*.yaml"))
    paths += list((UPSTREAM/"benchmark_utils").rglob("*.py"))
    records = []
    for path in sorted(set(paths)):
        dest = output/"source_snapshot"/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        records.append({"source": str(path), "snapshot": str(dest), "sha256": sha(dest)})
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("One allocated Slurm GPU required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = snapshot(args.output)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset_hash = sha(args.dataset_dir/"dataset.npz")
    meta = json.loads((args.dataset_dir/"manifest.json").read_text())
    selection = json.loads((args.training_dir/"selection.json").read_text())
    with np.load(args.dataset_dir/"dataset.npz", allow_pickle=False) as data:
        indices = np.arange(240, 256)
        noise = data["noise"][indices].copy()
        hard_reference = data["teacher_action"][indices].reshape(16, -1, 2).copy()
        if not np.all(data["split"][indices] == 2):
            raise ValueError("Expected first16 heldout indices")
    methods = ["teacher_original"]+[f"seed_{seed}/{method}" for method in ["distill_only", "fql_alpha_1"] for seed in [0, 1, 2]]
    if any(method not in selection["methods"] for method in methods[1:]):
        raise ValueError("Methods differ from the existing macro validation selection")
    report = {"status": "running", "purpose": "Warm actual deployment latency only; no quality reselection or backend speedup claim",
              "indices": indices.tolist(), "methods": methods, "timing_rounds": 3, "warmup_calls_per_method": 5,
              "source_snapshot": sources, "provenance": {"dataset_sha256": dataset_hash,
                 "selection_sha256": sha(args.training_dir/"selection.json"), "teacher_checkpoint_sha256": sha(CHECKPOINT),
                 "cache_sha256": sha(RUNTIME/"benchmark_cache"/f"{meta['args']['design']}.pt"),
                 "torch_version": torch.__version__, "numpy_version": np.__version__, "python": platform.python_version(),
                 "gpu": torch.cuda.get_device_name(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                 "threads": torch.get_num_threads(), "deterministic_algorithms": True,
                 "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                 "matmul_allow_tf32": False, "cudnn_allow_tf32": False},
              "scope": {"start": "Provided float32 noise vector already resident on the allocated GPU",
                        "included": "Input reshape, actual model forward/sampling, clipping/masks, required legalizer setup and execution, completion synchronization",
                        "end": "Final normalized macro coordinates resident on GPU",
                        "excluded": "Loading, compilation/first-call cold costs, five warmup calls, noise creation, host transfers, validation checks, DEF conversion, backend, data collection/training",
                        "teacher": "Official hard-constraint 20-step sampler; already legal output, no extra external projection",
                        "student": "Saved BC/FQL actor in batch1, clipping/fixed-mask, one default external legalizer",
                        "timing": "CUDA synchronize before perf_counter start and after result, including host launch/Python work",
                        "order": "All models resident, cyclic method order by round+noise local index to reduce fixed-order effects"},
              "cold": {}, "warmup": {}, "checks": {}, "summary": {}}
    write(args.output/"report.json", report)
    started = time.perf_counter()
    try:
        cold_start = time.perf_counter()
        problem = load_problem(meta["args"]["design"], "cuda")
        device_noise = torch.as_tensor(noise, device="cuda")
        torch.cuda.synchronize()
        report["cold"]["graph_and_noise_device_preparation_seconds"] = time.perf_counter()-cold_start
        sizes = problem.cond.x.cpu().numpy()
        torch.manual_seed(meta["args"]["model_init_seed"])
        (teacher, config), seconds = timed(lambda: load_model(problem, "cuda", steps=20, hard_constraint=True))
        report["cold"]["teacher_loading_seconds"] = seconds
        OmegaConf.save(config, args.output/"teacher_config.yaml")
        actors, expected = {}, {"teacher_original": {"final": hard_reference}}
        for method in methods[1:]:
            path = args.training_dir/method/"checkpoint.pt"
            cold_start = time.perf_counter()
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["data"]["sha256"] != dataset_hash:
                raise ValueError("Checkpoint dataset mismatch")
            actor = OneStepActor(saved["dim"], saved.get("actor_widths", saved["widths"]), saved["movable_mask"], saved["fixed_action"])
            actor.load_state_dict(saved["actor_state"], strict=True)
            actors[method] = actor.cuda().eval()
            torch.cuda.synchronize()
            report["cold"][method+"_loading_seconds"] = time.perf_counter()-cold_start
            with np.load(args.training_dir/method/"proposals.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["noise"][indices], noise)
                expected_raw = archive["raw_action"][indices].reshape(16, -1, 2).copy()
            with np.load(args.training_dir/"measured_heldout"/method/"evaluated.npz", allow_pickle=False) as archive:
                slots = [int(np.flatnonzero(archive["indices"] == index)[0]) for index in indices]
                expected_final = archive["action"][slots].copy()
            expected[method] = {"raw": expected_raw, "final": expected_final}
            report.setdefault("student_checkpoints", {})[method] = {"path": str(path), "sha256": sha(path)}
        def execute(method, local):
            if method == "teacher_original":
                final, _, _ = generate(teacher, problem, noise=device_noise[local].reshape(-1, 2), intermediate_every=0)
                return final, final
            raw = actors[method](device_noise[local].reshape(1, -1), clip=True).reshape(-1, 2)
            return raw, legalize(raw, problem)
        for method in methods:
            _, seconds = timed(lambda: execute(method, 0))
            report["cold"][method+"_first_call_seconds"] = seconds
            values = []
            for local in range(5):
                _, seconds = timed(lambda: execute(method, local))
                values.append(seconds)
            report["warmup"][method] = values
        write(args.output/"report.json", report)
        records, outputs, raw_outputs = [], [], []
        first_by_key = {}
        for round_id in range(3):
            for local, index in enumerate(indices):
                offset = (round_id+local) % len(methods)
                order = methods[offset:]+methods[:offset]
                for method in order:
                    (raw, final), seconds = timed(lambda: execute(method, local))
                    # All host transfers and checks below occur after the timer stops.
                    raw_cpu, final_cpu = raw.cpu().numpy(), final.cpu().numpy()
                    geometry = legality(final_cpu, sizes)
                    if not geometry["legal"]:
                        raise RuntimeError("Timed path produced an illegal output: "+method)
                    key = (method, int(index))
                    repeated_exact = key not in first_by_key or np.array_equal(first_by_key[key], final_cpu)
                    if key not in first_by_key:
                        first_by_key[key] = final_cpu.copy()
                    final_diff = differences(final_cpu, expected[method]["final"][local])
                    raw_diff = None if method == "teacher_original" else differences(raw_cpu, expected[method]["raw"][local])
                    if method == "teacher_original" and not final_diff["exact_equal"]:
                        raise AssertionError("Original teacher differs from deterministic cached baseline")
                    if not repeated_exact:
                        raise AssertionError("Repeated identical deployment input changed its final layout")
                    records.append({"method": method, "index": int(index), "round": round_id, "seconds": seconds,
                                    "training_seed": None if method == "teacher_original" else int(method.split("/")[0][5:]),
                                    "legal": True, "repeated_final_exact": repeated_exact,
                                    "final_vs_saved": final_diff, "raw_vs_saved_batch256": raw_diff})
                    outputs.append(final_cpu.copy()); raw_outputs.append(raw_cpu.copy())
            print(json.dumps({"phase": "timing", "round_completed": round_id+1}), flush=True)
        np.savez_compressed(args.output/"timed_outputs.npz", indices=indices, noise=noise,
                            final=np.stack(outputs), raw=np.stack(raw_outputs),
                            record_method=np.asarray([r["method"] for r in records]),
                            record_index=np.asarray([r["index"] for r in records]),
                            record_round=np.asarray([r["round"] for r in records]))
        with (args.output/"timings.jsonl").open("w") as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        for method in methods:
            rows = [r for r in records if r["method"] == method]
            per_noise = [{"index": int(index), "seconds": summarize([r["seconds"] for r in rows if r["index"] == index])} for index in indices]
            report["summary"][method] = {"all_calls_seconds": summarize([r["seconds"] for r in rows]),
                                         "per_noise_round_mean_seconds": summarize([r["seconds"]["mean"] for r in per_noise]),
                                         "per_noise": per_noise,
                                         "note": "Three repeated timings per fixed noise; these are not independent layouts or training seeds"}
            report["checks"][method] = {"all_final_exact_vs_saved": all(r["final_vs_saved"]["exact_equal"] for r in rows),
                                        "num_final_mismatched_calls": sum(not r["final_vs_saved"]["exact_equal"] for r in rows),
                                        "max_final_difference": max(r["final_vs_saved"]["max_abs"] for r in rows),
                                        "max_raw_difference_batch1_vs_saved_batch256": None if method == "teacher_original" else max(r["raw_vs_saved_batch256"]["max_abs"] for r in rows),
                                        "all_repeated_exact": True, "all_geometrically_legal": True,
                                        "scope": "Students use batch1 here; original proposals used batch256. Any kernel-rounding differences are retained, not silently redefined as exact baseline identity."}
        report["training_seed_variation"] = {}
        for family in ["distill_only", "fql_alpha_1"]:
            report["training_seed_variation"][family] = {
                "seed_means_seconds": {str(seed): report["summary"][f"seed_{seed}/{family}"]["all_calls_seconds"]["mean"] for seed in [0, 1, 2]},
                "summary_of_three_seed_means": summarize([report["summary"][f"seed_{seed}/{family}"]["all_calls_seconds"]["mean"] for seed in [0, 1, 2]])}
        report.update(status="completed", total_process_seconds=time.perf_counter()-started,
                      timing_records_sha256=sha(args.output/"timings.jsonl"), timed_outputs_sha256=sha(args.output/"timed_outputs.npz"),
                      source_changed_during_run=[r["source"] for r in sources if sha(r["source"]) != r["sha256"]])
        write(args.output/"report.json", report)
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}, total_process_seconds=time.perf_counter()-started)
        write(args.output/"report.json", report)
        raise


if __name__ == "__main__":
    main()
