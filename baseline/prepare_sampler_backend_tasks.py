"""Freeze/export eight exploratory sampler/anchor GRT tasks; never run GRT.

Aggregate plan also reuses three exact teacher_original backend requests.
Constants appear once, never as three purported independent noise samples.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import load_problem, load_model, save_sample, legalize, CHECKPOINT, RUNTIME, UPSTREAM
from baseline.diagnose_euler_sampler import sample
from baseline.diagnose_final_projection import sha, write, timed
from omegaconf import OmegaConf


def action_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def jsonl(path, values):
    Path(path).write_text("".join(json.dumps(v, allow_nan=False)+"\n" for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reuse-tasks", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="bp_fe_sampler_transfer_v1")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Generation and exports require an allocated Slurm GPU job")
    if Path(args.prefix).name != args.prefix:
        raise ValueError("Invalid run prefix")
    args.dataset_dir, args.reuse_tasks, args.anchor_dir, args.output = (
        p.resolve() for p in (args.dataset_dir, args.reuse_tasks, args.anchor_dir, args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.backends.cudnn.benchmark = False
    meta = json.loads((args.dataset_dir/"manifest.json").read_text())
    design = meta["args"]["design"]
    dataset_hash = sha(args.dataset_dir/"dataset.npz")
    with np.load(args.dataset_dir/"dataset.npz", allow_pickle=False) as archive:
        indices = np.flatnonzero(archive["split"] == 2)[:3]
        noise = archive["noise"][indices].reshape(3, -1, 2).copy()
        teacher = archive["teacher_action"][indices].reshape(noise.shape).copy()
    if indices.tolist() != [240, 241, 242]:
        raise ValueError("This bounded plan is explicitly frozen to indices240,241,242")
    reuse = []
    for line in args.reuse_tasks.read_text().splitlines():
        task = json.loads(line)
        if task["method"] != "teacher_original" or task["index"] not in indices:
            continue
        original = copy.deepcopy(task)
        if task["dataset_sha256"] != dataset_hash or task["split"] != 2:
            raise ValueError("Reused teacher belongs to another dataset/split")
        for path_key, hash_key in (("def_path", "def_sha256"), ("def_audit_path", "def_audit_sha256"),
                                   ("action_path", "action_file_sha256")):
            if sha(task[path_key]) != task[hash_key]:
                raise ValueError("Reused teacher artifact changed: "+path_key)
        expected_action = teacher[np.flatnonzero(indices == task["index"])[0]]
        if action_sha(expected_action) != task["action_sha256"]:
            raise ValueError("Reused teacher action does not match source cached hard teacher")
        task.update(role="paired_reference", noise_conditioning=True, reused_backend_measurement=True,
            reused_source_tasks_path=str(args.reuse_tasks), reused_source_tasks_sha256=sha(args.reuse_tasks),
            reused_source_line_sha256=hashlib.sha256(line.encode()).hexdigest(), reused_source_task=original)
        metrics = Path(task["metrics_path"])
        manifest = metrics.with_name("manifest.json")
        # Hash available existing artifacts without interpreting outcome values.
        if metrics.exists():
            task["reused_metrics_sha256_at_plan"] = sha(metrics)
        if manifest.exists():
            task["reused_backend_manifest_sha256_at_plan"] = sha(manifest)
        reuse.append(task)
    reuse.sort(key=lambda t: t["index"])
    if [t["index"] for t in reuse] != indices.tolist():
        raise ValueError("Need exactly three existing teacher requests")
    method_modes = [("released_nohard_final", "released_nohard"),
                    ("experimental_standard_euler_final", "experimental_standard_euler")]
    specs = []
    for method, mode in method_modes:
        for index in indices:
            specs.append({"method": method, "sampler_mode": mode, "index": int(index),
                          "role": "candidate", "noise_conditioning": True})
    for method, key in (("fixed_zero_anchor", "zero_centers"), ("fixed_train_mean_anchor", "train_mean")):
        specs.append({"method": method, "anchor_key": key, "index": 240,
                      "role": "fixed_control", "noise_conditioning": False})
    for j, spec in enumerate(specs):
        spec["run_id"] = f"{args.prefix}_n{j:02d}_i{spec['index']:04d}"
        spec["task_id"] = spec["run_id"]
    provenance = {"dataset_sha256": dataset_hash, "checkpoint_sha256": sha(CHECKPOINT),
        "cache_sha256": sha(RUNTIME/"benchmark_cache"/f"{design}.pt"),
        "adapter_sha256": sha(ROOT/"baseline/api.py"), "script_sha256": sha(__file__),
        "sampler_code_sha256": sha(ROOT/"baseline/diagnose_euler_sampler.py"),
        "upstream_models_sha256": sha(UPSTREAM/"flow_matching/models.py"),
        "anchor_report_sha256": sha(args.anchor_dir/"report.json"),
        "anchor_arrays_sha256": sha(args.anchor_dir/"arrays.npz"),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "model_init_seed": meta["args"]["model_init_seed"], "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "slurm_job_id": os.environ["SLURM_JOB_ID"]}
    for key in ("checkpoint_sha256", "cache_sha256", "adapter_sha256", "torch_version"):
        if provenance[key] != meta["provenance"][key]:
            raise ValueError("Source mismatch: "+key)
    anchor_report = json.loads((args.anchor_dir/"report.json").read_text())
    if anchor_report["status"] != "completed" or provenance["anchor_arrays_sha256"] != anchor_report["arrays_sha256"]:
        raise ValueError("Unverified fixed anchor source")
    if anchor_report["provenance"]["dataset_sha256"] != dataset_hash:
        raise ValueError("Fixed anchors belong to another dataset")
    with np.load(args.anchor_dir/"arrays.npz", allow_pickle=False) as archive:
        anchors = {k: archive[k] for k in archive.files}
    plan = {"phase": "exploratory", "design": design, "dataset_sha256": dataset_hash,
        "reference_method": "teacher_original",
        "purpose": "Frozen sampler/anchor extension measuring actual GRT transfer; no backend-based method selection",
        "backend_stage": "global_route_only", "not_final_table2": True,
        "no_backend_based_model_selection": True,
        "candidates": [{"method": t["method"], "index": t["index"], "role": t["role"],
                        "noise_conditioning": t["noise_conditioning"]} for t in reuse+specs],
        "task_ids": [t["task_id"] for t in reuse+specs],
        "new_execution_task_ids": [t["task_id"] for t in specs],
        "reused_task_ids": [t["task_id"] for t in reuse],
        "reused_source_tasks": str(args.reuse_tasks), "reused_source_tasks_sha256": sha(args.reuse_tasks),
        "indices_for_noise_conditioned_methods": indices.tolist(),
        "constant_index_semantics": "Fixed controls each appear exactly once. index240/split2 are evaluation bookkeeping only; no noise240 is used. Zero anchor is data-free; train mean uses only192 TRAIN original hard teachers.",
        "sampling": {"steps": 20, "prior": "uniform cached paired noise", "intermediate_hard_projection": False,
            "final_legalizer_calls_per_generated_layout": 1, "final_legalizer": "eager grid128 padding0.001 w_reg0.1 w_dist1",
            "per_step_clamp": [-2, 2], "external_pre_legalizer_clipping": False,
            "released_nohard": "Original upstream reverse_samples unchanged with hard=false",
            "experimental_standard_euler": "Experimental x_next=x+(t_next-t)*v, same forward/mask/clamp/timegrid; not official baseline or a silent bugfix"},
        "freeze_unix_time": time.time(), "provenance": provenance}
    plan_path = args.output/"selection_plan.json"
    write(plan_path, plan)  # Freeze before generating new coordinates or seeing new backend outcomes.
    plan_hash = sha(plan_path)
    report = {"status": "generating", "plan_sha256": plan_hash, "provenance": provenance,
              "new_tasks": [], "reused_tasks": [t["task_id"] for t in reuse], "failures": []}
    write(args.output/"generation_manifest.json", report)
    tasks, execution = list(reuse), []
    try:
        problem = load_problem(design, "cuda")
        metric = MacroHPWL(problem.benchmark_info)
        for task in tasks:
            task["normalized_sizes"] = metric.sizes.tolist()
        torch.manual_seed(provenance["model_init_seed"])
        model, cfg = load_model(problem, "cuda", steps=20, hard_constraint=False)
        OmegaConf.save(cfg, args.output/"sampler_config.yaml")
        for spec in specs:
            method, index, run_id = spec["method"], spec["index"], spec["run_id"]
            folder = args.output/run_id
            conditioning = spec["noise_conditioning"]
            if conditioning:
                z = noise[np.flatnonzero(indices == index)[0]]
                (raw_gpu, path_gpu), generation_s = timed(lambda: sample(spec["sampler_mode"], model, problem,
                                                                         torch.as_tensor(z, device="cuda")))
                final_gpu, projection_s = timed(lambda: legalize(raw_gpu, problem))
                raw, x, trajectory = raw_gpu.cpu().numpy(), final_gpu.cpu().numpy(), path_gpu.cpu().numpy()
                original = None
            else:
                key = spec["anchor_key"]
                raw, x = anchors[key+"_raw"], anchors[key+"_final"]
                z = np.zeros_like(x)  # Explicit placeholder; never treated as conditioning noise.
                trajectory = None
                generation_s, projection_s = None, None
                original = {"path": str(args.anchor_dir/"arrays.npz"), "sha256": provenance["anchor_arrays_sha256"],
                            "raw_key": key+"_raw", "final_key": key+"_final", "regenerated": False}
            geom = legality(x, metric.sizes)
            if not geom["legal"]:
                raise ValueError(f"Predeclared candidate is illegal: {method}/{index}")
            save_sample(folder, torch.as_tensor(x), torch.as_tensor(z), problem,
                {"method": method, "index": index, "split": 2, "role": spec["role"],
                 "noise_conditioning": conditioning, "noise_placeholder_only": not conditioning,
                 "selection": "Exact frozen candidate; no best-of-K or replacement", "plan_sha256": plan_hash,
                 "slurm_job_id": os.environ["SLURM_JOB_ID"]})
            arrays = {"action": x[None], "raw_action": raw[None], "noise": z[None],
                      "indices": np.array([index]), "noise_conditioning": np.array([conditioning])}
            if trajectory is not None:
                arrays["trajectory"] = trajectory[None]
            action_path = folder/"executed_action.npz"
            np.savez_compressed(action_path, **arrays)
            def_path = folder/f"{design}.def"
            subprocess.run([sys.executable, str(ROOT/"backend/scripts/export_macro_def.py"),
                "--design", design, "--coordinates", str(folder/"macro_physical.json"),
                "--output", str(def_path)], check=True, stdout=subprocess.DEVNULL)
            audit_path = def_path.with_suffix(".audit.json")
            audit = json.loads(audit_path.read_text())
            if audit["legal_no_halo"] is not True or audit["output_sha256"] != sha(def_path):
                raise ValueError("Exported rounded DEF failed geometry/hash audit")
            task = {"task_id": run_id, "run_id": run_id, "method": method, "algorithm": method,
                "index": index, "split": 2, "training_seed": None, "design": design,
                "role": spec["role"], "noise_conditioning": conditioning,
                "index_is_bookkeeping_only": not conditioning,
                "fixed_control_semantics": None if conditioning else plan["constant_index_semantics"],
                "def_path": str(def_path), "def_sha256": sha(def_path),
                "def_audit_path": str(audit_path), "def_audit_sha256": sha(audit_path),
                "dataset_sha256": dataset_hash, "action_path": str(action_path),
                "action_file_sha256": sha(action_path), "action_key": "action", "action_row": 0,
                "normalized_action_flat": x.reshape(-1).tolist(), "action_sha256": action_sha(x),
                "normalized_sizes": metric.sizes.tolist(), "macro_hpwl": metric(x), "legal": True,
                "predicted_q": None, "q_reward_stage": None,
                "metrics_path": str(ROOT/"backend/results"/run_id/"metrics.json"),
                "plan_sha256": plan_hash, "sampler_provenance": provenance if conditioning else None,
                "sampler_settings": plan["sampling"] if conditioning else None,
                "sampler_mode": spec.get("sampler_mode"), "trajectory_key": "trajectory" if conditioning else None,
                "fixed_anchor_source": original, "generation_seconds": generation_s,
                "final_legalizer_seconds": projection_s, "reused_backend_measurement": False}
            tasks.append(task); execution.append(task)
            report["new_tasks"].append({"task_id": run_id, "action_sha256": task["action_sha256"],
                "def_sha256": task["def_sha256"], "legal": True, "macro_hpwl": task["macro_hpwl"]})
            write(args.output/"generation_manifest.json", report)
        if sha(plan_path) != plan_hash:
            raise RuntimeError("Frozen plan was modified during generation")
        if len(tasks) != 11 or len(execution) != 8:
            raise RuntimeError("Expected all11 aggregate tasks and new8 execution tasks")
        jsonl(args.output/"tasks.jsonl", tasks)
        jsonl(args.output/"execution_tasks.jsonl", execution)
        report.update(status="ready", aggregate_count=len(tasks), execution_count=len(execution),
            tasks_sha256=sha(args.output/"tasks.jsonl"), execution_tasks_sha256=sha(args.output/"execution_tasks.jsonl"),
            note="No backend process launched. Root schedules only execution_tasks.jsonl, then aggregates tasks.jsonl with frozen explicit candidates.")
        write(args.output/"generation_manifest.json", report)
        print(json.dumps({"status": "ready", "aggregate_count": len(tasks), "execution_count": len(execution),
                          "output": str(args.output), "new_task_ids": [t["task_id"] for t in execution]}, indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", failures=[{"type": type(exc).__name__, "message": str(exc)}])
        write(args.output/"generation_manifest.json", report)
        jsonl(args.output/"partial_tasks.jsonl", tasks)
        raise


if __name__ == "__main__":
    main()
