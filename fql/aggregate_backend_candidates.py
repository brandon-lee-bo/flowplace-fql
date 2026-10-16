"""Aggregate actual GRT results without mixing macro proxies or hiding failures.

Every method/index in the frozen plan must be present exactly once. Failures are
written to report.json, then the command fails; no success-only comparison is
produced. This module never launches backend evaluation or trains a model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.placement_metrics import legality, summarize
from fql.core import critic_diagnostics
from fql.relabel_backend import STAGE, action_sha256, numeric, resolve_from, sha256_file, write_json


def method_identity(method, supplied_seed=None):
    match = re.fullmatch(r"seed_([^/]+)/(.+)", method)
    if match:
        seed_text, configuration = match.groups()
        seed = int(seed_text) if seed_text.isdigit() else seed_text
        if supplied_seed is not None and str(supplied_seed) != str(seed):
            raise ValueError("training_seed disagrees with seed in method path")
    else:
        configuration, seed = method, supplied_seed
    return configuration, seed


def task_key(task):
    method, index = task.get("method"), task.get("index")
    if not isinstance(method, str) or not method:
        raise ValueError("Each task requires a method")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Each task requires a nonnegative noise index")
    method_identity(method, task.get("training_seed"))
    return method, index


def plan_keys(plan):
    if "candidates" in plan:
        entries = [task_key(row) for row in plan["candidates"]]
    elif "methods" in plan and "indices" in plan:
        entries = [(method, index) for method in plan["methods"] for index in plan["indices"]]
        for method, index in entries:
            task_key({"method": method, "index": index})
    else:
        raise ValueError("Frozen plan needs candidates[{method,index}] or methods plus indices")
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("Frozen plan is empty or has duplicate method/index pairs")
    return set(entries)


def executed_action(task, base):
    file_action = None
    if "action_path" in task:
        path = resolve_from(task["action_path"], base)
        if "action_file_sha256" in task and sha256_file(path) != task["action_file_sha256"]:
            raise ValueError("Executed action NPZ changed")
        with np.load(path, allow_pickle=False) as archive:
            file_action = archive[task.get("action_key", "action")][int(task["action_row"])]
    if "normalized_action_flat" in task:
        action = np.asarray(task["normalized_action_flat"], dtype=np.float32)
    elif file_action is not None:
        action = file_action
    else:
        raise ValueError("Need normalized_action_flat or action_path/key/row to verify action binding")
    action = np.asarray(action, dtype=np.float32).reshape(-1, 2)
    if not len(action) or not np.isfinite(action).all():
        raise ValueError("Executed layout has invalid coordinates")
    if action_sha256(action) != task.get("action_sha256"):
        raise ValueError("Executed action SHA mismatch")
    if file_action is not None and action_sha256(file_action) != task["action_sha256"]:
        raise ValueError("Inline action and source artifact disagree")
    return action


def q_information(task, base, measured_rwl):
    prediction = task.get("predicted_q")
    info = {"prediction": prediction, "calibration_allowed": False}
    if prediction is None:
        info["reason"] = "No predicted Q for this baseline"
        return info
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not len(prediction) or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predicted Q")
    info["prediction"] = prediction.tolist()
    stage = task.get("q_reward_stage")
    info["reward_stage"] = stage
    if stage != STAGE:
        info["reason"] = "Q is not explicitly trained on this global-route reward; no cross-fidelity Q error"
        return info
    path = resolve_from(task["q_source_manifest"], base)
    if sha256_file(path) != task.get("q_source_manifest_sha256"):
        raise ValueError("GRT Q source manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("reward_stage") != STAGE:
        raise ValueError("Q source manifest is not a GRT-labeled dataset")
    scale = numeric(task.get("q_reward_scale_train_only"), "q_reward_scale_train_only", required=True, positive=True)
    if scale != manifest.get("reward_scale_train_only"):
        raise ValueError("Q scale differs from recorded training-only backend normalization")
    # Bind both the scale manifest and actual dataset, not just a claimed stage string.
    dataset = path.with_name("dataset.npz")
    if sha256_file(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("Q source dataset changed since its manifest")
    info.update(calibration_allowed=True, measured_reward=-measured_rwl / scale,
                reward_scale_train_only=scale, source_manifest=str(path), source_manifest_sha256=sha256_file(path))
    return info


def validate_task(task, base, expected_stage=STAGE):
    method, index = task_key(task)
    configuration, seed = method_identity(method, task.get("training_seed"))
    if task.get("status", "ready") not in ("ready", "success", "completed"):
        raise ValueError(f"Task explicitly failed: {task.get('status')}")
    if task.get("legal") is False:
        raise ValueError("Task reports illegal executed layout")
    action = executed_action(task, base)
    coordinate_legality = None
    if "normalized_sizes" in task:
        coordinate_legality = legality(action, np.asarray(task["normalized_sizes"]).reshape(-1, 2))
        if not coordinate_legality["legal"]:
            raise ValueError("Independent normalized-coordinate legality check failed")
    def_path = resolve_from(task["def_path"], base)
    def_hash = sha256_file(def_path)
    if def_hash != task.get("def_sha256"):
        raise ValueError("Exported DEF SHA mismatch")
    audit_path = resolve_from(task["def_audit_path"], base) if "def_audit_path" in task else def_path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if "def_audit_sha256" in task and sha256_file(audit_path) != task["def_audit_sha256"]:
        raise ValueError("DEF geometry audit SHA mismatch")
    if audit.get("output_sha256") != def_hash or audit.get("legal_no_halo") is not True:
        raise ValueError("DEF lacks a successful matching rounded-geometry audit")
    metrics_path = resolve_from(task["metrics_path"], base)
    manifest_path = resolve_from(task["manifest_path"], base) if "manifest_path" in task else metrics_path.with_name("manifest.json")
    for key, path in [("metrics_sha256", metrics_path), ("manifest_sha256", manifest_path)]:
        if key in task and sha256_file(path) != task[key]:
            raise ValueError("Reused backend measurement changed after candidate export")
    metrics, manifest = json.loads(metrics_path.read_text()), json.loads(manifest_path.read_text())
    if manifest.get("success") is not True:
        raise ValueError("Backend did not succeed; preserve and retry the requested run")
    if metrics.get("stage") != expected_stage or manifest.get("stage") != expected_stage:
        raise ValueError("Wrong backend stage; global route is not final Table 2")
    if metrics.get("not_final_table2") is not True or manifest.get("not_final_table2") is not True:
        raise ValueError("GRT labels must explicitly identify not_final_table2")
    if manifest.get("def_sha256") != def_hash or sha256_file(resolve_from(manifest["def_path"], manifest_path.parent)) != def_hash:
        raise ValueError("Backend did not use the bound executed DEF")
    if task.get("design") != manifest.get("design"):
        raise ValueError("Task and backend designs differ")
    values = {key: numeric(metrics.get(key), key, required=key in ("rWL_um", "WNS_ns", "TNS_ns"), positive=key == "rWL_um")
              for key in ["rWL_um", "WNS_ns", "TNS_ns", "Power_W", "StdCellArea_um2"]}
    fixed_baseline_provenance = None
    if task.get("role") == "fixed_training_baseline":
        reference_path = resolve_from(task["best_train_baseline_path"], base)
        if sha256_file(reference_path) != task.get("best_train_baseline_sha256"):
            raise ValueError("Fixed training baseline record changed")
        reference = json.loads(reference_path.read_text())
        bound = reference["backend_record"]
        if reference.get("selection_split") != "train_only" or reference.get("parent_noise_index") != index:
            raise ValueError("Fixed baseline is not the training-selected source index")
        if reference.get("dataset_sha256") != task.get("dataset_sha256"):
            raise ValueError("Fixed baseline came from another training dataset")
        if reference.get("action_sha256") != task["action_sha256"] or bound.get("def_sha256") != def_hash or bound.get("metrics_sha256") != sha256_file(metrics_path):
            raise ValueError("Fixed baseline must reuse its exact executed action/DEF/backend measurement")
        fixed_baseline_provenance = {"path": str(reference_path), "sha256": sha256_file(reference_path)}
    consistency = {key: manifest.get(key) for key in ["design", "openroad_sha256", "design_config_sha256", "flow_script_patch"]}
    if not all(consistency.values()):
        raise ValueError("Incomplete backend implementation provenance")
    record = {"method": method, "configuration": configuration, "training_seed": seed, "index": index,
              "role": task.get("role", "candidate"), "reused_backend_measurement": task.get("reused_backend_measurement", False),
              "noise_conditioning": task.get("noise_conditioning", task.get("role") not in ("fixed_control", "fixed_training_baseline")),
              "split": task["split"], "task_id": task.get("task_id"), "run_id": task.get("run_id"),
              "status": "success", "values": values, "q": q_information(task, base, values["rWL_um"]),
              "coordinate_legality": coordinate_legality, "rounded_def_legal_no_halo": True,
              "fixed_training_baseline_provenance": fixed_baseline_provenance,
              "action_sha256": task["action_sha256"], "def_sha256": def_hash,
              "def_audit_sha256": sha256_file(audit_path), "metrics_path": str(metrics_path),
              "metrics_sha256": sha256_file(metrics_path), "backend_manifest_sha256": sha256_file(manifest_path),
              "backend_consistency": consistency, "backend_metrics": metrics, "backend_manifest": manifest}
    json.dumps(record, allow_nan=False)
    return record


def is_noise_conditioned(row):
    return row["role"] not in ("fixed_control", "fixed_training_baseline") and row.get("noise_conditioning", True)


def quality_summary(records, reference_method="teacher_relegalized"):
    if not isinstance(reference_method, str) or not reference_method:
        raise ValueError("reference_method must name an explicit method")
    by_method = {}
    for method in sorted({row["method"] for row in records}):
        rows = sorted((row for row in records if row["method"] == method), key=lambda row: row["index"])
        conditioned = [is_noise_conditioned(row) for row in rows]
        if len(set(conditioned)) != 1 or (not conditioned[0] and len(rows) != 1):
            raise ValueError("Fixed methods must have exactly one measured value, not copied noise replicates")
        result = {"configuration": rows[0]["configuration"], "training_seed": rows[0]["training_seed"],
                  "indices": [row["index"] for row in rows], "count": len(rows),
                  "noise_conditioning": conditioned[0], "independent_fixed_measurements": 0 if conditioned[0] else 1}
        for metric in ["rWL_um", "WNS_ns", "TNS_ns", "Power_W", "StdCellArea_um2"]:
            values = [row["values"][metric] for row in rows]
            result[metric] = summarize(values) if all(value is not None for value in values) else {
                "count": len(values), "available_count": sum(value is not None for value in values),
                "summary": None, "reason": "Missing optional values; no success-only average"}
        if all(row["q"]["calibration_allowed"] for row in rows):
            result["q_calibration"] = critic_diagnostics(np.asarray([row["q"]["prediction"] for row in rows]),
                                                        np.asarray([row["q"]["measured_reward"] for row in rows]))
        else:
            result["q_calibration"] = {"status": "not_available_for_all_rows", "reason": "Requires every Q to use verified GRT training normalization"}
        by_method[method] = result
    by_configuration = {}
    for config in sorted({row["configuration"] for row in records}):
        methods = {method: values for method, values in by_method.items() if values["configuration"] == config}
        indices = [values["indices"] for values in methods.values()]
        if any(value != indices[0] for value in indices):
            raise ValueError("Training seeds have unequal paired noise sets")
        seeds = [values["training_seed"] for values in methods.values()]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate training seed within a configuration")
        by_configuration[config] = {"methods": sorted(methods), "training_seeds": seeds,
                                   "noise_conditioning": all(value["noise_conditioning"] for value in methods.values()),
                                   "noise_count_per_seed": len(indices[0]) if all(value["noise_conditioning"] for value in methods.values()) else 0,
                                   "rwl_mean_across_seed_means": summarize([values["rWL_um"]["mean"] for values in methods.values()]),
                                   "note": "Seed means remain separate; paired noises are not independent training replicates"
                                   if all(value["noise_conditioning"] for value in methods.values()) else
                                   "One fixed measurement; index is bookkeeping, with no random-noise or training-seed replication"}
    pairs = []
    lookup = {(row["method"], row["index"]): row for row in records}
    for row in records:
        if not is_noise_conditioned(row):
            continue
        baselines = [reference_method]
        if row["training_seed"] is not None and row["configuration"].startswith(("fql_alpha_", "analytic_hpwl_alpha_")):
            baselines.append(f"seed_{row['training_seed']}/distill_only")
        for baseline in baselines:
            reference = lookup.get((baseline, row["index"]))
            if reference is None or row["method"] == baseline or not is_noise_conditioned(reference):
                continue
            pairs.append({"method": row["method"], "baseline": baseline, "training_seed": row["training_seed"],
                          "index": row["index"],
                          "rwl_reduction_percent": 100 * (reference["values"]["rWL_um"] - row["values"]["rWL_um"]) / reference["values"]["rWL_um"],
                          "wns_change_ns": row["values"]["WNS_ns"] - reference["values"]["WNS_ns"],
                          "tns_change_ns": row["values"]["TNS_ns"] - reference["values"]["TNS_ns"]})
    if sum(row["role"] == "fixed_training_baseline" for row in records) > 1:
        raise ValueError("Best-observed-TRAIN baseline must appear once, not once per noise")
    fixed = [row for row in records if not is_noise_conditioned(row)]
    fixed_comparisons = []
    for reference in fixed:
        for method, values in by_method.items():
            if not values["noise_conditioning"]:
                continue
            ref = reference["values"]["rWL_um"]
            fixed_comparisons.append({"method": method, "training_seed": values["training_seed"],
                                      "fixed_reference_method": reference["method"], "fixed_reference_role": reference["role"],
                                      "fixed_reference_rWL_um": ref, "fixed_reference_measurement_count": 1,
                                      "method_noise_count": values["count"], "method_rWL_um": values["rWL_um"],
                                      "method_mean_rwl_reduction_percent": 100 * (ref - values["rWL_um"]["mean"]) / ref,
                                      "note": "One fixed measurement versus a noise-conditioned method distribution; no paired-noise baseline replicates"})
    paired_summary = []
    for method, baseline in sorted({(row["method"], row["baseline"]) for row in pairs}):
        rows = [row for row in pairs if row["method"] == method and row["baseline"] == baseline]
        paired_summary.append({"method": method, "baseline": baseline, "count": len(rows),
                               **{metric: summarize([row[metric] for row in rows]) for metric in
                                  ["rwl_reduction_percent", "wns_change_ns", "tns_change_ns"]}})
    return {"reference_method": reference_method, "by_method": by_method, "by_configuration": by_configuration,
            "paired_records": pairs, "paired_summary": paired_summary,
            "fixed_measurements": [{"method": row["method"], "role": row["role"], "values": row["values"],
                                    "measurement_count": 1, "noise_conditioning": False} for row in fixed],
            "versus_fixed_references": fixed_comparisons,
            "versus_fixed_training_baseline": [row for row in fixed_comparisons if row["fixed_reference_role"] == "fixed_training_baseline"],
            "directions": "Positive rWL reduction and positive WNS/TNS changes indicate improvement; report timing separately from wirelength"}


def aggregate(tasks_path, output, plan_path=None):
    tasks_path, output = Path(tasks_path).resolve(), Path(output).resolve()
    plan_path = Path(plan_path).resolve() if plan_path is not None else tasks_path.with_name("selection_plan.json")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "validating", "stage": STAGE, "not_final_table2": True,
              "tasks_path": str(tasks_path), "plan_path": str(plan_path),
              "failures": [], "records": [], "quality": None}
    try:
        plan = json.loads(plan_path.read_text())
        expected = plan_keys(plan)
        report.update(tasks_sha256=sha256_file(tasks_path), plan_sha256=sha256_file(plan_path), plan=plan,
                      phase=plan.get("phase", "exploratory"), expected_count=len(expected))
        if report["phase"] not in ("validation", "heldout", "exploratory"):
            raise ValueError("Unknown evaluation phase")
        for name, binding in plan.get("source_bindings", {}).items():
            if sha256_file(resolve_from(binding["path"], plan_path.parent)) != binding["sha256"]:
                raise ValueError(f"Frozen candidate source changed: {name}")
        if "selection_path" in plan and sha256_file(resolve_from(plan["selection_path"], plan_path.parent)) != plan.get("selection_sha256"):
            raise ValueError("Upstream method selection changed after candidate plan was frozen")
        if report["phase"] == "heldout":
            if "selection_path" not in plan:
                raise ValueError("Heldout requires a bound sealed GRT validation selection; use exploratory for proxy transfer")
            selection = json.loads(resolve_from(plan["selection_path"], plan_path.parent).read_text())
            if selection.get("kind") != "sealed_GRT_validation_selection" or selection.get("selection_stage") != STAGE:
                raise ValueError("Heldout selection is not a sealed GRT validation decision")
            if sha256_file(selection["source_report"]) != selection.get("source_report_sha256"):
                raise ValueError("Validation source report changed after selection")
            if expected != plan_keys(selection["heldout_plan"]):
                raise ValueError("Heldout methods/noises differ from the sealed plan")
        seen = set()
        for line_number, line in enumerate(tasks_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            entry = None
            try:
                entry = json.loads(line)
                key = task_key(entry)
                if key in seen:
                    raise ValueError("Duplicate method/index task")
                seen.add(key)
                if key not in expected:
                    raise ValueError("Task not in frozen candidate plan")
                if "dataset_sha256" in plan and entry.get("dataset_sha256") != plan["dataset_sha256"]:
                    raise ValueError("Task belongs to another dataset")
                fixed_training = entry.get("role") == "fixed_training_baseline"
                if fixed_training and (entry.get("split") != 0 or entry.get("method") != "best_observed_train"):
                    raise ValueError("Fixed training baseline must come from train split and be identified explicitly")
                if report["phase"] == "validation" and not fixed_training and entry.get("split") != 1:
                    raise ValueError("Validation tasks must use split 1 only")
                if report["phase"] == "heldout" and not fixed_training and entry.get("split") != 2:
                    raise ValueError("Heldout tasks must use split 2 only")
                report["records"].append(validate_task(entry, tasks_path.parent))
            except Exception as exc:
                report["failures"].append({"mapping_line": line_number, "task": entry,
                                            "error_type": type(exc).__name__, "message": str(exc)})
        for method, index in sorted(expected - seen):
            report["failures"].append({"method": method, "index": index, "message": "Predeclared task missing from JSONL"})
        records = report["records"]
        if "task_ids" in plan:
            actual_ids = [row.get("task_id") for row in records]
            if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(plan["task_ids"]):
                report["failures"].append({"message": "Task IDs do not match frozen plan exactly"})
        if records:
            consistency = records[0]["backend_consistency"]
            for row in records[1:]:
                if row["backend_consistency"] != consistency:
                    report["failures"].append({"method": row["method"], "index": row["index"], "message": "Mixed backend provenance across candidates"})
            report["backend_consistency"] = consistency
            if "backend_consistency" in plan and consistency != plan["backend_consistency"]:
                report["failures"].append({"message": "Backend implementation differs from frozen candidate plan"})
            if report["phase"] == "heldout" and consistency != selection["backend_consistency"]:
                report["failures"].append({"message": "Backend implementation differs from validation selection"})
        report["successful_count"] = len(records)
        if report["failures"] or len(records) != len(expected):
            raise ValueError("Incomplete or invalid backend comparison; no success-only aggregate generated")
        report["quality"] = quality_summary(records, plan.get("reference_method", "teacher_relegalized"))
        report.update(status="completed", selection_allowed=report["phase"] == "validation",
                      notes=["GRT is an intermediate backend stage, not final detailed-route Table 2",
                             "No Q calibration against rWL unless GRT reward stage and training-only scale are verified",
                             "Method/seed and paired-noise records remain separate; no seed cherry-picking",
                             "Exploratory cross-fidelity results cannot be relabeled as GRT validation selection"])
        write_json(output / "report.json", report)
        return report
    except Exception as exc:
        report.update(status="failed", selection_allowed=False,
                      failure={"type": type(exc).__name__, "message": str(exc)})
        write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use Slurm for numeric aggregation")
    report = aggregate(args.tasks, args.output, args.plan)
    print(json.dumps({"status": report["status"], "phase": report["phase"],
                      "count": report["successful_count"], "output": str(Path(args.output).resolve())}), flush=True)


if __name__ == "__main__":
    main()
