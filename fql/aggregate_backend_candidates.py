"""Aggregate actual GRT results without mixing macro proxies or hiding failures.

Every method/index in the frozen plan must be present exactly once. Strict mode
rejects failures. An explicit exploratory-only mode admits tightly verified
terminal physical failures, preserving denominators and suppressing incomplete
method/configuration means. This module never launches backends or training.
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


def validated_input(task, base):
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
    record = {"method": method, "configuration": configuration, "training_seed": seed, "index": index,
              "role": task.get("role", "candidate"), "reused_backend_measurement": task.get("reused_backend_measurement", False),
              "noise_conditioning": task.get("noise_conditioning", task.get("role") not in ("fixed_control", "fixed_training_baseline")),
              "split": task["split"], "task_id": task.get("task_id"), "run_id": task.get("run_id"),
              "coordinate_legality": coordinate_legality, "rounded_def_legal_no_halo": True,
              "action_sha256": task["action_sha256"], "def_sha256": def_hash, "def_path": str(def_path),
              "def_audit_sha256": sha256_file(audit_path), "metrics_path": str(metrics_path),
              "backend_manifest_path": str(manifest_path)}
    return record, def_path, audit_path, metrics_path, manifest_path


def backend_identity(task, manifest, manifest_path, def_hash, expected_stage):
    if manifest.get("stage") != expected_stage or manifest.get("not_final_table2") is not True:
        raise ValueError("Expected explicitly intermediate global-route backend")
    if manifest.get("def_sha256") != def_hash or sha256_file(resolve_from(manifest["def_path"], manifest_path.parent)) != def_hash:
        raise ValueError("Backend did not use the bound executed DEF")
    if task.get("design") != manifest.get("design"):
        raise ValueError("Task and backend designs differ")
    consistency = {key: manifest.get(key) for key in ["design", "openroad_sha256", "design_config_sha256", "flow_script_patch"]}
    if not all(consistency.values()):
        raise ValueError("Incomplete backend implementation provenance")
    return consistency


def validate_task(task, base, expected_stage=STAGE):
    record, def_path, audit_path, metrics_path, manifest_path = validated_input(task, base)
    method, index, def_hash = record["method"], record["index"], record["def_sha256"]
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
    consistency = backend_identity(task, manifest, manifest_path, def_hash, expected_stage)
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
    record.update(status="success", values=values, q=q_information(task, base, values["rWL_um"]),
                  fixed_training_baseline_provenance=fixed_baseline_provenance,
                  metrics_sha256=sha256_file(metrics_path), backend_manifest_sha256=sha256_file(manifest_path),
                  backend_consistency=consistency, backend_metrics=metrics, backend_manifest=manifest)
    json.dumps(record, allow_nan=False)
    return record


def validate_physical_failure(task, base, tasks_path, expected_stage=STAGE):
    """Only the independently diagnosed CTS RSZ-0060 outcome is authorized."""
    record, def_path, audit_path, metrics_path, manifest_path = validated_input(task, base)
    if task.get("role") == "fixed_training_baseline":
        raise ValueError("A reused successful best-TRAIN label cannot become a new physical-failure observation")
    sidecar_path = metrics_path.with_name("physical_failure.json")
    sidecar, manifest = json.loads(sidecar_path.read_text()), json.loads(manifest_path.read_text())
    if sidecar.get("schema_version") != 1 or sidecar.get("status") != "physical_failure":
        raise ValueError("Unknown physical-failure sidecar schema/status")
    expected = {"failure_class": "cts_hold_repair_max_buffer_count", "error_code": "RSZ-0060",
                "error_line": "[ERROR RSZ-0060] Max buffer count reached.",
                "stage": "floorplan_place_cts", "substage": "clock_tree_synthesis_hold_repair",
                "make_exit_code": 2, "openroad_exit_code": 1, "slurm_state": "FAILED",
                "original_manifest_unchanged": True, "grt_metrics_available": False,
                "not_parser_failure": True, "not_preemption": True}
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise ValueError("Unrecognized or nonterminal physical failure; parser/preemption failures are not admissible")
    if manifest.get("success") is not False or metrics_path.exists():
        raise ValueError("Physical failure requires unsuccessful original manifest and no GRT metrics")
    if "floorplan_place_cts exited 2" not in str(manifest.get("failure", "")):
        raise ValueError("Original manifest does not identify the failed CTS subprocess")
    start = numeric(manifest.get("time_start"), "time_start", required=True, positive=True)
    end = numeric(manifest.get("time_end"), "time_end", required=True, positive=True)
    if end < start or sidecar.get("time_end") != end or not sidecar.get("slurm_end"):
        raise ValueError("Missing/mismatched terminal timing evidence")
    if not re.fullmatch(r"[1-9]\d*:0", str(sidecar.get("slurm_exit_code", ""))):
        raise ValueError("Sidecar lacks a normal nonzero process exit (signals/preemption rejected)")
    if sidecar.get("slurm_job_id") != manifest.get("slurm_job_id") or sidecar.get("run_id") != task.get("run_id"):
        raise ValueError("Sidecar refers to another backend execution")
    if sidecar.get("physical_flow_revision") != manifest.get("physical_flow_revision") or not manifest.get("physical_flow_revision"):
        raise ValueError("Missing/mismatched physical-flow identity")
    action_path = resolve_from(task["action_path"], base)
    bindings = {"manifest": manifest_path, "def": def_path, "log": metrics_path.with_name("floorplan_place_cts.log"),
                "task_manifest": Path(tasks_path).resolve(), "def_audit": audit_path}
    for name, path in bindings.items():
        if resolve_from(sidecar[name + "_path"], sidecar_path.parent) != path.resolve() or sha256_file(path) != sidecar.get(name + "_sha256"):
            raise ValueError(f"Physical failure {name} path/hash binding mismatch")
    if resolve_from(sidecar["run_path"], sidecar_path.parent) != metrics_path.parent.resolve():
        raise ValueError("Sidecar run directory mismatch")
    if resolve_from(sidecar["action_path"], sidecar_path.parent) != action_path or sha256_file(action_path) != sidecar.get("action_file_sha256"):
        raise ValueError("Physical failure action file binding mismatch")
    for key in ["task_id", "method", "index", "dataset_sha256", "action_sha256"]:
        if key not in task or sidecar.get(key) != task[key]:
            raise ValueError(f"Physical failure task/action identity mismatch: {key}")
    if sidecar.get("action_key") != task.get("action_key", "action") or sidecar.get("action_row") != task["action_row"]:
        raise ValueError("Physical failure action locator mismatch")
    lines = bindings["log"].read_text(errors="replace").splitlines()
    line_number = sidecar.get("error_line_number")
    if isinstance(line_number, bool) or not isinstance(line_number, int) or not 1 <= line_number <= len(lines) or lines[line_number - 1].strip() != expected["error_line"]:
        raise ValueError("Classified error is absent at its bound raw log location")
    if not any("do-4_1_cts] Error 1" in line for line in lines) or not any("do-cts] Error 2" in line for line in lines):
        raise ValueError("Raw log does not prove terminal CTS make failure")
    consistency = backend_identity(task, manifest, manifest_path, record["def_sha256"], expected_stage)
    record.update(status="physical_failure", values={key: None for key in ["rWL_um", "WNS_ns", "TNS_ns", "Power_W", "StdCellArea_um2"]},
                  q={"prediction": task.get("predicted_q"), "calibration_allowed": False,
                     "reason": "No terminal GRT reward exists for this physical failure"},
                  metrics_sha256=None, backend_manifest_sha256=sha256_file(manifest_path),
                  backend_consistency=consistency, backend_metrics=None, backend_manifest=manifest,
                  physical_failure_path=str(sidecar_path), physical_failure_sha256=sha256_file(sidecar_path),
                  physical_failure=sidecar)
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


def quality_with_outcomes(records, reference_method):
    """Keep failure denominators, without success-only method or seed means."""
    by_method = {}
    for method in sorted({row["method"] for row in records}):
        rows = sorted((row for row in records if row["method"] == method), key=lambda row: row["index"])
        conditioned = [is_noise_conditioned(row) for row in rows]
        if len(set(conditioned)) != 1 or (not conditioned[0] and len(rows) != 1):
            raise ValueError("Fixed methods must have exactly one outcome, not copied noise replicates")
        success = sum(row["status"] == "success" for row in rows)
        physical = sum(row["status"] == "physical_failure" for row in rows)
        by_method[method] = {"configuration": rows[0]["configuration"], "training_seed": rows[0]["training_seed"],
                             "indices": [row["index"] for row in rows], "count": len(rows), "total_count": len(rows),
                             "success_count": success, "physical_failure_count": physical,
                             "physical_failure_fraction": physical / len(rows), "comparison_complete": success == len(rows),
                             "noise_conditioning": conditioned[0], "independent_fixed_measurements": 0 if conditioned[0] else success}
    complete_methods = {method for method, counts in by_method.items() if counts["comparison_complete"]}
    quality = quality_summary([row for row in records if row["method"] in complete_methods], reference_method)
    for method, counts in by_method.items():
        if method in complete_methods:
            quality["by_method"][method].update(counts)
        else:
            quality["by_method"][method] = {**counts, **{key: None for key in
                ["rWL_um", "WNS_ns", "TNS_ns", "Power_W", "StdCellArea_um2", "q_calibration"]},
                "reason": "Physical failure in the frozen method budget; successful-only metric means are prohibited"}
    for config in sorted({row["configuration"] for row in records}):
        methods = {name: counts for name, counts in by_method.items() if counts["configuration"] == config}
        indices = [value["indices"] for value in methods.values()]
        if any(index != indices[0] for index in indices):
            raise ValueError("Training seeds have unequal requested noise sets")
        seeds = [value["training_seed"] for value in methods.values()]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate training seed within a configuration")
        complete = all(value["comparison_complete"] for value in methods.values())
        total = sum(value["total_count"] for value in methods.values())
        physical = sum(value["physical_failure_count"] for value in methods.values())
        counts = {"methods": sorted(methods), "training_seeds": seeds, "total_count": total,
                  "success_count": sum(value["success_count"] for value in methods.values()),
                  "physical_failure_count": physical, "physical_failure_fraction": physical / total,
                  "comparison_complete": complete,
                  "noise_conditioning": all(value["noise_conditioning"] for value in methods.values()),
                  "noise_count_per_seed": len(indices[0]) if all(value["noise_conditioning"] for value in methods.values()) else 0}
        if complete:
            quality["by_configuration"][config].update(counts)
        else:
            quality["by_configuration"][config] = {**counts, "rwl_mean_across_seed_means": None,
                "note": "At least one seed failed a requested physical evaluation; no successful-seed-only configuration mean"}
    quality["complete_methods_for_numerical_summaries"] = sorted(complete_methods)
    quality["numerical_scope"] = "Numerical means/pairs only for complete methods; configuration means require every predeclared seed complete"
    return quality


def aggregate(tasks_path, output, plan_path=None, allow_exploratory_physical_failures=False):
    tasks_path, output = Path(tasks_path).resolve(), Path(output).resolve()
    plan_path = Path(plan_path).resolve() if plan_path is not None else tasks_path.with_name("selection_plan.json")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "validating", "stage": STAGE, "not_final_table2": True,
              "tasks_path": str(tasks_path), "plan_path": str(plan_path),
              "failures": [], "physical_failures": [], "records": [], "quality": None,
              "allow_exploratory_physical_failures": allow_exploratory_physical_failures,
              "outcomes_complete": False, "comparison_complete": False}
    try:
        plan = json.loads(plan_path.read_text())
        expected = plan_keys(plan)
        report.update(tasks_sha256=sha256_file(tasks_path), plan_sha256=sha256_file(plan_path), plan=plan,
                      phase=plan.get("phase", "exploratory"), expected_count=len(expected))
        if report["phase"] not in ("validation", "heldout", "exploratory"):
            raise ValueError("Unknown evaluation phase")
        if allow_exploratory_physical_failures and report["phase"] != "exploratory":
            raise ValueError("Physical-failure reporting mode is permitted only for exploratory comparisons")
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
                sidecar = resolve_from(entry["metrics_path"], tasks_path.parent).with_name("physical_failure.json")
                if allow_exploratory_physical_failures and sidecar.exists():
                    record = validate_physical_failure(entry, tasks_path.parent, tasks_path)
                    report["physical_failures"].append(record)
                else:
                    record = validate_task(entry, tasks_path.parent)
                record.update(task_manifest_path=str(tasks_path), task_mapping_line=line_number)
                report["records"].append(record)
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
        report["successful_count"] = sum(row["status"] == "success" for row in records)
        report["physical_failure_count"] = len(report["physical_failures"])
        report["total_count"] = len(expected)
        report["verified_outcome_count"] = len(records)
        if report["failures"] or len(records) != len(expected):
            raise ValueError("Incomplete or invalid backend comparison; no success-only aggregate generated")
        report["quality"] = quality_with_outcomes(records, plan.get("reference_method", "teacher_relegalized"))
        report.update(status="completed_with_physical_failures" if report["physical_failures"] else "completed",
                      outcomes_complete=True, comparison_complete=not report["physical_failures"],
                      selection_allowed=report["phase"] == "validation",
                      notes=["GRT is an intermediate backend stage, not final detailed-route Table 2",
                             "No Q calibration against rWL unless GRT reward stage and training-only scale are verified",
                             "Method/seed and paired-noise records remain separate; no seed cherry-picking",
                             "Verified physical failures retain the full requested denominator; incomplete method/configuration numerical means are null",
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
    parser.add_argument("--allow-exploratory-physical-failures", action="store_true",
                        help="Explicit exploratory-only verified CTS physical-failure outcomes; no incomplete-method means")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use Slurm for numeric aggregation")
    report = aggregate(args.tasks, args.output, args.plan, args.allow_exploratory_physical_failures)
    print(json.dumps({"status": report["status"], "phase": report["phase"],
                      "count": report["successful_count"], "output": str(Path(args.output).resolve())}), flush=True)


if __name__ == "__main__":
    main()
