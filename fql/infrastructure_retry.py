"""Independent aggregation checks for an explicit successful GRT infra retry."""
import hashlib
import json
from pathlib import Path

from fql.relabel_backend import STAGE, numeric, resolve_from, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def load_retry_map(path, tasks_path):
    if path is None:
        return None
    path, tasks_path = Path(path).resolve(), Path(tasks_path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("kind") != "same_input_infrastructure_retries":
        raise ValueError("Unknown infrastructure retry-map schema")
    if resolve_from(data["source_tasks_path"], path.parent) != tasks_path or data["source_tasks_sha256"] != sha256_file(tasks_path):
        raise ValueError("Retry map does not bind the unchanged full tasks file")
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    requested = [row.get("run_id") for row in rows]
    if not data.get("retries") or any(requested.count(key) != 1 for key in data["retries"]):
        raise ValueError("Retry map must identify unique originally requested run IDs")
    return {"path": str(path), "sha256": sha256_file(path), "data": data}


def current_flow_files(manifest):
    """Hash actual available inputs, rather than accepting inherited hash strings."""
    backend = ROOT / "backend"
    flow = backend / "thirdparty/OpenROAD-flow-scripts"
    components = manifest["physical_flow_components"]
    paths = {
        "tool_image_sha256": (Path("/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif"), manifest["tool_image_sha256"]),
        "author_flow_archive_sha256": (backend / "assets/OpenROAD-flow-scripts.tar.gz", components["author_flow_archive_sha256"]),
        "design_config_sha256": (Path(manifest["design_config"]), components["design_config_sha256"]),
        "platform_config_sha256": (flow / "platforms/nangate45/config.mk", components["platform_config_sha256"]),
        "patched_write_ref_sdc_sha256": (flow / "scripts/write_ref_sdc.tcl", components["patched_write_ref_sdc_sha256"]),
        "global_route_optimization_sha256": (flow / "scripts/global_route.tcl", components["global_route_optimization_sha256"]),
    }
    proofs = {}
    for key, (path, expected) in paths.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Actual retry flow input hash differs: {key}")
        proofs[key] = {"path": str(path.resolve()), "sha256": actual}
    if (components["openroad_binary_sha256"] != manifest["openroad_sha256"]
            or components["design_config_sha256"] != manifest["design_config_sha256"]
            or components["platform_config_sha256"] != manifest["platform_config_sha256"]):
        raise ValueError("Runtime component hashes disagree with manifest fields")
    return proofs


def resolve_successful_retry(task, tasks_path, retry_map):
    if retry_map is None or task.get("run_id") not in retry_map["data"]["retries"]:
        return task, None
    if task.get("reused_backend_measurement") or task.get("role") == "fixed_training_baseline":
        raise ValueError("A reused successful reference is not an interrupted candidate")
    base = Path(tasks_path).resolve().parent
    item = retry_map["data"]["retries"][task["run_id"]]
    old_metrics = resolve_from(task["metrics_path"], base)
    old_path = old_metrics.with_name("manifest.json")
    event_path = old_metrics.with_name("preemption.json")
    if old_path.parent.name != task["run_id"]:
        raise ValueError("Original requested run ID does not match its result directory")
    for name, path in [("source_manifest", old_path), ("preemption", event_path)]:
        if resolve_from(item[name + "_path"], base) != path or item[name + "_sha256"] != sha256_file(path):
            raise ValueError(f"Retry original {name} path/hash mismatch")
    old, event = json.loads(old_path.read_text()), json.loads(event_path.read_text())
    if old.get("success") is not False or old.get("time_end") or old_metrics.exists() or old_metrics.with_name("physical_failure.json").exists():
        raise ValueError("Retry requires an incomplete original infrastructure attempt without metrics/physical failure")
    if event.get("kind") != "slurm_preemption" or event.get("state") != "PREEMPTED" or event.get("original_manifest_unmodified") is not True:
        raise ValueError("Original attempt lacks explicit preserved preemption evidence")
    if (resolve_from(event["manifest_path"], event_path.parent) != old_path
            or event["manifest_sha256"] != sha256_file(old_path)
            or str(event["slurm_job_id"]) != str(old["slurm_job_id"])):
        raise ValueError("Original preemption manifest/job binding differs")
    raw_rows = [line.split("|") for line in event["sacct_record"].splitlines() if line.strip()]
    if not any(len(row) >= 8 and row[0] == str(old["slurm_job_id"]) and row[2] == "PREEMPTED" for row in raw_rows):
        raise ValueError("Original raw Slurm record is not PREEMPTED for this job")
    log_path = resolve_from(event["last_log_path"], event_path.parent)
    if log_path.parent != old_path.parent or sha256_file(log_path) != event["last_log_sha256"]:
        raise ValueError("Original interrupted log changed")
    new_path = resolve_from(item["retry_manifest_path"], base)
    metrics = resolve_from(item["retry_metrics_path"], base)
    if (new_path.name != "manifest.json" or new_path.parent == old_path.parent
            or new_path.parent.name != item["retry_run_id"] or metrics != new_path.with_name("metrics.json")):
        raise ValueError("Retry run ID/manifest/metrics location differs")
    if sha256_file(new_path) != item["retry_manifest_sha256"] or sha256_file(metrics) != item["retry_metrics_sha256"]:
        raise ValueError("Frozen retry manifest/metrics changed")
    new = json.loads(new_path.read_text())
    if new.get("success") is not True or str(new.get("slurm_job_id")) != str(item["retry_slurm_job_id"]) or new_path.with_name("physical_failure.json").exists():
        raise ValueError("Only the declared successful retry is accepted")
    if numeric(new.get("time_end"), "retry time_end", required=True) <= numeric(new.get("time_start"), "retry time_start", required=True):
        raise ValueError("Retry lacks a completed interval")
    completed = [line.split("|") for line in item["retry_sacct_record"].splitlines() if line.strip()]
    if not any(len(row) >= 3 and row[:3] == [str(new["slurm_job_id"]), "COMPLETED", "0:0"] for row in completed):
        raise ValueError("Retry raw Slurm record is not successful for the declared job")
    for key in ["design", "def_sha256", "design_config_sha256", "openroad_sha256", "tool_image_sha256",
                "platform_config_sha256", "physical_flow_revision", "physical_flow_components", "flow_script_patch",
                "stage", "not_final_table2", "num_cores"]:
        if key not in old or key not in new or old[key] != new[key]:
            raise ValueError(f"Retry changes requested input or physical flow: {key}")
    if new["stage"] != STAGE or new["not_final_table2"] is not True or new["design"] != task["design"]:
        raise ValueError("Retry is not the requested fixed GRT evaluator/design")
    canonical = hashlib.sha256(json.dumps(new["physical_flow_components"], sort_keys=True).encode()).hexdigest()
    if canonical != new["physical_flow_revision"]:
        raise ValueError("Retry physical-flow revision is inconsistent")
    def_path = resolve_from(task["def_path"], base)
    if (any(resolve_from(m["def_path"], base) != def_path for m in [old, new, event])
            or any(m["def_sha256"] != task["def_sha256"] for m in [old, new, event])
            or sha256_file(def_path) != task["def_sha256"]):
        raise ValueError("Original/requested/retried actual DEF differs")
    proof = {"requested_run_id": task["run_id"], "executed_run_id": item["retry_run_id"],
             "retry_map_path": retry_map["path"], "retry_map_sha256": retry_map["sha256"],
             "original_manifest_path": str(old_path), "original_manifest_sha256": sha256_file(old_path),
             "original_manifest": old, "preemption_path": str(event_path), "preemption_sha256": sha256_file(event_path),
             "preemption": event, "original_metrics_path": str(old_metrics),
             "retry_manifest_path": str(new_path), "retry_manifest_sha256": sha256_file(new_path),
             "retry_metrics_path": str(metrics), "retry_metrics_sha256": sha256_file(metrics),
             "original_slurm_job_id": old["slurm_job_id"], "retry_slurm_job_id": new["slurm_job_id"],
             "current_actual_flow_file_checks": current_flow_files(new),
             "file_check_scope": "Actual files verified at aggregation; not a retroactive startup snapshot",
             "same_logical_candidate": True, "extra_candidate_denominator": 0}
    effective = dict(task, metrics_path=str(metrics), manifest_path=str(new_path),
                     metrics_sha256=sha256_file(metrics), manifest_sha256=sha256_file(new_path))
    return effective, proof
