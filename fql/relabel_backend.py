"""Replace selected critic rewards with measured global-route wirelength labels.

No backend runs are launched. Missing or failed requests are errors. An explicit
--allow-unlabeled-subset permits budget-selected, unrequested source rows to be
excluded, but never excuses failure of a requested row. All teacher data stays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


TEACHER_KEYS = {"noise", "teacher_action", "split", "group_id", "teacher_seconds"}
CRITIC_KEYS = {"raw_action", "action", "reward", "critic_split", "family_id", "variant",
               "critic_group_id", "critic_parent_index", "legal", "macro_hpwl"}
GLOBAL_KEYS = {"movable_mask", "fixed_action", "coordinate_shape", "reward_stage"}
STAGE = "global_route_only"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_sha256(action):
    """Canonical row hash: flattened C-order little-endian float32 bytes."""
    return hashlib.sha256(np.asarray(action, dtype="<f4").reshape(-1).tobytes(order="C")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numeric(value, label, required=False, positive=False):
    if value is None:
        if required:
            raise ValueError(f"Missing required metric {label}")
        return None
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid metric: {label}")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Invalid metric {label}: {value!r}")
    return result


def resolve_from(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def validate_record(entry, line_number, labels_path, arrays, source_hash, design, require_row_hashes):
    index = entry.get("row_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("row_index must be an integer")
    if not 0 <= index < len(arrays["reward"]):
        raise ValueError(f"row_index {index} is outside original critic rows")
    if entry.get("status", "success") not in ("success", "completed"):
        raise ValueError(f"Requested backend row has status={entry.get('status')!r}")
    metrics_value = entry.get("metrics_path", entry.get("metrics_json"))
    if not metrics_value:
        raise ValueError("metrics_path is required")
    metrics_path = resolve_from(metrics_value, labels_path.parent)
    manifest_path = resolve_from(entry["manifest_path"], labels_path.parent) if "manifest_path" in entry else metrics_path.with_name("manifest.json")
    metrics = json.loads(metrics_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("success") is not True:
        raise ValueError("Backend manifest does not prove success=true")
    if metrics.get("stage") != STAGE or manifest.get("stage") != STAGE:
        raise ValueError("Expected global_route_only in both metrics and manifest; refusing mixed stages")
    if metrics.get("not_final_table2") is not True or manifest.get("not_final_table2") is not True:
        raise ValueError("Global-route labels must explicitly identify not_final_table2=true")
    if design is not None and manifest.get("design") != design:
        raise ValueError(f"Backend design {manifest.get('design')!r} != source design {design!r}")
    if "legal" in arrays and not bool(arrays["legal"][index]):
        raise ValueError("Source action is independently marked illegal; success of backend cannot erase this")
    actual_action_hash = action_sha256(arrays["action"][index])
    hashes = {"dataset_sha256": source_hash, "action_sha256": actual_action_hash}
    for key, expected in hashes.items():
        if require_row_hashes and key not in entry:
            raise ValueError(f"Strict row binding requires {key}")
        if key in entry and entry[key] != expected:
            raise ValueError(f"{key} mismatch")
    def_path = resolve_from(manifest["def_path"], manifest_path.parent)
    actual_def_hash = sha256_file(def_path)
    if manifest.get("def_sha256") != actual_def_hash:
        raise ValueError("Backend input DEF no longer matches manifest hash")
    if require_row_hashes and "def_sha256" not in entry:
        raise ValueError("Strict row binding requires def_sha256 in requested mapping")
    if "def_sha256" in entry and entry["def_sha256"] != actual_def_hash:
        raise ValueError("Requested DEF hash does not match backend input DEF")
    rwl = numeric(metrics.get("rWL_um"), "rWL_um", required=True, positive=True)
    # Timing is validation data, not part of the optimized reward.
    diagnostic = {key: numeric(metrics.get(key), key, required=key in ("WNS_ns", "TNS_ns"))
                  for key in ["WNS_ns", "TNS_ns", "Power_W", "StdCellArea_um2"]}
    consistency = {}
    for key in ["openroad_sha256", "design_config_sha256", "flow_script_patch"]:
        if not manifest.get(key):
            raise ValueError(f"Missing backend provenance field {key}")
        consistency[key] = manifest[key]
    consistency["design"] = manifest.get("design")
    return {"row_index": index, "mapping_line": line_number, "metrics_path": str(metrics_path),
            "request_record": entry,
            "manifest_path": str(manifest_path), "def_path": str(def_path),
            "metrics_sha256": sha256_file(metrics_path), "manifest_sha256": sha256_file(manifest_path),
            "def_sha256": actual_def_hash, "action_sha256": actual_action_hash,
            "strict_row_hashes_present": all(key in entry for key in ["dataset_sha256", "action_sha256", "def_sha256"]),
            "critic_split": int(arrays["critic_split"][index]), "rWL_um": rwl,
            "diagnostic_metrics": diagnostic, "backend_consistency": consistency,
            "metrics": metrics, "backend_manifest": manifest}


def relabel(dataset, labels, output, allow_unlabeled_subset=False, require_row_hashes=False,
            extra_critic_keys=(), extra_teacher_keys=(), selection_plan=None):
    dataset, labels, output = Path(dataset).resolve(), Path(labels).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "validating", "source_dataset": str(dataset), "labels_jsonl": str(labels),
             "allow_unlabeled_subset": allow_unlabeled_subset, "require_row_hashes": require_row_hashes,
             "errors": [], "successful_requested_rows": [], "excluded_unrequested_rows": []}
    try:
        source_hash = sha256_file(dataset)
        audit["source_dataset_sha256"] = source_hash
        plan_path = Path(selection_plan).resolve() if selection_plan is not None else labels.with_name("selection_plan.json")
        plan_rows = None
        if selection_plan is not None or plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan.get("dataset_sha256") != source_hash:
                raise ValueError("Frozen selection plan has a different source dataset SHA")
            plan_rows = plan.get("selected_rows")
            if not isinstance(plan_rows, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in plan_rows):
                raise ValueError("Selection plan selected_rows must be a list of integers")
            if len(set(plan_rows)) != len(plan_rows):
                raise ValueError("Selection plan contains duplicate row indices")
            plan_rows = set(plan_rows)
            audit.update(selection_plan=str(plan_path), selection_plan_sha256=sha256_file(plan_path))
        with np.load(dataset, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for key in ["noise", "teacher_action", "split", "raw_action", "action", "reward", "critic_split"]:
            if key not in arrays:
                raise ValueError(f"Source dataset missing {key}")
        n, m = len(arrays["noise"]), len(arrays["reward"])
        if "proxy_reward" in arrays or "backend_rwl_um" in arrays:
            raise ValueError("Input is already backend-relabeled; use the original proxy dataset")
        source_manifest_path = dataset.with_name("manifest.json")
        source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}
        design = source_manifest.get("args", {}).get("design")
        teacher_keys, critic_keys = TEACHER_KEYS | set(extra_teacher_keys), CRITIC_KEYS | set(extra_critic_keys)
        if teacher_keys & critic_keys:
            raise ValueError("A field cannot be both teacher and critic indexed")
        for key, value in arrays.items():
            if key in teacher_keys and (value.ndim == 0 or len(value) != n):
                raise ValueError(f"Teacher field {key} length does not match noise")
            if key in critic_keys and (value.ndim == 0 or len(value) != m):
                raise ValueError(f"Critic field {key} length does not match reward")
            if key not in teacher_keys | critic_keys | GLOBAL_KEYS and value.ndim and len(value) in (n, m):
                raise ValueError(f"Unknown row-indexed field {key}; specify --extra-critic-keys or --extra-teacher-keys")
        if not np.isin(arrays["critic_split"], [0, 1, 2]).all():
            raise ValueError("Invalid source critic split labels")
        if "critic_parent_index" in arrays:
            parent = arrays["critic_parent_index"]
            if np.any(parent < 0) or np.any(parent >= n) or np.any(arrays["critic_split"] != arrays["split"][parent]):
                raise ValueError("Source parent noise and critic splits do not agree")
        audit.update(original_teacher_rows=n, original_critic_rows=m)
        seen, accepted = set(), []
        for line_number, line in enumerate(labels.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            entry = None
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("JSONL row must be an object")
                index = entry.get("row_index")
                if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < m:
                    if index in seen:
                        raise ValueError(f"Duplicate row_index {index}")
                    seen.add(index)
                record = validate_record(entry, line_number, labels, arrays, source_hash, design, require_row_hashes)
                # Reject nonfinite values anywhere in retained source JSON before writing a dataset.
                json.dumps(record, allow_nan=False)
                accepted.append(record)
            except Exception as exc:
                audit["errors"].append({"mapping_line": line_number,
                                        "row_index": entry.get("row_index") if isinstance(entry, dict) else None,
                                        "error_type": type(exc).__name__, "message": str(exc)})
        omitted = sorted(set(range(m)) - seen)
        if plan_rows is not None and seen != plan_rows:
            audit["errors"].append({"message": "Request mapping differs from the frozen selection plan",
                                    "planned_but_missing_rows": sorted(plan_rows - seen),
                                    "unplanned_requested_rows": sorted(seen - plan_rows)})
        audit["excluded_unrequested_rows"] = omitted
        audit["successful_requested_rows"] = sorted(record["row_index"] for record in accepted)
        if omitted and not allow_unlabeled_subset:
            audit["errors"].append({"message": "Unlabeled source critic rows; pass --allow-unlabeled-subset only for a budget-selected request subset",
                                    "count": len(omitted)})
        if not accepted:
            audit["errors"].append({"message": "No valid backend labels"})
        if accepted:
            consistent = accepted[0]["backend_consistency"]
            for record in accepted[1:]:
                if record["backend_consistency"] != consistent:
                    audit["errors"].append({"row_index": record["row_index"], "message": "Mixed backend executable/config/patch/design provenance"})
            counts = {str(value): sum(record["critic_split"] == value for record in accepted) for value in (0, 1, 2)}
            audit["retained_critic_split_counts"] = counts
            if any(count == 0 for count in counts.values()):
                audit["errors"].append({"message": "Retained critic train/validation/heldout splits must all be nonempty"})
        if audit["errors"]:
            raise ValueError(f"Backend relabel validation failed with {len(audit['errors'])} error(s); see audit.json")
        accepted.sort(key=lambda record: record["row_index"])
        indices = np.asarray([record["row_index"] for record in accepted], dtype=np.int64)
        rwl = np.asarray([record["rWL_um"] for record in accepted], dtype=np.float64)
        kept_splits = arrays["critic_split"][indices]
        scale = float(rwl[kept_splits == 0].mean())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Training-only backend normalization is invalid")
        new_reward = (-rwl / scale).astype(np.float32)
        if not np.isfinite(new_reward).all():
            raise ValueError("Normalized rewards are not finite float32 values")
        output_arrays = {key: value[indices] if key in critic_keys else value.copy() for key, value in arrays.items()}
        output_arrays["proxy_reward"] = output_arrays["reward"].copy()
        proxy_stage = str(arrays["reward_stage"].item()) if "reward_stage" in arrays else source_manifest.get("metric", {}).get("metric", "macro_only_proxy_unspecified")
        output_arrays.update(reward=new_reward,
                             reward_stage=np.asarray(STAGE), proxy_reward_stage=np.asarray(proxy_stage),
                             source_row_index=indices, backend_rwl_um=rwl,
                             backend_metrics_sha256=np.asarray([r["metrics_sha256"] for r in accepted]),
                             backend_manifest_sha256=np.asarray([r["manifest_sha256"] for r in accepted]),
                             backend_def_sha256=np.asarray([r["def_sha256"] for r in accepted]))
        for source, target in [("WNS_ns", "backend_wns_ns"), ("TNS_ns", "backend_tns_ns"),
                               ("Power_W", "backend_power_w"), ("StdCellArea_um2", "backend_stdcell_area_um2")]:
            values = [record["diagnostic_metrics"][source] for record in accepted]
            output_arrays[target] = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
            output_arrays[target + "_available"] = np.asarray([value is not None for value in values])
        np.savez_compressed(output / "dataset.npz", **output_arrays)
        with (output / "backend_labels.jsonl").open("w") as stream:
            for record in accepted:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        manifest = {"kind": "backend-relabelled terminal critic dataset", "args": source_manifest.get("args", {}),
                    "source_dataset": str(dataset), "source_dataset_sha256": source_hash,
                    "dataset_sha256": sha256_file(output / "dataset.npz"),
                    "labels_jsonl": str(labels), "labels_jsonl_sha256": sha256_file(labels),
                    "selection_plan": audit.get("selection_plan"), "selection_plan_sha256": audit.get("selection_plan_sha256"),
                    "metric": {"metric": "globalroute_rwl_um", "stage": STAGE, "not_final_table2": True},
                    "reward": "-globalroute_rWL_um / mean_globalroute_rWL_um_of_retained_training_rows_only",
                    "reward_scale_train_only": scale, "reward_scale_training_row_count": int((kept_splits == 0).sum()),
                    "reward_stage": STAGE, "proxy_reward_stage": proxy_stage,
                    "num_teacher": n, "num_critic": len(indices), "original_num_critic": m,
                    "selected_source_rows": indices.tolist(), "excluded_unrequested_source_rows": omitted,
                    "requested_failure_count": 0, "strict_row_hashes_required": require_row_hashes,
                    "row_hash_coverage": sum(r["strict_row_hashes_present"] for r in accepted) / len(accepted),
                    "retained_critic_split_counts": audit["retained_critic_split_counts"],
                    "backend_consistency": accepted[0]["backend_consistency"],
                    "source_proxy_manifest": source_manifest,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "notes": ["All teacher/noise/split rows preserved exactly", "No timing/power metric is mixed into reward",
                              "Global route is not detailed-route final Table 2 PPA",
                              "Backend failures are not silently dropped or assigned fabricated wirelength",
                              "Unrequested rows may be excluded only with explicit subset opt-in",
                              "A proxy-only evaluator cannot calibrate this backend-trained critic"]}
        write_json(output / "manifest.json", manifest)
        audit.update(status="success", output_dataset=str(output / "dataset.npz"),
                     output_dataset_sha256=manifest["dataset_sha256"], reward_scale_train_only=scale)
        write_json(output / "audit.json", audit)
        return manifest
    except Exception as exc:
        audit["status"] = "failed"
        audit["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output / "audit.json", audit)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unlabeled-subset", action="store_true")
    parser.add_argument("--require-row-hashes", action="store_true")
    parser.add_argument("--extra-critic-keys", default="")
    parser.add_argument("--extra-teacher-keys", default="")
    parser.add_argument("--selection-plan", help="Frozen plan; default checks selection_plan.json beside JSONL if present")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use Slurm for dataset relabeling and numeric processing")
    manifest = relabel(args.dataset, args.labels_jsonl, args.output, args.allow_unlabeled_subset,
                       args.require_row_hashes, [x for x in args.extra_critic_keys.split(",") if x],
                       [x for x in args.extra_teacher_keys.split(",") if x], args.selection_plan)
    print(json.dumps({"output": args.output, "num_critic": manifest["num_critic"],
                      "reward_scale_train_only": manifest["reward_scale_train_only"],
                      "excluded_unrequested_count": len(manifest["excluded_unrequested_source_rows"])}), flush=True)


if __name__ == "__main__":
    main()
