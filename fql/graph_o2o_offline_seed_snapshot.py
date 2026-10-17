"""Audit the completed offline comparison while online seed runs continue.

Uses all three previously registered seeds/four noises. It never reads online
metrics or changes training/model selection. Frozen baselines are not retrained
replicates. All actual final outcomes and checkpoint deltas are independently
checked rather than trusting an incomplete whole-experiment controller report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch

# Import this first so the upstream adapter cannot shadow project analysis.
from .graph_o2o_train import ROOT
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests, validate_request
from .graph_o2o_collect import audit_outcome
from .graph_o2o_design_policy_report import actual_weight_changes
from .graph_o2o_final_evidence import audit_final_outcome, LEGACY_REPEAT_PLAN

KEYS = ["DRT_WL", "WNS", "TNS", "Power", "StdCellArea", "DRC"]


def tensor_signature(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update((key + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legacy_actual_changes(saved, parent):
    """Compute changes without inventing an absent historical delta record."""
    result = {}
    for module in ["flow", "actor", "critic"]:
        before, after = parent[module], saved[module]
        if set(before) != set(after):
            raise ValueError("Legacy parent model inventory differs")
        count, square, maximum = 0, 0., 0.
        for name, value in after.items():
            old = before[name]
            if value.dtype != old.dtype or value.shape != old.shape:
                raise ValueError("Legacy parameter shape/type changed")
            if not value.is_floating_point():
                if not torch.equal(value, old):
                    raise ValueError("Legacy nonfloating buffer changed")
                continue
            if not torch.isfinite(value).all():
                raise ValueError("Legacy nonfinite tensor")
            if not torch.equal(value, old):
                delta = value.double() - old.double()
                count += 1
                square += float(delta.square().sum())
                maximum = max(maximum, float(delta.abs().max()))
        if count <= 0:
            raise ValueError("Legacy network did not change")
        result[module] = dict(changed_parameter_tensors=count, parameter_delta_l2=square**.5,
                              max_absolute_parameter_delta=maximum)
    return result


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Offline snapshot source changed: " + name)
    for item in plan["bindings"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Offline snapshot frozen input changed: " + item["path"])
    original_plan = json.loads(Path(plan["replication_plan"]).read_text())
    experiments = {0: Path(original_plan["source_experiment"]),
                   **{int(k): Path(v) for k, v in original_plan["experiments"].items()}}
    if sorted(experiments) != original_plan["independent_training_seeds"] or sorted(experiments) != [0, 1, 2]:
        raise ValueError("All three predeclared training seeds are required")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    configs = {seed: json.loads((exp / "config.json").read_text()) for seed, exp in experiments.items()}
    for cfg in configs.values():
        if any(cfg[key] != configs[0][key] for key in plan["matched_hyperparameters"]):
            raise ValueError("Offline training hyperparameters differ")
    signatures, changes, training = {}, {}, {}
    layouts, audits, paired_noises, frozen_geometry = [], [], {}, {}
    common_identity = None
    tasks = {(r["seed"], r["group"], r["index"]): r for r in plan["outcomes"]}
    if len(tasks) != 37:
        raise ValueError("Expected24 learned-policy,12 frozen-policy and1 fixed-reference outcomes")
    for seed, exp in sorted(experiments.items()):
        folder = exp / f"seed_{seed}/round_0"
        status = json.loads((folder / "training_status.json").read_text())
        if (status["status"] != "completed" or status["seed"] != seed or status["round"] != 0
                or status["config_sha256"] != sha(exp / "config.json")):
            raise ValueError("Actual offline seed training is incomplete or mismatched")
        training[str(seed)] = dict(job=status["slurm_job_id"], status_sha256=sha(folder / "training_status.json"))
        for arm in ["bc", "fql"]:
            completion = json.loads((folder / arm / "completed.json").read_text())
            checkpoint = folder / arm / "checkpoint.pt"
            parent = Path(completion.get("parent_checkpoint", folder / "common.pt"))
            parent_sha = sha(parent)
            if (sha(checkpoint) != completion["checkpoint_sha256"]
                    or parent_sha != completion.get("parent_checkpoint_sha256", parent_sha)
                    or completion["seed"] != seed or completion["round"] != 0
                    or completion["online_training_rows"] != 0 or completion["train_rows"] != 64):
                raise ValueError("Offline checkpoint or training-data binding differs")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            before = torch.load(parent, map_location="cpu", weights_only=False)
            if saved["iteration"] != 1000 or saved["seed"] != seed:
                raise ValueError("Fixed offline checkpoint selection changed")
            if "parameter_changes" in completion and "parent_checkpoint_sha256" in completion:
                changes[f"{seed}/{arm}"] = actual_weight_changes(saved, before, completion["parameter_changes"])
            elif seed == 0:
                changes[f"{seed}/{arm}"] = legacy_actual_changes(saved, before)
                training[str(seed)]["legacy_parent_note"] = (
                    "Original offline completion predates parent-hash and parameter-delta fields. "
                    "Differences are recomputed from the retained common.pt now, not matched "
                    "against an invented historical delta or runtime parent hash.")
                training[str(seed)]["retrospective_common_checkpoint_sha256"] = parent_sha
            else:
                raise ValueError("New seed is missing its required parent/delta record")
            if sha(parent) != parent_sha:
                raise ValueError("Parent checkpoint changed during snapshot")
            signatures[f"{seed}/{arm}"] = tensor_signature(saved["actor"])
            del saved, before
        rows, _ = load_requests(exp, seed, 0)
        for arm in ["bc", "fql", "flowplace"]:
            chosen = [r for r in rows if r["role"] == "evaluation" and r["arm"] == arm]
            if [r["index"] for r in chosen] != list(range(4)):
                raise ValueError("Lost predeclared offline evaluation noise")
            for row in chosen:
                validate_request(row)
                measured = audit_outcome(row, configs[seed], status["reward_scale"])
                if row["training_feedback_eligible"] or row["noise_seed"] != original_plan["paired_evaluation_noise_seed"]:
                    raise ValueError("Evaluation data entered training or pairing changed")
                with np.load(Path(row["sample_dir"]) / "actions.npz", allow_pickle=False) as arrays:
                    noise = arrays["noise"].copy()
                    action = arrays["action"].copy()
                index = row["index"]
                if index in paired_noises and not np.array_equal(noise, paired_noises[index]):
                    raise ValueError("Offline evaluation noises are not exactly paired")
                paired_noises[index] = noise
                if arm == "flowplace":
                    if index in frozen_geometry and not np.array_equal(action, frozen_geometry[index]):
                        raise ValueError("Frozen baseline layout differs; cannot deduplicate its interpretation")
                    frozen_geometry[index] = action
                task = tasks[(seed, arm, index)]
                outcome = json.loads(Path(task["path"]).read_text())
                if outcome["status"] == "completed":
                    checked = audit_final_outcome(outcome, design="bp_fe", route_cap=20, legacy_plan=LEGACY_REPEAT_PLAN)
                    if common_identity is None:
                        common_identity = checked["physical_identity"]
                    if common_identity != checked["physical_identity"]:
                        raise ValueError("Final physical flow differs")
                    if (Path(checked["source_run"]).resolve() != Path(measured["physical_run"]).resolve()
                            or outcome["spec"]["source_manifest_sha256"] != measured["physical_manifest_sha256"]):
                        raise ValueError("Routed result is not bound to this actual policy request")
                    audits.append(dict(path=task["path"], sha256=task["sha256"],
                                       final_macro_count=checked["final_macro_count"],
                                       legacy_exact_repeat_evidence=checked["legacy_exact_repeat_evidence"]))
                elif outcome["status"] != "upstream_physical_failure":
                    raise ValueError("Unresolved offline physical outcome")
                metrics = outcome.get("metrics") or {}
                layouts.append(dict(seed=seed, arm=arm, noise_index=index, status=outcome["status"],
                                    **{k: metrics.get(k) for k in KEYS},
                                    outcome=task["path"], outcome_sha256=task["sha256"]))
    for arm in ["bc", "fql"]:
        if len({signatures[f"{seed}/{arm}"] for seed in experiments}) != 3:
            raise ValueError("Independent training seeds unexpectedly share actor tensors")
    reference = json.loads(Path(tasks[(0, "best_train", 0)]["path"]).read_text())
    reference_audit = audit_final_outcome(reference, design="bp_fe", route_cap=20, legacy_plan=LEGACY_REPEAT_PLAN)
    if reference_audit["physical_identity"] != common_identity:
        raise ValueError("Fixed TRAIN reference physical flow differs")
    per_seed = []
    for seed in experiments:
        for arm in ["bc", "fql", "flowplace"]:
            rows = [r for r in layouts if r["seed"] == seed and r["arm"] == arm]
            complete = all(r["status"] == "completed" for r in rows)
            per_seed.append(dict(seed=seed, arm=arm, requested=4,
                completed=sum(r["status"] == "completed" for r in rows),
                drc_clean=sum(r["DRC"] == 0 for r in rows),
                **{k: statistics.mean(r[k] for r in rows) if complete else None for k in KEYS}))
    groups = {}
    for arm in ["bc", "fql"]:
        rows = [r for r in per_seed if r["arm"] == arm]
        complete = all(r["completed"] == 4 for r in rows)
        groups[arm] = dict(independent_training_seeds=3, requested=12,
            completed=sum(r["completed"] for r in rows), drc_clean=sum(r["drc_clean"] for r in rows),
            mean={k: statistics.mean(r[k] for r in rows) for k in KEYS} if complete else None,
            sample_sd_across_seed_means={k: statistics.stdev(r[k] for r in rows) for k in KEYS} if complete else None)
    comparisons = []
    for seed in experiments:
        a = next(r for r in per_seed if r["seed"] == seed and r["arm"] == "bc")
        b = next(r for r in per_seed if r["seed"] == seed and r["arm"] == "fql")
        comparisons.append(dict(seed=seed, fql_rwl_change_vs_bc_pct=
                                100 * (b["DRT_WL"] / a["DRT_WL"] - 1) if a["DRT_WL"] and b["DRT_WL"] else None))
    dest = Path(plan["output"])
    dest.mkdir(parents=True, exist_ok=False)
    atomic_json(dest / "evidence.json", dict(status="all_three_predeclared_offline_training_seeds_audited",
        plan_sha256=sha(plan_path), groups=groups, per_seed=per_seed, paired_comparisons=comparisons,
        actual_training=training, actual_weight_changes=changes, actor_tensor_signatures=signatures,
        native_route_audits=audits, single_fixed_TRAIN_reference=reference,
        fixed_reference_native_audit=reference_audit,
        frozen_baseline=next(r for r in per_seed if r["seed"] == 0 and r["arm"] == "flowplace"),
        frozen_baseline_is_same_four_layouts_not_three_independent_trainings=True,
        scope="bp_fe offline only, all predeclared seeds0/1/2; online results deliberately incomplete",
        online_metrics_read=False, model_selection_changed=False, goal_completed=False,
        slurm_job_id=os.environ["SLURM_JOB_ID"]))
    write_csv(dest / "per_layout.csv", layouts)
    write_csv(dest / "per_training_seed.csv", per_seed)
    lines = ["# 三个训练 seed 的离线阶段详细布线结果", "",
        "bp_fe，统一 OpenROAD/Nangate45、八线程和 cap20。每个训练 seed 先平均四个固定评估噪声，再计算三个 seed 的均值和样本标准差。",
        "这里只汇总已完成的离线阶段；在线训练和第二电路仍在进行，没有选择最好 seed。", "",
        "| 训练 seed | BC rWL（m） | FQL rWL（m） | FQL相对BC变化 |",
        "| --- | --- | --- | --- |"]
    for row in comparisons:
        seed = row["seed"]
        a = next(r for r in per_seed if r["seed"] == seed and r["arm"] == "bc")
        b = next(r for r in per_seed if r["seed"] == seed and r["arm"] == "fql")
        if a["DRT_WL"] is not None and b["DRT_WL"] is not None:
            lines.append(f"| {seed} | {a['DRT_WL']/1e6:.6f} | {b['DRT_WL']/1e6:.6f} | {row['fql_rwl_change_vs_bc_pct']:+.3f}% |")
        else:
            lines.append(f"| {seed} | — | — | 存在失败，保留分母 |")
    lines += ["", "变化为负表示 FQL 线长更低。", "",
        "| 方法 | 完成/请求 | DRC=0 | rWL（m，均值±seed间标准差） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, group in groups.items():
        m, sd = group["mean"], group["sample_sd_across_seed_means"]
        values = (f"{m['DRT_WL']/1e6:.6f} ± {sd['DRT_WL']/1e6:.6f} | " +
                  " | ".join(f"{m[k]:.6g}" for k in KEYS[1:-1])) if m else "— | — | — | — | —"
        lines.append(f"| {arm} offline | {group['completed']}/12 | {group['drc_clean']} | {values} |")
    frozen = next(r for r in per_seed if r["seed"] == 0 and r["arm"] == "flowplace")
    lines += ["", f"冻结 FlowPlace 的同四个布局平均rWL为{frozen['DRT_WL']/1e6:.6f}m；没有重新训练，不能重复计算成三个训练seed。",
        f"best64 TRAIN 是单个实测参考，rWL={reference['metrics']['DRT_WL']/1e6:.6f}m。",
        "所有失败保留分母，非零DRC不剔除。样本量只有三个训练seed，当前阶段不能据此宣称稳定收益或整体PPA改善。", ""]
    (dest / "summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status="reported", output=str(dest), comparisons=comparisons)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use Slurm for native-file and actual-checkpoint audits")
    run(args.plan)
