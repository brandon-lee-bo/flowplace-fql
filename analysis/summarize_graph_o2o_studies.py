"""Consolidate completed graph-FQL studies without selecting new checkpoints.

Inputs are the three independent native-PPA audits, not training logs. Rebuild
group statistics from their bound layout records; keep failure denominators,
training seeds and frozen-generator repeats distinct. This does not replace
the upstream native DEF/log/checkpoint audits or mark the goal complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

KEYS = ("DRT_WL", "WNS", "TNS", "Power", "StdCellArea", "DRC")
POLICIES = ("flowplace_r0", "bc_r0", "fql_r0", "bc_r2", "fql_r2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("Bound report changed: " + str(path))
    return json.loads(raw), dict(path=str(path.resolve()), sha256=digest, bytes=len(raw))


def summarize(records):
    """Keep physical failures and refuse unresolved states or invented PPA."""
    groups = {}
    for row in records:
        status, metrics = row["status"], row["metrics"]
        if status == "completed":
            if metrics is None or any(not math.isfinite(metrics[k]) for k in KEYS):
                raise ValueError("A completed layout lacks finite native PPA")
            if metrics["DRT_WL"] <= 0 or metrics["DRC"] < 0 or int(metrics["DRC"]) != metrics["DRC"]:
                raise ValueError("Invalid wirelength or DRC count")
        elif status != "upstream_physical_failure" or metrics is not None:
            raise ValueError("Unresolved failure or fabricated failure metrics")
        groups.setdefault(row["group"], []).append(row)
    result = {}
    for name, rows in groups.items():
        good = [r for r in rows if r["status"] == "completed"]
        result[name] = dict(requested=len(rows), completed=len(good),
            drc_clean=sum(r["metrics"]["DRC"] == 0 for r in good),
            mean=({k: statistics.mean(r["metrics"][k] for r in good) for k in KEYS}
                  if len(good) == len(rows) else None))
    return result


def compare_groups(actual, declared):
    if set(actual) != set(declared):
        raise ValueError("A declared group disappeared")
    for name, group in actual.items():
        saved = declared[name]
        for key in ("requested", "completed", "drc_clean"):
            if group[key] != saved[key]:
                raise ValueError("Changed physical denominator: " + name)
        if (group["mean"] is None) != (saved["mean"] is None):
            raise ValueError("Partial cohort was converted to a complete mean")
        if group["mean"] is not None:
            for key in KEYS:
                if not math.isclose(group["mean"][key], saved["mean"][key], rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError("Aggregate differs from bound layout data: " + name + "/" + key)


def route_groups(path, expected_sha, expected_count):
    summary, binding = read_bound(path, expected_sha)
    if summary["status"] != "all_requested_outcomes_accounted" or summary["requested"] != expected_count:
        raise ValueError("Final routing scope is incomplete")
    rows = [dict(group=r["task"]["group"], status=r["result"]["status"],
                 metrics=r["result"].get("metrics")) for r in summary["records"]]
    if len(rows) != expected_count:
        raise ValueError("Missing final-route request")
    groups = summarize(rows)
    compare_groups(groups, summary["groups"])
    if any(groups[k]["requested"] != 4 for k in POLICIES):
        raise ValueError("Policy evaluation must preserve all four fixed noises")
    return groups, binding


def common_backend(identity):
    """Across circuits only the design configuration may differ."""
    if identity["num_cores"] != 8 or identity["route_end_iteration"] != 20:
        raise ValueError("Unexpected backend thread or repair budget")
    components = identity["physical_flow_components"]
    return dict(tool_image_sha256=identity["tool_image_sha256"],
        num_cores=8, route_end_iteration=20,
        physical_flow_components={k: v for k, v in components.items() if k != "design_config_sha256"},
        detailed_route_flow_components=identity["detailed_route_flow_components"])


def load_main(plan):
    report, binding = read_bound(plan["main_report"], plan["main_report_sha256"])
    if report["status"] != "all_predeclared_training_seeds_accounted" or set(report["bindings"]) != {"0", "1", "2"}:
        raise ValueError("The main three-seed experiment is incomplete")
    per_seed, bound = {}, [binding]
    for seed, row in report["bindings"].items():
        groups, item = route_groups(Path(row["experiment"]) / "full_route/summary.json", row["route_summary_sha256"], 21)
        per_seed[seed] = groups
        bound.append(item)
    declared = {r["group"]: r for r in report["groups"]}
    if set(declared) != set(POLICIES) or not report["frozen_FlowPlace_not_independently_retrained"]:
        raise ValueError("Main policy or frozen-generator identities changed")
    aggregate = {}
    for name in POLICIES:
        rows = [per_seed[s][name] for s in ("0", "1", "2")]
        mean = ({k: statistics.mean(r["mean"][k] for r in rows) for k in KEYS}
                if all(r["mean"] is not None for r in rows) else None)
        group = dict(requested=sum(r["requested"] for r in rows),
            completed=sum(r["completed"] for r in rows), drc_clean=sum(r["drc_clean"] for r in rows), mean=mean)
        compare_groups({name: group}, {name: declared[name]})
        sd = ({k: statistics.stdev(r["mean"][k] for r in rows) for k in KEYS} if mean else None)
        if sd is not None:
            for k in KEYS:
                if not math.isclose(sd[k], declared[name]["sd_across_training_seed_means"][k], rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError("Training-seed spread changed")
        aggregate[name] = dict(group, sd_across_training_seed_means=sd)
    return report, aggregate, per_seed, bound


def delta(fql, bc):
    if fql["mean"] is None or bc["mean"] is None:
        return None
    return 100 * (fql["mean"]["DRT_WL"] / bc["mean"]["DRT_WL"] - 1)


def run(args):
    plan_path = Path(args.plan)
    plan, plan_binding = read_bound(plan_path)
    if sha(__file__) != plan["summary_source_sha256"]:
        raise ValueError("Consolidation source changed after registration")
    main, main_groups, seeds, bindings = load_main(plan)
    if args.check_main:
        print(json.dumps(dict(status="actual_main_statistics_reproduced", seeds=list(seeds),
            logical_final_requests=63, online_delta_percent=delta(main_groups["fql_r2"], main_groups["bc_r2"]))))
        return
    expanded, b = read_bound(plan["expanded_report"])
    bindings.append(b)
    second, b = read_bound(plan["second_report"])
    bindings.append(b)
    if (expanded["status"] != "expanded_offline_to_online_accounted" or
            expanded["report_source_sha256"] != plan["expanded_report_source_sha256"] or
            not expanded["final_route_reports"] or not expanded["actual_native_route_audits"]):
        raise ValueError("Expanded-policy native audit is incomplete or changed")
    if (second["status"] != "second_fixed_design_training_online_and_final_PPA_audited" or
            second["source_sha256"] != plan["second_report_source_sha256"] or second["design"] != "bp_be" or
            second["actual_backend_outcomes"] != 76 or second["requested_final_routes"] != 23 or
            second["final_used_online_labels_per_arm"] != 16 or second["independent_training_seeds"] != 1):
        raise ValueError("Second-circuit native/online audit is incomplete or changed")
    reference = expanded["final_route_reports"]["original64"]
    if reference["sha256"] != main["bindings"]["0"]["route_summary_sha256"]:
        raise ValueError("The expanded study changed its matched seed0 reference")
    item = expanded["final_route_reports"]["expanded384"]
    expanded_groups, b = route_groups(item["path"], item["sha256"], 21)
    bindings.append(b)
    second_rows = [dict(group=r["group"], status=r["status"],
        metrics={k: r[k] for k in KEYS} if r["status"] == "completed" else None) for r in second["final_layouts"]]
    if len(second_rows) != 23:
        raise ValueError("Missing second-circuit final layouts")
    second_groups = summarize(second_rows)
    compare_groups(second_groups, second["groups"])
    backend = common_backend(main["common_physical_flow"])
    if any(common_backend(i) != backend for i in [expanded["common_final_flow"], second["common_physical_identity"]]):
        raise ValueError("Cross-study tool/platform/physical settings differ")
    studies = [("bp_fe_64", "bp_fe", 64, 3, main_groups),
               ("bp_fe_384", "bp_fe", 384, 1, expanded_groups),
               ("bp_be_64", "bp_be", 64, 1, second_groups)]
    table, comparisons = [], []
    for name, design, count, n, groups in studies:
        if any(groups[k]["requested"] != 4 * n for k in POLICIES):
            raise ValueError("Study noise/seed denominator changed")
        for method in POLICIES:
            group = groups[method]
            row = dict(study=name, design=design, offline_TRAIN_labels=count,
                used_online_labels_per_arm=16 if method.endswith("r2") else 0,
                method=method, independent_policy_training_seeds=0 if method == "flowplace_r0" else n,
                frozen_FlowPlace=method == "flowplace_r0", requested=group["requested"],
                completed=group["completed"], drc_clean=group["drc_clean"])
            row.update({k: (group["mean"][k] if group["mean"] else None) for k in KEYS})
            row["rWL_training_seed_SD_m"] = (group["sd_across_training_seed_means"]["DRT_WL"] / 1e6
                if n > 1 and method != "flowplace_r0" and group["mean"] else None)
            table.append(row)
        for stage in (0, 2):
            paired = ([delta(seeds[s][f"fql_r{stage}"], seeds[s][f"bc_r{stage}"]) for s in ("0", "1", "2")]
                      if n == 3 else [delta(groups[f"fql_r{stage}"], groups[f"bc_r{stage}"])])
            comparisons.append(dict(study=name, stage=stage, independent_training_seeds=n,
                delta_percent_ratio_of_means=delta(groups[f"fql_r{stage}"], groups[f"bc_r{stage}"]),
                per_training_seed_delta_percent=paired, improved_seeds=sum(x < 0 for x in paired if x is not None)))
    output = Path(plan["output"]) / ("job_" + os.environ["SLURM_JOB_ID"])
    output.mkdir(parents=True, exist_ok=False)
    for i, item in enumerate(bindings):
        copy = output / f"input_{i:02d}_{Path(item['path']).name}"
        shutil.copyfile(item["path"], copy)
        if sha(copy) != item["sha256"] or sha(item["path"]) != item["sha256"]:
            raise ValueError("Completed-study evidence changed during consolidation")
    shutil.copyfile(__file__, output / Path(__file__).name)
    evidence = dict(status="three_completed_studies_consolidated", studies=table, comparisons=comparisons,
        input_bindings=bindings, plan_binding=plan_binding, common_backend_excluding_design_config=backend,
        native_audits_are_upstream=True, no_checkpoint_or_layout_selection=True,
        single_seed_studies_do_not_establish_stability=True, goal_completed=False,
        slurm_job_id=os.environ["SLURM_JOB_ID"], source_sha256=sha(__file__))
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    with (output / "policy_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    lines = ["# FlowPlace＋Flow-Q：完整实验汇总", "",
        "统一 OpenROAD/Nangate45、八线程、详细布线 cap20。各组完整分母保留，失败组不报告完整均值。",
        "主实验先平均每个训练 seed 的四个噪声，再统计三个 seed；另外两组各只有一个训练 seed。",
        "冻结 FlowPlace 没有独立重训；重复评估次数不等于训练 seed 数。", "",
        "| 实验 / TRAIN标签 | 方法 | 完成/请求 | DRC=0 | rWL（m） | WNS（ns） | TNS（ns） | 功耗（W） | 面积（μm²） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        values = ([row["DRT_WL"] / 1e6] + [row[k] for k in KEYS[1:5]]) if row["DRT_WL"] is not None else [None] * 5
        cells = ["—" if v is None else f"{v:.6g}" for v in values]
        if row["rWL_training_seed_SD_m"] is not None:
            cells[0] += f" ± {row['rWL_training_seed_SD_m']:.6f}"
        cells = " | ".join(cells)
        lines.append(f"| {row['study']} | {row['method']} | {row['completed']}/{row['requested']} | {row['drc_clean']} | {cells} |")
    lines += ["", "| 实验 | 阶段 | FQL相对BC的最终线长变化 | 改善的训练seed |",
              "| --- | --- | --- | --- |"]
    for row in comparisons:
        value = row["delta_percent_ratio_of_means"]
        cell = "—" if value is None else f"{value:+.3f}%"
        lines.append(f"| {row['study']} | {'离线' if row['stage']==0 else '在线两轮'} | {cell} | {row['improved_seeds']}/{row['independent_training_seeds']} |")
    lines += ["", "负百分比表示线长降低；单个训练 seed 的结果不能证明稳定收益。",
        "r0表示离线训练结束，r2表示两轮在线更新结束；±为三个训练seed均值之间的样本标准差。",
        "各最终在线策略实际使用16条自采反馈；最后额外8条未进入该 checkpoint。",
        "扩大标签同时改变 Q 的 TRAIN-only 特征归一化，行为目标和更新预算不变；其评分预算更大。",
        "所有时序原值保留；DRC=0不等于时序收敛。best64/80/384/400均为单个已测参考，见各原始报告，不能当成策略均值。",
        "本汇总只整合已完成研究，不改变选择规则，也不自动声明整个 goal 完成。", ""]
    (output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=evidence["status"], output=str(output), policy_rows=len(table))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--check-main", action="store_true")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run completed-study consolidation through Slurm")
    run(args)
