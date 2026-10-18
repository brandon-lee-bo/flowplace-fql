"""Predeclare and audit development-only, matched multi-seed final routing.

This module never launches a model, scheduler job or EDA executable.  The plan
is compatible with graph_o2o_route_controller, whose historical single-seed
aggregate is deliberately ignored. Resolve seals completed GRT inputs; report
re-reads their native evidence and audits actual final routing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics

import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_final_evidence import audit_final_outcome
from .qstudy_actor_report import check_training, native_fingerprints
from .qstudy_backend import context, checked_request, checked_existing

METRICS = ("DRT_WL", "WNS", "TNS", "Power", "StdCellArea", "DRC")
UNITS = dict(DRT_WL="um", WNS="ns", TNS="ns", Power="W", StdCellArea="um^2", DRC="count")
SEEDS = (0, 1, 2)
ROUNDS = (0, 2)
INDICES = (0, 1, 2, 3)
SOURCES = ("fql/graph_o2o_route_controller.py", "fql/graph_o2o_full_route.py",
           "fql/graph_o2o_full_route.slurm", "fql/graph_o2o_finish.py", "fql/graph_o2o_finish.sh",
           "fql/graph_o2o_final_evidence.py", "fql/qstudy_actor.py", "fql/qstudy_models.py",
           "fql/qstudy_backend.py", "fql/graph_o2o_backend.py", "fql/graph_o2o_geometry.py",
           "fql/qstudy_actor_report.py", "fql/qstudy_final_route.py")


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    """Resume exact same declaration; never silently replace a binding."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("Immutable declaration changed: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def task_key(arm, seed, round_number, index):
    return f"{arm}_s{seed}_r{round_number}_n{index}"


def build_plan(output, experiments, prefix, concurrency=2):
    """Metadata only: never reads source outcomes or selects on any metric."""
    output = Path(output).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix) or concurrency < 1:
        raise ValueError("Invalid namespace or concurrency")
    configs = {arm: read(Path(exp) / "config.json") for arm, exp in experiments.items()}
    if set(configs) != {"bc", "legacy"}:
        raise ValueError("Exactly BC and legacy fixed-policy arms are required")
    # Q pressure is the declared contrast; all other training settings match.
    allowed = {"arm", "run_prefix", "q_weight"}
    keys = set(configs["bc"]) | set(configs["legacy"])
    for name in keys - allowed:
        if configs["bc"].get(name) != configs["legacy"].get(name):
            raise ValueError("Unmatched policy configuration: " + name)
    for arm, cfg in configs.items():
        if (cfg["arm"] != arm or cfg["design"] != "bp_fe" or cfg["seeds"] != list(SEEDS)
                or cfg["evaluation_samples"] != 8 or cfg.get("online_rounds", 2) != 2
                or cfg["evaluation_purpose"] != "development_validation"
                or cfg["backend_threads"] != 8 or cfg["terminal_mask"] != 0
                or cfg["q_weight"] != (0 if arm == "bc" else 1)):
            raise ValueError("Configuration differs from the bounded development protocol")
    bindings = {arm: dict(path=str(Path(exp).resolve()),
        config_sha256=sha(Path(exp) / "config.json"), config=configs[arm])
        for arm, exp in experiments.items()}
    tasks = []
    for arm in ("bc", "legacy"):
        exp = Path(bindings[arm]["path"])
        for seed in SEEDS:
            for round_number in ROUNDS:
                for index in INDICES:
                    key = task_key(arm, seed, round_number, index)
                    tasks.append(dict(key=key, group=f"{arm}_r{round_number}",
                        method=f"{arm}_round_{round_number}", arm=arm, seed=seed,
                        round=round_number, noise_index=index,
                        source_outcome=str(exp / f"seed_{seed}/round_{round_number}/requests" /
                                           arm / "evaluation" / str(index) / "outcome.json"),
                        output=str(output / "full_route" / key), run_id=prefix + "_" + key))
    return dict(version=1, study="qstudy_development_final_route_v1", design="bp_fe",
        purpose="development_validation", fresh_confirmation=False,
        historical_table2=False, requested=48, actor_seeds=list(SEEDS), rounds=list(ROUNDS),
        evaluation_indices=list(INDICES), original_evaluation_samples=8,
        selection="first four predeclared indices; no per-policy Q/backend/candidate selection",
        policy_estimand="mean over fixed noise indices within each actor seed, then across seeds",
        final_metrics=list(METRICS), metric_units=UNITS,
        detailed_route_end_iteration=20, backend_threads=8, backend_storage="scratch",
        concurrency=concurrency, run_id_prefix=prefix, actor_experiments=bindings,
        source_sha256={name: sha(ROOT / name) for name in SOURCES}, tasks=tasks,
        baseline_note="BC is a separately trained one-step policy, not the historical public FlowPlace baseline",
        failure_protocol="Retain all48; no numerical PPA for physical failures or missing/infrastructure tasks",
        congestion_note="GRT congestion is not a final Table2 routed-congestion measurement",
        extension_note="Any later fixed/best measured candidate reference must stay separate from policy means; inherited physical calls must be deduplicated")


def validate_plan(experiment):
    experiment = Path(experiment).resolve()
    path = experiment / "full_route_plan.json"
    plan = read(path)
    declaration = read(experiment / "development_route_request.json")
    if declaration["plan_sha256"] != sha(path) or declaration["requested"] != 48:
        raise ValueError("Predeclared request no longer binds this plan")
    expected = build_plan(experiment, {arm: value["path"] for arm, value in
        plan["actor_experiments"].items()}, plan["run_id_prefix"], plan["concurrency"])
    if plan != expected:
        raise ValueError("Predeclared task/configuration/source identity changed")
    return plan


def plan_command(args):
    output = Path(args.output).resolve()
    plan = build_plan(output, dict(bc=args.bc, legacy=args.legacy), args.run_prefix, args.concurrency)
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / "full_route_plan.json", plan)
    freeze(output / "development_route_request.json", dict(version=1,
        plan_sha256=sha(output / "full_route_plan.json"), requested=48,
        frozen_without_reading_native_metrics=True, purpose="development_validation",
        helper_sha256=sha(__file__)))
    print(json.dumps(dict(status="predeclared", requested=48, output=str(output))))


def verify_actor_chain(experiment, cfg, seed, round_number):
    config_sha = sha(experiment / "config.json")
    current = None
    for r in range(round_number + 1):
        folder = experiment / f"seed_{seed}/round_{r}"
        current = check_training(folder, cfg, seed, r, config_sha, curve_snapshot={})
        if not current["completed"]:
            raise FileNotFoundError("Actor round is not completed: " + str(folder))
        binding = read(folder / "binding.json")
        if r == 0:
            if binding["parent"] is not None or binding["replay"]:
                raise ValueError("Offline round unexpectedly has online replay/parent")
            continue
        previous = experiment / f"seed_{seed}/round_{r-1}"
        expected_parent = dict(checkpoint_sha256=sha(previous / "checkpoint.pt"),
            completed_sha256=sha(previous / "completed.json"), binding_sha256=sha(previous / "binding.json"))
        if binding["parent"] != expected_parent:
            raise ValueError("Online actor parent chain changed")
        expected = {str((experiment / f"seed_{seed}/round_{p}/requests" / cfg["arm"] /
                         "collect" / str(i) / "outcome.json").resolve())
                    for p in range(r) for i in range(cfg["online_samples"])}
        observed = []
        for entry in binding["replay"]:
            path = Path(entry["path"]).resolve()
            outcome = read(path)
            if (entry["sha256"] != sha(path) or entry["status"] != outcome["status"]
                    or outcome["arm"] != cfg["arm"] or outcome["seed"] != seed
                    or outcome["role"] != "collect" or not outcome["training_feedback_eligible"]
                    or outcome["round"] >= r):
                raise ValueError("Online replay is changed, unmatched or evaluation-contaminated")
            observed.append(str(path))
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("Online replay denominator differs from declared own-collection data")
    return current


def round_context(plan, task, cache):
    arm, seed, r = task["arm"], task["seed"], task["round"]
    key = (arm, seed, r)
    if key not in cache:
        binding = plan["actor_experiments"][arm]
        experiment = Path(binding["path"])
        cfg = binding["config"]
        for paths, digests in (("common_checkpoints", "common_checkpoint_sha256"),
                               ("critic_checkpoints", "critic_checkpoint_sha256")):
            if sha(cfg[paths][str(seed)]) != cfg[digests][str(seed)]:
                raise ValueError("Frozen actor initialization file changed: " + paths)
        training = verify_actor_chain(experiment, cfg, seed, r)
        rows, path = load_requests(experiment, seed, r)
        done = read(path.with_name("completed.json"))
        if (done.get("status") != "completed" or done.get("selection") != "all predetermined noises; no Q-selection"
                or done.get("evaluation_never_replayed") is not True):
            raise ValueError("Export did not preserve all predeclared evaluation noises")
        counts = dict(evaluation=cfg["evaluation_samples"])
        if r < cfg.get("online_rounds", 2):
            counts["collect"] = cfg["online_samples"]
        if set(row["role"] for row in rows) != set(counts):
            raise ValueError("Export role set changed")
        for role, n in counts.items():
            subset = [row for row in rows if row["role"] == role]
            if len(subset) != n or {row["index"] for row in subset} != set(range(n)):
                raise ValueError("Export count/index set changed")
        cache[key] = (training, rows, path, context(experiment))
    return cache[key]


def inspect_source(plan, task, cache):
    """Strict read-only validation, including a guard against an EDA audit call."""
    training, rows, tasks, (cfg, verification, binding) = round_context(plan, task, cache)
    matches = [row for row in rows if row["role"] == "evaluation" and row["index"] == task["noise_index"]]
    if len(matches) != 1:
        raise ValueError("Selected evaluation index missing or duplicated")
    row = matches[0]
    sample = checked_request(row, argparse.Namespace(seed=task["seed"], round=task["round"]), cfg, binding)
    if (row["arm"] != task["arm"] or row["checkpoint_sha256"] != training["checkpoint_sha256"]
            or row["noise_seed"] != cfg["evaluation_noise_seed"]
            or row.get("evaluation_purpose") != "development_validation"
            or row["training_feedback_eligible"] is not False
            or (sample / "outcome.json").resolve() != Path(task["source_outcome"]).resolve()):
        raise ValueError("Evaluation request identity changed")
    out = read(task["source_outcome"])
    for name in ("arm", "seed", "round", "role", "index", "graph", "training_feedback_eligible"):
        if out.get(name) != row.get(name):
            raise ValueError("GRT outcome/request identity changed: " + name)
    if out.get("physical_run"):
        run = Path(out["physical_run"])
        if read(run / "manifest.json").get("success") and not (run / "backend_macro_audit.json").is_file():
            raise ValueError("Missing saved native geometry audit; read-only reporter cannot create it")
    checked_existing(out, row, sample, tasks, binding, cfg, verification)
    if out["status"] not in {"success", "geometry_failure", "physical_failure"}:
        raise ValueError("Unclassified GRT outcome is not an admitted physical result")
    with np.load(sample / "actions.npz", allow_pickle=False) as arrays:
        values = {name: arrays[name].tolist() for name in ("noise", "raw_action", "action")}
    return dict(version=1, key=task["key"], arm=task["arm"], seed=task["seed"], round=task["round"],
        noise_index=task["noise_index"], status=out["status"],
        source_outcome=str(Path(task["source_outcome"]).resolve()), source_outcome_sha256=sha(task["source_outcome"]),
        request_sha256=sha(sample / "request.json"), tasks_sha256=sha(tasks),
        export_completion_sha256=sha(tasks.with_name("completed.json")), actions_npz_sha256=sha(sample / "actions.npz"),
        config_sha256=binding["config_sha256"], dataset_sha256=binding["dataset_sha256"],
        actor=training, raw_action_sha256=row["raw_action_sha256"], action_sha256=row["action_sha256"],
        noise_sha256=row["noise_sha256"], noise_seed=row["noise_seed"], exact_coordinates_and_noise=values,
        def_sha256=row.get("def_sha256"), native=native_fingerprints(out),
        physical_flow_revision=binding["physical_flow_revision"], failure_class=out.get("failure_class"),
        GRT_metrics=out["metrics"], final_metrics=None)


def inspect_sources(experiment, plan, *, seal):
    cache, records = {}, []
    for task in plan["tasks"]:
        path = experiment / "source_bindings" / (task["key"] + ".json")
        item = dict(key=task["key"], state="missing", binding=None)
        try:
            binding = inspect_source(plan, task, cache)
            if not seal and (not path.is_file() or read(path) != binding):
                raise ValueError("Source not sealed or source binding changed; run resolve before report")
            item.update(state="resolved", binding=binding)
        except FileNotFoundError as error:
            item.update(state="missing", reason=str(error))
            if path.exists():
                item.update(state="invalid", reason="Previously sealed evidence missing: " + str(error))
        except Exception as error:
            item.update(state="invalid", reason=str(error))
        records.append(item)
    # A mismatch invalidates every available member of its matched index, not
    # only the last inspected task. Never leave partially valid pooled tables.
    index_digests = defaultdict(set)
    for item in records:
        if item["binding"]:
            index_digests[item["binding"]["noise_index"]].add(item["binding"]["noise_sha256"])
    bad = {i for i, values in index_digests.items() if len(values) > 1}
    for task, item in zip(plan["tasks"], records):
        if task["noise_index"] in bad:
            item.update(state="invalid", reason="Matched noise mismatch")
        if item["state"] == "resolved":
            path = experiment / "source_bindings" / (task["key"] + ".json")
            try:
                if seal:
                    freeze(path, item["binding"])
                item["binding_sha256"] = sha(path)
            except Exception as error:
                item.update(state="invalid", reason=str(error))
    return records


def summary(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values) > 1 else None)


def metric_means(rows):
    return {name: statistics.mean(row["metrics"][name] for row in rows) for name in METRICS} if rows else None


def aggregate(records):
    """Noise means first; independent actor seeds are the only SD replicates."""
    seed_rows = []
    for arm in ("bc", "legacy"):
        for r in ROUNDS:
            for seed in SEEDS:
                rows = [row for row in records if row["arm"] == arm and row["round"] == r and row["seed"] == seed]
                good = [row for row in rows if row["state"] == "completed"]
                full = len(rows) == len(INDICES) and len(good) == len(INDICES)
                seed_rows.append(dict(arm=arm, round=r, seed=seed, planned=len(INDICES),
                    counts=dict(Counter(row["state"] for row in rows)), complete=full,
                    drc_clean=sum(row["metrics"]["DRC"] == 0 for row in good),
                    mean=metric_means(good) if full else None,
                    successful_only_mean=metric_means(good)))
    groups = []
    for arm in ("bc", "legacy"):
        for r in ROUNDS:
            subset = [row for row in seed_rows if row["arm"] == arm and row["round"] == r]
            good = [row for row in subset if row["complete"]]
            layouts = [row for row in records if row["arm"] == arm and row["round"] == r]
            groups.append(dict(arm=arm, round=r, planned_actor_seeds=3, complete_actor_seeds=len(good),
                planned_layouts=12, counts=dict(Counter(row["state"] for row in layouts)),
                drc_clean=sum(row["state"] == "completed" and row["metrics"]["DRC"] == 0 for row in layouts),
                statistics={name: summary([row["mean"][name] for row in good]) for name in METRICS}
                    if len(good) == 3 else None,
                complete_seed_only_statistics={name: summary([row["mean"][name] for row in good]) for name in METRICS}))
    pairs = []
    comparisons = [(f"legacy_minus_bc_r{r}", ("legacy", r), ("bc", r)) for r in ROUNDS]
    comparisons += [(f"{arm}_round2_minus_round0", (arm, 2), (arm, 0)) for arm in ("bc", "legacy")]
    for label, lhs, rhs in comparisons:
        per_seed = []
        for seed in SEEDS:
            a = [row for row in records if (row["arm"], row["round"], row["seed"]) == (*lhs, seed)]
            b = [row for row in records if (row["arm"], row["round"], row["seed"]) == (*rhs, seed)]
            by_a, by_b = {row["noise_index"]: row for row in a}, {row["noise_index"]: row for row in b}
            matched = [i for i in INDICES if i in by_a and i in by_b
                and by_a[i]["state"] == by_b[i]["state"] == "completed"
                and by_a[i]["noise_sha256"] == by_b[i]["noise_sha256"]]
            delta = {name: statistics.mean(by_a[i]["metrics"][name] - by_b[i]["metrics"][name]
                     for i in matched) for name in METRICS} if len(matched) == 4 else None
            per_seed.append(dict(seed=seed, matched_completed_indices=matched, delta=delta))
        full = all(row["delta"] is not None for row in per_seed)
        pairs.append(dict(comparison=label, sign="lhs minus rhs in native units; negative helps WL/Power/Area/DRC, positive helps WNS/TNS",
            per_seed=per_seed, independent_complete_seeds=sum(row["delta"] is not None for row in per_seed),
            statistics={name: summary([row["delta"][name] for row in per_seed]) for name in METRICS} if full else None))
    return dict(seed_means=seed_rows, policy_groups=groups, matched_deltas=pairs)


def inspect_final(experiment, plan, task, source):
    row = {name: task[name] for name in ("key", "arm", "seed", "round", "noise_index")}
    row.update(state="missing", metrics=None, noise_sha256=None)
    if source["state"] != "resolved":
        row.update(state="source_" + source["state"], reason=source.get("reason"))
        return row
    binding = source["binding"]
    row.update(noise_sha256=binding["noise_sha256"], source_binding_sha256=source["binding_sha256"],
        source_outcome_sha256=binding["source_outcome_sha256"])
    path = Path(task["output"]) / "outcome.json"
    if binding["status"] != "success":
        if path.exists():
            out = read(path)
            if (out["status"] != "upstream_physical_failure" or out["metrics"] is not None
                    or out["source_outcome_sha256"] != binding["source_outcome_sha256"]
                    or Path(out["source_outcome"]).resolve() != Path(task["source_outcome"]).resolve()
                    or out["failure_class"] != binding["failure_class"]):
                raise ValueError("Final upstream failure is inconsistent with sealed GRT failure")
            row["outcome_sha256"] = sha(path)
        row.update(state="upstream_physical_failure", failure_class=binding["failure_class"],
            upstream_failure_sidecar_present=path.exists())
        return row
    if not path.exists():
        row.update(state="missing_final", scheduler_check_required=True)
        return row
    out = read(path)
    spec_path = Path(task["output"]) / "spec.json"
    expected = dict(source_run=str(Path(binding["native"]["physical_run"]).resolve()),
        source_manifest_sha256=binding["native"]["physical_manifest_sha256"],
        source_outcome=task["source_outcome"], source_outcome_sha256=binding["source_outcome_sha256"],
        method=task["method"], run_id=task["run_id"], route_end_iteration=20, storage="scratch")
    if (out["status"] != "completed" or out["spec"] != expected or read(spec_path) != expected
            or out["spec_sha256"] != sha(spec_path)):
        raise ValueError("Final outcome does not bind the exact predeclared source and route request")
    audit = audit_final_outcome(out, design=plan["design"], route_cap=20)
    if Path(audit["source_run"]).resolve() != Path(expected["source_run"]).resolve():
        raise ValueError("Final native audit changed source")
    row.update(state="completed", outcome_sha256=sha(path), metrics=out["metrics"],
        physical_run=str(Path(out["physical_run"]).resolve()), physical_manifest_sha256=out["physical_manifest_sha256"],
        metrics_sha256=out["metrics_sha256"], physical_identity=audit["physical_identity"],
        final_native_audit=audit, route_drc_clean=out["route_drc_clean"])
    return row


def csv_write(path, rows):
    if not rows:
        return
    flat = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows]
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def snapshot(experiment, stage, value):
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + "_job" + os.environ.get("SLURM_JOB_ID", "test")
    folder = experiment / stage / "snapshots" / token
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "report.json", value)
    atomic_json(experiment / stage / "latest.json", dict(path=str(folder), report_sha256=sha(folder / "report.json")))
    return folder


def resolve_command(args):
    experiment = Path(args.experiment).resolve()
    plan = validate_plan(experiment)
    rows = inspect_sources(experiment, plan, seal=True)
    report = dict(status="complete" if all(row["state"] == "resolved" for row in rows) else "partial",
        plan_sha256=sha(experiment / "full_route_plan.json"), requested=48,
        counts=dict(Counter(row["state"] for row in rows)), records=rows)
    folder = snapshot(experiment, "source_resolution", report)
    csv_write(folder / "sources.csv", rows)
    print(json.dumps(dict(report=str(folder), counts=report["counts"])))


def report_command(args):
    experiment = Path(args.experiment).resolve()
    plan = validate_plan(experiment)
    sources = inspect_sources(experiment, plan, seal=False)
    rows = []
    for task, source in zip(plan["tasks"], sources):
        try:
            rows.append(inspect_final(experiment, plan, task, source))
        except Exception as error:
            rows.append(dict(**{k: task[k] for k in ("key", "arm", "seed", "round", "noise_index")},
                             state="invalid", reason=str(error), metrics=None, noise_sha256=None))
    identities = {json.dumps(row["physical_identity"], sort_keys=True) for row in rows if row["state"] == "completed"}
    if len(identities) > 1:
        for row in rows:
            if row["state"] == "completed":
                row.update(state="invalid", reason="Final physical identity differs across requested tasks", metrics=None)
    seen = defaultdict(list)
    for row in rows:
        if row["state"] == "completed":
            seen[row["physical_run"]].append(row["key"])
    duplicates = {run: keys for run, keys in seen.items() if len(keys) > 1}
    if duplicates:
        for row in rows:
            if row.get("physical_run") in duplicates:
                row.update(state="invalid", reason="Distinct current policy tasks reuse the same physical call", metrics=None)
    result = aggregate(rows)
    counts = dict(Counter(row["state"] for row in rows))
    accounted = sum(counts.get(state, 0) for state in ("completed", "upstream_physical_failure"))
    report = dict(version=1, purpose="development_validation", fresh_confirmation=False,
        status="all_requested_accounted" if accounted == 48 else "partial_or_invalid",
        requested=48, counts=counts, metric_units=UNITS, plan_sha256=sha(experiment / "full_route_plan.json"),
        records=rows, **result,
        inference="Development evidence only; evaluation noise is not an independent training replicate; no candidate selection",
        failures="Failure/missing PPA stays null; complete policy mean withheld unless all four noises for all3seeds complete",
        congestion="No final routed congestion inferred from native GRT quantities",
        goal_complete=False)
    folder = snapshot(experiment, "development_report", report)
    flat = [{**{k: v for k, v in row.items() if k not in ("metrics", "final_native_audit")},
             **{name: (row.get("metrics") or {}).get(name) for name in METRICS}} for row in rows]
    csv_write(folder / "layouts.csv", flat)
    csv_write(folder / "seed_means.csv", result["seed_means"])
    csv_write(folder / "policy_groups.csv", result["policy_groups"])
    csv_write(folder / "matched_deltas.csv", result["matched_deltas"])
    missing = [task for task, row in zip(plan["tasks"], rows) if row["state"] == "missing_final"]
    atomic_json(folder / "missing_final_tasks.json", dict(tasks=missing,
        scheduler_check_required=True, automatically_submitted=False,
        note="Resume existing route controller only after checking its live jobs; never duplicate completed or active attempts"))
    lines = ["# Q-study development final routing", "", "48 fixed tasks: BC / legacy ×3 actor seeds ×rounds0,2 ×first4 fixed noises.",
        "Development evidence, not fresh confirmation. No per-policy candidate selection.", "", str(counts), "",
        "| Policy / round | Complete seeds | Complete layouts | DRC=0 | rWL um | WNS ns | TNS ns | Power W | Area um² | DRC |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for group in result["policy_groups"]:
        stats = group["statistics"]
        cells = [f"{stats[name]['mean']:.6g} ± {stats[name]['sd']:.4g}" for name in METRICS] if stats else ["—"] * 6
        lines.append(f"| {group['arm']} / {group['round']} | {group['complete_actor_seeds']}/3 | "
            f"{group['counts'].get('completed', 0)}/12 | {group['drc_clean']}/12 | " + " | ".join(cells) + " |")
    lines += ["", "Means first average four noises within each actor seed; ± is sample SD across the3actor-seed means.",
        "Nonzero DRC remains numerical evidence. Missing and physical failures stay in the48-task denominator; their PPA is unavailable.",
        "Native GRT overflow is not treated as final Table2 routed congestion. Historical candidate references, if added later, must remain separate."]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(report=str(folder), counts=counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bc", type=Path, required=True)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--run-prefix", default="go2o_qstdev1")
    p.add_argument("--concurrency", type=int, default=2)
    p.set_defaults(function=plan_command)
    for name, function in (("resolve", resolve_command), ("report", report_command)):
        p = sub.add_parser(name)
        p.add_argument("--experiment", type=Path, required=True)
        p.set_defaults(function=function)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use a bounded CPU Slurm job; this helper never submits production EDA")
    args.function(args)


if __name__ == "__main__":
    main()
