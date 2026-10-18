"""Read-only, fixed-eight-candidate development selection diagnostics.

No model/EDA/scheduler is run. Native eight-layout pools are revalidated from
an explicitly hashed actor-report snapshot without reading learning curves.
Policy means and selected-versus-selected results are separate estimands.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .qstudy_actor_report import (ADMITTED, METRICS, check_training, controlled_pair,
    native_fingerprints, train_scales, utility)
from .qstudy_backend import context, checked_existing, checked_request

RULES = ("fixed_index0", "min_executed_macro_hpwl", "max_recorded_current_q",
         "oracle_min_grt_wl", "oracle_max_common_utility")


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values) > 1 else None)


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite " + name)
    return value


def common_utility(metrics, status, weights, scales):
    if status not in ADMITTED:
        raise ValueError("Missing/infrastructure states have no physical utility")
    return -4. if status != "success" else finite(utility(metrics,
        np.asarray(weights, dtype=float), np.asarray(scales, dtype=float)), "common utility")


def select_candidates(rows):
    """No outcome-based filtering for the three pre-backend selectors."""
    if len(rows) != 8 or {row["index"] for row in rows} != set(range(8)):
        raise ValueError("A selection pool must contain all eight declared indices")
    if any(row["state"] not in ADMITTED for row in rows):
        raise ValueError("Every outcome must be accounted before reporting selection")
    rows = sorted(rows, key=lambda row: row["index"])
    for row in rows:
        finite(row["macro_hpwl"], "executed HPWL")
        finite(row["predicted_q"], "recorded current Q")
        finite(row["common_utility"], "common utility")
    chosen = dict(fixed_index0=rows[0],
        min_executed_macro_hpwl=min(rows, key=lambda row: (row["macro_hpwl"], row["index"])),
        max_recorded_current_q=min(rows, key=lambda row: (-row["predicted_q"], row["index"])),
        oracle_min_grt_wl=min(rows, key=lambda row: (row["state"] != "success",
            row["metrics"]["rWL_um"] if row["state"] == "success" else 0., row["index"])),
        oracle_max_common_utility=min(rows, key=lambda row: (-row["common_utility"], row["index"])))
    return chosen


def audit_snapshot(path, expected_sha):
    path = Path(path).resolve()
    if sha(path) != expected_sha:
        raise ValueError("Explicit actor-report SHA changed")
    marker_path = path.with_name("completed.json")
    marker, report = read(marker_path), read(path)
    if marker["status"] != "completed" or marker["artifacts"][path.name] != expected_sha:
        raise ValueError("Actor report lacks a bound immutable completion marker")
    if report["stage"] != "global_route_only" or report["not_final_table2"] is not True:
        raise ValueError("Only audited GRT development reports are supported")
    for name, digest in report["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Audited report source changed: " + name)
    for exp, binding in report["experiment_bindings"].items():
        cfg = read(Path(exp) / "config.json")
        if sha(Path(exp) / "config.json") != binding["config_sha256"] or cfg != binding["config"]:
            raise ValueError("Audited actor configuration changed")
        if cfg["seeds"] != [0, 1, 2] or cfg["evaluation_samples"] != 8 or cfg["physical_failure_reward"] != -4:
            raise ValueError("Expected three actor seeds and eight fixed evaluation samples")
    identities = [(r["experiment"], r["seed"], r["round"]) for r in report["rounds"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate round identity in parent report")
    return report, dict(path=str(path), sha256=expected_sha, completion_sha256=sha(marker_path))


def audit_pool(record, binding, backend_context, weights, scales):
    if record["status"] != "complete" or record["errors"]:
        raise ValueError("Incomplete/invalid audited round; selection withheld")
    exp, seed, r = Path(record["experiment"]), record["seed"], record["round"]
    cfg = binding["config"]
    folder = exp / f"seed_{seed}/round_{r}"
    for kind in ("common", "critic"):
        if sha(cfg[kind+"_checkpoints"][str(seed)]) != cfg[kind+"_checkpoint_sha256"][str(seed)]:
            raise ValueError("Frozen initial actor/critic artifact changed")
    training = check_training(folder, cfg, seed, r, binding["config_sha256"], curve_snapshot={})
    for key in ("completed", "checkpoint_sha256", "completion_sha256", "binding_sha256", "steps"):
        if training.get(key) != record["training"].get(key) or training.get("completed") is not True:
            raise ValueError("Audited completed actor identity changed")
    requests, tasks = load_requests(exp, seed, r)
    marker = read(tasks.with_name("completed.json"))
    if (sha(tasks) != record["tasks_sha256"] or sha(tasks.with_name("completed.json")) != record["export_completion_sha256"]
            or marker.get("selection") != "all predetermined noises; no Q-selection"
            or marker.get("evaluation_never_replayed") is not True):
        raise ValueError("Original full-pool export binding changed")
    requested = [row for row in requests if row["role"] == "evaluation"]
    saved = [row for row in record["rows"] if row["role"] == "evaluation"]
    for rows in (requested, saved):
        if len(rows) != 8 or {row["index"] for row in rows} != set(range(8)):
            raise ValueError("Missing/duplicate candidate in fixed eight-layout pool")
    saved = {row["index"]: row for row in saved}
    bcfg, verification, backend_binding = backend_context
    if backend_binding != binding["backend_binding"]:
        raise ValueError("Audited dataset/reference/physical backend binding changed")
    rows = []
    for request in sorted(requested, key=lambda row: row["index"]):
        old = saved[request["index"]]
        if old["state"] not in ADMITTED:
            raise ValueError("Unaccounted candidate; no partial-pool selection")
        sample = checked_request(request, argparse.Namespace(seed=seed, round=r), bcfg, backend_binding)
        for field in ("arm", "noise_sha256", "noise_seed", "raw_action_sha256", "action_sha256", "checkpoint_sha256"):
            if request[field] != old[field]:
                raise ValueError("Candidate request identity changed: " + field)
        if (request["arm"] != cfg["arm"] or request["checkpoint_sha256"] != training["checkpoint_sha256"]
                or request["noise_seed"] != cfg["evaluation_noise_seed"]
                or request["training_feedback_eligible"] is not False
                or request.get("evaluation_purpose") != "development_validation"):
            raise ValueError("Candidate is not the declared completed development policy")
        for name, field in (("request.json", "request_sha256"), ("actions.npz", "actions_npz_sha256"),
                            ("outcome.json", "outcome_sha256")):
            if sha(sample / name) != old[field]:
                raise ValueError("Audited candidate artifact changed: " + name)
        out = read(sample / "outcome.json")
        for name in ("arm", "seed", "round", "role", "index", "graph", "training_feedback_eligible"):
            if out.get(name) != request.get(name):
                raise ValueError("Native outcome/request identity changed")
        if out.get("physical_run") and read(Path(out["physical_run"]) / "manifest.json").get("success"):
            if not (Path(out["physical_run"]) / "backend_macro_audit.json").is_file():
                raise ValueError("Missing saved geometry audit; read-only report cannot launch EDA")
        checked_existing(out, request, sample, tasks, backend_binding, bcfg, verification)
        native = native_fingerprints(out)
        if out["status"] != old["state"] or out["metrics"] != old["metrics"] or native != old["native"]:
            raise ValueError("Native evidence differs from audited snapshot")
        if out["status"] == "success":
            for name in METRICS:
                finite(out["metrics"][name], "native " + name)
        value = finite(np.asarray(request["predicted_q"], dtype=float).mean(), "recorded Q")
        if value != old["predicted_q"]:
            raise ValueError("Saved Q score changed")
        rows.append(dict(index=request["index"], state=out["status"], predicted_q=value,
            macro_hpwl=finite(request["macro_hpwl"], "executed macro HPWL"), noise_sha256=request["noise_sha256"],
            metrics=out["metrics"], common_utility=common_utility(out["metrics"], out["status"], weights, scales),
            sample_dir=str(sample), request_sha256=old["request_sha256"], outcome_sha256=old["outcome_sha256"],
            actions_npz_sha256=old["actions_npz_sha256"], native=native, failure_class=out.get("failure_class")))
    if len({row["noise_sha256"] for row in rows}) != 8:
        raise ValueError("Duplicate noise in an ostensibly eight-candidate pool")
    return rows


def metric_summary(rows):
    successful = [row for row in rows if row["state"] == "success"]
    return dict(counts=dict(Counter(row["state"] for row in rows)),
        utility=stats([row["common_utility"] for row in rows]),
        native_full={name: stats([row["metrics"][name] for row in rows]) for name in METRICS}
            if len(successful) == len(rows) and rows else None,
        successful_only_native={name: stats([row["metrics"][name] for row in successful]) for name in METRICS})


def pool_mean(rows):
    value = metric_summary(rows)
    return dict(counts=value["counts"], common_utility_mean=value["utility"]["mean"],
        native_full_mean={k:v["mean"] for k,v in value["native_full"].items()} if value["native_full"] else None,
        successful_only_native_mean={k:v["mean"] for k,v in value["successful_only_native"].items()})


def aggregate_pool_means(pools, bindings, endpoints):
    result = []
    for exp in bindings:
        for r in range(endpoints[exp]+1):
            rows = [p for p in pools if (p["experiment"], p["round"]) == (exp, r) and p["state"] == "complete"]
            full = len(rows) == 3 and {p["seed"] for p in rows} == {0, 1, 2}
            means = [p["policy_pool_mean"] for p in rows]
            result.append(dict(experiment=exp, round=r, complete_actor_seeds=len(rows),
                common_utility=stats([p["common_utility_mean"] for p in means]) if full else None,
                native_full={name:stats([p["native_full_mean"][name] for p in means]) for name in METRICS}
                    if full and all(p["native_full_mean"] is not None for p in means) else None))
    return result


def paired_selection(pools, selected, reference, bindings):
    indexed = {(row["experiment"], row["seed"], row["round"]): row for row in pools}
    picks = {(row["experiment"], row["seed"], row["round"], row["rule"]): row for row in selected}
    result = []
    for (exp, seed, r), pool in indexed.items():
        if exp == reference:
            continue
        base = indexed.get((reference, seed, r))
        if pool["state"] != "complete" or not base or base["state"] != "complete":
            continue
        if pool["noise_pool"] != base["noise_pool"]:
            raise ValueError("Compared candidate pools use different ordered eight-noise vectors")
        a, b = bindings[exp]["config"], bindings[reference]["config"]
        for field in ("design", "dataset_sha256", "cache_sha256"):
            if a[field] != b[field]:
                raise ValueError("Matched candidate-pool context differs: " + field)
        for rule in RULES:
            lhs, rhs = picks[(exp, seed, r, rule)], picks[(reference, seed, r, rule)]
            result.append(dict(experiment=exp, reference=reference, seed=seed, round=r, rule=rule,
                selected_index=lhs["index"], reference_selected_index=rhs["index"],
                common_utility_delta=lhs["common_utility"]-rhs["common_utility"],
                native_delta={name: lhs["metrics"][name]-rhs["metrics"][name] for name in METRICS}
                    if lhs["state"] == rhs["state"] == "success" else None,
                controlled_eligibility=controlled_pair(a, b, seed, r), matched_noise_pool=pool["noise_pool"]))
    return result


def aggregate_selected(selected, bindings, endpoints):
    groups = []
    for exp, binding in bindings.items():
        for r in range(endpoints[exp]+1):
            for rule in RULES:
                rows = [row for row in selected if (row["experiment"], row["round"], row["rule"]) == (exp, r, rule)]
                complete = {row["seed"] for row in rows} == {0, 1, 2} and len(rows) == 3
                groups.append(dict(experiment=exp, round=r, rule=rule, planned_actor_seeds=3,
                    complete_actor_seeds=len(rows), full_three_seed_summary=metric_summary(rows) if complete else None,
                    partial_seed_summary=metric_summary(rows)))
    return groups


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def run(args):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run bounded report audits through CPU Slurm")
    report, parent = audit_snapshot(args.report, args.report_sha256)
    bindings = report["experiment_bindings"]
    reference = str(Path(args.reference_experiment).resolve())
    if reference not in bindings:
        raise ValueError("Explicit reference experiment is absent from audited snapshot")
    reference_cfg = bindings[reference]["config"]
    scales, normalization = train_scales(reference_cfg)
    if normalization != bindings[reference]["normalization"]:
        raise ValueError("Explicit initial TRAIN-only reference normalization changed")
    weights = reference_cfg["weights"]
    contexts = {exp: context(exp) for exp in bindings}
    pools, selected = [], []
    for record in report["rounds"]:
        exp, seed, r = record["experiment"], record["seed"], record["round"]
        pool = dict(experiment=exp, arm=record["arm"], seed=seed, round=r, state="withheld", requested=8)
        try:
            rows = audit_pool(record, bindings[exp], contexts[exp], weights, scales)
            chosen = select_candidates(rows)
            calls = sorted({str(Path(row["native"]["physical_run"]).resolve()) for row in rows if row["native"].get("physical_run")})
            pool.update(state="complete", noise_pool=[row["noise_sha256"] for row in rows], rows=rows,
                policy_pool_mean=pool_mean(rows), existing_unique_native_calls=calls)
            for rule, row in chosen.items():
                selected.append(dict(experiment=exp, arm=record["arm"], seed=seed, round=r, rule=rule, **row,
                    pool_candidates=8, backend_assessments_available=8,
                    deployment_selected_evaluation_budget=8 if rule.startswith("oracle_") else 1,
                    critic_state="initial_frozen" if r == 0 or bindings[exp]["config"].get("q_weight", 1.) == 0 else "online_updated",
                    selector_note="current critic + generator system; BC critic frozen, FQL critic updated online; Q scores raw coordinates"
                        if rule == "max_recorded_current_q" else "fixed common rule within each policy's full eight-layout pool"))
        except Exception as error:
            pool["reason"] = str(error)
        pools.append(pool)
    endpoints = {exp: int(binding["report_round_endpoint"]) for exp, binding in bindings.items()}
    groups = aggregate_selected(selected, bindings, endpoints)
    pool_groups = aggregate_pool_means(pools, bindings, endpoints)
    pairs = paired_selection(pools, selected, reference, bindings)
    paired_groups = []
    for exp in bindings:
        if exp == reference:
            continue
        for r in range(min(endpoints[exp], endpoints[reference])+1):
            for rule in RULES:
                rows = [row for row in pairs if (row["experiment"], row["round"], row["rule"]) == (exp, r, rule)]
                full = len(rows) == 3 and {row["seed"] for row in rows} == {0, 1, 2}
                paired_groups.append(dict(experiment=exp, reference=reference, round=r, rule=rule, paired_seeds=len(rows),
                    controlled_eligible_seeds=sum(row["controlled_eligibility"]["eligible"] for row in rows),
                    all_paired_controls_eligible=full and all(row["controlled_eligibility"]["eligible"] for row in rows),
                    common_utility_delta=stats([row["common_utility_delta"] for row in rows]) if full else None,
                    native_delta={name: stats([row["native_delta"][name] for row in rows]) for name in METRICS}
                        if full and all(row["native_delta"] is not None for row in rows) else None))
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(args.report).resolve().parent) or any(output.is_relative_to(Path(exp)) for exp in bindings):
        raise ValueError("Do not write into immutable input reports or actor experiment trees")
    snapshot = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")+"_job"+os.environ["SLURM_JOB_ID"])
    snapshot.mkdir(parents=True, exist_ok=False)
    calls = {run for pool in pools for run in pool.get("existing_unique_native_calls", [])}
    result = dict(schema="qstudy_eight_candidate_selection_v1", parent=parent, stage="global_route_only",
        development_only=True, fresh_confirmation=False, new_backend_calls=0, new_model_inference=0,
        existing_unique_native_calls=len(calls), reference=dict(experiment=reference, normalization=normalization,
            weights=weights, scales=scales.tolist(), failure_utility=-4),
        report_source_sha256=sha(__file__), pools=pools, selections=selected, selected_aggregates=groups,
        policy_pool_mean_aggregates=pool_groups,
        selected_vs_selected_pairs=pairs, selected_vs_selected_aggregates=paired_groups,
        definitions=dict(pool="eight fixed noises, all outcomes accounted; no partial-pool selection",
            ties="lowest declared noise index", prebackend="fixed/Q/HPWL rules never remove candidates using backend outcomes; no fallback",
            wl_oracle="all successes precede failures; no numerical failed WL; all-failed pool chooses index0",
            utility_oracle="uses common initial TRAIN-only utility, including exactly-4 for confirmed failures",
            seeds="one selected result per independent actor seed; sampleSD across3seeds, never across8candidates or rounds",
            budget="all rules inspect same8-candidate pool; Q/HPWL selectors need one selected backend evaluation in deployment, oracle needs8; historical calls reused across rules, not multiplied",
            limits="current-Q selects a generator+critic system; oracle exploits measured feedback and is a fixed-pool reference, not deployable free selection; no policy-mean, fresh-generalization, DRT, DRC or final-congestion claim",
            initial_q="not reconstructed: FQL online pools need explicitly authorized frozen-initial-Q rescoring; no model inference here"))
    atomic_json(snapshot / "report.json", result)
    csv_write(snapshot / "selected.csv", [{**{k:v for k,v in row.items() if k not in ("metrics", "native")},
        **{name:(row["metrics"] or {}).get(name) for name in METRICS}} for row in selected])
    csv_write(snapshot / "policy_pool_means.csv", [{k:v for k,v in pool.items() if k != "rows"} for pool in pools])
    csv_write(snapshot / "policy_pool_mean_aggregates.csv", pool_groups)
    csv_write(snapshot / "selected_aggregates.csv", groups)
    csv_write(snapshot / "selected_vs_selected.csv", pairs)
    csv_write(snapshot / "selected_vs_selected_aggregates.csv", paired_groups)
    atomic_json(snapshot / "completed.json", dict(status="completed", report_sha256=sha(snapshot / "report.json"),
        complete_pools=sum(pool["state"] == "complete" for pool in pools), total_pools=len(pools), new_backend_calls=0))
    print(json.dumps(dict(snapshot=str(snapshot), complete_pools=sum(pool["state"] == "complete" for pool in pools), total_pools=len(pools))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--reference-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
