"""Read-only round-zero source comparison with an audited reused BC control.

Native outcomes are revalidated by the frozen actor reporter. Every arm uses
one O64 TRAIN WL scale; source-specific Q predictions/utilities are not compared.
No fitting, inference, physical evaluation, replay insertion, or selection occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np

from . import qstudy_actor_report as native

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_SHA = "7a325459aeabe9ee9b9b0995a2b7b2cc3cc867a12e7de41cdff1dc2f3e3c8823"
ARMS = ("BC", "O64", "T64")
SEEDS = (0, 1, 2)
COUNT = 8
CONTRASTS = (("O64", "BC"), ("T64", "BC"), ("T64", "O64"))
SETTINGS = ("design", "cache_sha256", "seeds", "alpha", "offline_actor_steps", "batch_size",
            "flow_steps", "actor_lr", "online_flow_update", "offline_critic_update",
            "evaluation_samples", "evaluation_noise_seed", "evaluation_purpose",
            "backend_threads", "backend_storage", "common_checkpoint_sha256")


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=native.sha(path))


def verify_certificate(path, plans):
    """Bind the specific no-new-training audit to all current declarations."""
    path = Path(path).resolve()
    if native.sha(path) != CERTIFICATE_SHA:
        raise ValueError("Unrecognized BC effective-control certificate SHA")
    cert = native.read(path)
    if (cert.get("schema") != "qstudy_bc_effective_control_certificate_v1"
            or cert.get("status") != "verified"
            or cert.get("exact_effective_update_control_eligible") is not True
            or cert.get("no_new_training") is not True
            or cert.get("independently_retrained_bitwise_equivalence_demonstrated") is not False):
        raise ValueError("Certificate does not admit the declared reused effective control")
    for field in ("actor_source", "implementation", "cache"):
        ref = cert[field]
        if native.sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Certificate evidence changed: " + field)
    if native.sha(ROOT / "fql/qstudy_actor.py") != cert["actor_source"]["sha256"]:
        raise ValueError("Effective-control training implementation changed")
    if set(plans) != set(ARMS):
        raise ValueError("Exactly BC, O64 and T64 declarations are required")
    for arm, (experiment, cfg) in plans.items():
        evidence = cert["configs"][arm]
        config_path = Path(experiment).resolve() / "config.json"
        if config_path != Path(evidence["path"]).resolve() or native.sha(config_path) != evidence["sha256"]:
            raise ValueError("Certificate config binding changed: " + arm)
        for key in SETTINGS:
            if cfg.get(key) != cert["effective_settings"].get(key):
                raise ValueError("Effective-control setting differs: " + arm + "/" + key)
        if cfg["seeds"] != list(SEEDS) or cfg["evaluation_samples"] != COUNT:
            raise ValueError("This report requires the declared three-seed/eight-noise plan")
        if (cfg.get("weights") != [1, 0, 0, 0, 0] or cfg.get("critic_variant") != "legacy"
                or cfg.get("physical_failure_reward") != -4 or cfg.get("terminal_mask") != 0):
            raise ValueError("Unexpected source-contrast objective/backend contract")
        if arm != "BC" and cfg.get("online_rounds") != 0:
            raise ValueError("Source comparison is strictly offline round zero")
        if (arm == "BC" and cfg.get("q_weight") != 0) or (arm != "BC" and cfg.get("q_weight", 0) <= 0):
            raise ValueError("BC/treatment Q intervention differs from declaration")
    if set(r["seed"] for r in cert["seeds"]) != set(SEEDS):
        raise ValueError("Certificate independent seeds changed")
    for evidence in cert["seeds"]:
        seed = str(evidence["seed"])
        common = evidence["common"]
        if native.sha(common["path"]) != common["sha256"]:
            raise ValueError("Certified common initialization changed")
        for arm, (_, cfg) in plans.items():
            if (Path(cfg["common_checkpoints"][seed]).resolve() != Path(common["path"]).resolve()
                    or cfg["common_checkpoint_sha256"][seed] != common["sha256"]):
                raise ValueError("Common initialization path/hash differs: " + arm)
            critic = evidence["zero_q_critic_probes"][arm]["checkpoint"]
            if (native.sha(critic["path"]) != critic["sha256"]
                    or cfg["critic_checkpoint_sha256"][seed] != critic["sha256"]
                    or Path(cfg["critic_checkpoints"][seed]).resolve() != Path(critic["path"]).resolve()):
                raise ValueError("Certified finite source critic evidence changed: " + arm)
    return cert


def common_utility(row, wl_scale):
    """Measured, unweighted native WL utility in one common TRAIN scale."""
    if not math.isfinite(wl_scale) or wl_scale <= 0:
        raise ValueError("Invalid O64 reference WL scale")
    if row["state"] == "success":
        value = row["metrics"]["rWL_um"]
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid successful native wirelength")
        return -float(value) / wl_scale
    return -4. if row["state"] in ("physical_failure", "geometry_failure") else None


def evaluation_stream(record, arm, wl_scale, certificate):
    rows = [dict(r) for r in record["rows"] if r.get("role") == "evaluation"]
    errors = list(record.get("errors", []))
    training = record.get("training", {})
    if not training.get("completed"):
        # Not-yet-trained is unresolved; changed completed evidence is invalid.
        if training.get("integrity_error"):
            errors.append("Training integrity: " + training["integrity_error"])
    if arm == "BC" and training.get("completed"):
        proof = next(r for r in certificate["seeds"] if r["seed"] == record["seed"])
        for field in ("checkpoint_sha256", "completion_sha256", "binding_sha256", "steps"):
            if training.get(field) != proof["completed_training"][field]:
                errors.append("Reused BC training differs from certificate: " + field)
    indices = [r["index"] for r in rows]
    noises = [r.get("noise_sha256") for r in rows]
    if rows and (len(rows) != COUNT or set(indices) != set(range(COUNT)) or len(set(indices)) != COUNT):
        errors.append("Evaluation export differs from fixed eight-index plan")
    if rows and (any(not isinstance(n, str) or not n for n in noises) or len(set(noises)) != len(noises)):
        errors.append("Evaluation noise hashes are absent or duplicated")
    for row in rows:
        if row["state"] == "success":
            metrics = row.get("metrics") or {}
            if any(type(metrics.get(k)) not in (int, float) or not math.isfinite(metrics[k]) for k in native.METRICS):
                row.update(state="invalid", metrics=None, integrity_error="Seven finite native metrics required")
        if row["state"] != "success":
            # Keep no invented PPA for missing/invalid/confirmed failure.
            row["metrics"] = None
        row["common_o64_train_wl_utility"] = common_utility(row, wl_scale)
        row.pop("measured_utility", None)
        row.pop("predicted_q", None)
    counts = {name: sum(r["state"] == name for r in rows)
              for name in ("success", "geometry_failure", "physical_failure", "missing", "invalid")}
    accounted = sum(counts[k] for k in native.ADMITTED)
    complete = (not errors and training.get("completed") is True and len(rows) == COUNT and accounted == COUNT)
    successes = [r for r in rows if r["state"] == "success"]
    return dict(arm=arm, seed=record["seed"], round=0, experiment=record["experiment"], errors=errors,
        training=training, tasks_sha256=record.get("tasks_sha256"),
        export_completion_sha256=record.get("export_completion_sha256"),
        planned=COUNT, exported=len(rows), not_exported=max(COUNT-len(rows), 0), **counts,
        fully_accounted=bool(complete), integrity_invalid=bool(errors or counts["invalid"]),
        actual_noise_by_index={str(r["index"]): r["noise_sha256"] for r in rows},
        conditional_PPA=bool(counts["physical_failure"]+counts["geometry_failure"]),
        successful_native_metric_means={k: native.mean([r["metrics"][k] for r in successes]) for k in native.METRICS},
        common_utility_mean=native.mean([r["common_o64_train_wl_utility"] for r in rows]) if complete else None,
        raw_to_legalizer_rms=native.describe([r.get("raw_to_legalizer_rms") for r in rows]), rows=rows)


def complete_statistics(values, expected):
    """Never silently reduce the independent-seed denominator for missing PPA."""
    result = native.describe(values)
    return dict(result, eligible=result["n"] == expected) if result["n"] == expected else dict(
        n=result["n"], mean=None, sd=None, eligible=False)


def aggregate(streams):
    arm_rows, pairs, contrasts = [], [], []
    for arm in ARMS:
        chosen = [s for s in streams if s["arm"] == arm]
        eligible = (len(chosen) == len(SEEDS) and {s["seed"] for s in chosen} == set(SEEDS)
                    and all(s["fully_accounted"] for s in chosen)
                    and all(s["actual_noise_by_index"] == chosen[0]["actual_noise_by_index"] for s in chosen))
        arm_rows.append(dict(arm=arm, planned_seeds=list(SEEDS),
            completed_seeds=[s["seed"] for s in chosen if s["fully_accounted"]],
            aggregate_eligible=bool(eligible), independent_seed_n=3 if eligible else 0,
            request_accounting={key:sum(s[key] for s in chosen) for key in
                ("planned", "exported", "not_exported", "success", "geometry_failure", "physical_failure", "missing", "invalid")},
            conditional_PPA=any(s["conditional_PPA"] for s in chosen),
            metrics={k: complete_statistics([s["successful_native_metric_means"][k] for s in chosen], 3)
                     for k in native.METRICS} if eligible else None,
            common_utility=native.describe([s["common_utility_mean"] for s in chosen]) if eligible else None))
    lookup = {(s["arm"], s["seed"]): s for s in streams}
    for target, baseline in CONTRASTS:
        per_seed = []
        for seed in SEEDS:
            a, b = lookup[(target, seed)], lookup[(baseline, seed)]
            same_plan = bool(len(a["rows"]) == len(b["rows"]) == COUNT
                             and a["actual_noise_by_index"] == b["actual_noise_by_index"])
            ready = same_plan and a["fully_accounted"] and b["fully_accounted"]
            local = []
            aa, bb = ({r["index"]: r for r in side["rows"]} for side in (a, b))
            for index in sorted(set(aa) & set(bb)):
                ar, br = aa[index], bb[index]
                matched = ar["noise_sha256"] == br["noise_sha256"]
                admitted = (matched and not a["integrity_invalid"] and not b["integrity_invalid"]
                            and ar["state"] in native.ADMITTED and br["state"] in native.ADMITTED)
                ok = admitted and ar["state"] == br["state"] == "success"
                row = dict(target=target, baseline=baseline, seed=seed, index=index,
                    actual_noise_sha256=ar["noise_sha256"] if matched else None,
                    same_actual_noise=matched, target_state=ar["state"], baseline_state=br["state"],
                    pair_admitted=admitted, both_successful=ok,
                    common_utility_delta=ar["common_o64_train_wl_utility"]-br["common_o64_train_wl_utility"] if admitted else None,
                    metric_delta_target_minus_baseline={k:ar["metrics"][k]-br["metrics"][k] for k in native.METRICS} if ok else None,
                    target_request_sha256=ar.get("request_sha256"), baseline_request_sha256=br.get("request_sha256"),
                    target_outcome_sha256=ar.get("outcome_sha256"), baseline_outcome_sha256=br.get("outcome_sha256"))
                local.append(row); pairs.append(row)
            per_seed.append(dict(seed=seed, fully_accounted=bool(ready), same_actual_noise_plan=same_plan,
                planned_pairs=COUNT, admitted_pairs=sum(r["pair_admitted"] for r in local),
                successful_pairs=sum(r["both_successful"] for r in local),
                conditional_PPA=any(not r["both_successful"] for r in local),
                common_utility_delta=native.mean([r["common_utility_delta"] for r in local]) if ready else None,
                metric_deltas={k:native.mean([r["metric_delta_target_minus_baseline"][k] for r in local
                    if r["both_successful"]]) for k in native.METRICS} if ready else None))
        # Cross-seed noise matching is part of the fixed shared evaluation plan.
        selected = [lookup[(arm, seed)] for arm in (target, baseline) for seed in SEEDS]
        shared = all(s["actual_noise_by_index"] == selected[0]["actual_noise_by_index"] for s in selected)
        eligible = shared and all(r["fully_accounted"] for r in per_seed)
        contrasts.append(dict(target=target, baseline=baseline, aggregate_eligible=eligible,
            controlled_effective_comparison_eligible=eligible, common_noise_plan_across_seeds=shared,
            planned_seeds=list(SEEDS), independent_seed_n=3 if eligible else 0,
            conditional_PPA=any(r["conditional_PPA"] for r in per_seed),
            common_utility_delta=native.describe([r["common_utility_delta"] for r in per_seed]) if eligible else None,
            metric_deltas={k:complete_statistics([r["metric_deltas"][k] for r in per_seed], 3)
                           for k in native.METRICS} if eligible else None, per_seed=per_seed))
    return arm_rows, pairs, contrasts


def run(args):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Read-only array/native validation requires a bounded CPU Slurm job")
    # Import local backend before anything that might add upstream analysis.py.
    from .qstudy_backend import context
    plans = {arm:(Path(getattr(args, arm.lower())).resolve(), None) for arm in ARMS}
    plans = {arm:(path, native.read(path/"config.json")) for arm,(path,_) in plans.items()}
    if len({p for p,_ in plans.values()}) != 3:
        raise ValueError("Duplicate source/control experiment")
    certificate = verify_certificate(args.certificate, plans)
    scales, normalization = native.train_scales(plans["O64"][1])
    if normalization["observed_counts"][0] != 64:
        raise ValueError("Common utility requires all 64 measured O64 TRAIN WL labels")
    streams, bindings = [], {}
    for arm,(experiment,cfg) in plans.items():
        backend = context(experiment)
        bindings[arm] = dict(config=fingerprint(experiment/"config.json"),
            dataset=fingerprint(cfg["dataset_path"]), backend=backend[2])
        for seed in SEEDS:
            stream = native.inspect_round(experiment, cfg, seed, 0, scales, backend)
            streams.append(evaluation_stream(stream, arm, float(scales[0]), certificate))
    revisions = {b["backend"]["physical_flow_revision"] for b in bindings.values()}
    if len(revisions) != 1:
        raise ValueError("Source arms differ in frozen physical evaluator revision")
    arm_rows, pairs, contrasts = aggregate(streams)
    invalid = any(s["integrity_invalid"] for s in streams)
    status = "integrity_invalid" if invalid else "complete" if all(s["fully_accounted"] for s in streams) else "partial"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output/(stamp+"_job"+os.environ["SLURM_JOB_ID"])
    snapshot.mkdir(exist_ok=False)
    result = dict(schema="qstudy_source_paired_report_v1", status=status, created_utc=stamp,
        slurm_job_id=os.environ["SLURM_JOB_ID"], stage="global_route_only", development_only=True,
        production_heldout_test=False, no_training=True, no_model_inference=True, backend_calls=0,
        certificate=fingerprint(args.certificate), experiment_bindings=bindings,
        normalization=normalization, common_reference="O64 explicit initial TRAIN only; native rWL_um mean",
        utility="-native_rWL_um/O64_TRAIN_mean_rWL_um on success; confirmed physical/geometry failure=-4; missing/invalid has no utility",
        reused_BC_control=dict(effective_update_control_verified=True, new_GPU_repeat=False,
            bitwise_retraining_demonstrated=False, dataset_file_bindings_identical=False,
            certificate_scope=certificate["scope"], limitations=certificate["limitations"]),
        definitions=dict(metric_names=list(native.METRICS),
            PPA="Seven native GRT metrics, conditional on successful runs; paired PPA requires both successes. No DRT/DRC inferred",
            failure_accounting="All eight declared requests per seed; unresolved/invalid never counted as failure. Success PPA and all-request failure-penalized utility kept separate",
            seeds="Equal independent seed means; SD is sample SD across 3 training seeds, never pooled layout SD",
            deltas="Target minus baseline on same actual saved noise/index/seed. Positive common utility delta is better; rWL/overflow/power/area negative deltas are smaller; signed WNS/TNS positive deltas are closer to zero",
            discarded="Source-specific measured_utility and predicted_Q are omitted; no source-normalized utility comparison",
            selection="No winner or promotion, even on complete development data"),
        source_sha256={name:native.sha(ROOT/name) for name in (
            "fql/qstudy_source_paired_report.py", "fql/qstudy_actor_report.py", "fql/qstudy_actor.py",
            "fql/qstudy_backend.py", "fql/qstudy_data.py", "fql/graph_o2o_backend.py")},
        per_seed_streams=streams, independent_seed_aggregates=arm_rows,
        matched_pairs=pairs, matched_contrasts=contrasts, selected_winner=None)
    (snapshot/"report.json").write_text(json.dumps(native.clean(result), indent=2, allow_nan=False)+"\n")
    native.write_csv(snapshot/"streams.csv", [{k:v for k,v in s.items() if k not in ("rows","training")} for s in streams])
    native.write_csv(snapshot/"seed_aggregates.csv", arm_rows)
    native.write_csv(snapshot/"matched_pairs.csv", pairs)
    native.write_csv(snapshot/"matched_contrasts.csv", contrasts)
    lines = ["# Source actor GRT comparison", "", f"Status **{status}**; Slurm `{os.environ['SLURM_JOB_ID']}`.", "",
        "BC is reused under the independently audited effective-update certificate; it was not retrained on a new GPU. All arms use the same O64 TRAIN WL scale. This is development GRT, with no final DRT/DRC or winner claim.", "",
        "| Arm | Seed | Planned | Success | Geometry failure | Physical failure | Missing/unexported | Invalid | Complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for s in streams:
        lines.append(f"| {s['arm']} | {s['seed']} | 8 | {s['success']} | {s['geometry_failure']} | {s['physical_failure']} | {s['missing']+s['not_exported']} | {s['invalid']} | {s['fully_accounted']} |")
    lines += ["", "Seven native metrics, three matched contrasts and sample SD across independent seed means are in the JSON/CSV sidecars. Aggregate comparisons are withheld until every declared matched request is accounted for. Failed runs have no invented PPA; their fixed penalty enters only the separately reported all-request utility."]
    (snapshot/"report.md").write_text("\n".join(lines)+"\n")
    artifacts = {p.name:native.sha(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    (snapshot/"completed.json").write_text(json.dumps(dict(status="completed", report_status=status,
        artifacts=artifacts, slurm_job_id=os.environ["SLURM_JOB_ID"]), indent=2)+"\n")
    print(json.dumps(dict(status=status, snapshot=str(snapshot),
        complete_streams=sum(s["fully_accounted"] for s in streams),
        invalid_streams=sum(s["integrity_invalid"] for s in streams),
        complete_contrasts=sum(c["aggregate_eligible"] for c in contrasts))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--"+arm.lower(), type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
