"""Read-only, paired native GRT evidence for uniform/quality-rank flow matching.

No model inference, training, EDA, selection, or replay. Eight shared generation
noises are nested within each of three independent, matched training seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

# Preserve the repository analysis package before upstream baseline imports.
from analysis.placement_metrics import legality  # noqa: F401
from fql import qstudy_actor_report as native
from fql import qstudy_flow_baseline_report as baseline

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("uniform", "quality_rank")
SEEDS = (0, 1, 2)
PROTOCOL = "docs/qstudy_quality_weighted_flow_protocol.md"
PROTOCOL_SHA = "670c26b1d3ed2ac1738df8e4bb73de28390da9d5137d4b3116857fa75099ebb3"
SOURCE_NAMES = ("analysis/qstudy_quality_weighted_results.py",
    "analysis/qstudy_quality_weighted_results.slurm", PROTOCOL,
    "fql/qstudy_quality_weighted_eval.py", "fql/qstudy_quality_weighted_flow.py",
    "fql/qstudy_flow_eval.py", "fql/qstudy_flow_baseline_report.py",
    "fql/qstudy_actor_report.py", "fql/qstudy_backend.py", "fql/graph_o2o_backend.py")


def artifact(path):
    return baseline.fingerprint(path)


def read(path):
    return native.read(path)


def checked_completion(experiment, declaration, physical, tasks):
    """Bind the immutable whole24 export marker, not an active controller file."""
    value = read(experiment/"export_completed.json")
    expected = dict(status="completed", count=24,
        declaration_sha256=native.sha(experiment/"declaration.json"),
        config_sha256=physical["config_sha256"],
        dispatch_sha256=native.sha(experiment/"dispatch.json"),
        seed_tasks={str(seed): artifact(tasks[seed]) for seed in SEEDS},
        backend_calls=0, production_native_submitted=False, evaluation_never_replayed=True)
    if value != expected:
        raise ValueError("Changed full24 quality-FM export completion")
    return artifact(experiment/"export_completed.json")


def metric_row(request, sample, tasks, context):
    item = baseline.inspect_outcome(request, sample, tasks, context)
    if item["state"] == "success":
        # Negative slack is valid; negative wirelength/overflow/power/area is not.
        if any(item["metrics"][k] < 0 for k in native.METRICS if k not in ("WNS_ns", "TNS_ns")):
            item.update(state="invalid", metrics=None, integrity_error="Negative nonnegative physical metric")
    item.update(macro_hpwl=request.get("macro_hpwl"),
        raw_macro_hpwl=request.get("raw_macro_hpwl"),
        source_training_artifacts=request.get("source_training_artifacts"))
    return item


def stream(arm, seed, rows, tasks):
    if len(rows) != 8 or [r["index"] for r in rows] != list(range(8)):
        raise ValueError("Each stream must retain all eight declared requests")
    if any(r["seed"] != seed for r in rows) or len({r["noise_sha256"] for r in rows}) != 8:
        raise ValueError("Changed training seed or distinct-noise identities")
    return dict(arm=arm, seed=seed, round=0, tasks=artifact(tasks),
        actual_noise_by_index={str(r["index"]): r["noise_sha256"] for r in rows},
        unique_executed_actions=len({r["action_sha256"] for r in rows}),
        duplicate_execution_count=8-len({r["action_sha256"] for r in rows}),
        integrity_errors=[r["integrity_error"] for r in rows if r["state"] == "invalid"],
        rows=rows, **baseline.summarize(rows))


def collect_arm(experiment, expected_arm):
    from fql import qstudy_quality_weighted_eval as exporter
    from fql import qstudy_backend as backend
    from fql.graph_o2o_backend import load_requests
    cfg, physical, pairs, plan, trained = exporter.settings(experiment)
    if cfg["arm"] != expected_arm:
        raise ValueError("Experiment arm differs from requested report role")
    declaration = exporter.checked_declaration(experiment, cfg, physical, pairs, plan, trained)
    tasks = {s: exporter.checked_seed(experiment, cfg, physical, declaration, s) for s in SEEDS}
    completion = checked_completion(experiment, declaration, physical, tasks)
    context = backend.context(experiment)
    streams = []
    for seed in SEEDS:
        requests, actual_tasks = load_requests(experiment, seed, 0)
        if actual_tasks != tasks[seed]:
            raise ValueError("Changed exact per-seed tasks path")
        rows = []
        for request in requests:
            pair = next(p for p in pairs if p["seed"] == seed and p["index"] == request["index"])
            sample = exporter.verify_row(request, cfg, physical, declaration, pair, experiment)
            rows.append(metric_row(request, sample, actual_tasks, context))
        streams.append(stream(expected_arm, seed, rows, actual_tasks))
    return dict(arm=expected_arm, experiment=str(experiment), config=artifact(experiment/"config.json"),
        declaration=artifact(experiment/"declaration.json"), completion=completion,
        dispatch=artifact(experiment/"dispatch.json"), physical_context=physical,
        physical_binding=context[2], training_manifest=cfg["training_manifest"],
        all_paired_training=trained, source_pairs=pairs, streams=streams)


def collect_common(experiment, control):
    """Optional unchanged common-flow reference; strict existing export checks only."""
    from fql import qstudy_flow_eval as original
    from fql import qstudy_backend as backend
    from fql.graph_o2o_backend import load_requests
    cfg, physical, controls = original.settings(experiment)
    declaration = read(experiment/"declaration.json")
    pairs = original.source_pairs(cfg, controls)
    if (declaration["source_kind"] != original.SOURCE_KIND or declaration["source_sampler"] != original.SAMPLER or
            declaration["pairs"] != pairs or pairs != control["source_pairs"] or
            declaration["physical_context"] != physical or declaration["expected_requests"] != 24):
        raise ValueError("Common flow source/noise/sampler declaration differs")
    for name, digest in declaration["source_sha256"].items():
        baseline.checked_file(dict(path=str(ROOT/name), sha256=digest))
        baseline.checked_file(dict(path=str(experiment/"flow_eval_sources"/name), sha256=digest))
    common_parent = read(control["declaration"]["path"])["source_flow_eval"]
    if (common_parent["config"] != artifact(experiment/"config.json") or
            common_parent["declaration"] != artifact(experiment/"declaration.json") or
            {k:v for k,v in physical.items() if k != "config_sha256"} !=
            {k:v for k,v in control["physical_context"].items() if k != "config_sha256"}):
        raise ValueError("Common reference is not the exact paired initialization/evaluator")
    done = read(experiment/"export_completed.json")
    if (done.get("status") != "completed" or done.get("count") != 24 or
            done.get("config_sha256") != native.sha(experiment/"config.json") or
            done.get("declaration_sha256") != native.sha(experiment/"declaration.json") or
            done.get("dispatch_sha256") != native.sha(experiment/"dispatch.json")):
        raise ValueError("Changed completed common-flow export")
    context, streams = backend.context(experiment), []
    for seed in SEEDS:
        requests, tasks = load_requests(experiment, seed, 0)
        marker = read(tasks.with_name("completed.json"))
        if (marker.get("status") != "completed" or marker.get("evaluation_never_replayed") is not True or
                marker.get("flow_eval_declaration_sha256") != original.object_sha(declaration) or
                done["seed_tasks_sha256"][str(seed)] != native.sha(tasks)):
            raise ValueError("Changed common per-seed completion")
        artifact_common = control["all_paired_training"]["uniform"][str(seed)]["common"]
        if artifact_common != dict(path=cfg["common_checkpoints"][str(seed)],
                sha256=cfg["common_checkpoint_sha256"][str(seed)]):
            raise ValueError("Common reference checkpoint differs from paired training initialization")
        rows = []
        for request in requests:
            pair = next(p for p in pairs if p["seed"] == seed and p["index"] == request["index"])
            sample = original.verify_row(request, argparse.Namespace(seed=seed, round=0), cfg, physical, declaration, pair)
            rows.append(metric_row(request, sample, tasks, context))
        streams.append(stream("common_flow", seed, rows, tasks))
    return dict(arm="common_flow", experiment=str(experiment), config=artifact(experiment/"config.json"),
        declaration=artifact(experiment/"declaration.json"), completion=artifact(experiment/"export_completed.json"),
        dispatch=artifact(experiment/"dispatch.json"), physical_context=physical,
        physical_binding=context[2], source_pairs=pairs, streams=streams,
        reference_scope="Additional source adaptation/1000-update diagnostic; not the primary quality-weighting contrast")


def matched_context(results):
    if len(results) not in (2, 3) or [r["arm"] for r in results[:2]] != list(ARMS) or \
            (len(results) == 3 and results[2]["arm"] != "common_flow"):
        raise ValueError("Require ordered uniform/quality_rank and optional common_flow reference")
    reference = results[0]
    for result in results:
        if (result["source_pairs"] != reference["source_pairs"] or
                {k:v for k,v in result["physical_context"].items() if k != "config_sha256"} !=
                {k:v for k,v in reference["physical_context"].items() if k != "config_sha256"}):
            raise ValueError("Arms do not share exact sources/noise/native physical context")
    if results[0]["training_manifest"] != results[1]["training_manifest"] or \
            results[0]["all_paired_training"] != results[1]["all_paired_training"]:
        raise ValueError("Two FM arms do not bind the same matched training matrix")
    noises = [s["actual_noise_by_index"] for r in results for s in r["streams"]]
    if not noises or any(n != noises[0] for n in noises):
        raise ValueError("All arms/seeds require the exact same eight actual noise arrays")


def aggregate(result):
    value = baseline.aggregate(result["streams"], "current")
    value.pop("baseline")
    value["arm"] = result["arm"]
    value["uncertainty_unit"] = "Sample SD of three equally weighted training-seed means; eight actual matched generation noises nested per seed, no confidence interval"
    return value


def paired(left, right):
    """left minus right. Every declared pair remains, even without numeric PPA."""
    seed_rows, pairs = [], []
    for seed in SEEDS:
        a = next(s for s in left["streams"] if s["seed"] == seed)
        b = next(s for s in right["streams"] if s["seed"] == seed)
        if a["actual_noise_by_index"] != b["actual_noise_by_index"]:
            raise ValueError("Paired physical differences require identical actual noise arrays")
        local = []
        for x, y in zip(a["rows"], b["rows"]):
            if (x["index"], x["noise_sha256"]) != (y["index"], y["noise_sha256"]):
                raise ValueError("Pair order/actual noise differs")
            both = x["state"] == y["state"] == "success"
            delta = {k: x["metrics"][k]-y["metrics"][k] for k in native.METRICS} if both else None
            local.append(dict(seed=seed, index=x["index"], noise_sha256=x["noise_sha256"],
                left_state=x["state"], right_state=y["state"], both_success=both, deltas=delta))
        pairs.extend(local)
        seed_rows.append(dict(seed=seed, planned_pairs=8, both_success=sum(r["both_success"] for r in local),
            metrics={k:native.describe([r["deltas"][k] for r in local if r["both_success"]]) for k in native.METRICS}))
    complete = all(s["fully_accounted"] and not s["integrity_errors"] for r in (left,right) for s in r["streams"])
    metrics = {}
    for key in native.METRICS:
        values = [r["metrics"][key]["mean"] for r in seed_rows]
        metrics[key] = native.describe(values) if complete and all(v is not None for v in values) else dict(n=sum(v is not None for v in values), mean=None, sd=None)
    return dict(left=left["arm"], right=right["arm"], sign="left minus right in native units; positive slack delta is better",
        complete_accounting=complete, planned_pairs=24, both_success=sum(r["both_success"] for r in pairs),
        success_conditional=any(not r["both_success"] for r in pairs),
        seed_means=seed_rows, three_seed_paired_deltas=metrics, pairs=pairs)


def protocol_gates(weighted, uniform):
    """Operational development thresholds only; never selects/promotes a model."""
    wa, ua = aggregate(weighted), aggregate(uniform)
    accounted = wa["aggregate_eligible"] and ua["aggregate_eligible"]
    fail = lambda x: x["request_accounting"]["physical_failure"]+x["request_accounting"]["geometry_failure"]
    no_added = fail(wa) <= fail(ua) if accounted else None
    # Failed layouts have no fabricated PPA. Full-policy numeric gates cannot be
    # decided from a success-filtered subset, even when both arms failed equally.
    numeric = accounted and wa["request_accounting"]["success"] == ua["request_accounting"]["success"] == 24
    gates = dict(no_added_physical_failures=dict(passed=no_added,
        weighted=fail(wa), uniform=fail(ua)), primary_WL=None, all_three_seed_WL_direction=None,
        WNS=None, TNS=None, horizontal_overflow_sum=None, vertical_overflow_sum=None, Power_W=None, StdCellArea_um2=None)
    if numeric:
        w = {k:v["mean"] for k,v in wa["metrics"].items()}
        u = {k:v["mean"] for k,v in ua["metrics"].items()}
        reduction = (u["rWL_um"]-w["rWL_um"])/u["rWL_um"] if u["rWL_um"] > 0 else None
        gates["primary_WL"] = dict(required_relative_reduction=.01, observed_relative_reduction=reduction,
            passed=reduction >= .01 if reduction is not None else None)
        directions = [next(s for s in weighted["streams"] if s["seed"] == seed)["successful_native_metric_means"]["rWL_um"] <
            next(s for s in uniform["streams"] if s["seed"] == seed)["successful_native_metric_means"]["rWL_um"] for seed in SEEDS]
        gates["all_three_seed_WL_direction"] = dict(seed_directions=directions, passed=all(directions))
        gates["WNS"] = dict(deterioration_ns=u["WNS_ns"]-w["WNS_ns"], tolerance_ns=.02,
            passed=u["WNS_ns"]-w["WNS_ns"] <= .02)
        # Violation magnitude is averaged per request, not inferred by negating
        # a possibly positive/negative mixed mean slack.
        violation = lambda result: native.mean([native.mean([max(-r["metrics"]["TNS_ns"],0.) for r in s["rows"]]) for s in result["streams"]])
        uv, wv = violation(uniform), violation(weighted)
        gates["TNS"] = dict(deterioration_violation_ns=wv-uv, tolerance_ns=max(1.,.05*uv), passed=wv-uv <= max(1.,.05*uv))
        for key in ("horizontal_overflow_sum", "vertical_overflow_sum"):
            tolerance = max(1.,.05*u[key])
            gates[key] = dict(deterioration=w[key]-u[key], tolerance=tolerance, passed=w[key]-u[key] <= tolerance)
        for key in ("Power_W", "StdCellArea_um2"):
            change = (w[key]-u[key])/u[key] if u[key] > 0 else None
            gates[key] = dict(relative_deterioration=change, tolerance=.01, passed=change <= .01 if change is not None else None)
    decisions = [v["passed"] if v is not None else None for v in gates.values()]
    status = "fails_declared_development_gate" if False in decisions else "passes_declared_development_gate" if all(v is True for v in decisions) else "inconclusive"
    return dict(status=status, gates=gates, full48_success_numeric_gates=numeric,
        interpretation="Engineering thresholds, not p-values or a stable-advantage claim. Success-filtered PPA never establishes full-policy gates. No automatic DRT submission, checkpoint selection or promotion.")


def write_report(output, results):
    if native.sha(ROOT/PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Frozen development protocol changed")
    matched_context(results)
    aggregates = [aggregate(r) for r in results]
    comparisons = [paired(results[1],results[0])]
    if len(results) == 3:
        comparisons += [paired(r,results[2]) for r in results[:2]]
    streams = [s for r in results for s in r["streams"]]
    status = "integrity_invalid" if any(s["integrity_errors"] for s in streams) else \
        "complete" if all(s["fully_accounted"] for s in streams) else "partial"
    for r in results:
        protected_paths = [ROOT, Path(r["experiment"]).resolve()]
        if r.get("training_manifest"):
            protected_paths.append(Path(r["training_manifest"]["path"]).resolve().parent)
        for protected in protected_paths:
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError("Report output overlaps native/export/training/source experiment")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+"_job"+os.environ["SLURM_JOB_ID"]
    snapshot = output/stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    value = dict(schema="qstudy_quality_weighted_native_results_v1", status=status,
        protocol=artifact(ROOT/PROTOCOL), stage="global_route_only", development_only=True,
        primary_contrast="quality_rank minus uniform", metric_names=list(native.METRICS),
        no_model_inference=True, new_backend_calls=0, no_replay=True, selected_winner=None,
        uncertainty="Three equally weighted training-seed means/sample SD; 8 generation noises per seed are not 24 independent training repetitions",
        PPA_scope="Seven native GRT metrics; no final DRT/DRC inferred. PPA means conditional on success; all failure/missing denominators retained.",
        experiments=results, aggregates=aggregates, comparisons=comparisons,
        development_gates=protocol_gates(results[1],results[0]),
        source_sha256={n:native.sha(ROOT/n) for n in SOURCE_NAMES})
    (snapshot/"report.json").write_text(json.dumps(native.clean(value),indent=2,allow_nan=False)+"\n")
    (snapshot/"reporter_source.py").write_bytes(Path(__file__).read_bytes())
    native.write_csv(snapshot/"streams.csv",[{k:v for k,v in s.items() if k != "rows"} for s in streams])
    native.write_csv(snapshot/"aggregates.csv",aggregates)
    native.write_csv(snapshot/"requests.csv",[dict(row, arm=s["arm"]) for s in streams for row in s["rows"]])
    native.write_csv(snapshot/"paired_deltas.csv",[dict(row,left=c["left"],right=c["right"]) for c in comparisons for row in c["pairs"]])
    native.write_csv(snapshot/"paired_seed_means.csv",[dict(row,left=c["left"],right=c["right"]) for c in comparisons for row in c["seed_means"]])
    lines=["# Quality-weighted FM development results","",f"Status: **{status}**. Gate: **{value['development_gates']['status']}**.","",
        "Means/SD below use three training-seed means, each from eight fixed generation noises. Failures/missing are retained; success-conditional metrics do not establish full-policy gates.","",
        "| Arm | Success/planned | Missing+invalid | GRT WL (um), mean ± training-seed SD |","| --- | ---: | ---: | ---: |"]
    for a in aggregates:
        count=a["request_accounting"]; m=(a.get("metrics") or {}).get("rWL_um",{})
        lines.append(f"| {a['arm']} | {count['success']}/{count['planned']} | {count['missing']+count['invalid']+count['not_exported']} | {m.get('mean')} ± {m.get('sd')} |")
    lines += ["","All seven native metrics, same-noise paired differences, seed summaries, exact source/training/request/native hashes and operational gates are in JSON/CSV. No DRT, fresh confirmation or automatic promotion is implied."]
    (snapshot/"report.md").write_text("\n".join(lines)+"\n")
    complete=dict(status="completed",report_status=status,backend_calls=0,model_inference_calls=0,
        artifacts={p.name:native.sha(p) for p in snapshot.iterdir() if p.is_file()})
    (snapshot/"completed.json").write_text(json.dumps(complete,indent=2)+"\n")
    output.mkdir(parents=True, exist_ok=True)
    latest=output/(".latest_"+os.environ["SLURM_JOB_ID"]+".tmp")
    latest.write_text(json.dumps(dict(snapshot=str(snapshot),report=artifact(snapshot/"report.json"),completion=artifact(snapshot/"completed.json")),indent=2)+"\n")
    latest.replace(output/"latest.json")
    return snapshot,value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform",type=Path,required=True)
    parser.add_argument("--quality-rank",type=Path,required=True)
    parser.add_argument("--common",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("All numerical/native evidence reporting requires CPU Slurm")
    results=[collect_arm(args.uniform.resolve(),"uniform"),collect_arm(args.quality_rank.resolve(),"quality_rank")]
    if args.common:
        results.append(collect_common(args.common.resolve(),results[0]))
    snapshot,value=write_report(args.output.resolve(),results)
    print(json.dumps(dict(status=value["status"],snapshot=str(snapshot),development_gate=value["development_gates"]["status"],new_backend_calls=0)),flush=True)


if __name__ == "__main__":
    main()
