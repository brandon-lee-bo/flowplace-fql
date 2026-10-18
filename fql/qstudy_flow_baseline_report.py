"""Read-only native GRT reports for current24 and released-public8 flow baselines.

Current means three independently trained common flows on eight shared noises.
Public means one released checkpoint on eight generation noises, never three
training seeds. No model inference, fitting, EDA, replay, or selection occurs.
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
AUDIT_SHA = "c5c11fc5ebbd6e81b835c2508cd38e9dc0b64c6d6b6b834243b66313a4b619ac"
PLAN_SHA = dict(current="5d4dc80e9fe7c3552dce646037f2f6f4b9caaccf9f8bbda40a27b28967b9766c",
                public="a18034f582a574be72e15a241c109370b28663a7b299cb33c12c148af5e77ca1")
SEEDS = dict(current=[0, 1, 2], public=[0])
KINDS = dict(current="current_fql_behavior_flow_evaluation_only",
             public="released_public_flowplace_evaluation_only")


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=native.sha(path))


def checked_file(binding):
    if native.sha(binding["path"]) != binding["sha256"]:
        raise ValueError("Changed frozen baseline artifact: " + str(binding["path"]))
    return Path(binding["path"]).resolve()


def checked_audit(path):
    if native.sha(path) != AUDIT_SHA:
        raise ValueError("Unrecognized complete actual-array export audit")
    audit = native.read(path)
    if (audit.get("schema") != "qstudy_authorized_baseline_native_submission_v1"
            or any(audit.get(k) is not True for k in (
                "all32_actual_arrays_verified", "eight_actual_noises_identical_across_three_current_seeds_and_public",
                "no_DRT", "no_train_or_replay")) or set(audit["baselines"]) != {"current", "public"}):
        raise ValueError("Wrong baseline source/noise audit scope")
    checked_file(audit["source"])
    return audit


def checked_plan(experiment, label, audit):
    path = experiment/"native_plan/plan.json"
    plan = native.read(path)
    evidence = audit["baselines"][label]
    if (native.sha(path) != PLAN_SHA[label] or evidence["plan"] != fingerprint(path)
            or plan["requests"] != 8*len(SEEDS[label]) or plan["seeds"] != SEEDS[label]
            or plan["indices"] != list(range(8)) or plan["source_kind"] != KINDS[label]
            or plan["role"] != "evaluation" or plan["training_or_replay_eligible"] is not False
            or plan["threads_per_worker"] != 8 or plan["memory_per_worker"] != "24G"
            or plan["concurrency"] != 2):
        raise ValueError("Baseline immutable native plan/budget differs")
    for key in ("config", "declaration", "export_completion", "matrix"):
        checked_file(plan[key])
        if plan[key] != evidence[key]:
            raise ValueError("Export audit/native plan binding differs: " + key)
    if checked_file(plan["config"]) != experiment/"config.json":
        raise ValueError("Native plan belongs to a different experiment")
    for name,digest in plan["worker_sources"].items():
        if native.sha(ROOT/name) != digest:
            raise ValueError("Frozen native worker changed: " + name)
    checked_file(plan["preparation_source"])
    cfg, declaration = native.read(experiment/"config.json"), native.read(experiment/"declaration.json")
    if (cfg["seeds"] != SEEDS[label] or cfg["native_budget"] != plan["requests"]
            or declaration["source_kind"] != KINDS[label] or declaration["expected_requests"] != plan["requests"]
            or declaration["seeds"] != SEEDS[label] or declaration["role"] != "evaluation"
            or declaration["training_feedback_eligible"] is not False
            or declaration["offline_train_eligible"] is not False or declaration["terminal_mask"] != 0):
        raise ValueError("Wrong declaration/seed/evaluation eligibility")
    archive = experiment/("flow_eval_sources" if label == "current" else "public_eval_sources")
    for name,digest in declaration["source_sha256"].items():
        if native.sha(ROOT/name) != digest or native.sha(archive/name) != digest:
            raise ValueError("Frozen source/archive differs: " + name)
    if label == "current":
        done = native.read(checked_file(plan["export_completion"]))
        if (done["status"] != "completed" or done["count"] != 24
                or done["config_sha256"] != native.sha(experiment/"config.json")
                or done["declaration_sha256"] != native.sha(experiment/"declaration.json")
                or done["dispatch_sha256"] != native.sha(experiment/"dispatch.json")):
            raise ValueError("Current-flow completed export binding differs")
    else:
        checked_file(plan["sampler_config"])
        model = native.read(plan["sampler_config"]["path"])["model"]
        if (model["max_diffusion_steps"] != 20 or model["use_hard_constraint_in_sampling"] is not True
                or model["hard_constraint_w_reg"] != .1 or model["hard_constraint_w_dist"] != 1.):
            raise ValueError("Released public sampler differs from official20hard settings")
    matrix = [json.loads(line) for line in Path(plan["matrix"]["path"]).read_text().splitlines() if line.strip()]
    if len(matrix) != plan["requests"]:
        raise ValueError("Native dispatch denominator differs")
    return cfg, declaration, plan, matrix


def verify_request_audit(row, evidence):
    """Rehash the earlier actual fixed-node/trajectory audit, never rerun a flow."""
    if evidence["seed"] != row["seed"] or evidence["index"] != row["index"]:
        raise ValueError("Request no longer matches audited export identity")
    for key in ("noise_sha256", "raw_action_sha256", "action_sha256"):
        if evidence[key] != row[key]:
            raise ValueError("Audited noise/action identity changed: " + key)
    sample = Path(row["sample_dir"]).resolve()
    if (checked_file(evidence["request"]) != sample/"request.json"
            or checked_file(evidence["actions"]) != sample/"actions.npz"
            or evidence["fixed_raw_and_executed_preserved"] is not True):
        raise ValueError("Changed actual-array/fixed-node export evidence")
    for key in ("DEF", "trajectory", "repeat_determinism"):
        if evidence.get(key):
            checked_file(evidence[key])


def dispatch_record(experiment, seed, row, sample, tasks):
    """Reconstruct the full record written by both frozen native plan preparers."""
    return dict(experiment=str(experiment), seed=seed, round=0, index=row["index"],
        sample_dir=str(sample), tasks_sha256=native.sha(tasks),
        request_sha256=native.sha(sample/"request.json"),
        **{key:row[key] for key in ("noise_sha256", "action_sha256", "raw_action_sha256",
            "checkpoint_sha256", "run_id", "role", "training_feedback_eligible",
            "offline_train_eligible")})


def verify_dispatch(expected, actual):
    # Exact ordered equality also binds all extended provenance/eligibility keys.
    if expected != actual:
        raise ValueError("Native dispatch rows differ from completed immutable exports")


def inspect_outcome(row, sample, tasks, backend_context):
    """Read native evidence only. Missing saved geometry must never launch EDA."""
    from .qstudy_backend import checked_existing
    cfg, verification, binding = backend_context
    item = {k:row.get(k) for k in ("seed", "index", "role", "source_kind", "source_sampler",
        "noise_sha256", "checkpoint_sha256", "raw_action_sha256", "action_sha256", "sample_dir")}
    item.update(state="missing", metrics=None, request=fingerprint(sample/"request.json"),
                actions=fingerprint(sample/"actions.npz"), legalizer_jump_rms=row.get("legalizer_jump_rms"))
    path = sample/"outcome.json"
    if not path.exists():
        return item
    try:
        out = native.read(path)
        for key in ("seed", "round", "arm", "role", "index", "graph", "training_feedback_eligible"):
            if out.get(key) != row.get(key):
                raise ValueError("Native outcome/request identity differs: " + key)
        if out.get("physical_run"):
            run = Path(out["physical_run"])
            if native.read(run/"manifest.json").get("success") and not (run/"backend_macro_audit.json").is_file():
                raise ValueError("Missing saved native geometry audit; report must not run OpenROAD")
        checked_existing(out, row, sample, tasks, binding, cfg, verification)
        if out["status"] not in native.ADMITTED:
            raise ValueError("Nonadmitted native outcome is not a physical failure label")
        values = out["metrics"] if out["status"] == "success" else None
        if values is not None and any(type(values.get(k)) not in (int,float) or not math.isfinite(values[k])
                                      for k in native.METRICS):
            raise ValueError("All seven finite native GRT measurements are required")
        item.update(state=out["status"], metrics=values, outcome=fingerprint(path),
            native=native.native_fingerprints(out), failure_class=out.get("failure_class"), stage=out["stage"],
            backend_macro_legal=out.get("backend_macro_legal"), detailed_route_DRC=None)
    except Exception as error:
        item.update(state="invalid", metrics=None, integrity_error=str(error))
    return item


def summarize(rows, expected=8):
    counts = {k:sum(r["state"] == k for r in rows) for k in
              ("success", "geometry_failure", "physical_failure", "missing", "invalid")}
    successes = [r for r in rows if r["state"] == "success"]
    return dict(planned=expected, exported=len(rows), not_exported=max(expected-len(rows),0), **counts,
        fully_accounted=len(rows)==expected and sum(counts[k] for k in native.ADMITTED)==expected,
        conditional_PPA=len(successes)!=expected,
        successful_native_metric_n=len(successes),
        successful_native_metric_means={k:native.mean([r["metrics"][k] for r in successes]) for k in native.METRICS},
        generation_noise_statistics_success_only={k:native.describe([r["metrics"][k] for r in successes]) for k in native.METRICS},
        legalizer_jump_rms=native.describe([r.get("legalizer_jump_rms") for r in rows]))


def aggregate(streams, label):
    expected = SEEDS[label]
    complete = (len(streams)==len(expected) and {s["seed"] for s in streams}==set(expected)
        and all(s["fully_accounted"] and not s.get("integrity_errors") for s in streams)
        and all(s["actual_noise_by_index"]==streams[0]["actual_noise_by_index"] for s in streams))
    result = dict(baseline=label, aggregate_eligible=bool(complete), planned_seeds=expected,
        completed_seeds=[s["seed"] for s in streams if s["fully_accounted"]],
        request_accounting={k:sum(s[k] for s in streams) for k in
            ("planned","exported","not_exported","success","geometry_failure","physical_failure","missing","invalid")},
        conditional_PPA=any(s["conditional_PPA"] for s in streams), metrics=None,
        independent_training_seed_n=3 if complete and label=="current" else 0,
        released_public_checkpoint_n=1 if label=="public" else 0)
    if complete:
        if label=="current":
            result["metrics"] = {}
            for key in native.METRICS:
                stat = native.describe([s["successful_native_metric_means"][key] for s in streams])
                result["metrics"][key] = dict(stat, eligible=True) if stat["n"]==3 else dict(n=stat["n"],mean=None,sd=None,eligible=False)
            result["uncertainty_unit"] = "Sample SD of three equally weighted independent common-flow training-seed means; eight shared actual generation noises per seed"
        else:
            result["metrics"] = streams[0]["generation_noise_statistics_success_only"]
            result["uncertainty_unit"] = "Descriptive sample SD across successful generation noises for ONE fixed released checkpoint; not training-seed SD or confidence interval"
    return result


def collect(experiment, label, audit):
    # Preserve the local analysis package before upstream baseline adds paths.
    from . import qstudy_backend as backend
    from .graph_o2o_backend import load_requests
    from . import qstudy_flow_eval as current
    from . import qstudy_public_flow_eval as public
    cfg, declaration, plan, matrix = checked_plan(experiment, label, audit)
    if label=="current":
        actual_cfg, physical, controls = current.settings(experiment)
        if current.source_pairs(actual_cfg, controls) != declaration["pairs"]:
            raise ValueError("Current-flow matched parent source bindings changed")
    else:
        actual_cfg, physical, _ = public.settings(experiment)
    if cfg != actual_cfg or declaration["physical_context"] != physical:
        raise ValueError("Baseline physical/declaration context changed")
    backend_context = backend.context(experiment)
    evidence = {(r["seed"],r["index"]):r for r in audit["baselines"][label]["rows"]}
    if set(evidence) != {(s,i) for s in SEEDS[label] for i in range(8)}:
        raise ValueError("Actual-array audit denominator differs")
    streams, expected_matrix = [], []
    for seed in SEEDS[label]:
        requests, tasks = load_requests(experiment, seed, 0)
        marker = native.read(tasks.with_name("completed.json"))
        key = "flow_eval_declaration_sha256" if label=="current" else "public_eval_declaration_sha256"
        if (marker.get("status") != "completed" or marker.get(key) != current.object_sha(declaration)
                or marker.get("evaluation_never_replayed") is not True or len(requests)!=8
                or [r["index"] for r in requests]!=list(range(8))):
            raise ValueError("Completed baseline export/denominator differs")
        if label=="public" and (marker.get("independent_public_checkpoints")!=1
                or marker.get("sampler_config_sha256") != plan["sampler_config"]["sha256"]):
            raise ValueError("Public checkpoint/sampler completion differs")
        if label=="current":
            done=native.read(plan["export_completion"]["path"])
            if done["seed_tasks_sha256"][str(seed)] != native.sha(tasks):
                raise ValueError("Current seed export differs from top completion")
        rows = []
        for row in requests:
            if (row["role"]!="evaluation" or row["terminal_mask"]!=0
                    or row["training_feedback_eligible"] is not False or row["offline_train_eligible"] is not False):
                raise ValueError("Baseline row is not strictly evaluation-only")
            if label=="current":
                pair=next(p for p in declaration["pairs"] if p["seed"]==seed and p["index"]==row["index"])
                sample=current.verify_row(row,argparse.Namespace(seed=seed,round=0),cfg,physical,declaration,pair)
            else:
                sample=public.verify_row(row,cfg,physical,declaration,declaration["noises"][row["index"]])
            verify_request_audit(row,evidence[(seed,row["index"])])
            rows.append(inspect_outcome(row,sample,tasks,backend_context))
            expected_matrix.append(dispatch_record(experiment,seed,row,sample,tasks))
        if len({r["noise_sha256"] for r in rows})!=8:
            raise ValueError("Eight distinct actual noises are required")
        streams.append(dict(baseline=label,seed=seed,round=0,experiment=str(experiment),
            tasks=fingerprint(tasks),export_completion=fingerprint(tasks.with_name("completed.json")),
            actual_noise_by_index={str(r["index"]):r["noise_sha256"] for r in rows},
            integrity_errors=[r["integrity_error"] for r in rows if r["state"]=="invalid"],
            rows=rows,**summarize(rows)))
    verify_dispatch(expected_matrix,matrix)
    return dict(baseline=label,experiment=str(experiment),
        plan=fingerprint(experiment/"native_plan/plan.json"),config=plan["config"],declaration=plan["declaration"],
        matrix=plan["matrix"],source_kind=KINDS[label],physical_binding=backend_context[2],
        streams=streams,aggregate=aggregate(streams,label))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current",type=Path,required=True)
    parser.add_argument("--public",type=Path,required=True)
    parser.add_argument("--export-audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Read-only array/native reporting requires bounded CPU Slurm")
    audit=checked_audit(args.export_audit)
    results=[collect(getattr(args,label).resolve(),label,audit) for label in ("current","public")]
    streams=[s for r in results for s in r["streams"]]
    same=all(s["actual_noise_by_index"]==streams[0]["actual_noise_by_index"] for s in streams)
    if not same:
        raise ValueError("Current/public baseline actual shared-noise plan differs")
    if len({r["physical_binding"]["physical_flow_revision"] for r in results})!=1:
        raise ValueError("Current/public physical evaluator revisions differ")
    invalid=any(s["integrity_errors"] for s in streams)
    status="integrity_invalid" if invalid else "complete" if all(s["fully_accounted"] for s in streams) else "partial"
    output=args.output.resolve()
    for protected in (args.current.resolve(),args.public.resolve()):
        if output==protected or output in protected.parents or protected in output.parents:
            raise ValueError("Report output must be separate from physical/export experiments")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot=output/(stamp+"_job"+os.environ["SLURM_JOB_ID"])
    snapshot.mkdir(parents=True,exist_ok=False)
    result=dict(schema="qstudy_flow_baseline_native_report_v1",status=status,created_utc=stamp,
        slurm_job_id=os.environ["SLURM_JOB_ID"],export_audit=fingerprint(args.export_audit),
        same_actual_eight_noises=True,stage="global_route_only",development_only=True,
        no_fitting=True,model_inference_calls=0,backend_calls=0,no_replay=True,selected_winner=None,
        conventions=dict(current="3 independent common-flow checkpoints x8 shared actual noises; Euler10 without intermediate hard constraints then final legalizer",
            public="1 released FlowPlace checkpoint x8 actual noises; official20hard sampler then external final legalizer; NOT three training seeds",
            PPA="Seven native GRT measurements, success-conditional; no final DRT/DRC inferred",
            accounting="Missing/invalid are unresolved, not failures. Confirmed geometry/physical failures get no invented PPA. Fixed denominators24/8",
            uncertainty="Current: sample SD across three equal training-seed means. Public: descriptive sample SD across successful generation noises from one fixed checkpoint. No cross-checkpoint uncertainty inferred",
            selection="All declared layouts and failures reported; no quality filtering, winner or actor promotion"),
        metric_names=list(native.METRICS),baselines=results,
        source_sha256={name:native.sha(ROOT/name) for name in (
            "fql/qstudy_flow_baseline_report.py","fql/qstudy_actor_report.py","fql/qstudy_backend.py",
            "fql/qstudy_flow_eval.py","fql/qstudy_public_flow_eval.py","fql/graph_o2o_backend.py")})
    (snapshot/"report.json").write_text(json.dumps(native.clean(result),indent=2,allow_nan=False)+"\n")
    (snapshot/"reporter_source.py").write_bytes(Path(__file__).read_bytes())
    native.write_csv(snapshot/"streams.csv",[{k:v for k,v in s.items() if k!="rows"} for s in streams])
    native.write_csv(snapshot/"aggregates.csv",[r["aggregate"] for r in results])
    native.write_csv(snapshot/"requests.csv",[dict(baseline=s["baseline"],**row) for s in streams for row in s["rows"]])
    lines=["# Flow baseline native GRT report","",f"Status **{status}**, Slurm `{os.environ['SLURM_JOB_ID']}`.","",
        "Current: three independently trained common flows, each on eight matched noises. Public: one fixed released checkpoint on eight generation noises. These uncertainty units differ. No final DRT/DRC or winner claim.","",
        "| Baseline | Seed | Planned | Success | Geometry failure | Physical failure | Missing | Invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in streams:
        lines.append(f"| {s['baseline']} | {s['seed']} | {s['planned']} | {s['success']} | {s['geometry_failure']} | {s['physical_failure']} | {s['missing']+s['not_exported']} | {s['invalid']} |")
    lines += ["","Seven native metric means and their explicitly defined SDs, all request/source/native fingerprints and conditional-failure accounting are in JSON/CSV. Aggregate values are withheld until each baseline's complete declared plan is accounted for."]
    (snapshot/"report.md").write_text("\n".join(lines)+"\n")
    artifacts={p.name:native.sha(p) for p in snapshot.iterdir() if p.is_file()}
    (snapshot/"completed.json").write_text(json.dumps(dict(status="completed",report_status=status,
        artifacts=artifacts,backend_calls=0,slurm_job_id=os.environ["SLURM_JOB_ID"]),indent=2)+"\n")
    print(json.dumps(dict(status=status,snapshot=str(snapshot),
        verified_outcomes=sum(s["success"]+s["physical_failure"]+s["geometry_failure"] for s in streams),
        planned_outcomes=32,invalid=sum(s["invalid"] for s in streams))),flush=True)


if __name__=="__main__":
    main()
