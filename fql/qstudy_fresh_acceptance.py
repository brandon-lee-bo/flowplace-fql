"""Pure saved-report operational gates for the locked fresh confirmation.

No model/native imports, measurement, selection or fitting. A resolved physical
failure never receives PPA. BC failures make the prescribed comparison
unassessable; candidate failures fail its complete-denominator primary gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


RULES_SHA256 = "b6553120d9a4f410740d9f45ee3214dfa62b48f27b0a75a8c4cfe262f9b940ab"
GRT_METRICS = ("rWL_um", "WNS_ns", "TNS_ns", "horizontal_overflow_sum",
               "vertical_overflow_sum", "Power_W", "StdCellArea_um2")
DRT_METRICS = ("DRT_WL", "WNS", "TNS", "Power", "StdCellArea", "DRC")
GRT_RESOLVED = {"success", "geometry_failure", "physical_failure"}
DRT_RESOLVED = {"completed", "upstream_physical_failure"}


def capture(path, expected_sha):
    path = Path(path).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha:
        raise ValueError("Pinned file SHA mismatch: " + str(path))
    return json.loads(data), dict(path=str(path), sha256=digest)


def bound(entry):
    if (not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or
            not Path(entry["path"]).is_absolute()):
        raise ValueError("Expected absolute path/SHA binding")
    return capture(entry["path"], entry["sha256"])


def checked_rules(rules, binding):
    if (binding["sha256"] != RULES_SHA256 or rules.get("version") != "qstudy_prospective_acceptance_v1" or
            rules.get("improvement_or_tradeoff_rule", {}).get("classification_precedence") !=
            ["inconclusive", "no_stable_gain", "tradeoff", "overall_improvement"]):
        raise ValueError("Only exact locked prospective acceptance v1 is supported")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def checked_output_root(path, protected):
    output=Path(path).resolve()
    for value in protected:
        require(isinstance(value,str) and Path(value).is_absolute(), "Missing absolute immutable evidence root")
        source=Path(value).resolve()
        require(output!=source and source not in output.parents,
                "Acceptance output must be outside immutable selection/DRT evidence roots")
    return output


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values) > 1 else None)


def summarize(rows, names, success):
    good = [r for r in rows if r["state"] == success]
    return dict(planned=len(rows), success=len(good), all_success=len(good) == len(rows),
                states={s:sum(r["state"] == s for r in rows) for s in sorted({r["state"] for r in rows})},
                PPA_is_success_conditional=len(good) != len(rows),
                metric_means={k:statistics.mean(r["metrics"][k] for r in good) if good else None for k in names},
                violation_means={k:statistics.mean(max(-r["metrics"][k], 0.) for r in good) if good else None
                                 for k in names if k in {"TNS", "TNS_ns"}})


def metric_summary(seed_rows, names):
    return {k:stats([s["metric_means"][k] for s in seed_rows])
            if all(s["metric_means"][k] is not None for s in seed_rows)
            else dict(n=sum(s["metric_means"][k] is not None for s in seed_rows), mean=None, sd=None)
            for k in names}


def evaluate(grt, drt, lock, declaration, route_plan, rules, bindings, candidate, baseline):
    """Return gate values from captured JSON only; malformed contracts fail closed."""
    result = dict(version="fresh_acceptance_v1", classification="inconclusive", gates=[],
                  candidate=candidate, baseline=baseline, bindings=bindings,
                  integrity_errors=[], metrics={}, selection_performed=False,
                  protected_evidence_roots=[lock.get("output_root"),route_plan.get("output")],
                  model_forward_calls=0, backend_calls=0, training_updates=0,
                  interpretation="Operational predeclared study gates; not statistical significance or winner selection")

    def gate(name, passed, value=None, threshold=None, reason=""):
        result["gates"].append(dict(name=name, passed=bool(passed), value=value,
                                    threshold=threshold, reason=reason))
        return bool(passed)

    try:
        checked_rules(rules, bindings["rules"])
        require(grt["version"] == "fresh_native_report_v1" and drt["version"] == "fresh_detailed_route_v1",
                "Unexpected fresh report versions")
        selection = bindings["selection"]
        require(grt["selection"] == drt["selection"] == route_plan["selection"] == declaration["selection_lock"] == selection,
                "Reports/declaration/route plan do not share the pinned selection")
        require(grt["declaration"] == drt["declaration"] == route_plan["declaration"] == bindings["declaration"],
                "GRT/DRT declaration bindings differ")
        require(drt["plan"] == bindings["route_plan"], "DRT report differs from captured route plan")
        require(bindings["rules"] in lock["development_inventory"], "Rules JSON is not in the locked development inventory")
        require(lock["status"] == "selection_locked", "Selection is not locked")
        require(grt["design"] == route_plan["design"] == lock["design"] and
                grt["native_revision"] == lock["physical_flow_revision"], "Native design/revision binding differs")
        require(declaration["role"] == "evaluation" and declaration["terminal_mask"] == 0 and
                declaration["evaluation_purpose"] == "fresh_confirmation" and
                declaration["training_feedback_eligible"] is False and declaration["offline_train_eligible"] is False,
                "Declaration admits training/replay or a non-confirmation role")
        require(grt["training_feedback_eligible"] is False and grt["offline_train_eligible"] is False and
                drt["feedback_eligible"] is False, "Report admits training feedback")
        require(drt["indices"] == route_plan["indices"] == [0, 1, 2, 3] and
                drt["route_cap"] == route_plan["route_cap"] == 20 and drt["backend_threads"] == 8 and
                drt["route_settings"] == route_plan["route_settings"], "DRT indices/repair budget/settings changed")
        noise = grt["noise"]
        require(noise == route_plan["noise"] and noise["selection_lock_sha256"] == selection["sha256"] and
                noise["noise_seed"] == 85000000 and noise["generator_device"] == "cuda" and
                noise["dtype"] == "float32" and noise["no_training_or_replay"] is True and
                len(noise["noise_sha256"]) == len(set(noise["noise_sha256"])) == 8,
                "GRT/DRT actual fresh noise inventory differs")
        require(lock["evaluation"]["noise_seed"] == 85000000 and lock["evaluation"]["samples"] == 8 and
                lock["evaluation"]["drt_indices"] == [0, 1, 2, 3] and lock["evaluation"]["drt_repair_cap"] == 20,
                "Locked fresh evaluation budget differs")
        streams = lock["streams"]
        require(len({s["id"] for s in streams}) == len(streams), "Duplicate locked stream IDs")
        require(all(s["seeds"] == ([0] if s["kind"] == "public_official20" else [0,1,2]) for s in streams),
                "Public reference is one frozen checkpoint; other streams require all three seeds")
        by_id = {s["id"]:s for s in streams}
        require(candidate != baseline and candidate in by_id and baseline in by_id and
                by_id[candidate]["purpose"] == "candidate" and by_id[baseline]["purpose"] == "bc" and
                by_id[candidate]["kind"] == by_id[baseline]["kind"] == "one_step" and
                by_id[candidate]["seeds"] == by_id[baseline]["seeds"] == [0, 1, 2],
                "Explicit candidate/BC IDs or all-three-seed contract differs")
        identities = {(s["id"], seed) for s in streams for seed in s["seeds"]}
        require(len(declaration["streams"]) == len(streams) and
                {(s["id"],seed) for s in declaration["streams"] for seed in s["seeds"]} == identities and
                declaration["evaluation"] == lock["evaluation"], "Declaration stream/evaluation identity differs")
        require(len(grt["streams"]) == len(identities) and
                {(r["stream"],r["seed"]) for r in grt["streams"]} == identities,
                "GRT stream/seed denominator changed")
        require(grt["expected_requests"] == 8*len(identities) and
                drt["requested"] == route_plan["requested"] == 4*len(identities),
                "Requested whole-study denominators changed")
        grt_rows = {}; drt_rows = {}; unresolved = []
        for record in grt["streams"]:
            sid, seed = record["stream"], record["seed"]
            require(record["purpose"] == by_id[sid]["purpose"] and record["kind"] == by_id[sid]["kind"],
                    "GRT stream role/kind differs from selection")
            require(record["planned"] == 8, "GRT per-seed denominator is not eight")
            rows = record["rows"]
            require(len({r["index"] for r in rows}) == len(rows), "Duplicate GRT index")
            for row in rows:
                index = row["index"]
                require(type(index) is int and 0 <= index < 8 and row["noise_sha256"] == noise["noise_sha256"][index],
                        "GRT row does not share the exact actual noise/index")
                require(row["state"] in GRT_RESOLVED | {"missing", "invalid"}, "Unknown GRT state")
                grt_rows[(sid,seed,index)] = row
            for index in range(8):
                key = (sid,seed,index)
                if key not in grt_rows:
                    grt_rows[key] = dict(index=index, state="not_exported", metrics=None)
                if grt_rows[key]["state"] not in GRT_RESOLVED:
                    unresolved.append(dict(stage="GRT", key=list(key), state=grt_rows[key]["state"]))
        require(len(drt["records"]) == 4*len(identities), "DRT row denominator changed")
        for row in drt["records"]:
            sid, seed, index = row["stream"], row["seed"], row["index"]
            key = (sid,seed,index)
            require((sid,seed) in identities and type(index) is int and index in [0,1,2,3] and key not in drt_rows,
                    "Unexpected/duplicate DRT identity")
            require(row["noise_sha256"] == noise["noise_sha256"][index], "DRT actual noise/index differs from GRT")
            require(row["state"] in DRT_RESOLVED | {"missing_GRT", "missing_DRT", "invalid"}, "Unknown DRT state")
            drt_rows[key] = row
            if row["state"] not in DRT_RESOLVED:
                unresolved.append(dict(stage="DRT", key=list(key), state=row["state"]))
            if row["state"] == "completed":
                require(grt_rows[key]["state"] == "success", "Completed DRT has no successful GRT source")
            if row["state"] == "upstream_physical_failure":
                require(grt_rows[key]["state"] in {"geometry_failure", "physical_failure"}, "DRT upstream failure differs from GRT")
        for rows, names, good in [(grt_rows,GRT_METRICS,"success"),(drt_rows,DRT_METRICS,"completed")]:
            for row in rows.values():
                if row["state"] == good:
                    require(isinstance(row["metrics"], dict) and set(names) <= set(row["metrics"]) and
                            all(finite(row["metrics"][k]) for k in names), "Successful PPA metrics missing/nonfinite")
                    require(all(row["metrics"][k] >= 0 for k in names if k not in {"WNS","TNS","WNS_ns","TNS_ns"}),
                            "Negative physical cost/DRC")
                else:
                    require(row.get("metrics") is None, "Failed/unresolved row has fabricated PPA")
        require(not drt.get("integrity_errors"), "DRT report has integrity errors")
        result["request_accounting"] = {
            stage:[dict(stream=sid,seed=seed,planned=count,
                        states={state:sum(rows[(sid,seed,i)]["state"]==state for i in range(count))
                                for state in sorted({rows[(sid,seed,i)]["state"] for i in range(count)})})
                   for sid,seed in sorted(identities)]
            for stage,rows,count in [("GRT",grt_rows,8),("DRT",drt_rows,4)]}
        gate("integrity_and_shared_identity", True, reason="Pinned reports, rules, selection, declaration, actual noise and budgets match")
        resolved = gate("all_required_requests_resolved", not unresolved, unresolved,
                        "Every locked stream: eight GRT and four DRT requests per declared seed")
        summaries = {}
        for stage, rows, count, names, good in [("GRT",grt_rows,8,GRT_METRICS,"success"),
                                               ("DRT",drt_rows,4,DRT_METRICS,"completed")]:
            summaries[stage] = {}
            for sid in [candidate,baseline]:
                per_seed = [dict(seed=seed, **summarize([rows[(sid,seed,i)] for i in range(count)],names,good))
                            for seed in range(3)]
                summaries[stage][sid] = dict(per_seed=per_seed, all_success=all(s["all_success"] for s in per_seed),
                    metrics=metric_summary(per_seed,names))
        result["metrics"] = summaries
        candidate_good = gate("candidate_all24_GRT_and12_DRT_success",
                              all(summaries[s][candidate]["all_success"] for s in ["GRT","DRT"]),
                              {s:sum(r["success"] for r in summaries[s][candidate]["per_seed"]) for s in ["GRT","DRT"]},
                              dict(GRT=24,DRT=12))
        baseline_good = gate("BC_full_native_comparison_assessable",
                             all(summaries[s][baseline]["all_success"] for s in ["GRT","DRT"]),
                             {s:sum(r["success"] for r in summaries[s][baseline]["per_seed"]) for s in ["GRT","DRT"]},
                             dict(GRT=24,DRT=12), "Conditional BC PPA cannot establish primary superiority")
        if not resolved or not baseline_good:
            result["reason"] = "Unresolved required evidence" if not resolved else "BC physical failure: prescribed native comparison is unassessable"
            return result
        grt_c, grt_b = summaries["GRT"][candidate], summaries["GRT"][baseline]
        # Primary is GRT-only; DRT failure belongs to the later feasibility/quality gate.
        primary_available = grt_c["all_success"]
        cmean = grt_c["metrics"]["rWL_um"]["mean"] if primary_available else None
        bmean = grt_b["metrics"]["rWL_um"]["mean"]
        require(bmean > 0, "BC WL must be positive to evaluate the declared fractional threshold")
        reduction = (bmean-cmean)/bmean if cmean is not None else None
        deltas = [a["metric_means"]["rWL_um"]-b["metric_means"]["rWL_um"] if a["all_success"] else None
                  for a,b in zip(grt_c["per_seed"],grt_b["per_seed"])]
        primary = gate("GRT_primary_at_least1percent_and_all3_lower",
                       primary_available and cmean <= .99*bmean and all(d < 0 for d in deltas),
                       dict(candidate_mean=cmean,BC_mean=bmean,reduction_fraction=reduction,paired_seed_deltas=deltas),
                       dict(minimum_fraction=.01,each_seed_delta_strictly_below=0))
        quality = []
        for stage, names in [("GRT",GRT_METRICS),("DRT",DRT_METRICS)]:
            left, right = summaries[stage][candidate], summaries[stage][baseline]
            continuous = {}
            for key in names:
                x,y = left["metrics"][key]["mean"],right["metrics"][key]["mean"]
                continuous[key] = dict(candidate=x,BC=y,delta_candidate_minus_BC=x-y if x is not None else None)
            result["metrics"][stage]["continuous_deltas"] = continuous
            if not left["all_success"]:
                quality.append(gate(stage+"_full_success_quality_assessable",False,reason="No substitute for failed layout PPA"))
                continue
            mapping = [("WNS_ns" if stage=="GRT" else "WNS","signed_lower",.02),
                       ("TNS_ns" if stage=="GRT" else "TNS","violation",None),
                       ("Power_W" if stage=="GRT" else "Power","relative",.01),
                       ("StdCellArea_um2" if stage=="GRT" else "StdCellArea","relative",.01)]
            if stage == "GRT":
                mapping += [("horizontal_overflow_sum","cost",None),("vertical_overflow_sum","cost",None)]
            else:
                mapping += [("DRT_WL","relative",0.)]
            for key,mode,epsilon in mapping:
                x,y = continuous[key]["candidate"],continuous[key]["BC"]
                if mode == "signed_lower":
                    passed,limit = x >= y-epsilon,y-epsilon
                elif mode == "violation":
                    x = statistics.mean(s["violation_means"][key] for s in left["per_seed"])
                    y = statistics.mean(s["violation_means"][key] for s in right["per_seed"])
                    limit=y+max(1.,.05*y);passed=x <= limit
                elif mode == "cost":
                    limit=y+max(1.,.05*y);passed=x <= limit
                else:
                    limit=(1.+epsilon)*y;passed=x <= limit
                quality.append(gate(stage+"_"+key,passed,
                    dict(candidate=x,BC=y,delta_candidate_minus_BC=x-y),
                    dict(candidate_minimum=limit) if mode=="signed_lower" else dict(candidate_maximum=limit),
                    "Mean positive TNS violation per request" if mode=="violation" else "Native equally weighted training-seed mean"))
        drcs = [drt_rows[(candidate,s,i)]["metrics"]["DRC"] for s in range(3) for i in range(4)
                if drt_rows[(candidate,s,i)]["state"] == "completed"]
        drc_ok = gate("DRT_all_candidate_DRC_zero",len(drcs)==12 and all(x==0 for x in drcs),drcs,
                      "All12 values equal zero; physical completion alone is not DRC clean")
        if not primary:
            result["classification"] = "no_stable_gain"
        elif not candidate_good or not all(quality) or not drc_ok:
            result["classification"] = "tradeoff"
        else:
            result["classification"] = "overall_improvement"
        result["reason"] = dict(no_stable_gain="Complete primary rule fails; individual/secondary improvements do not override it",
            tradeoff="Primary passes but secondary, feasibility or detailed-route gate fails",
            overall_improvement="All predeclared primary/secondary/feasibility/DRT gates pass")[result["classification"]]
        return result
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result["integrity_errors"].append(str(error))
        gate("integrity_and_shared_identity",False,reason=str(error))
        result["classification"] = "inconclusive"
        result["reason"] = "Pinned/schema/provenance contract is unresolved; no numerical acceptance claim"
        return result


def collect(grt_path, grt_sha, drt_path, drt_sha, rules_path, rules_sha, candidate, baseline):
    grt,g = capture(grt_path,grt_sha);drt,d = capture(drt_path,drt_sha);rules,r = capture(rules_path,rules_sha)
    checked_rules(rules,r)
    lock,s = bound(grt["selection"]);declaration,decl = bound(grt["declaration"])
    route_plan,p = bound(drt["plan"])
    # Rehash the common saved-noise artifact without loading/generating arrays.
    noise_artifact = grt["noise"]["artifact"]
    raw = Path(noise_artifact["path"]).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==noise_artifact["sha256"],"Saved actual noise artifact SHA differs")
    bindings=dict(GRT_report=g,DRT_report=d,rules=r,selection=s,declaration=decl,route_plan=p,actual_noise=noise_artifact)
    return evaluate(grt,drt,lock,declaration,route_plan,rules,bindings,candidate,baseline)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["grt","drt","rules"]:
        parser.add_argument("--"+key,type=Path,required=True);parser.add_argument("--"+key+"-sha256",required=True)
    parser.add_argument("--candidate",required=True);parser.add_argument("--bc",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run saved-report reduction under bounded CPU Slurm")
    result=collect(args.grt,args.grt_sha256,args.drt,args.drt_sha256,args.rules,args.rules_sha256,args.candidate,args.bc)
    output=checked_output_root(args.output,result["protected_evidence_roots"])/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ_job"+os.environ["SLURM_JOB_ID"])
    require(all(output!=Path(b["path"]) and output not in Path(b["path"]).parents for b in result["bindings"].values()),
            "Output may not overwrite or contain pinned evidence")
    output.mkdir(parents=True,exist_ok=False)
    result.update(reducer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),slurm_job_id=os.environ["SLURM_JOB_ID"])
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(classification=result["classification"],report=str(output/"report.json"))))


if __name__ == "__main__":
    main()
