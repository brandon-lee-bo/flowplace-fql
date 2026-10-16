"""Lock one FQL alpha across all seeds from complete GRT validation only.

Does not export candidates, inspect heldout outcomes, submit jobs or guarantee
an improvement. A frozen future heldout noise list must predate validation.
"""
import argparse
import json
from pathlib import Path

try:
    from .relabel_backend import STAGE, sha256_file, write_json
    from .physical_failure_policy import validate_bound_policy
except ImportError:
    from relabel_backend import STAGE, sha256_file, write_json
    from physical_failure_policy import validate_bound_policy


def seal(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Selection already exists; it cannot be silently replaced")
    report = json.loads(report_path.read_text())
    policy = report.get("physical_failure_policy")
    accepted_statuses = ("completed", "completed_with_physical_failures") if policy else ("completed",)
    if report.get("status") not in accepted_statuses or report.get("phase") != "validation" or report.get("stage") != STAGE:
        raise ValueError("Only complete GRT validation reports can select hyperparameters")
    if report.get("failures") or report.get("selection_allowed") is not True:
        raise ValueError("Report is incomplete or explicitly prohibits selection")
    if sha256_file(report["tasks_path"]) != report["tasks_sha256"] or sha256_file(report["plan_path"]) != report["plan_sha256"]:
        raise ValueError("Validation tasks/plan changed after aggregation")
    plan = report["plan"]
    if policy is not None:
        if validate_bound_policy(plan, Path(report["plan_path"]).parent) != policy:
            raise ValueError("Selection policy differs from the bound candidate plan")
        if report.get("outcomes_complete") is not True:
            raise ValueError("Policy selection requires every frozen terminal outcome")
        if report.get("physical_failures") and report.get("allow_preregistered_physical_failures") is not True:
            raise ValueError("Failed outcomes require the explicit preregistered policy mode")
    elif "physical_failure_policy" in plan:
        raise ValueError("Bound policy was not verified by aggregation")
    if plan.get("training_reward_stage") != STAGE:
        raise ValueError("This selector is for GRT-trained candidates, not exploratory proxy transfer")
    heldout = plan.get("heldout_indices")
    if not isinstance(heldout, list) or not heldout or len(set(heldout)) != len(heldout) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in heldout):
        raise ValueError("Validation plan must already contain a fixed heldout noise-index list")
    measured = report["quality"]["by_configuration"]
    if "distill_only" not in measured or "teacher_relegalized" not in measured or "best_observed_train" not in measured:
        raise ValueError("Require BC, teacher_relegalized and one best-observed-TRAIN baseline")
    if set(heldout) & {row["index"] for row in report["records"] if row["split"] == 1}:
        raise ValueError("Frozen heldout indices overlap validation")
    expected_seeds = set(measured["distill_only"]["training_seeds"])
    if not expected_seeds or any(not isinstance(seed, int) for seed in expected_seeds):
        raise ValueError("BC must contain explicitly identified training seeds")
    fql = {name: values for name, values in measured.items() if name.startswith("fql_alpha_")}
    if not fql:
        raise ValueError("No FQL configurations in validation report")
    noise_count = measured["distill_only"]["noise_count_per_seed"]
    by_method = report["quality"]["by_method"]
    reference_indices = by_method[measured["distill_only"]["methods"][0]]["indices"]
    for name, values in fql.items():
        if set(values["training_seeds"]) != expected_seeds or values["noise_count_per_seed"] != noise_count:
            raise ValueError("Every alpha and BC must use identical seeds and noise budgets")
        if any(by_method[method]["indices"] != reference_indices for method in values["methods"]):
            raise ValueError("Every alpha must use the exact BC validation noise indices")
    eligibility = {}
    for name, values in fql.items():
        rows = [row for row in report["records"] if row["configuration"] == name]
        eligible = all(row["status"] == "success" for row in rows) and values["rwl_mean_across_seed_means"] is not None
        if policy and len(rows) != 12:
            raise ValueError("Policy alpha must retain all 12 validation cases")
        eligibility[name] = {"eligible": eligible, "total_count": len(rows),
                             "success_count": sum(row["status"] == "success" for row in rows),
                             "physical_failure_count": sum(row["status"] == "physical_failure" for row in rows),
                             "reason": "all_requested_cases_successful" if eligible else "verified_physical_failure_in_fixed_budget"}
        if not policy and not eligible:
            raise ValueError("Strict selection requires all configurations complete")
    scores = {name: values["rwl_mean_across_seed_means"]["mean"] if eligibility[name]["eligible"] else None
              for name, values in fql.items()}
    eligible_scores = {name: value for name, value in scores.items() if value is not None}
    chosen = min(sorted(eligible_scores), key=lambda name: eligible_scores[name]) if eligible_scores else None
    methods = sorted(measured["distill_only"]["methods"] + fql[chosen]["methods"]) if chosen else []
    baseline_summary = measured["distill_only"]["rwl_mean_across_seed_means"]
    baseline_rwl = baseline_summary["mean"] if baseline_summary is not None else None
    best_train_rwl = measured["best_observed_train"]["rwl_mean_across_seed_means"]["mean"]
    fixed = [row for row in report["records"] if row["role"] == "fixed_training_baseline"]
    if len(fixed) != 1:
        raise ValueError("Best-observed-TRAIN must be one fixed measured reference")
    heldout_candidates = [{"method": method, "index": index} for method in ["teacher_relegalized", *methods] for index in heldout] if chosen else []
    if chosen:
        heldout_candidates.append({"method": "best_observed_train", "index": fixed[0]["index"]})
    result = {"kind": "sealed_GRT_validation_selection" if chosen else "sealed_GRT_validation_stop",
              "status": "selected" if chosen else "no_eligible_configuration", "selection_stage": STAGE,
              "selection_source": "validation_only", "source_report": str(report_path),
              "source_report_sha256": sha256_file(report_path), "validation_plan_sha256": report["plan_sha256"],
              "validation_tasks_sha256": report["tasks_sha256"], "backend_consistency": report["backend_consistency"],
              "dataset_sha256": plan.get("dataset_sha256"), "training_dir": plan.get("training_dir"),
              "rule": "All predeclared cases must succeed for eligibility; minimum mean GRT rWL across seed means; one alpha; lexical tie-break",
              "chosen_fql_configuration": chosen, "validation_rwl_by_configuration": scores,
              "eligibility_by_configuration": eligibility,
              "methods": methods, "training_seeds": sorted(expected_seeds),
              "validation_beats_BC": scores[chosen] < baseline_rwl if chosen and baseline_rwl is not None else None,
              "validation_beats_best_observed_train": scores[chosen] < best_train_rwl if chosen else None,
              "bc_comparison_complete": baseline_rwl is not None,
              "automatic_next_action": None,
              "heldout_plan": {"phase": "heldout", "training_reward_stage": STAGE, "indices": heldout,
                               "candidates": heldout_candidates, "new_candidate_backend_budget": len(methods) * len(heldout),
                               "fixed_training_baseline_repeated": False} if chosen else None,
              "notes": ["A chosen alpha can still be worse than BC or the fixed best-TRAIN layout",
                        "This selection never starts heldout evaluation automatically",
                        "Reuse teacher/best-TRAIN backend labels only for identical executed DEF and backend provenance",
                        "Timing/power remain separate diagnostics; rWL selection does not certify overall PPA improvement"]}
    if policy is not None:
        result["physical_failure_policy"] = policy
        if chosen:
            result["heldout_plan"]["physical_failure_policy"] = policy
        result["notes"].extend(["Physical failures retain denominators and make a configuration ineligible; no successful-subset means",
                                "All-ineligible is a terminal stop with no executable heldout plan",
                                "Heldout failures cannot change alpha; 12/12 validation success is not a population reliability claim"])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = seal(args.validation_report, args.output)
    print(json.dumps({key: result[key] for key in ["chosen_fql_configuration", "validation_beats_BC", "validation_beats_best_observed_train"]}), flush=True)


if __name__ == "__main__":
    main()
