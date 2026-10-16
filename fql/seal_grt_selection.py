"""Lock one FQL alpha across all seeds from complete GRT validation only.

Does not export candidates, inspect heldout outcomes, submit jobs or guarantee
an improvement. A frozen future heldout noise list must predate validation.
"""
import argparse
import json
from pathlib import Path

try:
    from .relabel_backend import STAGE, sha256_file, write_json
except ImportError:
    from relabel_backend import STAGE, sha256_file, write_json


def seal(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Selection already exists; it cannot be silently replaced")
    report = json.loads(report_path.read_text())
    if report.get("status") != "completed" or report.get("phase") != "validation" or report.get("stage") != STAGE:
        raise ValueError("Only complete GRT validation reports can select hyperparameters")
    if report.get("failures") or report.get("selection_allowed") is not True:
        raise ValueError("Report is incomplete or explicitly prohibits selection")
    if sha256_file(report["tasks_path"]) != report["tasks_sha256"] or sha256_file(report["plan_path"]) != report["plan_sha256"]:
        raise ValueError("Validation tasks/plan changed after aggregation")
    plan = report["plan"]
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
    scores = {name: values["rwl_mean_across_seed_means"]["mean"] for name, values in fql.items()}
    chosen = min(sorted(scores), key=lambda name: scores[name])
    methods = sorted(measured["distill_only"]["methods"] + fql[chosen]["methods"])
    baseline_rwl = measured["distill_only"]["rwl_mean_across_seed_means"]["mean"]
    best_train_rwl = measured["best_observed_train"]["rwl_mean_across_seed_means"]["mean"]
    fixed = [row for row in report["records"] if row["role"] == "fixed_training_baseline"]
    if len(fixed) != 1:
        raise ValueError("Best-observed-TRAIN must be one fixed measured reference")
    heldout_candidates = [{"method": method, "index": index} for method in ["teacher_relegalized", *methods] for index in heldout]
    heldout_candidates.append({"method": "best_observed_train", "index": fixed[0]["index"]})
    result = {"kind": "sealed_GRT_validation_selection", "selection_stage": STAGE,
              "selection_source": "validation_only", "source_report": str(report_path),
              "source_report_sha256": sha256_file(report_path), "validation_plan_sha256": report["plan_sha256"],
              "validation_tasks_sha256": report["tasks_sha256"], "backend_consistency": report["backend_consistency"],
              "dataset_sha256": plan.get("dataset_sha256"), "training_dir": plan.get("training_dir"),
              "rule": "Minimum mean GRT rWL across all seed means; one alpha shared by every seed; lexical tie-break",
              "chosen_fql_configuration": chosen, "validation_rwl_by_configuration": scores,
              "methods": methods, "training_seeds": sorted(expected_seeds),
              "validation_beats_BC": scores[chosen] < baseline_rwl,
              "validation_beats_best_observed_train": scores[chosen] < best_train_rwl,
              "automatic_next_action": None,
              "heldout_plan": {"phase": "heldout", "training_reward_stage": STAGE, "indices": heldout,
                               "candidates": heldout_candidates, "new_candidate_backend_budget": len(methods) * len(heldout),
                               "fixed_training_baseline_repeated": False},
              "notes": ["A chosen alpha can still be worse than BC or the fixed best-TRAIN layout",
                        "This selection never starts heldout evaluation automatically",
                        "Reuse teacher/best-TRAIN backend labels only for identical executed DEF and backend provenance",
                        "Timing/power remain separate diagnostics; rWL selection does not certify overall PPA improvement"]}
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
