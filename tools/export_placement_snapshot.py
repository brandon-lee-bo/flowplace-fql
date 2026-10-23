"""Export a small, read-only summary of the completed version-development reports.

This reads saved JSON only. It never runs a model, training, or physical tools.
Large source artifacts remain separately stored; their report digests are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path


REPORTS = {
    "v10_three_seeds": "v10_three_development_seeds_complete_v1/report.json",
    "v10_r1_controls": "v10_r1_attribution_report_v1/report.json",
    "v10_r2_controls": "v10_rounds_report_R2_complete_v1/report.json",
    "v10_fixed8": "v10_fixed8_report_complete_v1/report.json",
    "v2_final": "v2_final_recovered_saved_report_v1/report.json",
    "v2_q_support": "v2_final_saved_support_v1/report.json",
}


def read_bound(root, name):
    raw = (root / name).read_bytes()
    data = json.loads(raw)
    if data.get("integrity_errors"):
        raise ValueError("Source report has integrity errors: " + name)
    return data, dict(evidence_id=name, sha256=hashlib.sha256(raw).hexdigest(),
                      status=data["status"])


def group_summary(groups):
    fields = ("count", "requested", "successful", "failed", "complete", "mean", "sample_std")
    return {key: {field: row[field] for field in fields if field in row}
            for key, row in groups.items()}


def paired_summary(comparisons):
    return {key: {field: value for field, value in row.items() if field != "pairs"}
            for key, row in comparisons.items()}


def export(root):
    reports, identities = {}, {}
    for key, name in REPORTS.items():
        reports[key], identities[key] = read_bound(root, name)
    three = reports["v10_three_seeds"]
    seeds = {}
    for seed, row in three["reports"].items():
        groups = {arm: row["groups"][arm + "/evaluation"] for arm in ("bc", "fql")}
        if any(group["count"] != 8 or group.get("failed", 0) for group in groups.values()):
            raise ValueError("Unexpected development cohort denominator")
        seeds[seed] = dict(groups=group_summary(groups),
                           paired_fql_vs_bc=paired_summary({"fql_vs_bc": row["paired"]["fql_vs_bc"]}))
    v10 = dict(aggregate=three["aggregate"], seeds=seeds,
               public_reference_measured_once=group_summary({
                   "public_flowplace20": three["reports"]["0"]["groups"]["flowplace/evaluation"]}))
    for key in ("v10_r1_controls", "v10_r2_controls", "v10_fixed8"):
        report = reports[key]
        v10[key] = dict(groups=group_summary(report["groups"]),
                        paired=paired_summary(report.get("paired", {})))
    final = reports["v2_final"]
    v2 = {key: final[key] for key in ("arms", "native_summary", "attempt_accounting",
          "independent_training_seeds", "metric_better_direction")}
    v2["paired"] = final["paired"]
    # Keep the saved diagnostic's interpretation and independent-label counts;
    # omit its server paths and output file inventory.
    support = reports["v2_q_support"]
    v2["q_support"] = {key: support[key] for key in ("counts", "semantics", "critics")}
    for critic in v2["q_support"]["critics"].values():
        critic["checkpoint"] = Path(critic["checkpoint"]).name
        for cohort in critic["behavior_cohorts"].values():
            # Terminal calibration has a directly measured target. Keep the
            # full per-prefix descriptive statistics in the source report.
            cohort.pop("per_macro", None)
    v11 = {}
    for arm in ("bc", "fql"):
        name = "v11_matched_seed0/seed_0/round_0/" + arm + "/completed.json"
        marker, identity = read_bound(root, name)
        if marker["status"] != "completed":
            raise ValueError("Unfinished V1.1 arm")
        v11[arm] = dict(completion=identity, checkpoint_sha256=marker["checkpoint_sha256"],
                        parameter_changes=marker["parameter_changes"])
    return dict(schema="placement_versions_public_snapshot_v1", as_of="2026-10-23",
                design="bp_fe", macros=11, stage="global_route_only",
                reward="-GRT_rWL_um / 2341121.25", independent_confirmation=False,
                evidence_root_required_for_full_audit=True, source_reports=identities,
                v10=v10, v11=dict(training=v11, matched_backend_comparison_complete=False), v2=v2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("Wrote saved-report summary: " + str(args.output))


if __name__ == "__main__":
    main()
