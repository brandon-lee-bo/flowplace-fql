"""Summarize all six declared released-DEF fullflow attempts, retaining failures."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array-job", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use Slurm for numerical result aggregation")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paper_path = root / "backend/assets/paper_table2_flowplace.json"
    paper = json.loads(paper_path.read_text())
    runs = {"bp_fe": "released_bp_fe_03"}
    for i, design in enumerate(["ariane133", "ariane136", "bp", "bp_be", "swerv_wrapper"]):
        runs[design] = f"released_{design}_job{args.array_job}_{i}"
    columns = [("rWL_million_um", "DRT_WL", 1e-6), ("WNS_ns", "WNS", 1),
               ("TNS_ns", "TNS", 1), ("Power_W", "Power", 1), ("CellArea_um2", "StdCellArea", 1)]
    report = {"scope": "Released macro DEF through the released fullflow; not a reproduction of training or model generation",
              "status": "running", "source_paper": paper["source"], "paper_values_sha256": digest(paper_path),
              "array_job": args.array_job, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "source_sha256": digest(__file__), "designs": [], "failures": [],
              "limitations": ["Exact paper tool revision is unavailable; pinned image is newer",
                              "Released flow and paper narrative differ in timing/routability settings",
                              "Released DRT cap4 can leave violations; completion does not imply DRC clean",
                              "Power is analysis under released activity assumptions, not measured silicon power",
                              "No aggregate over successful designs is substituted for the complete six-design comparison"]}
    flat = []
    for design, reference in paper["designs"].items():
        run = root / "backend/results" / runs[design]
        row = {"design": design, "run": str(run), "paper_rounded": reference, "status": "unavailable"}
        try:
            manifest_path, metrics_path = run / "manifest.json", run / "metrics.json"
            manifest = json.loads(manifest_path.read_text())
            row.update(manifest_sha256=digest(manifest_path), slurm_job_id=manifest["slurm_job_id"])
            if manifest["design"] != design or digest(manifest["def_path"]) != manifest["def_sha256"]:
                raise ValueError("Declared design or input DEF differs from the run manifest")
            row.update(def_sha256=manifest["def_sha256"], openroad_sha256=manifest["openroad_sha256"])
            failure_path = run / "failure_analysis.json"
            if manifest.get("success") is not True and failure_path.exists():
                failure = json.loads(failure_path.read_text())
                if failure["manifest_sha256"] != digest(manifest_path) or failure["def_sha256"] != manifest["def_sha256"]:
                    raise ValueError("Failure analysis does not match original run inputs")
                for field in ["log", "raw_geometry_audit"]:
                    if digest(failure[field + "_path"]) != failure[field + "_sha256"]:
                        raise ValueError("Failure evidence changed after analysis")
                row.update(failure_analysis=failure, failure_analysis_sha256=digest(failure_path))
                raise ValueError(failure["error_line"])
            if manifest.get("success") is not True or manifest["design"] != design:
                raise ValueError("Fullflow did not complete successfully for the declared design")
            metrics = json.loads(metrics_path.read_text())
            audit_path = run / "fullflow_audit.json"
            audit = json.loads(audit_path.read_text())
            if audit.get("all_checks_pass") is not True or audit.get("stage") != "finish_after_detailed_route":
                raise ValueError("Final-report consistency audit did not pass")
            for relative, bound_hash in audit["source_report_sha256"].items():
                if digest(run / relative) != bound_hash:
                    raise ValueError("Raw final report changed after audit")
            measured, differences = {}, {}
            for name, key, scale in columns:
                value = float(metrics[key]) * scale
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {key}")
                measured[name] = value
                differences[name] = value - reference[name]
                flat.append({"design": design, "status": "completed", "error": "", "metric": name, "paper_rounded": reference[name],
                             "measured": value, "difference": differences[name],
                             "relative_to_abs_paper_percent": 100 * differences[name] / abs(reference[name])})
            drc = float(metrics["DRC"])
            if not math.isfinite(drc) or drc < 0 or not drc.is_integer():
                raise ValueError("Missing or invalid detailed-route violation count")
            row.update(status="completed", measured=measured, difference=differences, DRC=int(drc),
                       route_drc_clean=drc == 0, runtime_seconds=metrics["runtime"],
                       manifest_sha256=digest(manifest_path), metrics_sha256=digest(metrics_path),
                       def_sha256=manifest["def_sha256"], openroad_sha256=manifest["openroad_sha256"],
                       tool_image_sha256=manifest["tool_image_sha256"],
                       design_config_sha256=manifest["design_config_sha256"],
                       fullflow_audit_sha256=digest(audit_path),
                       slurm_job_id=manifest["slurm_job_id"])
        except Exception as error:
            row.update(status="failed_or_unavailable", error_type=type(error).__name__, error=str(error))
            report["failures"].append({"design": design, "run": str(run), "error": str(error)})
            flat = [entry for entry in flat if entry["design"] != design]
            for name, _, _ in columns:
                flat.append({"design": design, "status": row["status"], "error": str(error),
                             "metric": name, "paper_rounded": reference[name], "measured": None,
                             "difference": None, "relative_to_abs_paper_percent": None})
        report["designs"].append(row)
    successful = [row for row in report["designs"] if row["status"] == "completed"]
    report["completed_count"] = len(successful)
    report["same_tool_binary"] = len({row["openroad_sha256"] for row in successful}) <= 1
    report["status"] = "completed" if len(successful) == 6 and report["same_tool_binary"] else "incomplete"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "metric_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["design", "status", "error", "metric", "paper_rounded", "measured", "difference", "relative_to_abs_paper_percent"])
        writer.writeheader(); writer.writerows(flat)
    lines = ["# Released-DEF fullflow comparison", "", report["scope"], "",
             "| Design | Status | rWL (million um) | Paper rWL | WNS (ns) | TNS (ns) | Power (W) | Cell area (um²) | DRC |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["designs"]:
        if row["status"] != "completed":
            lines.append(f"| {row['design']} | {row['status']} | — | {row['paper_rounded']['rWL_million_um']} | — | — | — | — | — |")
        else:
            m = row["measured"]
            lines.append(f"| {row['design']} | completed | {m['rWL_million_um']:.6f} | {row['paper_rounded']['rWL_million_um']} | {m['WNS_ns']:.6f} | {m['TNS_ns']:.3f} | {m['Power_W']:.3f} | {m['CellArea_um2']:.1f} | {row['DRC']} |")
    lines += ["", "Paper values for every metric and signed differences are in metric_comparison.csv.", ""]
    if report["failures"]:
        lines += ["Failed or unavailable cases retain all five metric rows with blank measurements; they are not zero-valued results.", ""]
        lines += [f"- {failure['design']}: {failure['error']}" for failure in report["failures"]]
        lines.append("")
    lines += ["- " + note for note in report["limitations"]]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "completed": len(successful), "output": str(output)}))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
