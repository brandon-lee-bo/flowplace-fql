"""Descriptive plots from a complete, provenance-validated GRT comparison.

Run through Slurm. No model selection, filtering of failed candidates, or claims
of independent replication from different noises of one trained model.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.placement_metrics import MacroHPWL
from analysis.plot_results import read_cache, sha, short, write_json
from fql.aggregate_backend_candidates import executed_action

NOTE = "Global-route intermediate metrics, not detailed-route Table 2. Descriptive small-sample comparison."


def label(name):
    return (short(name).replace("released_nohard_final", "Released no-hard + projection")
            .replace("experimental_standard_euler_final", "Experimental Euler + projection")
            .replace("euler_final", "Euler + projection")
            .replace("fixed_zero_anchor", "Fixed zero anchor")
            .replace("fixed_train_mean_anchor", "Fixed TRAIN mean anchor")
            .replace("best_observed_train", "Best observed TRAIN"))


def save(fig, output, stem):
    fig.text(.5, .005, NOTE, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(report_path, cache_path, output):
    report_path, cache_path, output = map(lambda p: Path(p).resolve(), (report_path, cache_path, output))
    report = json.loads(report_path.read_text())
    physical_failures = report.get("physical_failures", [])
    acceptable = report.get("status") == "completed" or (
        report.get("status") == "completed_with_physical_failures" and report.get("outcomes_complete") is True)
    if not acceptable or report.get("failures"):
        raise ValueError("Plotting requires a complete validated comparison, not a subset of successes")
    if report.get("stage") != "global_route_only" or report.get("not_final_table2") is not True:
        raise ValueError("This plotter expects intermediate global-route results")
    plan = report["plan"]
    bound_cache_hash = plan.get("source_bindings", {}).get("benchmark_cache", {}).get("sha256")
    if bound_cache_hash is None:
        bound_cache_hash = plan.get("provenance", {}).get("cache_sha256")
    if bound_cache_hash is not None and sha(cache_path) != bound_cache_hash:
        raise ValueError("Benchmark cache differs from frozen candidate provenance")
    tasks_path = Path(report["tasks_path"])
    if sha(tasks_path) != report["tasks_sha256"]:
        raise ValueError("Task mapping changed after aggregation")
    tasks = {(t["method"], t["index"]): t for t in map(json.loads, tasks_path.read_text().splitlines())}
    if len(tasks) != len(report["records"]) or len(tasks) != report["expected_count"]:
        raise ValueError("Task count disagrees with validated report")
    metric = MacroHPWL(read_cache(cache_path))
    rows = []
    for record in report["records"]:
        task = tasks[(record["method"], record["index"])]
        successful = record["status"] == "success"
        if successful and sha(record["metrics_path"]) != record["metrics_sha256"]:
            raise ValueError("Measured backend file changed after aggregation")
        if not successful and sha(record["physical_failure_path"]) != record["physical_failure_sha256"]:
            raise ValueError("Physical-failure evidence changed after aggregation")
        action = executed_action(task, tasks_path.parent)
        hpwl = metric(action)
        if task.get("macro_hpwl") is not None and not np.isclose(hpwl, task["macro_hpwl"], rtol=1e-7, atol=1e-4):
            raise ValueError("Stored proxy differs from actual exported action")
        rows.append({"method": record["method"], "configuration": record["configuration"], "status": record["status"],
                     "training_seed": record["training_seed"], "noise_index": record["index"],
                     "noise_conditioning": record["noise_conditioning"], "role": record["role"],
                     "macro_hpwl_cache_units": hpwl, **record["values"],
                     "run_id": record["run_id"], "reused_backend_measurement": record.get("reused_backend_measurement", False),
                     "metrics_path": record["metrics_path"], "task_manifest_path": record["task_manifest_path"],
                     "failure_class": record.get("physical_failure", {}).get("failure_class")})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "candidate_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    methods = list(report["quality"]["by_method"])
    counts = {method: (sum(row["status"] == "success" for row in rows if row["method"] == method),
                       sum(row["method"] == method for row in rows)) for method in methods}
    method_labels = [label(m) + (f" [{counts[m][0]}/{counts[m][1]} passed]" if counts[m][0] < counts[m][1] else "")
                     for m in methods]
    configs = list(dict.fromkeys(row["configuration"] for row in rows))
    palette = plt.get_cmap("tab10")
    colors = {name: palette(i % 10) for i, name in enumerate(configs)}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, .42 * len(methods))))
    metrics = [("rWL_um", 1e-6, "GRT wirelength (million um), lower is better"),
               ("WNS_ns", 1, "WNS (ns), closer to zero is better"),
               ("TNS_ns", 1, "TNS (ns), closer to zero is better")]
    for axis, (name, scale, title) in zip(axes, metrics):
        for i, method in enumerate(methods):
            subset = [row for row in rows if row["method"] == method and row["status"] == "success"]
            if not subset:
                continue
            values = np.array([row[name] for row in subset]) * scale
            c = colors[subset[0]["configuration"]]
            axis.scatter(values, np.full(len(values), i), color=c, s=25, alpha=.7)
            if counts[method][0] == counts[method][1]:
                axis.plot([values.min(), values.max()], [i, i], color=c, alpha=.6)
                axis.scatter(values.mean(), i, marker="D" if subset[0]["noise_conditioning"] else "s",
                             color=c, edgecolor="black", linewidth=.5, s=45, zorder=3)
        axis.set_yticks(range(len(methods)))
        axis.set_yticklabels(method_labels if axis is axes[0] else [])
        axis.invert_yaxis(); axis.set_xlabel(title); axis.grid(axis="x", alpha=.2)
    fig.suptitle("All planned cases retained; incomplete methods have points but no mean\nDiamonds = complete noise means; squares = one fixed layout")
    save(fig, output, "backend_quality_by_method")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for config in configs:
        subset = [row for row in rows if row["configuration"] == config and row["status"] == "success"]
        if not subset:
            continue
        fixed = not subset[0]["noise_conditioning"]
        marker = "s" if fixed else "o"
        options = dict(color=colors[config], label=label(config), marker=marker, alpha=.8, s=45)
        axes[0].scatter([row["macro_hpwl_cache_units"] / 1000 for row in subset],
                        [row["rWL_um"] / 1e6 for row in subset], **options)
        axes[1].scatter([row["rWL_um"] / 1e6 for row in subset], [row["WNS_ns"] for row in subset], **options)
    axes[0].set(xlabel="Macro-only HPWL (thousand cache units)", ylabel="GRT wirelength (million um)",
                title=f"Measured points only; {len(physical_failures)} physical failures have no GRT value")
    axes[1].set(xlabel="GRT wirelength (million um)", ylabel="WNS (ns)",
                title="Wirelength and timing must be checked separately")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    save(fig, output, "proxy_transfer_and_timing")

    # A descriptive pooled coefficient is not a significance test: method and
    # noise points are dependent and can have different within-method trends.
    correlations = {"pooled_descriptive_pearson_R": None if physical_failures else correlation(
                        [r["macro_hpwl_cache_units"] for r in rows], [r["rWL_um"] for r in rows]),
                    "by_method": {m: (correlation([r["macro_hpwl_cache_units"] for r in rows if r["method"] == m],
                                                 [r["rWL_um"] for r in rows if r["method"] == m])
                                      if counts[m][0] == counts[m][1] else None) for m in methods},
                    "warning": "Correlated method/noise samples; no p-values or generalization claim. No pooled R with physical failures, no incomplete-method R. Fixed controls are single measurements."}
    write_json(output / "summary.json", {"status": "completed", "report_path": str(report_path),
               "report_sha256": sha(report_path), "cache_path": str(cache_path), "cache_sha256": sha(cache_path),
               "source_path": str(Path(__file__).resolve()), "source_sha256": sha(__file__),
               "slurm_job_id": os.environ["SLURM_JOB_ID"], "count": len(rows), "correlations": correlations,
               "physical_failure_count": len(physical_failures), "successful_count": report["successful_count"],
               "note": NOTE, "selection_performed": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run numeric analysis through Slurm")
    run(args.report, args.cache, args.output)


if __name__ == "__main__":
    main()
