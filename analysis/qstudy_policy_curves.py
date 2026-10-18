"""Plot saved, complete Q-study native GRT policy means without running models.

All eight evaluation outputs and all three training seeds must be successful.
Incomplete/failure-conditional evidence is rejected rather than silently plotted
as an unconditional policy mean. The native reporter remains authoritative.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


METRICS = ("rWL_um", "WNS_ns", "TNS_ns", "horizontal_overflow_sum",
           "vertical_overflow_sum", "Power_W", "StdCellArea_um2")
LABELS = ("GRT wirelength (m), lower", "WNS (ns), higher", "TNS (ns), higher",
          "Horizontal overflow, lower", "Vertical overflow, lower",
          "Power (W), lower", "Standard-cell area (um2), lower")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(report):
    if report.get("stage") != "global_route_only" or report.get("status") != "complete":
        raise ValueError("Only an authoritative complete GRT report can be plotted")
    bindings = report["experiment_bindings"]
    expected, names = set(), set()
    for experiment, binding in bindings.items():
        cfg = binding["config"]
        if sorted(cfg["seeds"]) != [0, 1, 2] or cfg["evaluation_samples"] != 8:
            raise ValueError("Require three training seeds and eight evaluation outputs")
        if cfg["arm"] in names:
            raise ValueError("Experiment arm names must be unique")
        names.add(cfg["arm"])
        endpoint = binding["report_round_endpoint"]
        if type(endpoint) is not int or endpoint < 0:
            raise ValueError("Missing explicit original-round endpoint")
        expected.update((experiment, seed, r) for seed in [0, 1, 2] for r in range(endpoint + 1))
    groups, seen, shared_noise = defaultdict(list), set(), None
    per_seed = []
    for record in report["rounds"]:
        key = (record["experiment"], record["seed"], record["round"])
        if key not in expected or key in seen or record["status"] != "complete":
            raise ValueError("Unexpected, duplicate or incomplete policy round")
        seen.add(key)
        cfg = bindings[key[0]]["config"]
        if record["arm"] != cfg["arm"]:
            raise ValueError("Round arm differs from its bound configuration")
        summary = record["roles"]["evaluation"]
        rows = [r for r in record["rows"] if r["role"] == "evaluation"]
        if (summary["planned"] != 8 or summary["success"] != 8 or
                not summary["fully_accounted"] or summary["metrics_conditional_on_success"] or
                len(rows) != 8 or any(r["state"] != "success" for r in rows)):
            raise ValueError("Failure-conditional or incomplete PPA cannot be plotted as a policy mean")
        noises = sorted(r["noise_sha256"] for r in rows)
        if len(set(noises)) != 8:
            raise ValueError("Eight evaluation noise identities must be distinct")
        if shared_noise is None:
            shared_noise = noises
        elif noises != shared_noise:
            raise ValueError("Policy curves require the same eight saved evaluation noises")
        means = {}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in rows]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                   not math.isfinite(v) for v in values):
                raise ValueError("Missing or nonfinite native metric")
            means[metric] = statistics.fmean(values)
            if not math.isclose(means[metric], summary["successful_native_metric_means"][metric],
                                rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("Saved summary differs from its native outcome rows")
        row = dict(arm=record["arm"], seed=record["seed"], round=record["round"], **means)
        per_seed.append(row)
        groups[(record["arm"], record["round"])].append(row)
    if seen != expected:
        raise ValueError("A declared policy round or training seed is missing")
    curves = []
    for (arm, original_round), rows in sorted(groups.items()):
        if sorted(r["seed"] for r in rows) != [0, 1, 2]:
            raise ValueError("Exactly three independent training-seed means are required")
        for metric in METRICS:
            values = [r[metric] for r in rows]
            curves.append(dict(arm=arm, round=original_round, metric=metric,
                mean=statistics.fmean(values), seed_sample_sd=statistics.stdev(values),
                training_seed_n=3, evaluation_outputs_per_seed=8))
    return dict(curves=curves, per_seed=sorted(per_seed, key=lambda r:(r["arm"], r["round"], r["seed"])),
                actual_noise_sha256=shared_noise,
                interpretation="Equal-weight seed means; error bars are sample SD of three training seeds, not confidence intervals. GRT is not final detailed routing.")


def render(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted({r["arm"] for r in data["curves"]})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, metric, label in zip(axes.flat, METRICS, LABELS):
        scale = 1e6 if metric == "rWL_um" else 1
        for arm in arms:
            rows = sorted((r for r in data["curves"] if r["arm"] == arm and r["metric"] == metric),
                          key=lambda r:r["round"])
            ax.errorbar([r["round"] for r in rows], [r["mean"]/scale for r in rows],
                        yerr=[r["seed_sample_sd"]/scale for r in rows], marker="o", capsize=3, label=arm)
        ax.set_title(label, fontsize=10)
        rounds = sorted({r["round"] for r in data["curves"]})
        ax.set_xticks(rounds, ["Offline" if r == 0 else f"Round {r}" for r in rounds])
        ax.grid(alpha=.2)
    axes.flat[7].axis("off")
    axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(), loc="upper left", fontsize=9)
    axes.flat[7].text(0, .45, "8 fixed noises per policy\n3 independent training seeds\nError bars: sample SD of seed means\nGRT stage; no final DRC claim\nFixed endpoints; no best-output selection",
                      va="top", fontsize=10, transform=axes.flat[7].transAxes)
    fig.suptitle("Measured development GRT quality across original training rounds", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / ("policy_curves." + suffix), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Render saved reports under a bounded CPU Slurm allocation")
    if sha(args.report) != args.sha256:
        raise ValueError("Report differs from explicit caller binding")
    data = extract(json.loads(args.report.read_text()))
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_job" + os.environ["SLURM_JOB_ID"])
    output.mkdir(parents=True, exist_ok=False)
    render(data, output)
    for name in ("curves", "per_seed"):
        with (output / (name + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[name][0]))
            writer.writeheader(); writer.writerows(data[name])
    result = dict(status="completed", stage="global_route_only", report=dict(path=str(args.report.resolve()), sha256=args.sha256),
        source_sha256=sha(__file__), model_forward_calls=0, optimizer_updates=0, backend_calls=0, **data)
    (output / "data.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "completed.json").write_text(json.dumps(dict(status="completed", artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()}), indent=2) + "\n")
    print(json.dumps(dict(status="completed", output=str(output))))


if __name__ == "__main__":
    main()
