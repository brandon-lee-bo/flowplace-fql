"""Independent audits and descriptive figures for the fixed-circuit macro proxy.

Run on Slurm. This script never launches training or evaluates routed PPA.
Seed and initial-noise variability are reported separately; repeated noise
across training seeds is not treated as independent replication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.placement_metrics import MacroHPWL, summarize

WARNING = "Macro-only HPWL proxy; excludes standard cells and explicit I/O. Not routed PPA."
SPLITS = {0: "train", 1: "validation", 2: "heldout"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.bbox": "tight"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def read_cache(path):
    # Official benchmark caches contain NumPy/Python data, not tensor storages.
    # Only the locally downloaded, provenance-checked official asset is loaded.
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        return pickle.loads(archive.read(name))


def independent_geometry(x, sizes, tol=1e-6):
    """Scalar pair loop, separate from the vectorized evaluator implementation."""
    x = np.asarray(x, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if x.shape != sizes.shape or not np.isfinite(x).all():
        return {"finite": False, "legal": False, "overlap_pairs": None,
                "out_of_bounds": None}
    lo, hi = x - sizes / 2, x + sizes / 2
    outside = sum(bool(np.any(lo[i] < -1-tol) or np.any(hi[i] > 1+tol))
                  for i in range(len(x)))
    overlap = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            overlap += bool(min(hi[i, 0], hi[j, 0]) - max(lo[i, 0], lo[j, 0]) > tol
                            and min(hi[i, 1], hi[j, 1]) - max(lo[i, 1], lo[j, 1]) > tol)
    return {"finite": True, "legal": outside == 0 and overlap == 0,
            "overlap_pairs": int(overlap), "out_of_bounds": int(outside)}


def save_fig(fig, output, stem):
    fig.text(.5, -.015, WARNING, ha="center", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)


def method_parts(name):
    return name.split("/", 1) if "/" in name else ("reference", name)


def short(name):
    name = re.sub(r"coordinate_hpwl_steps_(\d+)_anchor_([^_]+)_lr_([^/]+)",
                  r"Coordinate GD (anchor=\2)", name)
    return (name.replace("teacher_original", "Teacher (original)")
            .replace("teacher_relegalized", "Teacher (+projection)")
            .replace("distill_only", "Distillation")
            .replace("analytic_hpwl_alpha_", "Analytic actor a=")
            .replace("fql_alpha_", "FQL a=")
            .replace("coordinate_hpwl_", "Coordinate ")
            .replace("seed_control/", "").replace("seed_", "s"))


def audit_dataset(data, manifest, metric, output):
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
    n, d = data["noise"].shape
    shape = (len(metric.names), 2)
    check("teacher_shape", data["teacher_action"].shape == (n, d) and d == np.prod(shape))
    check("all_numeric_arrays_finite", all(np.isfinite(v).all() for v in data.values()))
    check("noise_in_uniform_support", np.all(np.abs(data["noise"]) <= 1))
    check("unique_noise_families", np.unique(data["noise"], axis=0).shape[0] == n)
    check("group_id_matches_family_index", np.array_equal(data["group_id"], np.arange(n)))
    family, variants = data["family_id"], data["variant"]
    check("valid_family_indices", np.all((family >= 0) & (family < n)))
    if not checks["valid_family_indices"]:
        raise ValueError("Critic family indices outside teacher range")
    check("critic_group_and_parent_match_family",
          np.array_equal(data["critic_group_id"], family) and
          np.array_equal(data["critic_parent_index"], family))
    check("all_variants_keep_parent_split", np.array_equal(data["critic_split"], data["split"][family]))
    counts = np.bincount(family, minlength=n)
    expected = manifest["args"]["perturbations"] + 1
    check("same_variant_count_each_family", np.all(counts == expected))
    check("variant_ids_unique_per_family", all(np.array_equal(np.sort(variants[family == i]),
                                                              np.arange(expected)) for i in range(n)))
    check("expected_split_counts", [int(np.sum(data["split"] == i)) for i in range(3)] ==
          [manifest["args"][k] for k in ("train", "val", "test")])
    original = data["teacher_action"].reshape(n, *shape)
    actions = data["action"].reshape(-1, *shape)
    teacher_geo = [independent_geometry(x, metric.sizes) for x in original]
    action_geo = [independent_geometry(x, metric.sizes) for x in actions]
    legal = np.array([r["legal"] for r in action_geo])
    check("critic_legality_matches_independent_geometry", np.array_equal(legal, data["legal"]))
    h = metric(actions)
    check("stored_hpwl_matches_recalculation", np.allclose(h, data["macro_hpwl"], rtol=1e-9, atol=1e-7))
    train_scale = h[(data["critic_split"] == 0) & legal].mean()
    check("reward_scale_uses_legal_training_rows_only",
          np.isclose(train_scale, manifest["reward_scale_train_only"], rtol=1e-12))
    check("stored_rewards_match_formula", np.allclose(data["reward"], -h/train_scale-(~legal)*10,
                                                     rtol=1e-6, atol=1e-6))
    baseline_rows = np.flatnonzero(variants == 0)
    baseline_rows = baseline_rows[np.argsort(family[baseline_rows])]
    check("variant_zero_raw_equals_original_teacher",
          np.array_equal(data["raw_action"][baseline_rows], data["teacher_action"]))
    check("one_variant_zero_per_teacher", np.array_equal(family[baseline_rows], np.arange(n)))
    projection = actions[baseline_rows]
    original_h, projection_h = metric(original), metric(projection)
    original_legal = np.array([r["legal"] for r in teacher_geo])
    delta = 100*(projection_h/original_h - 1)
    effects = {}
    for value, name in SPLITS.items():
        take = data["split"] == value
        effects[name] = {"noise_families": int(take.sum()),
            "original_legal_fraction": float(original_legal[take].mean()),
            "reprojected_legal_fraction": float(legal[baseline_rows][take].mean()),
            "original_macro_hpwl": summarize(original_h[take]),
            "reprojected_macro_hpwl": summarize(projection_h[take]),
            "paired_hpwl_change_percent": summarize(delta[take]),
            "coordinate_mse": summarize(np.square(projection-original).mean(axis=(1, 2))[take]),
            "identical_layout_count": int(np.sum(np.all(projection[take] == original[take], axis=(1, 2)))),
            "hpwl_improved_count": int(np.sum(delta[take] < -1e-6)),
            "hpwl_worsened_count": int(np.sum(delta[take] > 1e-6))}
    report = {"passed": all(checks.values()), "checks": checks,
              "n_teacher_families": n, "n_critic_rows": len(actions),
              "teacher_illegal_families": np.flatnonzero(~original_legal).tolist(),
              "critic_illegal_rows": np.flatnonzero(~legal).tolist(),
              "max_stored_hpwl_absolute_error": float(np.abs(h-data["macro_hpwl"]).max()),
              "max_stored_reward_absolute_error": float(np.abs(data["reward"]-(-h/train_scale-(~legal)*10)).max()),
              "teacher_final_projection_effect": effects,
              "independence_unit": "initial noise family; perturbations cannot cross splits",
              "warning": WARNING}
    write_json(output / "dataset_audit.json", report)
    if not report["passed"]:
        raise AssertionError(f"Dataset audit failed: {[k for k,v in checks.items() if not v]}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for value, name in SPLITS.items():
        take = data["split"] == value
        axes[0].scatter(original_h[take]/1000, projection_h[take]/1000, s=12, alpha=.65, label=name)
        axes[1].hist(delta[take], bins=np.linspace(min(-1, delta.min()), max(1, delta.max()), 30),
                     histtype="step", density=True, label=name)
    limits = [min(original_h.min(), projection_h.min())/1000, max(original_h.max(), projection_h.max())/1000]
    axes[0].plot(limits, limits, "k--", lw=1)
    axes[0].set(xlabel="Original teacher macro HPWL (thousand cache units)",
                ylabel="After one extra projection", title="Final legalization is not assumed idempotent")
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="Paired macro HPWL change (%)", ylabel="Density", title="Positive = extra projection worsened proxy")
    axes[0].legend(); axes[1].legend(); fig.tight_layout()
    save_fig(fig, output, "teacher_projection_effect")
    return report


def load_evaluations(training, phase, data, metric, scale, output):
    summary_path = training / f"measured_{phase}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    if phase == "heldout":
        selection = read_json(training / "selection.json")
        if sha(training / "measured_validation_summary.json") != selection["source_sha256"]:
            raise AssertionError("Validation selection source has changed")
        expected = set(selection["methods"]) | {"teacher_original", "teacher_relegalized"}
        if expected != set(summary["methods"]):
            raise AssertionError("Heldout methods differ from locked validation selection")
    result, audits = {}, {}
    expected_indices = np.flatnonzero(data["split"] == (1 if phase == "validation" else 2))
    for name in summary["methods"]:
        directory = training / f"measured_{phase}" / name
        records = read_json(directory / "records.json")
        arrays = read_npz(directory / "evaluated.npz")
        checks = {"complete_paired_indices": np.array_equal(arrays["indices"], expected_indices),
                  "records_match_saved_indices": [r["index"] for r in records] == arrays["indices"].tolist(),
                  "correct_split": np.array_equal(arrays["split"], data["split"][arrays["indices"]]),
                  "finite_coordinates": np.isfinite(arrays["action"]).all()}
        geometries = [independent_geometry(x, metric.sizes) for x in arrays["action"]]
        checks["independent_legality_matches"] = all(
            r[k] == g[k] for r, g in zip(records, geometries)
            for k in ("legal", "overlap_pairs", "out_of_bounds"))
        hpwl = metric(arrays["action"])
        checks["measured_hpwl_matches"] = np.allclose(hpwl, [r["macro_hpwl"] for r in records], rtol=1e-9)
        rewards = -hpwl/scale-np.array([not g["legal"] for g in geometries])*10
        checks["measured_reward_matches"] = np.allclose(rewards, [r["reward"] for r in records], rtol=1e-9)
        reported = summary["methods"][name][phase]
        checks["summary_matches_records"] = (reported["count"] == len(records) and
            np.isclose(reported["legal_fraction"], np.mean([g["legal"] for g in geometries])) and
            np.isclose(reported["macro_hpwl_all"]["mean"], hpwl.mean(), rtol=1e-12) and
            np.isclose(reported["reward"]["mean"], rewards.mean(), rtol=1e-12))
        if name.startswith("seed_"):
            proposal = read_npz(training / name / "proposals.npz")
            checks["proposal_noise_and_split_match"] = (np.array_equal(proposal["noise"], data["noise"])
                and np.array_equal(proposal["split"], data["split"]))
            predicted = proposal["predicted_q"][expected_indices]
            if predicted.ndim > 1:
                predicted = predicted.mean(axis=1)
            checks["q_values_match_proposals"] = np.allclose(predicted, [r["predicted_q"] for r in records])
            checks["q_error_matches_measured_reward"] = np.allclose(predicted-rewards, [r["q_error"] for r in records])
            mse = np.square(arrays["action"]-proposal["raw_action"][expected_indices].reshape(arrays["action"].shape)).mean(axis=(1, 2))
            checks["projection_mse_matches"] = np.allclose(mse, [r["legalization_mse"] for r in records], rtol=1e-5, atol=1e-8)
        audits[name] = {k: bool(v) for k,v in checks.items()}
        result[name] = {"records": records, **arrays}
    report = {"passed": all(all(v.values()) for v in audits.values()), "checks": audits,
              "phase": phase, "summary_sha256": sha(summary_path)}
    write_json(output / "evaluation_audit.json", report)
    if not report["passed"]:
        raise AssertionError("Evaluation audit failed; see evaluation_audit.json")
    return result


def series(item, key):
    return np.array([r[key] for r in item["records"]])


def grouped(results):
    groups = {}
    for name in results:
        seed, method = method_parts(name)
        groups.setdefault(method, []).append((seed, name))
    def order(method):
        for i, prefix in enumerate(("teacher_original", "teacher_relegalized", "distill_only",
                                    "fql_alpha_", "analytic_hpwl_alpha_", "coordinate_hpwl_")):
            if method.startswith(prefix):
                return (i, method)
        return (10, method)
    return {key: groups[key] for key in sorted(groups, key=order)}


def summarize_variability(results, output):
    base = results["teacher_relegalized"]
    base_h, base_legal = series(base, "macro_hpwl"), series(base, "legal")
    report, csv_rows = {}, []
    for method, entries in grouped(results).items():
        hpwl = np.array([series(results[name], "macro_hpwl") for _, name in entries])
        legal = np.array([series(results[name], "legal") for _, name in entries])
        reward = np.array([series(results[name], "reward") for _, name in entries])
        row = {"training_seeds": [s for s,_ in entries], "replicate_kind":
               "trained model" if any(s not in ("reference", "seed_control") for s,_ in entries) else "reference/control",
               "noise_families": hpwl.shape[1], "training_seed_count": len(entries),
               "between_training_seed_mean_hpwl_std": float(hpwl.mean(axis=1).std(ddof=1)) if len(entries)>1 else None,
               "within_seed_noise_hpwl_std_mean": float(hpwl.std(axis=1, ddof=1).mean()),
               "across_noise_seed_averaged_hpwl_std": float(hpwl.mean(axis=0).std(ddof=1)),
               "mean_macro_hpwl_all": float(hpwl.mean()), "legal_fraction": float(legal.mean()),
               "mean_measured_reward": float(reward.mean()), "per_seed": {}}
        for j, (seed, name) in enumerate(entries):
            paired = legal[j] & base_legal
            change = 100*(hpwl[j, paired]/base_h[paired] - 1)
            row["per_seed"][seed] = {"legal_fraction": float(legal[j].mean()),
                "macro_hpwl_all": summarize(hpwl[j]), "macro_hpwl_legal_only": summarize(hpwl[j, legal[j]]),
                "measured_reward": summarize(reward[j]),
                "paired_change_vs_projected_teacher_percent_legal_pairs": summarize(change)}
            csv_rows.append({"method": method, "seed": seed, "noise_families": len(hpwl[j]),
                             "legal_fraction": float(legal[j].mean()), "macro_hpwl_mean": float(hpwl[j].mean()),
                             "macro_hpwl_noise_std": float(hpwl[j].std(ddof=1)), "reward_mean": float(reward[j].mean()),
                             "paired_relative_hpwl_change_percent": float(change.mean()) if len(change) else ""})
        report[method] = row
    write_json(output / "variability_summary.json", {"methods": report,
        "interpretation": "Between-seed SD uses means over the same paired noise families. Noise SD is reported separately. No significance claim or pseudo-replicated confidence interval.",
        "warning": WARNING})
    with (output / "per_seed_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    return report


def plot_metrics(results, output, phase):
    groups = grouped(results)
    names = [name for entries in groups.values() for _, name in entries]
    colors = {method: plt.get_cmap("tab10")(i % 10) for i, method in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(max(9, len(names)*.7), 4))
    legal = [series(results[name], "legal").mean() for name in names]
    ax.bar(np.arange(len(names)), legal, color=[colors[method_parts(n)[1]] for n in names])
    ax.set(xticks=np.arange(len(names)), xticklabels=[short(n) for n in names], ylim=(0, 1.12),
           ylabel="Legal fraction", title=f"{phase.capitalize()}: independently checked geometry")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i, name in enumerate(names):
        valid = series(results[name], "legal")
        ax.text(i, legal[i]+.015, f"{valid.sum()}/{len(valid)}", ha="center", fontsize=8)
    fig.tight_layout(); save_fig(fig, output, "legal_fraction")

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, entries in groups.items():
        for j, (seed, name) in enumerate(entries):
            values = np.sort(series(results[name], "macro_hpwl")[series(results[name], "legal")])/1000
            if len(values):
                ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", color=colors[method],
                        alpha=.85, lw=1.5, ls=["-", "--", ":"][j % 3], label=short(method) if j == 0 else None)
    ax.set(xlabel="Macro-only HPWL (thousand cache units; lower is better)", ylabel="Empirical CDF",
           title="Legal layouts only; each line is one training seed/reference")
    ax.legend(fontsize=8); fig.tight_layout(); save_fig(fig, output, "macro_hpwl_ecdf")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for i, (method, entries) in enumerate(groups.items()):
        values = np.array([series(results[name], "macro_hpwl") for _,name in entries])/1000
        offsets = np.linspace(-.1, .1, len(entries)) if len(entries)>1 else [0]
        axes[0].scatter(i+np.asarray(offsets), values.mean(axis=1), color=colors[method], s=42, zorder=3)
        axes[1].boxplot([values.mean(axis=0)], positions=[i], widths=.5, showfliers=True,
                        medianprops={"color": colors[method]})
    labels = [short(n) for n in groups]
    for ax in axes:
        ax.set(xticks=np.arange(len(groups)), xticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    axes[0].set(ylabel="Macro HPWL (thousand cache units)", title="Training variation: each point = seed mean")
    axes[1].set(title="Noise variation: HPWL averaged across seeds")
    fig.tight_layout(); save_fig(fig, output, "training_seed_vs_noise_variation")

    baseline = results["teacher_relegalized"]
    base_h, base_legal = series(baseline, "macro_hpwl"), series(baseline, "legal")
    plot_names = [n for n in names if n != "teacher_relegalized"]
    changes = []
    for name in plot_names:
        paired = base_legal & series(results[name], "legal")
        changes.append(100*(series(results[name], "macro_hpwl")[paired]/base_h[paired]-1))
    fig, ax = plt.subplots(figsize=(max(9, len(plot_names)*.7), 4))
    ax.boxplot(changes, labels=[short(n) for n in plot_names], showfliers=True)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set(ylabel="Paired HPWL change vs projected teacher (%)",
           title="Same initial noise; only pairs with two legal layouts (negative is better)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout(); save_fig(fig, output, "paired_macro_hpwl_change")


def plot_layouts(results, metric, output):
    groups = grouped(results)
    chosen = [entries[0][1] for entries in groups.values()]
    index = int(results[chosen[0]]["indices"][0])
    columns = min(3, len(chosen)); rows = (len(chosen)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(5*columns, 4.5*rows), squeeze=False)
    for ax, name in zip(axes.flat, chosen):
        item = results[name]
        position = int(np.flatnonzero(item["indices"] == index)[0])
        centers = (item["action"][position]+1)*metric.canvas/2
        sizes = metric.sizes*metric.canvas/2
        for k, (center, size) in enumerate(zip(centers, sizes)):
            ax.add_patch(Rectangle(center-size/2, *size, facecolor=plt.get_cmap("tab20")(k % 20),
                                   edgecolor="black", linewidth=.7, alpha=.8))
            ax.text(*center, str(metric.names[k]), ha="center", va="center", fontsize=7)
        record = item["records"][position]
        ax.set(xlim=(0, metric.canvas[0]), ylim=(0, metric.canvas[1]), aspect="equal",
               xlabel="Cache x", ylabel="Cache y", title=f"{short(name)}\nHPWL {record['macro_hpwl']:,.0f}; legal={record['legal']}")
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle(f"Paired noise family {index}: first predeclared evaluation family, first training seed", fontsize=12)
    fig.tight_layout(h_pad=2.5); save_fig(fig, output, "paired_layouts_first_family")


def plot_q(results, output):
    qgroups = {method: entries for method, entries in grouped(results).items()
               if results[entries[0][1]]["records"][0].get("predicted_q") is not None}
    if not qgroups:
        return
    ncols = min(2, len(qgroups)); nrows = (len(qgroups)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 4*nrows), squeeze=False)
    fig_err, err_axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 4*nrows), squeeze=False)
    calibration = {}
    for ax, err_ax, (method, entries) in zip(axes.flat, err_axes.flat, qgroups.items()):
        calibration[method] = {}
        lows, highs = [], []
        for seed, name in entries:
            measured = series(results[name], "reward")
            predicted = series(results[name], "predicted_q")
            error = predicted-measured
            if not np.isfinite(predicted).all():
                raise ValueError(f"Nonfinite Q: {name}")
            calibration[method][seed] = {"count": len(error), "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean()),
                "pearson_r": float(np.corrcoef(measured, predicted)[0, 1])
                    if np.std(measured)>0 and np.std(predicted)>0 else None,
                "projection_mse": summarize(series(results[name], "legalization_mse")),
                "nearest_training_raw_mse": summarize(series(results[name], "nearest_training_raw_mse"))}
            ax.scatter(measured, predicted, s=16, alpha=.7, label=seed)
            err_ax.scatter(series(results[name], "nearest_training_raw_mse"), error, s=16, alpha=.7, label=seed)
            lows.append(min(measured.min(), predicted.min())); highs.append(max(measured.max(), predicted.max()))
        ax.plot([min(lows), max(highs)], [min(lows), max(highs)], "k--", lw=1)
        ax.set(xlabel="Measured proxy reward after projection", ylabel="Mean predicted Q", title=short(method))
        err_ax.axhline(0, color="k", ls="--", lw=1)
        err_ax.set(xlabel="Nearest training raw-action MSE", ylabel="Q minus measured proxy reward", title=short(method))
        ax.legend(fontsize=8); err_ax.legend(fontsize=8)
    for collection in (axes, err_axes):
        for ax in list(collection.flat)[len(qgroups):]:
            ax.set_visible(False)
    fig.suptitle("Critic calibration on actor proposals; identity line = accurate Q")
    fig_err.suptitle("Positive Q error = optimistic critic prediction")
    fig.tight_layout(); fig_err.tight_layout()
    save_fig(fig, output, "q_vs_measured_reward")
    save_fig(fig_err, output, "q_error_vs_training_support")
    write_json(output / "q_calibration_by_seed.json", {"methods": calibration, "warning": WARNING})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=["validation", "heldout"], default="heldout")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run plot/audit work in a Slurm CPU allocation")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.dataset_dir / "manifest.json")
    if manifest.get("reward_kind") != "macro_hpwl":
        raise ValueError("This script is for macro-proxy datasets, not routed/GRT labels")
    cache_path = ROOT / "baseline/runtime/benchmark_cache" / f"{manifest['args']['design']}.pt"
    if sha(cache_path) != manifest["provenance"]["cache_sha256"]:
        raise AssertionError("Official cache provenance mismatch")
    metric = MacroHPWL(read_cache(cache_path))
    data = read_npz(args.dataset_dir / "dataset.npz")
    dataset_audit = audit_dataset(data, manifest, metric, args.output)
    if not args.audit_only:
        if args.training_dir is None:
            raise ValueError("--training-dir is required for result figures")
        results = load_evaluations(args.training_dir, args.phase, data, metric,
                                   manifest["reward_scale_train_only"], args.output)
        summarize_variability(results, args.output)
        plot_metrics(results, args.output, args.phase)
        plot_layouts(results, metric, args.output)
        plot_q(results, args.output)
    provenance = {"dataset_dir": str(args.dataset_dir.resolve()),
                  "dataset_sha256": sha(args.dataset_dir / "dataset.npz"),
                  "cache_sha256": sha(cache_path), "script_sha256": sha(__file__),
                  "training_dir": str(args.training_dir.resolve()) if args.training_dir else None,
                  "phase": args.phase if not args.audit_only else "dataset_audit_only",
                  "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "outputs": sorted(p.name for p in args.output.iterdir()), "warning": WARNING}
    write_json(args.output / "figure_provenance.json", provenance)
    print(json.dumps({"output": str(args.output), "dataset_audit_passed": dataset_audit["passed"],
                      "artifacts": provenance["outputs"]}), flush=True)


if __name__ == "__main__":
    main()
