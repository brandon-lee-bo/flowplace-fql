"""Read-only, Slurm-only summaries of the fixed Q-study critic matrix.

No model is loaded, trained, advanced, or physically evaluated. Completed fold
predictions are checked against the audited dataset before OOF aggregation.
Stored utilities are reported per fold: different fold scales are never pooled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


METRICS = ("rWL_um", "WNS_ns", "TNS_ns", "horizontal_overflow_sum", "vertical_overflow_sum")
DEFINITIONS = ("GRT_wirelength_um", "max(-native_WNS_ns,0)", "max(-native_TNS_ns,0)",
               "GRT_horizontal_positive_demand_minus_capacity", "GRT_vertical_positive_demand_minus_capacity")
STATISTICS = ("mae", "normalized_mae", "spearman", "kendall", "top_regret", "family_pair_accuracy")
DEFAULT_MATRIX = Path("/fact_data/bohanli/flowplace_fql/qstudy_v1/screen_v1")
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def json_text(value):
    return json.dumps(clean(value), indent=2, allow_nan=False) + "\n"


def csv_text(rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(clean(v)) if isinstance(v, (dict, list, tuple)) else clean(v)
                         for k, v in row.items()})
    return buffer.getvalue()


def describe(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                minimum=float(values.min()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def diagnostics(prediction, truth, families):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    p, y, f = prediction[valid], truth[valid], families[valid]
    result = dict(n_rows=len(y), n_families=len(np.unique(f)), mae=None, spearman=None,
                  kendall=None, top_regret=None, family_pairs=0, family_ties=0,
                  family_pair_accuracy=None, family_prediction_ties=0, constant_truth=None)
    if not len(y):
        return result
    result.update(mae=float(np.abs(p-y).mean()), top_regret=float(y[np.argmin(p)]-y.min()),
                  constant_truth=bool(np.ptp(y) <= 1e-10))
    if len(y) > 2 and np.ptp(y) > 1e-10 and np.ptp(p) > 1e-10:
        result.update(spearman=float(spearmanr(p, y).statistic),
                      kendall=float(kendalltau(p, y).statistic))
    i, j = np.triu_indices(len(y), 1)
    same = f[i] == f[j]
    distinct = same & (np.abs(y[i]-y[j]) > 1e-9)
    result.update(family_pairs=int(distinct.sum()), family_ties=int((same & ~distinct).sum()))
    if distinct.any():
        result["family_prediction_ties"] = int((p[i[distinct]] == p[j[distinct]]).sum())
        result["family_pair_accuracy"] = float(np.mean(
            (p[i[distinct]]-p[j[distinct]]) * (y[i[distinct]]-y[j[distinct]]) > 0))
    return result


def family_means(prediction, truth, families):
    groups = np.unique(families)
    p = np.stack([prediction[families == group].mean(0) for group in groups])
    y = np.stack([truth[families == group].mean(0) for group in groups])
    return p, y, groups


def scheduler_evidence(job_id):
    if not job_id:
        return {}, dict(requested=False)
    if not job_id.isdigit():
        raise ValueError("--array-job-id must be a numeric Slurm array job ID")
    command = ["sacct", "-X", "-j", job_id, "--noheader", "--parsable2",
               "--format=JobID,State,ExitCode,Elapsed"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {}, dict(requested=True, error=str(error))
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0].startswith(job_id + "_") and "." not in fields[0]:
            states[fields[0]] = dict(job_id=fields[0], state=fields[1], exit_code=fields[2], elapsed=fields[3])
    return states, dict(requested=True, command=command, returncode=result.returncode,
                        stderr=result.stderr.strip(), records=len(states))


def load_dataset(task):
    path = Path(task["dataset"])
    if sha(path) != task["dataset_sha256"]:
        raise ValueError("Audited dataset changed: " + str(path))
    manifest = read_json(path.with_name("manifest.json"))
    if manifest["dataset_sha256"] != task["dataset_sha256"] or tuple(manifest["metric_names"][:5]) != METRICS:
        raise ValueError("Dataset manifest identity/metric order differs")
    if not manifest["no_test_or_heldout_exported"] or manifest["historical_online_included"]:
        raise ValueError("Dataset contamination flags reject reporting")
    data = {}
    with np.load(path, allow_pickle=False) as arrays:
        for split in ("train", "validation"):
            native = arrays[split + "_metrics"][:, :5].astype(float)
            mask = arrays[split + "_metric_mask"][:, :5].astype(bool)
            costs = native.copy()
            costs[:, 1:3] = np.maximum(-costs[:, 1:3], 0)
            costs[~mask] = np.nan
            successful = mask[:, 0] & (costs[:, 0] > 0)
            data[split] = dict(truth=costs[successful],
                family=arrays[split + "_family"][successful],
                executed=arrays[split + "_executed_action"][successful],
                source_indices=np.flatnonzero(successful), original_rows=len(successful),
                excluded_rows=int((~successful).sum()))
    if set(data["train"]["family"]) & set(data["validation"]["family"]):
        raise ValueError("TRAIN/validation family leakage")
    groups = np.unique(data["train"]["family"])
    np.random.default_rng(41307).shuffle(groups)
    data["folds"] = np.array_split(groups, task["folds"])
    data["manifest"] = manifest
    data["manifest_sha256"] = sha(path.with_name("manifest.json"))
    data["cache_path"] = ROOT / "baseline/runtime/benchmark_cache" / (manifest["design"] + ".pt")
    data["cache_sha256"] = sha(data["cache_path"])
    return data


def validate_binding(binding, task, dataset, plan):
    if binding["dataset_sha256"] != task["dataset_sha256"] or binding["manifest_sha256"] != dataset["manifest_sha256"]:
        raise ValueError("Run data/manifest binding differs")
    if binding["cache_sha256"] != dataset["cache_sha256"]:
        raise ValueError("Run netlist cache binding differs from the known current design cache")
    for key in ("dataset", "output", "variant", "seed", "steps", "folds"):
        if binding["config"][key] != task[key]:
            raise ValueError("Run configuration differs: " + key)
    if binding["heldout_used"] or binding["backend_calls"] != 0 or binding["fold_seed"] != 41307:
        raise ValueError("Screen run unexpectedly changed stage/data boundary")
    for name, digest in binding["source_sha256"].items():
        if plan["source_sha256"].get(name) != digest:
            raise ValueError("Run source differs from frozen matrix: " + name)


def read_fit(task, folder, fit_name, dataset, plan, verify_checkpoints):
    completed = folder / fit_name / "completed.json"
    if not completed.exists():
        return None
    done = read_json(completed)
    validate_binding(done["binding"], task, dataset, plan)
    if done["status"] != "completed" or done["steps"] != task["steps"]:
        raise ValueError("Unexpected fit completion/update budget")
    if done["variant"] != task["variant"] or done["seed"] != task["seed"]:
        raise ValueError("Fit identity differs")
    split = "validation" if fit_name == "refit" else "train"
    source = dataset[split]
    prediction_file = folder / fit_name / "predictions.npz"
    with np.load(prediction_file, allow_pickle=False) as arrays:
        arrays = {key: arrays[key].copy() for key in arrays.files}
    target = arrays["target_indices"]
    train = arrays["train_indices"]
    if not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(train.dtype, np.integer):
        raise ValueError("Noninteger prediction row indices")
    if fit_name == "refit":
        expected_target = np.arange(len(source["family"]))
        expected_train = np.arange(len(dataset["train"]["family"]))
    else:
        held = dataset["folds"][int(fit_name.split("_")[1])]
        expected_target = np.flatnonzero(np.isin(source["family"], held))
        expected_train = np.flatnonzero(~np.isin(source["family"], held))
        if set(source["family"][expected_target]) & set(source["family"][expected_train]):
            raise ValueError("CV family leakage")
    if not np.array_equal(target, expected_target) or not np.array_equal(train, expected_train):
        raise ValueError("Prediction rows differ from declared grouped split")
    if not np.array_equal(arrays["family"], source["family"][target]):
        raise ValueError("Prediction family IDs differ from audited rows")
    if not np.allclose(arrays["truth"], source["truth"][target], rtol=2e-6, atol=1e-6, equal_nan=True):
        raise ValueError("Prediction truth differs from audited physical costs")
    expected_columns = 1 if task["variant"] == "legacy" else len(METRICS)
    if arrays["prediction"].shape != (len(target), expected_columns) or not np.isfinite(arrays["prediction"]).all():
        raise ValueError("Prediction dimension/nonfinite value differs")
    if done["train_rows"] != len(train) or done["target_rows"] != len(target):
        raise ValueError("Fit row count differs from predictions")
    if verify_checkpoints and sha(folder / fit_name / "resume.pt") != done["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint SHA differs")
    return dict(done=done, arrays=arrays, completed_sha256=sha(completed),
                predictions_sha256=sha(prediction_file), split=split)


def summarize_seeds(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"], row["scope"], row["aggregation"], row["metric"])].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        if len({v["seed"] for v in values}) != len(values):
            raise ValueError("Seed summary contains duplicate seed estimates")
        row = dict(zip(("dataset", "variant", "scope", "aggregation", "metric"), key))
        row.update(seeds=sorted(v["seed"] for v in values), n_seeds=len(values))
        for statistic in STATISTICS:
            for name, value in describe([v.get(statistic) for v in values]).items():
                row[statistic + "_" + name] = value
        result.append(row)
    return result


def mechanism_proposals(rows, tasks, plan):
    """Descriptive TRAIN-CV ranks only; never outputs an executable selection."""
    indexed = {(r["dataset"], r["variant"], r["seed"], r["metric"]): r for r in rows
               if r["scope"] == "train_cv_oof" and r["aggregation"] == "rows" and r["complete_cv"]}
    datasets = [Path(task["dataset"]).parent.name for task in tasks]
    datasets = sorted(set(datasets))
    complete = all((d, v, s, METRICS[0]) in indexed
                   for d in datasets for v in plan["variants"] for s in plan["seeds"])
    if not complete:
        return dict(status="withheld_incomplete_TRAIN_CV", automatic_advancement=False,
                    explanation="No fast-finisher selection: wait for all planned variants/seeds/datasets.", candidates=[]), []
    tracks = (("wl_first", plan["variants"], (METRICS[0],)),
              ("balanced_wl_timing", [v for v in plan["variants"] if v != "legacy"], METRICS[:3]))
    rankings = []
    for track, variants, metrics in tracks:
        by_variant = defaultdict(list)
        for dataset in datasets:
            for seed in plan["seeds"]:
                components = defaultdict(list)
                for metric in metrics:
                    scores = [indexed.get((dataset, v, seed, metric), {}).get("spearman") for v in variants]
                    if any(score is None for score in scores):
                        return dict(status="withheld_missing_comparable_metric", automatic_advancement=False,
                                    explanation=f"Missing comparable Spearman for {track}/{dataset}/{seed}/{metric}", candidates=[]), []
                    for variant, rank in zip(variants, rankdata(-np.asarray(scores), method="average")):
                        components[variant].append(float(rank))
                for variant in variants:
                    by_variant[variant].append(dict(dataset=dataset, seed=seed,
                                                   rank=float(np.mean(components[variant]))))
        for variant, values in by_variant.items():
            # Average designs within a training seed before describing seed SD.
            per_seed = [np.mean([r["rank"] for r in values if r["seed"] == seed]) for seed in plan["seeds"]]
            stats = describe(per_seed)
            rankings.append(dict(track=track, variant=variant, mean_rank=stats["mean"],
                                 rank_sd_across_train_seeds=stats["sd"], n_train_seeds=stats["n"],
                                 n_designs=len(datasets), components=list(metrics), strata=values))
    candidates, used = [], set()
    for track, _, _ in tracks:
        eligible = sorted((r for r in rankings if r["track"] == track and r["variant"] != "legacy"),
                          key=lambda r: (r["mean_rank"], r["variant"]))
        choice = next((r for r in eligible if r["variant"] not in used), None)
        if choice:
            used.add(choice["variant"])
            candidates.append(dict(track=track, variant=choice["variant"], mean_rank=choice["mean_rank"],
                unrestricted_track_best=eligible[0]["variant"], distinct_mechanism_rule=True,
                interpretation="Provisional mechanism to inspect; not a configuration selected for advancement."))
    return dict(status="provisional_descriptive_TRAIN_CV_only", automatic_advancement=False,
        selection_rule="Rank within each dataset/seed by raw-cost OOF Spearman; WL alone, or equal WL/WNS-violation/TNS-violation component ranks. Average designs within seed, then seeds. Keep two distinct nonlegacy mechanisms; alphabetical tie-break.",
        caveat="Ranks omit gradient validity and backend actor gains; legacy remains the anchor. No historical-validation tuning enters these proposals.",
        candidates=candidates), rankings


def paired_deltas(rows):
    indexed = {(r["dataset"], r["variant"], r["seed"], r["scope"], r["aggregation"], r["metric"]): r
               for r in rows if r.get("complete_cv", True)}
    groups = defaultdict(list)
    for key, row in indexed.items():
        dataset, variant, seed, scope, aggregation, metric = key
        for baseline in ("legacy", "multi_huber"):
            if variant == baseline:
                continue
            reference = indexed.get((dataset, baseline, seed, scope, aggregation, metric))
            if reference is None:
                continue
            for statistic in STATISTICS:
                a, b = row.get(statistic), reference.get(statistic)
                if a is None or b is None:
                    continue
                groups[(dataset, variant, baseline, scope, aggregation, metric, statistic)].append((seed, a-b))
    result = []
    fields = ("dataset", "variant", "baseline", "scope", "aggregation", "metric", "statistic")
    for key, values in sorted(groups.items()):
        statistic = key[-1]
        lower = statistic in {"mae", "normalized_mae", "top_regret"}
        result.append(dict(zip(fields, key), **describe([v for _, v in values]),
            seeds=[s for s, _ in values], deltas=[v for _, v in values], definition="variant minus baseline",
            lower_delta_is_better=lower, improved_seeds=sum(v < 0 if lower else v > 0 for _, v in values)))
    return result


def aggregate(args):
    matrix = args.matrix.resolve()
    output = (args.output or matrix / "report").resolve()
    plan = read_json(matrix / "plan.json")
    task_path = matrix / "tasks.jsonl"
    if sha(task_path) != plan["tasks_sha256"]:
        raise ValueError("Frozen matrix tasks changed")
    tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    if len(tasks) != plan["count"]:
        raise ValueError("Matrix task count differs")
    if output == matrix or any(output == Path(t["output"]).resolve() or Path(t["output"]).resolve() in output.parents for t in tasks):
        raise ValueError("Report must not write into matrix task directories")
    states, scheduler = scheduler_evidence(args.array_job_id)
    datasets, status_rows, fit_rows, pooled_rows, prediction_rows, family_rows = {}, [], [], [], [], []
    provenance, errors = [], []
    for index, task in enumerate(tasks):
        folder = Path(task["output"])
        dataset_name = Path(task["dataset"]).parent.name
        identity = dict(dataset=dataset_name, variant=task["variant"], seed=task["seed"])
        status = dict(task_index=index, **identity, declared_steps=task["steps"], folds_expected=task["folds"],
                      completed_folds=0, completed_refit=False, root_completed=(folder / "completed.json").exists(),
                      scheduler=states.get(f"{args.array_job_id}_{index}"), integrity_errors=[])
        try:
            if task["dataset"] not in datasets:
                datasets[task["dataset"]] = load_dataset(task)
            dataset = datasets[task["dataset"]]
            if status["root_completed"]:
                root_done = read_json(folder / "completed.json")
                validate_binding(root_done["binding"], task, dataset, plan)
                if root_done["status"] != "completed" or len(root_done["outcomes"]) != task["folds"] + 1:
                    raise ValueError("Matrix task completion marker differs from expected fits")
            completed_cv = []
            for name in [f"cv_{fold}" for fold in range(task["folds"])] + ["refit"]:
                fit = read_fit(task, folder, name, dataset, plan, args.verify_checkpoints)
                if fit is None:
                    continue
                done, arrays = fit["done"], fit["arrays"]
                scope = "historical_validation" if name == "refit" else "train_cv_fold"
                if name == "refit":
                    status["completed_refit"] = True
                else:
                    status["completed_folds"] += 1
                    completed_cv.append((name, arrays))
                provenance.append(dict(**identity, fit=name, completed_sha256=fit["completed_sha256"],
                    predictions_sha256=fit["predictions_sha256"], checkpoint_sha256_declared=done["checkpoint_sha256"],
                    checkpoint_rehashed=args.verify_checkpoints, slurm_job_id=done["slurm_job_id"]))
                for kind, metrics in (("physical_cost", done["metrics"]), ("fold_scaled_utility", done["objectives"])):
                    for metric, values in metrics.items():
                        fit_rows.append(dict(**identity, fit=name, scope=scope, kind=kind, metric=metric,
                            steps=done["steps"], train_rows=done["train_rows"], target_rows=done["target_rows"], **values))
                if name == "refit":
                    add_pooled(identity, scope, arrays, dataset["validation"], True, pooled_rows,
                               prediction_rows, family_rows, fit_names=[name],
                               reporting_reference=dataset["train"]["truth"])
            if completed_cv:
                combined = {key: np.concatenate([arrays[key] for _, arrays in completed_cv])
                            for key in ("prediction", "truth", "family", "target_indices")}
                indices = combined["target_indices"]
                if len(np.unique(indices)) != len(indices):
                    raise ValueError("A TRAIN row has more than one OOF prediction")
                complete_cv = len(completed_cv) == task["folds"]
                if complete_cv and not np.array_equal(np.sort(indices), np.arange(len(dataset["train"]["family"]))):
                    raise ValueError("OOF predictions do not exhaust audited TRAIN rows")
                combined["fold"] = np.concatenate([np.full(len(a["family"]), name) for name, a in completed_cv])
                add_pooled(identity, "train_cv_oof" if complete_cv else "train_cv_oof_partial", combined,
                    dataset["train"], complete_cv, pooled_rows, prediction_rows, family_rows,
                    fit_names=[name for name, _ in completed_cv],
                    reporting_reference=dataset["train"]["truth"])
            fits_complete = status["completed_folds"] == task["folds"] and status["completed_refit"]
            status["status"] = "completed" if fits_complete and status["root_completed"] else "partial" if status["completed_folds"] or status["completed_refit"] else "no_completed_fit"
        except (OSError, ValueError, KeyError, IndexError) as error:
            status["status"] = "integrity_error"
            status["integrity_errors"].append(str(error))
            errors.append(dict(**identity, error=str(error)))
        status_rows.append(status)
    # Never rank an integrity-invalid task using the folds read before failure.
    invalid = {(r["dataset"], r["variant"], r["seed"]) for r in status_rows if r["status"] == "integrity_error"}
    def valid_rows(rows):
        return [r for r in rows if (r["dataset"], r["variant"], r["seed"]) not in invalid]
    fit_rows, pooled_rows, prediction_rows, family_rows = map(
        valid_rows, (fit_rows, pooled_rows, prediction_rows, family_rows))
    proposals, ranking_rows = mechanism_proposals(pooled_rows, tasks, plan)
    if any(row["status"] != "completed" for row in status_rows):
        proposals = dict(status="withheld_incomplete_matrix", automatic_advancement=False,
                         explanation="No mechanism proposals until all declared tasks complete without integrity errors.",
                         candidates=[])
        ranking_rows = []
    counts = dict(total=len(tasks), completed=sum(r["status"] == "completed" for r in status_rows),
                  cv_complete=sum(r["completed_folds"] == r["folds_expected"] and r["status"] != "integrity_error" for r in status_rows),
                  refit_complete=sum(r["completed_refit"] and r["status"] != "integrity_error" for r in status_rows),
                  integrity_errors=len(errors))
    snapshot = dict(version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ["SLURM_JOB_ID"], matrix=str(matrix), plan_sha256=sha(matrix/"plan.json"),
        tasks_sha256=sha(task_path), report_source_sha256=sha(__file__), snapshot_counts=counts,
        complete=counts["completed"] == counts["total"] and not errors,
        automatic_advancement=False, backend_calls=0, training_updates=0,
        scheduler_query=scheduler, errors=errors, proposals=proposals,
        semantics=dict(metrics=dict(zip(METRICS, DEFINITIONS)),
            oof="Concatenate disjoint/exhaustive held-out TRAIN rows across folds in inverse-transformed physical-cost units; partial folds use a separate scope.",
            family_means="Average the two variants within each family; supplementary to row ranks and within-family pair accuracy.",
            utilities="Stored fold utilities have different TRAIN-fitted scales; report per fold only; no pooled utility Spearman or beta reconstruction.",
            normalized_mae="Pooled reporting MAE divided by full offline TRAIN mean absolute cost; this is only a reporting scale, never a fold model fit.",
            uncertainty="Sample SD across independent model-training seeds, not confidence intervals; folds, critic members and data rows are not independent training seeds.",
            historical_validation="16 previously inspected rows / 8 noise families per design; tuning evidence, not fresh final evaluation.",
            top_regret="Best predicted candidate's true cost minus best true cost in this fixed candidate pool; not actor improvement.",
            pair_ties="True absolute differences <=1e-9 are excluded; predicted ties on a true non-tie count as incorrect, matching the frozen runner.",
            failures="Missing completion does not imply physical failure. Scheduler failure/preemption and integrity errors remain separate; no invented reward."),
        datasets={Path(path).parent.name: dict(train_rows=len(data["train"]["family"]),
            validation_rows=len(data["validation"]["family"]), excluded_train=data["train"]["excluded_rows"],
            excluded_validation=data["validation"]["excluded_rows"], dataset_sha256=data["manifest"]["dataset_sha256"],
            cache_path=str(data["cache_path"]), cache_sha256=data["cache_sha256"])
            for path, data in datasets.items()})
    tables = dict(tasks=status_rows, fit_metrics=fit_rows, pooled_metrics=pooled_rows,
                  seed_summary=summarize_seeds(pooled_rows), paired_deltas=paired_deltas(pooled_rows),
                  mechanism_ranks=ranking_rows, pooled_predictions=prediction_rows, family_predictions=family_rows)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_job{os.environ['SLURM_JOB_ID']}_{time.time_ns()%1000000:06d}"
    destination = output / "snapshots" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        for suffix, content in (("json", json_text(rows)), ("csv", csv_text(rows))):
            atomic_text(destination / (name + "." + suffix), content)
            atomic_text(output / (name + "." + suffix), content)
    atomic_text(destination / "provenance.json", json_text(provenance))
    snapshot["snapshot_directory"] = str(destination)
    text = render_markdown(snapshot, tables["seed_summary"])
    for name, content in (("summary.json", json_text(snapshot)), ("report.md", text)):
        atomic_text(destination / name, content)
        atomic_text(output / name, content)
    atomic_text(output / "latest.json", json_text(dict(snapshot_directory=str(destination), summary=str(destination/"summary.json"))))
    print(json.dumps(clean(dict(output=str(output), counts=counts, proposals=proposals)), allow_nan=False), flush=True)
    return 2 if errors else 0


def add_pooled(identity, scope, arrays, source, complete_cv, metrics_rows, prediction_rows, family_rows,
               fit_names, reporting_reference):
    prediction, truth, families = arrays["prediction"], arrays["truth"], arrays["family"]
    fp, fy, ff = family_means(prediction, truth, families)
    reference = reporting_reference
    for aggregation, p, y, f in (("rows", prediction, truth, families), ("family_means", fp, fy, ff)):
        for k, metric in enumerate(METRICS[:p.shape[1]]):
            stats = diagnostics(p[:, k], y[:, k], f)
            scale = float(np.nanmean(np.abs(reference[:, k])))
            stats["normalized_mae"] = stats["mae"] / scale if stats["mae"] is not None and scale > 1e-8 else None
            metrics_rows.append(dict(**identity, scope=scope, aggregation=aggregation, metric=metric,
                target_definition=DEFINITIONS[k], complete_cv=complete_cv, available_fits=fit_names,
                reporting_scale=scale, **stats))
    for row, target in enumerate(arrays["target_indices"]):
        for k, metric in enumerate(METRICS[:prediction.shape[1]]):
            prediction_rows.append(dict(**identity, scope=scope, target_index=int(target),
                source_row=int(source["source_indices"][target]), family=int(families[row]),
                fit=str(arrays["fold"][row]) if "fold" in arrays else "refit", metric=metric,
                prediction=float(prediction[row, k]), truth=float(truth[row, k])))
    for row, family in enumerate(ff):
        for k, metric in enumerate(METRICS[:fp.shape[1]]):
            family_rows.append(dict(**identity, scope=scope, family=int(family), metric=metric,
                n_variants=int((families == family).sum()), prediction=float(fp[row, k]), truth=float(fy[row, k])))


def render_markdown(snapshot, summaries):
    count = snapshot["snapshot_counts"]
    lines = ["# Q-study critic screening snapshot", "", f"Generated {snapshot['generated_utc']} · Slurm {snapshot['slurm_job_id']}", "",
             f"Completed tasks: {count['completed']}/{count['total']}; complete TRAIN CV: {count['cv_complete']}; historical-validation refits: {count['refit_complete']}; integrity errors: {count['integrity_errors']}.", "",
             "This report does not select or advance an actor configuration. Partial runs are listed, not counted as failed placements. Historical validation has already been inspected.", "",
             "## Complete pooled TRAIN-CV estimates across training seeds", "",
             "Each seed combines all three held-out folds in physical cost units. ± is sample SD across independent training seeds; n is explicit. WNS/TNS below mean positive violation costs, not signed native slacks. Correlation does not establish a useful actor gradient.", "",
             "| Dataset | Variant | Metric | Seeds | Spearman mean ± SD | MAE mean | Family-pair accuracy mean |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    fmt = lambda x: "NA" if x is None else f"{x:.4g}"
    for row in summaries:
        if row["scope"] != "train_cv_oof" or row["aggregation"] != "rows":
            continue
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['metric']} | {row['n_seeds']} | {fmt(row['spearman_mean'])} ± {fmt(row['spearman_sd'])} | {fmt(row['mae_mean'])} | {fmt(row['family_pair_accuracy_mean'])} |")
    lines += ["", "## Provisional mechanism proposals", "", snapshot["proposals"]["status"], ""]
    for candidate in snapshot["proposals"]["candidates"]:
        lines.append(f"- {candidate['track']}: {candidate['variant']} (mean within-stratum rank {candidate['mean_rank']:.3f}); inspection proposal only.")
    lines += ["", "Stored utility scores remain fold-specific because each fold fits its own utility scales. Family-mean ranks and within-family pair results are supplementary views, not additional independent training runs. The two candidate tracks are exploratory proposals, not preregistered evidence of physical improvement.", "",
              "CSV and JSON tables retain individual fits, OOF rows/families, training-seed summaries, matched-seed deltas, scheduler status and input hashes. Comparisons in incomplete snapshots may have different seed coverage; use paired_deltas for matched seeds and wait for complete coverage before proposing mechanisms.", "",
              "The frozen local-difference/global-ranking matrix changes both pair scope and loss type. A locality claim still requires matched-pair local-ranking controls; multihead utility claims require directly fitted scalar-utility controls.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--array-job-id")
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run report aggregation through Slurm")
    raise SystemExit(aggregate(args))


if __name__ == "__main__":
    main()
