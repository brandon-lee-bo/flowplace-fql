"""Immutable, read-only Slurm reports of declared Q-study actor experiments.

Seven GRT measurements stay in native units. Partial requests are not failures;
physical failures have utility but never fabricated PPA. Independent seed means
are aggregated only after every declared matched request is accounted for.
No checkpoint, actor, or physical tool is run, selected, advanced, or modified.
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
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("rWL_um", "WNS_ns", "TNS_ns", "horizontal_overflow_sum",
           "vertical_overflow_sum", "Power_W", "StdCellArea_um2")
ADMITTED = {"success", "geometry_failure", "physical_failure"}


def read(path):
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
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def describe(values):
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                sd=float(x.std(ddof=1)) if len(x) > 1 else None)


def mean(values):
    return describe(values)["mean"]


def costs(metrics):
    result = np.asarray([metrics.get(k, np.nan) for k in METRICS[:5]], dtype=float)
    result[1:3] = np.maximum(-result[1:3], 0)
    return result


def utility(metrics, weights, scales):
    c = costs(metrics)
    active = np.asarray(weights, dtype=float) != 0
    if not np.isfinite(c[active]).all():
        return None
    return float(-np.sum(np.asarray(weights)[active] * c[active] / scales[active]))


def train_scales(cfg):
    """Match critic objective magnitudes using observed TRAIN rows only."""
    from .qstudy_data import load_dataset
    path = Path(cfg["dataset_path"])
    if sha(path) != cfg["dataset_sha256"]:
        raise ValueError("Actor TRAIN dataset SHA changed")
    data, manifest = load_dataset(path)
    values = data["train"]["metrics"][:, :5].astype(float).copy()
    observed = data["train"]["metric_mask"][:, :5].astype(bool) & np.isfinite(values)
    values[:, 1:3] = np.maximum(-values[:, 1:3], 0)
    magnitudes = np.asarray([np.abs(values[observed[:, k], k]).mean()
                            if observed[:, k].any() else 0 for k in range(5)])
    scales = np.where(magnitudes > 1e-8, magnitudes, 1.)
    return scales, dict(dataset_sha256=sha(path), manifest_sha256=sha(path.with_name("manifest.json")),
        normalization="mean observed nonnegative cost on explicit initial TRAIN only; floor to1 if <=1e-8",
        metric_names=list(METRICS[:5]), scales=scales.tolist(),
        observed_counts=observed.sum(0).tolist(), historical_online_included=manifest["historical_online_included"])


def curve(folder):
    path = folder / "learning_curve.jsonl"
    if not path.exists():
        return dict(available=False)
    try:
        payload = path.read_bytes()
    except OSError as error:
        return dict(available=False, parse_status="integrity_invalid", integrity_error=str(error))
    metadata = dict(available=True, path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
        captured_bytes=len(payload), fingerprint_scope="Exact bytes captured for this snapshot; active file may later grow",
        parse_status="complete", truncated_final_line=False)
    lines = []
    chunks = payload.splitlines(keepends=True)
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            row = json.loads(chunk)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if index == len(chunks)-1 and not chunk.endswith((b"\n", b"\r")):
                # A writer can be interrupted partway through its final append.
                # Keep complete prior records without admitting the partial row.
                metadata.update(parse_status="partial_trailing_record", truncated_final_line=True,
                    ignored_final_bytes=len(chunk), ignored_line_number=index+1)
                continue
            return dict(metadata, parse_status="integrity_invalid", integrity_error=
                f"Malformed committed/interior JSONL record at line {index+1}: {error}",
                logged_rows=len(lines), diagnostics_withheld=True)
        if not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 1:
            return dict(metadata, parse_status="integrity_invalid", integrity_error=
                f"Invalid learning-curve row or step at line {index+1}", diagnostics_withheld=True)
        for space in ("action_gradients", "parameter_gradients"):
            if space in row and (not isinstance(row[space], dict) or any(
                    type(row[space].get(k)) not in (int, float) or not math.isfinite(row[space][k])
                    for k in ("gradient_cosine", "q_grad_norm", "distill_grad_norm"))):
                return dict(metadata, parse_status="integrity_invalid", integrity_error=
                    f"Invalid gradient diagnostics at line {index+1}", diagnostics_withheld=True)
        lines.append(row)
    # Resume may append a repeated step; retain the last record for that step.
    rows = list({int(r["step"]): r for r in lines}.values())
    rows.sort(key=lambda r: r["step"])
    result = dict(metadata, logged_rows=len(lines),
                  unique_steps=len(rows), latest=rows[-1] if rows else None)
    for space in ("action_gradients", "parameter_gradients"):
        diagnostics = [r[space] for r in rows if space in r]
        result[space] = dict(n=len(diagnostics),
            cosine=describe([d.get("gradient_cosine") for d in diagnostics]),
            q_norm=describe([d.get("q_grad_norm") for d in diagnostics]),
            distill_norm=describe([d.get("distill_grad_norm") for d in diagnostics]),
            q_over_distill_norm=describe([d["q_grad_norm"] / d["distill_grad_norm"]
                for d in diagnostics if d.get("distill_grad_norm", 0) > 0]))
    return result


def check_training(folder, cfg, seed, round_number, config_sha, curve_snapshot=None):
    diagnostics = curve(folder) if curve_snapshot is None else curve_snapshot
    marker = folder / "completed.json"
    if not marker.exists():
        return dict(completed=False, curve=diagnostics)
    done = read(marker)
    binding = read(folder / "binding.json")
    checkpoint = folder / "checkpoint.pt"
    if done["status"] != "completed" or done["binding"] != binding:
        raise ValueError("Actor completion/binding mismatch")
    if (binding["config_sha256"] != config_sha or binding["dataset_sha256"] != cfg["dataset_sha256"]
            or binding["seed"] != seed or binding["round"] != round_number
            or done["checkpoint_sha256"] != sha(checkpoint)
            or binding["common_sha256"] != cfg["common_checkpoint_sha256"][str(seed)]
            or binding["initial_critic_sha256"] != cfg["critic_checkpoint_sha256"][str(seed)]):
        raise ValueError("Actor completed checkpoint identity changed")
    expected_steps = cfg["offline_actor_steps"] if round_number == 0 else cfg["online_updates"]
    if done["steps"] != expected_steps:
        raise ValueError("Actor step budget differs from declaration")
    for name, digest in binding["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen actor training source changed: " + name)
    return dict(completed=True, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                completion_sha256=sha(marker), binding_sha256=sha(folder / "binding.json"),
                steps=done["steps"], curve=diagnostics)


def native_fingerprints(out):
    if not out.get("physical_run"):
        return {}
    run = Path(out["physical_run"])
    result = dict(physical_run=str(run), physical_manifest_sha256=sha(run / "manifest.json"),
        physical_flow_revision=out["physical_flow_revision"], physical_slurm_job_id=out.get("physical_slurm_job_id"))
    for name in ("metrics.json", "backend_macro_audit.json", "floorplan_place_cts.log"):
        if (run / name).is_file():
            result[name + "_sha256"] = sha(run / name)
    for file_name, field in (("metrics.json", "metrics_sha256"),
                             ("backend_macro_audit.json", "backend_macro_geometry_sha256"),
                             ("floorplan_place_cts.log", "failure_log_sha256")):
        if out.get(field) and result.get(file_name + "_sha256") != out[field]:
            raise ValueError("Native evidence SHA differs from outcome: " + file_name)
    if out["status"] == "success":
        geometry = read(run / "backend_macro_audit.json")
        if out.get("backend_macro_legal") is not True or geometry.get("legal_no_halo") is not True:
            raise ValueError("Successful PPA requires an admitted legal actual backend geometry")
        if out.get("backend_track_snapped_macro_count") != len(geometry["input_to_actual_changes"]):
            raise ValueError("Saved native track-snapping count differs")
    return result


def summarize_rows(rows, expected):
    valid = [r for r in rows if r["state"] in ADMITTED]
    successful = [r for r in valid if r["state"] == "success"]
    failures = [r for r in valid if r["state"] != "success"]
    counts = {name: sum(r["state"] == name for r in rows)
              for name in ("success", "geometry_failure", "physical_failure", "missing", "invalid")}
    qrows = [r for r in valid if r["predicted_q"] is not None and r["measured_utility"] is not None]
    p = np.asarray([r["predicted_q"] for r in qrows])
    y = np.asarray([r["measured_utility"] for r in qrows])
    corr = float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 and np.ptp(p) > 0 and np.ptp(y) > 0 else None
    return dict(planned=expected, exported=len(rows), **counts,
        not_exported=max(expected - len(rows), 0),
        fully_accounted=len(rows) == expected and len(valid) == expected,
        metrics_conditional_on_success=bool(failures),
        successful_native_metric_means={k: mean([r["metrics"].get(k) for r in successful]) for k in METRICS},
        utility_mean=mean([r["measured_utility"] for r in valid]),
        raw_to_legalizer_rms=describe([r["raw_to_legalizer_rms"] for r in rows if "raw_to_legalizer_rms" in r]),
        q_calibration=dict(n=len(qrows), pearson=corr,
            mae=float(np.abs(p-y).mean()) if len(y) else None,
            signed_prediction_minus_measurement=float((p-y).mean()) if len(y) else None,
            uncertainty_note="Exported predicted Q may include an uncalibrated ensemble penalty; realized utility uses true GRT/failure feedback"))


def inspect_round(experiment, cfg, seed, round_number, scales, backend_context):
    from .graph_o2o_backend import load_requests
    from .qstudy_backend import checked_existing, checked_request
    folder = experiment / f"seed_{seed}" / f"round_{round_number}"
    config_sha = sha(experiment / "config.json")
    record = dict(experiment=str(experiment), arm=cfg["arm"], design=cfg["design"], seed=seed,
                  round=round_number, errors=[], rows=[], roles={})
    # Export roles follow actor config, even if a controller/report stops earlier.
    role_counts = dict(evaluation=cfg["evaluation_samples"])
    if round_number < cfg.get("online_rounds", 2):
        role_counts["collect"] = cfg["online_samples"]
    diagnostics = curve(folder)
    if diagnostics.get("parse_status") == "integrity_invalid":
        record["errors"].append("learning_curve: " + diagnostics["integrity_error"])
    try:
        record["training"] = check_training(folder, cfg, seed, round_number, config_sha, diagnostics)
    except Exception as error:
        record["training"] = dict(completed=False, integrity_error=str(error), curve=diagnostics)
        record["errors"].append("training: " + str(error))
    tasks = folder / "requests/tasks.jsonl"
    if not tasks.exists() or not tasks.with_name("completed.json").exists():
        for role, count in role_counts.items():
            record["roles"][role] = summarize_rows([], count)
        record["status"] = "integrity_invalid" if record["errors"] else "partial_not_exported"
        return record
    try:
        requests, tasks = load_requests(experiment, seed, round_number)
        if set(r["role"] for r in requests) != set(role_counts):
            raise ValueError("Export roles differ from declared actor plan")
        for role, count in role_counts.items():
            subset = [r for r in requests if r["role"] == role]
            if len(subset) != count or {r["index"] for r in subset} != set(range(count)):
                raise ValueError("Export role count/index differs from declaration")
        record["tasks_sha256"] = sha(tasks)
        record["export_completion_sha256"] = sha(tasks.with_name("completed.json"))
    except Exception as error:
        record["errors"].append("request manifest: " + str(error))
        record["status"] = "integrity_invalid"
        for role, count in role_counts.items():
            record["roles"][role] = summarize_rows([], count)
        return record
    bcfg, verification, binding = backend_context
    for row in requests:
        item = {k: row.get(k) for k in ("arm", "role", "index", "noise_sha256", "noise_seed",
            "raw_action_sha256", "action_sha256", "checkpoint_sha256", "sample_dir", "predicted_q")}
        item.update(state="missing", metrics=None, measured_utility=None)
        try:
            sample = checked_request(row, argparse.Namespace(seed=seed, round=round_number), bcfg, binding)
            if row["arm"] != cfg["arm"]:
                raise ValueError("Actor request arm differs")
            expected_noise_seed = (cfg["evaluation_noise_seed"] if row["role"] == "evaluation" else
                cfg["collection_noise_seed"]+seed*100+round_number*1000)
            if row["noise_seed"] != expected_noise_seed:
                raise ValueError("Actor request noise stream differs from declaration")
            if not record["training"]["completed"] or row["checkpoint_sha256"] != record["training"]["checkpoint_sha256"]:
                raise ValueError("Request is not bound to completed declared actor")
            item["request_sha256"] = sha(sample / "request.json")
            item["actions_npz_sha256"] = sha(sample / "actions.npz")
            raw, executed = np.asarray(row["raw_action"]), np.asarray(row["executed_action"])
            # Fixed ports have zero delta; use all-coordinate RMS because actor exports
            # do not carry the movable mask. This definition is explicit in metadata.
            item["raw_to_legalizer_rms"] = float(np.sqrt(np.mean((executed-raw)**2)))
            value = np.asarray(row.get("predicted_q"), dtype=float).reshape(-1)
            item["predicted_q"] = float(value.mean()) if len(value) and np.isfinite(value).all() else None
            path = sample / "outcome.json"
            if path.exists():
                out = read(path)
                for key in ("seed", "round", "arm", "role", "index", "graph", "training_feedback_eligible"):
                    if out.get(key) != row.get(key):
                        raise ValueError("Outcome/request identity differs: " + key)
                # Guard before calling native validator: it must not run OpenROAD.
                if out.get("physical_run"):
                    run = Path(out["physical_run"])
                    if read(run / "manifest.json").get("success") and not (run / "backend_macro_audit.json").is_file():
                        raise ValueError("Missing saved native geometry audit; eight-CPU worker must supply it")
                checked_existing(out, row, sample, tasks, binding, bcfg, verification)
                if out["status"] not in ADMITTED:
                    raise ValueError("Unrecognized outcome has no physical failure label")
                item.update(state=out["status"], outcome_sha256=sha(path), metrics=out["metrics"],
                    measured_utility=utility(out["metrics"], cfg["weights"], scales) if out["status"] == "success" else cfg["physical_failure_reward"],
                    native=native_fingerprints(out), backend_macro_legal=out.get("backend_macro_legal"),
                    backend_track_snapped_macro_count=out.get("backend_track_snapped_macro_count"),
                    failure_class=out.get("failure_class"), stage=out["stage"], detailed_route_DRC=None)
        except Exception as error:
            item.update(state="invalid", integrity_error=str(error), metrics=None, measured_utility=None)
        record["rows"].append(item)
    for role, count in role_counts.items():
        record["roles"][role] = summarize_rows([r for r in record["rows"] if r["role"] == role], count)
    record["status"] = ("integrity_invalid" if record["errors"] or any(r["state"] == "invalid" for r in record["rows"])
        else "complete" if all(r["fully_accounted"] for r in record["roles"].values()) else "partial")
    return record


def seed_aggregates(records, plans):
    result = []
    for exp, cfg in plans:
        selected = [r for r in records if r["experiment"] == str(exp)]
        for round_number in sorted({r["round"] for r in selected}):
            for role in ("evaluation", "collect"):
                streams = [r for r in selected if r["round"] == round_number and role in r["roles"]]
                if not streams:
                    continue
                complete = {r["seed"] for r in streams if r["status"] != "integrity_invalid" and r["roles"][role]["fully_accounted"]}
                # Evaluation uses the same predetermined noise stream for every
                # independent training seed. Collection deliberately does not.
                noise_sets = [{r["noise_sha256"] for r in stream["rows"] if r["role"] == role}
                              for stream in streams]
                matched_plan = (role != "evaluation" or
                    bool(noise_sets) and all(s == noise_sets[0] for s in noise_sets))
                eligible = complete == set(cfg["seeds"]) and matched_plan
                row = dict(experiment=str(exp), arm=cfg["arm"], round=round_number, role=role,
                    planned_seeds=cfg["seeds"], completed_seeds=sorted(complete), aggregate_eligible=eligible,
                    same_evaluation_noise_plan=matched_plan,
                    independent_seed_n=len(complete) if eligible else 0,
                    physical_failure_conditioning=any(r["roles"][role]["metrics_conditional_on_success"] for r in streams))
                row["metrics"] = {k: describe([r["roles"][role]["successful_native_metric_means"][k] for r in streams])
                                  for k in METRICS} if eligible else None
                row["utility"] = describe([r["roles"][role]["utility_mean"] for r in streams]) if eligible else None
                result.append(row)
    return result


def controlled_pair(cfg, baseline, seed, round_number):
    """Distinguish descriptive same-noise deltas from controlled Q treatments."""
    checks = {"common_checkpoint_sha256": (
        cfg.get("common_checkpoint_sha256", {}).get(str(seed)),
        baseline.get("common_checkpoint_sha256", {}).get(str(seed)))}
    defaults = dict(actor_lr=1e-5, flow_lr=1e-5, batch_size=16, flow_steps=10,
                    alpha=10., online_flow_update=False)
    # Initial actor fitting, teacher evolution, and all historical collection/
    # update budgets must match before attributing a paired delta to Q treatment.
    for key in ("offline_actor_steps", "online_updates", "online_samples", *defaults):
        checks[key] = (cfg.get(key, defaults.get(key)), baseline.get(key, defaults.get(key)))
    if round_number:
        checks["collection_noise_seed"] = (cfg.get("collection_noise_seed"), baseline.get("collection_noise_seed"))
    reasons = [dict(field=key, target=values[0], baseline=values[1],
                    reason="missing declared identity" if values[0] is None or values[1] is None else "different declarations")
               for key, values in checks.items() if values[0] is None or values[1] is None or values[0] != values[1]]
    return dict(eligible=not reasons, mismatches=reasons, checked_fields=list(checks),
        intentional_treatment_fields=["critic_variant", "weights", "q_weight", "uncertainty_beta"])


def matched_deltas(records, plans, scale_by_exp):
    baselines = [(p, cfg) for p, cfg in plans if cfg.get("q_weight", 1) == 0]
    pairs, summaries = [], []
    for target, cfg in plans:
        if cfg.get("q_weight", 1) == 0:
            continue
        for baseline, bcfg in baselines:
            if any(cfg[k] != bcfg[k] for k in ("design", "dataset_sha256", "cache_sha256", "evaluation_samples", "evaluation_noise_seed")):
                continue
            if cfg.get("evaluation_purpose", "development_validation") != bcfg.get("evaluation_purpose", "development_validation"):
                continue
            for round_number in sorted({r["round"] for r in records if r["experiment"] == str(target)}):
                seed_rows = []
                expected_seeds = sorted(set(cfg["seeds"]) | set(bcfg["seeds"]))
                controls = {seed: controlled_pair(cfg, bcfg, seed, round_number) for seed in expected_seeds}
                for seed in expected_seeds:
                    matching = [r for r in records if r["round"] == round_number and r["seed"] == seed]
                    a = next((r for r in matching if r["experiment"] == str(target)), None)
                    b = next((r for r in matching if r["experiment"] == str(baseline)), None)
                    if not a or not b or a["status"] == "integrity_invalid" or b["status"] == "integrity_invalid":
                        continue
                    aa = {r["noise_sha256"]: r for r in a["rows"] if r["role"] == "evaluation"}
                    bb = {r["noise_sha256"]: r for r in b["rows"] if r["role"] == "evaluation"}
                    matched = set(aa) & set(bb)
                    fully = (a["roles"]["evaluation"]["fully_accounted"] and b["roles"]["evaluation"]["fully_accounted"]
                        and len(aa) == len(bb) == len(matched) == cfg["evaluation_samples"])
                    local = []
                    for noise in sorted(matched):
                        ar, br = aa[noise], bb[noise]
                        if ar["state"] not in ADMITTED or br["state"] not in ADMITTED:
                            continue
                        ok = ar["state"] == br["state"] == "success"
                        au = utility(ar["metrics"], cfg["weights"], scale_by_exp[str(target)]) if ar["state"] == "success" else cfg["physical_failure_reward"]
                        bu = utility(br["metrics"], cfg["weights"], scale_by_exp[str(target)]) if br["state"] == "success" else cfg["physical_failure_reward"]
                        item = dict(target=str(target), baseline=str(baseline), seed=seed, round=round_number,
                            noise_sha256=noise, target_state=ar["state"], baseline_state=br["state"],
                            controlled_comparison_eligible=controls[seed]["eligible"],
                            control_mismatches=controls[seed]["mismatches"],
                            utility_delta_target_objective=au-bu,
                            metric_delta_target_minus_baseline={k: ar["metrics"][k]-br["metrics"][k] for k in METRICS} if ok else None,
                            target_request_sha256=ar["request_sha256"], baseline_request_sha256=br["request_sha256"])
                        pairs.append(item); local.append(item)
                    if fully:
                        seed_rows.append(dict(seed=seed, utility_delta=mean([r["utility_delta_target_objective"] for r in local]),
                            conditional_PPA=any(r["metric_delta_target_minus_baseline"] is None for r in local),
                            metric_deltas={k: mean([r["metric_delta_target_minus_baseline"][k] for r in local if r["metric_delta_target_minus_baseline"] is not None]) for k in METRICS}))
                eligible = {r["seed"] for r in seed_rows} == set(expected_seeds)
                summaries.append(dict(target=str(target), baseline=str(baseline), round=round_number,
                    planned_seeds=expected_seeds, completed_matched_seeds=[r["seed"] for r in seed_rows],
                    aggregate_eligible=eligible, conditional_PPA=any(r["conditional_PPA"] for r in seed_rows),
                    controlled_comparison_eligible=eligible and all(c["eligible"] for c in controls.values()),
                    per_seed_control_eligibility=controls,
                    utility_delta=describe([r["utility_delta"] for r in seed_rows]) if eligible else None,
                    metric_deltas={k: describe([r["metric_deltas"][k] for r in seed_rows]) for k in METRICS} if eligible else None,
                    per_seed=seed_rows))
    return pairs, summaries


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v)) if isinstance(v, (dict, list, tuple)) else clean(v) for k, v in row.items()})


def report(args):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Report scanning, array loads and correlations require a bounded Slurm CPU allocation")
    from .qstudy_backend import context
    started = time.time()
    experiments = [Path(p).resolve() for p in args.experiments]
    if len(experiments) != len(set(experiments)):
        raise ValueError("Duplicate experiment arguments")
    plans = [(p, read(p / "config.json")) for p in experiments]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot = output / (stamp + "_job" + os.environ["SLURM_JOB_ID"])
    snapshot.mkdir(exist_ok=False)
    records, bindings, scale_by_exp = [], {}, {}
    for experiment, cfg in plans:
        scales, normalization = train_scales(cfg)
        scale_by_exp[str(experiment)] = scales
        bcontext = context(experiment)
        rounds = args.rounds if args.rounds is not None else cfg.get("online_rounds", 2)
        if rounds < 0:
            raise ValueError("Round endpoint must be nonnegative")
        bindings[str(experiment)] = dict(config_sha256=sha(experiment / "config.json"), config=cfg,
            report_round_endpoint=rounds, normalization=normalization, backend_binding=bcontext[2])
        for seed in cfg["seeds"]:
            for round_number in range(rounds + 1):
                records.append(inspect_round(experiment, cfg, seed, round_number, scales, bcontext))
    paired, paired_summary = matched_deltas(records, plans, scale_by_exp)
    aggregates = seed_aggregates(records, plans)
    result = dict(version=1, stage="global_route_only", not_final_table2=True,
        created_utc=stamp, slurm_job_id=os.environ["SLURM_JOB_ID"],
        status="complete" if all(r["status"] == "complete" for r in records) else "partial_or_integrity_invalid",
        elapsed_seconds=time.time()-started, experiment_bindings=bindings,
        source_sha256={name:sha(ROOT/name) for name in ("fql/qstudy_actor_report.py", "fql/qstudy_actor.py",
            "fql/qstudy_backend.py", "fql/qstudy_controller.py", "fql/qstudy_data.py", "fql/graph_o2o_backend.py",
            "fql/graph_o2o_geometry.py", "backend/scripts/grt_metrics.py")},
        definitions=dict(metric_names=list(METRICS), utility="negative weighted TRAIN-mean-normalized nonnegative costs; confirmed failure utility=-4",
            slacks="native signed ns; cost=max(-slack,0)", missing="no outcome is pending/missing, not physical failure",
            PPA="means over successful physical runs only; explicitly conditional when any physical/geometry failure exists",
            independent_seeds="SD is sample SD of equally weighted independent seed means; not pooled layout SD",
            gradient_cosine="Recorded cosine compares gradients of +(alpha*distill) and +(q_weight*Q). Actor minimizes alpha*distill-q_weight*Q, so when both norms are nonzero the cosine of the desired objective-improving update directions is the negative recorded value: negative logged cosine means aligned updates, positive means conflicting updates. Zero Q weight gives no Q update direction",
            legalizer_RMS="all normalized macro/port coordinates; ports have zero delta; actual native track snapping is separate",
            matched_delta="Descriptive target minus BC on exactly the same noise SHA and seed, same design/dataset/cache; target utility objective on both sides. Controlled eligibility additionally requires common init, actor/flow settings and update budgets matched per seed",
            no_selection="predetermined noises/checkpoints only; no model or actor promotion; no DRT/DRC claim"),
        rounds=records, independent_seed_aggregates=aggregates, matched_summary=paired_summary,
        matched_pairs=paired)
    (snapshot / "report.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    flat = []
    for row in records:
        for role, summary in row["roles"].items():
            flat.append(dict(experiment=row["experiment"], arm=row["arm"], seed=row["seed"], round=row["round"],
                role=role, status=row["status"], **summary))
    write_csv(snapshot / "rounds.csv", flat)
    write_csv(snapshot / "seed_aggregates.csv", aggregates)
    write_csv(snapshot / "matched_pairs.csv", paired)
    write_csv(snapshot / "matched_summary.csv", paired_summary)
    text = ["# Q-study actor GRT report", "", f"Snapshot `{stamp}`, Slurm `{os.environ['SLURM_JOB_ID']}`; **{result['status']}**.", "",
        "Every table is a frozen partial/full snapshot. Missing work is not counted as a physical failure. GRT is not detailed route and no DRC result is inferred.", "",
        "| Experiment | Seed | Round | Role | Planned | Success | Geometry failure | Physical failure | Missing/unexported | Invalid |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in flat:
        text.append(f"| {Path(row['experiment']).name} | {row['seed']} | {row['round']} | {row['role']} | {row['planned']} | {row['success']} | {row['geometry_failure']} | {row['physical_failure']} | {row['missing']+row['not_exported']} | {row['invalid']} |")
    text.extend(["", "Native seven-metric means, TRAIN-only Q calibration, gradient diagnostics, request/native fingerprints and matched same-noise deltas are in `report.json` and CSV sidecars.", "",
        "Independent seed aggregation is withheld until every declared seed has fully accounted matching request plans. Conditional PPA excludes failed runs and is labeled; utility includes their declared failure penalty. No automatic actor promotion."])
    (snapshot / "report.md").write_text("\n".join(text) + "\n")
    fingerprints = {p.name:sha(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    (snapshot / "completed.json").write_text(json.dumps(dict(status="completed", report_status=result["status"],
        artifacts=fingerprints, slurm_job_id=os.environ["SLURM_JOB_ID"]), indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], snapshot=str(snapshot), rounds=len(records),
        invalid_rounds=sum(r["status"] == "integrity_invalid" for r in records),
        verified_outcomes=sum(r["state"] in ADMITTED for row in records for r in row["rows"]))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, help="Explicit report endpoint when controller stops before actor default; stored in snapshot")
    report(parser.parse_args())


if __name__ == "__main__":
    main()
