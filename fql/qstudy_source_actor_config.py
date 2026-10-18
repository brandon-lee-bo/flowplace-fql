"""Prepare explicitly reviewed source-critic actor configs; never launch jobs.

Full-TRAIN refits and TRAIN-only utility scales are verified under Slurm. The
original datasets, BC targets, checkpoints and native reward semantics remain
unchanged. Legacy configs are offline-only: the frozen online actor cannot
rescale its failure target along with the source-dependent legacy Q weight.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

# Resolve the local analysis namespace before the upstream FlowPlace adapter
# adds its own analysis.py directory to sys.path.
from .qstudy_backend import PHYSICAL_REVISIONS
from .qstudy_branch import SAFE_NAME, atomic_json, recover_journal_temporaries
from . import qstudy_source as source
from . import qstudy_screen as screen
from .qstudy_data import METRIC_NAMES, bind, load_dataset, sha
from .qstudy_models import _masked_stats, measured_costs

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("fql/qstudy_source_actor_config.py", "fql/qstudy_source.py", "fql/qstudy_screen.py",
    "fql/qstudy_models.py", "fql/graph_o2o_models.py", "fql/qstudy_data.py", "fql/qstudy_actor.py",
    "fql/qstudy_backend.py", "fql/qstudy_branch.py", "baseline/api.py", "docs/qstudy_source_actor_protocol.md")


def review_note(path, report_sha, matrix):
    note = source.read_json(path)
    date.fromisoformat(note["date"])
    if not isinstance(note.get("reason"), str) or not note["reason"].strip():
        raise ValueError("A dated scientific selection reason is required")
    if note.get("source_report_sha256") != report_sha:
        raise ValueError("Review note is bound to a different source report")
    weights = np.asarray(note["reference_weights"], dtype=float)
    if weights.shape != (5,) or not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Reference preferences must be five finite nonnegative weights")
    selections = note.get("selections")
    if not isinstance(selections, list) or not selections:
        raise ValueError("Explicit reviewed source/critic selections are required")
    available = {(t["arm"], t["variant"]) for t in matrix["tasks"]}
    for row in selections:
        if set(row) != {"source_arm", "variant", "actor_arm", "run_prefix"}:
            raise ValueError("Each selection explicitly names source_arm/variant/actor_arm/run_prefix")
        if (row["source_arm"], row["variant"]) not in available:
            raise ValueError("Selected source/critic was not in the completed study")
        if not all(isinstance(row[k], str) and SAFE_NAME.fullmatch(row[k]) for k in ("actor_arm", "run_prefix")):
            raise ValueError("Unsafe actor arm/run prefix")
        if row["variant"] == "legacy" and weights.tolist() != [1., 0., 0., 0., 0.]:
            raise ValueError("The frozen legacy critic supports WL-only reference utility")
    for key in ("actor_arm", "run_prefix"):
        if len({row[key] for row in selections}) != len(selections):
            raise ValueError("Selected actor arms and run prefixes must be unique")
    if len({(r["source_arm"], r["variant"]) for r in selections}) != len(selections):
        raise ValueError("Source/critic selection duplicated")
    diagnostics = note.get("diagnostic_review_provenance", [])
    if not diagnostics or not any(p.get("role") == "direction_probe" for p in diagnostics):
        raise ValueError("Bind reviewed direction-probe evidence before actor preparation")
    for item in diagnostics:
        if not isinstance(item.get("role"), str) or not item["role"]:
            raise ValueError("Diagnostic evidence needs an explicit role")
        bind(item["path"], item["sha256"])
    return note, weights


def completed_study(study, report_path):
    matrix = source.load_matrix(study)
    declaration = source.load_plan(study)["declaration"]
    report = source.read_json(report_path)
    if (len(matrix["tasks"]) != 36 or report.get("status") != "complete" or
            report.get("planned_tasks") != 36 or report.get("completed_tasks") != 36 or
            report.get("integrity_errors") != [] or report.get("actor_training") is not False or
            report.get("automatic_promotion") is not False or report.get("backend_calls") != 0):
        raise ValueError("Source report must contain all36 completed tasks and zero integrity errors")
    if report["matrix"] != bind(study/"matrix.json") or report["plan"] != matrix["plan"]:
        raise ValueError("Source report plan/matrix binding differs")
    if report.get("report_source_sha256") != sha(ROOT/"fql/qstudy_source.py"):
        raise ValueError("Source report implementation differs")
    tasks = source.read_json(report_path.with_name("tasks.json"))
    if len(tasks) != 36 or len({r["index"] for r in tasks}) != 36:
        raise ValueError("Source report has duplicate/missing task identities")
    indexed = {r["index"]:r for r in tasks}
    expected = {(arm, variant, seed) for arm in source.ARMS for variant in source.VARIANTS for seed in (0, 1, 2)}
    if {(t["arm"], t["variant"], t["seed"]) for t in matrix["tasks"]} != expected:
        raise ValueError("Source study arm/variant/seed matrix differs")
    bindings = {str(report_path):sha(report_path), str(report_path.with_name("tasks.json")):sha(report_path.with_name("tasks.json"))}
    for task in matrix["tasks"]:
        row = indexed[task["index"]]
        if any(row.get(k) != v for k, v in task.items()) or row.get("status") != "completed" or row.get("completed_fits") != 4:
            raise ValueError("Source report task identity/status differs")
        _, state = source.checked_predictions(task, matrix, declaration)
        if state != {k:row[k] for k in ("status", "completed_fits", "root_sha256")}:
            raise ValueError("Source report became stale or an underlying fit changed")
        folder = Path(task["output"])
        paths = [folder/"completed.json", folder/"binding.json"]
        for fit in ("cv_0", "cv_1", "cv_2", "refit"):
            paths.extend(folder/fit/name for name in ("completed.json", "resume.pt", "predictions.npz"))
        for path in paths:
            bindings[str(path.resolve())] = sha(path)
    for group in matrix["datasets"].values():
        for binding in group.values():
            bindings[binding["path"]] = binding["sha256"]
    return matrix, declaration, bindings


def finite_train(path, expected_behavior=None):
    data, manifest = load_dataset(path)
    train = data["train"]
    if (not np.isfinite(train["metrics"]).all() or not train["metric_mask"].all() or
            not np.all(train["status"] == "success")):
        raise ValueError("Frozen actor requires fully finite successful initial TRAIN labels")
    if (len(data["behavior_action"]) != 256 or not np.isfinite(data["behavior_action"]).all() or
            data["behavior_action"].shape[1] != train["raw_action"].shape[1]):
        raise ValueError("Expected exactly256 finite BC targets in identical coordinate order")
    if expected_behavior is not None and any(not np.array_equal(data[k], expected_behavior[k])
                                             for k in ("behavior_action", "behavior_family")):
        raise ValueError("Original256 BC targets/families changed")
    if (manifest.get("stage") != "global_route_only" or manifest.get("not_final_table2") is not True or
            set(manifest["physical_flow_revisions"]) != {PHYSICAL_REVISIONS[manifest["design"]]}):
        raise ValueError("Source dataset physical stage/revision differs")
    costs, mask = measured_costs(torch.tensor(train["metrics"][:, :5], dtype=torch.float32),
        METRIC_NAMES[:5], torch.tensor(train["metric_mask"][:, :5]))
    if not bool(mask.all()) or not bool((costs[:, 0] > 0).all()):
        raise ValueError("Every initial TRAIN row needs positive WL and all five costs")
    return data, manifest, costs


def checked_refit(task, matrix, declaration, cache, movable, data, costs):
    path = Path(task["output"])/"refit/resume.pt"
    done = source.read_json(path.with_name("completed.json"))
    saved = torch.load(path, map_location="cpu", weights_only=False)
    train_indices = list(range(len(data["train"]["raw_action"])))
    if (sha(path) != done["checkpoint_sha256"] or saved["binding"] != done["binding"] or
            saved["binding"]["task"] != task or saved["binding"]["source_indices"]["train"] != train_indices or
            saved["train_indices"] != train_indices or saved["step"] != declaration["steps"] or
            saved["model_config"] != screen.VARIANTS[task["variant"]]):
        raise ValueError("Selected checkpoint is not the completed all-TRAIN refit")
    actions = torch.tensor(data["train"]["raw_action"], dtype=torch.float32).reshape(len(costs), -1, 2)
    families = torch.tensor(data["train"]["family"], dtype=torch.long)
    with torch.random.fork_rng():
        torch.manual_seed(task["seed"])
        model, normalization = screen.make_model(cache, actions, costs, torch.ones_like(costs, dtype=torch.bool),
            movable, task["variant"], families, task["seed"])
    if saved["normalization"] != normalization or done["normalization"] != normalization:
        raise ValueError("Selected critic normalization metadata differs from full TRAIN")
    # Every buffer is either TRAIN-fitted normalization or immutable cache geometry.
    # Compare before loading learned parameters so heldout-fitted scales cannot hide.
    for key, value in model.named_buffers():
        actual = saved["model"].get(key)
        if not isinstance(actual, torch.Tensor) or actual.dtype != value.dtype or not torch.equal(actual, value):
            raise ValueError("Checkpoint TRAIN/cache normalization buffer differs: "+key)
    model.load_state_dict(saved["model"], strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()
           if isinstance(v, torch.Tensor) and v.is_floating_point()):
        raise ValueError("Nonfinite selected critic parameter/buffer")
    scale = ([float(costs[:, 0].mean())] if task["variant"] == "legacy" else
             model.objective_scale.detach().cpu().tolist())
    return scale, dict(checkpoint=bind(path), completion=bind(path.with_name("completed.json")),
        binding=bind(path.parent.parent/"binding.json"), train_rows=len(costs), train_indices=train_indices,
        source_scale=scale, normalization=normalization)


def prepare(args):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Source/checkpoint verification requires a Slurm CPU allocation")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    study, report_path, base_path, note_path, output = (Path(p).resolve() for p in
        (args.study, args.source_report, args.base_config, args.review_note, args.output))
    base = source.read_json(base_path)
    if (base["seeds"] != [0, 1, 2] or base["alpha"] != 10 or base["flow_steps"] != 10 or
            base["online_flow_update"] is not False or base["offline_critic_update"] is not False or
            base["terminal_mask"] != 0 or base["backend_threads"] != 8 or
            base["physical_failure_reward"] != -4 or base.get("uncertainty_beta", 0) != 0):
        raise ValueError("Base config must preserve audited three-seed frozen-flow actor settings")
    matrix, declaration, evidence = completed_study(study, report_path)
    note, reference_weights = review_note(note_path, sha(report_path), matrix)
    if declaration["design"] != base["design"] or declaration["cache"]["sha256"] != base["cache_sha256"]:
        raise ValueError("Source design/cache differs from the matched base actor")
    original, _, _ = finite_train(declaration["old_source"]["dataset"]["path"])
    finite_train(base["dataset_path"], original)
    bind(base["dataset_path"], base["dataset_sha256"])
    reference_path = Path(matrix["datasets"]["O64"]["dataset"]["path"])
    reference, _, reference_costs = finite_train(reference_path, original)
    reference_scale = _masked_stats(reference_costs, torch.ones_like(reference_costs, dtype=torch.bool))[2].numpy()
    cache_path = Path(declaration["cache"]["path"])
    bind(cache_path, base["cache_sha256"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    from baseline.api import load_problem
    movable = ~load_problem(base["design"], "cpu").cond.is_ports.bool()
    common = {}
    for seed in base["seeds"]:
        key = str(seed)
        path = Path(base["common_checkpoints"][key])
        common[key] = dict(checkpoint=bind(path, base["common_checkpoint_sha256"][key]))
        status_path = path.with_name("training_status.json")
        status = source.read_json(status_path)
        if (status["status"] != "completed" or status["seed"] != seed or status["round"] != 0 or
                status["graph_cache_sha256"] != base["cache_sha256"] or status["no_heldout_labels_used"] is not True):
            raise ValueError("Common initialization metadata differs from audited independent seeds")
        init_config = path.parents[2]/"config.json"
        common[key].update(metadata=bind(status_path), original_config=bind(init_config, status["config_sha256"]))
    if len({c["checkpoint"]["sha256"] for c in common.values()}) != 3:
        raise ValueError("Three common initializations must be distinct independent-seed artifacts")
    if declaration["teacher_common"] != common["0"]["checkpoint"]:
        raise ValueError("Current teacher source must retain the declared common seed0 checkpoint")
    configs, selected_evidence = {}, []
    for selection in note["selections"]:
        arm, variant = selection["source_arm"], selection["variant"]
        dataset_path = Path(matrix["datasets"][arm]["dataset"]["path"])
        data, manifest, costs = finite_train(dataset_path, original)
        if manifest.get("source_arm") != arm:
            raise ValueError("Selected source dataset arm identity differs")
        cfg = copy.deepcopy(base)
        cfg.update(arm=selection["actor_arm"], run_prefix=selection["run_prefix"], critic_variant=variant,
            dataset_path=str(dataset_path), dataset_sha256=sha(dataset_path), online_flow_update=False,
            alpha=10., critic_checkpoints={}, critic_checkpoint_sha256={})
        scales, seed_evidence = [], {}
        for seed in base["seeds"]:
            candidates = [t for t in matrix["tasks"] if (t["arm"], t["variant"], t["seed"]) == (arm, variant, seed)]
            if len(candidates) != 1:
                raise ValueError("Selected source critic seed missing or duplicated")
            scale, info = checked_refit(candidates[0], matrix, declaration, cache, movable, data, costs)
            scales.append(scale); seed_evidence[str(seed)] = info
            cfg["critic_checkpoints"][str(seed)] = info["checkpoint"]["path"]
            cfg["critic_checkpoint_sha256"][str(seed)] = info["checkpoint"]["sha256"]
        if not all(scale == scales[0] for scale in scales):
            raise ValueError("Source normalization/objective scales differ across matched seeds")
        if variant == "legacy":
            ratio = scales[0][0]/float(reference_scale[0])
            cfg.update(q_weight=ratio, weights=[1, 0, 0, 0, 0], online_rounds=0)
        else:
            ratio = 1.
            cfg.update(q_weight=1., weights=(reference_weights*np.asarray(scales[0])/reference_scale).tolist(),
                online_rounds=base.get("online_rounds", 2))
        configs[cfg["arm"]] = cfg
        selected_evidence.append(dict(selection=selection, refits=seed_evidence, source_scale=scales[0],
            physical_q_multiplier=ratio, legacy_online_ready=variant != "legacy" or ratio == 1.,
            prepared_online_rounds=cfg["online_rounds"], successful_utility="-sum(w_ref,k * measured_cost_k / O64_TRAIN_scale_k)",
            failure_utility=(-4*ratio if variant == "legacy" else -4),
            limitation=("All legacy configs are offline-only. The frozen online legacy actor targets -4 in source Q units; after q_weight correction this becomes -4*S_source/S_ref, not common -4. A tested new actor must target -4/q_weight before source-matched online replay." if variant == "legacy" else
                        "Corrected native-unit metric weights and q_weight=1 preserve the frozen -4 utility failure target.")))
    protected = [study, base_path.parent, Path(base["dataset_path"]).parent, cache_path.parent,
        *[Path(c["checkpoint"]["path"]).parent for c in common.values()]]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("New config output must be disjoint from all preserved source artifacts")
    manifest = dict(schema="qstudy_reviewed_source_actor_configs_v1", status="prepared_configs_only",
        review_note=bind(note_path), source_report=bind(report_path), matrix=bind(study/"matrix.json"),
        plan=matrix["plan"], base_config=bind(base_path), common_initializations=common,
        reference=dict(source="O64 initial TRAIN only", dataset=bind(reference_path),
            scales=reference_scale.tolist(), weights=reference_weights.tolist(),
            note="Objective magnitude scales, never heldout scales or target regression z-scores"),
        behavior=dict(targets=256, exact_original_targets_and_families=True), selected=selected_evidence,
        evidence=evidence, source_sha256={name:sha(ROOT/name) for name in SOURCES},
        teacher_source_limitation="T64/M64 teacher layouts come from common seed0 only; three critic/actor seeds are not three independent teacher-source distributions.",
        original_artifacts_unchanged=True, automatic_selection=False, actor_jobs_submitted=0, backend_calls=0,
        heldout_used=False, validation_or_probes_in_replay=False,
        native_reward_semantics="Base reward_scale declaration/default and physical_failure_reward=-4 unchanged; physical Q correction is recorded separately.")
    output.mkdir(parents=True, exist_ok=True)
    if (output/".config_prepare.lock").is_symlink():
        raise ValueError("Config preparation lock cannot be a symlink")
    with (output/".config_prepare.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        recover_journal_temporaries(output, ("preparation_plan.json", "preparation.json"))
        if {p.name for p in output.iterdir()}-set(configs)-{".config_prepare.lock", "preparation_plan.json", "preparation.json"}:
            raise ValueError("Foreign artifacts in configuration output")
        plan_path = output/"preparation_plan.json"
        proposal = dict(manifest=manifest, configs=configs)
        if plan_path.is_symlink() or (plan_path.exists() and source.read_json(plan_path) != proposal):
            raise ValueError("Existing configuration preparation plan is immutable")
        if not plan_path.exists():
            atomic_json(plan_path, proposal)
        # Completed studies are immutable; refuse a concurrent change before
        # publishing any proposed configs rather than mixing two evidence states.
        for path, digest in evidence.items():
            bind(path, digest)
        for arm, cfg in configs.items():
            folder = output/arm
            if folder.is_symlink():
                raise ValueError("Actor config destination cannot be a symlink")
            folder.mkdir(exist_ok=True)
            recover_journal_temporaries(folder, ("config.json",))
            path = folder/"config.json"
            if path.is_symlink() or (path.exists() and source.read_json(path) != cfg):
                raise ValueError("Existing actor configuration is immutable")
            if not path.exists():
                atomic_json(path, cfg)
        manifest["configs"] = {arm:bind(output/arm/"config.json") for arm in configs}
        final = output/"preparation.json"
        if final.is_symlink() or (final.exists() and source.read_json(final) != manifest):
            raise ValueError("Existing preparation manifest is immutable")
        if not final.exists():
            atomic_json(final, manifest)
    print(json.dumps(dict(status="prepared_configs_only", manifest=str(final), arms=list(configs),
        actor_jobs_submitted=0, backend_calls=0)), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source-report", "base-config", "review-note", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
