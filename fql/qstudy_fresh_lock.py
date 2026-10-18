"""Explicit, metadata-only convenience builder for a reviewed fresh selection.

``python -m fql.qstudy_fresh_lock template`` prints an UNAPPROVED request.
After scientific review, the caller fills all bindings/rules and invokes
``build --request PATH --sha256 SHA``. No candidate, round, seed, acceptance
rule or development report is selected automatically. This helper only copies
JSON configs, hashes files and runs fresh_eval.validate_lock; it never declares
an evaluation, loads tensors, creates noise, runs inference or invokes EDA.

The caller must independently establish candidate_is_legacy, comparable training
budgets/BC reuse, and completeness of the inspected development inventory.
Passing the existing software contract does not establish those scientific facts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from . import qstudy_fresh_eval as fresh
from . import qstudy_fresh_route as route

VERSION = "fresh_lock_builder_request_v1"
RULES = ("scientific_reason", "primary_endpoint", "secondary_acceptance_rules",
         "failure_accounting", "improvement_or_tradeoff_rule", "training_budget_comparison")


def template():
    file = dict(path="/ABSOLUTE/REVIEWED/FILE", sha256="REPLACE_WITH_SHA256")
    trained = dict(source_config=file.copy(), selected_round="EXPLICIT_INTEGER")
    return dict(version=VERSION, scientific_selection_approved_by_caller=False,
        output_root="/ABSOLUTE/NEW/FRESH/METADATA_ROOT", run_namespace="REPLACE",
        native_reference_config=file.copy(), gpu_name="EXACT_ALLOCATED_GPU_NAME",
        candidate_is_legacy="EXPLICIT_BOOL", candidate=trained.copy(), bc=trained.copy(),
        behavior_flow=dict(source_config=file.copy()), public_reference=dict(source_config=file.copy()),
        legacy_anchor="REQUIRED_SOURCE_CONFIG_AND_ROUND_IF_CANDIDATE_IS_NOT_LEGACY",
        common_train_utility_reference=file.copy(), development_inventory=[file.copy()],
        inspected_noise_seeds=[900000, 4300000, 4500000, 4600000, 5300000],
        **{key:"REQUIRED_PROSPECTIVE_REVIEWED_RULE" for key in RULES})


def stream_metadata(purpose, item):
    cfg_path = fresh.checked_file(item["source_config"])
    cfg = fresh.read(cfg_path)
    kind = {"behavior_flow":"behavior_flow", "public_reference":"public_official20"}.get(purpose, "one_step")
    stage = {"behavior_flow":"common", "public_reference":"public"}.get(purpose, "trained")
    seeds = [0] if stage == "public" else [0, 1, 2]
    r = item.get("selected_round", 0)
    if type(r) is not int or r < 0 or (stage != "trained" and r != 0):
        raise ValueError("An explicit nonnegative original training round is required")
    if stage == "trained" and "selected_round" not in item:
        raise ValueError("Never choose a trained round automatically")
    checkpoints, chain = {}, {}
    for seed in seeds:
        if stage == "trained":
            experiment = cfg_path.parent
            # This is the frozen read-only metadata/hash verifier, not torch.load.
            fresh.verify_actor_chain(experiment, cfg, seed, r)
            checkpoint = experiment / f"seed_{seed}/round_{r}/checkpoint.pt"
            for previous in range(r + 1):
                folder = experiment / f"seed_{seed}/round_{previous}"
                for name in ("checkpoint.pt", "binding.json", "completed.json"):
                    entry = fresh.binding(folder / name)
                    chain[entry["path"]] = entry
                for replay in fresh.read(folder / "binding.json")["replay"]:
                    entry = fresh.binding(replay["path"])
                    chain[entry["path"]] = entry
        elif stage == "common":
            checkpoint = Path(cfg["common_checkpoints"][str(seed)])
        else:
            checkpoint = fresh.CHECKPOINT
        checkpoints[str(seed)] = fresh.binding(checkpoint)
    stream = dict(id=purpose, purpose=purpose, kind=kind, checkpoint_stage=stage,
        seeds=seeds, selected_round=r, source_config=fresh.binding(cfg_path),
        checkpoints=checkpoints, training_chain=[chain[key] for key in sorted(chain)])
    fresh.validate_source(stream, cfg)
    return stream, cfg


def build(request_path, expected_sha):
    request_path = fresh.checked_file(dict(path=str(Path(request_path).resolve()), sha256=expected_sha))
    request = fresh.read(request_path)
    if request.get("version") != VERSION or request.get("scientific_selection_approved_by_caller") is not True:
        raise ValueError("Caller must explicitly approve the scientific selection before building metadata")
    if type(request.get("candidate_is_legacy")) is not bool:
        raise ValueError("Caller must independently declare whether candidate is legacy")
    if any(not request.get(key) for key in RULES) or not request.get("development_inventory"):
        raise ValueError("Explicit prospective rules and inspected development inventory are required")
    namespace = request["run_namespace"]
    if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9]{1,16}", namespace):
        raise ValueError("Use a unique alphanumeric run_namespace of at most16 characters")
    purposes = ["candidate", "bc", "behavior_flow", "public_reference"]
    if not request["candidate_is_legacy"]:
        purposes.append("legacy_anchor")
    elif request.get("legacy_anchor") is not None:
        raise ValueError("Legacy candidate must not add a duplicate legacy anchor")
    output = Path(request["output_root"]).resolve()
    if request["output_root"] != str(output):
        raise ValueError("output_root must be canonical and absolute")
    reference_path = fresh.checked_file(request["native_reference_config"])
    reference, _, physical = fresh.backend.context(reference_path.parent)
    if reference_path.name != "config.json":
        raise ValueError("native_reference_config must name the exact evaluator config.json")
    streams, configs = [], {}
    for purpose in purposes:
        stream, original = stream_metadata(purpose, request[purpose])
        if Path(stream["source_config"]["path"]).is_relative_to(output):
            raise ValueError("Never place an original source config inside fresh output_root")
        # Change evaluator data/scale only. Source label support remains pinned
        # by source_config and every original training-chain binding.
        cfg = dict(original, dataset_path=physical["dataset_path"], dataset_sha256=physical["dataset_sha256"],
            reward_scale=physical["reward_scale"], arm=purpose, run_prefix=namespace+purpose.replace("_", ""),
            seeds=stream["seeds"], evaluation_noise_seed=fresh.NOISE_SEED, evaluation_samples=fresh.COUNT,
            evaluation_purpose="fresh_confirmation", online_rounds=0, backend_threads=8,
            physical_failure_reward=-4, training_feedback_eligible=False, offline_train_eligible=False)
        if original["design"] != reference["design"] or original["cache_sha256"] != reference["cache_sha256"]:
            raise ValueError("Source and common native evaluator must share design/cache")
        configs[purpose] = cfg
        streams.append(stream)
    # Builder resumes metadata only. Refuse an existing declaration/noise/export
    # or foreign file instead of silently changing an evaluation already started.
    allowed = {"selection.json", ".selection.json.lock"}
    allowed.update(f"{purpose}/{name}" for purpose in purposes for name in ("config.json", ".config.json.lock"))
    if output.exists() and any(str(p.relative_to(output)) not in allowed for p in output.rglob("*") if p.is_file()):
        raise ValueError("Fresh output already contains non-builder artifacts; no rebuilding after declaration/export")
    for stream in streams:
        path = output / stream["id"] / "config.json"
        fresh.immutable(path, configs[stream["id"]])
        stream["evaluation_config"] = fresh.binding(path)
    inventory = [*request["development_inventory"], fresh.binding(request_path)]
    for entry in inventory:
        fresh.checked_file(entry)
    fresh.checked_file(request["common_train_utility_reference"])
    sources = fresh.required_sources()
    sources.update({name:fresh.ROOT/name for name in route.SOURCE_NAMES})
    sources[str(Path(__file__).resolve().relative_to(fresh.ROOT))] = Path(__file__).resolve()
    geometry = [fresh.ROOT / f"backend/releases/flowplace_DEFS/openroad/{reference['design']}.def",
                *(fresh.ROOT / "backend/thirdparty/OpenROAD-flow-scripts/platforms/nangate45/lef").glob("*.lef")]
    lock = dict(version=fresh.VERSION, status="selection_locked", output_root=str(output),
        candidate_is_legacy=request["candidate_is_legacy"], design=reference["design"],
        dataset_sha256=physical["dataset_sha256"], cache_sha256=reference["cache_sha256"],
        physical_flow_revision=physical["physical_flow_revision"], streams=streams,
        hardware_software=fresh.environment_spec(request["gpu_name"]),
        common_train_utility_reference=request["common_train_utility_reference"], development_inventory=inventory,
        inspected_noise_seeds=request["inspected_noise_seeds"],
        source_sha256={name:fresh.sha(path) for name,path in sorted(sources.items())},
        geometry_inputs=[fresh.binding(path) for path in sorted(geometry)],
        evaluation=dict(noise_seed=fresh.NOISE_SEED, samples=fresh.COUNT, generator_device="cuda", dtype="float32",
            prior="uniform[-1,1]", shared_across_all_streams=True, drt_indices=fresh.DRT_INDICES,
            drt_repair_cap=fresh.REPAIR_CAP), **{key:request[key] for key in RULES})
    path = output / "selection.json"
    fresh.immutable(path, lock)
    digest = fresh.sha(path)
    fresh.validate_lock(path, digest)
    return dict(selection=fresh.binding(path), status="metadata_validated_not_declared",
                streams=len(streams), noise_created=False, tensors_loaded=False, native_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template")
    create = sub.add_parser("build")
    create.add_argument("--request", required=True)
    create.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(template() if args.command == "template" else build(args.request, args.sha256), indent=2))


if __name__ == "__main__":
    main()
