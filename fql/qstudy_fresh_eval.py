"""Locked, evaluation-only fresh exports; never train, score, select or run EDA.

The caller supplies an immutable selection JSON and its SHA, prepared only after
development review. ``validate``/``declare`` do not generate noise or load models.
Only ``export`` creates the shared CUDA noise once and evaluates a locked policy.
Native request format and the existing eight-thread backend remain unchanged.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

# Import the package-aware native verifier before upstream's analysis.py enters
# sys.path. These functions validate files; neither context nor checked_request
# executes a backend tool.
from . import qstudy_backend as backend
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .qstudy_actor_report import check_training
from .qstudy_final_route import verify_actor_chain
from .qstudy_data import action_sha
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import (CHECKPOINT, RUNTIME, generate, legalize, load_model,
                          load_problem, save_sample)
from backend.scripts.export_macro_def import convert
from .graph_o2o_models import GraphPolicy

ROOT = Path(__file__).resolve().parents[1]
VERSION = "fresh_eval_v1"
NOISE_SEED, COUNT, DRT_INDICES, REPAIR_CAP = 85000000, 8, [0, 1, 2, 3], 20
PUBLIC_SHA = "0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359"
SAMPLERS = {"one_step": "frozen_graph_actor_one_step_final_legalize",
            "behavior_flow": "frozen_flow_euler10_final_legalize",
            "public_official20": "public_reverse_samples_20hard_uniform_wreg0.1_wdist1"}
SOURCE_NAMES = ("fql/qstudy_fresh_eval.py", "fql/qstudy_backend.py",
    "fql/graph_o2o_backend.py", "fql/graph_o2o_archive.py", "fql/graph_o2o_models.py",
    "fql/physical_ridge.py", "fql/qstudy_data.py", "fql/qstudy_actor_report.py",
    "fql/graph_o2o_geometry.py", "analysis/placement_metrics.py",
    "analysis/audit_table2_macro_preservation.py", "baseline/api.py",
    "backend/scripts/export_macro_def.py", "backend/scripts/grt_metrics.py",
    "backend/scripts/grt_provenance.py", "fql/analytic.py", "fql/qstudy_final_route.py",
    "fql/graph_o2o_final_evidence.py", "upstream/flowplace/benchmark_utils/openroad_utils.py",
    "docs/qstudy_fresh_confirmation_protocol.md")


def read(path):
    return json.loads(Path(path).read_text())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked_file(item):
    if (set(item) != {"path", "sha256"} or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or binding(item["path"]) != item):
        raise ValueError("Changed or noncanonical locked file binding: " + str(item))
    return Path(item["path"])


def required_sources():
    """Explicit local dependencies plus the complete small upstream code/config tree.

    No environment-dependent sys.modules inventory or post-export code discovery.
    The caller can include additional reviewed files, but cannot omit these.
    """
    paths = {name: ROOT / name for name in SOURCE_NAMES}
    upstream = ROOT / "upstream/flowplace/flow_matching"
    for path in upstream.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts:
            paths[str(path.relative_to(ROOT))] = path
    return paths


def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name("." + path.name + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if path.exists():
            if read(path) != value:
                raise ValueError("Changed immutable fresh evaluation artifact: " + str(path))
        else:
            atomic_json(path, value)


def environment_spec(gpu_name):
    return dict(torch=str(torch.__version__), numpy=str(np.__version__), cuda=torch.version.cuda,
        gpu_name=gpu_name, deterministic_algorithms=True, warn_only=False,
        cudnn_benchmark=False, cudnn_deterministic=True, allow_tf32=False,
        float32_matmul_precision="highest", cublas_workspace_config=":4096:8")


def configure_export_environment(locked):
    """Apply reviewed precision settings, then verify actual allocated hardware."""
    if locked != environment_spec(torch.cuda.get_device_name(torch.cuda.current_device())):
        raise ValueError("Actual CUDA GPU/packages differ from locked hardware/software")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("Conflicting CUDA deterministic workspace configuration")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def source_checkpoint(stream, seed):
    return stream["checkpoints"][str(seed)]


def validate_source(stream, source_cfg):
    kind = stream["kind"]
    keys = {str(s) for s in stream["seeds"]}
    if set(stream["checkpoints"]) != keys:
        raise ValueError("Checkpoint inventory differs from every declared training seed")
    stage = stream["checkpoint_stage"]
    if stage not in ("trained", "common", "public") or not isinstance(stream["selected_round"], int) or stream["selected_round"] < 0:
        raise ValueError("Explicit original checkpoint stage/round is required")
    chain_files = {}
    for seed in stream["seeds"]:
        item = source_checkpoint(stream, seed)
        path = checked_file(item)
        if kind == "public_official20":
            if stage != "public" or item != binding(CHECKPOINT) or item["sha256"] != PUBLIC_SHA:
                raise ValueError("Public reference must use the released official checkpoint")
        elif stage == "trained":
            folder = path.parent
            if path.name != "checkpoint.pt" or folder.name != f"round_{stream['selected_round']}" or folder.parent.name != f"seed_{seed}":
                raise ValueError("Original selected training round/seed differs from checkpoint path")
            experiment = folder.parent.parent
            if experiment / "config.json" != Path(stream["source_config"]["path"]):
                raise ValueError("Selected checkpoint is not inside its original experiment")
            result = verify_actor_chain(experiment, source_cfg, seed, stream["selected_round"])
            if not result["completed"] or result["checkpoint_sha256"] != item["sha256"]:
                raise ValueError("Selected actor/flow is not a completed frozen checkpoint")
            for field in ("common", "critic"):
                checked_file(dict(path=str(Path(source_cfg[field+"_checkpoints"][str(seed)]).resolve()),
                    sha256=source_cfg[field+"_checkpoint_sha256"][str(seed)]))
            for r in range(stream["selected_round"]+1):
                previous = experiment / f"seed_{seed}/round_{r}"
                for name in ("checkpoint.pt", "completed.json", "binding.json"):
                    value = binding(previous / name)
                    chain_files[value["path"]] = value
                for entry in read(previous / "binding.json")["replay"]:
                    value = binding(entry["path"])
                    chain_files[value["path"]] = value
        elif kind != "behavior_flow" or stage != "common" or (
                item["sha256"] != source_cfg["common_checkpoint_sha256"][str(seed)] or
                path != Path(source_cfg["common_checkpoints"][str(seed)]).resolve()):
            raise ValueError("Only the bound common flow may use an untrained/common source")
    if stream.get("training_chain", []) != [chain_files[key] for key in sorted(chain_files)]:
        raise ValueError("Selected original training/replay chain is not exactly locked")


def validate_lock(path, expected_sha):
    """Read-only source/config/native-context gates; no tensors, RNG or outcomes."""
    path = checked_file(dict(path=str(Path(path).resolve()), sha256=expected_sha))
    lock = read(path)
    if lock.get("version") != VERSION or lock.get("status") != "selection_locked":
        raise ValueError("Fresh candidate selection is not explicitly locked")
    for key in ("scientific_reason", "primary_endpoint", "secondary_acceptance_rules",
                "failure_accounting", "improvement_or_tradeoff_rule", "hardware_software",
                "training_budget_comparison", "common_train_utility_reference", "development_inventory"):
        if not lock.get(key):
            raise ValueError("Missing prospective selection field: " + key)
    for item in lock["development_inventory"]:
        checked_file(item)
    checked_file(lock["common_train_utility_reference"])
    # Pin the inspected streams explicitly, without imposing a report file schema.
    if not {900000, 4300000, 4500000, 4600000, 5300000} <= set(lock.get("inspected_noise_seeds", [])):
        raise ValueError("Development inventory must disclose all inspected old noise streams")
    hardware = lock["hardware_software"]
    if not isinstance(hardware, dict) or not isinstance(hardware.get("gpu_name"), str) or not hardware["gpu_name"] or hardware != environment_spec(hardware["gpu_name"]):
        raise ValueError("Locked software versions/deterministic precision settings differ")
    evaluation = dict(noise_seed=NOISE_SEED, samples=COUNT, generator_device="cuda",
                      dtype="float32", prior="uniform[-1,1]", shared_across_all_streams=True,
                      drt_indices=DRT_INDICES, drt_repair_cap=REPAIR_CAP)
    if lock.get("evaluation") != evaluation:
        raise ValueError("Changed fresh noise/count/preselected DRT policy")
    sources = lock.get("source_sha256", {})
    if not set(required_sources()) <= set(sources):
        raise ValueError("Incomplete exporter/backend/upstream source closure")
    for name, digest in sources.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or sha(ROOT / rel) != digest:
            raise ValueError("Changed locked source code/config: " + name)
    export_inputs = {str(ROOT / "backend/releases/flowplace_DEFS/openroad" / (lock["design"] + ".def"))}
    export_inputs.update(str(p.resolve()) for p in (ROOT / "backend/thirdparty/OpenROAD-flow-scripts/platforms/nangate45/lef").glob("*.lef"))
    bound_inputs = lock["geometry_inputs"]
    if {x["path"] for x in bound_inputs} != export_inputs or len(bound_inputs) != len(export_inputs):
        raise ValueError("Incomplete locked DEF template/macro LEF geometry inputs")
    for item in bound_inputs:
        checked_file(item)
    output = Path(lock["output_root"]).resolve()
    if lock["output_root"] != str(output):
        raise ValueError("Fresh output_root must be canonical and absolute")
    streams = lock["streams"]
    if not streams or len({s["id"] for s in streams}) != len(streams):
        raise ValueError("Missing or duplicate stream identities")
    configs, physicals, experiments = {}, {}, set()
    purposes = {s["purpose"] for s in streams}
    if not {"candidate", "bc", "behavior_flow", "public_reference"} <= purposes or sum(s["purpose"] == "candidate" for s in streams) != 1:
        raise ValueError("One candidate plus BC/current flow/public reference must be locked")
    candidate = next(s for s in streams if s["purpose"] == "candidate")
    if not isinstance(lock.get("candidate_is_legacy"), bool):
        raise ValueError("Declare whether the candidate is the existing legacy anchor")
    expected_purposes = {"candidate", "bc", "behavior_flow", "public_reference"}
    if not lock["candidate_is_legacy"]:
        expected_purposes.add("legacy_anchor")
    if purposes != expected_purposes or len(streams) != len(expected_purposes):
        raise ValueError("A new method needs the locked legacy FQL anchor")
    for stream in streams:
        if (stream["kind"] not in SAMPLERS or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", stream["id"])
                or stream["seeds"] != ([0] if stream["kind"] == "public_official20" else [0, 1, 2])):
            raise ValueError("Invalid sampler or independent-training-seed inventory")
        if (stream["purpose"] == "public_reference") != (stream["kind"] == "public_official20"):
            raise ValueError("Public single-checkpoint reference cannot masquerade as three policies")
        if stream["purpose"] == "behavior_flow" and stream["kind"] != "behavior_flow":
            raise ValueError("Current behavior-flow control requires its Euler10 sampler")
        cfg_path = checked_file(stream["evaluation_config"])
        experiment = cfg_path.parent
        if (cfg_path.name != "config.json" or not experiment.is_relative_to(output) or experiment in experiments):
            raise ValueError("Fresh output paths must be distinct descendants of output_root")
        experiments.add(experiment)
        cfg, _, physical = backend.context(experiment)
        original = read(checked_file(stream["source_config"]))
        if stream["source_config"]["path"] == str(cfg_path):
            raise ValueError("Fresh config must not overwrite its original training config")
        for key in ("design", "cache_sha256"):
            if cfg[key] != original[key] or cfg[key] != lock[key]:
                raise ValueError("Changed original/locked design or cache: " + key)
        if cfg["dataset_sha256"] != lock["dataset_sha256"]:
            raise ValueError("Changed common native evaluation dataset")
        # Source label support may legitimately differ (O64/T64/384). Preserve
        # that identity instead of falsely rebinding it to the common evaluator.
        checked_file(dict(path=str(Path(original["dataset_path"]).resolve()), sha256=original["dataset_sha256"]))
        if stream["purpose"] == "bc" and original.get("q_weight") != 0:
            raise ValueError("BC reference must originate from the zero-Q BC training arm")
        if (cfg["arm"] != stream["id"] or cfg["seeds"] != stream["seeds"] or
                cfg["evaluation_noise_seed"] != NOISE_SEED or cfg["evaluation_samples"] != COUNT or
                cfg["evaluation_purpose"] != "fresh_confirmation" or cfg["online_rounds"] != 0 or
                cfg["backend_threads"] != 8 or cfg["physical_failure_reward"] != -4 or
                not re.fullmatch(r"[A-Za-z0-9]{1,32}", cfg["run_prefix"])):
            raise ValueError("Fresh evaluation config changed native/budget/role settings")
        if cfg.get("training_feedback_eligible", False) or cfg.get("offline_train_eligible", False):
            raise ValueError("Fresh layouts must never enter replay/offline TRAIN")
        checked_file(dict(path=str((RUNTIME / "benchmark_cache" / (cfg["design"] + ".pt")).resolve()), sha256=cfg["cache_sha256"]))
        if physical["physical_flow_revision"] != lock["physical_flow_revision"]:
            raise ValueError("Locked native physical revision differs")
        validate_source(stream, original)
        configs[stream["id"]], physicals[stream["id"]] = cfg, physical
    native_contexts = [{k: v for k, v in p.items() if k != "config_sha256"} for p in physicals.values()]
    if any(p != native_contexts[0] for p in native_contexts):
        raise ValueError("All fresh streams must share the exact native evaluator/TRAIN context")
    if candidate["kind"] != "one_step" or next(s for s in streams if s["purpose"] == "bc")["kind"] != "one_step":
        raise ValueError("Candidate and BC must be frozen one-step policies")
    if any(s["purpose"] == "legacy_anchor" and s["kind"] != "one_step" for s in streams):
        raise ValueError("Legacy anchor must be its frozen one-step actor")
    return lock, configs, physicals


def declare(path, expected_sha):
    lock, configs, physicals = validate_lock(path, expected_sha)
    root = Path(lock["output_root"])
    value = dict(version=VERSION, selection_lock=binding(path), expected_noise_arrays=COUNT,
        role="evaluation", terminal_mask=0, training_feedback_eligible=False,
        offline_train_eligible=False, evaluation_purpose="fresh_confirmation",
        evaluation=lock["evaluation"], streams=[dict(id=s["id"], kind=s["kind"],
            seeds=s["seeds"], expected_requests=COUNT*len(s["seeds"]),
            selected_original_round=s["selected_round"], source_config=s["source_config"],
            checkpoints=s["checkpoints"], source_sampler=SAMPLERS[s["kind"]],
            evaluation_config=s["evaluation_config"], native_context=physicals[s["id"]]) for s in lock["streams"]],
        backend_calls=0, optimizer_updates=0, selection="all locked seeds and indices0..7; no Q/metric selection",
        independent_public_checkpoints=1, drt_selection_independent_of_GRT=True)
    immutable(root / "declaration.json", value)
    return lock, configs, physicals, value


def noise_manifest(arrays, lock_sha, n):
    if (arrays.dtype != np.float32 or arrays.shape != (COUNT, n, 2) or
            not np.isfinite(arrays).all() or np.any(np.abs(arrays) > 1)):
        raise ValueError("Invalid actual shared fresh noise array")
    digests = [action_sha(x) for x in arrays]
    if len(set(digests)) != COUNT:
        raise ValueError("Eight distinct actual noise arrays are required")
    return dict(version=VERSION, selection_lock_sha256=lock_sha, noise_seed=NOISE_SEED,
        generator_device="cuda", dtype="float32", shape=list(arrays.shape),
        noise_sha256=digests, no_training_or_replay=True)


def shared_noise(root, lock_sha, n, *, create=False):
    """One CUDA RNG stream, shared actual bytes; safe recovery after NPZ publication."""
    root = Path(root)
    folder = root / "shared_noise"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = folder / "noises.npz"
        if not path.exists():
            if not create:
                raise FileNotFoundError("Fresh actual noise is not yet exported")
            if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
                raise RuntimeError("Fresh noise creation requires an allocated Slurm CUDA GPU")
            generator = torch.Generator(device="cuda").manual_seed(NOISE_SEED)
            arrays = (2*torch.rand(COUNT, n, 2, dtype=torch.float32, device="cuda", generator=generator)-1).cpu().numpy()
            tmp = path.with_suffix(".npz.tmp")
            with tmp.open("wb") as out:
                np.savez_compressed(out, noise=arrays, selection_lock_sha256=np.asarray(lock_sha))
            tmp.replace(path)
        with np.load(path, allow_pickle=False) as saved:
            arrays = saved["noise"].copy()
            if str(saved["selection_lock_sha256"].item()) != lock_sha:
                raise ValueError("Shared noise belongs to a different selection lock")
        value = dict(noise_manifest(arrays, lock_sha, n), artifact=binding(path))
        immutable(folder / "manifest.json", value)
        return arrays, value


def row_identity(stream, seed, index, cfg, physical, selection_sha, declaration, noise_info):
    checkpoint = source_checkpoint(stream, seed)
    return dict(arm=stream["id"], graph=cfg["design"], seed=seed, round=0, role="evaluation", index=index,
        terminal_mask=0, training_feedback_eligible=False, offline_train_eligible=False,
        evaluation_purpose="fresh_confirmation", source_kind=stream["kind"],
        source_sampler=SAMPLERS[stream["kind"]], source_original_round=stream["selected_round"],
        source_state_key="actor" if stream["kind"] == "one_step" else "flow" if stream["kind"] == "behavior_flow" else "model",
        source_checkpoint=checkpoint["path"], checkpoint_sha256=checkpoint["sha256"],
        source_config_binding=stream["source_config"], selection_lock_sha256=selection_sha,
        fresh_declaration_sha256=object_sha(declaration), shared_noise_binding=noise_info["artifact"],
        noise_seed=NOISE_SEED, noise_sha256=noise_info["noise_sha256"][index],
        config_sha256=physical["config_sha256"], dataset_sha256=cfg["dataset_sha256"], cache_sha256=cfg["cache_sha256"],
        predicted_q=None, stage="global_route_only", fresh_drt_preselected=index in DRT_INDICES,
        fresh_drt_repair_cap=REPAIR_CAP, run_id=f"go2o_qsfresh_{cfg['run_prefix']}_s{seed}_i{index}")


def verify_row(row, stream, seed, index, cfg, physical, selection_sha, declaration, noises, noise_info):
    expected = row_identity(stream, seed, index, cfg, physical, selection_sha, declaration, noise_info)
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Fresh request changed its locked source/noise/eligibility/DRT identity")
    sample = backend.checked_request(row, argparse.Namespace(seed=seed, round=0), cfg, physical)
    with np.load(sample / "actions.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["noise"], noises[index]):
            raise ValueError("Fresh request does not preserve the actual shared noise")
    if stream["kind"] == "public_official20" and row.get("trajectory_sha256") != sha(sample / "trajectory.pt"):
        raise ValueError("Official public sampling trajectory changed")
    return sample


def verify_completion(folder, cfg, selection_sha, declaration):
    marker = read(folder / "completed.json")
    expected = dict(status="completed", count=COUNT, config_sha256=sha(folder.parents[2] / "config.json"),
        tasks_sha256=sha(folder / "tasks.jsonl"), selection_lock_sha256=selection_sha,
        fresh_declaration_sha256=object_sha(declaration), backend_calls=0, optimizer_updates=0,
        evaluation_never_replayed=True)
    if any(marker.get(k) != v for k, v in expected.items()):
        raise ValueError("Completed fresh export marker changed")
    return marker


def build_sampler(stream, seed, design):
    """No critic/optimizer is constructed; every policy parameter stays frozen."""
    problem = load_problem(design, "cuda")
    if stream["kind"] == "public_official20":
        model, _ = load_model(problem, device="cuda", steps=20, hard_constraint=True, w_reg=.1, w_dist=1.)
    else:
        model = GraphPolicy(problem).eval()
        saved = torch.load(source_checkpoint(stream, seed)["path"], map_location="cuda", weights_only=False)
        key = "actor" if stream["kind"] == "one_step" else "flow"
        model.load_state_dict(saved[key], strict=True)
        del saved
    model.eval().requires_grad_(False)
    return problem, model


def generate_raw(stream, model, problem, noise):
    if stream["kind"] == "one_step":
        return model(noise[None])[0], None
    if stream["kind"] == "behavior_flow":
        return model.sample_flow(noise[None], steps=10)[0], None
    raw, used, trajectory = generate(model, problem, noise=noise, intermediate_every=1)
    if not torch.equal(used, noise):
        raise ValueError("Official public sampler changed the explicit shared noise")
    return raw, trajectory


def export_stream(path, expected_sha, stream_id, seed):
    lock, configs, physicals, declaration = declare(path, expected_sha)
    stream = next((s for s in lock["streams"] if s["id"] == stream_id), None)
    if stream is None or seed not in stream["seeds"]:
        raise ValueError("Undeclared fresh stream/training seed")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Frozen policy evaluation requires an allocated Slurm CUDA GPU")
    configure_export_environment(lock["hardware_software"])
    cfg, physical = configs[stream_id], physicals[stream_id]
    experiment = Path(stream["evaluation_config"]["path"]).parent
    folder = experiment / f"seed_{seed}/round_0/requests"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".fresh_export.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A completed export is validated without loading weights or new candidates.
        if (folder / "completed.json").exists():
            verify_completion(folder, cfg, expected_sha, declaration)
            rows, _ = load_requests(experiment, seed, 0)
            n = len(rows[0]["raw_action"])
            noises, noise_info = shared_noise(lock["output_root"], expected_sha, n)
            if len(rows) != COUNT or [r["index"] for r in rows] != list(range(COUNT)):
                raise ValueError("Fresh export denominator changed")
            for index, row in enumerate(rows):
                verify_row(row, stream, seed, index, cfg, physical, expected_sha, declaration, noises, noise_info)
            return dict(status="already_exported", count=COUNT, backend_calls=0)
        problem, model = build_sampler(stream, seed, cfg["design"])
        noises, noise_info = shared_noise(lock["output_root"], expected_sha, len(problem.x), create=True)
        metric = MacroHPWL(problem.benchmark_info)
        fixed = problem.cond.is_ports.flatten().bool()
        rows = []
        for index in range(COUNT):
            sample = folder / stream_id / "evaluation" / str(index)
            if (sample / "request.json").exists():
                row = read(sample / "request.json")
                verify_row(row, stream, seed, index, cfg, physical, expected_sha, declaration, noises, noise_info)
                rows.append(row)
                continue
            if (sample / "outcome.json").exists() or (sample / "backend_attempts").exists():
                raise ValueError("Refusing to regenerate an orphan physical request")
            sample.mkdir(parents=True, exist_ok=True)
            noise = torch.from_numpy(noises[index]).to("cuda")
            with torch.no_grad():
                raw, trajectory = generate_raw(stream, model, problem, noise)
                if not torch.equal(raw[fixed], problem.x[fixed]):
                    raise ValueError("Frozen sampler moved fixed nodes")
                action = torch.where(fixed[:, None], problem.x, legalize(raw, problem))
                if index == 0:
                    repeated, _ = generate_raw(stream, model, problem, noise)
                    if not torch.equal(raw, repeated) or not torch.equal(action, torch.where(fixed[:, None], problem.x, legalize(repeated, problem))):
                        raise ValueError("Explicit-noise frozen sampler/legalizer is not deterministic")
            raw_array, executed = raw.cpu().numpy(), action.cpu().numpy()
            geometry = legality(executed, metric.sizes)
            row = dict(row_identity(stream, seed, index, cfg, physical, expected_sha, declaration, noise_info),
                sample_dir=str(sample), raw_action=raw_array.tolist(), executed_action=executed.tolist(),
                raw_action_sha256=action_sha(raw_array), action_sha256=action_sha(executed),
                geometry=geometry, macro_hpwl=float(metric(executed)), raw_macro_hpwl=float(metric(raw_array)),
                legalizer_jump_rms=float((action[~fixed]-raw[~fixed]).square().mean().sqrt()))
            np.savez_compressed(sample / "actions.npz", noise=noises[index], raw_action=raw_array, action=executed)
            if trajectory is not None:
                torch.save(trajectory, sample / "trajectory.pt")
                row["trajectory_sha256"] = sha(sample / "trajectory.pt")
            if geometry["legal"]:
                save_sample(sample, action, noise, problem, row)
                output = sample / (cfg["design"] + ".def")
                output.unlink(missing_ok=True)
                output.with_suffix(".audit.json").unlink(missing_ok=True)
                audit = convert(cfg["design"], sample / "macro_physical.json", output)
                row.update(def_path=str(output), def_sha256=sha(output), rounded_geometry=audit["legal_no_halo"])
                if not audit["legal_no_halo"]:
                    row["pre_backend_failure"] = "rounded_DEF_geometry"
            else:
                row["pre_backend_failure"] = "normalized_geometry"
            atomic_json(sample / "request.json", row)
            verify_row(row, stream, seed, index, cfg, physical, expected_sha, declaration, noises, noise_info)
            rows.append(row)
        # Revalidate frozen source/code/config bytes before committing request inventory.
        validate_lock(path, expected_sha)
        tasks = folder / "tasks.jsonl"
        tmp = tasks.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        tmp.replace(tasks)
        immutable(folder / "completed.json", dict(status="completed", count=COUNT,
            config_sha256=physical["config_sha256"], tasks_sha256=sha(tasks),
            selection_lock_sha256=expected_sha, fresh_declaration_sha256=object_sha(declaration),
            backend_calls=0, optimizer_updates=0, evaluation_never_replayed=True,
            slurm_job_id=os.environ["SLURM_JOB_ID"]))
    return dict(status="exported", count=COUNT, backend_calls=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--stage", choices=("validate", "declare", "export"), required=True)
    parser.add_argument("--stream")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a bounded Slurm allocation for fresh export validation")
    if args.stage == "export":
        if args.stream is None or args.seed is None:
            parser.error("export requires --stream and --seed")
        torch.use_deterministic_algorithms(True)
        result = export_stream(args.selection_lock, args.selection_sha256, args.stream, args.seed)
    else:
        result = declare(args.selection_lock, args.selection_sha256) if args.stage == "declare" else validate_lock(args.selection_lock, args.selection_sha256)
        result = dict(status=args.stage + "_completed", streams=len(result[0]["streams"]), noise_arrays_created=0,
                      layout_generation_calls=0, backend_calls=0, optimizer_updates=0)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
