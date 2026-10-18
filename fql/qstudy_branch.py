"""Import an identical offline prefix for one frozen-vs-updated-flow control.

This is provenance-preserving measurement reuse, not new training or new PPA.
Only arm/run-prefix names and online_flow_update=False -> True may differ.
The latter is inactive during round zero in the frozen actor implementation.

Each seed prefix is staged and atomically published; branch_import.json is the
final transaction gate. Do not start the unchanged controller before that gate.
Native tools are never launched: saved geometry audits are mandatory.
Reserved journal ``*.tmp.PID`` files are uncommitted and discarded on restart.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np
import torch

from .graph_o2o_archive import sha
from .graph_o2o_backend import load_requests
from . import qstudy_backend as backend

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCES = ("fql/qstudy_actor.py", "fql/qstudy_models.py", "fql/qstudy_screen.py",
                 "fql/graph_o2o_models.py", "baseline/api.py")
IMPORT_SOURCES = ("fql/qstudy_branch.py", *TRAIN_SOURCES, "fql/qstudy_backend.py",
                  "fql/graph_o2o_backend.py", "fql/graph_o2o_geometry.py",
                  "fql/graph_o2o_design_verify.py", "fql/qstudy_controller.py")
SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
PAYLOAD_KEYS = ("flow", "actor", "critic", "optimizers", "rng", "cuda_rng", "generators", "step")


def read(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp."+str(os.getpid()))
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def recover_journal_temporaries(directory, names):
    """Discard only this protocol's uncommitted root journal writes.

    A process can die between fsync and replace; these are never published
    evidence. Other files, symlinks, and directories are never removed.
    """
    directory = Path(directory)
    reserved = re.compile(r"(?:"+"|".join(re.escape(name) for name in names)+r")\.tmp\.[0-9]+\Z")
    for path in directory.iterdir():
        if reserved.fullmatch(path.name):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Nonregular reserved journal temporary")
            path.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False)+"\n").encode())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def tree_sha(value):
    """Canonical values/dtypes/shapes/strides, including optimizer and RNG state."""
    digest = hashlib.sha256()

    def add(tag, payload=b""):
        digest.update(tag.encode()+b":"+str(len(payload)).encode()+b":"+payload)

    def walk(item):
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.device.type == "meta":
                raise ValueError("Unsupported checkpoint tensor layout")
            tensor = item.detach().cpu()
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite checkpoint state")
            add("tensor", str((str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()))).encode())
            add("bytes", tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Object arrays cannot bind checkpoint state")
            add("ndarray", str((item.dtype.str, item.shape, item.strides)).encode())
            add("bytes", item.tobytes())
        elif isinstance(item, np.generic):
            walk(item.item())
        elif isinstance(item, dict):
            add("dict", str(len(item)).encode())
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                walk(key); walk(item[key])
        elif isinstance(item, (list, tuple)):
            add(type(item).__name__, str(len(item)).encode())
            for child in item:
                walk(child)
        elif item is None:
            add("none")
        elif isinstance(item, bool):
            add("bool", str(item).encode())
        elif isinstance(item, int):
            add("int", str(item).encode())
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Nonfinite checkpoint scalar")
            add("float", item.hex().encode())
        elif isinstance(item, str):
            add("str", item.encode())
        else:
            raise ValueError("Unsupported checkpoint state type: "+type(item).__name__)
    walk(value)
    return digest.hexdigest()


def state_fingerprints(saved):
    if any(key not in saved for key in (*PAYLOAD_KEYS, "binding")):
        raise ValueError("Completed checkpoint lacks model/optimizer/RNG/generator state")
    if set(saved["optimizers"]) != {"flow", "actor", "critic"} or set(saved["generators"]) != {"q", "actor", "flow"}:
        raise ValueError("Checkpoint optimizer/generator identities differ")
    # Bind every payload field, including future fields, rather than cherry-pick weights.
    payload = {k:v for k,v in saved.items() if k not in {"binding", "prefix_import"}}
    return dict(all_payload=tree_sha(payload), **{key:tree_sha(saved[key]) for key in PAYLOAD_KEYS})


def configurations(source, target):
    if source.get("online_flow_update") is not False or target.get("online_flow_update") is not True:
        raise ValueError("Only the explicit online flow False -> True branch is supported")
    permitted = {"arm", "run_prefix", "online_flow_update", "online_rounds"}
    for key in set(source) | set(target):
        if key not in permitted and (key not in source or key not in target or source[key] != target[key]):
            raise ValueError("Unsupported branch configuration difference: "+key)
    a, b = source.get("online_rounds", 2), target.get("online_rounds", 2)
    if type(a) is not int or type(b) is not int or a != b or a < 1:
        raise ValueError("online_rounds normalization must be semantically identical and positive")
    for key in ("arm", "run_prefix"):
        if not all(isinstance(c.get(key), str) and SAFE_NAME.fullmatch(c[key]) for c in (source, target)):
            raise ValueError("Unsafe branch "+key)
        if source[key] == target[key]:
            raise ValueError("Branch needs distinct logical "+key)
    seeds = source.get("seeds")
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds) or target.get("seeds") != seeds):
        raise ValueError("Branch must preserve all declared independent seeds")
    return seeds


def inventory(directory):
    directory = Path(directory)
    result = {}
    for parent, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs+names:
            if (Path(parent)/name).is_symlink():
                raise ValueError("Imported logical artifacts must be regular files/directories")
        for name in names:
            path = Path(parent)/name
            if not path.is_file():
                raise ValueError("Nonregular imported artifact")
            result[str(path.relative_to(directory))] = sha(path)
    return result


def native_evidence(out, row, sample, tasks, cfg, verification, binding):
    """Use the unchanged native validators, guarded against any tool invocation."""
    for key in ("seed", "round", "arm", "role", "index", "graph", "training_feedback_eligible"):
        if out.get(key) != row.get(key):
            raise ValueError("Source outcome role/identity differs: "+key)
    for key in ("raw_action", "executed_action"):
        if out.get(key) != row.get(key):
            raise ValueError("Outcome coordinates differ from exact requested action: "+key)
    if out.get("status") not in {"success", "geometry_failure", "physical_failure"}:
        raise ValueError("Unresolved/infrastructure result cannot be imported")
    native = {}
    if out.get("physical_run"):
        run = Path(out["physical_run"])
        manifest = read(run/"manifest.json")
        if manifest.get("success") and not (run/"backend_macro_audit.json").is_file():
            raise ValueError("Missing native geometry audit; importer must never run OpenROAD")
        for name, field in (("manifest.json", "physical_manifest_sha256"),
                            ("metrics.json", "metrics_sha256"),
                            ("backend_macro_audit.json", "backend_macro_geometry_sha256"),
                            ("floorplan_place_cts.log", "failure_log_sha256")):
            path = run/name
            if out.get(field):
                if not path.is_file() or sha(path) != out[field]:
                    raise ValueError("Native evidence fingerprint differs: "+name)
                native[str(path.resolve())] = sha(path)
        if out["status"] == "success":
            if not all(out.get(k) for k in ("physical_manifest_sha256", "metrics_sha256", "backend_macro_geometry_sha256")):
                raise ValueError("Successful import requires complete native fingerprints")
            geometry = read(run/"backend_macro_audit.json")
            if (out.get("backend_macro_legal") is not True or geometry.get("legal_no_halo") is not True
                    or out.get("backend_track_snapped_macro_count") != len(geometry["input_to_actual_changes"])):
                raise ValueError("Illegal or inconsistent actual native macro geometry")
    backend.checked_existing(out, row, sample, tasks, binding, cfg, verification)
    # Raw/executed action and source checkpoint bytes are already checked by checked_request.
    if row.get("def_path"):
        path = Path(row["def_path"])
        native[str(path.resolve())] = sha(path)
        native[str(path.with_suffix(".audit.json").resolve())] = sha(path.with_suffix(".audit.json"))
    return native


def source_seed(source, cfg, seed, context):
    folder = source/f"seed_{seed}/round_0"
    done, binding = read(folder/"completed.json"), read(folder/"binding.json")
    checkpoint = folder/"checkpoint.pt"
    if (done.get("status") != "completed" or done.get("binding") != binding
            or done.get("checkpoint_sha256") != sha(checkpoint)
            or done.get("steps") != cfg["offline_actor_steps"]
            or done.get("offline_critic_frozen") is not True or done.get("online_flow_update") is not False):
        raise ValueError("Source offline training is incomplete or changed")
    expected = dict(config_sha256=sha(source/"config.json"), dataset_sha256=cfg["dataset_sha256"],
        common_sha256=cfg["common_checkpoint_sha256"][str(seed)],
        initial_critic_sha256=cfg["critic_checkpoint_sha256"][str(seed)],
        round=0, seed=seed, replay=[], parent=None,
        source_sha256={name:sha(ROOT/name) for name in TRAIN_SOURCES})
    if binding != expected:
        raise ValueError("Source training binding/source/config/initialization changed")
    for key in ("common", "critic"):
        path = Path(cfg[key+"_checkpoints"][str(seed)])
        if sha(path) != cfg[key+"_checkpoint_sha256"][str(seed)]:
            raise ValueError("Source initialization changed: "+key)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["binding"] != binding or saved["step"] != cfg["offline_actor_steps"] or "prefix_import" in saved:
        raise ValueError("Source checkpoint is not the declared original offline prefix")
    state = state_fingerprints(saved)
    del saved
    rows, tasks = load_requests(source, seed, 0)
    roles = {"collect":cfg["online_samples"], "evaluation":cfg["evaluation_samples"]}
    if set(row["role"] for row in rows) != set(roles):
        raise ValueError("Source export role plan differs")
    for role, count in roles.items():
        selected = [r for r in rows if r["role"] == role]
        if len(selected) != count or {r["index"] for r in selected} != set(range(count)):
            raise ValueError("Source export role count/index differs")
        if len({r["noise_sha256"] for r in selected}) != count:
            raise ValueError("Duplicate noise within source role plan")
    evidence = {str(path.resolve()):sha(path) for path in (
        folder/"completed.json", folder/"binding.json", checkpoint, tasks, tasks.with_name("completed.json"))}
    if (folder/"learning_curve.jsonl").exists():
        evidence[str((folder/"learning_curve.jsonl").resolve())] = sha(folder/"learning_curve.jsonl")
    bcfg, verification, physical_binding = context
    native_runs = []
    for row in rows:
        sample = backend.checked_request(row, argparse.Namespace(seed=seed, round=0), bcfg, physical_binding)
        if (row["arm"] != cfg["arm"] or row["checkpoint_sha256"] != sha(checkpoint)
                or Path(row["source_checkpoint"]).resolve() != checkpoint.resolve()):
            raise ValueError("Source request belongs to another actor checkpoint")
        expected_noise = (cfg["evaluation_noise_seed"] if row["role"] == "evaluation" else
                          cfg["collection_noise_seed"]+seed*100)
        if row["noise_seed"] != expected_noise or row.get("objective_weights") != cfg["weights"]:
            raise ValueError("Source export noise/objective plan differs")
        if row.get("evaluation_purpose", "development_validation") != cfg.get("evaluation_purpose", "development_validation"):
            raise ValueError("Source export evaluation purpose differs")
        if not (sample/"outcome.json").is_file():
            raise ValueError("Every source collect/evaluation request must have native terminal evidence")
        out = read(sample/"outcome.json")
        evidence.update(native_evidence(out, row, sample, tasks, bcfg, verification, physical_binding))
        for path in (sample/"request.json", sample/"actions.npz", sample/"outcome.json"):
            evidence[str(path.resolve())] = sha(path)
        if out.get("physical_run"):
            native_runs.append(str(Path(out["physical_run"]).resolve()))
    return dict(seed=seed, state=state, source_checkpoint=str(checkpoint.resolve()),
        source_checkpoint_sha256=sha(checkpoint), source_binding=binding,
        evidence=evidence, native_runs=sorted(set(native_runs)), source_rows=len(rows)), rows


def checked_files(files):
    for path, digest in files.items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError("Import provenance changed: "+path)


def build_seed(staged_seed, target, cfg, plan_id, spec, rows):
    seed = spec["seed"]
    staged = staged_seed/"round_0"
    staged.mkdir(parents=True)
    final = target/f"seed_{seed}/round_0"
    saved = torch.load(spec["source_checkpoint"], map_location="cpu", weights_only=False)
    if state_fingerprints(saved) != spec["state"]:
        raise ValueError("Source checkpoint payload changed while staging")
    binding = copy.deepcopy(saved["binding"])
    binding["config_sha256"] = sha(target/"config.json")
    provenance = dict(plan_sha256=plan_id, source_checkpoint=spec["source_checkpoint"],
        source_checkpoint_sha256=spec["source_checkpoint_sha256"], new_train_updates=0,
        changed_binding_fields=["config_sha256"], shared_counterfactual_prefix=True)
    saved["binding"], saved["prefix_import"] = binding, provenance
    torch.save(saved, staged/"checkpoint.pt")
    rebuilt = torch.load(staged/"checkpoint.pt", map_location="cpu", weights_only=False)
    if state_fingerprints(rebuilt) != spec["state"]:
        raise ValueError("Imported model/optimizer/RNG/step state differs")
    del rebuilt, saved
    checkpoint_sha = sha(staged/"checkpoint.pt")
    atomic_json(staged/"binding.json", binding)
    done = read(Path(spec["source_checkpoint"]).with_name("completed.json"))
    done.update(binding=binding, checkpoint_sha256=checkpoint_sha, online_flow_update=True,
        online_flow_update_applied=False, imported=True, new_train_updates=0,
        elapsed_this_attempt_s=0., prefix_import=provenance)
    atomic_json(staged/"completed.json", done)
    curve = Path(spec["source_checkpoint"]).with_name("learning_curve.jsonl")
    if curve.exists():
        shutil.copyfile(curve, staged/curve.name)
    requests = staged/"requests"
    requests.mkdir()
    new_rows, outcomes = [], []
    for row in rows:
        old_sample = Path(row["sample_dir"])
        relative = Path(cfg["arm"])/row["role"]/str(row["index"])
        sample = requests/relative
        sample.mkdir(parents=True)
        changed = copy.deepcopy(row)
        changed.update(arm=cfg["arm"], sample_dir=str(final/"requests"/relative),
            source_checkpoint=str(final/"checkpoint.pt"), checkpoint_sha256=checkpoint_sha,
            config_sha256=binding["config_sha256"],
            run_id=f"go2o_qs1_{cfg['run_prefix']}_s{seed}_r0_{row['role']}_{row['index']}",
            prefix_import=dict(plan_sha256=plan_id, source_request=str(old_sample/"request.json"),
                source_request_sha256=sha(old_sample/"request.json"), shared_prefix=True))
        shutil.copyfile(old_sample/"actions.npz", sample/"actions.npz")
        atomic_json(sample/"request.json", changed)
        out = read(old_sample/"outcome.json")
        out.update({key:changed[key] for key in ("run_id", "arm", "role", "index", "seed", "round", "graph",
            "checkpoint_sha256", "config_sha256", "training_feedback_eligible")})
        out.update(request_sha256=sha(sample/"request.json"), new_measurement=False,
            prefix_import=dict(plan_sha256=plan_id, source_outcome=str(old_sample/"outcome.json"),
                source_outcome_sha256=sha(old_sample/"outcome.json"), unique_physical_calls=0,
                shared_counterfactual_prefix=True))
        outcomes.append((sample, out)); new_rows.append(changed)
    tasks = requests/"tasks.jsonl"
    atomic_bytes(tasks, "".join(json.dumps(row, allow_nan=False)+"\n" for row in new_rows).encode())
    for sample, out in outcomes:
        out["tasks_sha256"] = sha(tasks)
        atomic_json(sample/"outcome.json", out)
    atomic_json(requests/"completed.json", dict(status="completed", count=len(new_rows),
        tasks_sha256=sha(tasks), config_sha256=binding["config_sha256"], evaluation_never_replayed=True,
        selection="import all original predetermined noises; no new training or physical selection",
        prefix_import=provenance))
    return dict(seed=seed, artifacts=inventory(staged), state=spec["state"],
        checkpoint_sha256=checkpoint_sha, plan_sha256=plan_id,
        new_train_updates=0, unique_physical_calls=0)


def validate_published(target, cfg, spec, ready, context, source_rows):
    folder = target/f"seed_{spec['seed']}/round_0"
    if inventory(folder) != ready["artifacts"]:
        raise ValueError("Published import artifact inventory changed")
    saved = torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=False)
    if state_fingerprints(saved) != spec["state"]:
        raise ValueError("Published checkpoint state differs from offline prefix")
    expected_binding = copy.deepcopy(spec["source_binding"])
    expected_binding["config_sha256"] = sha(target/"config.json")
    if saved["binding"] != expected_binding or read(folder/"binding.json") != expected_binding:
        raise ValueError("Imported checkpoint changed more than its logical config binding")
    expected_provenance = dict(plan_sha256=ready["plan_sha256"], source_checkpoint=spec["source_checkpoint"],
        source_checkpoint_sha256=spec["source_checkpoint_sha256"], new_train_updates=0,
        changed_binding_fields=["config_sha256"], shared_counterfactual_prefix=True)
    if saved.get("prefix_import") != expected_provenance:
        raise ValueError("Imported checkpoint logical provenance changed")
    del saved
    rows, tasks = load_requests(target, spec["seed"], 0)
    if len(rows) != len(source_rows):
        raise ValueError("Imported request count changed")
    bcfg, verification, physical = context
    request_fields = {"arm", "sample_dir", "source_checkpoint", "checkpoint_sha256", "config_sha256", "run_id", "prefix_import"}
    outcome_fields = {"run_id", "arm", "checkpoint_sha256", "config_sha256", "request_sha256", "tasks_sha256", "prefix_import", "new_measurement"}
    for row, original in zip(rows, source_rows):
        if {k:v for k,v in row.items() if k not in request_fields} != {
                k:v for k,v in original.items() if k not in request_fields}:
            raise ValueError("Imported coordinates/noise/DEF/geometry/export plan changed")
        if row["arm"] != cfg["arm"] or Path(row["source_checkpoint"]).resolve() != (folder/"checkpoint.pt").resolve():
            raise ValueError("Imported logical actor identity differs")
        sample = backend.checked_request(row, argparse.Namespace(seed=spec["seed"], round=0), bcfg, physical)
        out = read(sample/"outcome.json")
        original_out = read(Path(original["sample_dir"])/"outcome.json")
        if {k:v for k,v in out.items() if k not in outcome_fields} != {
                k:v for k,v in original_out.items() if k not in outcome_fields}:
            raise ValueError("Imported physical evidence/feedback changed")
        if out.get("new_measurement") is not False:
            raise ValueError("Imported native result must be marked reused")
        native_evidence(out, row, sample, tasks, bcfg, verification, physical)


def import_prefix(source, target, target_config=None, *, interrupt=None):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Checkpoint import/native validation requires a Slurm CPU allocation")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and target must be disjoint experiment trees")
    source_cfg = read(source/"config.json")
    if target_config is None:
        target_config = read(target/"config.json")
    cfg = copy.deepcopy(target_config)
    seeds = configurations(source_cfg, cfg)
    target.mkdir(parents=True, exist_ok=True)
    if (target/".qstudy_branch.lock").is_symlink():
        raise ValueError("Import lock must not be a symlink")
    with (target/".qstudy_branch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        recover_journal_temporaries(target, ("config.json", "import_plan.json", "branch_import.json"))
        if any((target/name).is_symlink() for name in ("config.json", "import_plan.json", "branch_import.json")):
            raise ValueError("Logical import journals must not be symlinks")
        if (target/"config.json").exists():
            if read(target/"config.json") != cfg:
                raise ValueError("Target configuration is immutable")
        else:
            atomic_json(target/"config.json", cfg)
        plan_path = target/"import_plan.json"
        if not plan_path.exists() and set(p.name for p in target.iterdir())-{"config.json", ".qstudy_branch.lock"}:
            raise ValueError("Target must be empty apart from its immutable config")
        source_context = backend.context(source)
        target_context = backend.context(target)
        cache = ROOT/"baseline/runtime/benchmark_cache"/(cfg["design"]+".pt")
        if sha(cache) != cfg["cache_sha256"]:
            raise ValueError("Frozen graph cache changed")
        # Exact config semantics already matched; contexts independently verify design,
        # dataset, pinned evaluator revision, reference image and eight-thread budget.
        source_binding, target_binding = source_context[2], target_context[2]
        if {k:v for k,v in source_binding.items() if k != "config_sha256"} != {
                k:v for k,v in target_binding.items() if k != "config_sha256"}:
            raise ValueError("Branch physical evaluator context differs")
        specifications, request_sets = [], {}
        for seed in seeds:
            spec, rows = source_seed(source, source_cfg, seed, source_context)
            specifications.append(spec); request_sets[seed] = rows
        declaration = dict(version=1, source_experiment=str(source), target_experiment=str(target),
            source_config_sha256=sha(source/"config.json"), target_config_sha256=sha(target/"config.json"),
            graph_cache=dict(path=str(cache.resolve()), sha256=sha(cache)),
            seeds=seeds, physical_context=source_binding,
            sources={name:sha(ROOT/name) for name in IMPORT_SOURCES}, prefixes=specifications,
            allowed_differences=["arm", "run_prefix", "online_flow_update=False->True"],
            online_rounds=cfg.get("online_rounds", 2), new_train_updates=0, unique_physical_calls=0,
            interpretation="Identical round0 prefix reused; no new independent training or measurement",
            launch_gate="branch_import.json must exist and validate before the frozen controller starts")
        plan_id = object_sha(declaration)
        if plan_path.exists():
            plan = read(plan_path)
            if plan != dict(declaration=declaration, declaration_sha256=plan_id):
                raise ValueError("Import source/config/code/provenance plan changed")
        else:
            atomic_json(plan_path, dict(declaration=declaration, declaration_sha256=plan_id))
        staging = target/".branch_staging"
        if staging.is_symlink():
            raise ValueError("Staging directory must not be a symlink")
        staging.mkdir(exist_ok=True)
        allowed = {"config.json", ".qstudy_branch.lock", "import_plan.json", ".branch_staging", "branch_import.json"}
        if set(p.name for p in target.iterdir())-allowed-{f"seed_{s}" for s in seeds}:
            raise ValueError("Foreign artifacts in reserved import target")
        ready_records = []
        for spec in specifications:
            seed = spec["seed"]
            checked_files(spec["evidence"])
            stage_seed = staging/f"seed_{seed}"
            ready_path = staging/f"ready_seed_{seed}.json"
            published = target/f"seed_{seed}"
            if published.is_symlink() or (published/"round_0").is_symlink():
                raise ValueError("Published logical prefix must not be a symlink")
            if ready_path.exists():
                ready = read(ready_path)
                if ready["plan_sha256"] != plan_id or ready["state"] != spec["state"]:
                    raise ValueError("Staged prefix no longer binds import plan")
            else:
                if published.exists():
                    raise ValueError("Unjournaled existing target seed cannot be overwritten")
                if stage_seed.exists():
                    if stage_seed.is_symlink():
                        raise ValueError("Staging path is a symlink")
                    shutil.rmtree(stage_seed)
                ready = build_seed(stage_seed, target, cfg, plan_id, spec, request_sets[seed])
                atomic_json(ready_path, ready)
            if interrupt:
                interrupt("staged", seed)
            if published.exists():
                if not (published/"round_0").is_dir():
                    raise ValueError("Foreign target seed directory")
            else:
                if inventory(stage_seed/"round_0") != ready["artifacts"]:
                    raise ValueError("Staged import artifact corrupted before publication")
                os.replace(stage_seed, published)
            if interrupt:
                interrupt("published", seed)
            validate_published(target, cfg, spec, ready, target_context, request_sets[seed])
            ready_records.append(ready)
        # Source evidence remains immutable throughout the transaction.
        for spec in specifications:
            checked_files(spec["evidence"])
        if sha(source/"config.json") != declaration["source_config_sha256"] or sha(target/"config.json") != declaration["target_config_sha256"]:
            raise ValueError("Configuration changed during prefix import")
        result = dict(status="completed", schema="qstudy_online_flow_prefix_import_v1",
            plan_sha256=plan_id, import_plan_sha256=sha(plan_path), prefixes=ready_records,
            source_experiment=str(source), target_experiment=str(target), seeds=seeds,
            new_train_updates=0, unique_physical_calls=0,
            reused_native_physical_runs=sorted({p for s in specifications for p in s["native_runs"]}),
            no_new_independent_samples=True, all_source_and_imported_native_results_validated=True)
        final = target/"branch_import.json"
        if final.exists():
            if read(final) != result:
                raise ValueError("Final immutable import manifest changed")
        else:
            atomic_json(final, result)
        print(json.dumps(dict(status="completed", manifest=str(final), seeds=seeds,
                             new_train_updates=0, unique_physical_calls=0)), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-config", type=Path,
        help="Optional proposed config JSON; otherwise target/config.json must already exist")
    args = parser.parse_args()
    import_prefix(args.source, args.target, read(args.target_config) if args.target_config else None)


if __name__ == "__main__":
    main()
