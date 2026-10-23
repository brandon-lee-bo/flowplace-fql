"""Audited V2 offline trajectories, matched BC/FQL training, and rollout export.

An existing measured *executed* layout is replayed in the deterministic macro
order.  It is usable only if each action is an identity under the sequential
environment, so the terminal layout still matches the measured backend input.
Its N prefixes are N transitions, not N independent backend measurements.
MC warmup labels describe replay of that recorded layout.  Subsequent critic
updates use genuine FQL Bellman targets under the current student's continuation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

# Must be bound before baseline.api prepends upstream paths.
import analysis  # noqa: F401
from .sequential_env import SequentialPlacementEnv, geometry_audit
from .sequential_models import (SequentialActor, SequentialCritics, actor_objective,
                                bellman_targets, frozen_target, polyak_update)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sequential_fql_v2_4"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def binding_digest(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_fingerprints():
    """Bind the adapters and their upstream model/configuration dependency tree.

    networks/__init__.py imports the complete network package, so include its
    Python modules, not only gnn.py.  Bind the Hydra configuration tree as well:
    changing an included size/conv/default config must invalidate resume/export.
    Checkpoint bytes and design graph bytes are bound separately.
    """
    names = {
        "analysis/__init__.py", "baseline/api.py", "backend/scripts/export_macro_def.py",
        "fql/sequential_models.py", "fql/sequential_train.py", "fql/sequential_env.py",
        "fql/sequential_teacher.py", "fql/flow_policy_config.py",
        "fql/released_flowplace_teacher.py", "fql/critic_gate.py", "fql/relabel_backend.py",
        "fql/core.py", "fql/provenance.py",
        "upstream/flowplace/benchmark_utils/openroad_utils.py",
    }
    upstream = ROOT / "upstream/flowplace/flow_matching"
    for name in ("__init__.py", "models.py", "schedulers.py", "pos_encoding.py",
                 "guidance.py", "hard_constraint_legalizer.py", "utils.py"):
        names.add(str((upstream / name).relative_to(ROOT)))
    for folder, suffixes in (("networks", {".py"}), ("configs", {".yaml", ".yml"})):
        paths = [p for p in (upstream / folder).rglob("*") if p.is_file() and p.suffix in suffixes]
        if not paths:
            raise ValueError("Missing source dependency tree: " + folder)
        names.update(str(p.relative_to(ROOT)) for p in paths)
    return {name: sha(ROOT / name) for name in sorted(names)}


def verify_source_binding(binding):
    actual = source_fingerprints()
    expected = binding.get("source_sha256", {})
    if actual != expected:
        changed = sorted(name for name in actual.keys() | expected.keys()
                         if actual.get(name) != expected.get(name))
        raise ValueError("Source/config dependency identity changed: " + ", ".join(changed))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed across resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def normalize_config(user):
    cfg = dict(schema=SCHEMA, design="bp_fe", batch_size=2, bc_steps=100,
               critic_mc_steps=100, train_steps=200, alpha=10., q_weight=1.,
               actor_lr=1e-5, critic_lr=3e-4, critic_weight_decay=1e-4,
               critic_width=128, gamma=1., target_tau=.005,
               target_reduction="mean", actor_q_reduction="mean", grad_clip=1.,
               checkpoint_every=25, log_every=10, env_padding=0., arms=["bc", "fql"])
    cfg.update(user)
    if cfg["schema"] != SCHEMA or cfg["gamma"] != 1. or cfg["env_padding"] != 0.:
        raise ValueError("Initial V2 protocol requires gamma=1 and identity-compatible padding=0")
    for key in ("batch_size", "critic_width", "checkpoint_every", "log_every"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError("Positive integer required: " + key)
    for key in ("bc_steps", "critic_mc_steps", "train_steps"):
        if not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError("Nonnegative integer required: " + key)
    for key in ("alpha", "q_weight"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("Nonnegative finite coefficient required: " + key)
    if not 0 < cfg["target_tau"] <= 1 or cfg["grad_clip"] <= 0:
        raise ValueError("Invalid target update/gradient clipping")
    if cfg["target_reduction"] not in ("mean", "min") or cfg["actor_q_reduction"] not in ("mean", "min"):
        raise ValueError("Declare twin-Q reductions explicitly")
    if cfg["arms"] != ["bc", "fql"]:
        raise ValueError("This entry point runs both matched BC and FQL arms")
    return cfg


def load_measured_layouts(path, design):
    """Use the existing strict GRT loader; audit selected labels against artifacts.

This initial input contract accepts the legal-layout critic_gate dataset schema.
Other schemas (including physical-failure utility rows) must have an explicit
adapter; they are not silently filtered into successful trajectories.
"""
    from .critic_gate import load_train_validation
    from .relabel_backend import action_sha256
    path = Path(path).resolve()
    data, manifest, manifest_path = load_train_validation(path)
    if manifest.get("args", {}).get("design") != design:
        raise ValueError("Dataset and netlist design differ")
    labels_path = path.with_name("backend_labels.jsonl")
    selected = {int(row) for split in data.values() for row in split["source_row_index"]}
    labels = {}
    for line in labels_path.read_text().splitlines():
        row = json.loads(line)
        if int(row["row_index"]) in selected:
            if int(row["row_index"]) in labels:
                raise ValueError("Repeated backend source row")
            labels[int(row["row_index"])] = row
    if set(labels) != selected:
        raise ValueError("Missing measured provenance for selected train/validation rows")
    bound = []
    for split_name, split in data.items():
        for i, source_id in enumerate(split["source_row_index"]):
            label = labels[int(source_id)]
            if label["critic_split"] != (0 if split_name == "train" else 1):
                raise ValueError("Backend label crossed its declared split")
            if label["action_sha256"] != action_sha256(split["action"][i]):
                raise ValueError("Measured action differs from the executed layout")
            if float(label["rWL_um"]) != float(split["backend_rwl_um"][i]):
                raise ValueError("Reward measurement changed")
            artifacts = {}
            for key in ("def", "metrics", "manifest"):
                artifact = Path(label[key + "_path"])
                digest = sha(artifact)
                if digest != label[key + "_sha256"]:
                    raise ValueError("Measured backend artifact changed: " + str(artifact))
                artifacts[str(artifact)] = digest
            bound.append(dict(source_row=int(source_id), split=split_name,
                              action_sha256=label["action_sha256"], artifacts=artifacts))
    binding = dict(dataset=str(path), dataset_sha256=sha(path),
                   manifest_sha256=sha(manifest_path), labels_sha256=sha(labels_path),
                   reward=manifest["reward"], reward_stage=manifest["reward_stage"],
                   reward_scale_train_only=manifest["reward_scale_train_only"],
                   selected_backend_labels=bound, confirmation_labels_used=False)
    return data, binding


def build_trajectories(problem, layouts, rewards, source_rows, groups, *, order=None):
    """Replay every measured layout exactly, retaining one terminal label per row."""
    device = problem.x.device
    layouts = torch.as_tensor(layouts, dtype=torch.float32, device=device).reshape(-1, len(problem.x), 2)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    source_rows, groups = np.asarray(source_rows), np.asarray(groups)
    if rewards.shape != (len(layouts),) or len(source_rows) != len(layouts) or len(groups) != len(layouts):
        raise ValueError("Layout identities and reward rows differ")
    if not len(layouts) or not bool(torch.isfinite(layouts).all() and torch.isfinite(rewards).all()):
        raise ValueError("Finite nonempty measured layouts required")
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError("Each backend source row must occur exactly once")
    env = SequentialPlacementEnv(problem, order=order, padding=0.)
    if not env.order:
        raise ValueError("At least one movable macro is required")
    fields = {key: [] for key in ("positions", "placed", "current_id", "action", "reward",
              "next_positions", "next_placed", "next_id", "done", "mc_return",
              "trajectory_id", "source_row", "group_id", "step_index")}
    for layout_id, (layout, terminal_reward) in enumerate(zip(layouts, rewards)):
        if not geometry_audit(layout, env.sizes)["is_legal"]:
            raise ValueError("Cannot replay an illegal measured layout without changing its label")
        if not torch.equal(layout[env.base_fixed], env.fixed_positions[env.base_fixed]):
            raise ValueError("Measured layout moved an original fixed node")
        state = env.reset()
        for step, macro in enumerate(env.order):
            if int(state["current_id"][0]) != macro:
                raise RuntimeError("Environment macro order changed")
            action = layout[macro].clone()
            after, _, done, info = env.step(action)
            if info["failure_reason"] is not None or not torch.equal(info["executed_action"], action):
                raise ValueError("Measured action is not an identity under the sequential executor")
            values = dict(positions=state["partial_positions"][0], placed=state["placed"][0],
                          current_id=state["current_id"][0], action=action,
                          reward=terminal_reward if done else terminal_reward.new_zeros(()),
                          next_positions=after["partial_positions"][0], next_placed=after["placed"][0],
                          next_id=after["current_id"][0], done=torch.tensor(done, device=device),
                          mc_return=terminal_reward, trajectory_id=torch.tensor(layout_id, device=device),
                          source_row=torch.tensor(int(source_rows[layout_id]), device=device),
                          group_id=torch.tensor(int(groups[layout_id]), device=device),
                          step_index=torch.tensor(step, device=device))
            for key, value in values.items():
                fields[key].append(value.detach().cpu().clone())
            state = after
        if not torch.equal(env.positions, layout):
            raise ValueError("Sequential replay changed the final measured layout")
    arrays = {key: torch.stack(value) for key, value in fields.items()}
    metadata = dict(order=list(env.order), backend_labels=len(layouts),
                    transitions=len(arrays["reward"]), noise_families=len(np.unique(groups)),
                    action_semantics="raw macro action; recorded legal actions are projection identities",
                    terminal_reward="one measured backend reward per complete layout",
                    mc_continuation="replay_remaining_coordinates_of_same_measured_layout",
                    independent_backend_measurements=len(layouts))
    return arrays, metadata


def prepare_data(problem, source):
    train_groups, valid_groups = set(source["train"]["group_id"]), set(source["validation"]["group_id"])
    if train_groups & valid_groups:
        raise ValueError("Noise family crosses train/validation")
    data, metadata = {}, {}
    for name in ("train", "validation"):
        rows = source[name]
        data[name], metadata[name] = build_trajectories(problem, rows["action"], rows["reward"],
                                                       rows["source_row_index"], rows["group_id"])
    return data, metadata


def sample_batch(data, count, device):
    indices = torch.randint(len(data["reward"]), (count,))
    return {key: value[indices].to(device) for key, value in data.items()}


def sample_noise(batch, device):
    return 2 * torch.rand(batch["positions"].shape, device=device) - 1


def optimizer_step(loss, optimizer, module, clip):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def gradient_norm(loss, module):
    values = torch.autograd.grad(loss, [p for p in module.parameters() if p.requires_grad],
                                 retain_graph=True, allow_unused=True)
    return float(torch.stack([g.detach().float().square().sum() for g in values if g is not None]).sum().sqrt())


def training_step(actor, critic, target, teacher, optimizers, batch, cfg, *, stage,
                  q_weight=1., log_components=False):
    device = batch["positions"].device
    record = {}
    if stage in ("mc", "bellman"):
        target_value = batch["mc_return"] if stage == "mc" else bellman_targets(
            actor, target, batch, sample_noise(batch, device), cfg["gamma"], cfg["target_reduction"])
        prediction = critic(batch["positions"], batch["placed"], batch["current_id"], batch["action"])
        q_loss = (prediction-target_value[:, None]).square().mean()
        record.update(critic_loss=float(q_loss.detach()), target_mean=float(target_value.mean()),
                      critic_grad_norm=optimizer_step(q_loss, optimizers["critic"], critic, cfg["grad_clip"]))
        polyak_update(critic, target, 1. if stage == "mc" else cfg["target_tau"])
    if stage in ("bc", "bellman"):
        noise = sample_noise(batch, device)
        weight = 0. if stage == "bc" else q_weight
        loss, distill, q, actor_info = actor_objective(actor, teacher, critic, batch, noise,
             cfg["alpha"], weight, cfg["actor_q_reduction"], return_info=True)
        record.update(actor_info)
        if log_components:
            record["distill_gradient_norm"] = gradient_norm(cfg["alpha"] * distill, actor)
            record["q_gradient_norm_unweighted"] = gradient_norm(-q, actor)
        record.update(actor_loss=float(loss.detach()), distill_mse=float(distill.detach()),
                      predicted_q=float(q.detach()), q_weight=weight,
                      actor_grad_norm=optimizer_step(loss, optimizers["actor"], actor, cfg["grad_clip"]))
    if not record:
        raise ValueError("Unknown training stage")
    return record


def checkpoint_state(actor, critic, target, optimizers, binding, *, stage, step, arm):
    return dict(schema=SCHEMA, actor=actor.state_dict(), critic=critic.state_dict(),
                target=target.state_dict(), optimizers={k: opt.state_dict() for k, opt in optimizers.items()},
                rng=rng_state(), binding=binding, stage=stage, step=step, arm=arm)


def validate_checkpoint_identity(saved, binding, *, arm=None, stage=None, step=None):
    if saved["schema"] != SCHEMA or saved["binding"] != binding:
        raise ValueError("Checkpoint/config/data/source identity changed")
    if arm is not None and saved["arm"] != arm:
        raise ValueError("Wrong arm checkpoint")
    if stage is not None and saved["stage"] != stage:
        raise ValueError("Wrong checkpoint stage")
    if step is not None and saved["step"] != step:
        raise ValueError("Wrong checkpoint step")


def restore_checkpoint(saved, actor, critic, target, optimizers, binding):
    validate_checkpoint_identity(saved, binding)
    actor.load_state_dict(saved["actor"])
    critic.load_state_dict(saved["critic"])
    target.load_state_dict(saved["target"])
    for key, optimizer in optimizers.items():
        # Adam may retain same-device momentum tensors from load_state_dict.
        # Without a copy, updating one restored arm mutates the saved common
        # optimizer state and contaminates every subsequent matched fork.
        optimizer.load_state_dict(copy.deepcopy(saved["optimizers"][key]))
    restore_rng(saved["rng"])


def read_completed_arm(directory, binding, arm):
    """Skip only a complete artifact belonging to this exact seed/config/arm."""
    directory = Path(directory)
    report = json.loads((directory / "completed.json").read_text())
    expected = dict(schema=SCHEMA, status="completed", arm=arm, seed=binding["seed"],
                    binding_sha256=binding_digest(binding))
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed report identity changed")
    checkpoint = directory / "checkpoint.pt"
    if report.get("checkpoint_sha256") != sha(checkpoint):
        raise ValueError("Completed checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(saved, binding, arm=arm, stage="completed",
                                 step=binding["config"]["train_steps"])
    return report


def construct(problem, train_returns, cfg, device, model_loader=None):
    actor = SequentialActor(problem, device=device, model_loader=model_loader)
    critic = SequentialCritics(problem, train_returns, device=device,
                               width=cfg["critic_width"], model_loader=model_loader)
    target = frozen_target(critic)
    actor.train()
    critic.train()
    optimizers = dict(actor=torch.optim.Adam(actor.parameters(), lr=cfg["actor_lr"]),
                      critic=torch.optim.AdamW(critic.parameters(), lr=cfg["critic_lr"],
                                              weight_decay=cfg["critic_weight_decay"]))
    return actor, critic, target, optimizers


@torch.no_grad()
def recorded_return_diagnostics(critic, data, device):
    """MC diagnostics against recorded continuation, not unbiased student values."""
    old_mode = critic.training
    critic.eval()
    predictions = []
    try:
        for start in range(0, len(data["reward"]), 32):
            batch = {key: value[start:start+32].to(device) for key, value in data.items()}
            predictions.append(critic(batch["positions"], batch["placed"], batch["current_id"],
                                       batch["action"]).cpu())
    finally:
        critic.train(old_mode)
    prediction = torch.cat(predictions)
    delta = prediction.mean(-1)-data["mc_return"]
    return dict(recorded_continuation_mae=float(delta.abs().mean()),
                recorded_continuation_rmse=float(delta.square().mean().sqrt()),
                twin_disagreement=float((prediction[:, 0]-prediction[:, 1]).abs().mean()),
                transitions=len(prediction), independent_layouts=len(data["trajectory_id"].unique()),
                interpretation="recorded-layout MC diagnostic; not student-rollout ground truth")


def train(config_path, output, seed, device, *, stop_after=None):
    from baseline.api import CHECKPOINT, RUNTIME, load_problem
    from .sequential_teacher import SequentialFlowPlaceTeacher
    cfg = normalize_config(json.loads(Path(config_path).read_text()))
    folder = Path(output).resolve() / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    problem = load_problem(cfg["design"], device)
    source, data_binding = load_measured_layouts(cfg["dataset"], cfg["design"])
    data, data_metadata = prepare_data(problem, source)
    binding = dict(config=cfg, seed=seed, data=data_binding,
                   public_checkpoint_sha256=sha(CHECKPOINT),
                   graph_sha256=sha(RUNTIME / "benchmark_cache" / (cfg["design"] + ".pt")),
                   source_sha256=source_fingerprints())
    binding_path = folder / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Experiment identity changed; use a new output directory")
    atomic_json(binding_path, binding)
    atomic_json(folder / "trajectory_manifest.json", data_metadata)
    for name, arrays in data.items():
        path = folder / (name + "_trajectories.npz")
        if not path.exists():
            with path.open("wb") as stream:
                np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()})
    teacher = SequentialFlowPlaceTeacher(problem, device)
    actor, critic, target, opts = construct(problem, source["train"]["reward"], cfg, device)
    stop = {"requested": False}
    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda *_: stop.update(requested=True))

    try:
        return run_training_stages(actor, critic, target, opts, teacher, data,
                                   binding, folder, device, stop_after=stop_after, stop=stop)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def read_common_bc(checkpoint, binding, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_identity(saved, binding, arm="common", stage="common_bc_complete",
                                 step=binding["config"]["bc_steps"])
    return saved


def common_bc_report(checkpoint, binding):
    """Describe this stage's checkpoint, never completion of the full experiment."""
    return dict(schema=SCHEMA, status="stage_completed", stage="common_bc",
                checkpoint_stage="common_bc_complete", arm="common", seed=binding["seed"],
                checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha(checkpoint),
                binding_sha256=binding_digest(binding), bc_updates=binding["config"]["bc_steps"],
                critic_mc_updates=0, bellman_updates=0, actor_q_weight=0.,
                full_training_completed=False, teacher_frozen=True,
                interpretation="Pure BC checkpoint before MC and before the matched BC/FQL fork")


def run_training_stages(actor, critic, target, opts, teacher, data, binding, folder,
                        device, *, stop_after=None, stop=None):
    """Run/resume a fixed configuration; stop_after changes scheduling only.

    common_bc.pt is an immutable boundary with full optimizer and RNG state.
    Diagnostic exports can use it while later invocation of this same function
    resumes MC exactly. No shorter config or artificial completed arm is needed.
    """
    cfg, seed, folder = binding["config"], binding["seed"], Path(folder)
    if stop_after not in (None, "common_bc"):
        raise ValueError("Unsupported stage boundary")
    if stop_after == "common_bc" and (cfg["bc_steps"] < 1 or cfg["alpha"] <= 0):
        raise ValueError("A pure BC gate requires positive bc_steps and alpha")
    folder.mkdir(parents=True, exist_ok=True)
    stop = {"requested": False} if stop is None else stop

    def run_stage(stage, total, begin, arm, destination, weight):
        started = time.perf_counter()
        for i in range(begin, total):
            batch = sample_batch(data["train"], cfg["batch_size"], device)
            log_now = i == begin or (i+1) % cfg["log_every"] == 0 or i+1 == total
            record = training_step(actor, critic, target, teacher, opts, batch, cfg,
                                   stage=stage, q_weight=weight, log_components=log_now)
            if log_now:
                record.update(stage=stage, arm=arm, step=i+1, elapsed_seconds=time.perf_counter()-started)
                with (destination.parent / "training.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
            if (i+1) % cfg["checkpoint_every"] == 0 or i+1 == total or stop["requested"]:
                atomic_torch(destination, checkpoint_state(actor, critic, target, opts, binding,
                                                           stage=stage, step=i+1, arm=arm))
            if stop["requested"]:
                raise InterruptedError("Slurm signal received; resumable checkpoint saved")
        atomic_torch(destination, checkpoint_state(actor, critic, target, opts, binding,
                                                   stage=stage, step=total, arm=arm))

    common_path, resume_path = folder / "common.pt", folder / "common_resume.pt"
    bc_path, bc_report_path = folder / "common_bc.pt", folder / "common_bc.json"
    saved = None
    if common_path.exists():
        common = torch.load(common_path, map_location="cpu", weights_only=False)
        validate_checkpoint_identity(common, binding, arm="common", stage="common_complete", step=0)
        del common
        if not bc_path.exists():
            raise ValueError("Missing immutable BC boundary for completed common training")
    elif resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        validate_checkpoint_identity(saved, binding, arm="common")
        if saved["stage"] not in ("bc", "mc") or not 0 <= saved["step"] <= (
                cfg["bc_steps"] if saved["stage"] == "bc" else cfg["critic_mc_steps"]):
            raise ValueError("Invalid common resume stage/step")
        if saved["stage"] == "mc" and not bc_path.exists():
            raise ValueError("MC resume is missing its immutable BC boundary")
        if bc_path.exists() and saved["stage"] == "bc" and saved["step"] != cfg["bc_steps"]:
            raise ValueError("BC resume predates the published BC boundary")

    if not bc_path.exists():
        if bc_report_path.exists():
            raise ValueError("BC marker exists but its checkpoint is missing")
        if saved:
            restore_checkpoint(saved, actor, critic, target, opts, binding)
        begin = saved["step"] if saved else 0
        run_stage("bc", cfg["bc_steps"], begin, "common", resume_path, 0.)
        atomic_torch(bc_path, checkpoint_state(actor, critic, target, opts, binding,
                                             stage="common_bc_complete", step=cfg["bc_steps"], arm="common"))
    # Re-load even a just-written boundary: identical path for staged and
    # uninterrupted runs, and serialized moments cannot alias live optimizers.
    bc_saved = read_common_bc(bc_path, binding)
    bc_report = common_bc_report(bc_path, binding)
    if bc_report_path.exists():
        if json.loads(bc_report_path.read_text()) != bc_report:
            raise ValueError("Pure BC checkpoint/marker identity changed")
    else:
        atomic_json(bc_report_path, bc_report)
    if stop_after == "common_bc":
        return dict(status="stopped_after_common_bc", common_bc=bc_report,
                    requested_full_config_unchanged=True,
                    interpretation="Stage checkpoint only; no claim that MC or either arm is complete")

    if not common_path.exists():
        if saved and saved["stage"] == "mc":
            restore_checkpoint(saved, actor, critic, target, opts, binding)
            begin = saved["step"]
        else:
            restore_checkpoint(bc_saved, actor, critic, target, opts, binding)
            begin = 0
        run_stage("mc", cfg["critic_mc_steps"], begin, "common", resume_path, 0.)
        atomic_torch(common_path, checkpoint_state(actor, critic, target, opts, binding,
                                                  stage="common_complete", step=0, arm="common"))
    summaries = {}
    for arm in cfg["arms"]:
        destination = folder / arm
        destination.mkdir(exist_ok=True)
        completed = destination / "completed.json"
        if completed.exists():
            summaries[arm] = read_completed_arm(destination, binding, arm)
            continue
        checkpoint = destination / "resume.pt"
        # Read independently for every arm, even though restore also copies
        # optimizer tensors.  The serialized common fork stays immutable.
        saved = torch.load(checkpoint if checkpoint.exists() else common_path,
                           map_location=device, weights_only=False)
        if checkpoint.exists():
            validate_checkpoint_identity(saved, binding, arm=arm, stage="bellman")
            if not 0 <= saved["step"] <= cfg["train_steps"]:
                raise ValueError("Invalid arm resume step")
        else:
            validate_checkpoint_identity(saved, binding, arm="common", stage="common_complete", step=0)
        restore_checkpoint(saved, actor, critic, target, opts, binding)
        if not checkpoint.exists():
            # Same RNG stream, starting state and optimizer state for both arms.
            seed_all(100000 + seed)
        begin = saved["step"] if checkpoint.exists() else 0
        run_stage("bellman", cfg["train_steps"], begin, arm, checkpoint,
                  cfg["q_weight"] if arm == "fql" else 0.)
        final = destination / "checkpoint.pt"
        atomic_torch(final, checkpoint_state(actor, critic, target, opts, binding,
                                             stage="completed", step=cfg["train_steps"], arm=arm))
        report = dict(schema=SCHEMA, status="completed", arm=arm, seed=seed,
                      binding_sha256=binding_digest(binding), checkpoint_sha256=sha(final),
                      train=recorded_return_diagnostics(critic, data["train"], device),
                      validation=recorded_return_diagnostics(critic, data["validation"], device),
                      backend_evaluation_pending=True, teacher_frozen=True,
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        atomic_json(completed, report)
        summaries[arm] = report
    atomic_json(folder / "training_completed.json", dict(schema=SCHEMA, status="completed", arms=summaries,
                seed=seed, binding_sha256=binding_digest(binding),
                teacher_variant="public20_conditional_prefix_v1", backend_evaluation_pending=True))
    return summaries


@torch.no_grad()
def rollout(actor, problem, noise_seed, *, order=None):
    env = SequentialPlacementEnv(problem, order=order, padding=0.)
    state = env.reset()
    generator = torch.Generator(device=problem.x.device).manual_seed(noise_seed)
    raw_layout = env.positions.clone()
    noises, records = [], []
    while not env.done:
        noise = 2 * torch.rand((1, len(problem.x), 2), device=problem.x.device, generator=generator) - 1
        action = actor(state["partial_positions"], state["placed"], state["current_id"], noise)[0]
        current = int(state["current_id"][0])
        raw_layout[current] = action
        after, _, done, info = env.step(action)
        records.append(dict(current_id=current, raw_action=action.cpu().tolist(),
                            executed_action=None if info["executed_action"] is None else info["executed_action"].cpu().tolist(),
                            projection_distance=info["projection_distance"], failure_reason=info["failure_reason"],
                            done=done, completed=info["completed"],
                            prefix_geometry=geometry_audit(env.positions, env.sizes, env.placed)))
        noises.append(noise[0].cpu())
        state = after
    return dict(centers=env.positions.clone(), raw_layout=raw_layout,
                noises=torch.stack(noises), records=records, order=list(env.order),
                completed=env.failure_reason is None,
                failure_reason=env.failure_reason)


def validate_export_checkpoint(saved, role):
    binding, cfg = saved["binding"], saved["binding"]["config"]
    if role == "bc_gate":
        validate_checkpoint_identity(saved, binding, arm="common", stage="common_bc_complete",
                                     step=cfg["bc_steps"])
        if cfg["bc_steps"] < 1 or cfg["alpha"] <= 0:
            raise ValueError("A pure BC export requires positive BC training")
    elif role in ("evaluation", "collect"):
        validate_checkpoint_identity(saved, binding, stage="completed", step=cfg["train_steps"])
        if saved["arm"] not in cfg["arms"]:
            raise ValueError("Invalid completed export arm")
    else:
        raise ValueError("Unsupported export role")
    if not isinstance(binding["seed"], int):
        raise ValueError("Invalid export seed identity")


def prefix_quality_summary(requests, macro_count):
    """Geometry/coverage only; neither a teacher completion test nor PPA quality."""
    if not requests or macro_count < 1:
        raise ValueError("Nonempty rollout cohort and positive macro count required")
    count = len(requests)
    completed = [r for r in requests if r["rollout_completed"]]
    decisions = [d for r in requests for d in r["decisions"]]
    projection = [float(d["projection_distance"]) for d in decisions
                  if d["executed_action"] is not None]
    stages = []
    for step in range(macro_count):
        reached = [r["decisions"][step] for r in requests if len(r["decisions"]) > step]
        executed = [d for d in reached if d["executed_action"] is not None]
        stages.append(dict(prefix_length=step+1, attempted=len(reached), executed=len(executed),
                           geometrically_legal=sum(d["prefix_geometry"]["is_legal"] for d in executed),
                           survived_fraction=len(executed)/count))
    failures = {}
    for row in requests:
        if row["sequential_failure_reason"] is not None:
            key = row["sequential_failure_reason"]
            failures[key] = failures.get(key, 0)+1
    vectors = [np.asarray(r["executed_action"], dtype=np.float64).reshape(-1) for r in completed]
    distances = [float(np.sqrt(np.mean((left-right)**2))) for i, left in enumerate(vectors)
                 for right in vectors[i+1:]]
    return dict(rollouts=count, completed=len(completed), completion_fraction=len(completed)/count,
                normalized_geometry_legal=sum(bool(r["geometry"]["legal"]) for r in requests),
                failures=failures, prefixes=stages,
                projected_actions=sum(value > 0 for value in projection), executed_actions=len(projection),
                mean_projection_distance=float(np.mean(projection)) if projection else None,
                max_projection_distance=max(projection) if projection else None,
                unique_complete_layouts=len({r["action_sha256"] for r in completed}),
                complete_layout_pairwise_coordinate_rms=float(np.mean(distances)) if distances else None,
                backend_measurements=0, teacher_completion_tests=0,
                interpretation="Student geometry, projection and diversity diagnostic; dead ends retained; no backend quality or conditional-teacher feasibility claim")


def export(checkpoint_path, output, count, noise_seed, device, role="evaluation"):
    from baseline.api import CHECKPOINT, RUNTIME, load_problem, save_sample
    from backend.scripts.export_macro_def import convert
    if count < 1 or role not in ("evaluation", "collect", "bc_gate"):
        raise ValueError("Declare a positive count and valid rollout role")
    checkpoint_path, output = Path(checkpoint_path).resolve(), Path(output).resolve()
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    binding, cfg = saved["binding"], saved["binding"]["config"]
    validate_export_checkpoint(saved, role)
    if sha(CHECKPOINT) != binding["public_checkpoint_sha256"]:
        raise ValueError("Released initialization changed")
    if sha(RUNTIME / "benchmark_cache" / (cfg["design"] + ".pt")) != binding["graph_sha256"]:
        raise ValueError("Export graph differs from training")
    verify_source_binding(binding)
    problem = load_problem(cfg["design"], device)
    actor = SequentialActor(problem, device)
    actor.load_state_dict(saved["actor"])
    actor.eval()
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                    count=count, noise_seed=noise_seed, role=role, schema=SCHEMA,
                    seed=binding["seed"], arm=saved["arm"], binding_sha256=binding_digest(binding),
                    checkpoint_stage=saved["stage"])
    identity_path = output / "export_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Export directory belongs to different inputs")
    atomic_json(identity_path, identity)
    requests = []
    for index in range(count):
        sample = output / str(index)
        request_path = sample / "request.json"
        if request_path.exists():
            request = json.loads(request_path.read_text())
            expected = dict(arm=saved["arm"], seed=binding["seed"], index=index, role=role,
                            graph=cfg["design"], checkpoint_sha256=identity["checkpoint_sha256"],
                            binding_sha256=identity["binding_sha256"], noise_seed=noise_seed+index,
                            checkpoint_stage=saved["stage"])
            if any(request.get(key) != value for key, value in expected.items()):
                raise ValueError("Completed rollout identity changed")
            if "def_path" in request and sha(request["def_path"]) != request["def_sha256"]:
                raise ValueError("Completed rollout DEF changed")
            requests.append(request)
            continue
        sample.mkdir(exist_ok=True)
        start = time.perf_counter()
        result = rollout(actor, problem, noise_seed + index)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        centers, raw = result["centers"], result["raw_layout"]
        geometry = geometry_audit(centers, problem.cond.x)
        geometry["legal"] = result["completed"] and geometry["is_legal"]
        record = dict(arm=saved["arm"], seed=binding["seed"], index=index, role=role,
                      graph=cfg["design"], stage="global_route_only", terminal_mask=0,
                      round=0, predicted_q=None,
                      source_checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                      binding_sha256=identity["binding_sha256"],
                      checkpoint_stage=saved["stage"],
                      sample_dir=str(sample), noise_seed=noise_seed+index,
                      raw_action=raw.cpu().tolist(), executed_action=centers.cpu().tolist(),
                      geometry=geometry, training_feedback_eligible=role == "collect",
                      action_sha256=hashlib.sha256(np.asarray(centers.cpu(), dtype="<f4").reshape(-1).tobytes()).hexdigest(),
                      run_id=f"go2o_seqv2_{saved['arm']}_s{binding['seed']}_{identity['checkpoint_sha256'][:12]}_{role}_{noise_seed+index}",
                      generation_seconds=elapsed, actor_forward_calls=len(result["records"]),
                      fresh_latent_each_decision=True, decisions=result["records"],
                      rollout_completed=result["completed"],
                      sequential_failure_reason=result["failure_reason"])
        np.savez_compressed(sample / "actions.npz", action=centers.cpu().numpy(),
                            raw_action=raw.cpu().numpy(), step_noise=result["noises"].numpy())
        if geometry["legal"] and role != "bc_gate":
            save_sample(sample, centers, result["noises"][0], problem, record)
            def_path = sample / (cfg["design"] + ".def")
            audit = convert(cfg["design"], sample / "macro_physical.json", def_path)
            record.update(def_path=str(def_path), def_sha256=sha(def_path),
                          rounded_geometry=audit["legal_no_halo"])
            if not audit["legal_no_halo"]:
                record["pre_backend_failure"] = "rounded_DEF_geometry"
        elif not geometry["legal"]:
            # The existing backend worker has two physical geometry categories.
            # Preserve the precise sequential cause separately, including the
            # terminal decision, rather than dropping dead ends from the plan.
            record["pre_backend_failure"] = "normalized_geometry"
        atomic_json(request_path, record)
        requests.append(record)
    tasks = output / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(record, allow_nan=False)+"\n" for record in requests))
    quality = prefix_quality_summary(requests, len(SequentialPlacementEnv(problem).order))
    atomic_json(output / "prefix_quality.json", dict(**identity, **quality))
    status = "exported_bc_gate_diagnostics" if role == "bc_gate" else "exported_backend_pending"
    atomic_json(output / "completed.json", dict(**identity, status=status,
                                                prefix_quality_sha256=sha(output / "prefix_quality.json"),
                                                tasks_sha256=sha(tasks)))
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "export"))
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=2900000)
    parser.add_argument("--role", choices=("evaluation", "collect", "bc_gate"), default="evaluation")
    parser.add_argument("--stop-after", choices=("common_bc",),
                        help="Pause at the pure BC checkpoint using the unchanged full configuration")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("All network execution must run in a Slurm allocation")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "train":
        if not args.config:
            parser.error("train requires --config")
        train(args.config, args.output, args.seed, args.device, stop_after=args.stop_after)
    else:
        if not args.checkpoint:
            parser.error("export requires --checkpoint")
        if args.stop_after:
            parser.error("--stop-after applies only to train")
        export(args.checkpoint, args.output, args.count, args.noise_seed, args.device, args.role)


if __name__ == "__main__":
    main()
