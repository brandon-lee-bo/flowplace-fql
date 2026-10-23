"""Controlled whole-layout v1.0/v1.1 experiment, isolated from legacy jobs.

Teachers, students and critics have separate reproducible initialization RNG.
Legacy flow warmup is explicit, then both teachers stay frozen. Public teacher
targets are freshly sampled through the original 20-step sampler, not cached.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

# Import the project package before the upstream adapter prepends its paths.
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import RUNTIME, CHECKPOINT, load_problem, load_model, generate, legalize, save_sample
from backend.scripts.export_macro_def import convert
from .core import critic_diagnostics
from .critic_gate import load_train_validation
from .graph_o2o_models import GraphPolicy, FeatureCritics, actor_objective
from .released_flowplace_teacher import make_behavior_flow
from .flow_policy_config import uses_released_teacher
from .provenance import capture_source_provenance

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ["fql/versions_whole_train.py", "fql/versions_whole_train.slurm",
                "fql/graph_o2o_models.py", "fql/physical_ridge.py", "fql/critic_gate.py",
                "fql/core.py", "fql/released_flowplace_teacher.py", "fql/flow_policy_config.py",
                "fql/provenance.py", "baseline/api.py", "analysis/placement_metrics.py",
                "backend/scripts/export_macro_def.py", "upstream/flowplace/flow_matching/models.py",
                "upstream/flowplace/flow_matching/configs/config_eval.yaml",
                "upstream/flowplace/flow_matching/networks/gnn.py",
                "upstream/flowplace/flow_matching/hard_constraint_legalizer.py"]
ACTIVE_SOURCE_BUNDLE_SHA = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def update(loss, optimizer, module):
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def component_gradient_norm(loss, module):
    gradients = torch.autograd.grad(loss, [p for p in module.parameters() if p.requires_grad],
                                    retain_graph=True, allow_unused=True)
    squares = [g.detach().float().square().sum() for g in gradients if g is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.


def initialize_problem(args):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    problem = load_problem("bp_fe", "cuda")
    data, manifest, _ = load_train_validation(ROOT / "datasets/bp_fe_grt_v1/dataset.npz")
    cache = torch.load(RUNTIME / "benchmark_cache/bp_fe.pt", map_location="cpu", weights_only=False)
    shape = (-1, len(problem.x), 2)
    train = torch.tensor(data["train"]["raw_action"], device="cuda").reshape(shape)
    reward = torch.tensor(data["train"]["reward"], device="cuda")
    valid = torch.tensor(data["validation"]["raw_action"], device="cuda").reshape(shape)
    vreward = data["validation"]["reward"]
    with np.load(ROOT / "datasets/bp_fe_macro_deterministic_v1/dataset.npz", allow_pickle=False) as a:
        # These are unlabeled additional TRAIN actions, never validation/heldout.
        extra = a["teacher_action"][a["split"] == 0].copy().reshape(shape)
    behavior = torch.cat([train, torch.tensor(extra, device="cuda")])
    return problem, data, manifest, cache, train, reward, valid, vreward, behavior


def state_of(flow, actor, critic, optimizers, **metadata):
    return dict(flow=flow.state_dict(), actor=actor.state_dict(), critic=critic.state_dict(),
                optimizers={k: v.state_dict() for k, v in optimizers.items()},
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), **metadata)


def construct(problem, cache, x, y, saved=None, cfg=None):
    cfg = cfg or {}
    # The teacher architecture/construction must not consume the student's or
    # critic's initialization randomness in a matched replacement experiment.
    init_seed = 310000 + int(cfg["training_seed"]) * 1000
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(init_seed + 1)
        flow = make_behavior_flow(problem, cfg)
        torch.manual_seed(init_seed + 2)
        actor = GraphPolicy(problem)
        torch.manual_seed(init_seed + 3)
        critic = FeatureCritics(cache, x, y, actor.movable, width=cfg.get("critic_width", 128)).cuda()
    optimizers = dict(flow=torch.optim.Adam(flow.parameters(), lr=cfg.get("flow_lr", 1e-5)),
                      actor=torch.optim.Adam(actor.parameters(), lr=cfg.get("actor_lr", 1e-5)),
                      critic=torch.optim.AdamW(critic.parameters(), lr=cfg.get("critic_lr", 3e-4), weight_decay=1e-4))
    if saved:
        if saved.get("source_bundle_sha256") != cfg["source_bundle_sha256"]:
            raise ValueError("Training source/data bundle differs from the checkpoint")
        flow.load_state_dict(saved["flow"])
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        for key, value in optimizers.items():
            value.load_state_dict(saved["optimizers"][key])
        torch.set_rng_state(saved["torch_rng"].cpu())
        # map_location="cuda" also moves serialized ByteTensor RNG states.
        # CUDA's generator setter still requires host ByteTensors.
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    return flow, actor, critic, optimizers


def critic_stats(critic, x, y):
    with torch.no_grad():
        pred = critic(x).cpu().numpy()
    return critic_diagnostics(pred, np.asarray(y))


def parameter_changes(module, before):
    changed, squared, maximum = 0, 0., 0.
    for key, value in module.named_parameters():
        current = value.detach().cpu()
        original = before[key].cpu()
        if not torch.equal(current, original):
            changed += 1
        delta = current.double()-original.double()
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return dict(changed_parameter_tensors=changed, parameter_delta_l2=squared**.5,
                max_absolute_parameter_delta=maximum)


def module_identity(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_point(folder, name, flow, actor, critic, opts, metadata, full=True):
    metadata = dict(metadata, source_bundle_sha256=ACTIVE_SOURCE_BUNDLE_SHA)
    value = state_of(flow, actor, critic, opts, **metadata)
    if not full:
        value.pop("optimizers")
    atomic_torch(folder / name, value)


def train(args):
    global ACTIVE_SOURCE_BUNDLE_SHA
    exp = Path(args.experiment).resolve()
    cfg_path = exp / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("protocol") != "matched_frozen_teachers_v1" or cfg.get("online_flow_update") is not False:
        raise ValueError("This study requires explicitly frozen teachers")
    cfg["training_seed"] = args.seed
    if cfg["design"] != "bp_fe":
        raise ValueError("Pilot graph/context differs")
    problem, data, manifest, cache, x, y, vx, vy, behavior = initialize_problem(args)
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    for name, expected in cfg.get("source_sha256", {}).items():
        if sources.get(name) != expected:
            raise ValueError("Declared source changed: " + name)
    dependencies = dict(sources=sources, dataset_sha256=manifest["dataset_sha256"],
        behavior_extra_sha256=sha(ROOT / "datasets/bp_fe_macro_deterministic_v1/dataset.npz"),
        graph_cache_sha256=sha(RUNTIME / "benchmark_cache/bp_fe.pt"), pretrained_sha256=sha(CHECKPOINT))
    ACTIVE_SOURCE_BUNDLE_SHA = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
    cfg["source_bundle_sha256"] = ACTIVE_SOURCE_BUNDLE_SHA
    seed_root = exp / f"seed_{args.seed}"
    phase = f"round_{args.round}"
    folder = seed_root / phase
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "training_status.json"
    status = dict(status="training", seed=args.seed, round=args.round,
                  slurm_job_id=os.environ["SLURM_JOB_ID"], config_sha256=sha(cfg_path),
                  graph_cache_sha256=sha(RUNTIME / "benchmark_cache/bp_fe.pt"),
                  pretrained_sha256=sha(CHECKPOINT),
                  offline_dataset_sha256=manifest["dataset_sha256"],
                  reward_scale=manifest["reward_scale_train_only"],
                  reward="-measured global-route rWL / fixed offline TRAIN mean",
                  context="fixed netlist", terminal_mask=0,
                  prior="uniform[-1,1]", no_heldout_labels_used=True)
    status.update(dependencies=dependencies, source_bundle_sha256=ACTIVE_SOURCE_BUNDLE_SHA)
    source_dir = folder / ("source_snapshot_job_" + os.environ["SLURM_JOB_ID"])
    if not source_dir.exists():
        status["source"] = capture_source_provenance(source_dir, extra_paths={
            name: ROOT / name for name in SOURCE_FILES})
    atomic_json(status_path, status)
    resume_common = folder / "common_resume.pt"
    if args.round == 0 and not (folder / "common.pt").exists():
        saved = torch.load(resume_common, map_location="cuda", weights_only=False) if resume_common.exists() else None
        flow, actor, critic, opts = construct(problem, cache, x, y, saved, cfg)
        if saved and saved.get("config_sha256") != sha(cfg_path):
            raise ValueError("Common-stage configuration changed")
        if not saved:
            atomic_json(folder / "initialization.json", dict(
                training_seed=args.seed, actor_sha256=module_identity(actor),
                critic_sha256=module_identity(critic), teacher_sha256=module_identity(flow),
                rng_rule="separate fixed seeds per module; actor/Q identical across teacher arms"))
        started = time.perf_counter()
        warm_total = cfg["flow_warmup_steps"] + cfg["actor_warmup_steps"]
        begin = (warm_total if saved and saved.get("phase") == "critic"
                 else saved.get("iteration", 0) if saved else 0)
        if begin >= cfg["flow_warmup_steps"]:
            flow.requires_grad_(False)
            flow.eval()
        for i in range(begin, warm_total):
            if i == cfg["flow_warmup_steps"]:
                torch.manual_seed(410000 + args.seed)
                flow.requires_grad_(False)
                flow.eval()
            z = 2 * torch.rand(cfg["batch_size"], len(problem.x), 2, device="cuda") - 1
            t = torch.rand(len(z), device="cuda")
            if i < cfg["flow_warmup_steps"]:
                idx = torch.randint(len(behavior), (len(z),), device="cuda")
                loss = flow.flow_loss(behavior[idx], z, t)
                update(loss, opts["flow"], flow)
            else:
                loss, _, _ = actor_objective(actor, flow, critic, z,
                    alpha=1., q_weight=0., flow_steps=cfg["flow_steps"])
                update(loss, opts["actor"], actor)
            if (i + 1) % 50 == 0 or i + 1 == warm_total:
                print(json.dumps(dict(stage="common_warmup", step=i+1, loss=float(loss),
                                      seconds=time.perf_counter()-started)), flush=True)
            if (i + 1) % 50 == 0 or i + 1 == warm_total:
                save_point(folder, "common_resume.pt", flow, actor, critic, opts,
                           dict(iteration=i+1, phase="warmup", config_sha256=sha(cfg_path)))
        # Terminal Q regression before the policy is allowed to optimize it.
        q_begin = saved.get("critic_iteration", 0) if saved and saved.get("phase") == "critic" else 0
        if q_begin == 0:
            torch.manual_seed(420000 + args.seed)
        for i in range(q_begin, cfg["critic_warmup_steps"]):
            idx = torch.randint(len(x), (64,), device="cuda")
            loss = (critic(x[idx]) - y[idx, None]).square().mean()
            update(loss, opts["critic"], critic)
            if (i + 1) % 250 == 0 or i + 1 == cfg["critic_warmup_steps"]:
                save_point(folder, "common_resume.pt", flow, actor, critic, opts,
                           dict(iteration=warm_total, phase="critic", critic_iteration=i+1,
                                config_sha256=sha(cfg_path)))
        flow.requires_grad_(False)
        flow.eval()
        save_point(folder, "common.pt", flow, actor, critic, opts,
                   dict(iteration=0, common_warmup_complete=True, config_sha256=sha(cfg_path)))
        atomic_json(folder / "common_critic.json", dict(
            train=critic_stats(critic, x, y.cpu().numpy()), validation=critic_stats(critic, vx, vy),
            critic_sha256=module_identity(critic), teacher_sha256=module_identity(flow)))
        del flow, actor, critic, opts
        torch.cuda.empty_cache()
    status["arms"] = {}
    for arm in ["bc", "fql"]:
        dest = folder / arm
        dest.mkdir(exist_ok=True)
        if (dest / "completed.json").exists():
            complete = json.loads((dest / "completed.json").read_text())
            checkpoint = torch.load(dest / "checkpoint.pt", map_location="cpu", weights_only=False)
            if (complete["checkpoint_sha256"] != sha(dest / "checkpoint.pt")
                    or checkpoint.get("config_sha256") != sha(cfg_path)
                    or checkpoint.get("source_bundle_sha256") != ACTIVE_SOURCE_BUNDLE_SHA
                    or any(checkpoint.get(k) != v for k, v in dict(arm=arm, seed=args.seed, round=args.round).items())):
                raise ValueError("Completed branch identity changed")
            del checkpoint
            status["arms"][arm] = "completed_previously"
            continue
        previous = folder / "common.pt" if args.round == 0 else seed_root / f"round_{args.round-1}" / arm / "checkpoint.pt"
        replay = json.loads((seed_root / f"replay_round_{args.round}.json").read_text()) if args.round else None
        resume = dest / "resume.pt"
        saved = torch.load(resume if resume.exists() else previous, map_location="cuda", weights_only=False)
        if saved.get("config_sha256") != sha(cfg_path):
            raise ValueError("Training configuration changed since the checkpoint")
        if resume.exists() and replay and saved.get("replay_sha256") != sha(seed_root/f"replay_round_{args.round}.json"):
            raise ValueError("Online replay changed during a resumed round")
        flow, actor, critic, opts = construct(problem, cache, x, y, saved, cfg)
        flow.requires_grad_(False)
        flow.eval()
        xx, yy, bb = x, y, behavior
        new_count = 0
        if replay:
            if (replay["for_training_round"] != args.round or replay["config_sha256"] != sha(cfg_path)
                or replay["offline_dataset_sha256"] != manifest["dataset_sha256"]
                or replay["reward_scale"] != manifest["reward_scale_train_only"]):
                raise ValueError("Replay provenance/configuration differs")
            rows = replay["arms"][arm]
            if not rows:
                raise ValueError("Online update without newly measured training data")
            for row in rows:
                if row["terminal_mask"] != 0 or sha(row["outcome_path"]) != row["outcome_sha256"]:
                    raise ValueError("Replay measurement changed or is not terminal")
            new_actions = torch.tensor([r["raw_action"] for r in rows], device="cuda").reshape(-1, len(problem.x), 2)
            new_reward = torch.tensor([r["reward"] for r in rows], device="cuda", dtype=torch.float32)
            xx = torch.cat([x, new_actions]); yy = torch.cat([y, new_reward])
            # Only successful actions are behavior imitation targets. Failures still train Q.
            good = torch.tensor([r["status"] == "success" for r in rows], device="cuda")
            bb = torch.cat([behavior, new_actions[good]])
            new_count = len(new_actions)
        begin = saved.get("iteration", 0) if resume.exists() else 0
        total = cfg["offline_steps"] if args.round == 0 else cfg["online_update_steps"]
        # Arm-local RNG is identical for the matched controls before online replay diverges.
        if not resume.exists():
            torch.manual_seed(100000 + args.seed * 1000 + args.round)
        start = time.perf_counter()
        with (dest / "training.jsonl").open("a") as log:
            for i in range(begin, total):
                n = cfg["batch_size"]
                index = torch.randint(len(x), (n,), device="cuda")
                if new_count:
                    online_n = round(n * cfg["online_batch_fraction"])
                    index[:online_n] = len(x) + torch.randint(new_count, (online_n,), device="cuda")
                bindex = torch.randint(len(behavior), (n,), device="cuda")
                if new_count and len(bb) > len(behavior):
                    # Use the optional balanced_sampling behavior implemented
                    # in official main.py, not the paper's unbalanced default.
                    # Extra unlabeled teacher targets remain in the offline half.
                    bindex[:online_n] = len(behavior) + torch.randint(len(bb)-len(behavior), (online_n,), device="cuda")
                z = 2 * torch.rand(n, len(problem.x), 2, device="cuda") - 1
                zbc = 2 * torch.rand_like(z) - 1
                t = torch.rand(n, device="cuda")
                # Compute all targets before updates, matching FQL's simultaneous losses.
                aloss, distill, aq = actor_objective(actor, flow, critic, z,
                    alpha=cfg["alpha"], q_weight=float(arm == "fql"), flow_steps=cfg["flow_steps"])
                floss = z.new_zeros(())
                qloss = (critic(xx[index]) - yy[index, None]).square().mean()
                component_norms = {}
                if (i+1) % 50 == 0 or i+1 == total:
                    component_norms = dict(distill_gradient_norm=component_gradient_norm(cfg["alpha"]*distill, actor),
                                           q_gradient_norm_unweighted=component_gradient_norm(-aq, actor))
                actor_norm = update(aloss, opts["actor"], actor)
                update(qloss, opts["critic"], critic)
                if (i+1) % 50 == 0 or i+1 == total:
                    row = dict(step=i+1, actor_loss=float(aloss), flow_loss=float(floss),
                               critic_loss=float(qloss), distill=float(distill), q=float(aq),
                               seconds=time.perf_counter()-start, online_training_rows=new_count,
                               actor_gradient_norm=actor_norm, **component_norms)
                    log.write(json.dumps(row)+"\n"); log.flush()
                    print(json.dumps(dict(arm=arm, round=args.round, **row)), flush=True)
                if (i+1) % 50 == 0 or i+1 == total:
                    metadata = dict(iteration=i+1, arm=arm, round=args.round, seed=args.seed,
                                    config_sha256=sha(cfg_path), offline_rows=len(x), online_rows=new_count,
                                    replay_sha256=None if not replay else sha(seed_root/f"replay_round_{args.round}.json"))
                    save_point(dest, "resume.pt", flow, actor, critic, opts, metadata)
                    if i+1 in cfg.get("diagnostic_weight_milestones", []):
                        save_point(dest, f"weights_step_{i+1}.pt", flow, actor, critic, opts, metadata, full=False)
            save_point(dest, "checkpoint.pt", flow, actor, critic, opts,
                       dict(iteration=total, arm=arm, round=args.round, seed=args.seed,
                            config_sha256=sha(cfg_path), replay_sha256=None if not replay else sha(seed_root/f"replay_round_{args.round}.json")))
        parent = torch.load(previous, map_location="cpu", weights_only=False)
        changes = {name: parameter_changes(module, parent[name])
                   for name, module in [("flow", flow), ("actor", actor), ("critic", critic)]}
        del parent
        atomic_json(dest / "completed.json", dict(status="completed", checkpoint_sha256=sha(dest/"checkpoint.pt"),
            arm=arm, seed=args.seed, round=args.round, train_rows=len(xx), online_training_rows=new_count,
            parent_checkpoint=str(previous), parent_checkpoint_sha256=sha(previous), parameter_changes=changes,
            critic_train=critic_stats(critic, xx, yy.cpu().numpy()), critic_validation=critic_stats(critic, vx, vy),
            slurm_job_id=os.environ["SLURM_JOB_ID"], seconds=time.perf_counter()-start))
        status["arms"][arm] = "completed"
        atomic_json(status_path, status)
        del flow, actor, critic, opts
        torch.cuda.empty_cache()
    export(args, cfg, problem, cache, x, y, folder)
    status["status"] = "completed"
    atomic_json(status_path, status)


def export(args, cfg, problem, cache, x, y, folder):
    dest = folder / "requests"
    if (dest / "completed.json").exists():
        done = json.loads((dest / "completed.json").read_text())
        if done["config_sha256"] != sha(Path(args.experiment)/"config.json") or done["tasks_sha256"] != sha(dest/"tasks.jsonl"):
            raise ValueError("Completed export configuration or tasks changed")
        return
    dest.mkdir(exist_ok=True)
    metric = MacroHPWL(problem.benchmark_info)
    records = []
    for arm in ["bc", "fql"] + (["flowplace"] if args.round == 0 else []):
        actor = critic = None
        if arm != "flowplace":
            source = folder / arm / "checkpoint.pt"
            saved = torch.load(source, map_location="cuda", weights_only=False)
            _, actor, critic, _ = construct(problem, cache, x, y, saved, cfg)
        else:
            source = CHECKPOINT
            model, _ = load_model(problem)
        roles = ["evaluation"] if arm == "flowplace" else ["collect", "evaluation"]
        for role in roles:
            count = cfg["online_samples"] if role == "collect" else cfg["evaluation_samples"]
            noise_seed = (700000 + args.round * 1000 + args.seed * 100 if role == "collect" else 900000)
            generator = torch.Generator(device="cuda").manual_seed(noise_seed)
            noises = 2 * torch.rand(count, len(problem.x), 2, generator=generator, device="cuda") - 1
            for i, noise in enumerate(noises):
                sample_dir = dest / arm / role / str(i)
                record_path = sample_dir / "request.json"
                if record_path.exists():
                    prior = json.loads(record_path.read_text())
                    if prior["source_checkpoint"] != str(source) or prior["checkpoint_sha256"] != sha(source):
                        raise ValueError("Interrupted export checkpoint identity changed")
                    records.append(prior)
                    continue
                sample_dir.mkdir(parents=True, exist_ok=True)
                if arm == "flowplace":
                    action, _, _ = generate(model, problem, noise=noise)
                    raw = action.detach()
                    q = None
                else:
                    with torch.no_grad():
                        raw = actor(noise[None])[0]
                        q = critic(raw[None]).cpu().tolist()[0]
                    action = legalize(raw, problem)
                a = action.detach().cpu().numpy()
                geom = legality(a, metric.sizes)
                record = dict(arm=arm, seed=args.seed, round=args.round, role=role, index=i,
                    noise_seed=noise_seed, graph="bp_fe", terminal_mask=0, raw_action=raw.cpu().tolist(),
                    executed_action=a.tolist(), predicted_q=q, source_checkpoint=str(source),
                    checkpoint_sha256=sha(source), geometry=geom,
                    action_sha256=hashlib.sha256(np.asarray(a, dtype="<f4").reshape(-1).tobytes()).hexdigest(),
                    sample_dir=str(sample_dir), run_id=f"{cfg.get('run_id_prefix', 'go2o_v1')}_s{args.seed}_r{args.round}_{arm}_{role}_{i}",
                    stage="global_route_only", training_feedback_eligible=role == "collect",
                    macro_hpwl=float(metric(a)))
                np.savez_compressed(sample_dir/"actions.npz", noise=noise.cpu().numpy(),
                                    raw_action=raw.cpu().numpy(), action=a)
                if geom["legal"]:
                    save_sample(sample_dir, action, noise, problem, record)
                    # Preserve an interrupted export rather than overwrite it.
                    def_path = sample_dir / ("bp_fe_job" + os.environ["SLURM_JOB_ID"] + ".def")
                    audit = convert("bp_fe", sample_dir/"macro_physical.json", def_path)
                    record.update(def_path=str(def_path), def_sha256=sha(def_path), rounded_geometry=audit["legal_no_halo"])
                    if not audit["legal_no_halo"]:
                        record["pre_backend_failure"] = "rounded_DEF_geometry"
                else:
                    record["pre_backend_failure"] = "normalized_geometry"
                atomic_json(record_path, record)
                records.append(record)
        del actor, critic
        if arm == "flowplace":
            del model
        torch.cuda.empty_cache()
    path = dest / "tasks.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in records))
    atomic_json(dest / "completed.json", dict(status="completed", tasks_sha256=sha(path), count=len(records),
        config_sha256=sha(Path(args.experiment)/"config.json"), all_outcomes_required=True,
        selection="All predetermined samples, no Q selection; fixed evaluation noises never enter replay"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Use Slurm GPU")
    train(args)


if __name__ == "__main__":
    main()
