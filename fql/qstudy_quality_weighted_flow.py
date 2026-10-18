"""Paired measured-quality flow-matching control; preparation never trains.

Uniform and quality arms use the SAME measured TRAIN raw layouts, common
per-seed flow checkpoint, uniform row sampling, noise/time streams and budget.
Only a detached per-layout loss weight differs. This is neither full IQL nor
an FQL actor update: there is no critic, action-Q gradient, one-step actor,
sampling/export, legalization or backend call in this module.

The fixed initial protocol uses audited bp_fe384 TRAIN (historical public
FlowPlace20-hard proposals and perturbations), not the separate 256 unmeasured
BC targets or the newer current-flow64 source. All retained layouts keep
strictly positive weight. This preserves support, not a guarantee against
finite-model mode collapse or a claim of improved physical quality.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from .qstudy_data import bind, load_dataset, sha

ROOT = Path(__file__).resolve().parents[1]
VERSION = "measured_quality_weighted_flow_v1"
DATASET_SHA = "0daecf483b64283af46e3aec101161912ebff2984a493d0778fba5b2ebba99e4"
ARMS, SEEDS = ("uniform", "quality_rank"), (0, 1, 2)
SETTINGS = dict(steps=1000, batch_size=16, lr=1e-5, gradient_clip=1.,
                rank_log_range=math.log(4.), checkpoint_every=50,
                stream_seed=8640000, model_mode="eval_with_gradients")
SOURCE_NAMES = ("fql/qstudy_quality_weighted_flow.py", "fql/qstudy_quality_weighted_flow.slurm",
                "fql/qstudy_data.py", "fql/graph_o2o_models.py", "fql/physical_ridge.py",
                "fql/analytic.py", "baseline/api.py",
                "upstream/flowplace/benchmark_utils/openroad_utils.py",
                "docs/qstudy_quality_weighted_flow_protocol.md")
STOP = False
SCOPE = dict(training_split="train", validation_used=False, heldout_used=False,
    historical_online_used=False, behavior256_used=False, critic_updates=0,
    actor_updates=0, layout_generation_calls=0, backend_calls=0,
    checkpoint_selection="fixed final update; no best-of-validation", automatic_launch=False,
    source_population="historical public20-hard proposals and perturbations, not current Euler10 teacher")


def read(path):
    return json.loads(Path(path).read_text())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dataset/checkpoint preparation and training require Slurm")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))


def sources():
    paths = {name: ROOT/name for name in SOURCE_NAMES}
    # The known, small upstream model/config tree closes Hydra and graph imports.
    for path in (ROOT/"upstream/flowplace/flow_matching").rglob("*"):
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json"):
            paths[str(path.relative_to(ROOT))] = path
    return {name: sha(path) for name, path in sorted(paths.items())}


def train_population(data):
    """Select successful measured TRAIN only; never inspect validation labels.

    Exclusion is shared by both arms. Confirmed physical/geometry failures are
    not useful target layouts for this positive-weight imitation control;
    unresolved/missing active labels are also excluded, not invented rewards.
    """
    rows = data["train"]
    raw = np.asarray(rows["raw_action"], dtype=np.float32)
    metrics, mask = np.asarray(rows["metrics"]), np.asarray(rows["metric_mask"])
    statuses = np.asarray(rows["status"])
    n = len(raw)
    if (raw.ndim != 2 or raw.shape[1] % 2 or not np.isfinite(raw).all() or
            metrics.shape != (n, 7) or mask.shape != (n, 7) or mask.dtype != np.bool_ or
            statuses.shape != (n,) or any(np.asarray(rows[k]).shape != (n,)
                for k in ("family", "source_row", "variant", "provenance_index"))):
        raise ValueError("Invalid audited TRAIN shapes/coordinates/identities")
    valid = (statuses == "success") & mask[:, 0] & np.isfinite(metrics[:, 0]) & (metrics[:, 0] > 0)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError("At least two successful measured TRAIN layouts are required")
    identities = [dict(provenance_index=int(rows["provenance_index"][i]),
                       source_row=int(rows["source_row"][i]), family=int(rows["family"][i]),
                       variant=int(rows["variant"][i]), raw_action_sha256=array_sha(raw[i]))
                  for i in indices]
    if len({r["provenance_index"] for r in identities}) != len(identities):
        raise ValueError("TRAIN row identities must be unique")
    wl = np.asarray(metrics[indices, 0], dtype=np.float64)
    reference = float(wl.mean())
    utility = -wl/reference
    return dict(indices=indices, targets=raw[indices].copy(), utility=utility,
                reference_wl_um=reference, row_ids=identities,
                exclusion=dict(total_train=n, retained=len(indices),
                    confirmed_failure=int(np.isin(statuses, ["geometry_failure", "physical_failure"]).sum()),
                    unsuccessful_other=int((~np.isin(statuses, ["success", "geometry_failure", "physical_failure"])).sum()),
                    success_missing_or_invalid_wl=int(((statuses == "success") & ~valid).sum()),
                    retained_families=len({r["family"] for r in identities})))


def quality_weights(utility):
    """Detached, full-TRAIN normalized bounded average-rank weights.

    u=-measured_WL/TRAIN_mean_WL; ties receive their average zero-based rank.
    p=rank/(n-1); w=exp(log(4)*(p-.5))/TRAIN_mean(exp(...)).
    The maximum/minimum weight ratio is at most four; all weights are positive.
    Equal utilities yield exactly uniform weights. No minibatch normalization
    or quality-dependent resampling changes the paired stochastic streams.
    """
    u = np.asarray(utility, dtype=np.float64)
    if u.ndim != 1 or len(u) < 2 or not np.isfinite(u).all():
        raise ValueError("Weights require finite TRAIN utilities")
    _, inverse, counts = np.unique(u, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts)-counts
    ranks = (starts+(counts-1)/2.)[inverse]
    percentiles = ranks/(len(u)-1)
    unnormalized = np.exp(SETTINGS["rank_log_range"]*(percentiles-.5))
    weights = unnormalized/unnormalized.mean()
    info = dict(formula="exp(log(4)*(average_zero_based_rank/(n-1)-.5))/fullTRAIN_mean_exp",
        utility="-measured_rWL_um/retainedTRAIN_mean_rWL_um", split="train",
        rank_ties="average", rank_log_range=SETTINGS["rank_log_range"],
        utility_mean=float(u.mean()), normalization=float(unnormalized.mean()),
        min=float(weights.min()), max=float(weights.max()), mean=float(weights.mean()),
        effective_sample_size=float(weights.sum()**2/np.square(weights).sum()),
        theoretical_min=.25, theoretical_max=4., theoretical_max_to_min_ratio=4.,
        effective_float32_sha256=array_sha(weights.astype(np.float32)))
    return weights.astype(np.float32), info


def population_binding(population):
    weights, info = quality_weights(population["utility"])
    return dict(coordinates="raw normalized centers; fixed nodes restored by GraphPolicy",
        targets_sha256=array_sha(population["targets"]), row_ids=population["row_ids"],
        eligible_train_indices=population["indices"].tolist(),
        row_ids_sha256=object_sha(population["row_ids"]), exclusion=population["exclusion"],
        reference_wl_um=population["reference_wl_um"], weights=weights.tolist(), weighting=info)


def _inputs(base_path, cfg):
    if (cfg.get("design") != "bp_fe" or cfg.get("seeds") != list(SEEDS) or
            cfg.get("online_flow_update") is not False or cfg.get("flow_steps") != 10 or
            cfg.get("q_weight") != 0 or cfg.get("offline_critic_update") is not False or
            cfg.get("dataset_sha256") != DATASET_SHA):
        raise ValueError("Initial control requires bp_fe and the same three frozen common flows")
    dataset = bind(cfg["dataset_path"], cfg["dataset_sha256"])
    cache = bind(ROOT/"baseline/runtime/benchmark_cache/bp_fe.pt", cfg["cache_sha256"])
    public = bind(ROOT/"baseline/runtime/logs/debug/mask.61.train_mask_flowplace_uniform.61/best.ckpt")
    common = {str(s): bind(cfg["common_checkpoints"][str(s)], cfg["common_checkpoint_sha256"][str(s)]) for s in SEEDS}
    if len({c["sha256"] for c in common.values()}) != len(SEEDS):
        raise ValueError("Three seed-specific common checkpoints must be independently bound")
    data, manifest = load_dataset(dataset["path"])
    if (manifest.get("design") != "bp_fe" or manifest.get("stage") != "global_route_only" or
            manifest.get("split_counts") != dict(train=384, validation=16)):
        raise ValueError("Expected the existing audited measured bp_fe384/16 dataset")
    return data, dict(base_config=bind(base_path), dataset=dataset, cache=cache, public_checkpoint=public,
        dataset_manifest=bind(Path(dataset["path"]).with_name("manifest.json")),
        dataset_rows=bind(Path(dataset["path"]).with_name("rows.jsonl")), common=common)


def prepare(base_config, output):
    """Write an immutable six-task proposal only; never starts training."""
    slurm()
    base_path, output = Path(base_config).resolve(), Path(output).resolve()
    cfg = read(base_path)
    data, inputs = _inputs(base_path, cfg)
    pop = population_binding(train_population(data))
    # Protect every input directory, including ancestors: never publish into a
    # source experiment/data/model tree or rename an existing scientific root.
    protected = [ROOT, base_path.parent, *[Path(v["path"]).parent for k,v in inputs.items()
                 if k not in ("base_config", "common")], *[Path(v["path"]).parent for v in inputs["common"].values()]]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("Output must be disjoint from existing source artifacts")
    plan = dict(version=VERSION, output=str(output), inputs=inputs, source_sha256=sources(),
        settings=dict(SETTINGS), population=pop, arms=list(ARMS), seeds=list(SEEDS),
        software=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda),
        tasks=[dict(index=i, arm=a, seed=s) for i,(a,s) in enumerate((a,s) for a in ARMS for s in SEEDS)],
        scope=dict(SCOPE))
    if output.exists() and (not (output/"plan.json").is_file() or read(output/"plan.json") != plan):
        raise ValueError("Output exists without the exact immutable proposal")
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in plan["source_sha256"].items():
        destination = output/"sources"/name
        if destination.exists():
            if sha(destination) != digest:
                raise ValueError("Prepared source snapshot changed: "+name)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = (ROOT/name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("Source changed during proposal snapshot")
            temporary = destination.with_name(destination.name+".tmp."+str(os.getpid()))
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
    if not (output/"plan.json").exists():
        atomic_json(output/"plan.json", plan)
    return plan


def checked_plan(path):
    slurm()
    path = Path(path).resolve()
    plan = read(path)
    if (set(plan) != {"version", "output", "inputs", "source_sha256", "settings", "population",
                     "arms", "seeds", "software", "tasks", "scope"} or
            plan.get("version") != VERSION or plan.get("settings") != SETTINGS or plan.get("scope") != SCOPE or
            plan.get("arms") != list(ARMS) or plan.get("seeds") != list(SEEDS) or
            path != Path(plan.get("output", ""))/"plan.json" or plan.get("source_sha256") != sources()):
        raise ValueError("Changed/noncanonical flow-control plan or source closure")
    cfg = read(plan["inputs"]["base_config"]["path"])
    data, inputs = _inputs(plan["inputs"]["base_config"]["path"], cfg)
    population = train_population(data)
    if (inputs != plan["inputs"] or population_binding(population) != plan["population"] or
            plan["tasks"] != [dict(index=i, arm=a, seed=s) for i,(a,s) in enumerate((a,s) for a in ARMS for s in SEEDS)]):
        raise ValueError("Measured TRAIN data, normalization, tasks or initial checkpoint bindings changed")
    expected = dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda)
    if plan["software"] != expected:
        raise ValueError("Training software differs from prepared contract")
    if any(sha(Path(plan["output"])/"sources"/name) != digest for name,digest in plan["source_sha256"].items()):
        raise ValueError("Prepared source snapshots changed")
    return plan, population


def per_layout_flow_loss(flow, targets, noise, t):
    """Same interpolation/velocity target as GraphPolicy.flow_loss, before mean."""
    if targets.shape != noise.shape or targets.ndim != 3 or targets.shape[-1] != 2 or t.shape != (len(targets),):
        raise ValueError("Flow input shapes differ")
    x0, target = flow.restore(noise), flow.restore(targets)
    xt = (1-t[:, None, None])*x0+t[:, None, None]*target
    return (flow.velocity(xt, t)[:, flow.movable]-(target-x0)[:, flow.movable]).square().mean((1, 2))


def weighted_flow_loss(flow, targets, noise, t, weights):
    errors = per_layout_flow_loss(flow, targets, noise, t)
    weights = weights.detach().to(errors)
    if weights.shape != errors.shape or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("Flow weights must be detached positive finite per-layout values")
    return (errors*weights).mean(), errors.mean()


def draw_batch(targets, batch, generators):
    device = targets.device
    indices = torch.randint(len(targets), (batch,), device=device, generator=generators["rows"])
    noise = 2*torch.rand(batch, targets.shape[1], 2, device=device, generator=generators["noise"])-1
    t = torch.rand(batch, device=device, generator=generators["time"])
    identity = dict(indices=indices.detach().cpu().tolist(), noise_sha256=array_sha(noise.detach().cpu().numpy()),
                    time_sha256=array_sha(t.detach().cpu().numpy()))
    return indices, noise, t, identity


def load_flow_state(flow, state):
    for name in ("fixed", "movable"):
        actual = state.get(name)
        expected = getattr(flow, name)
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.to(expected.device), expected):
            raise ValueError("Initial/resumed flow graph geometry changed: "+name)
    flow.load_state_dict(state, strict=True)


def fit(flow, targets, weights, settings, seed, binding, folder, stop_requested=lambda: False):
    """Deterministic/resumable core, also exercised with a tiny CPU test model."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    flow.eval()  # The existing GraphPolicy flow updates retain upstream eval mode.
    optimizer = torch.optim.Adam(flow.parameters(), lr=settings["lr"])
    generators = {k: torch.Generator(device=targets.device).manual_seed(settings["stream_seed"]+100*seed+i)
                  for i,k in enumerate(("rows", "noise", "time"))}
    resume, final = folder/"resume.pt", folder/"checkpoint.pt"
    curve, begin, chain = [], 0, object_sha(dict(seed=seed, stream_seed=settings["stream_seed"]))
    recovery = resume if resume.exists() else final
    if recovery.exists():
        saved = torch.load(recovery, map_location=targets.device, weights_only=False)
        if saved["binding"] != binding or not 0 <= saved["step"] <= settings["steps"]:
            raise ValueError("Flow resume binding/step changed")
        load_flow_state(flow, saved["flow"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"].cpu())
        if targets.is_cuda:
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
        for k,g in generators.items():
            g.set_state(saved["generators"][k].cpu())
        curve, begin, chain = saved["curve"], saved["step"], saved["draw_chain_sha256"]
    completed_path = folder/"completed.json"
    if completed_path.exists():
        done = read(completed_path)
        if (done.get("binding") != binding or done.get("checkpoint_sha256") != sha(final) or
                done.get("curve_sha256") != sha(folder/"learning_curve.json") or
                done.get("compact_flow_sha256") != sha(folder/"flow.pt") or
                begin != settings["steps"] or done.get("steps") != begin):
            raise ValueError("Completed flow checkpoint/curve/binding changed")
        return done
    started = time.monotonic()
    for step in range(begin, settings["steps"]):
        indices, noise, t, draw = draw_batch(targets, settings["batch_size"], generators)
        chain = object_sha(dict(previous=chain, step=step+1, **draw))
        loss, uniform = weighted_flow_loss(flow, targets[indices], noise, t, weights[indices])
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite flow-matching loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(flow.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        stopped = bool(stop_requested())
        if (step+1) % settings["checkpoint_every"] == 0 or step+1 == settings["steps"] or stopped:
            curve.append(dict(step=step+1, weighted_loss=float(loss.detach()), uniform_loss=float(uniform.detach()),
                gradient_norm=float(grad), batch_weight_mean=float(weights[indices].mean()),
                draw_chain_sha256=chain, last_draw=draw, slurm_job_id=os.environ.get("SLURM_JOB_ID")))
            value = dict(flow=flow.state_dict(), optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if targets.is_cuda else [],
                generators={k:g.get_state() for k,g in generators.items()}, binding=binding,
                step=step+1, curve=curve, draw_chain_sha256=chain)
            atomic_torch(resume, value)
            atomic_json(folder/"learning_curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if stopped:
            raise SystemExit(85)
    # A completed checkpoint without its final marker is recoverable even if
    # the previous process stopped between replace() and marker publication.
    if resume.exists():
        os.replace(resume, final)
    if not final.exists():
        raise ValueError("No completed flow checkpoint")
    atomic_json(folder/"learning_curve.json", curve)
    atomic_torch(folder/"flow.pt", dict(flow=flow.state_dict(), binding=binding, step=settings["steps"]))
    done = dict(status="completed", binding=binding, steps=settings["steps"],
        checkpoint_sha256=sha(final), curve_sha256=sha(folder/"learning_curve.json"),
        compact_flow_sha256=sha(folder/"flow.pt"),
        draw_chain_sha256=chain, flow_updates=settings["steps"], actor_updates=0, critic_updates=0,
        layout_generation_calls=0, backend_calls=0, elapsed_this_attempt_s=time.monotonic()-started)
    atomic_json(completed_path, done)
    return done


def train(plan_path, index):
    slurm()
    if not torch.cuda.is_available():
        raise RuntimeError("Real graph flow training requires a Slurm GPU")
    plan, population = checked_plan(plan_path)
    if isinstance(index, bool) or not 0 <= index < len(plan["tasks"]):
        raise ValueError("Undeclared flow task")
    task = plan["tasks"][index]
    seed, arm = task["seed"], task["arm"]
    folder = Path(plan["output"])/arm/f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Source/data rechecked inside the worker lock, before state restoration.
        plan, population = checked_plan(plan_path)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        from baseline.api import load_problem
        from .graph_o2o_models import GraphPolicy
        problem = load_problem("bp_fe", "cuda")
        flow = GraphPolicy(problem)
        common = plan["inputs"]["common"][str(seed)]
        source = torch.load(common["path"], map_location="cuda", weights_only=False)
        load_flow_state(flow, source["flow"])
        del source
        targets = torch.as_tensor(population["targets"], device="cuda").reshape(-1, len(problem.x), 2)
        w = (np.ones(len(targets), np.float32) if arm == "uniform" else
             np.asarray(plan["population"]["weights"], np.float32))
        weights = torch.as_tensor(w, device="cuda")
        binding = dict(plan=bind(plan_path), task=task, common=common,
            source_sha256=plan["source_sha256"], population_sha256=object_sha(plan["population"]),
            effective_weight_sha256=array_sha(w), initial_state_key="flow",
            fixed_nodes="restored in x0/target; excluded from velocity loss", model_mode=SETTINGS["model_mode"],
            gpu=torch.cuda.get_device_name(), device_count=torch.cuda.device_count())
        if (folder/"binding.json").exists() and read(folder/"binding.json") != binding:
            raise ValueError("Existing task binding differs")
        atomic_json(folder/"binding.json", binding)
        def stop(*_):
            global STOP
            STOP = True
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, stop)
        return fit(flow, targets, weights, SETTINGS, seed, binding, folder, lambda: STOP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.base_config, args.output) if args.command == "prepare" else train(args.plan, args.index)
    print(json.dumps(result if args.command == "train" else dict(status="prepared_only", tasks=len(result["tasks"]),
        plan=str(Path(result["output"])/"plan.json"), backend_calls=0, training_jobs_submitted=0)), flush=True)


if __name__ == "__main__":
    main()
