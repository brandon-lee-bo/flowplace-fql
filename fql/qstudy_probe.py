"""Fresh, matched directional probes of learned Q gradients after legalization.

These are local diagnostics, not new policies, independent training seeds or
replay data. Each common-actor base has one physical baseline, and matched
positive/negative critic-gradient and random perturbations at fixed RMS radii.
All measured labels are kept outside TRAIN and historically inspected validation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import CHECKPOINT, RUNTIME, legalize, load_problem, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_models import GraphPolicy
from .qstudy_backend import checked_existing, checked_request, context
from .qstudy_data import METRIC_NAMES, action_sha, load_dataset
from .qstudy_models import measured_costs
from .qstudy_screen import VARIANTS, make_model

ROOT = Path(__file__).resolve().parents[1]
METRICS = tuple(METRIC_NAMES[:5])
SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def read_json(path):
    return json.loads(Path(path).read_text())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def checked_file(path, expected):
    path = Path(path).resolve()
    if not isinstance(expected, str) or sha(path) != expected:
        raise ValueError("Frozen input changed: " + str(path))
    return dict(path=str(path), sha256=expected)


def settings(experiment):
    cfg, verification, physical = context(experiment)
    # The backend's generic context permits absent expected dataset SHA. Probes
    # require it because the screened model must bind exactly this TRAIN dataset.
    checked_file(cfg["dataset_path"], cfg["dataset_sha256"])
    cache = RUNTIME/"benchmark_cache"/(cfg["design"]+".pt")
    checked_file(cache, cfg["cache_sha256"])
    checked_file(cfg["common_checkpoint"], cfg["common_checkpoint_sha256"])
    if (not isinstance(cfg["base_count"], int) or not 0 < cfg["base_count"] <= 32 or
            not isinstance(cfg["base_noise_seed"], int) or cfg["base_noise_seed"] < 0 or
            not SAFE_NAME.fullmatch(cfg["run_prefix"])):
        raise ValueError("Invalid predeclared base count/noise seed/run prefix")
    radii = cfg["radii"]
    if (not radii or len(set(radii)) != len(radii) or
            any(not math.isfinite(r) or not 0 < r <= .25 for r in radii)):
        raise ValueError("Radii must be distinct, positive normalized-coordinate RMS values")
    critics = cfg["probe_critics"]
    if not critics or len({c["name"] for c in critics}) != len(critics):
        raise ValueError("Probe critics must have distinct names")
    for critic in critics:
        if not SAFE_NAME.fullmatch(critic["name"]) or critic["variant"] not in VARIANTS:
            raise ValueError("Unsafe critic name or unknown screened architecture")
        checked_file(critic["checkpoint"], critic["sha256"])
        weights = np.asarray(critic["weights"], dtype=np.float64)
        if (weights.shape != (5,) or not np.isfinite(weights).all() or
                (weights < 0).any() or not (weights > 0).any() or
                not math.isfinite(critic.get("beta", 0.)) or critic.get("beta", 0.) < 0):
            raise ValueError("Each critic needs five nonnegative objective weights and beta >= 0")
        if critic["variant"] == "legacy" and critic["weights"] != [1, 0, 0, 0, 0]:
            raise ValueError("Legacy scalar-WL critic cannot represent other metrics")
    return cfg, verification, physical


def source_files():
    """Freeze imported local Python sources and the exact Hydra defaults used.

    sys.modules is a bounded list of already imported files, not a storage scan.
    Relevant upstream architecture/legalizer sources are imported by baseline.api.
    """
    files = {"fql/qstudy_probe.slurm": ROOT/"fql/qstudy_probe.slurm"}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name or not str(name).endswith(".py"):
            continue
        path = Path(name).absolute()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        # The environment's installed packages are versioned below, not copied.
        if relative.parts[0] == ".venv":
            continue
        files[str(relative)] = path
    for name in ["config_eval.yaml", "model/att-gnn.yaml", "model/conv-layer/gat.yaml",
                 "model/size/large.yaml", "guidance/none.yaml", "legalizer/none.yaml",
                 "sc_placer/sgd.yaml"]:
        relative = "upstream/flowplace/flow_matching/configs/"+name
        files[relative] = ROOT/relative
    files["fql/qstudy_probe.py"] = Path(__file__)
    return files


def freeze(experiment, cfg, physical):
    paths = source_files()
    binding = dict(schema_version=1, config_sha256=sha(experiment/"config.json"),
        backend_context=physical,
        cache=checked_file(RUNTIME/"benchmark_cache"/(cfg["design"]+".pt"), cfg["cache_sha256"]),
        common_checkpoint=checked_file(cfg["common_checkpoint"], cfg["common_checkpoint_sha256"]),
        construction_pretrained_checkpoint=dict(path=str(CHECKPOINT.resolve()), sha256=sha(CHECKPOINT)),
        critics=[dict(name=c["name"], variant=c["variant"],
            checkpoint=checked_file(c["checkpoint"], c["sha256"]),
            weights=c["weights"], beta=c.get("beta", 0.)) for c in cfg["probe_critics"]],
        source_sha256={name: sha(path) for name, path in sorted(paths.items())},
        packages=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda),
        prior="uniform[-1,1]", actor_clip=True, generator_device="cpu", generator_dtype="float32",
        legalizer=dict(w_reg=.1, w_dist=1., grid_res=128, padding=.001),
        terminal_mask=0, role="diagnostic", training_feedback_eligible=False,
        direction_seed=cfg.get("direction_seed", cfg["base_noise_seed"]+1000000),
        expected_count=cfg["base_count"]*(1+len(cfg["radii"])*2*(len(cfg["probe_critics"])+1)))
    path = experiment/"probe_binding.json"
    if path.exists():
        if read_json(path) != binding:
            raise ValueError("Refusing to mix changed configuration/data/checkpoint/source into probe export")
    else:
        archive = experiment/"probe_sources"
        archive.mkdir(exist_ok=True)
        for name, source in paths.items():
            target = archive/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != binding["source_sha256"][name]:
                raise ValueError("Conflicting source snapshot: "+name)
            shutil.copyfile(source, target)
        atomic_json(path, binding)
    for name, digest in binding["source_sha256"].items():
        if sha(experiment/"probe_sources"/name) != digest:
            raise ValueError("Probe source snapshot changed: "+name)
    return binding


def verify_probe_request(row, args, cfg, physical, binding):
    sample = checked_request(row, args, cfg, physical)
    expected = dict(probe_binding_sha256=object_sha(binding), role="diagnostic",
                    training_feedback_eligible=False, terminal_mask=0)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Changed probe identity or TRAIN eligibility")
    if row["predictions_sha256"] != object_sha(row["predictions"]):
        raise ValueError("Changed frozen critic predictions")
    if row["common_checkpoint"] != binding["common_checkpoint"]:
        raise ValueError("Changed base policy identity")
    if row["kind"] == "gradient":
        source = next(c for c in binding["critics"] if c["name"] == row["critic_name"])
        if row["probe_critic"] != source:
            raise ValueError("Changed gradient-generating critic")
        expected_source = source["checkpoint"]
    elif row["kind"] in ["baseline", "random"]:
        expected_source = binding["common_checkpoint"]
    else:
        raise ValueError("Unknown probe direction kind")
    if (row["source_checkpoint"] != expected_source["path"] or
            row["checkpoint_sha256"] != expected_source["sha256"]):
        raise ValueError("Request does not bind its actual generating checkpoint")
    with np.load(sample/"actions.npz", allow_pickle=False) as arrays:
        if (action_sha(arrays["direction"]) != row["direction_sha256"] or
                action_sha(arrays["base_raw_action"]) != row["base_raw_action_sha256"] or
                action_sha(arrays["base_executed_action"]) != row["base_action_sha256"]):
            raise ValueError("Changed controlled direction or matched base")
    return sample


def build_models(cfg, problem):
    data, manifest = load_dataset(cfg["dataset_path"])
    if manifest["design"] != cfg["design"]:
        raise ValueError("Dataset belongs to a different circuit")
    train = data["train"]
    x = torch.tensor(train["raw_action"], dtype=torch.float32).reshape(len(train["raw_action"]), -1, 2)
    costs, mask = measured_costs(torch.tensor(train["metrics"][:, :5], dtype=torch.float32),
                                METRICS, torch.tensor(train["metric_mask"][:, :5]))
    successful = mask[:, 0] & (costs[:, 0] > 0)
    x, costs, mask = x[successful], costs[successful], mask[successful]
    families = torch.tensor(train["family"], dtype=torch.long)[successful]
    cache = torch.load(RUNTIME/"benchmark_cache"/(cfg["design"]+".pt"), map_location="cpu", weights_only=False)
    movable = ~problem.cond.is_ports.bool().cpu()
    models, metadata = {}, {}
    for spec in cfg["probe_critics"]:
        checkpoint = Path(spec["checkpoint"]).resolve()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        complete = read_json(checkpoint.with_name("completed.json"))
        identity = saved["binding"]
        if (identity["dataset_sha256"] != cfg["dataset_sha256"] or
                identity["cache_sha256"] != cfg["cache_sha256"] or
                identity["config"]["variant"] != spec["variant"] or
                complete["binding"] != identity or complete["checkpoint_sha256"] != spec["sha256"] or
                complete["status"] != "completed" or
                saved["step"] != identity["config"]["steps"] or
                saved["train_indices"] != list(range(len(x)))):
            raise ValueError("Critic is not the complete full-TRAIN refit: "+spec["name"])
        for relative, digest in identity["source_sha256"].items():
            if sha(ROOT/relative) != digest:
                raise ValueError("Screened critic implementation changed: "+relative)
        seed = identity["config"]["seed"]
        torch.manual_seed(seed)
        model, _ = make_model(cache, x, costs, mask, movable, spec["variant"], families, seed)
        model.load_state_dict(saved["model"], strict=True)
        model = model.cuda().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[spec["name"]] = model
        legacy = spec["variant"] == "legacy"
        scale = [float(costs[:, 0].mean())]+[1.]*4 if legacy else model.objective_scale.detach().cpu().tolist()
        metadata[spec["name"]] = dict(metric_names=list(METRICS), objective_scale=scale,
            weights=spec["weights"], beta=spec.get("beta", 0.), variant=spec["variant"],
            checkpoint_sha256=spec["sha256"], screening_binding=identity,
            note="TRAIN scales frozen; ensemble disagreement is a heuristic, not a confidence bound")
    common = torch.load(cfg["common_checkpoint"], map_location="cuda", weights_only=False)
    if not all(key in common for key in ["flow", "actor"]):
        raise ValueError("The common checkpoint must contain the completed flow and independent actor")
    actor = GraphPolicy(problem).eval()
    state = common["actor"]
    if (not torch.equal(state["movable"].bool(), actor.movable) or
            not torch.equal(state["fixed"], actor.fixed)):
        raise ValueError("Common policy coordinates/movable mask differ from the frozen graph context")
    actor.load_state_dict(state, strict=True)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, models, metadata


def critic_reward(model, action, spec):
    if spec["variant"] == "legacy":
        members = model(action)
        return members.mean(1)-spec.get("beta", 0.)*members.std(1, unbiased=False)
    return model.reward(action, spec["weights"], uncertainty_beta=spec.get("beta", 0.))


@torch.no_grad()
def predictions(models, action, cfg):
    result = {}
    for spec in cfg["probe_critics"]:
        model = models[spec["name"]]
        legacy = spec["variant"] == "legacy"
        members = model(action[None]) if legacy else model.member_rewards(action[None], spec["weights"])
        costs = None if legacy else model(action[None])[0].mean(0).cpu().tolist()
        result[spec["name"]] = dict(member_rewards=members[0].cpu().tolist(),
            reward_mean=float(members.mean()), reward_std=float(members.std(unbiased=False)),
            objective=float(critic_reward(model, action[None], spec)[0]), mean_costs=costs)
    return result


def rms(delta, movable):
    return float(delta[movable].square().mean().sqrt())


def unit_direction(vector, movable):
    vector = vector*movable[:, None]
    norm = vector[movable].square().mean().sqrt()
    if not bool(torch.isfinite(vector).all()) or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("A predeclared probe has zero/nonfinite direction; do not invent gradient evidence")
    return vector/norm, float(norm)


def emit(args, cfg, physical, binding, problem, metric, models, records, base, raw,
         direction, kind, radius, radius_index, sign, spec=None, norm=None):
    index = len(records)
    name = "base" if kind == "baseline" else ("grad_"+spec["name"] if kind == "gradient" else "random")
    suffix = "base" if kind == "baseline" else f"r{radius_index}_{'plus' if sign == 1 else 'minus'}"
    arm = name+"_"+suffix
    sample = args.experiment/f"seed_0/round_0/requests/{arm}/diagnostic/{base['index']}"
    candidate = raw.detach().cpu().numpy().astype(np.float32)
    n = base["noise"].detach().cpu().numpy().astype(np.float32)
    source = binding["common_checkpoint"] if spec is None else checked_file(spec["checkpoint"], spec["sha256"])
    base_identity = dict(base_raw_action_sha256=action_sha(base["raw"].cpu().numpy()),
                         base_action_sha256=action_sha(base["action"].cpu().numpy()))
    expected = dict(index=index, base_index=base["index"], kind=kind, sign=sign,
        radius_rms=radius, radius_index=radius_index, arm=arm,
        raw_action_sha256=action_sha(candidate), noise_sha256=action_sha(n),
        direction_sha256=action_sha(direction.cpu().numpy()), **base_identity)
    path = sample/"request.json"
    if path.exists():
        old = read_json(path)
        verify_probe_request(old, args, cfg, physical, binding)
        if any(old[key] != value for key, value in expected.items()):
            raise ValueError("Interrupted export changed a matched base, proposal or direction")
        records.append(old)
        return
    sample.mkdir(parents=True, exist_ok=True)
    if (sample/"outcome.json").exists():
        raise ValueError("Orphan physical outcome without its immutable request")
    with torch.no_grad():
        action = base["action"] if kind == "baseline" else legalize(raw, problem)
        pred = predictions(models, raw, cfg)
    values = action.cpu().numpy().astype(np.float32)
    geometry = legality(values, metric.sizes)
    movable = ~problem.cond.is_ports.bool()
    raw_step = rms(raw-base["raw"], movable)
    executed_step = rms(action-base["action"], movable)
    row = dict(**expected, seed=0, round=0, role="diagnostic", graph=cfg["design"],
        terminal_mask=0, training_feedback_eligible=False, sample_dir=str(sample),
        config_sha256=physical["config_sha256"], dataset_sha256=cfg["dataset_sha256"],
        cache_sha256=cfg["cache_sha256"], probe_binding_sha256=object_sha(binding),
        common_checkpoint=binding["common_checkpoint"],
        source_checkpoint=source["path"], checkpoint_sha256=source["sha256"],
        raw_action=candidate.tolist(), executed_action=values.tolist(), action_sha256=action_sha(values),
        predictions=pred, predictions_sha256=object_sha(pred),
        predicted_q=None if spec is None else pred[spec["name"]]["member_rewards"],
        critic_name=None if spec is None else spec["name"], direction_original_rms=norm,
        noise_seed=cfg["base_noise_seed"], direction_seed=binding["direction_seed"],
        base_request_index=base["request_index"], geometry=geometry, stage="global_route_only",
        run_id=f"go2o_qsp_{cfg['run_prefix']}_b{base['index']}_{arm}",
        macro_hpwl=float(metric(values)), raw_macro_hpwl=float(metric(candidate)), actual_raw_step_rms=raw_step,
        executed_step_rms=executed_step, legalizer_jump_rms=rms(action-raw, movable),
        legalizer_displacement_change_rms=rms((action-raw)-(base["action"]-base["raw"]), movable),
        executed_to_raw_step_ratio=None if raw_step == 0 else executed_step/raw_step,
        same_executed_as_base=action_sha(values) == base_identity["base_action_sha256"],
        proposal_kind="fresh local gradient diagnostic; no trained-policy or replay claim")
    if spec is not None:
        row["probe_critic"] = next(c for c in binding["critics"] if c["name"] == spec["name"])
    np.savez_compressed(sample/"actions.npz", noise=n, raw_action=candidate, action=values,
        direction=direction.cpu().numpy(), base_raw_action=base["raw"].cpu().numpy(),
        base_executed_action=base["action"].cpu().numpy())
    if geometry["legal"]:
        save_sample(sample, action, base["noise"], problem, row)
        output = sample/(cfg["design"]+".def")
        # Only unsealed partial export artifacts are rebuilt. Sealed requests
        # and measured outcomes are validated above and never overwritten.
        output.unlink(missing_ok=True)
        output.with_suffix(".audit.json").unlink(missing_ok=True)
        audit = convert(cfg["design"], sample/"macro_physical.json", output)
        row.update(def_path=str(output), def_sha256=sha(output), rounded_geometry=audit["legal_no_halo"])
        if not audit["legal_no_halo"]:
            row["pre_backend_failure"] = "rounded_DEF_geometry"
    else:
        row["pre_backend_failure"] = "normalized_geometry"
    atomic_json(path, row)
    verify_probe_request(row, args, cfg, physical, binding)
    records.append(row)


def export(args, cfg, physical, binding):
    folder = args.experiment/"seed_0/round_0/requests"
    if (folder/"completed.json").exists():
        rows, _ = load_requests(args.experiment, 0, 0)
        if len(rows) != binding["expected_count"]:
            raise ValueError("Completed export changed its predeclared denominator")
        for row in rows:
            verify_probe_request(row, args, cfg, physical, binding)
        print(json.dumps(dict(status="already_exported_verified", requests=len(rows))), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Probe actor inference requires an allocated Slurm GPU")
    problem = load_problem(cfg["design"], "cuda")
    actor, models, metadata = build_models(cfg, problem)
    metadata_path = args.experiment/"probe_models.json"
    if metadata_path.exists() and read_json(metadata_path) != metadata:
        raise ValueError("Frozen objective/normalization metadata changed")
    atomic_json(metadata_path, metadata)
    metric = MacroHPWL(problem.benchmark_info)
    # Separate CPU generators make the fresh matched noise/directions independent
    # of critic initialization, checkpoint order and CUDA global RNG consumption.
    generator = torch.Generator(device="cpu").manual_seed(cfg["base_noise_seed"])
    direction_generator = torch.Generator(device="cpu").manual_seed(binding["direction_seed"])
    noises = 2*torch.rand(cfg["base_count"], len(problem.x), 2, generator=generator)-1
    records = []
    for base_index, host_noise in enumerate(noises):
        noise = host_noise.cuda()
        with torch.no_grad():
            raw = actor(noise[None], clip=True)[0]
            action = legalize(raw, problem)
        base = dict(index=base_index, raw=raw, action=action, noise=noise, request_index=len(records))
        emit(args, cfg, physical, binding, problem, metric, models, records, base, raw,
             torch.zeros_like(raw), "baseline", 0., -1, 0)
        directions = []
        for spec in cfg["probe_critics"]:
            differentiable = raw.detach().clone().requires_grad_(True)
            q = critic_reward(models[spec["name"]], differentiable[None], spec)[0]
            gradient = torch.autograd.grad(q, differentiable)[0]
            direction, norm = unit_direction(gradient.detach(), actor.movable)
            directions.append((spec, direction, norm))
        for radius_index, radius in enumerate(cfg["radii"]):
            random, random_norm = unit_direction(
                torch.randn(raw.shape, generator=direction_generator).cuda(), actor.movable)
            for spec, direction, norm in directions+[(None, random, random_norm)]:
                for sign in [-1, 1]:
                    candidate = actor.restore((raw+sign*radius*direction)[None], clip=True)[0]
                    emit(args, cfg, physical, binding, problem, metric, models, records, base, candidate,
                         direction, "random" if spec is None else "gradient", radius, radius_index,
                         sign, spec, norm)
    if len(records) != binding["expected_count"] or len({r["run_id"] for r in records}) != len(records):
        raise ValueError("Incomplete or duplicated directional-probe denominator")
    folder.mkdir(parents=True, exist_ok=True)
    tasks = folder/"tasks.jsonl"
    temp = tasks.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in records))
    temp.replace(tasks)
    atomic_json(folder/"completed.json", dict(status="completed", count=len(records),
        tasks_sha256=sha(tasks), config_sha256=physical["config_sha256"],
        probe_binding_sha256=object_sha(binding), models_metadata_sha256=sha(metadata_path),
        slurm_job_id=os.environ["SLURM_JOB_ID"], selection="all predeclared probes; no Q filtering",
        labels_enter_replay=False, independent_training_seeds=0, policy_changed=False))
    print(json.dumps(dict(status="exported", count=len(records), labels_enter_replay=False)), flush=True)


def utility(outcome, spec, metadata, physical_failure_reward):
    if outcome["status"] != "success":
        return float(physical_failure_reward)
    metrics = outcome["metrics"]
    native = np.asarray([metrics.get(name, np.nan) for name in METRICS], dtype=np.float64)
    costs = native.copy()
    costs[1:3] = np.maximum(-costs[1:3], 0.)
    weights = np.asarray(spec["weights"], dtype=np.float64)
    scale = np.asarray(metadata[spec["name"]]["objective_scale"], dtype=np.float64)
    if not np.isfinite(costs[weights > 0]).all():
        return None
    return -float(np.sum(np.where(weights > 0, costs, 0.)/scale*weights))


def report(args, cfg, verification, physical, binding):
    rows, tasks = load_requests(args.experiment, 0, 0)
    marker = read_json(tasks.with_name("completed.json"))
    models_path = args.experiment/"probe_models.json"
    if (len(rows) != binding["expected_count"] or
            marker["models_metadata_sha256"] != sha(models_path)):
        raise ValueError("Probe denominator or frozen objective normalization changed")
    metadata = read_json(models_path)
    outcomes, missing, outcome_bindings = {}, [], []
    for row in rows:
        sample = verify_probe_request(row, args, cfg, physical, binding)
        path = sample/"outcome.json"
        if not path.exists():
            missing.append(row["index"])
            continue
        out = read_json(path)
        checked_existing(out, row, sample, tasks, physical, cfg, verification)
        outcomes[row["index"]] = out
        outcome_bindings.append(dict(path=str(path), sha256=sha(path)))
    lookup = {(r["base_index"], r["kind"], r["critic_name"], r["radius_index"], r["sign"]): r for r in rows}
    pairs = []
    for base in range(cfg["base_count"]):
        baseline = lookup[(base, "baseline", None, -1, 0)]
        for radius_index, radius in enumerate(cfg["radii"]):
            for spec in cfg["probe_critics"]:
                # Both gradient and matched random use the same critic objective.
                for kind, source_name in [("gradient", spec["name"]), ("random", None)]:
                    plus = lookup[(base, kind, source_name, radius_index, 1)]
                    minus = lookup[(base, kind, source_name, radius_index, -1)]
                    ids = [baseline["index"], plus["index"], minus["index"]]
                    available = all(index in outcomes for index in ids)
                    measured = [utility(outcomes[i], spec, metadata, cfg["physical_failure_reward"])
                                for i in ids] if available else [None]*3
                    valid = available and all(value is not None for value in measured)
                    success = available and all(outcomes[i]["status"] == "success" for i in ids)
                    predicted = (plus["predictions"][spec["name"]]["objective"]-
                                 minus["predictions"][spec["name"]]["objective"])
                    delta = measured[1]-measured[2] if valid else None
                    pair = dict(base_index=base, critic_name=spec["name"], kind=kind, radius_rms=radius,
                        request_indices=ids, all_measured=available, all_physical_success=success,
                        statuses=[outcomes[i]["status"] if i in outcomes else "missing" for i in ids],
                        predicted_plus_minus=predicted, measured_plus_minus=delta,
                        plus_vs_base=None if not valid else measured[1]-measured[0],
                        minus_vs_base=None if not valid else measured[2]-measured[0],
                        prediction_sign_agrees=None if not valid or abs(delta) <= 1e-12 or abs(predicted) <= 1e-12
                        else bool(delta*predicted > 0), measured_tie=None if not valid else abs(delta) <= 1e-12,
                        plus_raw_step_rms=plus["actual_raw_step_rms"], minus_raw_step_rms=minus["actual_raw_step_rms"],
                        plus_executed_step_rms=plus["executed_step_rms"], minus_executed_step_rms=minus["executed_step_rms"],
                        plus_legalizer_jump_rms=plus["legalizer_jump_rms"],
                        minus_legalizer_jump_rms=minus["legalizer_jump_rms"],
                        same_legalized_layout=plus["action_sha256"] == minus["action_sha256"],
                        plus_same_as_base=plus["same_executed_as_base"], minus_same_as_base=minus["same_executed_as_base"])
                    pairs.append(pair)
    grouped = []
    for spec in cfg["probe_critics"]:
        for radius in cfg["radii"]:
            for kind in ["gradient", "random"]:
                selected = [p for p in pairs if p["critic_name"] == spec["name"] and
                            p["radius_rms"] == radius and p["kind"] == kind]
                valid = [p for p in selected if p["measured_plus_minus"] is not None]
                sign = [p for p in valid if p["prediction_sign_agrees"] is not None]
                success = [p for p in valid if p["all_physical_success"]]
                grouped.append(dict(critic_name=spec["name"], radius_rms=radius, kind=kind,
                    predeclared_bases=len(selected), measured_pairs=len(valid), all_success_pairs=len(success),
                    tied_pairs=sum(p["measured_tie"] for p in valid),
                    sign_comparisons=len(sign), sign_agreements=sum(p["prediction_sign_agrees"] for p in sign),
                    mean_measured_plus_minus=None if not valid else float(np.mean([p["measured_plus_minus"] for p in valid])),
                    mean_plus_vs_base=None if not valid else float(np.mean([p["plus_vs_base"] for p in valid])),
                    successful_mean_plus_vs_base=None if not success else float(np.mean([p["plus_vs_base"] for p in success])),
                    positive_vs_base=sum(p["plus_vs_base"] > 1e-12 for p in valid),
                    same_legalized_pairs=sum(p["same_legalized_layout"] for p in selected)))
    dest = args.experiment/"report"
    dest.mkdir(exist_ok=True)
    atomic_json(dest/"evidence.json", dict(status="partial" if missing else "all_requested_accounted",
        requested=len(rows), measured=len(outcomes), missing_request_indices=missing,
        config_sha256=physical["config_sha256"], probe_binding_sha256=object_sha(binding),
        tasks_sha256=sha(tasks), objective_metadata_sha256=sha(models_path),
        pairs=pairs, grouped=grouped, outcome_bindings=outcome_bindings,
        labels_entered_replay=False, policy_changed=False, independent_training_seeds=0,
        scope=f"{cfg['base_count']} fresh common-actor neighborhoods per design; matched local diagnostics, not policy-quality validation",
        utility_note="Measured TRAIN-scaled cost utility; failure utility -4 is bookkeeping, no fabricated PPA. Beta only affects predicted Q.",
        not_final_table2=True, detailed_route_DRC=None, source_sha256=sha(__file__)))
    print(json.dumps(dict(status="reported", measured=len(outcomes), requested=len(rows),
                         evidence=str(dest/"evidence.json"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=["export", "report"], default="export")
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    args.seed = 0
    args.round = 0
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Probe export, tensor loading and reports must run under Slurm")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, verification, physical = settings(args.experiment)
    with (args.experiment/".probe_export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = freeze(args.experiment, cfg, physical)
        if args.stage == "export":
            export(args, cfg, physical, binding)
        else:
            report(args, cfg, verification, physical, binding)


if __name__ == "__main__":
    main()
