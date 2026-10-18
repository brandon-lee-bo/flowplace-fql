"""Evaluation-only, paired Euler10 exports of measured-quality FM controls.

Each experiment contains one of uniform/quality_rank, three trained flows and
eight existing development noises per seed.  Declaration is a separate CPU
stage.  This module never trains, draws noise, loads a one-step actor, selects
outputs, submits EDA, or admits evaluation data to replay.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import RUNTIME, legalize, load_problem, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_models import GraphPolicy
from . import qstudy_flow_eval as original
from . import qstudy_quality_weighted_flow as training
from .qstudy_backend import checked_request, context
from .qstudy_data import action_sha
from .qstudy_teacher import file_binding, object_sha, source_files

ROOT = Path(__file__).resolve().parents[1]
ARMS, SEEDS, COUNT = ("uniform", "quality_rank"), (0, 1, 2), 8
SOURCE_KIND = "quality_weighted_flow_development"
SAMPLER = "trained_flow_euler10_no_intermediate_constraints_final_legalize"
OWN_SOURCES = ("fql/qstudy_quality_weighted_eval.py", "fql/qstudy_quality_weighted_eval.slurm")


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    original.immutable_json(path, value)


def checked_artifact(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("An explicit path/SHA artifact binding is required")
    actual = file_binding(value["path"], value["sha256"])
    if actual != value:
        raise ValueError("Artifact paths must be canonical")
    return actual


def training_artifacts(cfg):
    """Validate all six completions; hashing tensors never deserializes them."""
    manifest = checked_artifact(cfg["training_manifest"])
    plan, _ = training.checked_plan(Path(manifest["path"]))
    root = Path(cfg["training_root"]).resolve()
    if (Path(manifest["path"]) != root/"plan.json" or Path(plan["output"]) != root or
            plan["settings"]["steps"] != 1000 or plan["arms"] != list(ARMS) or
            plan["seeds"] != list(SEEDS)):
        raise ValueError("Changed fixed six-task/1000-update training manifest")
    result = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            folder = root/arm/f"seed_{seed}"
            artifacts = {name: file_binding(folder/filename) for name, filename in (
                ("checkpoint", "checkpoint.pt"), ("completion", "completed.json"),
                ("binding", "binding.json"), ("curve", "learning_curve.json"))}
            done, binding, curve = (read(artifacts[k]["path"]) for k in ("completion", "binding", "curve"))
            task = dict(index=ARMS.index(arm)*3+seed, arm=arm, seed=seed)
            expected_weight = (np.ones(len(plan["population"]["weights"]), np.float32)
                if arm == "uniform" else np.asarray(plan["population"]["weights"], np.float32))
            if (done.get("status") != "completed" or done.get("steps") != 1000 or
                    done.get("flow_updates") != 1000 or
                    any(done.get(k) != 0 for k in ("actor_updates", "critic_updates", "layout_generation_calls", "backend_calls")) or
                    done.get("checkpoint_sha256") != artifacts["checkpoint"]["sha256"] or
                    done.get("curve_sha256") != artifacts["curve"]["sha256"] or done.get("binding") != binding or
                    binding.get("plan") != manifest or binding.get("task") != task or
                    binding.get("common") != plan["inputs"]["common"][str(seed)] or
                    binding.get("source_sha256") != plan["source_sha256"] or
                    binding.get("population_sha256") != object_sha(plan["population"]) or
                    binding.get("effective_weight_sha256") != training.array_sha(expected_weight) or
                    binding.get("initial_state_key") != "flow" or
                    binding.get("model_mode") != plan["settings"]["model_mode"] or
                    not curve or curve[-1].get("step") != 1000 or
                    curve[-1].get("draw_chain_sha256") != done.get("draw_chain_sha256")):
                raise ValueError("Incomplete/changed trained flow or TRAIN/stream binding")
            artifacts.update(common=binding["common"], draw_chain_sha256=done["draw_chain_sha256"],
                             task=task, trained_state_key="flow", steps=1000)
            result[arm][str(seed)] = artifacts
    for seed in SEEDS:
        if result["uniform"][str(seed)]["draw_chain_sha256"] != result["quality_rank"][str(seed)]["draw_chain_sha256"]:
            raise ValueError("Paired FM arms did not use identical row/noise/time streams")
    selected = cfg["trained_artifacts"]
    if set(selected) != {str(s) for s in SEEDS} or any(
            selected[str(s)] != {k: result[cfg["arm"]][str(s)][k]
                for k in ("checkpoint", "completion", "binding", "curve")} for s in SEEDS):
        raise ValueError("Config does not bind the exact three trained checkpoints/completions")
    return plan, result


def settings(experiment):
    cfg, _, physical = context(experiment)
    if (cfg.get("arm") not in ARMS or cfg.get("seeds") != list(SEEDS) or
            cfg.get("flow_steps") != 10 or cfg.get("evaluation_noise_seed") != 4600000 or
            cfg.get("evaluation_samples") != COUNT or cfg.get("native_budget") != 24 or
            cfg.get("evaluation_purpose") != "development_validation" or
            cfg.get("online_flow_update") is not False or cfg.get("online_rounds") != 0 or
            not isinstance(cfg.get("backend_concurrency"), int) or
            not 1 <= cfg["backend_concurrency"] <= 4 or
            not cfg.get("run_prefix", "").isalnum() or len(cfg["run_prefix"]) > 32):
        raise ValueError("Changed eval-only 3x8 per-arm Euler10 declaration")
    parent = Path(cfg["flow_eval_experiment"]).resolve()
    protected = (ROOT, parent, Path(cfg["training_root"]).resolve())
    if any(experiment == p or p in experiment.parents or experiment in p.parents for p in protected):
        raise ValueError("New evaluation experiment must be disjoint from source artifacts")
    file_binding(parent/"config.json", cfg["flow_eval_config_sha256"])
    file_binding(parent/"declaration.json", cfg["flow_eval_declaration_sha256"])
    pcfg, pphysical, controls = original.settings(parent)
    declared = read(parent/"declaration.json")
    pairs = original.source_pairs(pcfg, controls)
    if (declared.get("source_kind") != original.SOURCE_KIND or
            declared.get("source_sampler") != original.SAMPLER or declared.get("pairs") != pairs or
            declared.get("expected_requests") != 24):
        raise ValueError("Original common-flow noise declaration changed")
    for name, digest in declared["source_sha256"].items():
        file_binding(ROOT/name, digest)
        file_binding(parent/"flow_eval_sources"/name, digest)
    for key in original.CONTEXT_FIELDS:
        if cfg.get(key) != pcfg.get(key):
            raise ValueError("Changed common-flow evaluation context: "+key)
    if {k:v for k,v in physical.items() if k != "config_sha256"} != {
            k:v for k,v in pphysical.items() if k != "config_sha256"}:
        raise ValueError("Paired FM uses a different native physical context")
    plan, trained = training_artifacts(cfg)
    if (plan["inputs"]["dataset"]["sha256"] != cfg["dataset_sha256"] or
            plan["inputs"]["cache"]["sha256"] != cfg["cache_sha256"] or
            plan["inputs"]["base_config"] != file_binding(pcfg["bc_experiment"]+"/config.json")):
        raise ValueError("Training and common-flow data/cache/BC initialization differ")
    for seed in SEEDS:
        if plan["inputs"]["common"][str(seed)] != file_binding(
                pcfg["common_checkpoints"][str(seed)], pcfg["common_checkpoint_sha256"][str(seed)]):
            raise ValueError("Changed per-seed current behavior-flow initialization")
    return cfg, physical, pairs, plan, trained


def declaration_value(experiment, cfg, physical, pairs, plan, trained):
    files = source_files()
    files.update({name: ROOT/name for name in OWN_SOURCES})
    return dict(version=1, source_kind=SOURCE_KIND, arm=cfg["arm"], source_state_key="flow",
        source_sampler=SAMPLER, seeds=list(SEEDS), indices_per_seed=list(range(COUNT)),
        expected_requests=24, stage="global_route_only", role="evaluation", native_budget=24,
        backend_concurrency=cfg["backend_concurrency"], evaluation_purpose="development_validation",
        config=file_binding(experiment/"config.json"), physical_context=physical,
        training_manifest=cfg["training_manifest"], training_population_sha256=object_sha(plan["population"]),
        all_paired_training=trained, source_flow_eval=dict(
            config=file_binding(Path(cfg["flow_eval_experiment"])/"config.json"),
            declaration=file_binding(Path(cfg["flow_eval_experiment"])/"declaration.json")),
        pairs=pairs, source_noise_seed=4600000, source_noise="verbatim saved matched CUDA arrays; no RNG draw",
        flow_steps=10, intermediate_hard_constraints=False, intermediate_coordinate_clipping=False,
        final_coordinate_clipping=True, fixed_ports_restored=True,
        legalizer=dict(w_reg=.1, w_dist=1., grid_res=128, padding=.001),
        training_feedback_eligible=False, offline_train_eligible=False, terminal_mask=0,
        selection="all three fixed final checkpoints and eight fixed development noises; no selection",
        common_flow_reuse="existing24 common-flow native measurements only under identical source/noise/sampler/physical bindings",
        public_FlowPlace_baseline=False,
        source_sha256={name:sha(path) for name,path in sorted(files.items())})


def dispatch_value(experiment, cfg, pairs):
    requests = [dict(experiment=str(experiment), arm=cfg["arm"], seed=p["seed"], round=0,
        index=p["index"], role="evaluation", noise_sha256=p["noise_sha256"], source_kind=SOURCE_KIND,
        run_id=f"go2o_qsw_{cfg['run_prefix']}_s{p['seed']}_i{p['index']}") for p in pairs]
    return dict(declaration_sha256=sha(experiment/"declaration.json"), count=24,
                requests=requests, production_native_submitted=False)


def declare(experiment, cfg, physical, pairs, plan, trained):
    value = declaration_value(experiment, cfg, physical, pairs, plan, trained)
    target = experiment/"declaration.json"
    if not target.exists() and any(experiment.glob("seed_*")):
        raise ValueError("Cannot declare after any evaluation output exists")
    archive = experiment/"quality_flow_eval_sources"
    for name, digest in value["source_sha256"].items():
        destination = archive/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            file_binding(destination, digest)
        else:
            shutil.copyfile(ROOT/name, destination)
            file_binding(destination, digest)
    immutable(target, value)
    immutable(experiment/"dispatch.json", dispatch_value(experiment, cfg, pairs))
    return value


def checked_declaration(experiment, cfg, physical, pairs, plan, trained):
    value = read(experiment/"declaration.json")
    if value != declaration_value(experiment, cfg, physical, pairs, plan, trained):
        raise ValueError("Evaluation declaration/source closure changed")
    for name, digest in value["source_sha256"].items():
        file_binding(experiment/"quality_flow_eval_sources"/name, digest)
    if read(experiment/"dispatch.json") != dispatch_value(experiment, cfg, pairs):
        raise ValueError("Changed predeclared dispatch")
    return value


def trained_flow(cfg, declaration, seed):
    if not torch.cuda.is_available():
        raise RuntimeError("Flow export requires an allocated Slurm GPU")
    artifact = declaration["all_paired_training"][cfg["arm"]][str(seed)]
    file_binding(artifact["checkpoint"]["path"], artifact["checkpoint"]["sha256"])
    saved = torch.load(artifact["checkpoint"]["path"], map_location="cuda", weights_only=False)
    if (saved.get("step") != 1000 or saved.get("binding") != read(artifact["binding"]["path"]) or
            saved.get("draw_chain_sha256") != artifact["draw_chain_sha256"]):
        raise ValueError("Tensor checkpoint differs from completed training contract")
    problem = load_problem(cfg["design"], "cuda")
    flow = GraphPolicy(problem).eval()
    state = saved["flow"]
    if not torch.equal(state["movable"].bool(), flow.movable) or not torch.equal(state["fixed"], flow.fixed):
        raise ValueError("Trained flow fixed-node buffers differ from benchmark")
    flow.load_state_dict(state, strict=True)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    return problem, flow


def row_identity(cfg, declaration, pair):
    artifact = declaration["all_paired_training"][cfg["arm"]][str(pair["seed"])]
    return dict(arm=cfg["arm"], seed=pair["seed"], round=0, role="evaluation", index=pair["index"],
        source_kind=SOURCE_KIND, source_sampler=SAMPLER, source_state_key="flow", source_sampler_steps=10,
        training_feedback_eligible=False, offline_train_eligible=False,
        evaluation_purpose="development_validation", terminal_mask=0,
        intermediate_hard_constraints=False, intermediate_coordinate_clipping=False,
        final_coordinate_clipping=True, fixed_ports_restored=True,
        quality_flow_declaration_sha256=object_sha(declaration), source_noise_binding=pair,
        source_training_artifacts=artifact, training_manifest=cfg["training_manifest"],
        source_checkpoint=artifact["checkpoint"]["path"], checkpoint_sha256=artifact["checkpoint"]["sha256"],
        noise_seed=4600000, noise_sha256=pair["noise_sha256"], predicted_q=None,
        run_id=f"go2o_qsw_{cfg['run_prefix']}_s{pair['seed']}_i{pair['index']}")


def verify_row(row, cfg, physical, declaration, pair, experiment):
    args = argparse.Namespace(seed=pair["seed"], round=0)
    sample = checked_request(row, args, cfg, physical)
    expected = row_identity(cfg, declaration, pair)
    expected["sample_dir"] = str(experiment/f"seed_{pair['seed']}/round_0/requests"/cfg["arm"]/"evaluation"/str(pair["index"]))
    if any(row.get(k) != v for k,v in expected.items()):
        raise ValueError("Changed quality-FM checkpoint/stream/noise/eligibility binding")
    with np.load(sample/"actions.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["noise"], original.source_noise(pair)):
            raise ValueError("Quality-FM export did not use exact saved CUDA noise")
    return sample


def emit(experiment, cfg, physical, declaration, pair, problem, flow, metric):
    sample = experiment/f"seed_{pair['seed']}/round_0/requests"/cfg["arm"]/"evaluation"/str(pair["index"])
    if (sample/"request.json").exists():
        row = read(sample/"request.json")
        verify_row(row, cfg, physical, declaration, pair, experiment)
        return row
    if sample.exists() and any(sample.iterdir()):
        raise ValueError("Refuse to overwrite an orphan export or physical outcome")
    sample.mkdir(parents=True, exist_ok=True)
    noise = torch.from_numpy(original.source_noise(pair)).to(flow.fixed.device)
    with torch.no_grad():
        raw = flow.sample_flow(noise[None], steps=10)[0]
        action = legalize(raw, problem)
        if pair["index"] == 0:
            repeated = flow.sample_flow(noise[None], steps=10)[0]
            if not torch.equal(raw, repeated) or not torch.equal(action, legalize(repeated, problem)):
                raise ValueError("Fixed-noise trained flow/legalizer is not deterministic")
        if not torch.equal(raw[~flow.movable], flow.fixed[~flow.movable]):
            raise ValueError("Trained flow moved fixed ports")
    raw_array, executed = raw.cpu().numpy(), action.cpu().numpy()
    geometry = legality(executed, metric.sizes)
    row = dict(row_identity(cfg, declaration, pair), graph=cfg["design"], stage="global_route_only",
        config_sha256=physical["config_sha256"], dataset_sha256=cfg["dataset_sha256"], cache_sha256=cfg["cache_sha256"],
        sample_dir=str(sample), raw_action=raw_array.tolist(), executed_action=executed.tolist(),
        raw_action_sha256=action_sha(raw_array), action_sha256=action_sha(executed), geometry=geometry,
        macro_hpwl=float(metric(executed)), raw_macro_hpwl=float(metric(raw_array)),
        legalizer_jump_rms=float((action[flow.movable]-raw[flow.movable]).square().mean().sqrt()))
    np.savez_compressed(sample/"actions.npz", noise=noise.cpu().numpy(), raw_action=raw_array, action=executed)
    if geometry["legal"]:
        save_sample(sample, action, noise, problem, row)
        output = sample/(cfg["design"]+".def")
        audit = convert(cfg["design"], sample/"macro_physical.json", output)
        row.update(def_path=str(output), def_sha256=sha(output), rounded_geometry=audit["legal_no_halo"])
        if not audit["legal_no_halo"]:
            row["pre_backend_failure"] = "rounded_DEF_geometry"
    else:
        row["pre_backend_failure"] = "normalized_geometry"
    atomic_json(sample/"request.json", row)
    verify_row(row, cfg, physical, declaration, pair, experiment)
    return row


def checked_seed(experiment, cfg, physical, declaration, seed):
    rows, tasks = load_requests(experiment, seed, 0)
    marker = read(tasks.with_name("completed.json"))
    if (marker.get("status") != "completed" or marker.get("evaluation_never_replayed") is not True or
            marker.get("selection") != declaration["selection"] or
            marker.get("quality_flow_declaration_sha256") != object_sha(declaration) or
            marker.get("count") != 8 or [r["index"] for r in rows] != list(range(COUNT))):
        raise ValueError("Changed fixed8 quality-FM export manifest")
    pairs = [p for p in declaration["pairs"] if p["seed"] == seed]
    for row, pair in zip(rows, pairs):
        verify_row(row, cfg, physical, declaration, pair, experiment)
    return tasks


def export_seed(experiment, cfg, physical, declaration, seed):
    folder = experiment/f"seed_{seed}/round_0/requests"
    folder.mkdir(parents=True, exist_ok=True)
    with (experiment/f".export_seed_{seed}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/"completed.json").exists():
            checked_seed(experiment, cfg, physical, declaration, seed)
        else:
            problem, flow = trained_flow(cfg, declaration, seed)
            metric = MacroHPWL(problem.benchmark_info)
            pairs = [p for p in declaration["pairs"] if p["seed"] == seed]
            rows = [emit(experiment, cfg, physical, declaration, p, problem, flow, metric) for p in pairs]
            tasks = folder/"tasks.jsonl"
            temp = tasks.with_suffix(".jsonl.tmp")
            temp.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
            temp.replace(tasks)
            atomic_json(folder/"completed.json", dict(status="completed", count=COUNT,
                tasks_sha256=sha(tasks), config_sha256=physical["config_sha256"],
                quality_flow_declaration_sha256=object_sha(declaration),
                selection=declaration["selection"], evaluation_never_replayed=True))
            checked_seed(experiment, cfg, physical, declaration, seed)
    with (experiment/".completion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if all((experiment/f"seed_{s}/round_0/requests/completed.json").is_file() for s in SEEDS):
            tasks = {str(s):file_binding(checked_seed(experiment, cfg, physical, declaration, s)) for s in SEEDS}
            immutable(experiment/"export_completed.json", dict(status="completed", count=24,
                declaration_sha256=sha(experiment/"declaration.json"), config_sha256=physical["config_sha256"],
                dispatch_sha256=sha(experiment/"dispatch.json"), seed_tasks=tasks,
                backend_calls=0, production_native_submitted=False, evaluation_never_replayed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("declare", "export"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("All source/data/noise checks and inference require Slurm")
    if args.stage == "export" and args.seed is None:
        parser.error("GPU export requires an explicit declared --seed")
    experiment = args.experiment.resolve()
    cfg, physical, pairs, plan, trained = settings(experiment)
    if args.stage == "declare":
        with (experiment/".declaration.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            declare(experiment, cfg, physical, pairs, plan, trained)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Export requires an allocated Slurm GPU")
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        declaration = checked_declaration(experiment, cfg, physical, pairs, plan, trained)
        export_seed(experiment, cfg, physical, declaration, args.seed)
    print(json.dumps(dict(stage=args.stage, arm=cfg["arm"], seed=args.seed,
        planned_requests=24, new_noise_draws=0, training_calls=0, backend_calls=0)), flush=True)


if __name__ == "__main__":
    main()
