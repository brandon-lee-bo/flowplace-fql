"""Released public FlowPlace20hard on eight saved matched CUDA noises, eval only.

One fixed public checkpoint is NOT three independently trained flow models.
CPU declaration freezes all 3 BC/FQL source bindings; GPU export generates only
seed_0/index0..7. No training, replay, native execution or quality selection.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import CHECKPOINT, generate, legalize, load_model, load_problem, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .qstudy_backend import checked_request, context
from .qstudy_data import action_sha
from .qstudy_flow_eval import (COUNT, ROOT, immutable_json, read, settings as control_settings,
                               source_noise, source_pairs)
from .qstudy_teacher import file_binding, object_sha, source_files

PUBLIC_SHA = "0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359"
KIND = "released_public_flowplace_evaluation_only"
SAMPLER = "public_reverse_samples_20hard_uniform_wreg0.1_wdist1"
SELECTION = "all eight predeclared actual noises; one public checkpoint; no quality selection"


def settings(experiment):
    cfg, _, physical = context(experiment)
    source = Path(cfg["source_control_experiment"]).resolve()
    file_binding(source / "config.json", cfg["source_control_config_sha256"])
    file_binding(source / "declaration.json", cfg["source_control_declaration_sha256"])
    control, parent_physical, parents = control_settings(source)
    for key in ("design", "dataset_path", "dataset_sha256", "cache_sha256", "evaluation_noise_seed",
                "evaluation_samples", "physical_failure_reward", "backend_threads", "evaluation_purpose"):
        if cfg[key] != control[key]:
            raise ValueError("Public/control context differs: " + key)
    for key in physical:
        if key != "config_sha256" and physical[key] != parent_physical[key]:
            raise ValueError("Public/control physical evaluator differs: " + key)
    if (cfg["arm"] != "public_flowplace" or cfg["seeds"] != [0] or cfg["flow_steps"] != 20 or
            cfg["native_budget"] != 8 or cfg["online_flow_update"] is not False or
            cfg["backend_concurrency"] != 2 or cfg["public_checkpoint_sha256"] != PUBLIC_SHA):
        raise ValueError("Expected one released public checkpoint and eight eval-only exports")
    if not cfg["run_prefix"].isalnum() or len(cfg["run_prefix"]) > 32:
        raise ValueError("Unsafe public evaluation run prefix")
    file_binding(CHECKPOINT, PUBLIC_SHA)
    parent = read(source / "declaration.json")
    for name, digest in parent["source_sha256"].items():
        file_binding(ROOT / name, digest)
    pairs = source_pairs(control, parents)
    if pairs != parent["pairs"] or parent["expected_requests"] != 24:
        raise ValueError("Frozen matched-source declaration changed")
    return cfg, physical, pairs


def declare(experiment, cfg, physical, pairs):
    noises = []
    for index in range(COUNT):
        sources = [p for p in pairs if p["index"] == index]
        if sorted(p["seed"] for p in sources) != [0, 1, 2] or len({p["noise_sha256"] for p in sources}) != 1:
            raise ValueError("Source does not establish identical actual noise across all actor seeds")
        noises.append(dict(index=index, noise_sha256=sources[0]["noise_sha256"],
            noise_shape=sources[0]["noise_shape"], matched_pairs=sorted(sources, key=lambda p:p["seed"])))
    if len({n["noise_sha256"] for n in noises}) != COUNT:
        raise ValueError("Eight distinct saved noises are required")
    files = source_files()
    files["fql/qstudy_public_flow_eval.py"] = Path(__file__)
    files["fql/qstudy_public_flow_eval.slurm"] = ROOT / "fql/qstudy_public_flow_eval.slurm"
    source = Path(cfg["source_control_experiment"])
    value = dict(version=1, source_kind=KIND, source_sampler=SAMPLER, expected_requests=8,
        seeds=[0], independent_public_checkpoints=1, distinct_noise_arrays=8, native_budget=8,
        source_checkpoint=file_binding(CHECKPOINT, PUBLIC_SHA), physical_context=physical,
        source_control_config=file_binding(source / "config.json", cfg["source_control_config_sha256"]),
        source_control_declaration=file_binding(source / "declaration.json", cfg["source_control_declaration_sha256"]),
        source_actual_noise="verbatim saved CUDA arrays4600000; no RNG regeneration",
        role="evaluation", stage="global_route_only", evaluation_purpose="development_validation",
        source_sampler_steps=20, intermediate_hard_constraints=True, prior="uniform[-1,1]",
        hard_constraint_w_reg=.1, hard_constraint_w_dist=1., upstream_torch_compile_unchanged=True,
        intermediate_clip=[-2, 2], external_legalizer=dict(w_reg=.1, w_dist=1., grid_res=128, padding=.001),
        fixed_ports_restored=True, training=False, training_feedback_eligible=False,
        offline_train_eligible=False, terminal_mask=0, selection=SELECTION,
        limitation="one public checkpoint/eight generation replicates; not3trainingseeds or fresh confirmation; GRT only",
        source_sha256={name:sha(path) for name,path in sorted(files.items())}, noises=noises)
    archive = experiment / "public_eval_sources"
    for name, path in files.items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            file_binding(target, value["source_sha256"][name])
        else:
            shutil.copyfile(path, target)
    immutable_json(experiment / "declaration.json", value)
    immutable_json(experiment / "dispatch.json", dict(count=8, declaration_sha256=sha(experiment / "declaration.json"),
        requests=[dict(experiment=str(experiment), seed=0, round=0, role="evaluation", index=i,
            run_id=f"go2o_qsp_{cfg['run_prefix']}_s0_i{i}") for i in range(COUNT)], production_native_submitted=False))
    return value


def noise_array(item):
    return source_noise(item["matched_pairs"][0])


def verify_row(row, cfg, physical, declaration, item):
    sample = checked_request(row, argparse.Namespace(seed=0, round=0), cfg, physical)
    expected = dict(arm="public_flowplace", role="evaluation", seed=0, round=0, index=item["index"],
        source_kind=KIND, source_sampler=SAMPLER, source_sampler_steps=20, training=False,
        training_feedback_eligible=False, offline_train_eligible=False, evaluation_purpose="development_validation",
        public_eval_declaration_sha256=object_sha(declaration), source_noise_binding=item,
        noise_sha256=item["noise_sha256"], noise_seed=4600000,
        source_checkpoint=str(CHECKPOINT.resolve()), checkpoint_sha256=PUBLIC_SHA,
        run_id=f"go2o_qsp_{cfg['run_prefix']}_s0_i{item['index']}")
    if any(row.get(key) != value for key,value in expected.items()):
        raise ValueError("Released public source/sampler/noise/evaluation binding changed")
    with np.load(sample / "actions.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["noise"], noise_array(item)):
            raise ValueError("Public sampler noise differs from matched saved CUDA array")
    file_binding(sample / "trajectory.pt", row["trajectory_sha256"])
    return sample


def emit(experiment, cfg, physical, declaration, item, problem, model, metric):
    index = item["index"]
    sample = experiment / f"seed_0/round_0/requests/public_flowplace/evaluation/{index}"
    if (sample / "request.json").exists():
        row = read(sample / "request.json")
        verify_row(row, cfg, physical, declaration, item)
        return row
    if (sample / "outcome.json").exists():
        raise ValueError("Cannot overwrite orphan native result")
    sample.mkdir(parents=True, exist_ok=True)
    noise = torch.from_numpy(noise_array(item)).to(device=model.device)
    with torch.no_grad():
        raw, used, trajectory = generate(model, problem, noise=noise, intermediate_every=1)
        fixed = problem.cond.is_ports.flatten().bool()
        if not torch.equal(used, noise) or not torch.equal(raw[fixed], problem.x[fixed]):
            raise ValueError("Official sampler changed explicit noise or fixed ports")
        action = torch.where(fixed[:, None], problem.x, legalize(raw, problem))
        if index == 0:
            repeated, _, _ = generate(model, problem, noise=noise)
            delta = float((raw - repeated).abs().max())
            immutable_json(sample / "determinism.json", dict(raw_max_abs_difference=delta,
                bitwise_equal=torch.equal(raw, repeated)))
            if not torch.equal(raw, repeated):
                raise ValueError("Fixed-noise public sampler non-determinism; see determinism.json; no fallback")
    raw_array, executed = raw.cpu().numpy(), action.cpu().numpy()
    geometry = legality(executed, metric.sizes)
    torch.save(trajectory, sample / "trajectory.pt")
    row = dict(arm="public_flowplace", seed=0, round=0, role="evaluation", index=index,
        graph=cfg["design"], terminal_mask=0, training=False, training_feedback_eligible=False,
        offline_train_eligible=False, evaluation_purpose="development_validation",
        source_kind=KIND, source_sampler=SAMPLER, source_sampler_steps=20,
        source_checkpoint=str(CHECKPOINT.resolve()), checkpoint_sha256=PUBLIC_SHA,
        public_eval_declaration_sha256=object_sha(declaration), source_noise_binding=item,
        config_sha256=physical["config_sha256"], dataset_sha256=cfg["dataset_sha256"],
        cache_sha256=cfg["cache_sha256"], sample_dir=str(sample),
        raw_action=raw_array.tolist(), executed_action=executed.tolist(),
        raw_action_sha256=action_sha(raw_array), action_sha256=action_sha(executed),
        noise_sha256=item["noise_sha256"], noise_seed=4600000, predicted_q=None,
        trajectory_sha256=sha(sample / "trajectory.pt"), geometry=geometry,
        macro_hpwl=float(metric(executed)), raw_macro_hpwl=float(metric(raw_array)),
        legalizer_jump_rms=float((action[~fixed]-raw[~fixed]).square().mean().sqrt()),
        stage="global_route_only", run_id=f"go2o_qsp_{cfg['run_prefix']}_s0_i{index}")
    np.savez_compressed(sample / "actions.npz", noise=used.cpu().numpy(), raw_action=raw_array, action=executed)
    if geometry["legal"]:
        save_sample(sample, action, used, problem, row)
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
    verify_row(row, cfg, physical, declaration, item)
    return row


def export(experiment, cfg, physical, declaration):
    folder = experiment / "seed_0/round_0/requests"
    if (folder / "completed.json").exists():
        marker = read(folder / "completed.json")
        if (marker.get("public_eval_declaration_sha256") != object_sha(declaration) or
                marker.get("selection") != SELECTION or marker.get("independent_public_checkpoints") != 1 or
                marker.get("sampler_config_sha256") != sha(experiment / "public_sampler_config.json")):
            raise ValueError("Changed completed public export/sampler binding")
        rows, _ = load_requests(experiment, 0, 0)
        if [r["index"] for r in rows] != list(range(COUNT)):
            raise ValueError("Changed public evaluation denominator")
        for row,item in zip(rows, declaration["noises"]):
            verify_row(row, cfg, physical, declaration, item)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Public inference requires an allocated Slurm CUDA GPU")
    problem = load_problem(cfg["design"], "cuda")
    model, sampler_cfg = load_model(problem, device="cuda", steps=20, hard_constraint=True, w_reg=.1, w_dist=1.)
    immutable_json(experiment / "public_sampler_config.json", OmegaConf.to_container(sampler_cfg, resolve=True))
    metric = MacroHPWL(problem.benchmark_info)
    rows = [emit(experiment, cfg, physical, declaration, item, problem, model, metric) for item in declaration["noises"]]
    tasks = folder / "tasks.jsonl"
    tmp = tasks.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
    tmp.replace(tasks)
    atomic_json(folder / "completed.json", dict(status="completed", count=8, tasks_sha256=sha(tasks),
        config_sha256=physical["config_sha256"], public_eval_declaration_sha256=object_sha(declaration),
        sampler_config_sha256=sha(experiment / "public_sampler_config.json"), selection=SELECTION,
        evaluation_never_replayed=True, independent_public_checkpoints=1, backend_calls=0,
        slurm_job_id=os.environ["SLURM_JOB_ID"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=("declare", "export"), required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU declaration and GPU inference must run through Slurm")
    experiment = args.experiment.resolve()
    with (experiment / ".public_flow_eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, physical, pairs = settings(experiment)
        declaration = declare(experiment, cfg, physical, pairs)
        if args.stage == "export":
            torch.use_deterministic_algorithms(True)
            export(experiment, cfg, physical, declaration)
        print(json.dumps(dict(status=args.stage+"_completed", requests=8, public_checkpoints=1,
            source_kind=KIND, training_or_replay_eligible=False, backend_calls=0)), flush=True)


if __name__ == "__main__":
    main()
