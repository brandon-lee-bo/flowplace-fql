"""Matched development-only evaluation of the CURRENT FQL behavior flow.

Use the actual, saved CUDA noise arrays from matched BC/FQL round-zero exports.
Never regenerate noises, train a policy, select layouts, or add rows to replay.
This is common['flow'] + Euler10 + final legalizer, NOT released FlowPlace20hard.
The 24-request declaration is sealed before any layout or backend metric exists.
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
from baseline.api import RUNTIME, legalize, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .qstudy_actor_report import check_training
from .qstudy_backend import checked_request, context
from .qstudy_data import action_sha
from .qstudy_teacher import build_flow, file_binding, object_sha, source_files

ROOT = Path(__file__).resolve().parents[1]
SOURCE_KIND = "current_fql_behavior_flow_evaluation_only"
SAMPLER = "common_flow_euler10_no_intermediate_constraints_final_legalize"
SEEDS = [0, 1, 2]
COUNT = 8
CONTEXT_FIELDS = ("design", "dataset_path", "dataset_sha256", "cache_sha256",
                  "evaluation_noise_seed", "evaluation_samples", "flow_steps",
                  "online_flow_update", "physical_failure_reward", "backend_threads")


def read(path):
    return json.loads(Path(path).read_text())


def assert_matching_configs(cfg, left, right):
    for key in CONTEXT_FIELDS:
        if cfg[key] != left[key] or cfg[key] != right[key]:
            raise ValueError("Mismatched BC/FQL/current-flow context: " + key)
    for parent in (left, right):
        if parent["seeds"] != SEEDS:
            raise ValueError("The matched controls require exactly actor seeds 0/1/2")
        for seed in SEEDS:
            for key in ("common_checkpoints", "common_checkpoint_sha256"):
                if parent[key][str(seed)] != cfg[key][str(seed)]:
                    raise ValueError("Matched control has a different common initialization")
    if (cfg["flow_steps"] != 10 or cfg["online_flow_update"] is not False or
            cfg["evaluation_samples"] != COUNT or cfg["evaluation_noise_seed"] != 4600000 or
            cfg["evaluation_purpose"] != "development_validation" or
            cfg.get("seeds") != SEEDS or cfg.get("native_budget") != 24 or
            cfg.get("backend_concurrency") != 2 or cfg["arm"] != "behavior_flow"):
        raise ValueError("Changed fixed eval-only 3x8 Euler10 baseline declaration")
    if not cfg["run_prefix"].isalnum() or len(cfg["run_prefix"]) > 32:
        raise ValueError("Unsafe current-flow run prefix")
    if left["arm"] != "bc" or left["q_weight"] != 0 or right["q_weight"] <= 0:
        raise ValueError("Expected a genuine BC and Q-optimized matched control")


def matched_noise(left, right, left_row, right_row):
    """Compare actual arrays; matching generator integers alone is insufficient."""
    a = np.asarray(left)
    b = np.asarray(right)
    if (a.dtype != np.float32 or b.dtype != np.float32 or a.ndim != 2 or
            a.shape[1] != 2 or a.shape != b.shape or not np.isfinite(a).all() or
            np.any(np.abs(a) > 1) or not np.array_equal(a, b)):
        raise ValueError("Actual BC/FQL evaluation noises differ or are invalid")
    digest = action_sha(a)
    if any(r["noise_sha256"] != digest or r["noise_seed"] != 4600000
           for r in (left_row, right_row)):
        raise ValueError("Saved noise array does not match its source request")
    return digest


def settings(experiment):
    cfg, verification, physical = context(experiment)
    controls = []
    for name in ("bc_experiment", "fql_experiment"):
        folder = Path(cfg[name]).resolve()
        file_binding(folder / "config.json", cfg[name + "_config_sha256"])
        controls.append(read(folder / "config.json"))
    assert_matching_configs(cfg, *controls)
    file_binding(RUNTIME / "benchmark_cache" / (cfg["design"] + ".pt"), cfg["cache_sha256"])
    for seed in SEEDS:
        file_binding(cfg["common_checkpoints"][str(seed)], cfg["common_checkpoint_sha256"][str(seed)])
    return cfg, physical, controls


def source_pairs(cfg, controls):
    """Source-only validation: no outcomes, native PPA, or labels are consulted."""
    pairs = []
    cross_seed = {}
    for seed in SEEDS:
        parents = []
        for name, pcfg in zip(("bc_experiment", "fql_experiment"), controls):
            experiment = Path(cfg[name]).resolve()
            folder = experiment / f"seed_{seed}/round_0"
            training = check_training(folder, pcfg, seed, 0, sha(experiment / "config.json"), {})
            rows, tasks = load_requests(experiment, seed, 0)
            selected = sorted((r for r in rows if r["role"] == "evaluation"), key=lambda r: r["index"])
            if len(selected) != COUNT or [r["index"] for r in selected] != list(range(COUNT)):
                raise ValueError("Matched parent evaluation plan differs from eight fixed noises")
            backend_cfg, _, backend_binding = context(experiment)
            checked = []
            for row in selected:
                sample = checked_request(row, argparse.Namespace(seed=seed, round=0), backend_cfg, backend_binding)
                if (row["checkpoint_sha256"] != training["checkpoint_sha256"] or
                        row["arm"] != pcfg["arm"] or row["training_feedback_eligible"]):
                    raise ValueError("Source noise is not from the completed matched eval-only actor")
                with np.load(sample / "actions.npz", allow_pickle=False) as arrays:
                    noise = arrays["noise"].copy()
                binding = dict(experiment=str(experiment), arm=pcfg["arm"], seed=seed, round=0,
                    index=row["index"], sample_dir=str(sample),
                    config=file_binding(experiment / "config.json"), tasks=file_binding(tasks),
                    export_completion=file_binding(tasks.with_name("completed.json")),
                    request=file_binding(sample / "request.json"),
                    actions_npz=file_binding(sample / "actions.npz"),
                    actor_checkpoint=file_binding(training["checkpoint"], training["checkpoint_sha256"]),
                    actor_completion=file_binding(folder / "completed.json"),
                    actor_training_binding=file_binding(folder / "binding.json"),
                    common_checkpoint=file_binding(cfg["common_checkpoints"][str(seed)],
                        cfg["common_checkpoint_sha256"][str(seed)]))
                checked.append((noise, row, binding))
            parents.append(checked)
        for index, (left, right) in enumerate(zip(*parents)):
            digest = matched_noise(left[0], right[0], left[1], right[1])
            if index in cross_seed and cross_seed[index] != digest:
                raise ValueError("Actual evaluation noise differs across the declared actor seeds")
            cross_seed[index] = digest
            pairs.append(dict(seed=seed, index=index, noise_sha256=digest,
                noise_shape=list(left[0].shape), source_bc=left[2], source_fql=right[2]))
    if len(set(cross_seed.values())) != COUNT:
        raise ValueError("The eight development evaluation noises are not distinct")
    return pairs


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("Changed immutable evaluation declaration: " + str(path))
    else:
        atomic_json(path, value)


def declare(experiment, cfg, physical, controls):
    pairs = source_pairs(cfg, controls)
    files = source_files()
    files["fql/qstudy_flow_eval.py"] = Path(__file__)
    files["fql/qstudy_flow_eval.slurm"] = ROOT / "fql/qstudy_flow_eval.slurm"
    declaration = dict(version=1, source_kind=SOURCE_KIND, source_state_key="flow",
        source_sampler=SAMPLER, expected_requests=24, seeds=SEEDS, indices_per_seed=list(range(COUNT)),
        role="evaluation", stage="global_route_only", evaluation_purpose="development_validation",
        source_actual_noise="copied verbatim from matched completed BC/FQL round-zero actions.npz; never regenerated",
        noise_generator_original_device="cuda", source_noise_seed=4600000,
        cross_seed_actual_noise_identical=True, distinct_noise_arrays=8,
        common_checkpoints={str(s): file_binding(cfg["common_checkpoints"][str(s)],
            cfg["common_checkpoint_sha256"][str(s)]) for s in SEEDS},
        flow_steps=10, intermediate_hard_constraints=False, intermediate_coordinate_clipping=False,
        final_coordinate_clipping=True, fixed_ports_restored=True,
        legalizer=dict(w_reg=.1, w_dist=1., grid_res=128, padding=.001),
        training_feedback_eligible=False, offline_train_eligible=False, terminal_mask=0,
        selection="all predeclared seeds and all eight actual noises; no quality or Q selection",
        backend_budget=24, backend_concurrency=2, physical_context=physical,
        round_reuse="frozen per-seed flow and noises; do not duplicate identical round1/2 evaluations",
        later_DRT_subset="first four of eight indices per seed only if separately authorized; no DRT submitted",
        public_FlowPlace_baseline=False,
        source_sha256={n: sha(p) for n, p in sorted(files.items())}, pairs=pairs)
    archive = experiment / "flow_eval_sources"
    archive.mkdir(exist_ok=True)
    for name, path in files.items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            file_binding(target, declaration["source_sha256"][name])
        else:
            shutil.copyfile(path, target)
    immutable_json(experiment / "declaration.json", declaration)
    dispatch = [dict(experiment=str(experiment), seed=p["seed"], round=0, index=p["index"],
        role="evaluation", noise_sha256=p["noise_sha256"], source_kind=SOURCE_KIND,
        run_id=f"go2o_qsf_{cfg['run_prefix']}_s{p['seed']}_i{p['index']}") for p in pairs]
    immutable_json(experiment / "dispatch.json", dict(declaration_sha256=sha(experiment / "declaration.json"),
        count=24, requests=dispatch, production_native_submitted=False))
    return declaration


def source_noise(pair):
    binding = pair["source_bc"]["actions_npz"]
    file_binding(binding["path"], binding["sha256"])
    with np.load(binding["path"], allow_pickle=False) as arrays:
        noise = arrays["noise"].copy()
    if action_sha(noise) != pair["noise_sha256"] or list(noise.shape) != pair["noise_shape"]:
        raise ValueError("Saved declared evaluation noise changed")
    return noise


def verify_row(row, args, cfg, physical, declaration, pair):
    sample = checked_request(row, args, cfg, physical)
    expected = dict(arm="behavior_flow", role="evaluation", seed=pair["seed"], round=0,
        index=pair["index"], source_kind=SOURCE_KIND, source_sampler=SAMPLER,
        source_state_key="flow", source_sampler_steps=10,
        training_feedback_eligible=False, offline_train_eligible=False,
        flow_eval_declaration_sha256=object_sha(declaration),
        source_noise_binding=pair, noise_sha256=pair["noise_sha256"], noise_seed=4600000,
        source_checkpoint=cfg["common_checkpoints"][str(pair["seed"])],
        checkpoint_sha256=cfg["common_checkpoint_sha256"][str(pair["seed"])],
        run_id=f"go2o_qsf_{cfg['run_prefix']}_s{pair['seed']}_i{pair['index']}")
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Current-flow evaluation source/sampler/noise/eligibility binding changed")
    with np.load(sample / "actions.npz", allow_pickle=False) as arrays:
        if not np.array_equal(arrays["noise"], source_noise(pair)):
            raise ValueError("Current flow no longer uses the source's actual CUDA noise array")
    return sample


def emit(experiment, cfg, physical, declaration, pair, problem, flow, metric):
    seed, index = pair["seed"], pair["index"]
    args = argparse.Namespace(seed=seed, round=0)
    sample = experiment / f"seed_{seed}/round_0/requests/behavior_flow/evaluation/{index}"
    if (sample / "request.json").exists():
        row = read(sample / "request.json")
        verify_row(row, args, cfg, physical, declaration, pair)
        return row
    if (sample / "outcome.json").exists():
        raise ValueError("Cannot overwrite orphan backend result")
    sample.mkdir(parents=True, exist_ok=True)
    noise = torch.from_numpy(source_noise(pair)).to(device="cuda")
    with torch.no_grad():
        raw = flow.sample_flow(noise[None], steps=10)[0]
        action = legalize(raw, problem)
        if index == 0:
            repeated = flow.sample_flow(noise[None], steps=10)[0]
            if not torch.equal(raw, repeated) or not torch.equal(action, legalize(repeated, problem)):
                raise ValueError("Fixed-noise behavior flow/final legalizer is not deterministic")
        if not torch.equal(raw[~flow.movable], flow.fixed[~flow.movable]):
            raise ValueError("Current behavior flow moved fixed ports")
    raw_array, executed = raw.cpu().numpy(), action.cpu().numpy()
    geometry = legality(executed, metric.sizes)
    row = dict(arm="behavior_flow", seed=seed, round=0, role="evaluation", index=index,
        graph=cfg["design"], terminal_mask=0, training_feedback_eligible=False,
        offline_train_eligible=False, evaluation_purpose="development_validation",
        source_kind=SOURCE_KIND, source_sampler=SAMPLER, source_state_key="flow",
        source_sampler_steps=10, intermediate_hard_constraints=False,
        intermediate_coordinate_clipping=False, final_coordinate_clipping=True,
        source_checkpoint=cfg["common_checkpoints"][str(seed)],
        checkpoint_sha256=cfg["common_checkpoint_sha256"][str(seed)],
        flow_eval_declaration_sha256=object_sha(declaration), source_noise_binding=pair,
        config_sha256=physical["config_sha256"], dataset_sha256=cfg["dataset_sha256"],
        cache_sha256=cfg["cache_sha256"], sample_dir=str(sample),
        raw_action=raw_array.tolist(), executed_action=executed.tolist(),
        raw_action_sha256=action_sha(raw_array), action_sha256=action_sha(executed),
        noise_sha256=pair["noise_sha256"], noise_seed=4600000, predicted_q=None,
        geometry=geometry, macro_hpwl=float(metric(executed)), raw_macro_hpwl=float(metric(raw_array)),
        legalizer_jump_rms=float((action[flow.movable]-raw[flow.movable]).square().mean().sqrt()),
        stage="global_route_only", run_id=f"go2o_qsf_{cfg['run_prefix']}_s{seed}_i{index}")
    np.savez_compressed(sample / "actions.npz", noise=noise.cpu().numpy(), raw_action=raw_array, action=executed)
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
    verify_row(row, args, cfg, physical, declaration, pair)
    return row


def export(experiment, cfg, physical, declaration):
    for seed in SEEDS:
        pairs = [p for p in declaration["pairs"] if p["seed"] == seed]
        folder = experiment / f"seed_{seed}/round_0/requests"
        if (folder / "completed.json").exists():
            rows, _ = load_requests(experiment, seed, 0)
            if [r["index"] for r in rows] != list(range(COUNT)):
                raise ValueError("Changed evaluation export denominator")
            for row, pair in zip(rows, pairs):
                verify_row(row, argparse.Namespace(seed=seed, round=0), cfg, physical, declaration, pair)
            continue
        local = dict(cfg, common_checkpoint=cfg["common_checkpoints"][str(seed)])
        problem, flow = build_flow(local)
        metric = MacroHPWL(problem.benchmark_info)
        rows = [emit(experiment, cfg, physical, declaration, p, problem, flow, metric) for p in pairs]
        tasks = folder / "tasks.jsonl"
        tmp = tasks.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
        tmp.replace(tasks)
        atomic_json(folder / "completed.json", dict(status="completed", count=COUNT,
            tasks_sha256=sha(tasks), config_sha256=physical["config_sha256"],
            flow_eval_declaration_sha256=object_sha(declaration),
            selection=declaration["selection"], evaluation_never_replayed=True,
            slurm_job_id=os.environ["SLURM_JOB_ID"]))
        del flow, problem
        torch.cuda.empty_cache()
    atomic_json(experiment / "export_completed.json", dict(status="completed", count=24,
        declaration_sha256=sha(experiment / "declaration.json"),
        config_sha256=physical["config_sha256"], dispatch_sha256=sha(experiment / "dispatch.json"),
        seed_tasks_sha256={str(s): sha(experiment / f"seed_{s}/round_0/requests/tasks.jsonl") for s in SEEDS},
        backend_calls=0, production_native_submitted=False, slurm_job_id=os.environ["SLURM_JOB_ID"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=("declare", "export"), required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Source tensor/noise validation and inference must run under Slurm")
    experiment = args.experiment.resolve()
    with (experiment / ".flow_eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, physical, controls = settings(experiment)
        declaration = declare(experiment, cfg, physical, controls)
        if args.stage == "export":
            torch.use_deterministic_algorithms(True)
            export(experiment, cfg, physical, declaration)
        print(json.dumps(dict(status=args.stage + "_completed", requests=24,
            actual_noise_arrays=8, training_or_replay_eligible=False,
            source_kind=SOURCE_KIND, backend_calls=0)), flush=True)


if __name__ == "__main__":
    main()
