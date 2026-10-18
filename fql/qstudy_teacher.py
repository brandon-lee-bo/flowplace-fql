"""Collect true GRT labels for the current FQL behavior flow's fresh layouts.

This source is common['flow'], integrated with ten explicit Euler steps and no
intermediate hard constraints. It is distinct from both the independent one-step
actor and the original released FlowPlace twenty-step hard-constrained sampler.
Noise families are assigned to offline TRAIN/validation before generation. The
paired random variant never crosses that family boundary. No online replay data
or historical labels are automatically merged into this collection.
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
from .qstudy_data import METRIC_NAMES, _native, action_sha, load_dataset, summarize

ROOT = Path(__file__).resolve().parents[1]
SOURCE_KIND = "current_fql_behavior_flow"
SOURCE_SAMPLER = "fql_bc_flow_euler_10_final_legalize"
SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def read_json(path):
    return json.loads(Path(path).read_text())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_binding(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen teacher input changed: "+str(path))
    return dict(path=str(path), sha256=digest)


def settings(experiment):
    cfg, verification, physical = context(experiment)
    for key in ["noise_seed", "train_families", "val_families"]:
        if not isinstance(cfg[key], int) or cfg[key] < (0 if key == "noise_seed" else 1):
            raise ValueError("Invalid predeclared teacher "+key)
    if (cfg["train_families"]+cfg["val_families"] > 256 or
            not SAFE_NAME.fullmatch(cfg["run_prefix"])):
        raise ValueError("Invalid bounded teacher collection size/run prefix")
    if (cfg.get("flow_steps", 10) != 10 or
            cfg.get("variants", ["base", "fixed_random"]) != ["base", "fixed_random"] or
            not math.isfinite(cfg["radius_rms"]) or not 0 < cfg["radius_rms"] <= .25):
        raise ValueError("Teacher source requires Euler10 and base/fixed-random paired variants")
    if cfg.get("common_state_key", "flow") != "flow":
        raise ValueError("An actor checkpoint is not the behavior-flow source")
    file_binding(cfg["dataset_path"], cfg["dataset_sha256"])
    file_binding(RUNTIME/"benchmark_cache"/(cfg["design"]+".pt"), cfg["cache_sha256"])
    file_binding(cfg["common_checkpoint"], cfg["common_checkpoint_sha256"])
    return cfg, verification, physical


def source_files():
    # Imported project files are a small in-memory list, not a recursive scan.
    files = {"fql/qstudy_teacher.slurm": ROOT/"fql/qstudy_teacher.slurm",
             "fql/qstudy_teacher.py": Path(__file__)}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name or not str(name).endswith(".py"):
            continue
        path = Path(name).absolute()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if relative.parts[0] != ".venv":
            files[str(relative)] = path
    for name in ["config_eval.yaml", "model/att-gnn.yaml", "model/conv-layer/gat.yaml",
                 "model/size/large.yaml", "guidance/none.yaml", "legalizer/none.yaml",
                 "sc_placer/sgd.yaml"]:
        relative = "upstream/flowplace/flow_matching/configs/"+name
        files[relative] = ROOT/relative
    return files


def freeze(experiment, cfg, physical):
    files = source_files()
    binding = dict(version=1, source_kind=SOURCE_KIND, source_sampler=SOURCE_SAMPLER,
        state_key="flow", integration_steps=10, intermediate_hard_constraints=False,
        intermediate_coordinate_clipping=False, final_coordinate_clipping=True,
        legalizer=dict(w_reg=.1, w_dist=1., grid_res=128, padding=.001),
        prior="uniform[-1,1]", generator_device="cpu", generator_dtype="float32",
        noise_seed=cfg["noise_seed"], direction_seed=cfg.get("direction_seed", cfg["noise_seed"]+1000000),
        radius_rms=cfg["radius_rms"], variants=["base", "fixed_random"],
        train_families=cfg["train_families"], val_families=cfg["val_families"],
        expected_requests=2*(cfg["train_families"]+cfg["val_families"]),
        common_checkpoint=file_binding(cfg["common_checkpoint"], cfg["common_checkpoint_sha256"]),
        graph_cache=file_binding(RUNTIME/"benchmark_cache"/(cfg["design"]+".pt"), cfg["cache_sha256"]),
        construction_pretrained_checkpoint=file_binding(CHECKPOINT),
        reference_dataset=file_binding(cfg["dataset_path"], cfg["dataset_sha256"]),
        reference_dataset_manifest=file_binding(Path(cfg["dataset_path"]).with_name("manifest.json")),
        reference_rows=file_binding(Path(cfg["dataset_path"]).with_name("rows.jsonl")),
        backend_context=physical, config_sha256=sha(experiment/"config.json"),
        online_training_feedback_eligible=False, terminal_mask=0,
        family_namespace_fields=["source_kind", "design", "common_checkpoint_sha256", "noise_seed", "family_index"],
        split_rule="first declared train_families noises TRAIN; remaining val_families validation; paired variants stay together",
        source_sha256={name: sha(path) for name, path in sorted(files.items())},
        packages=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda))
    path = experiment/"teacher_binding.json"
    if path.exists() and read_json(path) != binding:
        raise ValueError("Refusing changed teacher source/configuration/data in existing experiment")
    archive = experiment/"teacher_sources"
    archive.mkdir(exist_ok=True)
    for name, source in files.items():
        target = archive/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != binding["source_sha256"][name]:
                raise ValueError("Frozen teacher source snapshot changed: "+name)
        else:
            shutil.copyfile(source, target)
    if not path.exists():
        atomic_json(path, binding)
    return binding


def build_flow(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("Behavior-flow sampling requires an allocated Slurm GPU")
    problem = load_problem(cfg["design"], "cuda")
    saved = torch.load(cfg["common_checkpoint"], map_location="cuda", weights_only=False)
    if not all(key in saved for key in ["flow", "actor"]):
        raise ValueError("Expected a completed matching common flow/actor warmup")
    flow = GraphPolicy(problem).eval()
    state = saved["flow"]
    if (not torch.equal(state["movable"].bool(), flow.movable) or
            not torch.equal(state["fixed"], flow.fixed)):
        raise ValueError("Common behavior flow belongs to different graph coordinates/movable nodes")
    flow.load_state_dict(state, strict=True)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    return problem, flow


def rms(value, movable):
    return float(value[movable].square().mean().sqrt())


def family_inputs(cfg, flow, node_count):
    generator = torch.Generator(device="cpu").manual_seed(cfg["noise_seed"])
    direction_generator = torch.Generator(device="cpu").manual_seed(cfg.get("direction_seed", cfg["noise_seed"]+1000000))
    for index in range(cfg["train_families"]+cfg["val_families"]):
        noise = (2*torch.rand(node_count, 2, generator=generator)-1).cuda()
        vector = torch.randn(node_count, 2, generator=direction_generator).cuda()*flow.movable[:, None]
        norm = vector[flow.movable].square().mean().sqrt()
        if not torch.isfinite(norm) or norm <= 0 or not torch.isfinite(vector).all():
            raise ValueError("Cannot fabricate a zero/nonfinite fixed perturbation direction")
        yield index, noise, vector/norm


def verify_teacher_request(row, args, cfg, physical, binding):
    sample = checked_request(row, args, cfg, physical)
    split = "train" if row["family_index"] < cfg["train_families"] else "validation"
    role = "data_train" if split == "train" else "data_validation"
    namespace = dict(source_kind=SOURCE_KIND, design=cfg["design"],
        common_checkpoint_sha256=cfg["common_checkpoint_sha256"], noise_seed=cfg["noise_seed"],
        family_index=row["family_index"])
    expected = dict(source_kind=SOURCE_KIND, source_sampler=SOURCE_SAMPLER,
        source_state_key="flow", teacher_binding_sha256=object_sha(binding),
        split=split, role=role, training_feedback_eligible=False,
        offline_train_eligible=split == "train", family=cfg["noise_seed"]+row["family_index"],
        source_family_namespace=namespace, source_family_sha256=object_sha(namespace),
        source_checkpoint=binding["common_checkpoint"]["path"], checkpoint_sha256=cfg["common_checkpoint_sha256"],
        source_sampler_steps=10, intermediate_hard_constraints=False)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Teacher request changed source, predeclared split or TRAIN eligibility")
    if (row["variant"] not in [0, 1] or row["index"] != 2*row["family_index"]+row["variant"] or
            row["variant_kind"] != binding["variants"][row["variant"]]):
        raise ValueError("Teacher family/variant identity changed")
    with np.load(sample/"actions.npz", allow_pickle=False) as arrays:
        if (action_sha(arrays["direction"]) != row["direction_sha256"] or
                action_sha(arrays["base_raw_action"]) != row["base_raw_action_sha256"]):
            raise ValueError("Teacher base or paired perturbation changed")
    return sample


def emit(args, cfg, physical, binding, problem, metric, flow, family_index, noise,
         direction, base_raw, base_action, variant):
    split = "train" if family_index < cfg["train_families"] else "validation"
    role = "data_train" if split == "train" else "data_validation"
    index = family_index*2+variant
    sample = args.experiment/f"seed_0/round_0/requests/teacher_flow/{role}/{index}"
    with torch.no_grad():
        raw = (base_raw if variant == 0 else
               flow.restore((base_raw+cfg["radius_rms"]*direction)[None], clip=True)[0])
    raw_array = raw.cpu().numpy()
    noise_array = noise.cpu().numpy()
    expected = dict(index=index, family_index=family_index, variant=variant,
        raw_action_sha256=action_sha(raw_array), noise_sha256=action_sha(noise_array),
        direction_sha256=action_sha(direction.cpu().numpy()), base_raw_action_sha256=action_sha(base_raw.cpu().numpy()))
    path = sample/"request.json"
    if path.exists():
        row = read_json(path)
        verify_teacher_request(row, args, cfg, physical, binding)
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("Interrupted teacher export changed its deterministic proposal/noise")
        return row
    sample.mkdir(parents=True, exist_ok=True)
    if (sample/"outcome.json").exists():
        raise ValueError("Orphan backend outcome cannot be overwritten")
    with torch.no_grad():
        action = base_action if variant == 0 else legalize(raw, problem)
    executed = action.cpu().numpy()
    geometry = legality(executed, metric.sizes)
    namespace = dict(source_kind=SOURCE_KIND, design=cfg["design"],
        common_checkpoint_sha256=cfg["common_checkpoint_sha256"], noise_seed=cfg["noise_seed"], family_index=family_index)
    row = dict(**expected, arm="teacher_flow", seed=0, round=0, graph=cfg["design"],
        source_kind=SOURCE_KIND, source_sampler=SOURCE_SAMPLER, source_state_key="flow",
        source_sampler_steps=10, intermediate_hard_constraints=False,
        intermediate_coordinate_clipping=False, final_coordinate_clipping=True,
        source_family_namespace=namespace, source_family_sha256=object_sha(namespace),
        variant_kind=binding["variants"][variant], family=cfg["noise_seed"]+family_index,
        split=split, role=role, terminal_mask=0, training_feedback_eligible=False,
        offline_train_eligible=split == "train", sample_dir=str(sample),
        source_checkpoint=binding["common_checkpoint"]["path"], checkpoint_sha256=cfg["common_checkpoint_sha256"],
        teacher_binding_sha256=object_sha(binding), config_sha256=physical["config_sha256"],
        dataset_sha256=cfg["dataset_sha256"], cache_sha256=cfg["cache_sha256"],
        raw_action=raw_array.tolist(), executed_action=executed.tolist(), action_sha256=action_sha(executed),
        predicted_q=None, noise_seed=cfg["noise_seed"], direction_seed=binding["direction_seed"],
        radius_rms=0. if variant == 0 else cfg["radius_rms"],
        actual_raw_step_rms=rms(raw-base_raw, flow.movable),
        executed_step_rms=rms(action-base_action, flow.movable),
        legalizer_jump_rms=rms(action-raw, flow.movable),
        geometry=geometry, macro_hpwl=float(metric(executed)), raw_macro_hpwl=float(metric(raw_array)),
        stage="global_route_only", run_id=f"go2o_qst_{cfg['run_prefix']}_f{family_index}_v{variant}")
    np.savez_compressed(sample/"actions.npz", noise=noise_array, raw_action=raw_array, action=executed,
        direction=direction.cpu().numpy(), base_raw_action=base_raw.cpu().numpy())
    if geometry["legal"]:
        save_sample(sample, action, noise, problem, row)
        output = sample/(cfg["design"]+".def")
        # Rebuild only an unsealed partial export. Sealed requests returned above.
        output.unlink(missing_ok=True)
        output.with_suffix(".audit.json").unlink(missing_ok=True)
        audit = convert(cfg["design"], sample/"macro_physical.json", output)
        row.update(def_path=str(output), def_sha256=sha(output), rounded_geometry=audit["legal_no_halo"])
        if not audit["legal_no_halo"]:
            row["pre_backend_failure"] = "rounded_DEF_geometry"
    else:
        row["pre_backend_failure"] = "normalized_geometry"
    atomic_json(path, row)
    verify_teacher_request(row, args, cfg, physical, binding)
    return row


def check(args, cfg, binding):
    problem, flow = build_flow(cfg)
    _, noise, direction = next(family_inputs(cfg, flow, len(problem.x)))
    with torch.no_grad():
        first = flow.sample_flow(noise[None], steps=10)[0]
        second = flow.sample_flow(noise[None], steps=10)[0]
        baseline = legalize(first, problem)
        repeated = legalize(second, problem)
        perturbation = flow.restore((first+cfg["radius_rms"]*direction)[None], clip=True)[0]
        legal_perturbation = legalize(perturbation, problem)
    if not torch.equal(first, second) or not torch.equal(baseline, repeated):
        raise ValueError("Behavior flow or final legalizer is not deterministic on repeated fixed noise")
    if (not torch.equal(first[~flow.movable], flow.fixed[~flow.movable]) or
            not torch.equal(perturbation[~flow.movable], flow.fixed[~flow.movable])):
        raise ValueError("Flow or perturbation moved fixed port coordinates")
    metric = MacroHPWL(problem.benchmark_info)
    result = dict(status="passed", teacher_binding_sha256=object_sha(binding),
        source_kind=SOURCE_KIND, source_sampler=SOURCE_SAMPLER, source_state_key="flow",
        repeated_raw_equal=True, repeated_legalized_equal=True, fixed_ports_preserved=True,
        noise_sha256=action_sha(noise.cpu().numpy()), raw_action_sha256=action_sha(first.cpu().numpy()),
        action_sha256=action_sha(baseline.cpu().numpy()),
        baseline_geometry=legality(baseline.cpu().numpy(), metric.sizes),
        perturbation_geometry=legality(legal_perturbation.cpu().numpy(), metric.sizes),
        actual_raw_step_rms=rms(perturbation-first, flow.movable),
        legalizer_jump_rms=rms(baseline-first, flow.movable),
        requests_exported=0, backend_calls=0, slurm_job_id=os.environ["SLURM_JOB_ID"])
    atomic_json(args.experiment/"teacher_check.json", result)
    print(json.dumps(result), flush=True)


def export(args, cfg, physical, binding):
    folder = args.experiment/"seed_0/round_0/requests"
    if (folder/"completed.json").exists():
        rows, _ = load_requests(args.experiment, 0, 0)
        if len(rows) != binding["expected_requests"]:
            raise ValueError("Changed teacher export denominator")
        for row in rows:
            verify_teacher_request(row, args, cfg, physical, binding)
        print(json.dumps(dict(status="already_exported_verified", count=len(rows))), flush=True)
        return
    problem, flow = build_flow(cfg)
    metric = MacroHPWL(problem.benchmark_info)
    rows = []
    for family_index, noise, direction in family_inputs(cfg, flow, len(problem.x)):
        with torch.no_grad():
            base_raw = flow.sample_flow(noise[None], steps=10)[0]
            base_action = legalize(base_raw, problem)
        for variant in [0, 1]:
            rows.append(emit(args, cfg, physical, binding, problem, metric, flow, family_index,
                             noise, direction, base_raw, base_action, variant))
    if (len(rows) != binding["expected_requests"] or
            [r["index"] for r in rows] != list(range(binding["expected_requests"])) or
            len({r["run_id"] for r in rows}) != len(rows)):
        raise ValueError("Incomplete/duplicate teacher export")
    folder.mkdir(parents=True, exist_ok=True)
    tasks = folder/"tasks.jsonl"
    temporary = tasks.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
    temporary.replace(tasks)
    atomic_json(folder/"completed.json", dict(status="completed", count=len(rows),
        tasks_sha256=sha(tasks), config_sha256=physical["config_sha256"],
        teacher_binding_sha256=object_sha(binding), slurm_job_id=os.environ["SLURM_JOB_ID"],
        train_requests=2*cfg["train_families"], validation_requests=2*cfg["val_families"],
        split_rule=binding["split_rule"], selection="all predeclared families/variants; no quality filtering",
        online_replay_eligible=False, source_sampler=SOURCE_SAMPLER))
    print(json.dumps(dict(status="exported", count=len(rows), train=2*cfg["train_families"],
                         validation=2*cfg["val_families"], source_sampler=SOURCE_SAMPLER)), flush=True)


def write_dataset(args, cfg, binding, tasks, rows, outcome_bindings):
    reference, _ = load_dataset(cfg["dataset_path"])
    groups = {split: [row for row in rows if row["split"] == split] for split in ["train", "validation"]}
    if (len(groups["train"]) != 2*cfg["train_families"] or
            len(groups["validation"]) != 2*cfg["val_families"] or
            set(r["family"] for r in groups["train"]) & set(r["family"] for r in groups["validation"])):
        raise ValueError("New teacher dataset changed its family-atomic split")
    collisions = {}
    for key in ["raw_action_sha256", "action_sha256"]:
        collisions[key] = sorted({r[key] for r in groups["train"]} & {r[key] for r in groups["validation"]})
    if any(collisions.values()):
        atomic_json(args.experiment/"teacher_cross_split_collision.json", dict(
            status="blocked_dataset_export_coordinate_collision", exact_cross_split_collisions=collisions,
            note="All native requests/outcomes remain intact; no resampling or silent row dropping"))
        raise ValueError("Collapsed teacher coordinates cross TRAIN/validation; do not silently claim independence")
    source_bindings = dict(teacher_binding_sha256=object_sha(binding), config_sha256=sha(args.experiment/"config.json"),
        tasks_sha256=sha(tasks), outcomes=outcome_bindings,
        reference_dataset=binding["reference_dataset"], teacher_common=binding["common_checkpoint"])
    folder = args.experiment/"dataset"
    if folder.exists():
        _, manifest = load_dataset(folder)
        if manifest["source_bindings"] != source_bindings:
            raise ValueError("Existing teacher dataset's measured/source inputs changed")
        return manifest
    staging = args.experiment/"dataset.building"
    staging.mkdir(exist_ok=True)
    arrays, ordered = {}, []
    for split, selected in groups.items():
        metrics = np.asarray([[np.nan if row["metrics"][name] is None else row["metrics"][name]
                               for name in METRIC_NAMES] for row in selected], dtype=np.float64)
        fields = dict(raw_action=np.asarray([row["raw_action"] for row in selected], dtype=np.float32),
            executed_action=np.asarray([row["executed_action"] for row in selected], dtype=np.float32),
            metrics=metrics, metric_mask=np.isfinite(metrics),
            family=np.asarray([row["family"] for row in selected], dtype=np.int64),
            variant=np.asarray([row["variant"] for row in selected], dtype=np.int64),
            source_row=np.asarray([row["source_row"] for row in selected], dtype=np.int64),
            status=np.asarray([row["status"] for row in selected], dtype="U24"),
            provenance_index=np.arange(len(ordered), len(ordered)+len(selected), dtype=np.int64))
        arrays.update({split+"_"+name: value for name, value in fields.items()})
        ordered.extend(selected)
    # The intervention is critic-label source only. Preserve the reference's
    # exact TRAIN behavior targets, rather than implicitly changing the BC task.
    arrays["behavior_action"] = reference["behavior_action"]
    arrays["behavior_family"] = reference["behavior_family"]
    np.savez_compressed(staging/"dataset.npz", **arrays)
    (staging/"rows.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in ordered))
    revisions = {}
    for row in ordered:
        if row["backend"]:
            revision = row["backend"]["physical_flow_revision"]
            revisions[revision] = revisions.get(revision, 0)+1
    manifest = dict(version=1, design=cfg["design"], metric_names=list(METRIC_NAMES),
        metric_units=["um", "ns", "ns", "positive_demand_minus_capacity", "positive_demand_minus_capacity", "W", "um2"],
        stage="global_route_only", not_final_table2=True, detailed_route_DRC=None,
        dataset_sha256=sha(staging/"dataset.npz"), rows_sha256=sha(staging/"rows.jsonl"),
        source_kind=SOURCE_KIND, source_sampler=SOURCE_SAMPLER, source_bindings=source_bindings,
        split_counts={name: len(selected) for name, selected in groups.items()},
        no_test_or_heldout_exported=True, historical_online_included=False,
        validation_status="Fresh teacher-flow noise families; designated tuning validation, not a final heldout test",
        family_namespace_fields=binding["family_namespace_fields"],
        behavior_targets=len(arrays["behavior_action"]),
        behavior_source="Unchanged audited reference TRAIN behavior targets; this intervention changes Q labels only",
        executed_action_semantics="normalized final-legalizer output before DEF rounding/native track snapping; actual backend boxes are separate native geometry evidence",
        physical_flow_revisions=revisions, exact_cross_split_collisions=collisions,
        summary={name: summarize(selected) for name, selected in groups.items()},
        slurm_job_id=os.environ["SLURM_JOB_ID"], assembly_source_sha256=sha(__file__))
    atomic_json(staging/"manifest.json", manifest)
    load_dataset(staging)
    staging.replace(folder)
    return manifest


def collect(args, cfg, verification, physical, binding):
    requests, tasks = load_requests(args.experiment, 0, 0)
    if len(requests) != binding["expected_requests"]:
        raise ValueError("Changed teacher request denominator")
    rows, missing, outcomes = [], [], []
    for request in requests:
        sample = verify_teacher_request(request, args, cfg, physical, binding)
        path = sample/"outcome.json"
        if not path.exists():
            missing.append(request["index"])
            continue
        outcome = read_json(path)
        checked_existing(outcome, request, sample, tasks, physical, cfg, verification)
        digest = sha(path)
        outcome.update(outcome_path=str(path), outcome_sha256=digest)
        row = _native(outcome, cfg["design"], request["raw_action"], request["executed_action"],
            request["split"], request["family"], request["variant"], request["index"],
            SOURCE_KIND, tasks, sha(tasks))
        row.update(source_kind=SOURCE_KIND, source_sampler=SOURCE_SAMPLER,
            source_state_key="flow", source_sampler_steps=10,
            source_family_namespace=request["source_family_namespace"],
            source_family_sha256=request["source_family_sha256"],
            noise_sha256=request["noise_sha256"], source_checkpoint=request["source_checkpoint"],
            checkpoint_sha256=request["checkpoint_sha256"],
            online_training_feedback_eligible=False, offline_train_eligible=request["offline_train_eligible"],
            family_index=request["family_index"], variant_kind=request["variant_kind"],
            direction_sha256=request["direction_sha256"], raw_step_rms=request["actual_raw_step_rms"],
            legalizer_jump_rms=request["legalizer_jump_rms"], request_path=str(sample/"request.json"),
            outcome_path=str(path), outcome_sha256=digest)
        rows.append(row)
        outcomes.append(dict(path=str(path), sha256=digest))
    result = dict(status="waiting_for_native_labels" if missing else "all_requested_accounted",
        source_kind=SOURCE_KIND, source_sampler=SOURCE_SAMPLER, requested=len(requests),
        measured=len(rows), missing_request_indices=missing,
        successful=sum(row["status"] == "success" for row in rows),
        failed=sum(row["status"] != "success" for row in rows),
        teacher_binding_sha256=object_sha(binding), tasks_sha256=sha(tasks),
        slurm_job_id=os.environ["SLURM_JOB_ID"], automatic_merge=False)
    if not missing:
        manifest = write_dataset(args, cfg, binding, tasks, rows, outcomes)
        result.update(dataset_path=str(args.experiment/"dataset/dataset.npz"),
            dataset_sha256=manifest["dataset_sha256"], split_counts=manifest["split_counts"])
    atomic_json(args.experiment/"teacher_collection.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=["check", "export", "collect"], default="export")
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    args.seed = 0
    args.round = 0
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Teacher inference, tensor loading and native-label assembly require Slurm")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, verification, physical = settings(args.experiment)
    with (args.experiment/".teacher_export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = freeze(args.experiment, cfg, physical)
        if args.stage == "export":
            export(args, cfg, physical, binding)
        elif args.stage == "check":
            check(args, cfg, binding)
        else:
            collect(args, cfg, verification, physical, binding)


if __name__ == "__main__":
    main()
