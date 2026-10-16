"""Export a frozen GRT-trained candidate comparison, without launching backends.

Full tasks include exact reused teacher and best-TRAIN labels. Submit only
pending_tasks.jsonl. Validation precedes sealed heldout; failures never shrink
the requested comparison. Run numerical export inside a Slurm allocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.placement_metrics import legality
from fql.aggregate_backend_candidates import plan_keys, validate_task
from fql.core import Critics, OneStepActor
from fql.provenance import capture_source_provenance, checkpoint_provenance
from fql.physical_failure_policy import bind_policy, validate_bound_policy
from fql.relabel_backend import STAGE, action_sha256, sha256_file, write_json


def numbers(text, convert, label):
    result = [convert(value) for value in text.split(",") if value]
    if not result or len(set(result)) != len(result) or any(not np.isfinite(value) for value in result):
        raise ValueError(f"Invalid {label}")
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def record_binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def check_bindings(bindings):
    for name, item in bindings.items():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"Source artifact changed during/between exports: {name}")


def benchmark_cache_path(design):
    return ROOT / "baseline/runtime/benchmark_cache" / f"{design}.pt"


def prepare_plan(args):
    """Read metadata/arrays and bind all sources before any legalization/outcomes."""
    dataset_dir, training, gate_dir = [Path(value).resolve() for value in
                                      [args.dataset_dir, args.training_dir, args.gate_dir]]
    dataset_path, manifest_path = dataset_dir / "dataset.npz", dataset_dir / "manifest.json"
    manifest, run, gate = [read_json(path) for path in
                           [manifest_path, training / "run.json", gate_dir / "report.json"]]
    dataset_hash = sha256_file(dataset_path)
    if manifest.get("reward_stage") != STAGE or manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("Require a bound real-GRT labeled dataset")
    if gate.get("status") != "completed" or gate.get("dataset_sha256") != dataset_hash or gate.get("stage") != STAGE:
        raise ValueError("Require a completed critic gate for this exact GRT dataset")
    if gate.get("source_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Dataset manifest changed since the gate")
    if run.get("data", {}).get("sha256") != dataset_hash or run["args"].get("seal_heldout") is not True:
        raise ValueError("Training dataset mismatch or heldout diagnostics were not sealed")
    if Path(run["args"]["pretrained_critic_dir"]).resolve() != gate_dir:
        raise ValueError("Training used a different gate")
    if read_json(training / "summary.json").get("status") != "training_completed_external_evaluation_required":
        raise ValueError("Actor training did not complete")
    seeds = numbers(args.seeds, int, "seeds")
    alphas = numbers(args.alphas, float, "alphas")
    if any(seed < 0 for seed in seeds) or any(alpha <= 0 for alpha in alphas):
        raise ValueError("Seeds must be nonnegative and alphas positive")
    if set(seeds) != set(numbers(run["args"]["seeds"], int, "training seeds")):
        raise ValueError("All predeclared training seeds must participate")
    if set(seeds) != {item["seed"] for item in gate["seeds"]}:
        raise ValueError("Export seeds differ from the frozen gate seeds")
    validation_indices = numbers(args.validation_indices, int, "validation indices")
    heldout_indices = numbers(args.heldout_indices, int, "heldout indices")
    with np.load(dataset_path, allow_pickle=False) as archive:
        # Do not load or summarize heldout reward/PPA labels here.
        keys = ["noise", "teacher_action", "split", "action", "raw_action", "critic_split",
                "critic_parent_index", "variant", "source_row_index", "legal"]
        arrays = {key: archive[key].copy() for key in keys}
    for indices, split in [(validation_indices, 1), (heldout_indices, 2)]:
        if any(index < 0 or index >= len(arrays["noise"]) or arrays["split"][index] != split for index in indices):
            raise ValueError("Declared evaluation indices are outside their required split")
    bindings = {name: record_binding(path) for name, path in {
        "dataset": dataset_path, "dataset_manifest": manifest_path,
        "backend_labels": dataset_dir / "backend_labels.jsonl",
        "training_run": training / "run.json", "training_summary": training / "summary.json",
        "gate_report": gate_dir / "report.json", "best_train": gate_dir / "best_train_layout.json",
        "best_train_action": gate_dir / "best_train_layout.npz",
        "legalizer_adapter": ROOT / "baseline/api.py",
        "legalizer": ROOT / "upstream/flowplace/flow_matching/hard_constraint_legalizer.py",
        "def_exporter": ROOT / "backend/scripts/export_macro_def.py"}.items()}
    bindings["benchmark_cache"] = record_binding(benchmark_cache_path(manifest["args"]["design"]))
    policy_path = getattr(args, "physical_failure_policy", None)
    policy_binding = bind_policy(policy_path) if policy_path else None
    if policy_binding is not None:
        bindings["physical_failure_policy"] = {key: policy_binding[key] for key in ["path", "sha256"]}
    best = read_json(gate_dir / "best_train_layout.json")
    if best.get("selection_split") != "train_only" or best.get("dataset_sha256") != dataset_hash:
        raise ValueError("Best-TRAIN baseline must be selected using this dataset's training rows only")
    best_row = np.flatnonzero(arrays["source_row_index"] == best["source_row_index"])
    if len(best_row) != 1 or arrays["critic_split"][best_row[0]] != 0:
        raise ValueError("Best-TRAIN source row is missing or outside training")
    if action_sha256(arrays["action"][best_row[0]]) != best["action_sha256"]:
        raise ValueError("Best-TRAIN selected action differs from the dataset")
    base_methods = [f"seed_{seed}/{method}" for seed in seeds for method in
                    ["distill_only", *[f"fql_alpha_{alpha:g}" for alpha in alphas]]]
    selection = None
    if args.phase == "validation":
        if args.selection:
            raise ValueError("Validation cannot be populated using a post-validation selection")
        if set(alphas) != set(numbers(run["args"]["alphas"], float, "training alphas")):
            raise ValueError("Validation must include every predeclared trained FQL alpha")
        methods, indices = base_methods, validation_indices
    else:
        if not args.selection:
            raise ValueError("Heldout requires a sealed GRT validation selection")
        selection_path = Path(args.selection).resolve()
        selection = read_json(selection_path)
        if selection.get("kind") != "sealed_GRT_validation_selection" or selection.get("selection_stage") != STAGE:
            raise ValueError("Wrong heldout selection kind/stage")
        if selection.get("physical_failure_policy") != policy_binding:
            raise ValueError("Heldout must explicitly bind the identical validation physical-failure policy")
        if sha256_file(selection["source_report"]) != selection.get("source_report_sha256"):
            raise ValueError("Validation report changed after selection")
        if selection.get("dataset_sha256") != dataset_hash or Path(selection["training_dir"]).resolve() != training:
            raise ValueError("Heldout selection is bound to another dataset/training run")
        if selection.get("backend_consistency") != manifest.get("backend_consistency"):
            raise ValueError("Selected validation and training-label backends differ")
        validation_plan = read_json(selection["source_report"])["plan"]
        for name, item in validation_plan["source_bindings"].items():
            if name in bindings and item != bindings[name]:
                raise ValueError(f"Heldout source differs from validation: {name}")
        check_bindings(validation_plan["source_bindings"])
        if sorted(selection["training_seeds"]) != sorted(seeds) or selection["heldout_plan"]["indices"] != heldout_indices:
            raise ValueError("Heldout seeds/noises differ from the predeclared selection")
        methods, indices = selection["methods"], heldout_indices
        if not set(methods) <= set(base_methods):
            raise ValueError("Selected heldout methods are outside the predeclared sweep")
        bindings["selection"] = record_binding(selection_path)
    for seed in seeds:
        expected = checkpoint_provenance(gate_dir / f"seed_{seed}/critic.pt")
        if run["gate_checkpoint_provenance"].get(str(seed)) != expected:
            raise ValueError("Training gate checkpoint SHA changed")
        bindings[f"gate_seed_{seed}"] = expected
    for method in methods:
        for name, filename in [("proposals", "proposals.npz"), ("checkpoint", "checkpoint.pt")]:
            bindings[f"{method}/{name}"] = record_binding(training / method / filename)
    candidates = [{"method": method, "index": index} for method in
                  ["teacher_relegalized", *methods] for index in indices]
    candidates.append({"method": "best_observed_train", "index": best["parent_noise_index"]})
    if selection is not None and plan_keys({"candidates": candidates}) != plan_keys(selection["heldout_plan"]):
        raise ValueError("Heldout candidate keys differ from sealed validation selection")
    plan = {"phase": args.phase, "args": vars(args), "design": manifest["args"]["design"],
            "training_reward_stage": STAGE, "backend_stage": STAGE, "not_final_table2": True,
            "reward_scale_train_only": manifest["reward_scale_train_only"],
            "dataset_sha256": dataset_hash, "training_dir": str(training), "gate_dir": str(gate_dir),
            "source_bindings": bindings, "reference_method": "teacher_relegalized",
            "indices": indices, "validation_indices": validation_indices, "heldout_indices": heldout_indices,
            "training_seeds": seeds, "predeclared_alphas": alphas, "methods": methods, "candidates": candidates,
            "task_ids": [f"{args.prefix}_{args.phase}_{i:03d}" for i in range(len(candidates))],
            "new_candidate_backend_budget": len(methods) * len(indices), "reused_backend_count": len(indices) + 1,
            "backend_consistency": manifest["backend_consistency"],
            "fixed_training_baseline_repeated": False,
            "legalizer": {"calls_per_new_candidate": 1, "grid_res": 128, "padding": .001,
                          "w_reg": .1, "w_dist": 1., "device": args.device,
                          "raw_action": "clipped network proposal; identical to critic input",
                          "reused_baselines": "No additional legalization or backend execution"},
            "freeze_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "script_sha256": sha256_file(__file__),
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "notes": ["Only pending_tasks.jsonl requires backend jobs; full tasks.jsonl includes exact reused references",
                      "No observed outcome is used to exclude, replace or resample a candidate",
                      "Best observed TRAIN is one measured fixed layout, not a per-noise independent replicate",
                      "Actual GRT rWL selects alpha; WNS/TNS/power remain separate diagnostics"]}
    if selection is not None:
        plan.update(selection_path=bindings["selection"]["path"], selection_sha256=bindings["selection"]["sha256"])
    if policy_binding is not None:
        plan["physical_failure_policy"] = policy_binding
        validate_bound_policy(plan, Path(args.output).resolve())
    return plan, arrays, manifest, best


def reused_task(plan, item, task_id, arrays, manifest, best, labels, output):
    method, index = item["method"], item["index"]
    if method == "teacher_relegalized":
        rows = np.flatnonzero((arrays["critic_parent_index"] == index) & (arrays["variant"] == 0))
        role = "paired_reference"
    else:
        rows = np.flatnonzero(arrays["source_row_index"] == best["source_row_index"])
        role = "fixed_training_baseline"
    if len(rows) != 1:
        raise ValueError(f"No unique source backend label for {method}/{index}")
    row = int(rows[0]); source_row = int(arrays["source_row_index"][row])
    source = labels[source_row]
    action = arrays["action"][row].reshape(-1, 2)
    if not arrays["legal"][row] or source["action_sha256"] != action_sha256(action):
        raise ValueError("Reused source action is illegal or hash-mismatched")
    if role == "paired_reference" and not np.array_equal(arrays["raw_action"][row].reshape(-1), arrays["teacher_action"][index].reshape(-1)):
        raise ValueError("Variant zero is not the exact teacher-before-final-legalization input")
    for name, key in [("def", "def_path"), ("metrics", "metrics_path"), ("manifest", "manifest_path")]:
        if sha256_file(source[key]) != source[f"{name}_sha256"]:
            raise ValueError(f"Original reused {name} changed")
    task = {**item, "task_id": task_id, "run_id": source["request_record"].get("run_id"),
            "split": int(arrays["critic_split"][row]), "training_seed": None, "design": plan["design"],
            "role": role, "reused_backend_measurement": True, "requires_backend_execution": False,
            "noise_conditioning": role == "paired_reference",
            "source_row_index": source_row, "dataset_sha256": plan["dataset_sha256"],
            "action_path": plan["source_bindings"]["dataset"]["path"],
            "action_file_sha256": plan["dataset_sha256"], "action_key": "action", "action_row": row,
            "normalized_action_flat": action.reshape(-1).tolist(), "action_sha256": action_sha256(action),
            "def_path": source["def_path"], "def_sha256": source["def_sha256"],
            "metrics_path": source["metrics_path"], "manifest_path": source["manifest_path"],
            "metrics_sha256": source["metrics_sha256"], "manifest_sha256": source["manifest_sha256"],
            "legal": True, "predicted_q": None}
    if role == "fixed_training_baseline":
        task.update(best_train_baseline_path=plan["source_bindings"]["best_train"]["path"],
                    best_train_baseline_sha256=plan["source_bindings"]["best_train"]["sha256"])
    checked = validate_task(task, output)
    if checked["backend_consistency"] != manifest["backend_consistency"]:
        raise ValueError("Reused backend has different implementation provenance")
    return task


def load_proposals(method, plan, arrays, device):
    """Bind proposals to exact actor, frozen gate critic, and paired teacher data."""
    folder = Path(plan["training_dir"]) / method
    checkpoint = torch.load(folder / "checkpoint.pt", map_location=device, weights_only=False)
    seed = int(method.split("/")[0][5:])
    gate_binding = plan["source_bindings"][f"gate_seed_{seed}"]
    if checkpoint.get("gate_checkpoint_provenance") != gate_binding:
        raise ValueError("Actor checkpoint does not bind the same frozen gate")
    if checkpoint["data"]["sha256"] != plan["dataset_sha256"] or checkpoint["seed"] != seed or checkpoint["method"] != method.split("/")[1]:
        raise ValueError("Actor checkpoint dataset/seed/method mismatch")
    gate = torch.load(gate_binding["path"], map_location=device, weights_only=False)
    if gate.get("reward_stage") != STAGE or gate["dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("Gate checkpoint is not for the bound GRT dataset")
    if gate.get("reward_scale_train_only") != plan["reward_scale_train_only"]:
        raise ValueError("Gate checkpoint reward normalization differs from the bound dataset")
    for key, value in gate["critic_state"].items():
        if not torch.equal(value, checkpoint["critic_state"][key]):
            raise ValueError("Actor training modified the supposedly frozen gate critic")
    with np.load(folder / "proposals.npz", allow_pickle=False) as archive:
        proposal = {key: archive[key].copy() for key in ["noise", "teacher_action", "split", "raw_action", "predicted_q"]}
    for name in ["noise", "teacher_action"]:
        expected = arrays[name].reshape(len(arrays[name]), -1)
        if not np.array_equal(proposal[name].reshape(len(expected), -1), expected):
            raise ValueError(f"Proposal {name} is not exactly paired with the source dataset")
    if not np.array_equal(proposal["split"], arrays["split"]):
        raise ValueError("Proposal split changed")
    actor = OneStepActor(checkpoint["dim"], checkpoint["actor_widths"], checkpoint["movable_mask"], checkpoint["fixed_action"]).to(device).eval()
    actor.load_state_dict(checkpoint["actor_state"], strict=True)
    critic = Critics(checkpoint["dim"], checkpoint["critic_widths"]).to(device).eval()
    critic.load_state_dict(checkpoint["critic_state"], strict=True)
    # Same 256-row chunking as train.py. Inference does not inspect heldout labels.
    with torch.no_grad():
        noise = torch.as_tensor(proposal["noise"], dtype=torch.float32, device=device)
        raw = torch.cat([actor(part, clip=True) for part in noise.split(256)])
        q = torch.cat([critic(part) for part in raw.split(256)])
    indices = plan["indices"]
    np.testing.assert_allclose(raw.cpu().numpy()[indices], proposal["raw_action"][indices], rtol=1e-5, atol=1e-6,
                               err_msg="Proposals do not reproduce their actor checkpoint")
    np.testing.assert_allclose(q.cpu().numpy()[indices], proposal["predicted_q"][indices], rtol=1e-5, atol=1e-6,
                               err_msg="Predicted Q does not reproduce its frozen gate critic")
    return proposal


def export(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "preflight", "full_tasks": None, "pending_tasks": None}
    try:
        source_provenance = capture_source_provenance(output / "source_snapshot", {
            "baseline/api.py": ROOT / "baseline/api.py",
            "upstream/hard_constraint_legalizer.py": ROOT / "upstream/flowplace/flow_matching/hard_constraint_legalizer.py",
            "backend/export_macro_def.py": ROOT / "backend/scripts/export_macro_def.py"}, "startup_before_candidate_export")
        write_json(output / "source_provenance.json", source_provenance)
        plan, arrays, manifest, best = prepare_plan(args)
        plan["source_provenance"] = source_provenance
        # This record is immutable: later failures produce status.json, not a reduced plan.
        write_json(output / "selection_plan.json", plan)
        labels = {}
        for line in Path(plan["source_bindings"]["backend_labels"]["path"]).read_text().splitlines():
            record = json.loads(line)
            if record["row_index"] in labels:
                raise ValueError("Duplicate source backend row")
            labels[record["row_index"]] = record
        from baseline.api import legalize, load_problem, save_sample
        problem = load_problem(plan["design"], args.device)
        sizes = problem.cond.x.detach().cpu().numpy().reshape(-1, 2)
        shape = sizes.shape
        tasks, pending, proposals = [], [], {}
        for item, task_id in zip(plan["candidates"], plan["task_ids"]):
            method, index = item["method"], item["index"]
            if method in ("teacher_relegalized", "best_observed_train"):
                task = reused_task(plan, item, task_id, arrays, manifest, best, labels, output)
                task["normalized_sizes"] = sizes.tolist()
                if not legality(np.asarray(task["normalized_action_flat"]).reshape(shape), sizes)["legal"]:
                    raise ValueError("Reused source geometry failed independent legality")
            else:
                if (ROOT / "backend/results" / task_id).exists():
                    raise FileExistsError(f"Backend run ID already exists: {task_id}; use a unique prefix")
                if method not in proposals:
                    proposals[method] = load_proposals(method, plan, arrays, args.device)
                proposal = proposals[method]
                raw = proposal["raw_action"][index].reshape(shape)
                action = legalize(torch.as_tensor(raw, device=args.device), problem).cpu().numpy()
                if not legality(action, sizes)["legal"]:
                    raise ValueError(f"Illegal candidate {method}/{index}; no replacement or success-only export")
                folder = output / task_id
                save_sample(folder, torch.as_tensor(action), torch.as_tensor(arrays["noise"][index].reshape(shape)), problem,
                            {"method": method, "index": index, "split": int(arrays["split"][index]),
                             "purpose": "Frozen GRT comparison; one identical external legalization"})
                action_path = folder / "executed.npz"
                np.savez_compressed(action_path, action=action[None], raw_action=raw[None], indices=[index],
                                    split=arrays["split"][[index]], noise=arrays["noise"][[index]])
                def_path = folder / f"{plan['design']}.def"
                subprocess.run([sys.executable, str(ROOT / "backend/scripts/export_macro_def.py"),
                                "--design", plan["design"], "--coordinates", str(folder / "macro_physical.json"),
                                "--output", str(def_path)], check=True, stdout=subprocess.DEVNULL)
                audit_path = def_path.with_suffix(".audit.json")
                audit = read_json(audit_path)
                if audit.get("output_sha256") != sha256_file(def_path) or audit.get("legal_no_halo") is not True:
                    raise ValueError("Rounded exported DEF failed its binding/legality audit")
                task = {**item, "task_id": task_id, "run_id": task_id, "design": plan["design"],
                        "split": int(arrays["split"][index]), "training_seed": int(method.split("/")[0][5:]),
                        "role": "candidate", "reused_backend_measurement": False, "requires_backend_execution": True,
                        "noise_conditioning": True,
                        "dataset_sha256": plan["dataset_sha256"], "action_path": str(action_path),
                        "action_file_sha256": sha256_file(action_path), "action_key": "action", "action_row": 0,
                        "normalized_action_flat": action.reshape(-1).tolist(), "normalized_sizes": sizes.tolist(),
                        "action_sha256": action_sha256(action), "raw_action_sha256": action_sha256(raw),
                        "legalization_mse": float(np.square(action - raw).mean()),
                        "def_path": str(def_path), "def_sha256": sha256_file(def_path),
                        "def_audit_path": str(audit_path), "def_audit_sha256": sha256_file(audit_path),
                        "metrics_path": str(ROOT / "backend/results" / task_id / "metrics.json"),
                        "legal": True, "predicted_q": proposal["predicted_q"][index].tolist(),
                        "q_reward_stage": STAGE, "q_reward_scale_train_only": manifest["reward_scale_train_only"],
                        "q_source_manifest": plan["source_bindings"]["dataset_manifest"]["path"],
                        "q_source_manifest_sha256": plan["source_bindings"]["dataset_manifest"]["sha256"],
                        "source_proposals": plan["source_bindings"][f"{method}/proposals"],
                        "source_checkpoint": plan["source_bindings"][f"{method}/checkpoint"]}
                pending.append(task)
            tasks.append(task)
            # Partial evidence survives a failure; status forbids submitting it as a full plan.
            (output / "tasks.partial.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in tasks))
        check_bindings(plan["source_bindings"])
        if len(pending) != plan["new_candidate_backend_budget"] or len(tasks) != len(plan["candidates"]):
            raise ValueError("Exported counts differ from frozen budgets")
        (output / "tasks.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in tasks))
        (output / "pending_tasks.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in pending))
        (output / "tasks.partial.jsonl").unlink()
        status.update(status="completed", full_tasks=len(tasks), pending_tasks=len(pending),
                      plan_sha256=sha256_file(output / "selection_plan.json"),
                      tasks_sha256=sha256_file(output / "tasks.jsonl"),
                      pending_tasks_sha256=sha256_file(output / "pending_tasks.jsonl"),
                      note="Submit only pending_tasks.jsonl; aggregate full tasks.jsonl after all successes")
    except Exception as exc:
        status.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)},
                      note="Do not submit partial tasks; preserve failed frozen plan and diagnose")
        write_json(output / "status.json", status)
        raise
    write_json(output / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset-dir", "training-dir", "gate-dir", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--phase", choices=["validation", "heldout"], default="validation")
    parser.add_argument("--selection")
    parser.add_argument("--physical-failure-policy", help="Explicit pre-outcome GRT actor policy JSON; omitted means strict")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--alphas", default="1,10,100")
    parser.add_argument("--validation-indices", default="192,193,194,195")
    parser.add_argument("--heldout-indices", default="240,241,242,243")
    parser.add_argument("--prefix", default="bp_fe_grt_actor_v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run candidate inference, legalization and exports through Slurm")
    if not args.prefix or Path(args.prefix).name != args.prefix:
        raise ValueError("Invalid backend run prefix")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(json.dumps(export(args)), flush=True)


if __name__ == "__main__":
    main()
