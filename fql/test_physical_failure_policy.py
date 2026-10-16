"""Synthetic software fixtures only: no placement or backend-quality evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fql.aggregate_backend_candidates import aggregate
from fql.physical_failure_policy import bind_policy, validate_bound_policy
from fql.relabel_backend import action_sha256, sha256_file
from fql.seal_grt_selection import seal


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "fql/grt_actor_physical_policy_v1.json"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fixture(root, phase="validation", selection_path=None):
    binding = bind_policy(POLICY)
    policy = binding["document"]
    selection = json.loads(selection_path.read_text()) if selection_path else None
    methods = selection["methods"] if selection else [f"seed_{seed}/{config}" for seed in policy["training_seeds"]
        for config in ["distill_only", "fql_alpha_1", "fql_alpha_10", "fql_alpha_100"]]
    indices = policy[phase + "_indices"]
    candidates = [{"method": method, "index": index} for method in ["teacher_relegalized", *methods] for index in indices]
    candidates.append({"method": "best_observed_train", "index": 0})
    dataset = root / "source/dataset.npz"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"SYNTHETIC SOFTWARE PROVENANCE FIXTURE, NOT TRAINING DATA\n")
    data_sha = sha256_file(dataset)
    write(dataset.with_name("manifest.json"), {"reward_stage": "global_route_only", "dataset_sha256": data_sha})
    write(root / "train/run.json", {"data": {"sha256": data_sha}, "args": {"seal_heldout": True}})
    write(root / "train/summary.json", {"status": "training_completed_external_evaluation_required"})
    write(root / "gate/report.json", {"status": "completed"})
    sources = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in {
        "dataset": dataset, "dataset_manifest": dataset.with_name("manifest.json"),
        "training_run": root / "train/run.json", "training_summary": root / "train/summary.json",
        "gate_report": root / "gate/report.json"}.items()}
    sources["physical_failure_policy"] = {key: binding[key] for key in ["path", "sha256"]}
    rows = []
    for number, item in enumerate(candidates):
        method, index = item["method"], item["index"]
        folder = root / f"backend_{number:03d}"
        folder.mkdir()
        def_path = folder / "fixture.def"
        def_path.write_text("SYNTHETIC DEF SOFTWARE FIXTURE ONLY\n")
        def_sha = sha256_file(def_path)
        write(def_path.with_suffix(".audit.json"), {"output_sha256": def_sha, "legal_no_halo": True})
        rwl = 100. if "alpha_1" in method else 120.
        write(folder / "metrics.json", {"stage": "global_route_only", "not_final_table2": True,
              "rWL_um": rwl + index * .01, "WNS_ns": -.5, "TNS_ns": -5.})
        write(folder / "manifest.json", {"stage": "global_route_only", "not_final_table2": True,
              "success": True, "design": "toy", "def_path": str(def_path), "def_sha256": def_sha,
              "openroad_sha256": "a" * 64, "design_config_sha256": "b" * 64, "flow_script_patch": "fixture"})
        action = np.asarray([-.5, -.5, .5, .5], dtype=np.float32)
        action_path = folder / "executed.npz"
        np.savez(action_path, action=action.reshape(1, 2, 2))
        role = "fixed_training_baseline" if method == "best_observed_train" else "paired_reference" if method == "teacher_relegalized" else "candidate"
        row = {**item, "task_id": f"task_{number}", "run_id": folder.name,
               "role": role, "split": 0 if role == "fixed_training_baseline" else 1 if phase == "validation" else 2,
               "design": "toy", "legal": True, "requires_backend_execution": role == "candidate",
               "dataset_sha256": data_sha, "metrics_path": str(folder / "metrics.json"),
               "def_path": str(def_path), "def_sha256": def_sha,
               "action_path": str(action_path), "action_file_sha256": sha256_file(action_path),
               "action_row": 0, "action_key": "action", "action_sha256": action_sha256(action),
               "normalized_action_flat": action.tolist(), "normalized_sizes": [[.1, .1], [.1, .1]]}
        if role == "fixed_training_baseline":
            reference = root / "gate/best_train_layout.json"
            write(reference, {"selection_split": "train_only", "parent_noise_index": 0,
                  "dataset_sha256": data_sha, "action_sha256": row["action_sha256"],
                  "backend_record": {"def_sha256": def_sha, "metrics_sha256": sha256_file(folder / "metrics.json")}})
            row.update(best_train_baseline_path=str(reference), best_train_baseline_sha256=sha256_file(reference))
        rows.append(row)
    tasks = root / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
    plan = {"phase": phase, "training_reward_stage": "global_route_only", "physical_failure_policy": binding,
            "training_seeds": policy["training_seeds"], "validation_indices": policy["validation_indices"],
            "heldout_indices": policy["heldout_indices"], "predeclared_alphas": policy["alphas"],
            "indices": indices, "methods": methods, "candidates": candidates,
            "task_ids": [row["task_id"] for row in rows], "source_bindings": sources,
            "dataset_sha256": data_sha, "training_dir": str(root / "train"),
            "freeze_utc": datetime.now(timezone.utc).isoformat()}
    if selection:
        plan.update(selection_path=str(selection_path), selection_sha256=sha256_file(selection_path))
    write(root / "selection_plan.json", plan)
    return tasks, rows


def fail_case(tasks, row, error_code="RSZ-0060"):
    metrics = Path(row["metrics_path"])
    metrics.unlink()
    manifest_path = metrics.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(success=False, failure="RuntimeError('floorplan_place_cts exited 2; see retained log')",
                    time_start=10., time_end=11., slurm_job_id="123", physical_flow_revision="c" * 64)
    write(manifest_path, manifest)
    log = metrics.with_name("floorplan_place_cts.log")
    log.write_text("[ERROR RSZ-0060] Max buffer count reached.\nmake[1]: *** [Makefile:750: do-4_1_cts] Error 1\nmake: *** [Makefile:756: do-cts] Error 2\n")
    sidecar = {"schema_version": 1, "status": "physical_failure", "failure_class": "cts_hold_repair_max_buffer_count",
        "error_code": error_code, "error_line": "[ERROR RSZ-0060] Max buffer count reached.", "error_line_number": 1,
        "stage": "floorplan_place_cts", "substage": "clock_tree_synthesis_hold_repair", "make_exit_code": 2,
        "openroad_exit_code": 1, "slurm_job_id": "123", "slurm_state": "FAILED", "slurm_exit_code": "1:0",
        "slurm_end": "2026-10-17T01:12:36", "time_end": 11., "physical_flow_revision": "c" * 64,
        "original_manifest_unchanged": True, "grt_metrics_available": False, "not_parser_failure": True,
        "not_preemption": True, "run_id": row["run_id"], "run_path": str(metrics.parent),
        "action_path": row["action_path"], "action_file_sha256": row["action_file_sha256"],
        "action_key": "action", "action_row": 0,
        **{key: row[key] for key in ["task_id", "method", "index", "dataset_sha256", "action_sha256"]}}
    for name, path in {"manifest": manifest_path, "def": Path(row["def_path"]), "log": log,
                       "task_manifest": tasks, "def_audit": Path(row["def_path"]).with_suffix(".audit.json")}.items():
        sidecar[name + "_path"] = str(path)
        sidecar[name + "_sha256"] = sha256_file(path)
    write(metrics.with_name("physical_failure.json"), sidecar)


class PhysicalFailurePolicyTests(unittest.TestCase):
    def test_one_alpha_failure_and_bc_failure_do_not_become_success_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, rows = fixture(root)
            for method in ["seed_0/fql_alpha_1", "seed_1/distill_only"]:
                fail_case(tasks, next(row for row in rows if row["method"] == method))
            with self.assertRaises(ValueError):
                aggregate(tasks, root / "strict")
            report = aggregate(tasks, root / "allowed", allow_preregistered_physical_failures=True)
            self.assertEqual((report["total_count"], report["successful_count"], report["physical_failure_count"]), (53, 51, 2))
            decision = seal(root / "allowed/report.json", root / "selection.json")
            self.assertEqual(decision["chosen_fql_configuration"], "fql_alpha_10")
            self.assertFalse(decision["eligibility_by_configuration"]["fql_alpha_1"]["eligible"])
            self.assertIsNone(decision["validation_rwl_by_configuration"]["fql_alpha_1"])
            self.assertIsNone(decision["validation_beats_BC"])
            self.assertEqual(decision["heldout_plan"]["indices"], [244, 245, 246, 247])
            self.assertEqual(decision["heldout_plan"]["new_candidate_backend_budget"], 24)

    def test_all_alphas_failed_is_terminal_stop_without_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, rows = fixture(root)
            for alpha in [1, 10, 100]:
                fail_case(tasks, next(row for row in rows if row["method"] == f"seed_0/fql_alpha_{alpha}"))
            aggregate(tasks, root / "report", allow_preregistered_physical_failures=True)
            decision = seal(root / "report/report.json", root / "stop.json")
            self.assertEqual(decision["status"], "no_eligible_configuration")
            self.assertEqual(decision["kind"], "sealed_GRT_validation_stop")
            self.assertIsNone(decision["heldout_plan"])
            self.assertIsNone(decision["chosen_fql_configuration"])
            self.assertEqual(decision["methods"], [])

    def test_unknown_missing_and_unbound_policy_remain_errors(self):
        for mutation in ["unknown", "missing", "no_policy", "changed_policy", "old_heldout", "exploratory"]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tasks, rows = fixture(root)
                if mutation == "unknown":
                    fail_case(tasks, next(row for row in rows if row["role"] == "candidate"), error_code="UNKNOWN")
                elif mutation == "missing":
                    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
                else:
                    plan_path = root / "selection_plan.json"
                    plan = json.loads(plan_path.read_text())
                    if mutation == "no_policy":
                        del plan["physical_failure_policy"]
                    elif mutation == "changed_policy":
                        plan["physical_failure_policy"]["sha256"] = "0" * 64
                    elif mutation == "old_heldout":
                        plan["heldout_indices"] = [240, 241, 242, 243]
                    else:
                        plan["phase"] = "exploratory"
                    write(plan_path, plan)
                with self.assertRaises(ValueError):
                    aggregate(tasks, root / "rejected", allow_preregistered_physical_failures=True)
                report = json.loads((root / "rejected/report.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["outcomes_complete"])
                self.assertIsNone(report["quality"])

    def test_heldout_failure_is_reported_without_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation"; validation.mkdir()
            tasks, _ = fixture(validation)
            aggregate(tasks, validation / "report", allow_preregistered_physical_failures=True)
            selection_path = validation / "selection.json"
            decision = seal(validation / "report/report.json", selection_path)
            sealed_sha = sha256_file(selection_path)
            heldout = root / "heldout"; heldout.mkdir()
            tasks, rows = fixture(heldout, "heldout", selection_path)
            selected = decision["chosen_fql_configuration"]
            fail_case(tasks, next(row for row in rows if row["method"] == f"seed_0/{selected}"))
            report = aggregate(tasks, heldout / "report", allow_preregistered_physical_failures=True)
            self.assertEqual((report["total_count"], report["successful_count"]), (29, 28))
            self.assertFalse(report["selection_allowed"])
            self.assertIsNone(report["quality"]["by_configuration"][selected]["rwl_mean_across_seed_means"])
            self.assertEqual(sha256_file(selection_path), sealed_sha)
            with self.assertRaisesRegex(ValueError, "Only complete GRT validation"):
                seal(heldout / "report/report.json", heldout / "forbidden_selection.json")
            plan_path = heldout / "selection_plan.json"
            plan = json.loads(plan_path.read_text())
            other = "fql_alpha_100" if selected != "fql_alpha_100" else "fql_alpha_1"
            plan["methods"] = [method.replace(selected, other) for method in plan["methods"]]
            for item in plan["candidates"]:
                item["method"] = item["method"].replace(selected, other)
            write(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "differ from the sealed plan"):
                aggregate(tasks, heldout / "changed_choice", allow_preregistered_physical_failures=True)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run software numerical fixtures through Slurm")
    unittest.main()
