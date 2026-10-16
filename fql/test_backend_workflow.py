"""Software fixtures only: backend provenance, sealed labels and critic reuse."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fql.aggregate_backend_candidates import aggregate
from fql.critic_gate import load_train_validation, run_gate
from fql.relabel_backend import action_sha256, relabel, sha256_file
from fql import test_relabel_backend
from fql.seal_grt_selection import seal


class BackendWorkflowTests(unittest.TestCase):
    def candidate_fixture(self, root):
        methods = ["teacher_relegalized", "seed_0/distill_only", "seed_1/distill_only",
                   "seed_0/fql_alpha_10", "seed_1/fql_alpha_10"]
        records = []
        for method in methods:
            for index in [2, 3]:
                folder = root / method / str(index)
                folder.mkdir(parents=True)
                def_path = folder / "fixture.def"
                def_path.write_text("SYNTHETIC SOFTWARE FIXTURE ONLY\n")
                def_hash = sha256_file(def_path)
                audit = {"output_sha256": def_hash, "legal_no_halo": True}
                def_path.with_suffix(".audit.json").write_text(json.dumps(audit))
                rwl = 100. + index + (10. if "distill" in method else 0.)
                (folder / "metrics.json").write_text(json.dumps({"stage": "global_route_only", "not_final_table2": True,
                    "rWL_um": rwl, "WNS_ns": -1., "TNS_ns": -10., "Power_W": None, "StdCellArea_um2": 12.}))
                (folder / "manifest.json").write_text(json.dumps({"stage": "global_route_only", "not_final_table2": True,
                    "success": True, "design": "toy", "def_sha256": def_hash, "def_path": str(def_path),
                    "openroad_sha256": "a" * 64, "design_config_sha256": "b" * 64, "flow_script_patch": "test"}))
                action = np.array([-.5, -.5, .5, .5], dtype=np.float32)
                action_path = folder / "executed.npz"
                np.savez(action_path, action=action.reshape(1, 2, 2))
                records.append({"method": method, "index": index, "split": 1, "design": "toy",
                    "task_id": f"{method}:{index}", "metrics_path": str(folder / "metrics.json"),
                    "def_path": str(def_path), "def_sha256": def_hash, "normalized_action_flat": action.tolist(),
                    "normalized_sizes": [[.1, .1], [.1, .1]], "action_sha256": action_sha256(action),
                    "action_path": str(action_path), "action_file_sha256": sha256_file(action_path), "action_row": 0,
                    "predicted_q": -1., "q_reward_stage": "macro_only_hpwl", "legal": True})
        plan = {"phase": "validation", "methods": methods, "indices": [2, 3], "task_ids": [row["task_id"] for row in records]}
        tasks = root / "tasks.jsonl"
        tasks.write_text("".join(json.dumps(row) + "\n" for row in records))
        (root / "selection_plan.json").write_text(json.dumps(plan))
        return tasks, records

    def test_seed_separation_and_no_macro_q_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, _ = self.candidate_fixture(root)
            report = aggregate(tasks, root / "report")
            self.assertEqual(report["status"], "completed")
            group = report["quality"]["by_configuration"]["fql_alpha_10"]
            self.assertEqual(set(group["training_seeds"]), {0, 1})
            self.assertEqual(group["noise_count_per_seed"], 2)
            self.assertEqual(group["rwl_mean_across_seed_means"]["count"], 2)
            self.assertNotIn("mae", report["quality"]["by_method"]["seed_0/fql_alpha_10"]["q_calibration"])
            pairs = [row for row in report["quality"]["paired_records"] if row["baseline"] == "seed_0/distill_only"]
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(row["rwl_reduction_percent"] > 0 for row in pairs))

    def test_missing_and_failed_requests_do_not_form_success_subset(self):
        for kind in ["missing", "failed", "changed_action"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tasks, rows = self.candidate_fixture(root)
                if kind == "missing":
                    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
                elif kind == "failed":
                    path = Path(rows[0]["metrics_path"]).with_name("manifest.json")
                    obj = json.loads(path.read_text()); obj["success"] = False
                    path.write_text(json.dumps(obj))
                else:
                    np.savez(rows[0]["action_path"], action=np.zeros((1, 2, 2)))
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    aggregate(tasks, root / "report")
                report = json.loads((root / "report/report.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertIsNone(report["quality"])
                self.assertFalse(report["selection_allowed"])

    def test_fixed_control_index_does_not_create_noise_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, rows = self.candidate_fixture(root)
            for row in rows:
                if row["method"] == "teacher_relegalized":
                    row["method"] = "teacher_original"
            fixed = dict(rows[0], method="fixed_zero_anchor", index=2, role="fixed_control",
                         noise_conditioning=False, task_id="fixed_anchor_once")
            rows.append(fixed)
            plan = {"phase": "exploratory", "reference_method": "teacher_original",
                    "candidates": [{"method": row["method"], "index": row["index"]} for row in rows]}
            plan_path = root / "selection_plan.json"
            tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            plan_path.write_text(json.dumps(plan))
            before = aggregate(tasks, root / "before")["quality"]
            self.assertTrue(any(row["baseline"] == "teacher_original" for row in before["paired_summary"]))
            self.assertFalse(any(row["method"] == "fixed_zero_anchor" for row in before["paired_records"]))
            self.assertEqual(before["by_configuration"]["fixed_zero_anchor"]["noise_count_per_seed"], 0)
            self.assertEqual(len(before["fixed_measurements"]), 1)
            fixed["index"] = 99
            plan["candidates"][-1]["index"] = 99
            tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            plan_path.write_text(json.dumps(plan))
            after = aggregate(tasks, root / "after")["quality"]
            for key in ["paired_records", "paired_summary", "fixed_measurements", "versus_fixed_references"]:
                self.assertEqual(before[key], after[key])

    def make_grt(self, root):
        source, labels, _, _ = test_relabel_backend.RelabelTests().create_fixture(root)
        relabel(source, labels, root / "grt", require_row_hashes=True)
        return root / "grt/dataset.npz"

    def test_verified_grt_q_scale_and_wrong_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"; source_root.mkdir()
            dataset = self.make_grt(source_root)
            tasks_root = root / "tasks"; tasks_root.mkdir()
            tasks, rows = self.candidate_fixture(tasks_root)
            manifest_path = dataset.with_name("manifest.json")
            scale = json.loads(manifest_path.read_text())["reward_scale_train_only"]
            for row in rows:
                rwl = json.loads(Path(row["metrics_path"]).read_text())["rWL_um"]
                row.update(predicted_q=-rwl / scale, q_reward_stage="global_route_only",
                           q_source_manifest=str(manifest_path), q_source_manifest_sha256=sha256_file(manifest_path),
                           q_reward_scale_train_only=scale)
            tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = aggregate(tasks, root / "good")
            self.assertEqual(report["quality"]["by_method"]["seed_0/fql_alpha_10"]["q_calibration"]["mae"], 0.)
            rows[0]["q_reward_scale_train_only"] = scale * 2
            tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                aggregate(tasks, root / "bad")

    def test_gate_does_not_use_heldout_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.make_grt(root)
            before, _, _ = load_train_validation(dataset)
            with np.load(dataset) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["reward"][arrays["critic_split"] == 2] = np.nan
            arrays["backend_rwl_um"][arrays["critic_split"] == 2] = np.nan
            np.savez(dataset, **arrays)
            manifest_path = dataset.with_name("manifest.json")
            manifest = json.loads(manifest_path.read_text()); manifest["dataset_sha256"] = sha256_file(dataset)
            manifest_path.write_text(json.dumps(manifest))
            after, _, _ = load_train_validation(dataset)
            for phase in ["train", "validation"]:
                np.testing.assert_array_equal(before[phase]["reward"], after[phase]["reward"])

    def test_seal_refuses_exploratory_cross_fidelity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, _ = self.candidate_fixture(root)
            plan_path = root / "selection_plan.json"
            plan = json.loads(plan_path.read_text()); plan["phase"] = "exploratory"
            plan_path.write_text(json.dumps(plan))
            aggregate(tasks, root / "report")
            with self.assertRaisesRegex(ValueError, "Only complete GRT validation"):
                seal(root / "report/report.json", root / "selection.json")

    def test_seal_success_and_reject_changed_heldout_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, rows = self.candidate_fixture(root)
            fixed = dict(rows[0], method="best_observed_train", index=0, split=0,
                         role="fixed_training_baseline", reused_backend_measurement=True,
                         task_id="best_train_once", dataset_sha256="d" * 64)
            baseline_path = root / "best_train_layout.json"
            baseline_path.write_text(json.dumps({"selection_split": "train_only", "parent_noise_index": 0,
                "dataset_sha256": fixed["dataset_sha256"], "action_sha256": fixed["action_sha256"],
                "backend_record": {"def_sha256": fixed["def_sha256"], "metrics_sha256": sha256_file(fixed["metrics_path"])}}))
            fixed.update(best_train_baseline_path=str(baseline_path), best_train_baseline_sha256=sha256_file(baseline_path))
            rows.append(fixed)
            tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            plan = {"phase": "validation", "training_reward_stage": "global_route_only", "heldout_indices": [4, 5],
                    "candidates": [{"method": row["method"], "index": row["index"]} for row in rows]}
            (root / "selection_plan.json").write_text(json.dumps(plan))
            report = aggregate(tasks, root / "report")
            self.assertEqual(report["quality"]["by_method"]["best_observed_train"]["count"], 1)
            chosen = seal(root / "report/report.json", root / "selection.json")
            self.assertEqual(chosen["chosen_fql_configuration"], "fql_alpha_10")
            self.assertEqual(chosen["training_seeds"], [0, 1])
            self.assertEqual(chosen["heldout_plan"]["new_candidate_backend_budget"], 8)
            changed = chosen["heldout_plan"]
            changed["selection_path"] = str(root / "selection.json")
            changed["selection_sha256"] = sha256_file(root / "selection.json")
            changed["candidates"][0]["index"] = 99
            (root / "tampered_heldout.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "differ from the sealed plan"):
                aggregate(tasks, root / "bad_heldout", root / "tampered_heldout.json")

    def test_gate_critic_reloads_with_larger_actor_and_sealed_heldout(self):
        if "SLURM_JOB_ID" not in os.environ:
            self.skipTest("Actual numeric subprocess smoke requires Slurm")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.make_grt(root)
            args = SimpleNamespace(dataset=str(dataset), output=str(root / "gate"), seeds="0", widths="4,4",
                                   steps=2, batch_size=2, lr=3e-4, weight_decay=1e-4, device="cpu")
            gate = run_gate(args)
            self.assertFalse(gate["heldout_used"])
            self.assertEqual(gate["best_train_baseline"]["source_row_index"], 0)
            run = root / "student"
            subprocess.run([sys.executable, str(ROOT / "fql/train.py"), "--dataset", str(dataset), "--output", str(run),
                            "--seeds", "0", "--alphas", "10", "--widths", "16,16", "--pretrained-critic-dir", str(root / "gate"),
                            "--critic-steps", "1", "--warmup-steps", "2", "--actor-steps", "2", "--log-every", "1",
                            "--latency-repeats", "0", "--device", "cpu"], check=True, stdout=subprocess.DEVNULL)
            checkpoint_path = run / "seed_0/fql_alpha_10/checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            gate_checkpoint = torch.load(root / "gate/seed_0/critic.pt", weights_only=False)
            self.assertEqual(checkpoint["actor_widths"], [16, 16])
            self.assertEqual(checkpoint["critic_widths"], [4, 4])
            for key, value in gate_checkpoint["critic_state"].items():
                torch.testing.assert_close(value, checkpoint["critic_state"][key], rtol=0, atol=0)
            self.assertNotIn("heldout", json.loads((run / "seed_0/critic_diagnostics.json").read_text()))
            self.assertNotIn("heldout", json.loads((run / "seed_0/fql_alpha_10/diagnostics.json").read_text())["splits"])
            subprocess.run([sys.executable, str(ROOT / "fql/predict.py"), "--checkpoint", str(checkpoint_path),
                            "--noise", str(dataset), "--output", str(root / "predicted.npz")], check=True, stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, str(ROOT / "fql/latency.py"), "--checkpoint", str(checkpoint_path),
                            "--noise", str(dataset), "--output", str(root / "latency.json"), "--batch-sizes", "1", "--warmup", "0", "--repeats", "1"],
                           check=True, stdout=subprocess.DEVNULL)
            cache = {"node_id_to_name": ["a", "b"], "ratio_x": 10/224, "ratio_y": 20/224,
                     "node_info": {name: {"x": 1, "y": 1} for name in "ab"},
                     "net_info": {"test": {"nodes": {"a": {}, "b": {}}}}}
            torch.save(cache, root / "cache.pt")
            subprocess.run([sys.executable, str(ROOT / "fql/coordinate_control.py"), "--dataset", str(dataset),
                            "--cache", str(root / "cache.pt"), "--training-dir", str(run), "--reference-checkpoint", str(checkpoint_path),
                            "--steps", "1", "--anchor-weights", "1", "--reward-scale", "20"], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main()
