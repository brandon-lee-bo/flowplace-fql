"""Synthetic provenance fixtures only; no fake backend-performance claims."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from .relabel_backend import action_sha256, relabel, sha256_file
except ImportError:
    from relabel_backend import action_sha256, relabel, sha256_file


class RelabelTests(unittest.TestCase):
    def create_fixture(self, root):
        source = root / "source.npz"
        noise = np.linspace(-0.7, 0.7, 24, dtype=np.float32).reshape(6, 4)
        split = np.array([0, 0, 1, 1, 2, 2])
        arrays = dict(noise=noise, teacher_action=noise * 0.5, split=split,
                      raw_action=noise * 0.5, action=noise * 0.5, reward=-np.arange(1, 7, dtype=np.float32),
                      critic_split=split, group_id=np.arange(6), critic_group_id=np.arange(6),
                      critic_parent_index=np.arange(6), legal=np.ones(6, dtype=bool),
                      macro_hpwl=np.arange(1, 7, dtype=np.float32))
        np.savez(source, **arrays)
        (root / "manifest.json").write_text(json.dumps({"args": {"design": "toy"},
                                                       "metric": {"metric": "toy_macro_proxy"}}))
        records = []
        for row, wirelength in enumerate([100., 200., 1000., 2000., 10000., 20000.]):
            backend = root / f"backend_{row}"
            backend.mkdir()
            def_path = backend / "toy.def"
            def_path.write_text(f"SOFTWARE FIXTURE ONLY ROW {row}\n")
            metrics = {"stage": "global_route_only", "not_final_table2": True, "rWL_um": wirelength,
                       "WNS_ns": -0.5, "TNS_ns": -5., "Power_W": None, "StdCellArea_um2": 100.}
            (backend / "metrics.json").write_text(json.dumps(metrics))
            manifest = {"stage": "global_route_only", "not_final_table2": True, "success": True,
                        "design": "toy", "def_path": str(def_path), "def_sha256": sha256_file(def_path),
                        "openroad_sha256": "a" * 64, "design_config_sha256": "b" * 64, "flow_script_patch": "toy"}
            (backend / "manifest.json").write_text(json.dumps(manifest))
            records.append({"row_index": row, "dataset_sha256": sha256_file(source),
                            "action_sha256": action_sha256(arrays["action"][row]),
                            "def_sha256": sha256_file(def_path), "def_path": str(def_path),
                            "metrics_path": str(backend / "metrics.json"), "run_id": f"toy_{row}"})
        labels = root / "requests.jsonl"
        labels.write_text("".join(json.dumps(record) + "\n" for record in records))
        return source, labels, arrays, records

    def test_train_only_scale_and_proxy_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, arrays, _ = self.create_fixture(root)
            manifest = relabel(source, labels, root / "output", require_row_hashes=True)
            self.assertEqual(manifest["reward_scale_train_only"], 150.)
            with np.load(root / "output/dataset.npz") as output:
                np.testing.assert_array_equal(output["noise"], arrays["noise"])
                np.testing.assert_array_equal(output["teacher_action"], arrays["teacher_action"])
                np.testing.assert_array_equal(output["proxy_reward"], arrays["reward"])
                np.testing.assert_allclose(output["reward"], -np.array([100, 200, 1000, 2000, 10000, 20000]) / 150., rtol=1e-6)
                self.assertEqual(str(output["reward_stage"]), "global_route_only")
                self.assertEqual(str(output["proxy_reward_stage"]), "toy_macro_proxy")

    def test_subset_requires_opt_in_and_preserves_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, arrays, records = self.create_fixture(root)
            subset = [0, 2, 4]
            labels.write_text("".join(json.dumps(records[i]) + "\n" for i in subset))
            with self.assertRaisesRegex(ValueError, "validation failed"):
                relabel(source, labels, root / "default_rejected")
            self.assertFalse((root / "default_rejected/dataset.npz").exists())
            manifest = relabel(source, labels, root / "explicit_subset", allow_unlabeled_subset=True, require_row_hashes=True)
            self.assertEqual(manifest["excluded_unrequested_source_rows"], [1, 3, 5])
            with np.load(root / "explicit_subset/dataset.npz") as output:
                np.testing.assert_array_equal(output["noise"], arrays["noise"])
                np.testing.assert_array_equal(output["action"], arrays["action"][subset])
                np.testing.assert_array_equal(output["raw_action"], arrays["raw_action"][subset])
                np.testing.assert_array_equal(output["critic_parent_index"], arrays["critic_parent_index"][subset])
                np.testing.assert_array_equal(output["source_row_index"], subset)
                self.assertEqual(len(output["teacher_action"]), 6)

    def test_requested_failure_cannot_be_dropped_with_subset_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, _, _ = self.create_fixture(root)
            path = root / "backend_2/manifest.json"
            record = json.loads(path.read_text())
            record["success"] = False
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "validation failed"):
                relabel(source, labels, root / "output", allow_unlabeled_subset=True)
            audit = json.loads((root / "output/audit.json").read_text())
            self.assertEqual(audit["status"], "failed")
            self.assertTrue(any(error.get("row_index") == 2 for error in audit["errors"]))
            self.assertFalse((root / "output/dataset.npz").exists())

    def test_wrong_row_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, _, records = self.create_fixture(root)
            records[3]["action_sha256"] = records[2]["action_sha256"]
            labels.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(ValueError, "validation failed"):
                relabel(source, labels, root / "output", require_row_hashes=True)
            self.assertFalse((root / "output/dataset.npz").exists())

    def test_requested_row_cannot_disappear_from_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, _, records = self.create_fixture(root)
            (root / "selection_plan.json").write_text(json.dumps({
                "dataset_sha256": sha256_file(source), "selected_rows": list(range(6))}))
            labels.write_text("".join(json.dumps(record) + "\n" for record in records if record["row_index"] != 3))
            with self.assertRaisesRegex(ValueError, "validation failed"):
                relabel(source, labels, root / "output", allow_unlabeled_subset=True)
            audit = json.loads((root / "output/audit.json").read_text())
            self.assertTrue(any(error.get("planned_but_missing_rows") == [3] for error in audit["errors"]))

    def test_mixed_stage_and_provenance_are_rejected(self):
        for changed_key, changed_value in [("stage", "detailed_route"), ("openroad_sha256", "c" * 64)]:
            with self.subTest(key=changed_key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, labels, _, _ = self.create_fixture(root)
                path = root / "backend_4/manifest.json"
                manifest = json.loads(path.read_text())
                manifest[changed_key] = changed_value
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    relabel(source, labels, root / "output")


if __name__ == "__main__":
    unittest.main()
