"""Software-only end-to-end GRT export contract using mock geometry/backend IO.

Real small actor/critic checkpoints are trained inside Slurm. Mocked DEF/PPA
fixtures are deliberately synthetic; they provide no placement evidence.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

REAL_SUBPROCESS_RUN = subprocess.run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fql.aggregate_backend_candidates import aggregate
from fql.critic_gate import run_gate
from fql.prepare_grt_candidate_tasks import export, prepare_plan
from fql.relabel_backend import relabel, sha256_file
from fql.seal_grt_selection import seal
from fql import test_relabel_backend


class GRTExportTests(unittest.TestCase):
    def fixture(self, root):
        source, labels, arrays, records = test_relabel_backend.RelabelTests().create_fixture(root)
        arrays["variant"] = np.zeros(6, dtype=np.int64)
        np.savez(source, **arrays)
        for record in records:
            record["dataset_sha256"] = sha256_file(source)
            Path(record["def_path"]).with_suffix(".audit.json").write_text(json.dumps({
                "output_sha256": record["def_sha256"], "legal_no_halo": True}))
        labels.write_text("".join(json.dumps(row) + "\n" for row in records))
        relabel(source, labels, root / "grt", require_row_hashes=True)
        run_gate(SimpleNamespace(dataset=str(root / "grt/dataset.npz"), output=str(root / "gate"),
                                 seeds="0", widths="4,4", steps=2, batch_size=2,
                                 lr=3e-4, weight_decay=1e-4, device="cpu"))
        subprocess.run([sys.executable, str(ROOT / "fql/train.py"), "--dataset", str(root / "grt/dataset.npz"),
                        "--output", str(root / "train"), "--seeds", "0", "--alphas", "10", "--widths", "8,8",
                        "--pretrained-critic-dir", str(root / "gate"), "--critic-steps", "1", "--warmup-steps", "2",
                        "--actor-steps", "2", "--latency-repeats", "0", "--device", "cpu"],
                       check=True, stdout=subprocess.DEVNULL)
        return SimpleNamespace(dataset_dir=str(root / "grt"), training_dir=str(root / "train"), gate_dir=str(root / "gate"),
                               output=str(root / "validation"), phase="validation", selection=None, seeds="0", alphas="10",
                               validation_indices="2,3", heldout_indices="4,5", prefix="synthetic_fixture", device="cpu")

    @staticmethod
    def fake_api():
        def save_sample(folder, *args):
            Path(folder).mkdir()
            (Path(folder) / "macro_physical.json").write_text("{}")
        # Mock legalization intentionally has no physical interpretation.
        return SimpleNamespace(load_problem=lambda *args: SimpleNamespace(cond=SimpleNamespace(x=torch.full((2, 2), .01))),
                               legalize=lambda raw, problem: torch.tensor([[-.5, -.5], [.5, .5]]), save_sample=save_sample)

    @staticmethod
    def fake_def_export(command, **kwargs):
        if command[0] == "git":
            # Source snapshots query real git metadata; only mock the DEF tool.
            return REAL_SUBPROCESS_RUN(command, **kwargs)
        path = Path(command[command.index("--output") + 1])
        path.write_text("SYNTHETIC DEF EXPORT FIXTURE ONLY\n")
        path.with_suffix(".audit.json").write_text(json.dumps({"output_sha256": sha256_file(path), "legal_no_halo": True}))

    def test_validation_reuse_binding_and_sealed_heldout(self):
        if "SLURM_JOB_ID" not in os.environ:
            self.skipTest("Numeric fixture runs only in Slurm")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            cache = root / "mock_cache.pt"
            cache.write_bytes(b"SYNTHETIC CACHE IDENTITY FIXTURE ONLY")
            cache_mock = patch("fql.prepare_grt_candidate_tasks.benchmark_cache_path", return_value=cache)
            cache_mock.start()
            self.addCleanup(cache_mock.stop)
            with patch.dict(sys.modules, {"baseline.api": self.fake_api()}), patch("fql.prepare_grt_candidate_tasks.subprocess.run", side_effect=self.fake_def_export):
                status = export(args)
            self.assertEqual((status["full_tasks"], status["pending_tasks"]), (7, 4))
            tasks_path = root / "validation/tasks.jsonl"
            rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
            self.assertEqual(sum(row["role"] == "fixed_training_baseline" for row in rows), 1)
            self.assertEqual(sum(row["reused_backend_measurement"] for row in rows), 3)
            template = json.loads((root / "backend_0/manifest.json").read_text())
            # Attach synthetic backend fixtures inside the temporary directory.
            # Task identities and frozen candidate plan remain unchanged.
            for i, row in enumerate(rows):
                if not row["requires_backend_execution"]:
                    continue
                folder = root / f"fake_new_backend_{i}"; folder.mkdir()
                (folder / "metrics.json").write_text(json.dumps({"stage": "global_route_only", "not_final_table2": True,
                    "rWL_um": 200. + i, "WNS_ns": -.5, "TNS_ns": -5.}))
                (folder / "manifest.json").write_text(json.dumps(dict(template, def_path=row["def_path"], def_sha256=row["def_sha256"])))
                row["metrics_path"] = str(folder / "metrics.json")
            tasks_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = aggregate(tasks_path, root / "validation_report")
            self.assertEqual(report["status"], "completed")
            selection_path = root / "selection.json"
            selection = seal(root / "validation_report/report.json", selection_path)
            self.assertEqual(selection["heldout_plan"]["new_candidate_backend_budget"], 4)
            args.phase, args.selection, args.output = "heldout", str(selection_path), str(root / "heldout")
            with patch.dict(sys.modules, {"baseline.api": self.fake_api()}), patch("fql.prepare_grt_candidate_tasks.subprocess.run", side_effect=self.fake_def_export):
                heldout = export(args)
            self.assertEqual((heldout["full_tasks"], heldout["pending_tasks"]), (7, 4))
            # The selection freezes source checkpoints/proposals, not merely method names.
            proposals = root / "train/seed_0/fql_alpha_10/proposals.npz"
            proposals.write_bytes(proposals.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "Source artifact changed"):
                prepare_plan(args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main()
