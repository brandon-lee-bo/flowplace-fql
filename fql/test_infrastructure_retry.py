"""Focused real-artifact recovery checks; run on a Slurm CPU allocation."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fql.aggregate_backend_candidates import validate_task
from fql.infrastructure_retry import load_retry_map, resolve_successful_retry
from fql.provenance import capture_source_provenance

TASKS = ROOT / "datasets/bp_fe_sampler_transfer_request_v1/tasks.jsonl"
MAP = ROOT / "backend/assets/sampler_infrastructure_retry_map_v1.json"


class RecoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
        cls.binding = load_retry_map(MAP, TASKS)
        cls.original_id = next(iter(cls.binding["data"]["retries"]))
        cls.task = next(row for row in cls.rows if row["run_id"] == cls.original_id)
        cls.item = cls.binding["data"]["retries"][cls.original_id]

    def test_01_real_eleven_requests_one_successful_recovery(self):
        self.assertEqual(len(self.rows), 11)
        recovered = []
        for task in self.rows:
            effective, proof = resolve_successful_retry(task, TASKS, self.binding)
            for key in ("run_id", "method", "index", "action_sha256", "def_sha256", "def_path"):
                self.assertEqual(task[key], effective[key])
            if proof is not None:
                recovered.append(proof)
                record = validate_task(effective, TASKS.parent)
                self.assertEqual(record["status"], "success")
                self.assertEqual(record["run_id"], task["run_id"])
                self.assertEqual(proof["extra_candidate_denominator"], 0)
                self.assertEqual(proof["original_slurm_job_id"], "1785399")
                self.assertEqual(proof["retry_slurm_job_id"], "1785454")
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["requested_run_id"], self.original_id)
        # Other original requests may be pending or physical failures; no success assumption.
        type(self).real_proof = recovered[0]

    def test_02_original_preemption_hash_change_rejected(self):
        binding = copy.deepcopy(self.binding)
        binding["data"]["retries"][self.original_id]["preemption_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "original preemption path/hash"):
            resolve_successful_retry(self.task, TASKS, binding)

    def test_03_original_raw_preemption_job_mismatch_rejected(self):
        original_text = Path(self.item["preemption_path"]).read_text()
        real_loads = json.loads
        modified = real_loads(original_text)
        modified["sacct_record"] = modified["sacct_record"].replace("1785399|", "9999999|")
        # Isolate the semantic check after the independent on-disk SHA check.
        def parse(text, *args, **kwargs):
            return modified if text == original_text else real_loads(text, *args, **kwargs)
        with patch("fql.infrastructure_retry.json.loads", side_effect=parse):
            with self.assertRaisesRegex(ValueError, "raw Slurm record is not PREEMPTED"):
                resolve_successful_retry(self.task, TASKS, self.binding)

    def test_04_retry_run_identity_change_rejected(self):
        binding = copy.deepcopy(self.binding)
        binding["data"]["retries"][self.original_id]["retry_run_id"] = "unrequested_retry"
        with self.assertRaisesRegex(ValueError, "Retry run ID/manifest/metrics location"):
            resolve_successful_retry(self.task, TASKS, binding)

    def test_05_requested_def_change_rejected(self):
        task = dict(self.task, def_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "actual DEF differs"):
            resolve_successful_retry(task, TASKS, self.binding)

    def test_06_retry_core_configuration_change_rejected(self):
        original_text = Path(self.item["retry_manifest_path"]).read_text()
        real_loads = json.loads
        modified = real_loads(original_text)
        modified["num_cores"] = "4"
        def parse(text, *args, **kwargs):
            return modified if text == original_text else real_loads(text, *args, **kwargs)
        with patch("fql.infrastructure_retry.json.loads", side_effect=parse):
            with self.assertRaisesRegex(ValueError, "physical flow: num_cores"):
                resolve_successful_retry(self.task, TASKS, self.binding)


if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this artifact/hash verification inside Slurm")
    output = ROOT / "runs" / ("software_infrastructure_retry_" + os.environ["SLURM_JOB_ID"])
    output.mkdir(parents=True, exist_ok=False)
    source = capture_source_provenance(output / "source", {"test_infrastructure_retry.py": Path(__file__)},
                                       "startup_before_infrastructure_retry_software_checks")
    (output / "source_provenance.json").write_text(json.dumps(source, indent=2) + "\n")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "software_validation_only": True,
              "tests_run": result.testsRun, "passed": result.wasSuccessful(),
              "failures": [str(item) for item in result.failures], "errors": [str(item) for item in result.errors],
              "original_requested_count": 11, "recovered_count": 1 if hasattr(RecoveryChecks, "real_proof") else None,
              "real_recovery_proof": getattr(RecoveryChecks, "real_proof", None),
              "source_provenance": source}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
