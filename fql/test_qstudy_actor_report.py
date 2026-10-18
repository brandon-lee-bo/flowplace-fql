"""Report-only controls; run these bounded CPU tests under Slurm, never EDA."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import qstudy_actor_report as report


def config():
    return dict(design="bp_fe", arm="bc", seeds=[0], dataset_sha256="data", cache_sha256="cache",
        evaluation_samples=2, evaluation_noise_seed=4600000, collection_noise_seed=4500000,
        common_checkpoint_sha256={"0": "common0", "1": "common1"},
        offline_actor_steps=1000, online_updates=600, online_samples=8,
        actor_lr=1e-5, flow_lr=1e-5, batch_size=16, flow_steps=10, alpha=10.,
        online_flow_update=False, q_weight=0., weights=[1, 0, 0, 0, 0],
        critic_variant="legacy", physical_failure_reward=-4)


def native(wirelength):
    return dict(zip(report.METRICS, [wirelength, -.4, -200., 3., 2., .17, 60000.]))


def row(index, wirelength, state="success"):
    return dict(role="evaluation", index=index, noise_sha256="noise"+str(index),
        state=state, metrics=native(wirelength) if state == "success" else None,
        predicted_q=-1., measured_utility=-wirelength if state == "success" else -4.,
        request_sha256="req"+str(wirelength), raw_to_legalizer_rms=.1)


def stream(experiment, seed, wirelength):
    rows = [row(0, wirelength), row(1, wirelength+2)]
    return dict(experiment=str(experiment), seed=seed, round=0, status="complete", rows=rows,
        roles=dict(evaluation=report.summarize_rows(rows, 2)))


class ActorReportTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("TMPDIR", "/scratch/bohanli/flowplace_graph/qstudy_report_tests"))
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def curve(self, payload):
        (self.folder/"learning_curve.jsonl").write_bytes(payload)
        return report.curve(self.folder)

    def test_truncated_last_append_keeps_only_complete_rows(self):
        payload = b'{"step":1,"loss":1}\n{"step":2,"loss":'
        result = self.curve(payload)
        self.assertEqual(result["parse_status"], "partial_trailing_record")
        self.assertTrue(result["truncated_final_line"])
        self.assertEqual(result["logged_rows"], 1)
        self.assertEqual(result["latest"]["step"], 1)
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())

    def test_interior_corruption_is_not_treated_as_active_tail(self):
        result = self.curve(b'{"step":1}\nBROKEN\n{"step":3}\n')
        self.assertEqual(result["parse_status"], "integrity_invalid")
        self.assertIn("line 2", result["integrity_error"])
        self.assertTrue(result["diagnostics_withheld"])
        self.assertNotIn("latest", result)

    def test_bad_final_record_with_newline_is_committed_corruption(self):
        result = self.curve(b'{"step":1}\n{"step":2,\n')
        self.assertEqual(result["parse_status"], "integrity_invalid")
        self.assertFalse(result["truncated_final_line"])

    def test_valid_unterminated_row_and_resumed_duplicate_steps(self):
        result = self.curve(b'{"step":1,"loss":2}\n{"step":1,"loss":1}')
        self.assertEqual(result["parse_status"], "complete")
        self.assertEqual(result["logged_rows"], 2)
        self.assertEqual(result["unique_steps"], 1)
        self.assertEqual(result["latest"]["loss"], 1)

    def test_semantically_invalid_curve_withholds_diagnostics(self):
        for payload in (b'[]\n', b'{"step":true}\n',
                        b'{"step":1,"action_gradients":{"q_grad_norm":1}}\n'):
            self.assertEqual(self.curve(payload)["parse_status"], "integrity_invalid")

    def test_inspect_does_not_reparse_curve_when_training_check_fails(self):
        cfg = config()
        (self.folder/"config.json").write_text(json.dumps(cfg))
        target = self.folder/"seed_0/round_0"
        target.mkdir(parents=True)
        (target/"learning_curve.jsonl").write_text('CORRUPT\n')
        with patch.object(report, "curve", wraps=report.curve) as parser, \
             patch.object(report, "check_training", side_effect=ValueError("checkpoint mismatch")):
            result = report.inspect_round(self.folder, cfg, 0, 0, np.ones(5), None)
        self.assertEqual(parser.call_count, 1)
        self.assertEqual(result["status"], "integrity_invalid")
        self.assertIn("checkpoint mismatch", result["training"]["integrity_error"])
        self.assertEqual(result["training"]["curve"]["parse_status"], "integrity_invalid")

    def test_critic_treatment_changes_keep_controlled_pair(self):
        base = config()
        candidate = dict(base, q_weight=1., critic_variant="huber_residual", weights=[1, .15, .35, 0, 0])
        self.assertTrue(report.controlled_pair(candidate, base, 0, 0)["eligible"])

    def test_common_identity_checked_per_seed(self):
        base = config()
        candidate = copy.deepcopy(base)
        candidate["common_checkpoint_sha256"]["1"] = "different"
        self.assertTrue(report.controlled_pair(candidate, base, 0, 0)["eligible"])
        result = report.controlled_pair(candidate, base, 1, 0)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["mismatches"][0]["field"], "common_checkpoint_sha256")

    def test_budget_flow_and_missing_init_prevent_control_claim(self):
        base = config()
        candidate = copy.deepcopy(base)
        candidate.update(online_updates=601, online_flow_update=True, common_checkpoint_sha256={})
        result = report.controlled_pair(candidate, base, 0, 1)
        self.assertFalse(result["eligible"])
        self.assertEqual({r["field"] for r in result["mismatches"]},
                         {"common_checkpoint_sha256", "online_updates", "online_flow_update"})

    def test_missing_is_not_failure_and_failures_have_no_numeric_PPA(self):
        rows = [row(0, 100), row(1, 0, "physical_failure"),
                dict(state="missing", predicted_q=None, measured_utility=None)]
        result = report.summarize_rows(rows, 3)
        self.assertFalse(result["fully_accounted"])
        self.assertEqual((result["success"], result["physical_failure"], result["missing"]), (1, 1, 1))
        self.assertEqual(result["successful_native_metric_means"]["rWL_um"], 100.)
        self.assertTrue(result["metrics_conditional_on_success"])
        self.assertEqual(result["utility_mean"], -52.)

    def test_seed_statistics_wait_for_complete_matching_plans(self):
        cfg = dict(config(), seeds=[0, 1, 2])
        exp = Path("/declared/test")
        records = [stream(exp, seed, 10+10*seed) for seed in cfg["seeds"]]
        result = report.seed_aggregates(records, [(exp, cfg)])[0]
        self.assertTrue(result["aggregate_eligible"])
        self.assertEqual(result["metrics"]["rWL_um"], dict(n=3, mean=21., sd=10.))
        records[-1]["roles"]["evaluation"]["fully_accounted"] = False
        result = report.seed_aggregates(records, [(exp, cfg)])[0]
        self.assertFalse(result["aggregate_eligible"])
        self.assertIsNone(result["metrics"])

    def test_same_noise_delta_retained_but_mismatched_init_is_descriptive(self):
        basepath, targetpath = Path("/declared/bc"), Path("/declared/fql")
        base = config()
        target = dict(base, arm="fql", q_weight=1.)
        records = [stream(basepath, 0, 100), stream(targetpath, 0, 90)]
        scales = {str(basepath): np.ones(5), str(targetpath): np.ones(5)}
        pairs, summaries = report.matched_deltas(records, [(basepath, base), (targetpath, target)], scales)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(summaries[0]["controlled_comparison_eligible"])
        self.assertEqual(summaries[0]["utility_delta"]["mean"], 10.)
        target["common_checkpoint_sha256"] = {"0": "another init"}
        pairs, summaries = report.matched_deltas(records, [(basepath, base), (targetpath, target)], scales)
        self.assertTrue(summaries[0]["aggregate_eligible"])
        self.assertFalse(summaries[0]["controlled_comparison_eligible"])
        self.assertTrue(all(not r["controlled_comparison_eligible"] for r in pairs))
        self.assertEqual(summaries[0]["utility_delta"]["mean"], 10.)

    def test_missing_geometry_audit_never_calls_backend_validator(self):
        cfg = dict(config(), online_rounds=0, evaluation_samples=1)
        (self.folder/"config.json").write_text(json.dumps(cfg))
        tasks = self.folder/"seed_0/round_0/requests/tasks.jsonl"
        tasks.parent.mkdir(parents=True)
        tasks.write_text("{}\n")
        tasks.with_name("completed.json").write_text("{}")
        sample, physical = self.folder/"sample", self.folder/"physical"
        sample.mkdir(); physical.mkdir()
        (physical/"manifest.json").write_text('{"success":true}')
        request = dict(arm="bc", seed=0, round=0, role="evaluation", index=0, graph="bp_fe",
            training_feedback_eligible=False, terminal_mask=0, checkpoint_sha256="checkpoint",
            noise_sha256="noise0", noise_seed=4600000, sample_dir=str(sample),
            raw_action=[[0., 0.]], executed_action=[[.1, .1]], predicted_q=[-1.])
        (sample/"request.json").write_text(json.dumps(request))
        (sample/"actions.npz").write_bytes(b"unused mock input")
        (sample/"outcome.json").write_text(json.dumps(dict(request, physical_run=str(physical), status="success")))
        with patch.object(report, "check_training", return_value=dict(completed=True, checkpoint_sha256="checkpoint")), \
             patch("fql.graph_o2o_backend.load_requests", return_value=([request], tasks)), \
             patch("fql.qstudy_backend.checked_request", return_value=sample), \
             patch("fql.qstudy_backend.checked_existing") as validator:
            result = report.inspect_round(self.folder, cfg, 0, 0, np.ones(5), (cfg, None, {}))
        validator.assert_not_called()
        self.assertEqual(result["rows"][0]["state"], "invalid")
        self.assertIn("Missing saved native geometry audit", result["rows"][0]["integrity_error"])


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run bounded report tests with Slurm")
    unittest.main()
