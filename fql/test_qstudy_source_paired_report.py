"""Meaningful bounded CPU tests; no training/inference/EDA is performed."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_source_paired_report as report


def metrics(wl):
    return dict(zip(report.native.METRICS, [wl, -.3, -50., 2., 3., .2, 70000.]))


def record(arm, seed=0, wl=100., failure=None):
    rows = []
    for index in range(8):
        state = failure if index == 0 and failure else "success"
        rows.append(dict(role="evaluation", index=index, noise_sha256="noise"+str(index),
            state=state, metrics=metrics(wl) if state == "success" else None,
            measured_utility=-999., predicted_q=-777., request_sha256=f"{arm}:{seed}:{index}",
            raw_to_legalizer_rms=.01))
    return dict(experiment="/"+arm, seed=seed, rows=rows, errors=[],
        training=dict(completed=True, checkpoint_sha256="checkpoint"+str(seed),
            completion_sha256="completion", binding_sha256="binding", steps=1000))


def proof():
    return dict(seeds=[dict(seed=seed, completed_training=record("BC", seed)["training"]) for seed in range(3)])


def streams():
    return [report.evaluation_stream(record(arm, seed, wl), arm, 100., proof())
            for arm, wl in (("BC", 100.), ("O64", 90.), ("T64", 95.)) for seed in range(3)]


class SourcePairedTests(unittest.TestCase):
    def test_common_utility_ignores_old_source_scaled_utility_and_Q(self):
        result = report.evaluation_stream(record("O64", wl=120.), "O64", 100., proof())
        self.assertEqual(result["common_utility_mean"], -1.2)
        self.assertNotIn("measured_utility", result["rows"][0])
        self.assertNotIn("predicted_q", result["rows"][0])

    def test_failure_has_penalty_without_fake_native_metrics(self):
        result = report.evaluation_stream(record("O64", failure="physical_failure"), "O64", 100., proof())
        self.assertTrue(result["fully_accounted"])
        self.assertEqual(result["physical_failure"], 1)
        self.assertEqual(result["common_utility_mean"], -11/8)
        self.assertIsNone(result["rows"][0]["metrics"])
        self.assertEqual(result["successful_native_metric_means"]["rWL_um"], 100.)
        self.assertTrue(result["conditional_PPA"])

    def test_missing_never_becomes_failure_or_partial_allrequest_utility(self):
        result = report.evaluation_stream(record("O64", failure="missing"), "O64", 100., proof())
        self.assertFalse(result["fully_accounted"])
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["physical_failure"], 0)
        self.assertIsNone(result["common_utility_mean"])

    def test_bad_successful_metric_is_invalid_and_cannot_be_utility(self):
        raw = record("O64")
        raw["rows"][0]["metrics"]["Power_W"] = float("nan")
        result = report.evaluation_stream(raw, "O64", 100., proof())
        self.assertTrue(result["integrity_invalid"])
        self.assertFalse(result["fully_accounted"])
        self.assertIsNone(result["rows"][0]["common_o64_train_wl_utility"])

    def test_certificate_BC_checkpoint_identity_is_rechecked(self):
        raw = record("BC")
        raw["training"]["checkpoint_sha256"] = "different"
        result = report.evaluation_stream(raw, "BC", 100., proof())
        self.assertTrue(result["integrity_invalid"])
        self.assertFalse(result["fully_accounted"])
        self.assertIn("checkpoint_sha256", result["errors"][0])

    def test_duplicate_noise_invalidates_complete_plan(self):
        raw = record("O64")
        raw["rows"][1]["noise_sha256"] = raw["rows"][0]["noise_sha256"]
        result = report.evaluation_stream(raw, "O64", 100., proof())
        self.assertTrue(result["integrity_invalid"])
        self.assertFalse(result["fully_accounted"])

    def test_same_noises_and_independent_seed_mean_SD(self):
        values = streams()
        for s in values:
            if s["arm"] == "O64":
                for r in s["rows"]:
                    r["metrics"]["rWL_um"] += s["seed"]*10
                s["successful_native_metric_means"]["rWL_um"] += s["seed"]*10
        arms, pairs, contrasts = report.aggregate(values)
        target = next(a for a in arms if a["arm"] == "O64")
        self.assertEqual(target["metrics"]["rWL_um"], dict(n=3, mean=100., sd=10., eligible=True))
        self.assertTrue(all(c["aggregate_eligible"] for c in contrasts))
        self.assertEqual(len(pairs), 72)

    def test_missing_one_seed_withholds_aggregate_and_winner(self):
        values = streams()
        values[-1] = report.evaluation_stream(record("T64", 2, failure="missing"), "T64", 100., proof())
        arms, _, contrasts = report.aggregate(values)
        self.assertFalse(arms[-1]["aggregate_eligible"])
        self.assertIsNone(arms[-1]["metrics"])
        self.assertFalse(next(c for c in contrasts if c["target"] == "T64")["aggregate_eligible"])

    def test_cross_seed_actual_noise_mismatch_blocks_comparison(self):
        values = streams()
        target = next(s for s in values if s["arm"] == "O64" and s["seed"] == 2)
        target["actual_noise_by_index"]["0"] = "other"
        target["rows"][0]["noise_sha256"] = "other"
        _, pairs, contrasts = report.aggregate(values)
        contrast = next(c for c in contrasts if c["target"] == "O64")
        self.assertFalse(contrast["aggregate_eligible"])
        self.assertFalse(contrast["common_noise_plan_across_seeds"])
        self.assertFalse(next(p for p in pairs if p["target"] == "O64" and p["seed"] == 2 and p["index"] == 0)["pair_admitted"])

    def test_partial_failure_pair_PPA_and_allrequest_utility_are_separate(self):
        values = streams()
        values[3] = report.evaluation_stream(record("O64", 0, wl=90., failure="geometry_failure"), "O64", 100., proof())
        _, _, contrasts = report.aggregate(values)
        c = contrasts[0]
        self.assertTrue(c["aggregate_eligible"])
        self.assertTrue(c["conditional_PPA"])
        self.assertEqual(c["per_seed"][0]["successful_pairs"], 7)
        self.assertAlmostEqual(c["per_seed"][0]["common_utility_delta"], (-3+7*.1)/8)
        self.assertEqual(c["per_seed"][0]["metric_deltas"]["rWL_um"], -10.)

    def test_no_success_seed_cannot_silently_reduce_PPA_seed_denominator(self):
        values = streams()
        raw = record("O64", 0)
        for row in raw["rows"]:
            row.update(state="physical_failure", metrics=None)
        values[3] = report.evaluation_stream(raw, "O64", 100., proof())
        arms, _, contrasts = report.aggregate(values)
        self.assertTrue(arms[1]["aggregate_eligible"])
        self.assertFalse(arms[1]["metrics"]["rWL_um"]["eligible"])
        self.assertIsNone(arms[1]["metrics"]["rWL_um"]["mean"])
        self.assertFalse(contrasts[0]["metric_deltas"]["rWL_um"]["eligible"])
        self.assertEqual(contrasts[0]["common_utility_delta"]["n"], 3)

    def test_integrity_invalid_stream_pair_never_exports_numeric_difference(self):
        values = streams()
        values[3]["integrity_invalid"] = True
        values[3]["fully_accounted"] = False
        _, pairs, _ = report.aggregate(values)
        bad = [p for p in pairs if p["target"] == "O64" and p["seed"] == 0]
        self.assertTrue(all(p["common_utility_delta"] is None for p in bad))

    def test_forged_certificate_rejected_before_any_evidence_load(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            path = Path(tmp)/"certificate.json"
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "certificate SHA"):
                report.verify_certificate(path, {})

    def test_valid_certificate_binds_configs_common_and_critic_files(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
            folder = Path(tmp)
            def artifact(name):
                path = folder/name
                path.write_text(name)
                return report.fingerprint(path)
            settings = dict(design="bp_fe", cache_sha256="cache", seeds=[0,1,2], alpha=10.,
                offline_actor_steps=1000, batch_size=16, flow_steps=10, actor_lr=1e-5,
                online_flow_update=False, offline_critic_update=False, evaluation_samples=8,
                evaluation_noise_seed=4600000, evaluation_purpose="development_validation",
                backend_threads=8, backend_storage="scratch")
            commons = {str(s):artifact("common"+str(s)) for s in range(3)}
            settings["common_checkpoint_sha256"] = {s:r["sha256"] for s,r in commons.items()}
            cert = dict(schema="qstudy_bc_effective_control_certificate_v1", status="verified",
                exact_effective_update_control_eligible=True, no_new_training=True,
                independently_retrained_bitwise_equivalence_demonstrated=False,
                actor_source=report.fingerprint(report.ROOT/"fql/qstudy_actor.py"),
                implementation=artifact("implementation"), cache=artifact("cache"),
                effective_settings=settings, configs={}, seeds=[])
            critics = {arm:{str(s):artifact(arm+str(s)+"critic") for s in range(3)} for arm in report.ARMS}
            plans = {}
            for arm in report.ARMS:
                exp = folder/arm
                exp.mkdir()
                cfg = dict(settings, weights=[1,0,0,0,0], critic_variant="legacy", physical_failure_reward=-4,
                    terminal_mask=0, online_rounds=0, q_weight=0 if arm=="BC" else 1,
                    common_checkpoints={s:r["path"] for s,r in commons.items()},
                    critic_checkpoints={s:r["path"] for s,r in critics[arm].items()},
                    critic_checkpoint_sha256={s:r["sha256"] for s,r in critics[arm].items()})
                (exp/"config.json").write_text(json.dumps(cfg))
                cert["configs"][arm] = report.fingerprint(exp/"config.json")
                plans[arm] = (exp,cfg)
            for seed in range(3):
                cert["seeds"].append(dict(seed=seed, common=commons[str(seed)],
                    zero_q_critic_probes={arm:dict(checkpoint=critics[arm][str(seed)]) for arm in report.ARMS}))
            path = folder/"certificate.json"
            path.write_text(json.dumps(cert))
            with patch.object(report,"CERTIFICATE_SHA",report.native.sha(path)):
                self.assertTrue(report.verify_certificate(path,plans)["exact_effective_update_control_eligible"])
                Path(commons["1"]["path"]).write_text("tampered")
                with self.assertRaisesRegex(ValueError,"common initialization changed"):
                    report.verify_certificate(path,plans)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run bounded report tests under Slurm")
    unittest.main()
