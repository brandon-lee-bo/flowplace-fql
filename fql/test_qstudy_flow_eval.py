"""Source identity and no-training safety tests for the matched flow baseline."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from . import qstudy_flow_eval as q


def configs():
    common = dict(design="bp_fe", dataset_path="data.npz", dataset_sha256="data",
        cache_sha256="cache", evaluation_noise_seed=4600000, evaluation_samples=8,
        flow_steps=10, online_flow_update=False, physical_failure_reward=-4,
        backend_threads=8, evaluation_purpose="development_validation", seeds=[0, 1, 2],
        common_checkpoints={str(s): f"common{s}.pt" for s in range(3)},
        common_checkpoint_sha256={str(s): f"sha{s}" for s in range(3)})
    cfg = dict(copy.deepcopy(common), native_budget=24, backend_concurrency=2, arm="behavior_flow", run_prefix="flev1")
    left = dict(copy.deepcopy(common), arm="bc", q_weight=0)
    right = dict(copy.deepcopy(common), arm="legacy", q_weight=1)
    return cfg, left, right


class FlowEvalTests(unittest.TestCase):
    def test_matched_control_configuration(self):
        q.assert_matching_configs(*configs())

    def test_reject_different_common_initialization(self):
        cfg, a, b = copy.deepcopy(configs())
        b["common_checkpoint_sha256"]["1"] = "different"
        with self.assertRaisesRegex(ValueError, "common initialization"):
            q.assert_matching_configs(cfg, a, b)

    def test_reject_updated_behavior_flow(self):
        cfg, a, b = configs()
        b["online_flow_update"] = True
        with self.assertRaisesRegex(ValueError, "context"):
            q.assert_matching_configs(cfg, a, b)

    def test_reject_changed_sample_budget(self):
        cfg, a, b = configs()
        cfg["native_budget"] = 25
        with self.assertRaisesRegex(ValueError, "3x8"):
            q.assert_matching_configs(cfg, a, b)

    def test_actual_noise_exact_match_not_seed_only(self):
        a = np.zeros((11, 2), dtype=np.float32)
        row = dict(noise_seed=4600000, noise_sha256=q.action_sha(a))
        self.assertEqual(q.matched_noise(a, a.copy(), row, row), row["noise_sha256"])
        b = a.copy()
        b[0, 0] = .25
        with self.assertRaisesRegex(ValueError, "Actual"):
            q.matched_noise(a, b, row, row)

    def test_reject_noise_bad_dtype_and_nonfinite(self):
        a = np.zeros((11, 2), dtype=np.float32)
        row = dict(noise_seed=4600000, noise_sha256=q.action_sha(a))
        for b in (a.astype(np.float64), np.full_like(a, np.nan), np.full_like(a, 1.1)):
            with self.assertRaisesRegex(ValueError, "Actual"):
                q.matched_noise(a, b, row, row)

    def test_reject_actual_noise_forged_hash(self):
        a = np.zeros((11, 2), dtype=np.float32)
        row = dict(noise_seed=4600000, noise_sha256="forged")
        with self.assertRaisesRegex(ValueError, "source request"):
            q.matched_noise(a, a, row, row)

    def test_declaration_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "declaration.json"
            q.immutable_json(path, dict(planned=24))
            before = path.read_bytes()
            q.immutable_json(path, dict(planned=24))
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "immutable"):
                q.immutable_json(path, dict(planned=23))

    def test_source_noise_load_exact_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actions.npz"
            a = np.arange(22, dtype=np.float32).reshape(11, 2) / 25
            np.savez(path, noise=a)
            pair = dict(source_bc=dict(actions_npz=q.file_binding(path)),
                noise_sha256=q.action_sha(a), noise_shape=[11, 2])
            self.assertTrue(np.array_equal(q.source_noise(pair), a))
            np.savez(path, noise=-a)
            with self.assertRaisesRegex(ValueError, "input changed"):
                q.source_noise(pair)

    def test_eval_request_replay_or_source_tampering_rejected(self):
        cfg, _, _ = configs()
        pair = dict(seed=0, index=0, noise_sha256="noise", noise_shape=[11, 2])
        declaration = dict(planned=24)
        row = dict(arm="behavior_flow", role="evaluation", seed=0, round=0, index=0,
            source_kind=q.SOURCE_KIND, source_sampler=q.SAMPLER, source_state_key="flow",
            source_sampler_steps=10, training_feedback_eligible=False, offline_train_eligible=False,
            flow_eval_declaration_sha256=q.object_sha(declaration), source_noise_binding=pair,
            noise_sha256="noise", noise_seed=4600000, source_checkpoint="common0.pt",
            checkpoint_sha256="sha0", run_id="go2o_qsf_flev1_s0_i0")
        for key, changed in (("training_feedback_eligible", True), ("offline_train_eligible", True),
                             ("source_state_key", "actor"), ("source_sampler_steps", 20)):
            bad = dict(row, **{key: changed})
            with patch.object(q, "checked_request", return_value=Path("unused")):
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    q.verify_row(bad, argparse.Namespace(seed=0, round=0), cfg, {}, declaration, pair)

    def test_export_declared_24_preserves_all_seed_noise_indices(self):
        cfg, _, _ = configs()
        declaration = dict(pairs=[dict(seed=s, index=i) for s in range(3) for i in range(8)],
                           selection="all predeclared")
        with tempfile.TemporaryDirectory() as folder:
            exp = Path(folder)
            (exp / "declaration.json").write_text("{}")
            (exp / "dispatch.json").write_text("{}")
            emitted = []

            def fake_emit(experiment, config, physical, binding, pair, problem, flow, metric):
                emitted.append((pair["seed"], pair["index"]))
                target = experiment / f"seed_{pair['seed']}/round_0/requests"
                target.mkdir(parents=True, exist_ok=True)
                return dict(index=pair["index"], role="evaluation", training_feedback_eligible=False)

            with patch.dict(q.os.environ, SLURM_JOB_ID="123"), \
                    patch.object(q, "build_flow", return_value=(type("Problem", (), {"benchmark_info": {}})(), object())), \
                    patch.object(q, "MacroHPWL"), patch.object(q, "emit", side_effect=fake_emit), \
                    patch.object(q.torch.cuda, "empty_cache"):
                q.export(exp, cfg, dict(config_sha256="cfg"), declaration)
            self.assertEqual(emitted, [(s, i) for s in range(3) for i in range(8)])
            self.assertEqual(q.read(exp / "export_completed.json")["backend_calls"], 0)
            for seed in range(3):
                marker = q.read(exp / f"seed_{seed}/round_0/requests/completed.json")
                self.assertEqual(marker["count"], 8)
                self.assertTrue(marker["evaluation_never_replayed"])


if __name__ == "__main__":
    unittest.main()
