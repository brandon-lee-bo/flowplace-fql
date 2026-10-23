"""Trajectory, split, Bellman and exact-resume checks for the V2 entry point."""
from __future__ import annotations

import io
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .sequential_train import (SCHEMA, atomic_json, atomic_torch, binding_digest,
                               build_trajectories, checkpoint_state, construct,
                               normalize_config, prepare_data, restore_checkpoint,
                               read_completed_arm, rollout, sample_batch, seed_all, sha,
                               source_fingerprints, training_step, verify_source_binding,
                               run_training_stages, validate_export_checkpoint, prefix_quality_summary)
from .test_sequential_models import TinyTeacher, tiny_loader, tiny_problem


def measured_rows():
    p = tiny_problem()
    layouts = np.stack([p.x.numpy(), p.x.numpy()])
    layouts[1, :3, 0] *= -1
    return dict(action=layouts, reward=np.array([-1., -.8], dtype=np.float32),
                source_row_index=np.array([10, 11]), group_id=np.array([5, 5]))


@unittest.skipUnless("SLURM_JOB_ID" in os.environ, "Network tests require Slurm")
class SequentialTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_all(31)
        self.problem = tiny_problem()
        self.rows = measured_rows()
        self.data, self.metadata = build_trajectories(self.problem, self.rows["action"],
            self.rows["reward"], self.rows["source_row_index"], self.rows["group_id"])
        self.cfg = normalize_config(dict(batch_size=2, critic_width=16, alpha=3.))

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def warmed_common(self):
        """Serialized common fork with nonzero actor AND critic Adam moments."""
        modules = construct(self.problem, self.rows["reward"], self.cfg, "cpu", tiny_loader)
        batch = sample_batch(self.data, 2, "cpu")
        for stage in ("bc", "mc"):
            training_step(*modules[:3], TinyTeacher(), modules[3], batch, self.cfg, stage=stage)
        binding = dict(seed=31, config=self.cfg, protocol="matched-common-regression")
        buffer = io.BytesIO()
        torch.save(checkpoint_state(*modules, binding, stage="common_complete", step=0,
                                    arm="common"), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location="cpu", weights_only=False)
        for opt in saved["optimizers"].values():
            self.assertTrue(any(bool(value["exp_avg"].ne(0).any()) for value in opt["state"].values()))
        return saved, binding

    def test_replay_has_real_prefixes_and_only_terminal_rewards(self):
        self.assertEqual(self.metadata["backend_labels"], 2)
        self.assertEqual(self.metadata["transitions"], 6)
        self.assertEqual(self.metadata["noise_families"], 1)
        self.assertTrue(torch.equal(self.data["done"], torch.tensor([False, False, True] * 2)))
        self.assertEqual(int((self.data["reward"] != 0).sum()), 2)
        self.assertTrue(bool((self.data["positions"][~self.data["placed"]] == 0).all()))
        self.assertTrue(bool((self.data["next_id"][self.data["done"]] == -1).all()))
        torch.testing.assert_close(self.data["next_positions"][2], torch.tensor(self.rows["action"][0]))

    def test_family_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "family"):
            prepare_data(self.problem, dict(train=self.rows, validation=self.rows))

    def test_illegal_complete_layout_cannot_reuse_old_backend_reward(self):
        bad = self.rows["action"].copy()
        bad[:, 1] = bad[:, 0]
        with self.assertRaisesRegex(ValueError, "illegal"):
            build_trajectories(self.problem, bad, self.rows["reward"],
                               self.rows["source_row_index"], self.rows["group_id"])

    def test_mc_then_bellman_updates_are_distinct_and_finite(self):
        actor, critic, target, opts = construct(self.problem, self.rows["reward"], self.cfg,
                                               "cpu", tiny_loader)
        batch = sample_batch(self.data, 2, "cpu")
        teacher = TinyTeacher()
        mc = training_step(actor, critic, target, teacher, opts, batch, self.cfg, stage="mc")
        self.assertNotIn("actor_loss", mc)
        record = training_step(actor, critic, target, teacher, opts, batch, self.cfg,
                                stage="bellman", log_components=True)
        self.assertIn("actor_loss", record)
        self.assertIn("q_gradient_norm_unweighted", record)
        self.assertIn("raw_saturation_fraction", record)
        self.assertTrue(0 <= record["raw_saturation_fraction"] <= 1)
        self.assertTrue(all(np.isfinite(value) for value in record.values()))

    def test_exact_rng_optimizer_target_resume(self):
        modules = construct(self.problem, self.rows["reward"], self.cfg, "cpu", tiny_loader)
        actor, critic, target, opts = modules
        teacher = TinyTeacher()
        batch = sample_batch(self.data, 2, "cpu")
        training_step(actor, critic, target, teacher, opts, batch, self.cfg, stage="bellman")
        binding = dict(seed=31, protocol="unit-test")
        buffer = io.BytesIO()
        torch.save(checkpoint_state(*modules, binding, stage="bellman", step=1, arm="fql"), buffer)
        next_batch = sample_batch(self.data, 2, "cpu")
        expected_log = training_step(actor, critic, target, teacher, opts, next_batch, self.cfg,
                                     stage="bellman")
        expected = [{k: v.clone() for k, v in model.state_dict().items()} for model in modules[:3]]
        reloaded = construct(self.problem, self.rows["reward"], self.cfg, "cpu", tiny_loader)
        buffer.seek(0)
        saved = torch.load(buffer, map_location="cpu", weights_only=False)
        restore_checkpoint(saved, *reloaded, binding)
        actual_batch = sample_batch(self.data, 2, "cpu")
        for key in next_batch:
            torch.testing.assert_close(next_batch[key], actual_batch[key], atol=0, rtol=0)
        actual_log = training_step(*reloaded[:3], teacher, reloaded[3], actual_batch, self.cfg,
                                   stage="bellman")
        self.assertEqual(expected_log, actual_log)
        for old, model in zip(expected, reloaded[:3]):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, old[key], atol=0, rtol=0)

    def test_resume_rejects_changed_binding(self):
        modules = construct(self.problem, self.rows["reward"], self.cfg, "cpu", tiny_loader)
        saved = checkpoint_state(*modules, {"seed": 31}, stage="bc", step=0, arm="common")
        with self.assertRaisesRegex(ValueError, "identity"):
            restore_checkpoint(saved, *modules, {"seed": 32})

    def test_bc_fork_does_not_mutate_common_optimizer_or_rng_snapshot(self):
        common, binding = self.warmed_common()
        before = copy.deepcopy(common)
        modules = construct(self.problem, self.rows["reward"], self.cfg, "cpu", tiny_loader)
        restore_checkpoint(common, *modules, binding)
        batch = sample_batch(self.data, 2, "cpu")
        training_step(*modules[:3], TinyTeacher(), modules[3], batch, self.cfg,
                      stage="bellman", q_weight=0., log_components=True)
        self.assert_nested_equal(common, before)

    def test_repeated_common_restore_yields_identical_first_q0_update(self):
        common, binding = self.warmed_common()
        modules = construct(self.problem, self.rows["reward"], self.cfg, "cpu", tiny_loader)
        runs = []
        for _ in range(2):
            restore_checkpoint(common, *modules, binding)
            batch = sample_batch(self.data, 2, "cpu")
            log = training_step(*modules[:3], TinyTeacher(), modules[3], batch, self.cfg,
                                stage="bellman", q_weight=0., log_components=True)
            after = checkpoint_state(*modules, binding, stage="bellman", step=1, arm="bc")
            runs.append(copy.deepcopy((batch, log, after)))
        self.assert_nested_equal(runs[0], runs[1])

    def test_completed_skip_checks_seed_arm_binding_and_finished_step(self):
        common, binding = self.warmed_common()
        common.update(stage="completed", step=self.cfg["train_steps"], arm="bc")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "checkpoint.pt"
            report = dict(schema=SCHEMA, status="completed", arm="bc", seed=31,
                          binding_sha256=binding_digest(binding))
            atomic_torch(checkpoint, common)
            report["checkpoint_sha256"] = sha(checkpoint)
            atomic_json(directory / "completed.json", report)
            self.assertEqual(read_completed_arm(directory, binding, "bc"), report)
            cases = [dict(arm="fql"), dict(stage="bellman"), dict(step=0),
                     dict(binding={**binding, "seed": 32}),
                     dict(binding={**binding, "config": {**self.cfg, "alpha": 9.}})]
            for changes in cases:
                with self.subTest(changes=changes):
                    altered = {**common, **changes}
                    atomic_torch(checkpoint, altered)
                    # A matching file hash alone does not establish arm identity.
                    report["checkpoint_sha256"] = sha(checkpoint)
                    atomic_json(directory / "completed.json", report)
                    with self.assertRaises(ValueError):
                        read_completed_arm(directory, binding, "bc")
            atomic_torch(checkpoint, common)
            report["checkpoint_sha256"] = sha(checkpoint)
            for changes in (dict(seed=32), dict(arm="fql"), dict(binding_sha256="other")):
                atomic_json(directory / "completed.json", {**report, **changes})
                with self.assertRaisesRegex(ValueError, "report identity"):
                    read_completed_arm(directory, binding, "bc")

    def test_upstream_model_config_and_legalizer_drift_invalidate_binding(self):
        source = source_fingerprints()
        required = ("upstream/flowplace/flow_matching/models.py",
                    "upstream/flowplace/flow_matching/networks/gnn.py",
                    "upstream/flowplace/flow_matching/hard_constraint_legalizer.py",
                    "upstream/flowplace/flow_matching/configs/config_eval.yaml",
                    "upstream/flowplace/flow_matching/configs/model/size/large.yaml",
                    "fql/sequential_train.py", "backend/scripts/export_macro_def.py")
        self.assertTrue(set(required).issubset(source))
        verify_source_binding(dict(source_sha256=source))
        for dependency in required:
            with self.subTest(dependency=dependency):
                altered = {**source, dependency: "changed"}
                with self.assertRaisesRegex(ValueError, "dependency identity"):
                    verify_source_binding(dict(source_sha256=altered))

    def test_rollout_executes_one_macro_and_resamples_latent_each_time(self):
        class FixedActor:
            def __init__(self, positions):
                self.positions, self.calls = positions, 0
            def __call__(self, positions, placed, ids, noise):
                self.calls += 1
                return self.positions[ids]
        actor = FixedActor(self.problem.x)
        result = rollout(actor, self.problem, 197)
        self.assertTrue(result["completed"])
        self.assertEqual(actor.calls, 3)
        self.assertEqual(result["noises"].shape, (3, 4, 2))
        self.assertFalse(torch.equal(result["noises"][0], result["noises"][1]))
        torch.testing.assert_close(result["centers"], self.problem.x, atol=0, rtol=0)

    def test_protocol_rejects_changed_discount_or_implicit_control(self):
        with self.assertRaises(ValueError):
            normalize_config(dict(gamma=.99))
        with self.assertRaises(ValueError):
            normalize_config(dict(arms=["fql"]))

    def stage_fixture(self):
        cfg = normalize_config(dict(batch_size=2, critic_width=16, alpha=3.,
                                    bc_steps=3, critic_mc_steps=2, train_steps=2,
                                    checkpoint_every=1, log_every=1))
        binding = dict(seed=31, config=cfg, protocol="stage-resume-regression")
        seed_all(31)
        modules = construct(self.problem, self.rows["reward"], cfg, "cpu", tiny_loader)
        # This fixture tests scheduling, not validation quality: the same tiny
        # rows are intentionally passed to the diagnostic-only evaluator.
        data = dict(train=self.data, validation=self.data)
        return modules, binding, data

    def assert_training_checkpoints_equal(self, left, right):
        for name in ("common_bc.pt", "common.pt", "bc/checkpoint.pt", "fql/checkpoint.pt"):
            a = torch.load(Path(left) / name, map_location="cpu", weights_only=False)
            b = torch.load(Path(right) / name, map_location="cpu", weights_only=False)
            self.assert_nested_equal(a, b)

    def test_bc_stage_pause_and_resume_exactly_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            continuous, staged = Path(temporary) / "continuous", Path(temporary) / "staged"
            modules, binding, data = self.stage_fixture()
            run_training_stages(*modules, TinyTeacher(), data, binding, continuous, "cpu")
            modules, binding, data = self.stage_fixture()
            untouched_critic = copy.deepcopy(modules[1].state_dict())
            untouched_target = copy.deepcopy(modules[2].state_dict())
            untouched_actor = copy.deepcopy(modules[0].state_dict())
            with patch("fql.sequential_train.training_step", wraps=training_step) as updates:
                status = run_training_stages(*modules, TinyTeacher(), data, binding, staged,
                                             "cpu", stop_after="common_bc")
            self.assertEqual([c.kwargs["stage"] for c in updates.call_args_list], ["bc"] * 3)
            self.assertEqual(status["status"], "stopped_after_common_bc")
            self.assertFalse(status["common_bc"]["full_training_completed"])
            self.assertEqual(status["common_bc"]["critic_mc_updates"], 0)
            self.assertFalse((staged / "common.pt").exists())
            self.assertFalse((staged / "training_completed.json").exists())
            self.assertFalse((staged / "bc").exists())
            self.assertFalse((staged / "fql").exists())
            self.assert_nested_equal(modules[1].state_dict(), untouched_critic)
            self.assert_nested_equal(modules[2].state_dict(), untouched_target)
            self.assertFalse(modules[3]["critic"].state)
            self.assertTrue(any(not torch.equal(value, untouched_actor[key])
                                for key, value in modules[0].state_dict().items()))
            boundary_sha = sha(staged / "common_bc.pt")
            # A repeated stop request performs no updates and does not rewrite
            # the checkpoint; unrelated RNG consumption between jobs is safe.
            modules, _, _ = self.stage_fixture()
            seed_all(983)
            with patch("fql.sequential_train.training_step", wraps=training_step) as updates:
                repeated = run_training_stages(*modules, TinyTeacher(), data, binding, staged,
                                               "cpu", stop_after="common_bc")
            updates.assert_not_called()
            self.assertEqual(status, repeated)
            self.assertEqual(sha(staged / "common_bc.pt"), boundary_sha)
            # The immutable BC checkpoint alone is sufficient to resume; an
            # ordinary transient BC resume file is not required afterward.
            (staged / "common_resume.pt").unlink()
            modules, _, _ = self.stage_fixture()
            seed_all(711)
            with patch("fql.sequential_train.training_step", wraps=training_step) as updates:
                run_training_stages(*modules, TinyTeacher(), data, binding, staged, "cpu")
            self.assertEqual([c.kwargs["stage"] for c in updates.call_args_list],
                             ["mc"] * 2 + ["bellman"] * 4)
            self.assertEqual(sha(staged / "common_bc.pt"), boundary_sha)
            self.assert_training_checkpoints_equal(continuous, staged)
            self.assertEqual(binding["config"]["bc_steps"], 3)
            self.assertEqual(binding["config"]["train_steps"], 2)
            self.assertNotIn("stop_after", binding["config"])

    def test_interrupted_bc_and_mc_resume_without_repeating_finished_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            continuous = Path(temporary) / "continuous"
            modules, binding, data = self.stage_fixture()
            run_training_stages(*modules, TinyTeacher(), data, binding, continuous, "cpu")
            for interrupted_stage, interrupted_step in (("bc", 2), ("mc", 1)):
                with self.subTest(stage=interrupted_stage):
                    directory = Path(temporary) / interrupted_stage
                    modules, _, _ = self.stage_fixture()
                    stop, counts = {"requested": False}, {"bc": 0, "mc": 0, "bellman": 0}
                    def interrupting_update(*args, **kwargs):
                        result = training_step(*args, **kwargs)
                        counts[kwargs["stage"]] += 1
                        if kwargs["stage"] == interrupted_stage and counts[interrupted_stage] == interrupted_step:
                            stop["requested"] = True
                        return result
                    with patch("fql.sequential_train.training_step", side_effect=interrupting_update):
                        with self.assertRaises(InterruptedError):
                            run_training_stages(*modules, TinyTeacher(), data, binding, directory,
                                                "cpu", stop=stop)
                    self.assertFalse((directory / "common.pt").exists())
                    self.assertFalse((directory / "training_completed.json").exists())
                    self.assertEqual((directory / "common_bc.pt").exists(), interrupted_stage == "mc")
                    partial = torch.load(directory / "common_resume.pt", weights_only=False)
                    self.assertEqual((partial["stage"], partial["step"]),
                                     (interrupted_stage, interrupted_step))
                    modules, _, _ = self.stage_fixture()
                    with patch("fql.sequential_train.training_step", wraps=training_step) as updates:
                        run_training_stages(*modules, TinyTeacher(), data, binding, directory, "cpu")
                    stages = [c.kwargs["stage"] for c in updates.call_args_list]
                    self.assertEqual(stages.count("bc"), 1 if interrupted_stage == "bc" else 0)
                    self.assertEqual(stages.count("mc"), 2 if interrupted_stage == "bc" else 1)
                    self.assert_training_checkpoints_equal(continuous, directory)

    def test_bc_gate_export_cannot_masquerade_as_completed_arm(self):
        common, binding = self.warmed_common()
        pure_bc = {**common, "stage": "common_bc_complete", "step": self.cfg["bc_steps"]}
        validate_export_checkpoint(pure_bc, "bc_gate")
        for role in ("evaluation", "collect"):
            with self.assertRaises(ValueError):
                validate_export_checkpoint(pure_bc, role)
        for altered in ({**pure_bc, "stage": "bc"}, {**pure_bc, "step": 0},
                        {**pure_bc, "arm": "bc"}, {**pure_bc, "schema": "sequential_fql_v2_3"}):
            with self.assertRaises(ValueError):
                validate_export_checkpoint(altered, "bc_gate")
        finished = {**common, "stage": "completed", "step": self.cfg["train_steps"], "arm": "fql"}
        validate_export_checkpoint(finished, "evaluation")
        with self.assertRaises(ValueError):
            validate_export_checkpoint(finished, "bc_gate")

    def test_prefix_gate_retains_dead_ends_in_all_denominators(self):
        class FixedActor:
            def __call__(self, positions, placed, ids, noise):
                return tiny_problem().x[ids]
        result = rollout(FixedActor(), self.problem, 197)
        success = dict(rollout_completed=True, decisions=result["records"],
                       sequential_failure_reason=None, geometry=dict(legal=True),
                       action_sha256="layout1", executed_action=result["centers"].tolist())
        failed_decision = {**result["records"][0], "executed_action": None,
                           "failure_reason": "no_legal_position", "done": True, "completed": False}
        failure = {**success, "rollout_completed": False, "decisions": [failed_decision],
                   "sequential_failure_reason": "no_legal_position", "geometry": dict(legal=False)}
        report = prefix_quality_summary([success, failure], 3)
        self.assertEqual(report["completion_fraction"], .5)
        self.assertEqual(report["failures"], {"no_legal_position": 1})
        self.assertEqual([p["attempted"] for p in report["prefixes"]], [2, 1, 1])
        self.assertEqual([p["survived_fraction"] for p in report["prefixes"]], [.5, .5, .5])
        self.assertEqual(report["unique_complete_layouts"], 1)
        self.assertEqual(report["backend_measurements"], 0)
        self.assertEqual(report["teacher_completion_tests"], 0)


if __name__ == "__main__":
    unittest.main()
