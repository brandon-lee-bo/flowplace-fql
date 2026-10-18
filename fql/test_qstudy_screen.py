"""Independent screening-runner tests, using tiny synthetic Slurm-only fits.

Tests check the actual runner's minibatch stream, checkpoint recovery and group
isolation. They do not evaluate a placement model or select hyperparameters.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_screen as screen
from .qstudy_models import MultiMetricCritic


class ScreeningRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run screening-runner fits under Slurm")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        scratch = os.environ.get("TMPDIR", "/scratch/bohanli/flowplace_graph")
        self.temporary = tempfile.TemporaryDirectory(prefix="qstudy-screen-test-", dir=scratch)
        self.root = Path(self.temporary.name)
        self.cache = dict(node_id_to_name=["a", "b", "c"], ratio_x=10/224,
                          ratio_y=20/224,
                          node_info={name: dict(x=1., y=1.2) for name in "abc"},
                          net_info={"n": {"nodes": {"a": {"x_offset": .2},
                                                      "b": {}, "c": {"y_offset": -.3}}}})
        x = torch.rand((12, 3, 2), generator=torch.Generator().manual_seed(19)) * 1.2 - .6
        costs = torch.stack((100 + 3*x[:, 0, 0], .4 + .1*x[:, 1, 0],
                             25 + x[:, 2, 1], (3*x[:, 0, 1]).clamp_min(0),
                             (4*x[:, 1, 1]).clamp_min(0)), -1)
        self.context = dict(x=x, costs=costs, mask=torch.ones_like(costs, dtype=torch.bool),
                            families=torch.arange(6).repeat_interleave(2),
                            movable=torch.ones(3, dtype=torch.bool), cache=self.cache,
                            binding={"test": "synthetic_fixed_identity_v1"})
        self.indices = np.arange(len(x))
        screen.STOP = False

    def tearDown(self):
        screen.STOP = False
        self.temporary.cleanup()

    def arguments(self, directory, variant="multi_huber", steps=7):
        return SimpleNamespace(output=self.root/directory, variant=variant, seed=3,
                               lr=3e-4, weight_decay=1e-4, batch_size=8, steps=steps)

    def fit(self, args, *, intercept=None):
        """Capture only training batches, before eval-mode diagnostics."""
        original_make = screen.make_model

        def make(*positional, **keyword):
            model, normalization = original_make(*positional, **keyword)
            if intercept is not None:
                if screen.VARIANTS[args.variant].get("legacy"):
                    def before_forward(module, inputs):
                        if module.training:
                            intercept(inputs[0])
                    model.register_forward_pre_hook(before_forward)
                else:
                    original_loss = model.loss
                    def recorded_loss(actions, *rest, **options):
                        intercept(actions)
                        return original_loss(actions, *rest, **options)
                    model.loss = recorded_loss
            return model, normalization

        with mock.patch.object(screen, "make_model", side_effect=make), contextlib.redirect_stdout(io.StringIO()):
            return screen.fit(args, self.context, "fit", self.indices,
                              self.indices[:4], self.context)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b), "checkpoint tensors differ")
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for first, second in zip(a, b):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(a, b)

    def test_minibatches_are_paired_across_architecture_initialization(self):
        recorded = {}
        for variant in ("legacy", "multi_huber", "huber_large"):
            batches = []
            self.fit(self.arguments(variant, variant, steps=4),
                     intercept=lambda x: batches.append(x.detach().clone()))
            self.assertEqual(len(batches), 4)
            recorded[variant] = batches
        for first, second, third in zip(recorded["legacy"], recorded["multi_huber"], recorded["huber_large"]):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            torch.testing.assert_close(first, third, atol=0, rtol=0)

    def test_train_row_ids_unique_despite_shared_families(self):
        c = self.context
        with mock.patch.object(screen, "MultiMetricCritic", wraps=MultiMetricCritic) as constructor:
            model, normalization = screen.make_model(self.cache, c["x"], c["costs"],
                c["mask"], c["movable"], "multi_mse", c["families"], 4)
        ids = constructor.call_args.kwargs["train_ids"]
        self.assertEqual(len(ids), len(c["x"]))
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(constructor.call_args.kwargs["train_splits"], ["train"] * len(ids))
        encoded = json.dumps(ids, separators=(",", ":")).encode()
        self.assertEqual(normalization["train_id_sha256"], hashlib.sha256(encoded).hexdigest())
        torch.testing.assert_close(model.target_mean, c["costs"].mean(0))

    def test_interrupted_resume_exactly_matches_uninterrupted_training(self):
        for variant in ("legacy", "huber_rank", "huber_local"):
            with self.subTest(variant=variant):
                full_args = self.arguments("full_"+variant, variant)
                resumed_args = self.arguments("resumed_"+variant, variant)
                self.fit(full_args)
                captured = []
                def interrupt_after_three(x):
                    captured.append(x.detach().clone())
                    if len(captured) == 3:
                        screen.STOP = True
                with self.assertRaises(SystemExit) as interrupted:
                    self.fit(resumed_args, intercept=interrupt_after_three)
                self.assertEqual(interrupted.exception.code, 85)
                self.assertFalse((resumed_args.output/"fit/completed.json").exists())
                interrupted_checkpoint = torch.load(resumed_args.output/"fit/resume.pt", weights_only=False)
                self.assertEqual(interrupted_checkpoint["step"], 3)
                screen.STOP = False
                self.fit(resumed_args)
                full = torch.load(full_args.output/"fit/resume.pt", weights_only=False)
                resumed = torch.load(resumed_args.output/"fit/resume.pt", weights_only=False)
                for key in ("model", "optimizer", "rng", "batch_rng", "train_indices", "step", "normalization"):
                    self.assert_nested_equal(full[key], resumed[key])
                with np.load(full_args.output/"fit/predictions.npz") as a, np.load(resumed_args.output/"fit/predictions.npz") as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])

    def test_resume_rejects_changed_source_or_train_identity(self):
        args = self.arguments("identity", "multi_mse")
        def interrupt(_):
            screen.STOP = True
        with self.assertRaises(SystemExit):
            self.fit(args, intercept=interrupt)
        screen.STOP = False
        changed = copy.deepcopy(self.context)
        changed["binding"]["test"] = "changed"
        with self.assertRaisesRegex(ValueError, "Resume identity"):
            screen.fit(args, changed, "fit", self.indices, self.indices[:4], changed)
        with self.assertRaisesRegex(ValueError, "Resume identity"):
            screen.fit(args, self.context, "fit", self.indices[::-1].copy(),
                       self.indices[:4], self.context)

    def test_main_keeps_whole_families_out_of_each_fold(self):
        c = self.context
        native = np.column_stack((c["costs"].numpy(), np.ones((12, 2))))
        native[:, 1:3] *= -1
        validation_action = c["x"][:4].numpy() + .05
        data = {"train": dict(raw_action=c["x"].flatten(1).numpy(), metrics=native,
                              metric_mask=np.ones_like(native, dtype=bool), family=c["families"].numpy()),
                "validation": dict(raw_action=validation_action.reshape(4, -1), metrics=native[:4],
                                   metric_mask=np.ones_like(native[:4], dtype=bool), family=np.array([100, 100, 101, 101]))}
        dataset = self.root/"dataset.npz"
        dataset.write_bytes(b"fixture-only-hash-binding")
        dataset.with_name("manifest.json").write_text('{"fixture":true}\n')
        calls = []
        def inspect(args, context, name, train_idx, target_idx, target_data):
            train_families = set(context["families"][train_idx].tolist())
            target_families = set(target_data["families"][target_idx].tolist())
            self.assertTrue(train_families.isdisjoint(target_families))
            self.assertTrue(train_families.issubset(set(range(6))))
            calls.append((name, train_idx.copy(), target_idx.copy(), target_families))
            return dict(status="fixture")
        argv = ["qstudy_screen", "--dataset", str(dataset), "--output", str(self.root/"main"),
                "--variant", "multi_mse", "--folds", "3", "--steps", "1"]
        # Cache/model load are boundary stubs; CV grouping itself runs unmodified.
        fake_problem = SimpleNamespace(cond=SimpleNamespace(is_ports=torch.zeros(3, dtype=torch.bool)))
        with mock.patch("sys.argv", argv), mock.patch.object(screen, "load_dataset", return_value=(data, {"design": "bp_fe"})), \
             mock.patch.object(screen.torch, "load", return_value=self.cache), \
             mock.patch("baseline.api.load_problem", return_value=fake_problem), \
             mock.patch.object(screen, "fit", side_effect=inspect), \
             mock.patch.object(screen.signal, "signal"):
            screen.main()
        self.assertEqual([call[0] for call in calls], ["cv_0", "cv_1", "cv_2", "refit"])
        heldout_rows = np.concatenate([call[2] for call in calls[:3]])
        np.testing.assert_array_equal(np.sort(heldout_rows), self.indices)
        for family in range(6):
            self.assertEqual(sum(family in call[3] for call in calls[:3]), 1)
        np.testing.assert_array_equal(calls[-1][1], self.indices)
        self.assertEqual(calls[-1][3], {100, 101})
        binding = json.loads((self.root/"main/binding.json").read_text())
        self.assertFalse(binding["heldout_used"])
        self.assertEqual(binding["backend_calls"], 0)

    def test_pairwise_diagnostic_counts_only_measured_same_family_pairs(self):
        families = np.array([0, 0, 1, 1, 2, 2])
        truth = np.array([1., 3., 10., 9., 1., np.nan])
        prediction = np.array([1., 3., 9., 10., 1., -100.])
        result = screen.diagnostics(prediction, truth, families)
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["family_pairs"], 2)
        self.assertEqual(result["family_pair_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
