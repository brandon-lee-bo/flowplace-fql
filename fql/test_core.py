"""Tiny correctness tests; no placement-performance claims."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    from .core import OneStepActor, PlacementData, critic_diagnostics, policy_loss
    from .analytic import DifferentiableMacroHPWL, optimize_coordinates
    from .latency import benchmark_actor
except ImportError:
    from core import OneStepActor, PlacementData, critic_diagnostics, policy_loss
    from analytic import DifferentiableMacroHPWL, optimize_coordinates
    from latency import benchmark_actor


class CorrectnessTests(unittest.TestCase):
    def test_objective_mean_scaling_and_gradient_direction(self):
        # Q = 2*a0 - a1; at the teacher, Q guidance should increase a0 and reduce a1.
        action = torch.tensor([[0.2, -0.1]], requires_grad=True)
        teacher = action.detach().clone()
        mask = torch.ones(2, dtype=torch.bool)
        q = (2 * action[:, 0] - action[:, 1])[:, None].repeat(1, 2)
        loss, distill, _ = policy_loss(action, teacher, q, mask, alpha=10)
        loss.backward()
        self.assertEqual(distill.item(), 0)
        torch.testing.assert_close(action.grad, torch.tensor([[-2., 1.]]))
        # A coordinate mean must not change when coordinates are repeated.
        a = torch.ones(1, 2)
        a4 = a.repeat(1, 2)
        l2 = policy_loss(a, torch.zeros_like(a), torch.zeros(1, 2), mask, alpha=3, q_weight=0)[0]
        l4 = policy_loss(a4, torch.zeros_like(a4), torch.zeros(1, 2), torch.ones(4, dtype=torch.bool), alpha=3, q_weight=0)[0]
        self.assertEqual(l2.item(), l4.item())

    def test_fixed_coordinates_are_restored(self):
        actor = OneStepActor(4, [8], [True, True, False, False], [0., 0., 0.8, -0.6])
        out = actor(torch.zeros(3, 4), clip=True)
        torch.testing.assert_close(out[:, 2:], torch.tensor([[0.8, -0.6]]).repeat(3, 1))
        self.assertTrue(bool((out.abs() <= 1).all()))

    def test_parent_split_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            data = dict(noise=np.zeros((3, 2)), teacher_action=np.zeros((3, 2)),
                        raw_action=np.zeros((3, 2)), action=np.zeros((3, 2)),
                        reward=np.zeros(3), split=np.array([0, 1, 2]),
                        critic_split=np.array([0, 2, 1]), critic_parent_index=np.arange(3))
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "different splits"):
                PlacementData.load(path)
            data["critic_split"] = np.array([0, 1, 2])
            np.savez(path, **data)
            loaded = PlacementData.load(path)
            self.assertEqual(loaded.metadata["teacher_counts"], {"0": 1, "1": 1, "2": 1})

    def test_calibration_is_measured_against_labels(self):
        truth = np.array([1., 2., 2., 4.])
        perfect = np.stack([truth, truth], axis=-1)
        metrics = critic_diagnostics(perfect, truth)
        self.assertEqual(metrics["mae"], 0)
        self.assertAlmostEqual(metrics["spearman_r"], 1)
        bad = critic_diagnostics(-perfect, truth)
        self.assertGreater(bad["mae"], 0)
        self.assertAlmostEqual(bad["spearman_r"], -1)

    @staticmethod
    def small_cache():
        return {"node_id_to_name": ["a", "b", "c"], "ratio_x": 10/224, "ratio_y": 20/224,
                "node_info": {n: {"x": 1, "y": 1} for n in "abc"},
                "net_info": {"n": {"nodes": {"a": {"x_offset": 0.2}, "b": {}, "c": {"y_offset": -0.3}}}}}

    def test_analytic_hpwl_and_gradient_match_hand_computation(self):
        model = DifferentiableMacroHPWL(self.small_cache())
        x = torch.tensor([[[-0.8, -0.7], [0.1, 0.2], [0.9, 0.8]]], dtype=torch.float64, requires_grad=True)
        physical = (x.detach().numpy()[0] + 1) * np.array([10., 20.]) / 2
        physical[0, 0] += 0.2
        physical[2, 1] -= 0.3
        expected = np.ptp(physical, axis=0).sum()
        self.assertAlmostEqual(model(x).item(), expected, places=10)
        torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-5)
        torch.testing.assert_close(model(x), model(x.reshape(1, -1)))

    def test_coordinate_control_improves_proxy_without_batch_scaling_change(self):
        model = DifferentiableMacroHPWL(self.small_cache())
        target = torch.tensor([[-0.8, -0.7, 0.1, 0.2, 0.9, 0.8]], dtype=torch.float64)
        mask = torch.ones(6, dtype=torch.bool)
        fixed = torch.zeros(6, dtype=torch.float64)
        one, _ = optimize_coordinates(target, model, 20., mask, fixed, steps=10, anchor_weight=0)
        two, _ = optimize_coordinates(target.repeat(2, 1), model, 20., mask, fixed, steps=10, anchor_weight=0)
        self.assertLess(model(one).item(), model(target).item())
        torch.testing.assert_close(one[0], two[0])
        torch.testing.assert_close(one[0], two[1])

    def test_latency_reports_batch_and_restores_training_mode(self):
        actor = OneStepActor(4, [8], [True] * 4, [0.] * 4).train()
        result = benchmark_actor(actor, torch.zeros(3, 4), batch_sizes=[1, 2], warmup=1, repeats=2)
        self.assertTrue(actor.training)
        self.assertEqual(result["batches"]["2"]["output_shape"], [2, 4])
        self.assertGreater(result["batches"]["1"]["mean_call_ms"], 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
