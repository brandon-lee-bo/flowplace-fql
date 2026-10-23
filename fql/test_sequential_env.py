"""V2 execution semantics and continuous projection checks; run under Slurm."""
import os
from types import SimpleNamespace
import unittest

import torch

from .sequential_env import (SequentialPlacementEnv, deterministic_macro_order,
                             geometry_audit, project_macro_action)


def problem(sizes, positions=None, fixed=None):
    sizes = torch.tensor(sizes, dtype=torch.float32)
    positions = torch.zeros_like(sizes) if positions is None else torch.tensor(positions, dtype=torch.float32)
    fixed = torch.zeros(len(sizes), dtype=torch.bool) if fixed is None else torch.tensor(fixed, dtype=torch.bool)
    return SimpleNamespace(x=positions, cond=SimpleNamespace(x=sizes, is_ports=fixed))


class SequentialEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run sequential environment tests through Slurm")
        torch.set_num_threads(1)

    def test_order_and_unknown_positions_do_not_leak_reference_answers(self):
        p = problem([[.2, .2], [.4, .4], [.4, .4], [0., 0.]],
                    [[.9, .9], [.8, .8], [.7, .7], [-1., 0.]],
                    [False, False, False, True])
        env = SequentialPlacementEnv(p)
        self.assertEqual(env.order, (1, 2, 0))
        self.assertEqual(deterministic_macro_order(p.cond.x, p.cond.is_ports), env.order)
        state = env.observe()
        self.assertEqual(tuple(state["partial_positions"].shape), (1, 4, 2))
        self.assertTrue(torch.equal(state["partial_positions"][0, :3], torch.zeros(3, 2)))
        self.assertTrue(torch.equal(state["partial_positions"][0, 3], p.x[3]))
        self.assertEqual(state["current_id"].tolist(), [1])
        self.assertEqual(state["placed"].tolist(), [[False, False, False, True]])
        state["partial_positions"].fill_(7)
        state["placed"].fill_(True)
        self.assertFalse(bool(env.placed[0]))
        self.assertTrue(torch.equal(env.positions[0], torch.zeros(2)))

    def test_legal_actions_identity_single_macro_changes_and_records(self):
        p = problem([[.3, .3], [.2, .2], [0., 0.]], fixed=[False, False, True],
                    positions=[[.8, .8], [.7, .7], [-1., 0.]])
        env = SequentialPlacementEnv(p)
        initial = env.observe()
        raw = torch.tensor([-.5, -.5], requires_grad=True)
        state, reward, done, info = env.step(raw)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["projection_distance"], 0.0)
        self.assertTrue(torch.equal(info["raw_action"], raw))
        self.assertTrue(torch.equal(info["executed_action"], raw))
        self.assertFalse(info["executed_action"].requires_grad)
        self.assertTrue(torch.equal(state["partial_positions"][0, 1:], initial["partial_positions"][0, 1:]))
        state, _, done, info = env.step(torch.tensor([[.4, .4]]))
        self.assertTrue(done and info["completed"] and info["terminal_reward_pending"])
        self.assertEqual(state["current_id"].tolist(), [-1])
        self.assertEqual(len(env.history), 2)
        self.assertTrue(env.history[-1]["done"])
        self.assertTrue(geometry_audit(env.positions, env.sizes)["is_legal"])
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            env.step(torch.zeros(2))

    def test_continuous_nearest_projection_and_raw_executed_distinction(self):
        env = SequentialPlacementEnv(problem([[.4, .4], [.4, .4]]))
        env.step(torch.zeros(2))
        prior = env.positions.clone()
        _, _, _, info = env.step(torch.zeros(2))
        # Equally close legal centers are +/- .4 on either axis; sorted x
        # resolves this deterministically to (-.4, 0), without a raster snap.
        torch.testing.assert_close(info["executed_action"], torch.tensor([-.4, 0.]))
        self.assertAlmostEqual(info["projection_distance"], .4, places=6)
        self.assertTrue(torch.equal(env.positions[0], prior[0]))
        self.assertTrue(torch.equal(info["raw_action"], torch.zeros(2)))
        self.assertTrue(geometry_audit(env.positions, env.sizes)["is_legal"])

    def test_boundary_projection_and_narrow_free_slot_are_not_grid_limited(self):
        p = problem([[.7, 1.8], [.7, 1.8], [.1, .2]],
                    [[-.65, 0.], [.65, 0.], [0., 0.]], [True, True, False])
        env = SequentialPlacementEnv(p)
        _, _, done, info = env.step(torch.tensor([3., .12345]))
        self.assertTrue(done and info["completed"])
        # The free channel permits x in [-.25,.25]; preserve the continuous y.
        torch.testing.assert_close(info["executed_action"], torch.tensor([.25, .12345]))
        isolated = SequentialPlacementEnv(problem([[.2, .4]]))
        _, _, _, info = isolated.step(torch.tensor([4., -4.]))
        torch.testing.assert_close(info["executed_action"], torch.tensor([.9, -.8]))

    def test_dead_end_is_terminal_and_does_not_forge_an_executed_action(self):
        env = SequentialPlacementEnv(problem([[2., 2.], [.1, .1]]))
        env.step(torch.zeros(2))
        previous = env.positions.clone()
        state, reward, done, info = env.step(torch.zeros(2))
        self.assertTrue(done)
        self.assertFalse(info["completed"])
        self.assertIsNone(info["executed_action"])
        self.assertEqual(info["failure_reason"], "no_legal_position")
        self.assertTrue(torch.equal(env.positions, previous))
        self.assertFalse(bool(env.placed[1]))
        self.assertEqual(reward, 0.0)  # Collector must attach declared failure reward.
        self.assertEqual(state["current_id"].item(), -1)

    def test_nonfinite_action_and_oversized_macro_fail_explicitly(self):
        for sizes, action, reason in [([[.2, .2]], [float("nan"), 0.], "nonfinite_action"),
                                      ([[2.1, .2]], [0., 0.], "macro_larger_than_canvas")]:
            with self.subTest(reason=reason):
                env = SequentialPlacementEnv(problem(sizes))
                _, _, done, info = env.step(torch.tensor(action))
                self.assertTrue(done)
                self.assertEqual(info["failure_reason"], reason)
                self.assertIsNone(info["executed_action"])

    def test_order_must_cover_movable_once_and_invalid_fixed_prefix_rejected(self):
        p = problem([[.4, .4], [.4, .4]])
        for order in [(0,), (0, 0), (0, 2)]:
            with self.subTest(order=order), self.assertRaises(ValueError):
                SequentialPlacementEnv(p, order)
        with self.assertRaisesRegex(ValueError, "fixed macros"):
            SequentialPlacementEnv(problem([[.4, .4], [.4, .4]], fixed=[True, True]))
        with self.assertRaisesRegex(ValueError, "fixed macros"):
            SequentialPlacementEnv(problem([[0., 0.]], positions=[[float("nan"), 0.]], fixed=[True]))

    def test_projection_chunks_and_padding_produce_same_legal_result(self):
        p = problem([[.4, .4], [.4, .4], [.2, .2]],
                    [[-.4, -.3], [.4, -.3], [0., 0.]], [True, True, False])
        first = project_macro_action(torch.tensor([-.4, -.3]), 2, p.x, p.cond.is_ports,
                                     p.cond.x, padding=.01, chunk_size=1)
        second = project_macro_action(torch.tensor([-.4, -.3]), 2, p.x, p.cond.is_ports,
                                      p.cond.x, padding=.01, chunk_size=2048)
        self.assertIsNone(first.failure_reason)
        self.assertTrue(torch.equal(first.executed, second.executed))
        placed = p.x.clone()
        placed[2] = first.executed
        self.assertTrue(geometry_audit(placed, p.cond.x)["is_legal"])


if __name__ == "__main__":
    unittest.main()
