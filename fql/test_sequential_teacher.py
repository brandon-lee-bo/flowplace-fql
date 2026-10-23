"""V2 conditional teacher contracts, with an optional public-weight check."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from . import sequential_teacher as module
from .flow_policy_config import PUBLIC_SHA


class FakeScheduler:
    def set_timesteps(self, steps):
        self.timesteps = torch.linspace(0, 1, steps+1)


class FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))
        self.mask_key = "is_ports"
        self.hard_constraint_scores = {"w_reg": .1, "w_dist": 1.}
        self._scheduler = FakeScheduler()
        self.calls = []

    def forward(self, x, cond, time):
        self.calls.append((x.clone(), cond.is_ports.clone(), time.clone()))
        # Intentionally moves even fixed nodes, so endpoint restoration is
        # necessary before calling the legalizer.
        return torch.ones_like(x) * self.weight


class SequentialTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run teacher tests through Slurm")
        torch.set_num_threads(1)

    def setUp(self):
        self.problem = SimpleNamespace(x=torch.tensor([[.8, .8], [.7, .7], [.75, .75]]),
            cond=SimpleNamespace(x=torch.full((3, 2), .2), is_ports=torch.tensor([False, False, True])))
        self.flow = FakeFlow()
        self.checksum = mock.patch.object(module, "checkpoint_sha", return_value=PUBLIC_SHA)
        self.loader = mock.patch.object(module, "load_model", return_value=(self.flow, None))
        self.checksum.start()
        self.load = self.loader.start()
        self.addCleanup(self.checksum.stop)
        self.addCleanup(self.loader.stop)
        self.teacher = module.SequentialFlowPlaceTeacher(self.problem, "cpu")

    def batch(self):
        positions = torch.tensor([[[-.7, -.5], [0., 0.], [.75, .75]],
                                  [[0., 0.], [.6, -.5], [.75, .75]]])
        placed = torch.tensor([[True, False, True], [False, True, True]])
        current = torch.tensor([1, 0])
        noise = torch.tensor([[[.1, .2], [.3, .4], [.5, .6]],
                              [[-.1, -.2], [-.3, -.4], [-.5, -.6]]])
        return positions, placed, current, noise

    def legalizer_factory(self, seen):
        positions, masks, _, _ = self.batch()
        def factory(**kwargs):
            index = len(seen)
            seen.append(kwargs)
            mask = kwargs["cond"].is_ports
            self.assertTrue(torch.equal(mask, masks[index]))
            def run(endpoint):
                self.assertTrue(torch.equal(endpoint[0, mask], positions[index, mask]))
                output = endpoint.clone()
                target = torch.tensor([[-.6, -.5], [.5, -.5], [.75, .75]])
                output[0, ~mask] = target[~mask]
                return output
            return SimpleNamespace(run=run)
        return factory

    def test_masks_known_positions_noise_and_public_options_are_consistent(self):
        positions, placed, current, noise = self.batch()
        seen = []
        before_positions, before_noise = positions.clone(), noise.clone()
        with mock.patch.object(module, "Legalizer", side_effect=self.legalizer_factory(seen)):
            output, reports = self.teacher.sample_flow(positions, placed, current, noise, return_info=True)
        self.load.assert_called_once_with(self.problem, device="cpu", steps=20,
                                         hard_constraint=True, w_reg=.1, w_dist=1.)
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(self.flow.calls), 40)
        for row in range(2):
            first, mask, time = self.flow.calls[row*20]
            self.assertTrue(torch.equal(mask, placed[row]))
            self.assertTrue(torch.equal(first[0, ~mask], noise[row, ~mask]))
            self.assertTrue(torch.equal(first[0, mask], positions[row, mask]))
            self.assertTrue(torch.equal(output[row, mask], positions[row, mask]))
            self.assertTrue(reports[row]["is_legal"])
            self.assertEqual(reports[row]["macro_forward_calls"], 20)
        self.assertTrue(torch.equal(positions, before_positions))
        self.assertTrue(torch.equal(noise, before_noise))
        self.assertTrue(torch.equal(self.problem.cond.is_ports, torch.tensor([False, False, True])))
        self.assertFalse(output.requires_grad)
        self.teacher.train()
        self.assertFalse(self.teacher.training or self.flow.training)
        self.assertFalse(any(p.requires_grad for p in self.teacher.parameters()))

    def test_teacher_does_not_consume_hidden_randomness(self):
        state = self.batch()
        with mock.patch.object(module, "Legalizer", side_effect=self.legalizer_factory([])):
            first = self.teacher.sample_flow(*state)
        torch.manual_seed(991)
        with mock.patch.object(module, "Legalizer", side_effect=self.legalizer_factory([])):
            second = self.teacher.sample_flow(*state)
        self.assertTrue(torch.equal(first, second))

    def test_frozen_checkpoint_and_terminal_completion(self):
        saved = copy.deepcopy(self.teacher.state_dict())
        self.teacher.load_state_dict(saved)
        wrong = copy.deepcopy(saved)
        wrong["model.weight"] += 1
        with self.assertRaisesRegex(ValueError, "Cannot replace"):
            self.teacher.load_state_dict(wrong)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            self.teacher.flow_loss(None)
        full = torch.tensor([[[-.6, -.5], [.5, -.5], [.75, .75]]])
        with mock.patch.object(module, "Legalizer") as legalizer:
            output, report = self.teacher.sample_flow(full, torch.ones(1, 3, dtype=torch.bool),
                    torch.tensor([-1]), torch.zeros_like(full), return_info=True)
        legalizer.assert_not_called()
        self.assertTrue(torch.equal(full, output))
        self.assertEqual(report[0]["macro_forward_calls"], 0)

    def test_invalid_inputs_cannot_smuggle_future_answers_or_bad_prefixes(self):
        position, placed, current, noise = self.batch()
        bad_future = position.clone()
        bad_future[0, 1] = torch.tensor([.1, .1])
        bad_fixed = position.clone()
        bad_fixed[0, 2, 0] += .01
        bad_prefix = position.clone()
        bad_prefix[0, 0] = bad_prefix[0, 2]
        cases = [(bad_future, placed, current, noise), (bad_fixed, placed, current, noise),
                 (bad_prefix, placed, current, noise), (position, placed, torch.tensor([0, 0]), noise),
                 (position, placed.float(), current, noise),
                 (position, placed, torch.tensor([-1, 0]), noise),
                 (position, placed, current, noise.double()),
                 (position[0], placed, current, noise)]
        for args in cases:
            with self.subTest(shapes=[tuple(x.shape) for x in args]), self.assertRaises(ValueError):
                self.teacher.sample_flow(*args)

    def test_fixed_node_movement_by_legalizer_is_rejected(self):
        state = tuple(x[:1] for x in self.batch())
        with mock.patch.object(module, "Legalizer", return_value=SimpleNamespace(run=lambda x: x+1)):
            with self.assertRaisesRegex(RuntimeError, "moved a fixed"):
                self.teacher.sample_flow(*state)

    def test_silent_illegal_completion_becomes_a_recorded_failure(self):
        state = tuple(x[:1] for x in self.batch())
        def invalid(x):
            out = x.clone()
            out[0, 1] = out[0, 0]
            return out
        with mock.patch.object(module, "Legalizer", return_value=SimpleNamespace(run=invalid)):
            with self.assertRaises(module.ConditionalCompletionError) as context:
                self.teacher.sample_flow(*state)
            self.assertFalse(context.exception.reports[0]["is_legal"])
            _, reports = self.teacher.sample_flow(*state, require_legal=False, return_info=True)
            self.assertGreater(reports[0]["overlap_pairs"], 0)


@unittest.skipUnless(os.environ.get("FQL_REAL_SEQUENTIAL_TEACHER") == "1",
                     "Opt-in actual public-checkpoint sampler test")
class RealSequentialTeacherTests(unittest.TestCase):
    def test_public_empty_prefix_equivalence_and_conditioned_invariants(self):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Real network execution requires Slurm")
        from baseline.api import generate, load_model, load_problem
        from .sequential_env import SequentialPlacementEnv
        device = os.environ.get("FQL_TEST_DEVICE", "cpu")
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        torch.manual_seed(20261022)
        p = load_problem("bp_fe", device)
        teacher = module.SequentialFlowPlaceTeacher(p, device)
        initial = {key: value.detach().clone() for key, value in teacher.state_dict().items()}
        env = SequentialPlacementEnv(p)
        state = env.observe()
        noise = 2*torch.rand(1, len(p.x), 2, device=device)-1
        first = teacher.sample_flow(**state, noise=noise)
        original, _ = load_model(p, device=device, steps=20, hard_constraint=True, w_reg=.1, w_dist=1.)
        expected = generate(original, p, noise=noise[0])[0]
        torch.testing.assert_close(first[0], expected, atol=0, rtol=0)
        del original
        prefix_reports = []
        for count in (1, max(1, len(env.order)//2), len(env.order)-1):
            fixed_ids = list(env.order[:count])
            positions = state["partial_positions"].clone()
            placed = state["placed"].clone()
            positions[:, fixed_ids] = first[:, fixed_ids]
            placed[:, fixed_ids] = True
            current = torch.tensor([env.order[count]], device=device)
            result, report = teacher.sample_flow(positions, placed, current, noise,
                                                 return_info=True, require_legal=False)
            self.assertTrue(torch.equal(result[placed], positions[placed]))
            self.assertTrue(bool(torch.isfinite(result).all()))
            prefix_reports.append(dict(prefix_count=count, **report[0]))
        again = teacher.sample_flow(**state, noise=noise)
        torch.testing.assert_close(again, first, atol=0, rtol=0)
        for key, value in initial.items():
            self.assertTrue(torch.equal(value, teacher.state_dict()[key]))
        report = dict(status="correctness_passed", checkpoint_sha256=PUBLIC_SHA,
                      empty_prefix_exact_public_match=True, fixed_prefix_preserved=True,
                      weights_unchanged=True, conditional_quality_is_not_asserted=True,
                      prefix_reports=prefix_reports, slurm_job_id=os.environ["SLURM_JOB_ID"], device=device)
        output = os.environ.get("FQL_SEQUENTIAL_TEST_REPORT")
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as target:
                target.write(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    unittest.main()
