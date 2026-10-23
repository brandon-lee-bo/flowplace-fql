"""V2 model contracts; run under Slurm, using a small injected graph backbone."""
from __future__ import annotations

import copy
import os
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from .sequential_models import (SequentialActor, SequentialCritics, actor_objective,
                                bellman_targets, frozen_target, polyak_update)


def tiny_problem():
    return SimpleNamespace(
        x=torch.tensor([[-.7, -.4], [.4, -.5], [.1, .5], [.9, .9]]),
        cond=SimpleNamespace(x=torch.tensor([[.2, .2], [.2, .2], [.2, .2], [0., 0.]]),
            is_ports=torch.tensor([False, False, False, True]),
            edge_index=torch.tensor([[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 1]]),
            edge_attr=torch.ones(6, 1)))


class TinyTime(nn.Module):
    def forward(self, value):
        return value[:, None].expand(-1, 4)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._layer = nn.Linear(5, 8)

    def forward(self, value, cond, t):
        size = cond.x[None].expand(len(value), -1, -1)
        fixed = cond.is_ports[None, :, None].expand(len(value), -1, -1).float()
        return self._layer(torch.cat((value, size, fixed), -1))


class TinyGraphBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, value, cond, t):
        return torch.tanh(self.linear(value + value.mean(1, keepdim=True)))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._layer = nn.Linear(8, 2)


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_encoder = TinyTime()
        self._reverse_model = nn.Module()
        self._reverse_model.dir_att_input = True
        self._reverse_model._gnn_blocks = nn.ModuleList([TinyEncoder(), TinyGraphBlock(), TinyDecoder()])


def tiny_loader(problem, device="cpu", **kwargs):
    return TinyFlow().to(device), None


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(.15), requires_grad=False)

    def sample_flow(self, positions, placed, ids, noise):
        return torch.where(placed[..., None], positions, .3 * noise + self.anchor)


def small_batch():
    p = tiny_problem()
    positions = torch.zeros(2, 4, 2)
    positions[:, 3] = p.x[3]
    positions[1, 0] = torch.tensor([-.6, -.6])
    placed = torch.tensor([[False, False, False, True], [True, False, False, True]])
    return dict(positions=positions, placed=placed, current_id=torch.tensor([0, 1]),
                action=torch.tensor([[-.5, -.5], [.5, -.5]]))


@unittest.skipUnless("SLURM_JOB_ID" in os.environ, "Network tests require Slurm")
class SequentialModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.problem, self.batch = tiny_problem(), small_batch()
        self.actor = SequentialActor(self.problem, "cpu", tiny_loader)
        self.critic = SequentialCritics(self.problem, [-1., -.8], "cpu", 16, tiny_loader)
        self.noise = 2 * torch.rand(2, 4, 2) - 1

    def predict(self, batch=None):
        b = batch or self.batch
        return self.actor(b["positions"], b["placed"], b["current_id"], self.noise)

    def test_one_action_shape_and_unknown_coordinates_are_not_read(self):
        prediction = self.predict()
        self.assertEqual(prediction.shape, (2, 2))
        poisoned = {key: value.clone() for key, value in self.batch.items()}
        poisoned["positions"][~poisoned["placed"]] = 9000
        torch.testing.assert_close(self.predict(poisoned), prediction, atol=0, rtol=0)
        self.assertEqual(float(self.actor.encoder.fixed_positions[:3].abs().sum()), 0.)

    def test_actual_prefix_affects_next_action(self):
        before = self.predict()
        self.batch["positions"][1, 0] = torch.tensor([.6, .6])
        self.assertFalse(torch.equal(self.predict()[1], before[1]))

    def test_unobserved_latent_is_visible_to_actor(self):
        # A different unplaced macro's latent reaches current action through graph pooling.
        before = self.predict()
        self.noise[:, 2] *= -1
        self.assertFalse(torch.equal(self.predict(), before))

    def test_invalid_current_or_moved_fixed_node_is_rejected(self):
        self.batch["current_id"][0] = -1
        with self.assertRaises(ValueError):
            self.predict()
        self.batch = small_batch()
        self.batch["positions"][0, 3] = 0
        with self.assertRaises(ValueError):
            self.predict()

    def test_twin_critics_are_independent_and_ignore_future_coordinates(self):
        self.assertNotEqual(next(self.critic.members[0].parameters()).data_ptr(),
                            next(self.critic.members[1].parameters()).data_ptr())
        for net in self.critic.members:
            nn.init.constant_(net.head[-1].weight, .1)
        b = self.batch
        before = self.critic(b["positions"], b["placed"], b["current_id"], b["action"])
        altered = b["positions"].clone()
        altered[~b["placed"]] = -9000
        after = self.critic(altered, b["placed"], b["current_id"], b["action"])
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_actor_loss_backpropagates_only_to_student(self):
        teacher = TinyTeacher()
        loss, distill, q = actor_objective(self.actor, teacher, self.critic,
                                          self.batch, self.noise, alpha=3.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.critic.parameters()))
        self.assertIsNone(teacher.anchor.grad)
        self.assertTrue(all(p.requires_grad for p in self.critic.parameters()))

    def test_q_zero_is_exact_coordinate_distillation_gradient(self):
        clone = copy.deepcopy(self.actor)
        teacher = TinyTeacher()
        loss, _, _ = actor_objective(self.actor, teacher, self.critic,
                                     self.batch, self.noise, alpha=3., q_weight=0.)
        loss.backward()
        b = self.batch
        with torch.no_grad():
            full = teacher.sample_flow(b["positions"], b["placed"], b["current_id"], self.noise)
            target = full[torch.arange(2), b["current_id"]]
        output = clone(b["positions"], b["placed"], b["current_id"], self.noise, clip=False)
        (3 * (output-target).square().mean()).backward()
        for a, c in zip(self.actor.parameters(), clone.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, c.grad, atol=0, rtol=0)

    def test_saturated_raw_action_keeps_bc_gradient_but_q_sees_bounded_action(self):
        class RecordingQ(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2))
                self.actions = None
            def forward(self, positions, placed, ids, action):
                self.actions = action.detach().clone()
                return (action * self.weight).sum(-1)[:, None].expand(-1, 2)
        with torch.no_grad():
            self.actor.head.weight.zero_()
            self.actor.head.bias.copy_(torch.tensor([1.4, -1.6]))
        critic, teacher = RecordingQ(), TinyTeacher()
        noise = torch.zeros_like(self.noise)
        loss, distill, q, info = actor_objective(self.actor, teacher, critic,
            self.batch, noise, alpha=3., q_weight=1., return_info=True)
        loss.backward()
        # Every current-node noise is zero: raw action equals the head bias.
        # Q's clamp derivative is zero here; only unbounded BC can correct it.
        torch.testing.assert_close(self.actor.head.bias.grad,
                                   3 * (torch.tensor([1.4, -1.6]) - teacher.anchor))
        self.assertGreater(float(self.actor.head.bias.grad[0]), 0.)
        self.assertLess(float(self.actor.head.bias.grad[1]), 0.)
        torch.testing.assert_close(critic.actions, torch.tensor([[1., -1.], [1., -1.]]))
        self.assertIsNone(critic.weight.grad)
        self.assertEqual(info["raw_saturation_fraction"], 1.)
        self.assertTrue(torch.isfinite(distill) and torch.isfinite(q))

    def test_target_is_frozen_and_polyak_is_correct(self):
        target = frozen_target(self.critic)
        old = [p.clone() for p in target.parameters()]
        with torch.no_grad():
            for p in self.critic.parameters():
                p.add_(2)
        polyak_update(self.critic, target, .25)
        for before, after in zip(old, target.parameters()):
            torch.testing.assert_close(after, before + .5)
            self.assertFalse(after.requires_grad)

    def test_terminal_target_does_not_evaluate_terminal_networks(self):
        class MustNotRun:
            def __call__(self, *args):
                raise AssertionError("Terminal state queried")
        batch = dict(reward=torch.tensor([-1., -.8]), done=torch.tensor([True, True]))
        target = bellman_targets(MustNotRun(), MustNotRun(), batch, self.noise)
        torch.testing.assert_close(target, batch["reward"], atol=0, rtol=0)

    def test_nonterminal_target_bootstraps_mean_with_done_mask(self):
        batch = dict(reward=torch.tensor([0., -.8]), done=torch.tensor([False, True]),
                     next_positions=self.batch["positions"], next_placed=self.batch["placed"],
                     next_id=torch.tensor([0, -1]))
        class ConstantQ:
            def __call__(self, pos, placed, ids, action):
                return torch.tensor([[2., 4.]]).expand(len(pos), -1)
        target = bellman_targets(self.actor, ConstantQ(), batch, self.noise, gamma=1.)
        torch.testing.assert_close(target, torch.tensor([3., -.8]))
        self.assertFalse(target.requires_grad)


if __name__ == "__main__":
    unittest.main()
