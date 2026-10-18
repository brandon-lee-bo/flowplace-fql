"""Correctness tests for cost objectives and learned action gradients.

These tiny synthetic tests establish implementation properties, not placement
performance. Run under the project's Slurm CPU/GPU test allocation.
"""
import copy
import io
import unittest
from unittest import mock

import torch
from torch import nn

from .qstudy_models import (MultiMetricCritic, frozen_critic, measured_costs,
                            multimetric_actor_loss, smooth_macro_hpwl)


class QStudyCriticTests(unittest.TestCase):
    @staticmethod
    def cache():
        return dict(node_id_to_name=["a", "b", "c"], ratio_x=10/224, ratio_y=20/224,
                    node_info={name: {"x": 1., "y": 1.2} for name in "abc"},
                    net_info={"n0": {"nodes": {"a": {"x_offset": .2}, "b": {},
                                                  "c": {"y_offset": -.3}}},
                              "n1": {"nodes": {"a": {}, "c": {}}}})

    def setUp(self):
        self.actions = torch.rand((8, 3, 2), generator=torch.Generator().manual_seed(14)) * 1.2 - .6
        self.costs = torch.stack((10 + self.actions[:, 0, 0],
                                 .4 + .1 * self.actions[:, 1, 0],
                                 30 + 2 * self.actions[:, 2, 1]), -1)
        self.names = ("rWL_um", "WNS_ns", "TNS_ns")
        self.movable = torch.tensor([True, True, True])

    def make(self, **kwargs):
        options = dict(width=12, depth=2, ensemble_size=2, seed=11)
        options.update(kwargs)
        return MultiMetricCritic(self.cache(), self.actions, self.costs,
                                 self.movable, self.names, **options)

    def test_native_slack_sign_missing_and_objective_units(self):
        native = torch.tensor([[1e6, -.5, -20.], [1.1e6, .1, float("nan")]])
        costs, mask = measured_costs(native, self.names)
        torch.testing.assert_close(costs[0], torch.tensor([1e6, .5, 20.]))
        self.assertEqual(costs[1, 1].item(), 0.)
        self.assertTrue(torch.isnan(costs[1, 2]))
        self.assertFalse(mask[1, 2])
        with self.assertRaisesRegex(ValueError, "negative observed cost"):
            measured_costs(torch.tensor([[-1.]]), ["rWL_um"])

    def test_train_only_stats_do_not_change_on_heldout_calls(self):
        model = self.make(train_splits=["train"] * 8, train_ids=[f"family{x}" for x in range(8)])
        torch.testing.assert_close(model.target_mean, self.costs.mean(0))
        torch.testing.assert_close(model.objective_scale, self.costs.mean(0))
        before = copy.deepcopy(model.normalization_metadata())
        heldout = self.actions[:3] + 10
        heldout_costs = self.costs[:3] * 100
        model(heldout)
        model.loss(heldout, heldout_costs)
        self.assertEqual(model.normalization_metadata(), before)
        with self.assertRaisesRegex(ValueError, "TRAIN rows only"):
            self.make(train_splits=["train"] * 7 + ["validation"])
        with self.assertRaisesRegex(ValueError, "uniquely"):
            self.make(train_ids=["repeated"] * 8)

    def test_masked_stats_ignore_unknown_labels_and_zero_costs_are_finite(self):
        costs = self.costs.clone()
        costs[:, 1] = 0
        costs[:3, 2] = float("nan")
        model = MultiMetricCritic(self.cache(), self.actions, costs, self.movable, self.names,
                                 width=8, seed=5)
        torch.testing.assert_close(model.target_mean[2], self.costs[3:, 2].mean())
        self.assertEqual(model.train_observed_count.tolist(), [8, 8, 5])
        self.assertEqual(model.objective_scale[1].item(), 1.)
        self.assertTrue(torch.isfinite(model(self.actions)).all())
        scalar = model.observed_reward(costs, {"TNS_ns": 1})
        self.assertTrue(torch.isnan(scalar[:3]).all())
        self.assertTrue(torch.isfinite(scalar[3:]).all())
        self.assertTrue(torch.isfinite(model.observed_reward(costs, {"rWL_um": 1})).all())

    def test_reward_rewards_lower_physical_cost_and_independent_weights(self):
        model = self.make()
        base = model.target_mean[None].repeat(3, 1)
        base[1, 0] *= .8
        base[2, 2] *= .5
        wire_only = model.observed_reward(base, {"rWL_um": 1})
        self.assertGreater(wire_only[1], wire_only[0])
        self.assertEqual(wire_only[2].item(), wire_only[0].item())
        joint = model.observed_reward(base, {"rWL_um": 1, "TNS_ns": 1})
        self.assertGreater(joint[2], joint[0])
        torch.testing.assert_close(joint[0], torch.tensor(-2.))
        with self.assertRaisesRegex(ValueError, "unknown objective"):
            model.reward(self.actions, {"final_drt_not_measured": 1})
        with self.assertRaisesRegex(ValueError, "active metric"):
            model.reward(self.actions, [0, 0, 0])

    def test_uncertainty_penalty_is_mean_minus_disagreement(self):
        model = self.make()
        member_rewards = torch.tensor([[-1., -3.], [-2., -2.]])
        with mock.patch.object(model, "member_rewards", return_value=member_rewards):
            torch.testing.assert_close(model.reward(self.actions[:2], [1, 0, 0]), torch.tensor([-2., -2.]))
            torch.testing.assert_close(model.reward(self.actions[:2], [1, 0, 0], 2.), torch.tensor([-4., -2.]))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model.reward(self.actions, [1, 0, 0], -1)

    def test_ensemble_members_independent_and_seed_reproducible(self):
        before_rng = torch.random.get_rng_state().clone()
        first = self.make()
        torch.testing.assert_close(torch.random.get_rng_state(), before_rng)
        second = self.make()
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b)
        self.assertNotEqual(first.members[0][0].weight.data_ptr(), first.members[1][0].weight.data_ptr())
        self.assertFalse(torch.equal(first.members[0][0].weight, first.members[1][0].weight))
        # Default seeds must not accidentally clone every member through fork_rng.
        unseeded = self.make(seed=None)
        self.assertFalse(torch.equal(unseeded.members[0][0].weight, unseeded.members[1][0].weight))
        other_before = first.members[1][0].weight.detach().clone()
        with torch.no_grad():
            first.members[0][0].weight.add_(1)
        torch.testing.assert_close(first.members[1][0].weight, other_before)

    def test_checkpoint_roundtrip_preserves_normalizers_and_independent_values(self):
        model = self.make(hpwl_residual=True).eval()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        other = self.make(hpwl_residual=True, seed=99).eval()
        other.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(model(self.actions), other(self.actions))
        self.assertEqual(model.normalization_metadata(), other.normalization_metadata())
        self.assertNotEqual(model.members[0][0].weight.data_ptr(), other.members[0][0].weight.data_ptr())

    def test_checkpoint_rejects_changed_objective_semantics(self):
        model = self.make()
        other = MultiMetricCritic(self.cache(), self.actions, self.costs,
                                 self.movable, ("rWL_um", "TNS_ns", "WNS_ns"), width=12, seed=11)
        with self.assertRaisesRegex(ValueError, "specification differs"):
            other.load_state_dict(model.state_dict())

    def test_flat_and_grouped_shapes_match_for_both_architectures(self):
        for width, depth in ((128, 2), (256, 3)):
            model = self.make(width=width, depth=depth).eval()
            self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model.members[0]), depth + 1)
            self.assertEqual(tuple(model(self.actions).shape), (8, 2, 3))
            torch.testing.assert_close(model(self.actions), model(self.actions.flatten(1)))

    def test_pairwise_ranking_has_correct_cost_direction(self):
        model = self.make()
        costs = self.costs.clone()
        costs[:, 0] = torch.arange(1, 9)
        standardized = (costs - model.target_mean) / model.target_std
        perfect = standardized[:, None].expand(-1, 2, -1).clone().requires_grad_()
        wrong = (-standardized)[:, None].expand(-1, 2, -1).clone().requires_grad_()
        kwargs = dict(ranking_weight=1, difference_weight=1,
                      metric_weights={"rWL_um": 1}, pairs=[[0, 7], [1, 6]])
        with mock.patch.object(model, "standardized", return_value=perfect):
            _, good = model.loss(self.actions, costs, **kwargs)
        with mock.patch.object(model, "standardized", return_value=wrong):
            _, bad = model.loss(self.actions, costs, **kwargs)
        self.assertLess(good["ranking"], bad["ranking"])
        self.assertEqual(good["difference"].item(), 0.)
        self.assertGreater(bad["difference"].item(), 0.)
        # Minimize ranking loss: increase the predicted cost of the worse row.
        bad["ranking"].backward()
        self.assertLess(wrong.grad[7, :, 0].sum().item(), 0.)
        self.assertGreater(wrong.grad[0, :, 0].sum().item(), 0.)

    def test_pair_losses_ignore_ties_and_missing_values(self):
        model = self.make()
        costs = model.target_mean[None].repeat(8, 1)
        costs[0, 1] = float("nan")
        loss, parts = model.loss(self.actions, costs, regression="huber", ranking_weight=1,
                                 difference_weight=1, pairs=[[0, 1]], metric_weights={"WNS_ns": 1})
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["ranking"].item(), 0.)
        self.assertEqual(parts["difference"].item(), 0.)
        self.assertEqual(parts["pair_metric_count"], 0)
        _, ties = model.loss(self.actions, costs, ranking_weight=1, metric_weights={"rWL_um": 1})
        self.assertEqual(ties["ranking"].item(), 0.)
        with self.assertRaisesRegex(ValueError, "no observed active"):
            model.loss(self.actions, costs.new_full(costs.shape, float("nan")))

    def test_masked_regression_has_no_nan_gradient(self):
        model = self.make()
        costs = self.costs.clone()
        costs[0, 0] = float("nan")
        loss, parts = model.loss(self.actions, costs, regression="huber", ranking_weight=.2,
                                 difference_weight=.3, metric_weights=[.1, .2, .3])
        loss.backward()
        self.assertEqual(parts["observed_labels"], 23)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_frozen_critic_retains_action_gradient_and_restores_modes(self):
        model = self.make(dropout=.4).train()
        model.members[1].eval()
        first_parameter = next(model.parameters())
        first_parameter.requires_grad_(False)
        flags = [p.requires_grad for p in model.parameters()]
        modes = [m.training for m in model.modules()]
        actions = self.actions[:2].clone().requires_grad_()
        with frozen_critic(model):
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            q = model.reward(actions, {"rWL_um": 1})
            torch.testing.assert_close(q, model.reward(actions, {"rWL_um": 1}))
        q.sum().backward()
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(actions.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_actor_loss_stops_teacher_and_keeps_q_action_gradient(self):
        model = self.make()
        actions = self.actions[:2].clone().requires_grad_()
        target = actions.detach().clone().requires_grad_()
        loss, parts = multimetric_actor_loss(actions, target, model, {"rWL_um": 1}, self.movable)
        self.assertEqual(parts["distillation"].item(), 0.)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(actions.grad.abs().sum().item(), 0.)
        # Isolate the sign convention with a known measured-cost surrogate:
        # cost=a0+constant, so minimizing -reward must decrease coordinate a0.
        actions = self.actions[:2].clone().requires_grad_()
        with mock.patch.object(model, "forward", side_effect=lambda x: (
                x[:, 0, 0, None, None] + 20).expand(-1, 2, 3)):
            objective, _ = multimetric_actor_loss(actions, actions.detach(), model,
                                                 {"rWL_um": 1}, self.movable)
            objective.backward()
        self.assertTrue((actions.grad[:, 0, 0] > 0).all())

    def test_smooth_hpwl_is_an_explicit_analytic_prior_with_finite_gradient(self):
        model = self.make()
        x = self.actions[:1].double().requires_grad_()
        exact = model.extractor.hpwl(x)
        smooth = smooth_macro_hpwl(model.extractor, x, temperature=.02)
        self.assertGreaterEqual(smooth.item() + 1e-10, exact.item())
        self.assertTrue(torch.autograd.gradcheck(
            lambda p: smooth_macro_hpwl(model.extractor, p, .02), (x,), eps=1e-6, atol=1e-5))

    def test_residual_baseline_fits_train_only_and_only_selected_metric(self):
        probe = self.make()
        with torch.no_grad():
            hpwl = smooth_macro_hpwl(probe.extractor, self.actions, .05)
        costs = self.costs.clone()
        costs[:, 0] = 2 * hpwl + 100
        model = MultiMetricCritic(self.cache(), self.actions, costs, self.movable, self.names,
                                 width=8, seed=5, hpwl_residual=True, baseline_ridge=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        prediction = model(self.actions)
        torch.testing.assert_close(prediction[:, 0, 0], costs[:, 0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(model.baseline_slope[1:], torch.zeros(2))
        before = model.baseline_slope.detach().clone()
        model(self.actions + 100)
        torch.testing.assert_close(model.baseline_slope, before)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
