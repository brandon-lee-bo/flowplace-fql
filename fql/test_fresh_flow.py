"""Necessary checks for the new flow/critic gradient interfaces."""
import unittest
import numpy as np
import torch
from torch import nn
from .fresh_flow import BehaviorFlow, student_loss
from .core import OneStepActor, Critics


class FreshFlowTests(unittest.TestCase):
    def test_known_velocity_and_fixed_coordinates(self):
        flow = BehaviorFlow(2, [], [True, False], [0., .25])
        with torch.no_grad():
            flow.net[0].weight.zero_()
            flow.net[0].bias.copy_(torch.tensor([.4, 99.]))
        z = torch.tensor([[.1, -99.], [-.3, 99.]])
        np.testing.assert_allclose(flow.sample(z, 10).numpy(), [[.5, .25], [.1, .25]], atol=1e-6)
        # A straight path with displacement .4 has zero matching loss.
        a = torch.tensor([[.5, .25], [.1, .25]])
        self.assertLess(flow.bc_loss(a, z, torch.tensor([[.3], [.8]])).item(), 1e-12)

    def test_student_gradient_does_not_update_teacher_or_critic(self):
        torch.manual_seed(1)
        actor = OneStepActor(2, [8], [True, True], [0., 0.])
        critic = Critics(2, [8])
        flow = BehaviorFlow(2, [8], [True, True], [0., 0.])
        z = torch.randn(8, 2)
        target = flow.sample(z)
        loss, _, _ = student_loss(actor, z, target, critic, 10.)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        self.assertTrue(all(p.grad is None for p in flow.parameters()))
        self.assertTrue(all(p.requires_grad for p in critic.parameters()))

    def test_q_gradient_moves_action_toward_higher_value(self):
        actor = OneStepActor(1, [], [True], [0.])
        with torch.no_grad():
            actor.net[0].weight.zero_(); actor.net[0].bias.zero_()
        critic = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            critic.weight.fill_(2.)
        loss, _, _ = student_loss(actor, torch.zeros(2, 1), torch.zeros(2, 1), critic, 0.)
        loss.backward()
        self.assertLess(actor.net[0].bias.grad.item(), 0.)


if __name__ == '__main__':
    unittest.main()
