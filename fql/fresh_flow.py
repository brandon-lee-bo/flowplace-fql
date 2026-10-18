"""FQL's three networks for a fixed-context, terminal placement action.

The behavior flow is learned from offline raw proposals. Fresh Gaussian noise
is distilled throughout training. No RL gradients cross flow integration or
the external legalizer. Terminal targets are measured rewards, not bootstrap.
"""
from __future__ import annotations

import torch
from torch import nn

from .core import mlp


class BehaviorFlow(nn.Module):
    def __init__(self, dim, widths, movable_mask, fixed_action):
        super().__init__()
        self.dim = dim
        self.net = mlp(dim + 1, dim, widths)
        self.register_buffer('movable_mask', torch.as_tensor(movable_mask, dtype=torch.bool))
        self.register_buffer('fixed_action', torch.as_tensor(fixed_action, dtype=torch.float32))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1)) * self.movable_mask

    def bc_loss(self, actions, noise, time):
        x0 = torch.where(self.movable_mask, noise, self.fixed_action)
        xt = (1 - time) * x0 + time * actions
        return (self(xt, time)[:, self.movable_mask] -
                (actions - x0)[:, self.movable_mask]).square().mean()

    @torch.no_grad()
    def sample(self, noise, steps=10):
        if steps < 1:
            raise ValueError('At least one Euler step required')
        x = torch.where(self.movable_mask, noise, self.fixed_action)
        for index in range(steps):
            t = x.new_full((len(x), 1), index / steps)
            x = x + self(x, t) / steps
        return torch.where(self.movable_mask, x.clamp(-1, 1), self.fixed_action)


def student_loss(actor, noise, teacher_target, critic, alpha, q_weight=1.):
    """Return loss without accumulating teacher or critic parameter gradients."""
    raw = actor(noise, clip=False)
    distill = (raw[:, actor.movable_mask] - teacher_target.detach()[:, actor.movable_mask]).square().mean()
    # Freezing weights retains derivatives with respect to the actor's action.
    flags = [p.requires_grad for p in critic.parameters()]
    for p in critic.parameters():
        p.requires_grad_(False)
    try:
        action = torch.where(actor.movable_mask, raw.clamp(-1, 1), actor.fixed_action)
        q = critic(action).mean()
    finally:
        for p, flag in zip(critic.parameters(), flags):
            p.requires_grad_(flag)
    return alpha * distill - q_weight * q, distill, q
