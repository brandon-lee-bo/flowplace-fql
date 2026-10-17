"""Graph FQL for a fixed-netlist terminal placement environment.

Action = bounded raw macro centers. Environment = released deterministic
legalizer followed by the physical flow. Therefore Q(a) models R(L(a)); no
straight-through derivative of L is asserted. The critic derivative is learned.
The two graph policies retain the released GATv2/global-attention backbone.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from baseline.api import load_model
from .physical_ridge import PhysicalRidgeCritic


class GraphPolicy(nn.Module):
    def __init__(self, problem, device="cuda"):
        super().__init__()
        self.model, _ = load_model(problem, device, hard_constraint=False)
        self.cond = problem.cond
        self.register_buffer("movable", ~problem.cond.is_ports.bool())
        self.register_buffer("fixed", problem.x.detach().clone())

    def restore(self, x, clip=False):
        if clip:
            x = x.clamp(-1, 1)
        return torch.where(self.movable[None, :, None], x, self.fixed[None])

    def velocity(self, x, t):
        result = self.model(x, self.cond, t).float()
        return result * self.movable[None, :, None]

    def forward(self, noise, clip=True):
        # Independent one-step graph policy; weights do not share with BC flow.
        x = self.restore(noise)
        result = self.model(x, self.cond, x.new_zeros(len(x))).float()
        return self.restore(result, clip)

    def flow_loss(self, action, noise, t):
        x0 = self.restore(noise)
        target = self.restore(action)
        time = t.reshape(-1, 1, 1)
        xt = (1 - time) * x0 + time * target
        return (self.velocity(xt, t)[:, self.movable] -
                (target - x0)[:, self.movable]).square().mean()

    @torch.no_grad()
    def sample_flow(self, noise, steps=10):
        if steps < 1:
            raise ValueError("Flow integration needs a positive step count")
        x = self.restore(noise)
        for i in range(steps):
            t = x.new_full((len(x),), i / steps)
            x = self.restore(x + self.velocity(x, t) / steps)
        return self.restore(x, clip=True)


class FeatureCritics(nn.Module):
    """FQL twin critics with train-standardized coordinates and physical priors.

Only offline TRAIN features set the normalization. Online samples update the
two independent neural regressors, never relabel a prediction as a measurement.
"""
    def __init__(self, cache, train_actions, train_reward, movable, width=128):
        super().__init__()
        zero = dict(mean=np.zeros(17), std=np.ones(17), coeff=np.zeros(17), intercept=0.)
        self.extractor = PhysicalRidgeCritic(cache, zero).to(train_actions.device)
        self.register_buffer("movable", movable.detach().clone())
        with torch.no_grad():
            f = self.features(train_actions)
        self.register_buffer("mean", f.mean(0))
        std = f.std(0, unbiased=False)
        self.register_buffer("std", torch.where(std > 1e-5, std, torch.ones_like(std)))
        self.register_buffer("reward_mean", train_reward.mean())
        self.register_buffer("reward_std", train_reward.std(unbiased=False).clamp_min(.03))
        self.members = nn.ModuleList([
            nn.Sequential(nn.Linear(f.shape[1], width), nn.LayerNorm(width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
            for _ in range(2)])
        for net in self.members:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def features(self, action):
        flat = action.reshape(len(action), -1)
        geometry = self.extractor.features(flat).float()
        return torch.cat([geometry, action[:, self.movable].flatten(1)], -1)

    def forward(self, action):
        f = (self.features(action) - self.mean) / self.std
        return self.reward_mean + self.reward_std * torch.cat([net(f) for net in self.members], -1)


def actor_objective(actor, flow, critic, noise, alpha=10., q_weight=1., flow_steps=10):
    with torch.no_grad():
        target = flow.sample_flow(noise, flow_steps)
    prediction = actor(noise, clip=False)
    distill = (prediction[:, actor.movable] - target[:, actor.movable]).square().mean()
    flags = [p.requires_grad for p in critic.parameters()]
    for p in critic.parameters():
        p.requires_grad_(False)
    try:
        q = critic(actor.restore(prediction, clip=True)).mean()
    finally:
        for p, flag in zip(critic.parameters(), flags):
            p.requires_grad_(flag)
    return alpha * distill - q_weight * q, distill, q
