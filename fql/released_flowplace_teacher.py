"""Frozen public FlowPlace teacher, including its original constrained sampler.

The historical Euler teacher remains available only for old, untagged configs.
New configs explicitly select ``released_flowplace`` and never load a warmed-up
behavior flow into the public teacher.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

# Resolve the local package before upstream adds its own analysis.py to sys.path.
import analysis
from baseline.api import CHECKPOINT, generate, load_model
from .flow_policy_config import PUBLIC_SHA, STEPS, uses_released_teacher, validate_teacher_config


def checkpoint_sha():
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleasedFlowPlaceTeacher(nn.Module):
    """Run the unmodified upstream sampler once per batch member.

    Upstream hard-constraint sampling supports B=1 only. No Euler replacement,
    output clamp, extra projection, or changed noise pairing is introduced here.
    External student legalization remains the environment's responsibility.
    """
    def __init__(self, problem, device="cuda"):
        super().__init__()
        if checkpoint_sha() != PUBLIC_SHA:
            raise ValueError("Public FlowPlace checkpoint differs from the pinned release")
        self.model, _ = load_model(problem, device=device, steps=STEPS,
                                   hard_constraint=True, w_reg=.1, w_dist=1.)
        self.problem = problem
        self.register_buffer("movable", ~problem.cond.is_ports.flatten().bool())
        self.register_buffer("fixed", problem.x.detach().clone())
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # An enclosing trainer's train() call must not enable teacher training.
        return super().train(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = self.state_dict()
        if set(state_dict) != set(expected) or any(
                value.shape != state_dict[key].shape
                or value.dtype != state_dict[key].dtype
                or not torch.equal(value, state_dict[key].to(value.device))
                for key, value in expected.items()):
            raise ValueError("Cannot replace the frozen public FlowPlace teacher "
                             "with a legacy or modified flow checkpoint")
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    @torch.no_grad()
    def sample_flow(self, noise, steps=STEPS):
        if steps != STEPS:
            raise ValueError("The released FlowPlace teacher uses exactly 20 steps")
        if noise.ndim != 3 or tuple(noise.shape[1:]) != tuple(self.fixed.shape) or not len(noise):
            raise ValueError("Expected a nonempty [batch, macros, 2] noise tensor")
        if not bool(torch.isfinite(noise).all()):
            raise ValueError("Nonfinite FlowPlace teacher noise")
        outputs = []
        for row in noise:
            result, used, _ = generate(self.model, self.problem, noise=row)
            if not torch.equal(used, row.to(used.device)):
                raise ValueError("FlowPlace sampler changed the supplied distillation noise")
            if not bool(torch.isfinite(result).all()):
                raise ValueError("Nonfinite FlowPlace teacher output")
            if not torch.equal(result[~self.movable], self.fixed[~self.movable]):
                raise ValueError("FlowPlace sampler moved fixed graph nodes")
            outputs.append(result)
        return torch.stack(outputs)

    def forward(self, noise):
        return self.sample_flow(noise)

    def flow_loss(self, *args, **kwargs):
        raise RuntimeError("The released FlowPlace teacher is frozen; train actor and Q only")


def make_behavior_flow(problem, cfg, device="cuda"):
    validate_teacher_config(cfg)
    if uses_released_teacher(cfg):
        return ReleasedFlowPlaceTeacher(problem, device)
    from .graph_o2o_models import GraphPolicy
    return GraphPolicy(problem, device)
