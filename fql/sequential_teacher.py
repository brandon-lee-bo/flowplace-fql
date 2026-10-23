"""V2 fixed-prefix FlowPlace completion, separate from the exact V1.1 sampler.

This frozen adapter changes conditioning, not the released checkpoint.  Its
conditional completion quality is an experimental question.  It synchronizes
the graph mask, sampler restoration, and legalizer mask, and restores fixed
coordinates *before* legalizing each predicted endpoint.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

# Resolve this repository's package before upstream inserts analysis.py.
import analysis
from baseline.api import Legalizer, load_model
from .flow_policy_config import PUBLIC_SHA, STEPS
from .released_flowplace_teacher import checkpoint_sha
from .sequential_env import geometry_audit


class ConditionalCompletionError(RuntimeError):
    def __init__(self, reports):
        self.reports = reports
        failures = [str(i) for i, r in enumerate(reports) if not r["is_legal"]]
        super().__init__("Conditional teacher produced illegal completion in rows " + ",".join(failures))


class SequentialFlowPlaceTeacher(nn.Module):
    """Frozen public weights with explicit partial-layout conditioning.

    ``sample_flow(partial_positions, placed, current_id, noise)`` accepts
    [B,N,2], bool [B,N], long [B], [B,N,2] and returns full layouts [B,N,2].
    The student must see this same complete noise tensor, then learn only the
    position selected by ``current_id``.  No noise is sampled internally.
    """
    def __init__(self, problem, device="cuda"):
        super().__init__()
        if checkpoint_sha() != PUBLIC_SHA:
            raise ValueError("Public FlowPlace checkpoint differs from the pinned release")
        self.model, _ = load_model(problem, device=device, steps=STEPS,
                                   hard_constraint=True, w_reg=.1, w_dist=1.)
        if self.model.mask_key != "is_ports":
            raise ValueError("V2 adapter requires the released is_ports masking interface")
        self.cond = problem.cond.clone() if hasattr(problem.cond, "clone") else copy.deepcopy(problem.cond)
        if hasattr(self.cond, "to"):
            self.cond = self.cond.to(device)
        self.register_buffer("base_fixed", problem.cond.is_ports.flatten().bool().detach().clone().to(device))
        self.register_buffer("fixed_positions", torch.where(self.base_fixed[:, None],
                             problem.x.detach().to(device), torch.zeros_like(problem.x, device=device)))
        self.register_buffer("sizes", problem.cond.x.detach().clone().to(device))
        if (self.sizes.ndim != 2 or self.sizes.shape[-1] != 2
                or self.fixed_positions.shape != self.sizes.shape
                or self.base_fixed.shape != (len(self.sizes),)):
            raise ValueError("Invalid graph geometry for conditional teacher")
        if not bool(torch.isfinite(self.sizes).all()) or bool((self.sizes < 0).any()):
            raise ValueError("Teacher sizes must be finite and nonnegative")
        if not bool((self.sizes[~self.base_fixed] > 0).all()):
            raise ValueError("Movable macro sizes must be positive")
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = self.state_dict()
        if set(state_dict) != set(expected) or any(
                value.shape != state_dict[key].shape or value.dtype != state_dict[key].dtype
                or not torch.equal(value, state_dict[key].to(value.device))
                for key, value in expected.items()):
            raise ValueError("Cannot replace frozen V2 public teacher weights or geometry")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _validate(self, partial_positions, placed, current_id, noise):
        if partial_positions.ndim != 3:
            raise ValueError("partial_positions must be nonempty [B,N,2]")
        shape = (len(partial_positions), len(self.sizes), 2)
        if tuple(partial_positions.shape) != shape or not shape[0]:
            raise ValueError("partial_positions must be nonempty [B,N,2]")
        if tuple(noise.shape) != shape or placed.shape != shape[:2] or placed.dtype != torch.bool:
            raise ValueError("noise must be [B,N,2] and placed must be bool [B,N]")
        if current_id.shape != (shape[0],) or current_id.dtype != torch.long:
            raise ValueError("current_id must be long [B]")
        if any(x.device != self.sizes.device for x in (partial_positions, placed, current_id, noise)):
            raise ValueError("Teacher tensors must share the teacher device")
        if partial_positions.dtype != self.sizes.dtype or noise.dtype != self.sizes.dtype:
            raise ValueError("Teacher positions and noise must share the model coordinate dtype")
        if not bool(torch.isfinite(partial_positions).all() and torch.isfinite(noise).all()):
            raise ValueError("Nonfinite teacher state or noise")
        fixed = placed | self.base_fixed[None]
        for row in range(shape[0]):
            current = int(current_id[row])
            if current == -1:
                if not bool(fixed[row].all()):
                    raise ValueError("Terminal current_id requires all nodes fixed")
            elif not 0 <= current < shape[1] or bool(fixed[row, current]):
                raise ValueError("current_id must identify an unplaced macro")
            if not torch.equal(partial_positions[row, self.base_fixed], self.fixed_positions[self.base_fixed]):
                raise ValueError("State moved an original fixed node")
            if bool((partial_positions[row, ~fixed[row]] != 0).any()):
                raise ValueError("Unplaced coordinates must use zero plus the placed mask")
            if not geometry_audit(partial_positions[row], self.sizes, fixed[row])["is_legal"]:
                raise ValueError("Teacher received an illegal fixed prefix")
        return fixed

    @torch.no_grad()
    def sample_flow(self, partial_positions, placed, current_id, noise, *,
                    return_info=False, require_legal=True):
        fixed = self._validate(partial_positions, placed, current_id, noise)
        outputs, reports = [], []
        for row in range(len(noise)):
            cond = self.cond.clone() if hasattr(self.cond, "clone") else copy.deepcopy(self.cond)
            cond.is_ports = fixed[row].clone()
            mask = fixed[row][None, :, None]
            known = partial_positions[row:row+1]
            x = torch.where(mask, known, noise[row:row+1]).clone()
            calls = 0
            if not bool(fixed[row].all()):
                legalizer = Legalizer(cond=cond, grid_res=128,
                    scores=self.model.hard_constraint_scores, device=x.device,
                    dtype=x.dtype, padding=.001)
                self.model._scheduler.set_timesteps(STEPS)
                timesteps = self.model._scheduler.timesteps
                for t, t_next in zip(timesteps[:-1], timesteps[1:]):
                    t_vec = torch.as_tensor(t, device=x.device).expand(1)
                    next_vec = torch.as_tensor(t_next, device=x.device).expand(1)
                    velocity = self.model(x, cond, t_vec)
                    endpoint = x + (1-t_vec.view(1, 1, 1)) * velocity
                    # Critical: legalizer must see the actual fixed prefix,
                    # not velocities temporarily applied to fixed nodes.
                    endpoint = torch.where(mask, known, endpoint)
                    corrected = legalizer.run(endpoint)
                    if not torch.equal(corrected[mask.expand_as(corrected)], known[mask.expand_as(known)]):
                        raise RuntimeError("Conditional legalizer moved a fixed node")
                    x = (1-next_vec.view(1, 1, 1))*x + next_vec.view(1, 1, 1)*corrected
                    x = torch.where(mask, known, x.clamp(-2, 2))
                    calls += 1
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError("Conditional teacher generated nonfinite positions")
            if not torch.equal(x[mask.expand_as(x)], known[mask.expand_as(known)]):
                raise RuntimeError("Conditional teacher moved a fixed node")
            report = geometry_audit(x[0], self.sizes)
            report.update(fixed_nodes=int(fixed[row].sum()), macro_forward_calls=calls,
                          fixed_max_displacement=0.0,
                          teacher_variant="public20_conditional_prefix_v1")
            reports.append(report)
            outputs.append(x[0])
        if require_legal and any(not r["is_legal"] for r in reports):
            raise ConditionalCompletionError(reports)
        result = torch.stack(outputs)
        return (result, reports) if return_info else result

    def forward(self, partial_positions, placed, current_id, noise):
        return self.sample_flow(partial_positions, placed, current_id, noise)

    def flow_loss(self, *args, **kwargs):
        raise RuntimeError("V2 public teacher is frozen; conditional fine-tuning is a separate variant")
