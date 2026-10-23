"""Deterministic single-macro environment for the V2 placement experiment.

Unplaced coordinates are always zero and distinguished by ``placed``.  No
reference-layout coordinates are exposed for movable nodes.  Rewards from the
physical backend must be attached by the rollout collector; this environment
returns zero and does not invent terminal PPA measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def deterministic_macro_order(sizes, base_fixed):
    """Descending macro area, with graph-node ID as a stable tie breaker."""
    areas = (sizes[:, 0] * sizes[:, 1]).detach().cpu().tolist()
    fixed = base_fixed.detach().cpu().bool().tolist()
    return tuple(sorted((i for i in range(len(areas)) if not fixed[i]),
                        key=lambda i: (-areas[i], i)))


def geometry_audit(positions, sizes, active=None, tolerance=1e-6):
    """Check only known positive-area rectangles, allowing touching edges."""
    if positions.ndim != 2 or positions.shape != sizes.shape or positions.shape[-1] != 2:
        raise ValueError("positions and sizes must be matching [N,2] tensors")
    mask = torch.ones(len(sizes), dtype=torch.bool, device=sizes.device) if active is None else active.bool()
    if mask.shape != (len(sizes),):
        raise ValueError("active must have shape [N]")
    if not bool(torch.isfinite(positions[mask]).all() and torch.isfinite(sizes[mask]).all()):
        return dict(is_legal=False, overlap_pairs=None, out_of_bounds_macros=None,
                    failure_reason="nonfinite_geometry")
    if bool((sizes[mask] < 0).any()):
        return dict(is_legal=False, overlap_pairs=None, out_of_bounds_macros=None,
                    failure_reason="negative_size")
    # A zero-area fixed I/O is a condition, not an obstructing macro rectangle.
    mask = mask & (sizes > 0).all(-1)
    p, s = positions[mask], sizes[mask]
    if not bool(torch.isfinite(p).all() and torch.isfinite(s).all()):
        return dict(is_legal=False, overlap_pairs=None, out_of_bounds_macros=None,
                    failure_reason="nonfinite_geometry")
    if not len(p):
        return dict(is_legal=True, overlap_pairs=0, out_of_bounds_macros=0,
                    max_boundary_excess=0.0, failure_reason=None)
    lo, hi = p - s / 2, p + s / 2
    intersection = torch.minimum(hi[:, None], hi[None]) - torch.maximum(lo[:, None], lo[None])
    overlap = (intersection > tolerance).all(-1)
    overlap = torch.triu(overlap, diagonal=1)
    outside = ((lo < -1 - tolerance) | (hi > 1 + tolerance)).any(-1)
    count, boundary = int(overlap.sum()), int(outside.sum())
    return dict(is_legal=not (count or boundary), overlap_pairs=count,
                out_of_bounds_macros=boundary,
                max_boundary_excess=float(torch.maximum((-1-lo).clamp_min(0), (hi-1).clamp_min(0)).max()),
                failure_reason="overlap" if count else "out_of_bounds" if boundary else None)


@dataclass(frozen=True)
class ProjectionResult:
    executed: torch.Tensor | None
    distance: float | None
    failure_reason: str | None


@torch.no_grad()
def project_macro_action(raw_action, macro_id, positions, placed, sizes,
                         padding=0.0, tolerance=1e-7, chunk_size=2048):
    """Project one center into the remaining continuous rectangular free space.

    A valid input is an identity operation.  Otherwise the nearest point is
    found among the clamped input, board boundaries and obstacle boundaries.
    The Cartesian arrangement includes a nearest point in every free cell;
    unlike a raster search it does not invent dead ends from grid resolution.
    Ties follow sorted x then y.  Already placed rectangles never move.
    """
    if padding < 0 or tolerance < 0 or chunk_size < 1:
        raise ValueError("Invalid projection settings")
    if positions.ndim != 2 or positions.shape != sizes.shape or positions.shape[-1] != 2:
        raise ValueError("positions and sizes must have shape [N,2]")
    if placed.dtype != torch.bool or placed.shape != (len(sizes),):
        raise ValueError("placed must be bool [N]")
    if not 0 <= int(macro_id) < len(sizes) or bool(placed[int(macro_id)]):
        raise ValueError("Action must target an unplaced macro")
    raw = torch.as_tensor(raw_action, device=positions.device, dtype=positions.dtype)
    if raw.shape == (1, 2):
        raw = raw[0]
    if raw.shape != (2,):
        raise ValueError("A macro action must have shape [2] or [1,2]")
    if not bool(torch.isfinite(raw).all()):
        return ProjectionResult(None, None, "nonfinite_action")
    size = sizes[int(macro_id)]
    if not bool(torch.isfinite(size).all()) or not bool((size > 0).all()):
        raise ValueError("Movable macro sizes must be finite and positive")
    lower, upper = -1 + size / 2 + padding, 1 - size / 2 - padding
    if bool((lower > upper).any()):
        return ProjectionResult(None, None, "macro_larger_than_canvas")
    obstacle_mask = placed & (sizes > 0).all(-1)
    obstacle_pos, obstacle_sizes = positions[obstacle_mask], sizes[obstacle_mask]
    if not bool(torch.isfinite(obstacle_pos).all()):
        raise ValueError("Nonfinite placed coordinates")
    clearance = (size[None] + obstacle_sizes) / 2 + 2 * padding

    def feasible(candidates):
        in_bounds = ((candidates >= lower-tolerance) & (candidates <= upper+tolerance)).all(-1)
        if not len(obstacle_pos):
            return in_bounds
        blocked = ((candidates[:, None] - obstacle_pos[None]).abs() < clearance[None] - tolerance).all(-1).any(-1)
        return in_bounds & ~blocked

    clipped = torch.minimum(torch.maximum(raw, lower), upper)
    if bool(feasible(clipped[None])[0]):
        return ProjectionResult(clipped.clone(), float((clipped-raw).norm()), None)
    axes = []
    for dim in range(2):
        values = torch.cat((clipped[dim:dim+1], lower[dim:dim+1], upper[dim:dim+1],
                            obstacle_pos[:, dim]-clearance[:, dim],
                            obstacle_pos[:, dim]+clearance[:, dim]))
        values = values[(values >= lower[dim]-tolerance) & (values <= upper[dim]+tolerance)]
        axes.append(torch.unique(values.clamp(lower[dim], upper[dim]), sorted=True))
    # Enumerate in chunks to avoid allocating [all candidates, obstacles, 2].
    total, y_count = len(axes[0]) * len(axes[1]), len(axes[1])
    best, best_distance = None, float("inf")
    for start in range(0, total, chunk_size):
        ids = torch.arange(start, min(total, start+chunk_size), device=positions.device)
        candidates = torch.stack((axes[0][ids // y_count], axes[1][ids % y_count]), -1)
        candidates = candidates[feasible(candidates)]
        if not len(candidates):
            continue
        distance = (candidates-raw).square().sum(-1)
        index = int(distance.argmin())
        if float(distance[index]) < best_distance:
            best, best_distance = candidates[index].clone(), float(distance[index])
    if best is None:
        return ProjectionResult(None, None, "no_legal_position")
    return ProjectionResult(best, best_distance ** .5, None)


class SequentialPlacementEnv:
    """One environment transition places exactly one macro, never an entire graph."""
    def __init__(self, problem, order: Iterable[int] | None = None, padding=0.0):
        self.sizes = problem.cond.x.detach().clone()
        self.base_fixed = problem.cond.is_ports.flatten().bool().detach().clone()
        if self.sizes.ndim != 2 or self.sizes.shape != problem.x.shape or self.sizes.shape[-1] != 2 or not len(self.sizes):
            raise ValueError("Expected graph sizes and reference positions [N,2]")
        if self.base_fixed.shape != (len(self.sizes),):
            raise ValueError("Fixed mask must have one entry per graph node")
        if not self.sizes.is_floating_point() or not problem.x.is_floating_point():
            raise ValueError("Coordinates and sizes must be floating point")
        if self.sizes.device != problem.x.device or self.sizes.dtype != problem.x.dtype:
            raise ValueError("Coordinates and sizes must share device and dtype")
        if not bool(torch.isfinite(self.sizes).all()) or bool((self.sizes < 0).any()):
            raise ValueError("Sizes must be finite and nonnegative")
        if not bool((self.sizes[~self.base_fixed] > 0).all()):
            raise ValueError("Movable macro sizes must be positive")
        self.fixed_positions = torch.where(self.base_fixed[:, None], problem.x.detach(),
                                           torch.zeros_like(problem.x))
        self.order = deterministic_macro_order(self.sizes, self.base_fixed) if order is None else tuple(int(i) for i in order)
        expected = set((~self.base_fixed).nonzero(as_tuple=True)[0].tolist())
        if len(self.order) != len(expected) or set(self.order) != expected:
            raise ValueError("Order must contain every movable macro exactly once")
        if padding < 0:
            raise ValueError("padding must be nonnegative")
        self.padding = float(padding)
        self.reset()

    def reset(self):
        self.positions = self.fixed_positions.clone()
        self.placed = self.base_fixed.clone()
        if not geometry_audit(self.positions, self.sizes, self.placed)["is_legal"]:
            raise ValueError("Initial fixed macros are not geometrically legal")
        self.cursor = 0
        self.done = not self.order
        self.failure_reason = None
        self.history = []
        return self.observe()

    def observe(self):
        current = -1 if self.done else self.order[self.cursor]
        return dict(partial_positions=self.positions.clone().unsqueeze(0),
                    placed=self.placed.clone().unsqueeze(0),
                    current_id=torch.tensor([current], device=self.positions.device, dtype=torch.long))

    @torch.no_grad()
    def step(self, raw_action):
        if self.done:
            raise RuntimeError("Cannot step a terminal placement episode")
        current = self.order[self.cursor]
        before = self.observe()
        raw = torch.as_tensor(raw_action, device=self.positions.device, dtype=self.positions.dtype).detach().clone()
        if raw.shape == (1, 2):
            raw = raw[0]
        result = project_macro_action(raw, current, self.positions, self.placed,
                                      self.sizes, padding=self.padding)
        if result.executed is None:
            self.done, self.failure_reason = True, result.failure_reason
        else:
            self.positions[current] = result.executed
            self.placed[current] = True
            self.cursor += 1
            self.done = self.cursor == len(self.order)
            if not geometry_audit(self.positions, self.sizes, self.placed)["is_legal"]:
                raise RuntimeError("Geometry projection violated the environment contract")
        after = self.observe()
        info = dict(current_id=current, raw_action=raw,
                    executed_action=None if result.executed is None else result.executed.clone(),
                    projection_distance=result.distance, failure_reason=result.failure_reason,
                    completed=self.done and self.failure_reason is None,
                    terminal_reward_pending=self.done and self.failure_reason is None)
        self.history.append(dict(state=before, action=raw, next_state=after,
                                 reward=0.0, done=self.done, info=info))
        return after, 0.0, self.done, info
