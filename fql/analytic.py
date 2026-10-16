"""Differentiable exact macro-only HPWL in the same cache convention.

This provides an analytic control, NOT a differentiable legalization operation.
Gradients are subgradients of pre-legalization HPWL. The final proposal must use
the same external legalizer and evaluator as the learned policy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DifferentiableMacroHPWL(nn.Module):
    def __init__(self, cache, grid=224):
        super().__init__()
        self.names = list(cache["node_id_to_name"])
        ids = {name: i for i, name in enumerate(self.names)}
        canvas = np.asarray([cache["ratio_x"] * grid, cache["ratio_y"] * grid])
        self.register_buffer("canvas", torch.as_tensor(canvas, dtype=torch.float64))
        size = np.asarray([[cache["node_info"][n]["x"], cache["node_info"][n]["y"]] for n in self.names])
        self.register_buffer("sizes", torch.as_tensor(size * 2 / canvas, dtype=torch.float64))
        pin_nodes, pin_offsets, pin_nets = [], [], []
        self.num_nets = 0
        self.ignored_nets = 0
        for net in cache["net_info"].values():
            pins = [(ids[name], [pin.get("x_offset", 0), pin.get("y_offset", 0)])
                    for name, pin in net["nodes"].items() if name in ids]
            if len(pins) < 2:
                self.ignored_nets += 1
                continue
            for node, offset in pins:
                pin_nodes.append(node)
                pin_offsets.append(offset)
                pin_nets.append(self.num_nets)
            self.num_nets += 1
        self.register_buffer("pin_nodes", torch.tensor(pin_nodes, dtype=torch.long))
        self.register_buffer("pin_nets", torch.tensor(pin_nets, dtype=torch.long))
        self.register_buffer("pin_offsets", torch.as_tensor(np.asarray(pin_offsets).reshape(-1, 2), dtype=torch.float64))

    def forward(self, coordinates):
        # Accept flattened [B,2V] or explicitly grouped [B,V,2].
        if coordinates.ndim == 2:
            coordinates = coordinates.reshape(len(coordinates), len(self.names), 2)
        if coordinates.ndim != 3 or coordinates.shape[1:] != (len(self.names), 2):
            raise ValueError("coordinates must be [B,V,2] or [B,2V]")
        if self.num_nets == 0:
            return coordinates.sum(dim=(1, 2)) * 0
        physical = (coordinates + 1) * self.canvas.to(coordinates.dtype) / 2
        pins = physical[:, self.pin_nodes, :] + self.pin_offsets.to(coordinates.dtype)
        index = self.pin_nets.view(1, -1, 1).expand(len(coordinates), -1, 2)
        shape = (len(coordinates), self.num_nets, 2)
        minima = coordinates.new_full(shape, float("inf")).scatter_reduce(1, index, pins, reduce="amin", include_self=True)
        maxima = coordinates.new_full(shape, -float("inf")).scatter_reduce(1, index, pins, reduce="amax", include_self=True)
        return (maxima - minima).sum(dim=(1, 2))

    def metadata(self):
        return {"metric": "macro_only_hypernet_hpwl_cache_units", "num_macros": len(self.names),
                "included_nets": self.num_nets, "ignored_nets": self.ignored_nets,
                "canvas": self.canvas.detach().cpu().tolist(),
                "pin_offsets": "center-relative, as official graph conversion",
                "gradient": "exact pre-legalization HPWL subgradient, undefined choice at tied extrema handled by PyTorch",
                "not_included": ["standard cells", "explicit fixed I/O terminals", "legalizer Jacobian", "routed PPA"]}


def load_hpwl(path, device):
    path = Path(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    model = DifferentiableMacroHPWL(cache).to(device)
    meta = model.metadata()
    meta.update(cache=str(path.resolve()), cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return model, meta


def optimize_coordinates(teacher, hpwl, reward_scale, movable_mask, fixed_action,
                         steps=100, lr=0.01, anchor_weight=1.0):
    """Matched-initial-layout Adam control. No teacher calls or Q gradients.

    Optimize each layout independently. Summing per-layout losses keeps the
    gradient magnitude independent of how many layouts are batched together.
    Only coordinate-mean anchoring uses the dimension-normalized convention.
    """
    if steps < 0 or lr <= 0 or anchor_weight < 0 or reward_scale <= 0:
        raise ValueError("Invalid optimization parameters")
    proposal = nn.Parameter(teacher.detach().clone())
    optimizer = torch.optim.Adam([proposal], lr=lr)
    movable_mask = torch.as_tensor(movable_mask, device=teacher.device, dtype=torch.bool)
    fixed_action = torch.as_tensor(fixed_action, device=teacher.device, dtype=teacher.dtype)
    trace = []
    for step in range(steps):
        current = torch.where(movable_mask, proposal, fixed_action)
        values = hpwl(current) / reward_scale
        anchor = (current[:, movable_mask] - teacher[:, movable_mask]).square().mean(dim=-1)
        per_layout = values + anchor_weight * anchor
        loss = per_layout.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite direct HPWL objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            proposal.copy_(torch.where(movable_mask, proposal.clamp(-1, 1), fixed_action))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            trace.append({"step": step + 1, "mean_pre_update_hpwl_normalized": float(values.detach().mean()),
                          "mean_pre_update_anchor_mse": float(anchor.detach().mean())})
    return proposal.detach(), trace
