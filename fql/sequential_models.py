"""V2 partial-layout actor and twin graph critics.

Each actor call returns one two-dimensional action.  Unknown macro coordinates
are discarded before encoding; graph topology and sizes are the only static
netlist features.  Every encoder starts from the released FlowPlace GAT and
attention weights, with a new placed/current/progress embedding.  Actor and
both critics have independent parameters.  Q(s, raw_action) models the reward
after the environment's deterministic single-macro projection.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn


class _Condition:
    def __init__(self, **values):
        self.__dict__.update(values)

    def __getitem__(self, key):
        return getattr(self, key)


def _load_model(problem, device, **kwargs):
    # Upstream prepends its analysis.py; bind our package first.
    import analysis  # noqa: F401
    from baseline.api import load_model
    return load_model(problem, device=device, **kwargs)


class PartialGraphEncoder(nn.Module):
    """Released encoder with explicit dynamic features, excluding its decoder."""
    def __init__(self, problem, device="cuda", model_loader=None):
        super().__init__()
        model, _ = (model_loader or _load_model)(problem, device=device,
                                               hard_constraint=False)
        backbone = model._reverse_model
        blocks = list(backbone._gnn_blocks)
        if len(blocks) < 2 or not hasattr(blocks[0], "_layer") or not hasattr(blocks[-1], "_layer"):
            raise ValueError("Expected released linear encoder/decoder around GAT blocks")
        self.hidden_size = blocks[0]._layer.out_features
        decoder = blocks[-1]._layer
        if decoder.in_features != self.hidden_size or decoder.out_features != 2:
            raise ValueError("Released decoder does not match coordinate output")
        self.decoder_initial_state = {k: v.detach().cpu().clone()
                                      for k, v in decoder.state_dict().items()}
        self.blocks = nn.ModuleList(blocks[:-1])
        self.t_encoder = model.t_encoder
        self.dir_att_input = backbone.dir_att_input
        self.status_embedding = nn.Linear(3, self.hidden_size, bias=False,
                                          device=device)
        nn.init.zeros_(self.status_embedding.weight)
        sizes = problem.cond.x.detach().to(device).float()
        fixed = problem.cond.is_ports.flatten().detach().to(device).bool()
        positions = problem.x.detach().to(device).float()
        if sizes.ndim != 2 or sizes.shape[-1] != 2 or positions.shape != sizes.shape:
            raise ValueError("Expected [N,2] graph sizes and positions")
        self.register_buffer("sizes", sizes.clone())
        self.register_buffer("base_fixed", fixed.clone())
        self.register_buffer("fixed_positions", torch.where(fixed[:, None], positions,
                                                            torch.zeros_like(positions)))
        self.register_buffer("edge_index", problem.cond.edge_index.detach().to(device).long().clone())
        self.register_buffer("edge_attr", problem.cond.edge_attr.detach().to(device).float().clone())

    @property
    def node_count(self):
        return len(self.sizes)

    def spatial_input(self, partial_positions, placed, current_id, *, noise=None, action=None):
        shape = (len(partial_positions), self.node_count, 2)
        if partial_positions.shape != shape or not shape[0]:
            raise ValueError("partial_positions must be nonempty [B,N,2]")
        if placed.shape != shape[:2] or placed.dtype != torch.bool:
            raise ValueError("placed must be bool [B,N]")
        if current_id.shape != (shape[0],) or current_id.dtype != torch.long:
            raise ValueError("current_id must be long [B]")
        if any(t.device != self.sizes.device for t in (partial_positions, placed, current_id)):
            raise ValueError("State and model devices differ")
        if bool(((current_id < 0) | (current_id >= self.node_count)).any()):
            raise ValueError("Terminal states must not be passed to actor/critic")
        rows = torch.arange(shape[0], device=current_id.device)
        if bool(placed[rows, current_id].any()) or not bool(placed[:, self.base_fixed].all()):
            raise ValueError("Current macro must be unplaced; original fixed nodes must be placed")
        known = torch.where(placed[..., None], partial_positions,
                            torch.zeros_like(partial_positions))
        if not bool(torch.isfinite(known).all()):
            raise ValueError("Nonfinite known coordinates")
        if not torch.equal(known[:, self.base_fixed],
                           self.fixed_positions[None, self.base_fixed].expand(shape[0], -1, -1)):
            raise ValueError("Original fixed coordinates changed")
        if (noise is None) == (action is None):
            raise ValueError("Supply exactly one of actor noise or critic action")
        if noise is not None:
            if noise.shape != shape or noise.device != known.device or not bool(torch.isfinite(noise).all()):
                raise ValueError("Actor noise must be finite [B,N,2] on the model device")
            spatial = torch.where(placed[..., None], known, noise)
        else:
            if action.shape != (shape[0], 2) or action.device != known.device or not bool(torch.isfinite(action).all()):
                raise ValueError("Critic action must be finite [B,2] on the model device")
            current_mask = torch.nn.functional.one_hot(current_id, self.node_count).bool()
            spatial = torch.where(current_mask[..., None], action[:, None], known)
        current_flag = torch.nn.functional.one_hot(current_id, self.node_count).to(known.dtype)
        movable = ~self.base_fixed
        progress = placed[:, movable].to(known.dtype).mean(-1, keepdim=True)
        status = torch.stack((placed.to(known.dtype), current_flag,
                              progress.expand(-1, self.node_count)), -1)
        return spatial, status

    def forward(self, partial_positions, placed, current_id, *, noise=None, action=None):
        spatial, status = self.spatial_input(partial_positions, placed, current_id,
                                             noise=noise, action=action)
        cond = _Condition(x=self.sizes, edge_index=self.edge_index,
                          edge_attr=self.edge_attr, is_ports=self.base_fixed)
        t = spatial.new_zeros(len(spatial))
        time_embedding = self.t_encoder(t)
        # This follows released AttGNN.forward, stopping before its decoder.
        with torch.autocast(device_type=spatial.device.type,
                            enabled=spatial.device.type == "cuda"):
            hidden = self.blocks[0](spatial, cond, time_embedding)
            hidden = hidden + self.status_embedding(status)
            for block in self.blocks[1:]:
                name = block.__class__.__name__
                if name == "AttGNNBlock":
                    extra = spatial if self.dir_att_input else hidden
                    hidden = block(hidden, cond, time_embedding, att_extra_input=extra)
                elif name == "MLP":
                    hidden = block(hidden)
                else:
                    hidden = block(hidden, cond, time_embedding)
        rows = torch.arange(len(spatial), device=spatial.device)
        return hidden.float(), spatial[rows, current_id].float()


class SequentialActor(nn.Module):
    def __init__(self, problem, device="cuda", model_loader=None):
        super().__init__()
        self.encoder = PartialGraphEncoder(problem, device, model_loader)
        self.head = nn.Linear(self.encoder.hidden_size, 2, device=device)
        self.head.load_state_dict(self.encoder.decoder_initial_state)

    def forward(self, partial_positions, placed, current_id, noise, *, clip=True):
        hidden, skip = self.encoder(partial_positions, placed, current_id, noise=noise)
        rows = torch.arange(len(hidden), device=hidden.device)
        action = self.head(hidden[rows, current_id]) + skip
        return action.clamp(-1, 1) if clip else action


class _GraphQ(nn.Module):
    def __init__(self, problem, device, width, model_loader):
        super().__init__()
        self.encoder = PartialGraphEncoder(problem, device, model_loader)
        dim = self.encoder.hidden_size * 2 + 2
        self.head = nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)).to(device)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, partial_positions, placed, current_id, action):
        hidden, _ = self.encoder(partial_positions, placed, current_id, action=action)
        rows = torch.arange(len(hidden), device=hidden.device)
        features = torch.cat((hidden[rows, current_id], hidden.mean(1), action), -1)
        return self.head(features)


class SequentialCritics(nn.Module):
    """Independent graph critics; affine reward scale is fixed from TRAIN."""
    def __init__(self, problem, train_returns, device="cuda", width=128, model_loader=None):
        super().__init__()
        returns = torch.as_tensor(train_returns, device=device, dtype=torch.float32)
        if returns.ndim != 1 or not len(returns) or not bool(torch.isfinite(returns).all()):
            raise ValueError("Finite TRAIN trajectory returns are required")
        self.register_buffer("reward_mean", returns.mean())
        self.register_buffer("reward_scale", returns.std(unbiased=False).clamp_min(.03))
        self.members = nn.ModuleList([_GraphQ(problem, device, width, model_loader) for _ in range(2)])

    def forward(self, partial_positions, placed, current_id, action):
        values = torch.cat([net(partial_positions, placed, current_id, action)
                            for net in self.members], -1)
        return self.reward_mean + self.reward_scale * values


def reduced_q(values, reduction="mean"):
    if values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError("Expected two Q predictions per action")
    if reduction == "mean":
        return values.mean(-1)
    if reduction == "min":
        return values.min(-1).values
    raise ValueError("Q reduction must be mean or min")


def frozen_target(critic):
    target = copy.deepcopy(critic)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def polyak_update(critic, target, tau):
    if not 0 < tau <= 1:
        raise ValueError("Target update tau must be in (0,1]")
    parameters = dict(critic.named_parameters())
    for name, value in target.named_parameters():
        value.lerp_(parameters[name], tau)
    # Geometry and TRAIN normalization are immutable, not running statistics.
    buffers = dict(critic.named_buffers())
    for name, value in target.named_buffers():
        value.copy_(buffers[name])


@torch.no_grad()
def bellman_targets(actor, target_critic, batch, next_noise, gamma=1., reduction="mean"):
    if not 0 <= gamma <= 1:
        raise ValueError("Invalid discount")
    reward, done = batch["reward"], batch["done"]
    if reward.ndim != 1 or done.shape != reward.shape or done.dtype != torch.bool:
        raise ValueError("reward and bool done must be [B]")
    target = reward.clone()
    active = ~done
    # Terminal IDs are -1.  Do not evaluate a network and then multiply NaN by 0.
    if bool(active.any()):
        positions, placed, ids = (batch[key][active] for key in
                                  ("next_positions", "next_placed", "next_id"))
        action = actor(positions, placed, ids, next_noise[active])
        value = reduced_q(target_critic(positions, placed, ids, action), reduction)
        target[active] += gamma * value
    return target


@contextmanager
def frozen_parameters(module):
    flags = [parameter.requires_grad for parameter in module.parameters()]
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


def actor_objective(actor, teacher, critic, batch, noise, alpha=10., q_weight=1.,
                    q_reduction="mean", *, return_info=False):
    if alpha < 0 or q_weight < 0:
        raise ValueError("Actor coefficients must be nonnegative")
    positions, placed, ids = (batch[key] for key in ("positions", "placed", "current_id"))
    with torch.no_grad():
        complete = teacher.sample_flow(positions, placed, ids, noise)
        target = complete[torch.arange(len(ids), device=ids.device), ids]
    # Match FQL's order: distill the unbounded network output, then bound the
    # action queried by Q.  Clamping before MSE would give an overshooting
    # coordinate zero corrective BC gradient even when the teacher is inside.
    raw_prediction = actor(positions, placed, ids, noise, clip=False)
    distill = (raw_prediction-target).square().mean()  # mean per valid coordinate
    prediction = raw_prediction.clamp(-1, 1)
    with frozen_parameters(critic):
        q = reduced_q(critic(positions, placed, ids, prediction), q_reduction).mean()
    result = (alpha * distill - q_weight * q, distill, q)
    if return_info:
        return (*result, dict(raw_saturation_fraction=float(
            (raw_prediction.detach().abs() > 1).float().mean())))
    return result
