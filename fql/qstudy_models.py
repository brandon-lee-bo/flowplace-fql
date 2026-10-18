"""Small-data, multi-metric critics for the fixed-circuit placement Q study.

Measured costs describe the *executed* layout after the external legalizer and
physical flow; inputs are raw proposals available before that flow. Nothing in
this module assumes that hard legalization is differentiable. Target and feature
statistics are fitted once, from explicitly supplied TRAIN rows. Forward calls
and online updates do not refit them.

These models are terminal cost regressors, not a Bellman/CQL implementation.
The optional ensemble uncertainty penalty is a heuristic lower-confidence
reward and carries no calibrated confidence or conservative-Q guarantee.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .physical_ridge import PhysicalRidgeCritic


PHYSICAL_FEATURE_NAMES = (
    "macro_hpwl_over_canvas_sum", "area_centroid_x", "area_centroid_y",
    "span_x", "span_y", "area_weighted_squared_radius", "boundary_gap_mean",
    "boundary_gap_min", "boundary_violation_sum", "pair_l1_mean",
    "pair_l1_std", "pair_l1_min", "overlap_area_sum", "density_lower_left",
    "density_upper_left", "density_lower_right", "density_upper_right",
)
SIGNED_SLACK_NAMES = frozenset(("WNS_ns", "TNS_ns", "WNS", "TNS", "wns", "tns"))


def measured_costs(metrics: torch.Tensor, metric_names: Sequence[str],
                   metric_mask: torch.Tensor | None = None):
    """Convert native PPA labels to costs; missing labels remain NaN.

Only explicitly named slack metrics change sign: cost=max(-slack,0). Wirelength,
overflow, power and area are already nonnegative costs. Missing values are never
converted to successful zero-cost labels. This conversion does not normalize.
    """
    metrics = torch.as_tensor(metrics)
    if not metrics.is_floating_point():
        metrics = metrics.float()
    if metrics.ndim != 2 or metrics.shape[1] != len(metric_names):
        raise ValueError("metrics must be [rows, len(metric_names)]")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("metric_names must be unique")
    observed = torch.isfinite(metrics)
    if metric_mask is not None:
        metric_mask = torch.as_tensor(metric_mask, device=metrics.device)
        if metric_mask.shape != metrics.shape:
            raise ValueError("metric_mask must have the metric matrix shape")
        observed = observed & metric_mask.to(device=metrics.device, dtype=torch.bool)
    columns = []
    for i, name in enumerate(metric_names):
        column = (-metrics[:, i]).clamp_min(0) if name in SIGNED_SLACK_NAMES else metrics[:, i]
        if bool(((column < 0) & observed[:, i]).any()):
            raise ValueError(f"negative observed cost for {name}; check native label units/sign")
        columns.append(column)
    costs = torch.stack(columns, dim=-1)
    return costs.masked_fill(~observed, float("nan")), observed


def smooth_macro_hpwl(extractor: PhysicalRidgeCritic, actions: torch.Tensor,
                      temperature: float = .05):
    """Log-sum-exp macro HPWL, in cache units, differentiable in raw centers.

Temperature is in normalized coordinate units. This is a macro-only analytic
prior, not routed wirelength and not a legalizer derivative. The smooth span
includes its usual temperature-dependent positive offset; the fitted residual
baseline has an intercept and uses the same temperature throughout.
    """
    if temperature <= 0:
        raise ValueError("smooth HPWL temperature must be positive")
    hpwl = extractor.hpwl
    x = actions.reshape(len(actions), len(hpwl.names), 2)
    if hpwl.num_nets == 0:
        return x.sum((1, 2)) * 0
    canvas = hpwl.canvas.to(x.dtype)
    pins = x[:, hpwl.pin_nodes] + 2 * hpwl.pin_offsets.to(x.dtype) / canvas
    index = hpwl.pin_nets[None, :, None].expand(len(x), -1, 2)
    shape = (len(x), hpwl.num_nets, 2)

    def logsumexp_by_net(values):
        maximum = x.new_full(shape, -float("inf")).scatter_reduce(
            1, index, values, reduce="amax", include_self=True)
        exponential = torch.exp(values - maximum.gather(1, index))
        total = x.new_zeros(shape).scatter_add(1, index, exponential)
        return maximum + torch.log(total)

    spans = temperature * (logsumexp_by_net(pins / temperature) +
                           logsumexp_by_net(-pins / temperature))
    return (spans * canvas / 2).sum((1, 2))


def _masked_stats(values, observed, relative_floor=.03, absolute_floor=1e-8):
    """Per-column TRAIN mean, regression scale and objective magnitude scale."""
    count = observed.sum(0)
    safe_count = count.clamp_min(1).to(values.dtype)
    safe = torch.where(observed, values, torch.zeros_like(values))
    mean = safe.sum(0) / safe_count
    variance = torch.where(observed, (safe - mean).square(), torch.zeros_like(values)).sum(0) / safe_count
    # Relative objective weights refer to mean measured TRAIN magnitudes, not
    # the arbitrary spread of each validation/test distribution.
    magnitude = safe.abs().sum(0) / safe_count
    objective_scale = torch.where(magnitude > absolute_floor, magnitude, torch.ones_like(magnitude))
    std = torch.maximum(variance.sqrt(), relative_floor * objective_scale).clamp_min(absolute_floor)
    return mean, std, objective_scale, count


class MultiMetricCritic(nn.Module):
    """Independent MLP ensemble predicting multiple measured terminal costs.

``train_actions`` accepts [B,N,2] or [B,2N]; N follows cache node order.
``train_costs`` is [B,K] with nonnegative costs and NaN for missing labels.
Predictions are [B,ensemble,K] in original cost units. ``reward`` is [B].

Constructor inputs must contain only TRAIN rows. If split labels are supplied,
they are checked, not silently filtered. Independently selected validation/test
rows never contribute normalizers or the optional TRAIN analytic baseline.
    """
    def __init__(self, cache, train_actions, train_costs, movable, metric_names,
                 *, train_mask=None, train_ids=None, train_splits=None,
                 width=128, depth=2, ensemble_size=2, dropout=0., layer_norm=True,
                 feature_std_floor=1e-5, target_relative_floor=.03,
                 hpwl_residual=False, baseline_metric_names=None,
                 smooth_temperature=.05, baseline_ridge=.001, seed=None):
        super().__init__()
        if width < 1 or depth < 1 or ensemble_size < 2:
            raise ValueError("positive width/depth and at least two ensemble members are required")
        if not 0 <= dropout < 1 or feature_std_floor <= 0 or target_relative_floor <= 0:
            raise ValueError("invalid dropout or normalization floors")
        if baseline_ridge < 0 or smooth_temperature <= 0:
            raise ValueError("invalid analytic baseline configuration")
        self.metric_names = tuple(metric_names)
        if not self.metric_names or len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must be nonempty and unique")
        actions = torch.as_tensor(train_actions)
        if not actions.is_floating_point():
            actions = actions.float()
        costs = torch.as_tensor(train_costs, device=actions.device, dtype=actions.dtype)
        if costs.ndim != 2 or costs.shape != (len(actions), len(self.metric_names)) or len(actions) < 2:
            raise ValueError("need at least two TRAIN rows, with train_costs [rows,metrics]")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("TRAIN actions must be finite")
        if train_splits is not None:
            if len(train_splits) != len(actions) or any(str(s).lower() not in {"train", "0"} for s in train_splits):
                raise ValueError("critic normalization accepts TRAIN rows only")
        ids = [str(i) for i in range(len(actions))] if train_ids is None else [str(i) for i in train_ids]
        if len(ids) != len(actions) or len(set(ids)) != len(ids):
            raise ValueError("train_ids must identify every TRAIN row uniquely")
        observed = torch.isfinite(costs)
        if train_mask is not None:
            train_mask = torch.as_tensor(train_mask, device=actions.device)
            if train_mask.shape != costs.shape:
                raise ValueError("train_mask must have the cost matrix shape")
            observed &= train_mask.to(device=actions.device, dtype=torch.bool)
        if bool(((costs < 0) & observed).any()):
            raise ValueError("train_costs must be nonnegative; convert signed slacks first")
        if bool((observed.sum(0) < 2).any()):
            raise ValueError("each selected metric requires at least two measured TRAIN labels")
        zero = dict(mean=np.zeros(17), std=np.ones(17), coeff=np.zeros(17), intercept=0.)
        self.extractor = PhysicalRidgeCritic(cache, zero).to(device=actions.device, dtype=actions.dtype)
        self.register_buffer("movable", torch.as_tensor(movable, device=actions.device, dtype=torch.bool))
        if self.movable.ndim != 1 or len(self.movable) != len(self.extractor.hpwl.names) or not bool(self.movable.any()):
            raise ValueError("movable must select at least one node in cache order")
        if len(self.movable) < 2:
            raise ValueError("physical feature baseline requires at least two nodes")
        with torch.no_grad():
            features = self.features(actions)
            if not bool(torch.isfinite(features).all()):
                raise ValueError("nonfinite TRAIN physical features")
            feature_mean = features.mean(0)
            feature_std = features.std(0, unbiased=False)
            target_mean, target_std, objective_scale, count = _masked_stats(
                costs, observed, relative_floor=target_relative_floor)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", torch.where(feature_std > feature_std_floor, feature_std, torch.ones_like(feature_std)))
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("objective_scale", objective_scale)
        self.register_buffer("train_observed_count", count)
        self.register_buffer("baseline_slope", costs.new_zeros(len(self.metric_names)))
        self.register_buffer("baseline_intercept", costs.new_zeros(len(self.metric_names)))
        self.register_buffer("hpwl_mean", costs.new_zeros(()))
        self.register_buffer("hpwl_std", costs.new_ones(()))
        self.hpwl_residual = bool(hpwl_residual)
        self.smooth_temperature = float(smooth_temperature)
        selected_baseline = tuple(baseline_metric_names) if baseline_metric_names is not None else tuple(
            n for n in self.metric_names if n in {"rWL_um", "wirelength", "WL", "grt_wirelength_um"})
        if set(selected_baseline) - set(self.metric_names):
            raise ValueError("baseline metric is absent from metric_names")
        if self.hpwl_residual and not selected_baseline:
            raise ValueError("hpwl_residual needs an explicitly named wirelength metric")
        if self.hpwl_residual:
            with torch.no_grad():
                h = self._smooth_hpwl_feature(actions)
                self.hpwl_mean.copy_(h.mean())
                h_std = h.std(unbiased=False)
                self.hpwl_std.copy_(h_std if h_std > feature_std_floor else h_std.new_ones(()))
                z = (h - self.hpwl_mean) / self.hpwl_std
                for k, name in enumerate(self.metric_names):
                    if name not in selected_baseline:
                        continue
                    x = z[observed[:, k]]
                    y = (costs[observed[:, k], k] - target_mean[k]) / target_std[k]
                    xc, yc = x - x.mean(), y - y.mean()
                    slope = (xc * yc).sum() / (xc.square().sum() + baseline_ridge).clamp_min(1e-12)
                    self.baseline_slope[k] = slope
                    self.baseline_intercept[k] = y.mean() - slope * x.mean()
        # fork_rng makes per-member initialization reproducible without changing
        # the caller's global random stream. Each member has independent storage.
        members = []
        initialization_seed = int(seed) if seed is not None else int(torch.randint(0, 2**31 - 1, ()).item())
        for index in range(ensemble_size):
            # Linear layers initialize on CPU, then move to the action device.
            # Do not seed unrelated/unallocated GPU random generators.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(initialization_seed + 104729 * index)
                layers = []
                dimensions = features.shape[1]
                for layer in range(depth):
                    layers.append(nn.Linear(dimensions, width))
                    if layer_norm and layer == 0:
                        layers.append(nn.LayerNorm(width))
                    layers.append(nn.SiLU())
                    if dropout:
                        layers.append(nn.Dropout(dropout))
                    dimensions = width
                output = nn.Linear(width, len(self.metric_names))
                nn.init.normal_(output.weight, std=.001)
                nn.init.zeros_(output.bias)
                layers.append(output)
                members.append(nn.Sequential(*layers).to(device=actions.device, dtype=actions.dtype))
        self.members = nn.ModuleList(members)
        self.spec = dict(schema="qstudy_multimetric_critic_v1", metric_names=list(self.metric_names),
                         width=width, depth=depth, ensemble_size=ensemble_size, dropout=float(dropout),
                         layer_norm=bool(layer_norm), hpwl_residual=self.hpwl_residual,
                         smooth_temperature=self.smooth_temperature,
                         baseline_metric_names=list(selected_baseline), baseline_ridge=float(baseline_ridge),
                         feature_std_floor=float(feature_std_floor), target_relative_floor=float(target_relative_floor),
                         feature_dim=features.shape[1], normalization="explicit_offline_TRAIN_only",
                         uncertainty="heuristic ensemble std; no CQL/confidence guarantee")
        self._train_metadata = dict(rows=len(actions), train_id_sha256=hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest())

    def _coordinates(self, actions):
        if actions.ndim == 2 and actions.shape[1] == len(self.movable) * 2:
            return actions.reshape(len(actions), len(self.movable), 2)
        if actions.ndim == 3 and actions.shape[1:] == (len(self.movable), 2):
            return actions
        raise ValueError("actions must be [B,N,2] or [B,2N] in cache order")

    def features(self, actions):
        x = self._coordinates(actions)
        return torch.cat((self.extractor.features(x.flatten(1)), x[:, self.movable].flatten(1)), -1)

    def _smooth_hpwl_feature(self, actions):
        return smooth_macro_hpwl(self.extractor, self._coordinates(actions), self.smooth_temperature) / self.extractor.hpwl.canvas.sum()

    def baseline_standardized(self, actions):
        if not self.hpwl_residual:
            return actions.new_zeros((len(actions), len(self.metric_names)))
        h = (self._smooth_hpwl_feature(actions) - self.hpwl_mean) / self.hpwl_std
        return h[:, None] * self.baseline_slope + self.baseline_intercept

    def standardized(self, actions):
        """Cost predictions [B,E,K] in frozen TRAIN target-standardized units."""
        x = (self.features(actions) - self.feature_mean) / self.feature_std
        residual = torch.stack([member(x) for member in self.members], dim=1)
        return residual + self.baseline_standardized(actions)[:, None, :]

    def forward(self, actions):
        """Measured-cost predictions, original units; no nonnegative clipping.

Clipping predictions would create zero-gradient regions. Physically impossible
negative predictions should instead be tracked as a calibration diagnostic.
        """
        return self.target_mean + self.target_std * self.standardized(actions)

    def weights_tensor(self, weights, *, device=None, dtype=None):
        if isinstance(weights, Mapping):
            unknown = set(weights) - set(self.metric_names)
            if unknown:
                raise ValueError(f"unknown objective metrics: {sorted(unknown)}")
            weights = [weights.get(name, 0.) for name in self.metric_names]
        result = torch.as_tensor(weights, device=device or self.target_mean.device,
                                 dtype=dtype or self.target_mean.dtype)
        if result.shape != self.target_mean.shape or not bool(torch.isfinite(result).all()) or bool((result < 0).any()) or not bool((result > 0).any()):
            raise ValueError("objective weights must be finite, nonnegative and include an active metric")
        return result

    def member_rewards(self, actions, weights):
        costs = self(actions)
        w = self.weights_tensor(weights, device=costs.device, dtype=costs.dtype)
        return -(costs / self.objective_scale * w).sum(-1)

    def reward(self, actions, weights, uncertainty_beta=0.):
        """Mean reward minus heuristic ensemble disagreement penalty, [B]."""
        if uncertainty_beta < 0:
            raise ValueError("uncertainty_beta must be nonnegative")
        rewards = self.member_rewards(actions, weights)
        return rewards.mean(1) - uncertainty_beta * rewards.std(1, unbiased=False)

    def observed_reward(self, costs, weights, metric_mask=None):
        """Measured scalar reward; any missing active metric makes row NaN."""
        costs = torch.as_tensor(costs, device=self.target_mean.device, dtype=self.target_mean.dtype)
        if costs.ndim != 2 or costs.shape[1] != len(self.metric_names):
            raise ValueError("costs must be [rows,metrics]")
        observed = torch.isfinite(costs)
        if metric_mask is not None:
            metric_mask = torch.as_tensor(metric_mask, device=costs.device)
            if metric_mask.shape != costs.shape:
                raise ValueError("metric_mask shape differs from costs")
            observed &= metric_mask.to(device=costs.device, dtype=torch.bool)
        w = self.weights_tensor(weights)
        valid = (observed | (w == 0)[None]).all(1)
        safe = torch.where(observed, costs, torch.zeros_like(costs))
        rewards = -(safe / self.objective_scale * w).sum(-1)
        return rewards.masked_fill(~valid, float("nan"))

    def loss(self, actions, costs, metric_mask=None, *, regression="mse", huber_delta=1.,
             ranking_weight=0., difference_weight=0., pairs=None, metric_weights=None,
             ranking_temperature=1., tie_tolerance=.01):
        """Supervised regression plus optional per-metric pair objectives.

``pairs`` is [P,2] row indices. Supply explicit same-parent pairs to evaluate a
local-difference hypothesis; otherwise all within-batch pairs are used. Pair
losses compare standardized *cost* differences: larger true cost must have a
larger predicted cost. Missing labels and target ties are ignored for ranking.
No validation-based fitting, automatic pairing by heldout outcomes, or surrogate
predictions posing as measured targets occurs here.
        """
        if regression not in {"mse", "huber"} or huber_delta <= 0:
            raise ValueError("regression must be mse/huber with positive huber_delta")
        if min(ranking_weight, difference_weight, tie_tolerance) < 0 or ranking_temperature <= 0:
            raise ValueError("pair weights/tolerance must be nonnegative and temperature positive")
        predicted = self.standardized(actions)
        costs = torch.as_tensor(costs, device=predicted.device, dtype=predicted.dtype)
        if costs.shape != (len(actions), len(self.metric_names)):
            raise ValueError("costs must be [rows,metrics]")
        observed = torch.isfinite(costs)
        if metric_mask is not None:
            metric_mask = torch.as_tensor(metric_mask, device=costs.device)
            if metric_mask.shape != costs.shape:
                raise ValueError("metric_mask shape differs from costs")
            observed &= metric_mask.to(device=costs.device, dtype=torch.bool)
        if bool(((costs < 0) & observed).any()):
            raise ValueError("cost labels must be nonnegative")
        w = (self.weights_tensor(metric_weights) if metric_weights is not None else torch.ones_like(self.target_mean))
        if not bool((observed & (w > 0)[None]).any()):
            raise ValueError("batch has no observed active metric labels")
        safe = torch.where(observed, costs, self.target_mean[None])
        target = (safe - self.target_mean) / self.target_std
        residual = predicted - target[:, None]
        elements = residual.square() if regression == "mse" else F.huber_loss(
            predicted, target[:, None].expand_as(predicted), delta=huber_delta, reduction="none")
        reg = _weighted_masked_mean(elements, observed[:, None], w)
        rank, delta = predicted.sum() * 0, predicted.sum() * 0
        pair_count = 0
        if ranking_weight or difference_weight:
            if pairs is None:
                pair_indices = torch.triu_indices(len(actions), len(actions), offset=1, device=predicted.device).T
            else:
                pair_indices = torch.as_tensor(pairs, device=predicted.device, dtype=torch.long)
                if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
                    raise ValueError("pairs must be [P,2]")
                if pair_indices.numel() and (bool((pair_indices < 0).any()) or bool((pair_indices >= len(actions)).any())):
                    raise ValueError("pair indices are outside this batch")
            if len(pair_indices):
                i, j = pair_indices.unbind(1)
                truth_difference = target[i] - target[j]
                prediction_difference = predicted[i] - predicted[j]
                both = observed[i] & observed[j]
                pair_count = int((both & (w > 0)[None]).sum().detach())
                if ranking_weight:
                    distinct = both & (truth_difference.abs() > tie_tolerance)
                    element = F.softplus(-truth_difference.sign()[:, None] * prediction_difference / ranking_temperature)
                    rank = _weighted_masked_mean(element, distinct[:, None], w)
                if difference_weight:
                    element = (prediction_difference - truth_difference[:, None]).square()
                    delta = _weighted_masked_mean(element, both[:, None], w)
        total = reg + ranking_weight * rank + difference_weight * delta
        return total, {"regression": reg, "ranking": rank, "difference": delta,
                       "observed_labels": int(observed.sum().detach()), "pair_metric_count": pair_count}

    def normalization_metadata(self):
        """JSON-safe provenance and train-fitted statistics for run records."""
        return {**self._train_metadata, "metric_names": list(self.metric_names),
                "feature_mean": self.feature_mean.detach().cpu().tolist(),
                "feature_std": self.feature_std.detach().cpu().tolist(),
                "target_mean": self.target_mean.detach().cpu().tolist(),
                "target_std": self.target_std.detach().cpu().tolist(),
                "objective_scale": self.objective_scale.detach().cpu().tolist(),
                "train_observed_count": self.train_observed_count.detach().cpu().tolist(),
                "baseline_slope": self.baseline_slope.detach().cpu().tolist(),
                "baseline_intercept": self.baseline_intercept.detach().cpu().tolist(),
                "normalization": "explicit_offline_TRAIN_only"}

    def get_extra_state(self):
        return {"spec": self.spec, "train_metadata": self._train_metadata}

    def set_extra_state(self, state):
        if state["spec"] != self.spec:
            raise ValueError("checkpoint critic specification differs from constructed model")
        self._train_metadata = state["train_metadata"]


def _weighted_masked_mean(elements, mask, weights):
    # elements [B,E,K]; masks [B,1,K] are broadcast over independent members.
    weighted = mask.to(elements.dtype).expand_as(elements) * weights
    denominator = weighted.sum()
    return (elements * weighted).sum() / denominator.clamp_min(torch.finfo(elements.dtype).tiny)


@contextmanager
def frozen_critic(critic):
    """Disable critic updates/dropout while preserving gradients into actions."""
    flags = [(p, p.requires_grad) for p in critic.parameters()]
    modes = [(module, module.training) for module in critic.modules()]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    critic.eval()
    try:
        yield critic
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training


def multimetric_actor_loss(prediction, teacher_target, critic, weights, movable,
                           *, alpha=10., q_weight=1., uncertainty_beta=0.,
                           clip_for_q=True):
    """Same-noise distillation minus Q reward; teacher target never receives grad.

Caller supplies raw actor output with fixed nodes already restored and teacher
coordinates generated using the same circuit/noise. The raw output is used for
distillation; bounded coordinates are sent to Q when clip_for_q is enabled.
    """
    if alpha < 0 or q_weight < 0:
        raise ValueError("actor loss coefficients must be nonnegative")
    if prediction.shape != teacher_target.shape:
        raise ValueError("actor and teacher layouts must have identical shapes")
    x = critic._coordinates(prediction)
    teacher = critic._coordinates(teacher_target.detach())
    movable = torch.as_tensor(movable, device=x.device, dtype=torch.bool)
    if movable.shape != critic.movable.shape or not bool(movable.any()):
        raise ValueError("movable mask differs from cache node shape")
    distillation = (x[:, movable] - teacher[:, movable]).square().mean()
    with frozen_critic(critic):
        reward = critic.reward(x.clamp(-1, 1) if clip_for_q else x, weights, uncertainty_beta).mean()
    return alpha * distillation - q_weight * reward, {"distillation": distillation, "reward": reward}
