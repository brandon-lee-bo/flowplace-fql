"""Prospective bounded targets derived from measured native outcomes only.

Successful native costs C=sum_j(w_j*c_j/S_j) map to B=2*C/(1+C),
with utility -B. Confirmed physical/geometry failures have B=4. Missing or
unresolved feedback has no target. This module neither fits reference scales
nor transforms unrestricted model predictions, nor rewrites native rewards.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re

import torch


VERSION = "bounded_terminal_utility_v1"
METRIC_NAMES = (
    "rWL_um", "WNS_ns", "TNS_ns", "horizontal_overflow_sum",
    "vertical_overflow_sum",
)
SUCCESS_CAP = 2.0
FAILURE_COST = 4.0
CONFIRMED_FAILURES = frozenset(("geometry_failure", "physical_failure"))
_SPEC_KEYS = frozenset(("version", "metric_names", "weights", "scales",
                        "reference", "success_cap", "failure_cost"))
_REFERENCE_KEYS = frozenset(("split", "dataset_sha256", "train_rows_sha256"))
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _numbers(values, name, *, positive=False):
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain five finite numbers") from exc
    if len(values) != len(METRIC_NAMES):
        raise ValueError(f"{name} must contain five finite numbers")
    result = []
    for value in values:
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(f"{name} must contain numeric values")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
        if not math.isfinite(number) or (number <= 0 if positive else number < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        # Canonicalize negative zero without changing numerical preferences.
        result.append(0.0 if number == 0 else number)
    return result


def _reference(reference):
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_KEYS:
        raise ValueError("reference requires exactly split, dataset_sha256, train_rows_sha256")
    if reference["split"] != "train":
        raise ValueError("Utility scales must declare a TRAIN-only reference")
    result = {"split": "train"}
    for key in ("dataset_sha256", "train_rows_sha256"):
        value = reference[key]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"reference.{key} must be a nonempty SHA-256")
        result[key] = value.lower()
    return result


def make_spec(weights, scales, reference):
    """Declare immutable utility semantics; supplied scales are never fitted.

``reference`` is exactly ``{'split':'train', 'dataset_sha256':...,
'train_rows_sha256':...}``; hash the ordered reference TRAIN row identities.
Use the SAME reference/scales across sources in a matched source comparison.
    """
    return validate_spec(dict(version=VERSION, metric_names=list(METRIC_NAMES),
        weights=weights, scales=scales, reference=reference,
        success_cap=SUCCESS_CAP, failure_cost=FAILURE_COST))


def validate_spec(spec):
    """Return a canonical independent copy; reject unknown versions/semantics."""
    if not isinstance(spec, Mapping) or set(spec) != _SPEC_KEYS:
        raise ValueError("Utility spec has missing or unsupported fields")
    if spec["version"] != VERSION:
        raise ValueError("Unsupported utility spec version")
    if not isinstance(spec["metric_names"], (list, tuple)) or tuple(spec["metric_names"]) != METRIC_NAMES:
        raise ValueError("Utility metric order differs from native first-five metrics")
    for key, required in (("success_cap", SUCCESS_CAP), ("failure_cost", FAILURE_COST)):
        if isinstance(spec[key], (bool, str, bytes)) or spec[key] != required:
            raise ValueError(f"Unsupported utility constant {key}")
    weights = _numbers(spec["weights"], "weights")
    if not any(weight > 0 for weight in weights):
        raise ValueError("At least one utility weight must be active")
    return dict(version=VERSION, metric_names=list(METRIC_NAMES),
        weights=weights, scales=_numbers(spec["scales"], "scales", positive=True),
        reference=_reference(spec["reference"]), success_cap=SUCCESS_CAP,
        failure_cost=FAILURE_COST)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def scale_identity(spec):
    """Common physical scale identity, deliberately independent of weights."""
    spec = validate_spec(spec)
    return _identity({key: spec[key] for key in ("version", "metric_names", "scales", "reference")})


def utility_identity(spec):
    """Full objective identity, including the declared weights."""
    return _identity(validate_spec(spec))


def assert_common_scale(first, second):
    """Reject changed TRAIN reference or scales, with no approximate matching."""
    if scale_identity(first) != scale_identity(second):
        raise ValueError("Utility objectives do not share the exact common TRAIN scale")


def derive_cost_utility(native_metrics, statuses, spec, metric_mask=None):
    """Return measured ``cost`` B, ``utility`` -B and boolean validity masks.

Native metrics have shape [B,5] or [B,7] in the existing native order. Power
and area, if supplied, are not part of this version. Slack costs are
max(-signed_native_slack,0); active WL/overflow must already be nonnegative.
Only exact status ``success`` plus complete active metrics yields a successful
target. Confirmed ``physical_failure`` / ``geometry_failure`` yields B=4 even
without metrics. Other statuses and missing active labels stay NaN/invalid.

``metric_mask`` is optional boolean [B,5] or [B,7] (first five are used).
``native_metrics=None`` represents all missing labels, for example an all-
failure batch. Outputs preserve float32/float64 input dtype/device; other
numeric input dtypes promote to float32. Ordinary valid successful targets
retain gradients through native measurements, but unrestricted predicted
costs must NEVER be passed as native metrics or rationally transformed here.
    """
    spec = validate_spec(spec)
    if isinstance(statuses, (str, bytes)):
        raise ValueError("statuses must contain one native status string per row")
    try:
        statuses = list(statuses)
    except TypeError as exc:
        raise ValueError("statuses must contain one native status string per row") from exc
    if any(not isinstance(status, str) for status in statuses):
        raise ValueError("statuses must be native status strings")
    if native_metrics is None:
        metrics = torch.full((len(statuses), len(METRIC_NAMES)), float("nan"), dtype=torch.float32)
    else:
        metrics = torch.as_tensor(native_metrics)
        if metrics.is_complex() or metrics.dtype == torch.bool:
            raise ValueError("Native metrics must be real numeric values")
        if metrics.dtype not in (torch.float32, torch.float64):
            metrics = metrics.to(dtype=torch.float32)
    if metrics.ndim != 2 or metrics.shape[1] not in (5, 7):
        raise ValueError("Native metrics must be [rows,5] or [rows,7]")
    if len(statuses) != len(metrics):
        raise ValueError("Native status count differs from metric rows")
    observed = torch.isfinite(metrics[:, :5])
    if metric_mask is not None:
        mask = torch.as_tensor(metric_mask, device=metrics.device)
        if mask.ndim != 2 or mask.shape[0] != len(metrics) or mask.shape[1] not in (5, 7):
            raise ValueError("Native metric mask must be boolean [rows,5] or [rows,7]")
        if mask.dtype != torch.bool:
            raise ValueError("Native metric mask must be boolean")
        observed = observed & mask[:, :5]
    weights = torch.tensor(spec["weights"], dtype=torch.float64, device=metrics.device)
    scales = torch.tensor(spec["scales"], dtype=torch.float64, device=metrics.device)
    active = weights > 0
    values = metrics[:, :5].to(dtype=torch.float64)
    # Sanitize absent/inactive labels BEFORE arithmetic; NaN * zero is still NaN.
    safe = torch.where(observed & active, values, torch.zeros_like(values))
    costs = torch.cat((safe[:, :1], (-safe[:, 1:3]).clamp_min(0), safe[:, 3:]), dim=1)
    labels_valid = (observed[:, active] & (costs[:, active] >= 0)).all(dim=1)
    success_status = torch.tensor([s == "success" for s in statuses], dtype=torch.bool, device=metrics.device)
    confirmed_failure = torch.tensor([s in CONFIRMED_FAILURES for s in statuses], dtype=torch.bool, device=metrics.device)
    success = success_status & labels_valid
    valid = success | confirmed_failure
    result = metrics.new_full((len(metrics),), float("nan"))
    if bool(success.any()):
        # Preserve the direct ordinary path (and its ordinary gradients).
        # Finite inputs can still overflow/underflow an intermediate product
        # before division rescues it, so repair only affected rows in log space.
        selected = costs[success, :][:, active]
        product = selected * weights[active]
        terms = product / scales[active]
        c = terms.sum(dim=1)
        positive = selected > 0
        tiny = torch.finfo(torch.float64).tiny
        # Subnormal intermediates can lose substantial relative precision even
        # before rounding to zero; a small denominator can amplify that loss.
        exceptional = ((~torch.isfinite(terms)).any(dim=1) | ~torch.isfinite(c)
            | (positive & ((product < tiny) | (terms < tiny))).any(dim=1))
        bounded = torch.empty_like(c)
        small = ~exceptional & (c < 1)
        bounded[small] = SUCCESS_CAP * c[small] / (1 + c[small])
        large = ~exceptional & ~small
        bounded[large] = SUCCESS_CAP - SUCCESS_CAP / (1 + c[large])
        if bool(exceptional.any()):
            values = selected[exceptional]
            nonzero = values > 0
            # Avoid log(0) even in the unused branch, for clean zero gradients.
            log_terms = torch.where(nonzero, values, torch.ones_like(values)).log()
            log_terms = log_terms + weights[active].log() - scales[active].log()
            log_terms = log_terms.masked_fill(~nonzero, -float("inf"))
            log_c = torch.logsumexp(log_terms, dim=1)
            # Numerically evaluate 2*sigmoid(log(C)); some sigmoid kernels
            # prematurely return zero for log(C)<-709 even if B is representable.
            repaired = torch.empty_like(log_c)
            negative = log_c < 0
            repaired[negative] = torch.exp(log_c[negative] + math.log(SUCCESS_CAP)) / (1 + torch.exp(log_c[negative]))
            repaired[~negative] = SUCCESS_CAP * torch.sigmoid(log_c[~negative])
            bounded[exceptional] = repaired
        result[success] = bounded.to(dtype=metrics.dtype)
    result[confirmed_failure] = FAILURE_COST
    return dict(cost=result, utility=-result, valid=valid, success=success,
                confirmed_failure=confirmed_failure)
