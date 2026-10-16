"""Small, auditable one-step policy and terminal-reward critic.

The critic is queried on raw (clipped) proposals and predicts the reward obtained
after an external legalizer. This module never substitutes predicted rewards for
measured placement quality. The legalizer and backend belong to the caller.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


def mlp(input_dim: int, output_dim: int, widths: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in widths:
        layers.extend([nn.Linear(input_dim, width), nn.GELU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class OneStepActor(nn.Module):
    def __init__(self, dim: int, widths: list[int], movable_mask, fixed_action):
        super().__init__()
        self.net = mlp(dim, dim, widths)
        self.register_buffer("movable_mask", torch.as_tensor(movable_mask, dtype=torch.bool))
        self.register_buffer("fixed_action", torch.as_tensor(fixed_action, dtype=torch.float32))

    def forward(self, noise, clip: bool = False):
        action = self.net(noise)
        if clip:
            action = action.clamp(-1, 1)
        return torch.where(self.movable_mask, action, self.fixed_action)


class Critics(nn.Module):
    def __init__(self, dim: int, widths: list[int], ensemble_size: int = 2):
        super().__init__()
        self.members = nn.ModuleList(mlp(dim, 1, widths) for _ in range(ensemble_size))

    def forward(self, action):
        return torch.cat([member(action) for member in self.members], dim=-1)


def distillation_loss(actor_raw, teacher_action, movable_mask):
    """Coordinate MEAN, not sum: alpha is invariant to number of macros."""
    return (actor_raw[..., movable_mask] - teacher_action[..., movable_mask]).square().mean()


def policy_loss(actor_raw, teacher_action, critic_values, movable_mask, alpha,
                q_weight=1.0, normalize_q=False):
    distill = distillation_loss(actor_raw, teacher_action, movable_mask)
    q = critic_values.mean(dim=-1)
    q_loss = -q.mean()
    if normalize_q:
        q_loss = q_loss / q.abs().mean().clamp_min(1e-6).detach()
    return alpha * distill + q_weight * q_loss, distill, q_loss


def _flat(array, name):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"{name} needs a sample axis and coordinate dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite entries")
    return array.reshape(len(array), -1)


def _make_split(n, seed):
    if n < 10:
        raise ValueError("Need at least 10 records to create train/validation/test splits")
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(0.15 * n))
    n_test = max(1, int(0.15 * n))
    split = np.zeros(n, dtype=np.int64)
    split[order[:n_val]] = 1
    split[order[n_val:n_val + n_test]] = 2
    return split


def _validate_split(split, n, label):
    original = np.asarray(split)
    split = np.asarray(split, dtype=np.int64)
    if not np.array_equal(original, split):
        raise ValueError(f"{label} contains non-integer labels")
    if split.shape != (n,) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError(f"{label} must be length-{n} integer labels 0/1/2")
    if any(not np.any(split == value) for value in [0, 1, 2]):
        raise ValueError(f"{label} must contain nonempty train, validation, heldout subsets")
    return split


def _check_group_split(groups, split, label):
    if len(groups) != len(split):
        raise ValueError(f"{label} has the wrong length")
    for group in np.unique(groups):
        if len(np.unique(split[groups == group])) > 1:
            raise ValueError(f"{label} group {group!r} appears in multiple splits")


@dataclass
class PlacementData:
    noise: np.ndarray
    teacher: np.ndarray
    raw_action: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    split: np.ndarray
    critic_split: np.ndarray
    movable_mask: np.ndarray
    fixed_action: np.ndarray
    coordinate_shape: tuple[int, ...]
    metadata: dict

    @classmethod
    def load(cls, path, split_seed=20261017, allow_random_split=False):
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        required = {"noise", "teacher_action", "raw_action", "action", "reward"}
        missing = required - set(arrays)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
        coordinate_shape = tuple(arrays["teacher_action"].shape[1:])
        noise = _flat(arrays["noise"], "noise")
        teacher = _flat(arrays["teacher_action"], "teacher_action")
        raw_action = _flat(arrays["raw_action"], "raw_action")
        action = _flat(arrays["action"], "action")
        reward = np.asarray(arrays["reward"], dtype=np.float32).reshape(-1)
        if noise.shape != teacher.shape:
            raise ValueError("noise and teacher_action must be paired with equal shapes")
        if raw_action.shape != action.shape or action.shape[1] != noise.shape[1]:
            raise ValueError("action and raw_action must match and use teacher coordinate dimension")
        if reward.shape != (len(action),) or not np.isfinite(reward).all():
            raise ValueError("reward must be finite, one value per raw_action/action record")
        dim = noise.shape[1]
        mask = np.asarray(arrays.get("movable_mask", np.ones(dim)), dtype=bool).reshape(-1)
        if mask.shape != (dim,) or not mask.any():
            raise ValueError("movable_mask has wrong shape or contains no movable coordinate")
        if not mask.all() and "fixed_action" not in arrays:
            raise ValueError("fixed_action is required if there are nonmovable coordinates")
        fixed = np.asarray(arrays.get("fixed_action", np.zeros(dim)), dtype=np.float32).reshape(-1)
        if fixed.shape != (dim,) or not np.isfinite(fixed).all():
            raise ValueError("fixed_action has wrong shape or nonfinite entries")
        for name, value in [("teacher_action", teacher), ("raw_action", raw_action), ("action", action)]:
            if np.any(np.abs(value[:, mask]) > 1.0001):
                raise ValueError(f"{name} movable positions are outside normalized [-1,1]; clip proposals before collection")
            if not mask.all() and not np.allclose(value[:, ~mask], fixed[~mask], atol=1e-6):
                raise ValueError(f"{name} changed fixed coordinates")
        if not allow_random_split and ("split" not in arrays or "critic_split" not in arrays):
            raise ValueError("Explicit split and critic_split required; --allow-random-split is only for independent records")
        split = _validate_split(arrays["split"] if "split" in arrays else _make_split(len(noise), split_seed),
                                len(noise), "split")
        critic_split = _validate_split(arrays["critic_split"] if "critic_split" in arrays
                                       else _make_split(len(action), split_seed + 1), len(action), "critic_split")
        if "group_id" in arrays:
            _check_group_split(arrays["group_id"], split, "group_id")
        if "critic_group_id" in arrays:
            _check_group_split(arrays["critic_group_id"], critic_split, "critic_group_id")
        # Optional parent index makes teacher/critic cross-dataset leakage auditable.
        if "critic_parent_index" in arrays:
            parent = np.asarray(arrays["critic_parent_index"], dtype=np.int64)
            if parent.shape != (len(action),):
                raise ValueError("critic_parent_index has wrong length")
            known = parent >= 0
            if np.any(parent[known] >= len(noise)):
                raise ValueError("critic_parent_index out of range")
            if np.any(critic_split[known] != split[parent[known]]):
                raise ValueError("Critic record and parent teacher noise are in different splits")
        metadata = {
            "dataset": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "teacher_records": len(noise), "critic_records": len(action), "dim": dim,
            "movable_coordinates": int(mask.sum()), "coordinate_shape": list(coordinate_shape),
            "split_seed": split_seed, "explicit_teacher_split": "split" in arrays,
            "explicit_critic_split": "critic_split" in arrays,
            "teacher_counts": {str(i): int((split == i).sum()) for i in range(3)},
            "critic_counts": {str(i): int((critic_split == i).sum()) for i in range(3)},
            "reward_scale": "unchanged; dataset producer defines reward units",
        }
        return cls(noise, teacher, raw_action, action, reward, split, critic_split,
                   mask, fixed, coordinate_shape, metadata)


def average_rank(values):
    """Spearman tie handling without a scipy dependency."""
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = starts + (counts - 1) / 2
    return ranks[inverse]


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def critic_diagnostics(predictions, rewards):
    predictions, rewards = np.asarray(predictions), np.asarray(rewards)
    means = predictions.mean(axis=-1)
    return {
        "count": len(rewards), "mae": float(np.abs(means - rewards).mean()),
        "rmse": float(np.sqrt(np.square(means - rewards).mean())),
        "pearson_r": correlation(means, rewards),
        "spearman_r": correlation(average_rank(means), average_rank(rewards)),
        "reward_std": float(rewards.std()),
        "ensemble_disagreement": float(predictions.std(axis=-1).mean()),
        "mean_prediction": float(means.mean()), "mean_reward": float(rewards.mean()),
        "max_absolute_error": float(np.abs(means - rewards).max()),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
