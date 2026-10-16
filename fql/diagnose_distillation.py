"""Test whether student distillation generalizes beyond memorizing noise pairs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from .core import PlacementData, write_json
except ImportError:
    from core import PlacementData, write_json


def diagnostics(data, prediction=None):
    train = data.split == 0
    train_noise = data.noise[train][:, data.movable_mask]
    train_targets = data.teacher[train][:, data.movable_mask]
    center = train_targets.mean(axis=0)
    answer = {"interpretation": "Noise-to-layout fit diagnostics; no legality or PPA claim.",
              "macro_identity": "Coordinate labels are fixed; no arbitrary node permutations or Hungarian matching are applied.",
              "data": data.metadata, "splits": {}}
    for value, name in [(1, "validation"), (2, "heldout")]:
        select = data.split == value
        z = data.noise[select][:, data.movable_mask]
        y = data.teacher[select][:, data.movable_mask]
        nearest = []
        nearest_distances = []
        for chunk in np.array_split(z, max(1, (len(z)+63)//64)):
            squared = np.square(chunk[:, None, :] - train_noise[None, :, :]).mean(axis=-1)
            idx = squared.argmin(axis=1)
            nearest.append(train_targets[idx])
            nearest_distances.extend(squared[np.arange(len(idx)), idx].tolist())
        neighbor = np.concatenate(nearest)
        result = {"count": len(y), "train_mean_layout_mse": float(np.square(y-center).mean()),
                  "nearest_train_noise_layout_mse": float(np.square(y-neighbor).mean()),
                  "nearest_train_noise_mean_squared_distance": float(np.mean(nearest_distances)),
                  "teacher_coordinate_std": float(y.std(axis=0).mean()),
                  "teacher_unique_layouts_round_6_decimals": len(np.unique(y.round(6), axis=0))}
        if prediction is not None:
            p = prediction[select][:, data.movable_mask]
            mse = float(np.square(p-y).mean())
            denominator = result["train_mean_layout_mse"]
            result.update(student_mse=mse, student_coordinate_std=float(p.std(axis=0).mean()),
                          fraction_of_mean_baseline_mse=mse/denominator if denominator else None,
                          student_unique_layouts_round_6_decimals=len(np.unique(p.round(6), axis=0)))
        answer["splits"][name] = result
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--proposals", help="Optional paired student proposals NPZ")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run dataset-wide diagnostics in Slurm")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    data = PlacementData.load(args.dataset)
    prediction = None
    if args.proposals:
        with np.load(args.proposals, allow_pickle=False) as values:
            if not np.array_equal(data.noise, values["noise"]) or not np.array_equal(data.split, values["split"]):
                raise ValueError("Proposals do not use the same noise and split")
            prediction = values["raw_action"]
    result = diagnostics(data, prediction)
    result["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    write_json(args.output, result)


if __name__ == "__main__":
    main()
