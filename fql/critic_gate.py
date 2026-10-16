"""Train a small terminal GRT critic and inspect validation before actor work.

No heldout predictions/metrics are computed. The gate is advisory, and does not
launch actor training or candidate backends. Also freezes best-observed-TRAIN
executed layout as an essential one-circuit optimization baseline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

try:
    from .core import Critics, critic_diagnostics, write_json
    from .relabel_backend import STAGE, action_sha256, sha256_file
    from .provenance import capture_source_provenance
except ImportError:
    from core import Critics, critic_diagnostics, write_json
    from relabel_backend import STAGE, action_sha256, sha256_file
    from provenance import capture_source_provenance


def load_train_validation(path):
    path = Path(path).resolve()
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("reward_stage") != STAGE or manifest.get("dataset_sha256") != sha256_file(path):
        raise ValueError("Gate requires a bound real-GRT reward dataset")
    with np.load(path, allow_pickle=False) as data:
        split = data["critic_split"]
        train_rows, val_rows = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
        if not len(train_rows) or not len(val_rows):
            raise ValueError("Nonempty train and validation required")
        arrays = {}
        for name, indices in [("train", train_rows), ("validation", val_rows)]:
            arrays[name] = {key: data[key][indices].copy() for key in
                            ["raw_action", "action", "reward", "backend_rwl_um", "source_row_index", "critic_parent_index", "legal"]}
            arrays[name]["original_local_rows"] = indices
        # Splits/group identifiers are metadata, not heldout labels.
        groups = data["critic_group_id"]
        if set(groups[train_rows]) & set(groups[val_rows]):
            raise ValueError("Train and validation noise families overlap")
        arrays["train"]["group_id"] = groups[train_rows]
        arrays["validation"]["group_id"] = groups[val_rows]
    scale = float(manifest["reward_scale_train_only"])
    if not np.isfinite(scale) or scale <= 0 or not np.isclose(arrays["train"]["backend_rwl_um"].mean(), scale, rtol=1e-12):
        raise ValueError("Reward scale must match retained training rWL mean")
    for name, values in arrays.items():
        if not values["legal"].all():
            raise ValueError("Backend gate expects independently legal executed layouts")
        np.testing.assert_allclose(values["reward"], -values["backend_rwl_um"] / scale, rtol=1e-6, atol=1e-7)
        for key in ["raw_action", "action", "reward", "backend_rwl_um"]:
            if not np.isfinite(values[key]).all():
                raise ValueError(f"Nonfinite {name} {key}")
    return arrays, manifest, manifest_path


def best_training_baseline(data, manifest, dataset_path, output):
    train = data["train"]
    # Lowest observed TRAIN wirelength; stable source-row tie break. No val/test use.
    order = np.lexsort((train["source_row_index"], train["backend_rwl_um"]))
    index = int(order[0])
    source_row = int(train["source_row_index"][index])
    labels = Path(dataset_path).with_name("backend_labels.jsonl")
    selected = None
    for line in labels.read_text().splitlines():
        record = json.loads(line)
        if record["row_index"] == source_row:
            if selected is not None:
                raise ValueError("Duplicate best-TRAIN backend record")
            selected = record
    if selected is None or selected["critic_split"] != 0:
        raise ValueError("Best-TRAIN layout lacks its original backend record")
    if selected["rWL_um"] != float(train["backend_rwl_um"][index]):
        raise ValueError("Best-TRAIN metrics disagree with relabeled dataset")
    if selected["action_sha256"] != action_sha256(train["action"][index]):
        raise ValueError("Best-TRAIN action provenance mismatch")
    for key, path_key in [("def_sha256", "def_path"), ("metrics_sha256", "metrics_path"), ("manifest_sha256", "manifest_path")]:
        if sha256_file(selected[path_key]) != selected[key]:
            raise ValueError("Best-TRAIN backend artifact changed")
    np.savez_compressed(output / "best_train_layout.npz", raw_action=train["raw_action"][index],
                        action=train["action"][index], source_row_index=source_row)
    report = {"kind": "best_observed_training_executed_layout", "selection_split": "train_only",
              "rule": "Minimum TRAIN GRT rWL; original source row breaks exact ties",
              "source_row_index": source_row, "parent_noise_index": int(train["critic_parent_index"][index]),
              "eligible_training_layout_count": len(train["reward"]),
              "training_noise_families": len(np.unique(train["group_id"])),
              "observed_rWL_um": float(train["backend_rwl_um"][index]),
              "action_artifact": str(output / "best_train_layout.npz"), "action_sha256": selected["action_sha256"],
              "backend_record": selected, "dataset_sha256": manifest["dataset_sha256"],
              "label_budget_total": manifest["num_critic"], "teacher_data_count": manifest["num_teacher"],
              "reuse_policy": "Reuse exact executed action and matching DEF/backend provenance; do not relegalize and silently reuse old reward",
              "replication": "One previously measured layout; not an independent replicate for each noise or training seed",
              "cost_policy": "Charge the complete eligible training-label search and original teacher/data collection; it is not a zero-data baseline"}
    write_json(output / "best_train_layout.json", report)
    return report


def metrics_with_variance(prediction, target):
    report = critic_diagnostics(prediction, target)
    mean = prediction.mean(axis=-1)
    report["prediction_std"] = float(mean.std())
    report["prediction_to_label_std_ratio"] = float(mean.std() / target.std()) if target.std() > 0 else None
    return report


def run_gate(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_provenance = capture_source_provenance(output / "source_snapshot")
    dataset = Path(args.dataset).resolve()
    data, manifest, source_manifest_path = load_train_validation(dataset)
    widths = [int(value) for value in args.widths.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)) or not seeds or any(width < 1 for width in widths):
        raise ValueError("Invalid seeds/widths")
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimization budget")
    device = torch.device(args.device)
    report = {"status": "running", "args": vars(args), "dataset_sha256": manifest["dataset_sha256"],
              "source_manifest": str(source_manifest_path), "source_manifest_sha256": sha256_file(source_manifest_path),
              "stage": STAGE, "not_final_table2": True, "heldout_used": False,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seeds": [],
              "source_provenance": source_provenance,
              "gate_rule": "Advisory only: at least 2/3 predeclared seeds beat train-mean constant MAE and have positive validation Spearman; no heldout evaluation or automatic expansion",
              "limitations": ["64 rows correspond to only 32 paired noise families in the initial budget",
                              "22-dimensional action space and a discontinuous legalizer create extrapolation risk",
                              "Validation accuracy on cached actions does not guarantee actor-selected off-support accuracy",
                              "Few validation families make correlation unstable; this is a diagnostic gate, not statistical proof"]}
    write_json(output / "report.json", report)
    try:
        best = best_training_baseline(data, manifest, dataset, output)
        report["best_train_baseline"] = {key: best[key] for key in ["source_row_index", "observed_rWL_um", "eligible_training_layout_count"]}
        train, validation = data["train"], data["validation"]
        train_x = torch.as_tensor(train["raw_action"].reshape(len(train["reward"]), -1), device=device, dtype=torch.float32)
        train_y = torch.as_tensor(train["reward"], device=device, dtype=torch.float32)
        val_x = torch.as_tensor(validation["raw_action"].reshape(len(validation["reward"]), -1), device=device, dtype=torch.float32)
        dim = train_x.shape[1]
        constant = float(train["reward"].mean())
        report["constant_baseline"] = {"value": constant,
            "train": metrics_with_variance(np.full((len(train_y), 1), constant), train["reward"]),
            "validation": metrics_with_variance(np.full((len(validation["reward"]), 1), constant), validation["reward"])}
        report["counts"] = {name: {"rows": len(values["reward"]), "noise_families": len(np.unique(values["group_id"]))}
                            for name, values in data.items()}
        distance = np.square(validation["raw_action"].reshape(len(val_x), 1, -1) - train["raw_action"].reshape(1, len(train_y), -1)).mean(axis=-1)
        report["validation_nearest_train_raw_mse"] = distance.min(axis=1).tolist()
        for seed in seeds:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            critic = Critics(dim, widths).to(device)
            optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            rng = np.random.default_rng(seed + 10000)
            started = time.perf_counter()
            losses = []
            for step in range(args.steps):
                index = torch.as_tensor(rng.choice(len(train_y), min(args.batch_size, len(train_y)), replace=True), device=device)
                prediction = critic(train_x[index])
                loss = (prediction - train_y[index, None]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite critic loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.)
                optimizer.step()
                if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
                    losses.append({"step": step + 1, "train_minibatch_mse": float(loss.detach())})
            critic.eval()
            with torch.no_grad():
                pred_train, pred_val = critic(train_x).cpu().numpy(), critic(val_x).cpu().numpy()
            scores = {"seed": seed, "seconds": time.perf_counter() - started,
                      "train": metrics_with_variance(pred_train, train["reward"]),
                      "validation": metrics_with_variance(pred_val, validation["reward"]), "loss_log": losses}
            val = scores["validation"]
            scores["beats_constant_validation_mae"] = val["mae"] < report["constant_baseline"]["validation"]["mae"]
            scores["positive_validation_rank"] = val["spearman_r"] is not None and val["spearman_r"] > 0
            folder = output / f"seed_{seed}"
            folder.mkdir()
            np.savez_compressed(folder / "train_validation_predictions.npz", train_prediction=pred_train,
                                validation_prediction=pred_val, train_source_row=train["source_row_index"],
                                validation_source_row=validation["source_row_index"])
            torch.save({"critic_state": critic.state_dict(), "widths": widths, "critic_widths": widths, "dim": dim, "seed": seed,
                        "dataset_sha256": manifest["dataset_sha256"], "reward_stage": STAGE,
                        "reward_scale_train_only": manifest["reward_scale_train_only"], "args": vars(args),
                        "source_provenance": source_provenance}, folder / "critic.pt")
            report["seeds"].append(scores)
            write_json(output / "report.json", report)
            print(json.dumps(scores), flush=True)
        passes = sum(row["beats_constant_validation_mae"] and row["positive_validation_rank"] for row in report["seeds"])
        enough = passes >= max(1, int(np.ceil(2 * len(seeds) / 3)))
        report.update(status="completed", seeds_meeting_advisory_gate=passes,
                      continuation_recommendation="Consider the predeclared small actor experiment; no off-support guarantee" if enough else
                      "Do not expand actor/backend sweeps yet; diagnose weak critic or prefer best-observed-TRAIN layout",
                      automatic_next_action=None)
        write_json(output / "report.json", report)
        return report
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--widths", default="64,64")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Train/evaluate the gate under Slurm")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_gate(args)


if __name__ == "__main__":
    main()
