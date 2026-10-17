"""Preserve exact inference state from completed checkpoints without moving them.

Optimizer/resume state remains in the original checkpoint on scratch. The
durable copy contains every other field, including all three networks/buffers
and RNG state. This does not claim to archive native EDA or resume checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch

from backend.scripts.result_storage import MIN_HEADROOM, storage_status
from .graph_o2o_archive import atomic_json, sha


def same_tree(original, restored, path="root"):
    """Compare all fields and exact CPU tensor values, including model buffers."""
    if isinstance(original, torch.Tensor):
        if not isinstance(restored, torch.Tensor) or (
            original.dtype != restored.dtype or original.shape != restored.shape
            or not torch.equal(original, restored)
        ):
            raise ValueError("Preserved tensor differs: " + path)
        return 1
    if isinstance(original, dict):
        if not isinstance(restored, dict) or set(original) != set(restored):
            raise ValueError("Preserved keys differ: " + path)
        return sum(same_tree(v, restored[k], path + "." + str(k)) for k, v in original.items())
    if isinstance(original, (tuple, list)):
        if type(original) is not type(restored) or len(original) != len(restored):
            raise ValueError("Preserved sequence differs: " + path)
        return sum(same_tree(v, restored[i], path + "." + str(i)) for i, v in enumerate(original))
    if type(original) is not type(restored) or original != restored:
        raise ValueError("Preserved metadata differs: " + path)
    return 0


def preserve(plan_path):
    plan = json.loads(plan_path.read_text())
    if sha(__file__) != plan["implementation_sha256"]:
        raise ValueError("Preservation implementation changed")
    if len({r["id"] for r in plan["checkpoints"]}) != len(plan["checkpoints"]):
        raise ValueError("Duplicate checkpoint preservation IDs")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ["TMPDIR"])
    scratch.mkdir(parents=True, exist_ok=True)
    records = []
    for row in plan["checkpoints"]:
        source = Path(row["checkpoint"])
        completion = Path(row["completion"])
        if (sha(source) != row["checkpoint_sha256"] or sha(completion) != row["completion_sha256"]
                or sha(row["config"]) != row["config_sha256"]):
            raise ValueError("Completed checkpoint binding changed: " + row["id"])
        done = json.loads(completion.read_text())
        if done["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("Completion record does not bind this checkpoint")
        full = torch.load(source, map_location="cpu", weights_only=False)
        if not all(k in full for k in ("flow", "actor", "critic", "optimizers")):
            raise ValueError("Expected full graph-FQL checkpoint")
        inference = {k: v for k, v in full.items() if k != "optimizers"}
        folder = output / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / "inference_weights.pt"
        marker = folder / "preservation.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if (previous["source_checkpoint_sha256"] != row["checkpoint_sha256"]
                    or sha(dest) != previous["inference_weights_sha256"]):
                raise ValueError("Existing preservation differs")
            same_tree(inference, torch.load(dest, map_location="cpu", weights_only=False))
            records.append(previous)
            continue
        temporary = scratch / (row["id"] + ".pt")
        torch.save(inference, temporary)
        tensor_count = same_tree(inference, torch.load(temporary, map_location="cpu", weights_only=False))
        size = temporary.stat().st_size
        status = storage_status(Path("/fact_data/bohanli"), output)
        for name in ("filesystem_free_bytes", "quota_remaining_bytes"):
            if name in status and status[name] is not None and status[name] - size < MIN_HEADROOM:
                raise RuntimeError("Durable copy would breach8GiB reserve: " + name)
        if dest.exists():
            # An interrupted post-copy verification can be recovered only from
            # exact content; no completed file is overwritten or silently adopted.
            same_tree(inference, torch.load(dest, map_location="cpu", weights_only=False))
        else:
            pending = folder / ("pending_job_" + os.environ["SLURM_JOB_ID"] + "_restart_"
                                + os.environ.get("SLURM_RESTART_COUNT", "0") + ".pt")
            with pending.open("xb") as target, temporary.open("rb") as stream:
                shutil.copyfileobj(stream, target, length=4 * 1024**2)
            if sha(pending) != sha(temporary):
                raise ValueError("Durable copy checksum mismatch")
            pending.rename(dest)
        restored = torch.load(dest, map_location="cpu", weights_only=False)
        if same_tree(inference, restored) != tensor_count:
            raise ValueError("Durable model state verification failed")
        if sha(source) != row["checkpoint_sha256"]:
            raise ValueError("Original checkpoint changed during preservation")
        for name, binding, digest in [("training_completion.json", completion, row["completion_sha256"]),
                                      ("config.json", Path(row["config"]), row["config_sha256"])]:
            target = folder / name
            if target.exists():
                if sha(target) != digest:
                    raise ValueError("Preserved configuration or completion differs")
            else:
                shutil.copyfile(binding, target)
            if sha(target) != digest:
                raise ValueError("Metadata copy checksum mismatch")
        record = dict(status="exact_inference_state_preserved", source_checkpoint=str(source),
                      source_checkpoint_sha256=row["checkpoint_sha256"], source_completion=str(completion),
                      source_completion_sha256=row["completion_sha256"], inference_weights=str(dest),
                      source_config=str(row["config"]), source_config_sha256=row["config_sha256"],
                      inference_weights_sha256=sha(dest), inference_bytes=size,
                      source_bytes=source.stat().st_size, exact_tensor_count=tensor_count,
                      preserved_top_level_fields=list(inference), omitted_fields=["optimizers"],
                      resume_training_supported=False, original_checkpoint_unchanged=True,
                      plan_sha256=sha(plan_path), slurm_job_id=os.environ["SLURM_JOB_ID"],
                      storage_before=status)
        atomic_json(marker, record)
        records.append(record)
        print(json.dumps(dict(checkpoint=row["id"], bytes=size, exact_tensors=tensor_count)), flush=True)
        del full, inference, restored
        temporary.unlink()
    atomic_json(output / "manifest.json", dict(status="all_requested_inference_states_preserved",
                plan_sha256=sha(plan_path), records=records, count=len(records),
                durable_bytes=sum(r["inference_bytes"] for r in records),
                original_resume_checkpoints_remain_on_scratch=True,
                native_EDA_archived=False, goal_completed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Use Slurm for checkpoint loading and preservation")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    preserve(args.plan)
