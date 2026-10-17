"""Reload durable model states and compare actual GPU outputs to originals."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from .graph_o2o_archive import atomic_json, sha
# Register the project's analysis package before the upstream model adapter
# prepends its source path (which also contains an analysis.py module).
from .graph_o2o_train import initialize_problem
from .graph_o2o_models import GraphPolicy, FeatureCritics


def check(plan_path):
    plan = json.loads(plan_path.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, digest in plan["source_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Weight reload checker source changed: " + name)
    for binding in plan["input_bindings"]:
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Weight reload graph/data input changed")
    source_plan = Path(plan["preservation_plan"])
    if sha(source_plan) != plan["preservation_plan_sha256"]:
        raise ValueError("Original preservation plan changed")
    items = json.loads(source_plan.read_text())
    manifest_path = Path(items["output"]) / "manifest.json"
    if sha(manifest_path) != plan["preservation_manifest_sha256"]:
        raise ValueError("Completed preservation manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["count"] != len(items["checkpoints"]):
        raise ValueError("Preserved model denominator differs")
    problem, _, _, cache, x, y, _, _, _ = initialize_problem(argparse.Namespace(seed=0))
    flow = GraphPolicy(problem)
    actor = copy.deepcopy(flow)
    critic = FeatureCritics(cache, x, y, actor.movable, width=128).cuda()
    for model in (flow, actor, critic):
        model.eval()
    checks = []
    with torch.no_grad():
        for item, archived in zip(items["checkpoints"], manifest["records"]):
            source = Path(item["checkpoint"])
            cfg = json.loads(Path(item["config"]).read_text())
            if cfg["design"] != "bp_fe" or cfg["critic_width"] != 128:
                raise ValueError("This registered reload check is for bp_fe width128 only")
            if archived["source_checkpoint_sha256"] != item["checkpoint_sha256"]:
                raise ValueError("Preserved checkpoint identity mismatch")
            if sha(source) != item["checkpoint_sha256"]:
                raise ValueError("Original model changed")
            portable = Path(archived["inference_weights"])
            if sha(portable) != archived["inference_weights_sha256"]:
                raise ValueError("Durable model changed")
            requests = source.parent.parent / "requests" / source.parent.name / "evaluation"
            noises = []
            for i in range(4):
                with np.load(requests / str(i) / "actions.npz", allow_pickle=False) as data:
                    noises.append(data["noise"].copy())
            noise = torch.tensor(np.stack(noises), device="cuda")
            outcomes = []
            for checkpoint in (source, portable):
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if checkpoint == portable and "optimizers" in saved:
                    raise ValueError("Portable archive unexpectedly contains optimizer state")
                flow.load_state_dict(saved["flow"], strict=True)
                actor.load_state_dict(saved["actor"], strict=True)
                critic.load_state_dict(saved["critic"], strict=True)
                raw = actor(noise)
                outcomes.append(dict(actor=raw.cpu().clone(), critic=critic(raw).cpu().clone(),
                                     velocity=flow.velocity(noise, noise.new_full((4,), .35)).cpu().clone(),
                                     euler10=flow.sample_flow(noise, steps=10).cpu().clone()))
                del saved
            for name in outcomes[0]:
                if not torch.equal(outcomes[0][name], outcomes[1][name]):
                    raise ValueError("Reloaded actual output differs: " + item["id"] + "/" + name)
            checks.append(dict(id=item["id"], original_checkpoint_sha256=item["checkpoint_sha256"],
                               preserved_weights_sha256=archived["inference_weights_sha256"],
                               paired_actual_noise_count=4, exactly_equal_outputs=list(outcomes[0])))
            print(json.dumps(checks[-1]), flush=True)
    dest = Path(plan["output"])
    dest.mkdir(parents=True, exist_ok=False)
    atomic_json(dest / "evidence.json", dict(status="all_durable_models_reload_with_exact_GPU_outputs",
                plan_sha256=sha(plan_path), count=len(checks), checks=checks,
                gpu_model=torch.cuda.get_device_name(), torch_version=torch.__version__,
                original_and_durable_compared_on_same_GPU=True, new_training_steps=0, new_EDA_runs=0,
                optimizer_resume_not_tested=True, slurm_job_id=os.environ["SLURM_JOB_ID"], goal_completed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Use an allocated Slurm GPU")
    check(args.plan)
