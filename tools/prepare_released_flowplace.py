"""Create a fresh graph experiment using the frozen released FlowPlace teacher.

Run from the repository root with python -m tools.prepare_released_flowplace.
Only JSON is read/written here; training is a separate Slurm command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

from fql.flow_policy_config import RELEASED, STEPS, PUBLIC_SHA, validate_teacher_config


def prepare(source, output, run_prefix):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_prefix):
        raise ValueError("Use letters, digits, underscores or hyphens in the run prefix")
    cfg = json.loads(source.read_text())
    if "actor_warmup_steps" not in cfg or "offline_steps" not in cfg:
        raise ValueError("Expected a graph_o2o training configuration")
    if run_prefix == cfg.get("run_id_prefix", "go2o_v1"):
        raise ValueError("A new teacher experiment requires a distinct physical run prefix")
    cfg.update(flow_policy=RELEASED, flow_steps=STEPS, flow_checkpoint_sha256=PUBLIC_SHA,
               flow_warmup_steps=0, online_flow_update=False, run_id_prefix=run_prefix,
               configuration_template=str(source),
               configuration_template_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               flow_note="Frozen public FlowPlace weights and original 20-step constrained sampler; "
                         "fresh student distillation; only actor and critic update")
    validate_teacher_config(cfg)
    # Never mutate/reuse a historical run or import its warmed teacher/student.
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    args = parser.parse_args()
    cfg = prepare(args.source, args.output, args.run_prefix)
    print(json.dumps(dict(config=str(args.output.resolve()/"config.json"),
                          flow_policy=cfg["flow_policy"], flow_steps=cfg["flow_steps"])))
    if cfg["design"] == "bp_fe" and "dataset_binding_sha256" not in cfg:
        print("sbatch --time=12:00:00 fql/graph_o2o_train.slurm --experiment "
              + shlex.quote(str(args.output.resolve())) + " --seed "
              + str(cfg.get("seed", 0)) + " --round 0")


if __name__ == "__main__":
    main()
