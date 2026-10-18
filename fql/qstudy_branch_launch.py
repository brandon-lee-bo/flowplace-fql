"""Wait for an exact offline prefix, import it, then hand off to the controller.

The cheap wait gate reads only committed task/completion JSON and checks file
presence. Full checkpoint/native verification happens once through qstudy_branch
after the gate. No controller can run before the final validated import manifest.
Both experiment configs must already exist; this launcher never writes them.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from . import qstudy_branch as branch
from .graph_o2o_archive import sha

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SOURCES = tuple(dict.fromkeys((*branch.IMPORT_SOURCES,
    "fql/qstudy_branch_launch.py", "fql/qstudy_branch_launch.slurm",
    "fql/qstudy_actor.slurm", "fql/qstudy_backend_batch.slurm")))


def source_presence(source, cfg):
    """Return missing declared round-zero files; do not fit/load/verify tensors."""
    source = Path(source)
    missing = []
    roles = {"collect":cfg["online_samples"], "evaluation":cfg["evaluation_samples"]}
    for seed in cfg["seeds"]:
        folder = source/f"seed_{seed}/round_0"
        for name in ("checkpoint.pt", "binding.json", "completed.json"):
            if not (folder/name).is_file():
                missing.append(str(folder/name))
        if (folder/"completed.json").is_file():
            if branch.read(folder/"completed.json").get("status") != "completed":
                raise ValueError("Source training completion marker is not completed")
        tasks, marker = folder/"requests/tasks.jsonl", folder/"requests/completed.json"
        absent = [p for p in (tasks, marker) if not p.is_file()]
        if absent:
            missing.extend(map(str, absent))
            continue
        completion = branch.read(marker)
        rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
        if completion.get("status") != "completed" or completion.get("count") != sum(roles.values()):
            raise ValueError("Source export completion budget differs")
        if len(rows) != sum(roles.values()):
            raise ValueError("Source task budget differs")
        for role, count in roles.items():
            selected = [r for r in rows if r.get("role") == role]
            if len(selected) != count or {r.get("index") for r in selected} != set(range(count)):
                raise ValueError("Source task role/index plan differs")
        for row in rows:
            sample = Path(row["sample_dir"]).resolve()
            if (row.get("seed") != seed or row.get("round") != 0 or row.get("arm") != cfg["arm"]
                    or not sample.is_relative_to((folder/"requests").resolve())):
                raise ValueError("Source wait task identity differs")
            if not (sample/"outcome.json").is_file():
                missing.append(str(sample/"outcome.json"))
    return missing


def wait_source(source, target, cfg, source_config_sha, target_config_sha, *, wait=False,
                poll_seconds=30., max_wait_seconds=None, sleep=time.sleep, monotonic=time.monotonic):
    if not 0 < poll_seconds <= 60 or (max_wait_seconds is not None and max_wait_seconds <= 0):
        raise ValueError("Polling must be in (0,60] seconds with a positive optional timeout")
    started, last = monotonic(), None
    while True:
        if sha(Path(source)/"config.json") != source_config_sha or sha(Path(target)/"config.json") != target_config_sha:
            raise ValueError("Immutable source/target configuration changed while waiting")
        missing = source_presence(source, cfg)
        if not missing:
            return
        if not wait:
            raise RuntimeError("Source round0 prefix incomplete: "+str(len(missing))+" missing files")
        elapsed = monotonic()-started
        if max_wait_seconds is not None and elapsed >= max_wait_seconds:
            raise TimeoutError("Source prefix did not complete within declared wait limit")
        current = tuple(missing)
        if current != last:
            print(json.dumps(dict(status="waiting_source_round0", missing_files=len(missing),
                source=str(source), elapsed_seconds=elapsed)), flush=True)
            last = current
        delay = poll_seconds if max_wait_seconds is None else min(poll_seconds, max_wait_seconds-elapsed)
        sleep(delay)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or branch.read(path) != value:
            raise ValueError("Immutable launch journal changed: "+str(path))
    else:
        branch.atomic_json(path, value)


def launch(args, *, sleep=time.sleep, monotonic=time.monotonic, execv=os.execv):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Prefix wait/import/controller must run through Slurm")
    source, target, output = (Path(p).resolve() for p in (args.source, args.target, args.controller_output))
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and target experiment trees must be disjoint")
    if (output.parent != target.parent or output in (source, target) or output.name in {source.name, target.name}
            or output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError("Controller output must be a separate sibling of the target experiment")
    source_cfg, target_cfg = branch.read(source/"config.json"), branch.read(target/"config.json")
    seeds = branch.configurations(source_cfg, target_cfg)
    if target_cfg.get("online_rounds", 2) != 2:
        raise ValueError("This declared flow-update control uses exactly two online rounds")
    source_sha, target_sha = sha(source/"config.json"), sha(target/"config.json")
    declaration = dict(schema="qstudy_online_flow_launch_v1", source_experiment=str(source),
        target_experiment=str(target), controller_output=str(output),
        source_config_sha256=source_sha, target_config_sha256=target_sha, seeds=seeds,
        rounds=2, gpu_concurrency=2, backend_concurrency=2,
        source_sha256={name:sha(ROOT/name) for name in LAUNCH_SOURCES},
        new_prefix_training_updates=0, new_prefix_physical_calls=0,
        interpretation="Round0 is shared provenance, not independent measurements; round1+ may diverge")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output/".branch_launch.lock"
    if lock_path.is_symlink():
        raise ValueError("Launch lock must not be a symlink")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        branch.recover_journal_temporaries(output, ("branch_launch_plan.json", "branch_controller_handoff.json"))
        plan_path = output/"branch_launch_plan.json"
        if not plan_path.exists() and {p.name for p in output.iterdir()}-{".branch_launch.lock"}:
            raise ValueError("Controller destination contains foreign artifacts")
        immutable_json(plan_path, declaration)
        wait_source(source, target, source_cfg, source_sha, target_sha, wait=args.wait_source,
            poll_seconds=args.poll_seconds, max_wait_seconds=args.max_wait_seconds,
            sleep=sleep, monotonic=monotonic)
        # Recheck code after a potentially long wait, before the only full import.
        for name, digest in declaration["source_sha256"].items():
            if sha(ROOT/name) != digest:
                raise ValueError("Frozen launch source changed while waiting: "+name)
        result = branch.import_prefix(source, target)
        final = target/"branch_import.json"
        if (not final.is_file() or branch.read(final) != result or result.get("status") != "completed"
                or result.get("new_train_updates") != 0 or result.get("unique_physical_calls") != 0
                or result.get("seeds") != seeds or result.get("all_source_and_imported_native_results_validated") is not True):
            raise ValueError("No valid final branch import gate; controller launch refused")
        if sha(source/"config.json") != source_sha or sha(target/"config.json") != target_sha:
            raise ValueError("Configuration changed before controller handoff")
        command = [sys.executable, "-u", "-m", "fql.qstudy_controller",
            "--experiments", str(target), "--output", str(output), "--rounds", "2",
            "--gpu-concurrency", "2", "--backend-concurrency", "2"]
        immutable_json(output/"branch_controller_handoff.json", dict(status="validated_import_handoff",
            plan_sha256=sha(plan_path), import_manifest=str(final), import_manifest_sha256=sha(final),
            controller_command=command, no_new_round0_training=True, no_new_round0_backend=True))
        print(json.dumps(dict(status="exec_controller_after_validated_import", target=str(target),
            import_manifest_sha256=sha(final), rounds=2, gpu_concurrency=2, backend_concurrency=2)), flush=True)
        # Keep this launch lock across exec; the unchanged controller also owns its
        # ordinary controller.lock. Neither lock controls any foreign experiment.
        os.set_inheritable(lock.fileno(), True)
        os.chdir(ROOT)
        return execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--controller-output", type=Path, required=True)
    parser.add_argument("--wait-source", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--max-wait-seconds", type=float)
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
