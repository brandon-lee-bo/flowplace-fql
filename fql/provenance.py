"""Capture source-file identities at process startup, never retrospectively."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_source_provenance(archive_dir=None, extra_paths=None, capture_event="startup_before_training"):
    directory = Path(__file__).resolve().parent
    files = ["train.py", "core.py", "analytic.py", "latency.py", "critic_gate.py",
             "predict.py", "coordinate_control.py", "relabel_backend.py", "provenance.py",
             "prepare_grt_candidate_tasks.py", "aggregate_backend_candidates.py", "seal_grt_selection.py",
             "physical_failure_policy.py", "grt_actor_physical_policy_v1.json", "infrastructure_retry.py"]
    paths = {name: directory / name for name in files}
    paths.update(extra_paths or {})
    archive = Path(archive_dir).resolve() if archive_dir is not None else None
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=False)
    identities = {}
    for name, path in paths.items():
        path = Path(path).resolve()
        contents = path.read_bytes()
        identity = {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}
        if archive is not None:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Snapshot source name must remain inside its archive")
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            identity.update(archived_path=str(target), archived_sha256=file_sha256(target))
        identities[name] = identity
    snapshot = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "capture_event": capture_event, "retrospective": False,
                "archive_dir": None if archive is None else str(archive), "files": identities}
    try:
        top = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        head = subprocess.run(["git", "-C", top, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", top, "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, timeout=5, check=True).stdout.splitlines()
        snapshot["git"] = {"available": True, "root": top, "head": head, "dirty": bool(status),
                           "status_porcelain": status[:200], "status_truncated": len(status) > 200}
    except (OSError, subprocess.SubprocessError) as exc:
        snapshot["git"] = {"available": False, "error_type": type(exc).__name__, "head": None, "dirty": None}
    return snapshot


def checkpoint_provenance(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}
