"""Explicit pre-outcome policy for the bounded real-GRT actor experiment."""
from datetime import datetime
import json
from pathlib import Path

try:
    from .relabel_backend import STAGE, resolve_from, sha256_file
except ImportError:
    from relabel_backend import STAGE, resolve_from, sha256_file


POLICY_RULES = {
    "policy_id": "grt_actor_all_cases_success_v1",
    "schema_version": 1,
    "training_reward_stage": STAGE,
    "training_seeds": [0, 1, 2],
    "alphas": [1, 10, 100],
    "validation_indices": [192, 193, 194, 195],
    "heldout_indices": [244, 245, 246, 247],
    "eligible_rule": "all_12_validation_cases_successful",
    "ranking_rule": "minimum_mean_rwl_across_seed_means_lexical_tie_break",
    "incomplete_configuration_mean": None,
    "incomplete_bc_comparison": None,
    "all_ineligible_action": "stop_without_heldout",
    "heldout_action": "report_locked_choice_without_reselection",
    "unknown_infrastructure_parser_action": "hard_error",
    "permitted_physical_failure_classes": ["cts_hold_repair_max_buffer_count"],
}


def bind_policy(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if any(document.get(key) != value for key, value in POLICY_RULES.items()):
        raise ValueError("Unsupported preregistered physical-failure policy or budget")
    frozen = datetime.fromisoformat(document["frozen_at_utc"])
    if frozen.tzinfo is None:
        raise ValueError("Policy freeze time must include a timezone")
    return {"path": str(path), "sha256": sha256_file(path), "document": document}


def validate_bound_policy(plan, base):
    binding = plan.get("physical_failure_policy")
    if not isinstance(binding, dict):
        raise ValueError("Preregistered mode requires a frozen physical_failure_policy binding")
    actual = bind_policy(resolve_from(binding["path"], base))
    if actual != binding:
        raise ValueError("Frozen physical-failure policy document/path/SHA changed")
    policy = actual["document"]
    if plan.get("phase") not in ("validation", "heldout") or plan.get("training_reward_stage") != STAGE:
        raise ValueError("Preregistered policy applies only to real-GRT actor validation/heldout")
    for field in ["training_seeds", "validation_indices", "heldout_indices"]:
        if plan.get(field) != policy[field]:
            raise ValueError(f"Candidate plan violates frozen policy: {field}")
    if plan.get("predeclared_alphas") != policy["alphas"]:
        raise ValueError("Candidate plan violates frozen alpha sweep")
    if plan.get("indices") != policy[plan["phase"] + "_indices"]:
        raise ValueError("Candidate indices differ from the frozen policy phase")
    if datetime.fromisoformat(plan["freeze_utc"]) < datetime.fromisoformat(policy["frozen_at_utc"]):
        raise ValueError("Policy was not frozen before the candidate plan")
    source = plan.get("source_bindings", {})
    if source.get("physical_failure_policy") != {key: actual[key] for key in ["path", "sha256"]}:
        raise ValueError("Policy must also be included in frozen source bindings")
    for name in ["dataset", "dataset_manifest", "training_run", "training_summary", "gate_report"]:
        item = source.get(name, {})
        if not item.get("path") or sha256_file(resolve_from(item["path"], base)) != item.get("sha256"):
            raise ValueError(f"Missing/changed real-GRT actor source: {name}")
    manifest = json.loads(resolve_from(source["dataset_manifest"]["path"], base).read_text())
    run = json.loads(resolve_from(source["training_run"]["path"], base).read_text())
    if (manifest.get("reward_stage") != STAGE or manifest.get("dataset_sha256") != plan.get("dataset_sha256")
            or source["dataset"]["sha256"] != plan.get("dataset_sha256")
            or run.get("data", {}).get("sha256") != plan.get("dataset_sha256")
            or run.get("args", {}).get("seal_heldout") is not True):
        raise ValueError("Policy requires bound GRT training data and sealed actor training")
    candidates = [(item["method"], item["index"]) for item in plan["candidates"]]
    if len(candidates) != len(set(candidates)):
        raise ValueError("Duplicate frozen policy candidate")
    fixed = [key for key in candidates if key[0] == "best_observed_train"]
    if len(fixed) != 1:
        raise ValueError("Policy requires one measured best-TRAIN baseline")
    actor_methods = set(plan.get("methods", []))
    bc = {f"seed_{seed}/distill_only" for seed in policy["training_seeds"]}
    sweep = {f"seed_{seed}/fql_alpha_{alpha:g}" for seed in policy["training_seeds"] for alpha in policy["alphas"]}
    if plan["phase"] == "validation":
        if actor_methods != bc | sweep:
            raise ValueError("Policy validation must include all three seeds and all three alphas plus BC")
    else:
        choices = [{f"seed_{seed}/fql_alpha_{alpha:g}" for seed in policy["training_seeds"]} for alpha in policy["alphas"]]
        if not any(actor_methods == bc | choice for choice in choices):
            raise ValueError("Policy heldout must contain BC and exactly one alpha across all seeds")
    expected = {(method, index) for method in actor_methods | {"teacher_relegalized"} for index in plan["indices"]} | set(fixed)
    if set(candidates) != expected:
        raise ValueError("Policy candidate budget is incomplete or contains extra cases")
    return actual
