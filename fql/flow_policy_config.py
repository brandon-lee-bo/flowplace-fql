"""Behavior-teacher settings, kept independent of torch for config preparation."""

RELEASED = "released_flowplace"
LEGACY = "legacy_euler"
STEPS = 20
PUBLIC_SHA = "0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359"


def uses_released_teacher(cfg):
    # Untagged configs are immutable historical experiments, not new defaults.
    kind = cfg.get("flow_policy", LEGACY)
    if kind not in (RELEASED, LEGACY):
        raise ValueError("Unknown behavior flow: " + str(kind))
    return kind == RELEASED


def validate_teacher_config(cfg):
    if not uses_released_teacher(cfg):
        return
    if (cfg.get("flow_steps") != STEPS or cfg.get("flow_warmup_steps", 0) != 0
            or cfg.get("online_flow_update", False)
            or cfg.get("flow_checkpoint_sha256") != PUBLIC_SHA):
        raise ValueError("Released FlowPlace requires its pinned public weights, "
                         "20 constrained steps, zero flow warmup and no flow updates")
