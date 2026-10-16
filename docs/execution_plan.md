# Experiment plan — 2026-10-16

1. Pin official sources and assets; provision isolated dependencies.
2. Validate author-provided DEF through released OpenROAD flow, starting bp_fe. Record tool versions, exact cell libraries and report stage.
3. Generate pretrained layouts on the same circuit, independently check bounds and overlap, export DEF and evaluate. Extend to six Table 2 circuits when feasible.
4. Build FQL-style terminal contextual-bandit adaptation: graph context, whole-layout continuous action, identical hard legalization, explicit measured reward. A numerical flow integration step is NOT an environment transition.
5. Compare teacher FlowPlace, one-step distillation, FQL at matched noise/sample/backend budgets; use fixed heldout evaluation seeds and report distribution, not just best. Include direct proxy optimization where feasible to separate critic learning from objective access.
6. Separate macro-only proxy experiments from routed PPA. Preserve failed runs and identify implementation/tool/reward differences.

Stop only on tested end state; no assumed success from model loss. Pending full experimental requirements tracked here and in runtime manifests.
