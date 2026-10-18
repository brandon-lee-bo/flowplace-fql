# Reward extension: backend outcomes and placement rules

User steering 2026-10-18 UTC: actually explore several rewards beyond GRT
wirelength, including placement expertise and rules learned from evidence.
This is an active work plan, not a frozen experiment matrix or a claim that
the proposed rules improve PPA. The original FQL and current fixed weighted-WL
flow control remain references.

## What will be compared

1. **Final physical feedback.** Collect new TRAIN-only detailed-route labels
   for a predeclared subset of the existing measured layouts. Keep final WL,
   WNS, TNS, power, standard-cell area and DRC separate. Existing48 development
   detailed-route results remain evaluation data and cannot be relabeled TRAIN.
   Fit each normalization and any critic-selection statistic on TRAIN families.
   Compare GRT-based and final-route-based learning with the same labeled layout
   subset; otherwise label fidelity and dataset size are confounded.
2. **Expert rules as reward terms.** Audit perimeter preference, connected/grouped
   macro proximity, dataflow distance and I/O keepout against the official EXPlace
   definitions and actual cached metadata. Also audit a narrowly defined routing
   channel-risk proxy if geometry supports it. Test useful, nonconstant rules
   individually before combining them. Compare a rule-only control and backend
   feedback plus that rule, while preserving policy architecture and action space.
3. **Outcome constraints.** Keep WL as the objective and timing/congestion as
   separate costs with explicit budgets, inspired by constrained offline RL.
   This is distinct from one fixed weighted sum. Predicted constraint satisfaction
   is not proof of actual feasibility; every candidate still needs native checks.
4. **Measured preferences.** Consider same-family layout preferences derived from
   native outcomes or Pareto dominance, drawing on offline preference learning.
   Any tolerances and tradeoffs must be explicit. Reusing a measured outcome in
   several pairs does not create additional independent labels, and expert-rule
   preferences do not validate the same rule's downstream benefit.

The present terminal whole-layout problem has no Bellman bootstrap. Adding an
expert cost to its reward generally changes the objective; it is not an
automatic policy-invariant shaping guarantee. Expert experience supplies
testable hypotheses, not a substitute for final physical measurements.

## Immediate work and evidence gates

- Complete the already frozen uniform versus quality-ranked flow control.
  It tests measured-quality weighting and remains WL-only; it does not fulfill
  the entire reward extension by itself.
- Audit rule availability, units, numerical variation and family-level
  associations with the existing TRAIN seven-metric outcomes. Keep both raw
  proposal and executed-layout rule values; hard legalization may change them.
  Macro graph `is_ports` alone is insufficient to infer whether I/O metadata
  exists: inspect the cache's real port and blocking-rectangle fields.
- Prepare a new TRAIN-only final-route selection and restoration plan, with
  existing physical settings, eight threads, macro-preservation checks and
  unchanged route-repair cap. Record labels, attempts, storage and failures.
- Freeze the small initial reward matrix after those audits. Change reward
  separately from critic input/network, actor loss and action masks. A rule
  trial must not silently shrink legal actions as EXPlace's soft mask can do.
- Evaluate all physical metrics and failure rates, not just the designed reward.
  Compare multiple independent training seeds and keep matched controls.

Constant or unavailable rule/physical labels are reported as such, not scaled
by tiny denominators to manufacture a training signal. Physical failures,
infrastructure failures and missing evaluations remain separate. New evidence
may justify extending or closing an individual hypothesis; no blind Cartesian
hyperparameter sweep is authorized by this plan.

Related sources and exact definitions will be recorded in
`qstudy_rule_reward_literature.md`; the prior reward-method review is in
`qstudy_literature.md`. No new reward arm is called successful before its
native backend outcomes are complete. Final fresh-confirmation scope will be
locked only after this user-directed development extension is closed.
