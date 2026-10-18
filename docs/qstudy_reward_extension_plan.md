# Reward extension: backend outcomes and placement rules

User steering 2026-10-18 UTC: actually explore several rewards beyond GRT
wirelength, including placement expertise and rules learned from evidence.
This is an active work plan, not a frozen experiment matrix or a claim that
the proposed rules improve PPA. The original FQL and current fixed weighted-WL
flow control remain references.

## Audited execution status, 2026-10-19 HKT

The latest saved partial expert report is job1800252; its complete192-request
denominator is retained, not reduced to completed arms. Other unfinished
collections below remain queued under their existing job handles.

| Question | Actual experiment | Evidence at this snapshot |
| --- | --- | --- |
| Do timing and congestion rewards help? | Completed bounded WL / WL+timing / WL+timing+GRT H/V contrasts | Closed, three seeds. Congestion falls relative to timing-only, but WL worsens; no stable overall gain. |
| Can inexpensive expertise replace or supplement backend feedback? | Eight fixed objectives: GRT WL, macro HPWL, four separate rules, HPWL+four rules, GRT WL+four rules | All24 actors trained;92/192 native GRT outcomes audited. Only the two base controls are fully measured. |
| Does better macro HPWL mean better routing? | The two closed base controls,24 outputs each | Macro HPWL is lower, but GRT WL is about1.01% higher in all three seed means, with more congestion. This is a proxy/backend mismatch, not a conclusion about the six unfinished rule arms. |
| Does final-stage feedback change the answer? | Matched64 TRAIN DRT labels, then six fixed GRT/DRT-WL/timing/power/multi-PPA/expert objectives |29/64 labels complete. The whole64-label admission and power-quality gate must pass before fits/actors; the already submitted dependency chain is not yet a result. |
| Does return-weighted behavior learning avoid Q-gradient errors? | Uniform versus quality-weighted flow, same data and budget | GRT closed: weighted improves WL by about1.04% over uniform, but remains about2.30% worse than the common teacher. Final DRT9/24, unresolved. |

The rules are explicit, measured-layout hypotheses: keep central space by
preferring edges, keep cached groups compact, shorten static dataflow distance,
and reduce narrow facing channels. None is assumed to improve timing or routing
universally. The unavailable-provenance I/O term is excluded, and graph/dataflow
weights are not called STA criticality. No rule trial changes the legal action
space or imports EXPlace's soft action pruning.

Sources and mechanism boundaries are in
[the reward literature review](qstudy_rule_reward_literature.md). FQL supplies
the behavior-flow/one-step actor structure; CQL/Cal-QL/IQL/TD3+BC inform support,
critic and behavior constraints rather than prescribing a placement reward.
COptiDICE motivates separating task reward from cost budgets, but our scalar
mixtures do not reproduce its optimization or constraint guarantees.

## Next scientific online contrast

This is a prospective design for the next scientific stage, not a submitted
run, a selected new reward, or a claim that the integration smoke is an online
quality experiment. Close the eight-rule and matched-final-feedback programs
before choosing a new reward. Preserve negative results and use the declared
physical guards rather than a high predicted Q or a favorable single layout.

Use a genuine BC continuation, same-parent frozen-Q continuation and updated-Q
continuation. For an expert/final candidate, also retain a matched scalar
GRT-only online control; the historical legacy-WL matrix uses a different
reward scale and round RNG convention and is not that control. Keep the
teacher, original TRAIN normalizers, network, legalizer and physical flow fixed.
The intended scientific budget is two cycles of600 actor updates and eight own
measured collects per cycle/seed, with eight evaluation noises per stage,
three independent training seeds and complete cumulative replay. Final-feedback
objectives require actual DRT/QA for their collects, not substituted GRT labels.
Evaluation labels and noises remain excluded from replay. Freeze the exact
selected arms, source/checkpoint bindings and unique/reused call budget before
execution; no extra final-round collect is implied.

The sole proposed actor-loss intervention is alpha30 versus alpha10, with
Q weight1 and unchanged1:1 critic/actor updates. It tests whether stronger
behavior regularization reduces exploitation of imperfect Q directions. This
direction is motivated by the recorded growth in Q/distillation gradient
pressure and false-positive Q improvements; it is not evidence that alpha10
is too small. Both arms start from the same actual alpha10 offline checkpoint,
including Adam and named RNG state. They can share the exactly bound C0
collect; their first-cycle critic tensors should match and must be checked.
Their next C1 collects come from their own policies, so second-cycle critic
differences are part of the resulting feedback procedure.

This tests an online regularization intervention, not from-scratch alpha
optimization. Do not simultaneously sweep alpha3, reward scales, teacher
updates and update frequencies. Frozen-Q means no critic updates, not a2:1
or1:2 frequency ablation. Existing alpha3/30 gradient rescalings are analytic
diagnostics only. The one-update-per-cycle interface checks remain separate
from the proposed600-update scientific runs.

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
   individually before combining them. The frozen eight-arm experiment compares
   HPWL-only against HPWL plus individual/composite rules, and GRT-only against
   GRT plus the composite. A pure-rule objective was an earlier optional idea,
   not an implemented arm. Preserve policy architecture and action space.
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
