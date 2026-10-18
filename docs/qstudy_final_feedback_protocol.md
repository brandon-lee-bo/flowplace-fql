# Matched TRAIN64 final-feedback reward experiment

2026-10-19. Fixed protocol reviewed by root before source binding and any
preparation/training. No new TRAIN-DRT metric values or policy
results were inspected to choose these weights. The six objectives are fixed
contrasts, not a sweep, a best-reward search or a COptiDICE reproduction.

## Question and fixed source

Does learning from actual final-route feedback improve physical policy outputs
over a matched early-stage target, and do timing, power or one fixed expert
mixture add value? Existing experiments on384 GRT labels cannot answer this
matched64-label question. A negative result is valid; better predicted Q or
lower training loss does not establish a better placement.

Use only the64 original-TRAIN requests in
`/scratch/bohanli/flowplace_graph/qstudy_train_drt64_v1/plan.json`, SHA256
`18b109579d8076ee7f65afb84481b7c270a7edc101fdbb67d28216802c7c101f`.
The frozen selector chose32 families before DRT: eight SHA-selected families
in each of four GRT-family-mean-WL quartiles, both variants, with four fixed
family folds. Each CV fit has24 training families/48 rows and eight target
families/16 rows; refit uses32 families/64 rows. Keep the exact frozen folds.
The subset is GRT-stratified, not representative random sampling of all designs.

Use the unchanged eight-thread native continuation, same container and physical
revision, repair cap20, and preserved source macro geometry. The collector is
`fql/qstudy_train_final_route.py`: every accepted final label must pass its
`inspect_result`/`audit_final_outcome` chain and remain linked to the original
raw action, executed action, source GRT and final outcome. The existing48
development DRT observations and historical16 validation layouts are excluded
from this training/DRT-CV dataset, scale fitting and weight choice. No new DRT
validation labels are inferred for historical16. Final confirmation stays separate.

## Six target definitions

For a feasible completed layout, let `G,L,T,P,A` be native GRT WL in um, final
DRT WL in um, `max(-final_TNS_ns,0)`, final total power in W, and final standard
cell area in um2. Signed final WNS is retained separately. For each fit derive
`S_G,S_L,S_T,S_P,S_A` from that fit's feasible TRAIN rows only; define
`g=G/S_G, l=L/S_L, t=T/S_T, p=P/S_P, a=A/S_A`.

The four fixed expert costs and exact coordinate units come from
`analysis/qstudy_rule_audit.py`: nearest-edge clearance, cached cluster bounding
box/core area, cached dataflow-weighted center distance, and narrow facing-channel
risk. Compute them on the same executed layout used by the native source; bind
the raw-to-executed correspondence. If `E_k` denotes each nonnegative cost,
`e=(1/4)*sum_k(E_k/S_Ek)`, with every `S_Ek` fitted on the same fit-TRAIN rows.
No I/O term, new action mask or presumed STA-criticality weight is introduced.

| Fixed arm | Nonnegative scalar cost C | Primary contrast |
| --- | --- | --- |
| `grt_wl` | `g` | Matched64 early-stage target |
| `drt_wl` | `l` | Final WL versus early-stage WL |
| `drt_timing` | `.75*l + .25*t` | Aggregate final timing preference |
| `drt_power` | `.75*l + .25*p` | Final power preference |
| `drt_multi_ppa` | `.50*l + .25*t + .15*p + .10*a` | One fixed final PPA tradeoff |
| `drt_expert` | `.80*l + .20*e` | Fixed four-rule prior beyond final WL |

For every feasible target use `B=2*C/(1+C)`, reward `u=-B`, and actor
`Q=-mean(predicted B)` in native B units. No additional mean-B normalization,
actor-output clipping of predicted B, or rational transform of unrestricted
predicted linear costs is permitted. Each convex recipe has mean C=1 on its
feasible TRAIN reference; this does not equate its action gradients or mean B.
These operational weights are hypotheses, not literature-recommended optima.

TNS aggregates negative endpoint slack; WNS identifies the worst violation.
Both derive from related timing data, so WNS is not added again as a second
weighted timing penalty. Report signed WNS and its deterioration guard alongside
TNS. A TNS gain can still hide a worse worst path; neither is timing closure.

## Feasibility, missingness and variation gates

- All six arms use the same final feasibility convention. A completed final
  route with DRC>0, a confirmed geometry failure, or a confirmed physical failure
  receives B=4/u=-4 in every arm. Preserve native status and all measured PPA:
  a completed DRC-infeasible route is not relabeled as a tool execution failure.
  Boundary/overlap/fixed-node checks remain actual hard execution checks.
- Thus `grt_wl` means **GRT WL for successful layouts under the same final
  feasibility predicate**. It is not a pure original GRT-only reward: the shared
  feasibility information deliberately comes from DRT for a controlled comparison.
- Infrastructure interruption, unknown/tool failure, invalid provenance or a
  missing label is NA, not B4, zero PPA or a synthesized DRT label. All64 must be
  resolved and audited before training. Do not replace inconvenient families or
  train different subsets by objective. Confirmed failures need no fabricated PPA.
- Cost scales use only feasible, finite, observed fit-TRAIN values. Before
  training, check every required base/rule column in all four folds and refit:
  at least two observations, mean>1e-8 and span>1e-12, matching the existing rule
  screen's explicit checks. All feasible rows must have all required native/rule
  fields. Constant/unavailable required terms make the **whole six-arm protocol
  inadmissible**; do not silently drop a term, change weights, insert epsilon
  labels or run a selected subset. Any revision requires a new prospective version.

## Matched learning and physical budget

Use the same raw-coordinate plus17 pre-backend features, two independent critic
members with two hidden layers of width128, scalar Huber supervision, batch64,
AdamW lr3e-4/weight decay1e-4 and2000 updates. Fit feature/target normalizers on
fit TRAIN only. Seeds0/1/2, fold memberships, valid rows, initialization/minibatch
streams and fixed last-update checkpoints are matched across all six arms.
There are18 tasks, each four family-CV fits plus full refit:90 fits total.
No Bellman bootstrap is added to this whole-layout terminal bandit.

Each offline actor uses the same corresponding original common seed state,
frozen Euler10 behavior teacher, fixed256 behavior targets and1000 updates with
`10*coordinate_distillation-Q`. Keep the original flow/actor architecture,
sampler, final legalizer, DEF conversion, optimizer recipe and matched streams.
Critic64-label provenance is separate from the unchanged behavior-target and
physical-context bindings. There are18 actor runs; no online branch or alpha,
head-count or weight sweep is part of this contrast. Record actual distillation/Q
action and parameter gradient pressure: equal alpha/B range is not equal pressure.

All six policies use the same eight existing saved development noise arrays
per seed:144 GRT requests. Also predeclare final routing of indices0..3 for
**every arm and all three seeds**, with the same cap20:72 DRT requests. These
are new policy evaluations, separate from the64 TRAIN DRT labels. Do not
route only favorable outputs or select the best sample instead of policy means.
This physical budget is authorized for implementation after source review and
the complete64-label admission gate; no submission occurs from this document.

Report all7 native GRT metrics and all6 final DRT metrics, full requested
failure/NA denominators, successful-conditional PPA explicitly, each seed mean,
three-seed sample SD, and paired same-noise deltas. GRT and final DRT have
different noise denominators and are not interchangeable. There is no invented
final congestion metric. Preserve linear C, bounded B and all native units.
Score each arm under every declared preference without treating different
objectives' B-MAE as one common physical quality. CV remains fold-local then
seed-aggregated; never pool B across differing fold scales. Retain constant
TRAIN-mean baselines, family-local pairs/top-regret and out-of-range predictions.

The main stage-fidelity comparison is `drt_wl` versus matched `grt_wl`; preference
comparisons use `drt_wl` as control. Timing/power/multi-PPA outcomes can be useful
tradeoffs even without a WL win, but must retain actual deterioration elsewhere,
DRC/failures and all three seeds. The existing practical WNS/TNS/power/area
tolerances can be reported as guardrails, not significance tests or proof of
constraint satisfaction. This extension does not automatically promote a new
fresh-confirmation candidate or alter an already sealed confirmation rule.

## Power quality and literature boundary

Power is an **OpenSTA estimate under fixed default activity-model conditions**,
not measured silicon power or a measured workload's energy. The inspected
`report_metrics.tcl` calls `report_power`/`report_power_metric`; final extraction
and SPEF reading precede reporting. No VCD/SAIF read or explicit activity setting
was found in the inspected design config/load/final-report path. Audit the
actual bound image/scripts, libraries (including fakeram power models), SDC,
clock frequency, corners, voltage and activity provenance for consistency before
using this term. Check finite positive totals and raw internal/switching/leakage
components where available; do not infer annotation coverage merely from a total.

[OpenSTA's documented inputs](https://openroad.readthedocs.io/en/latest/main/src/sta/README.html)
include VCD/SAIF activities. Its current upstream
[power implementation](https://github.com/The-OpenROAD-Project/OpenSTA/blob/master/power/Power.cc)
also supplies default input activity when unannotated. The current upstream
numeric default is not proof of the exact value in our frozen binary. Report
that value as unverified if the bound version is not established; do not add
activity annotation to only one arm or claim workload-power generalization.

[FQL](https://proceedings.mlr.press/v267/park25f.html) motivates the separate
behavior flow/one-step Q-guided actor, not these physical reward weights.
[COptiDICE Eq.(2)-(5)](https://arxiv.org/html/2204.08957) separates task reward
from budgeted expected costs and explicitly addresses offline estimation error.
Our six fixed scalar targets do not implement its density-ratio optimization,
cost-conservative upper bounds or constraint guarantee; even adding a hinge
would not make them COptiDICE. Here WNS/feasibility guards are checked on measured
outputs rather than assumed from predicted Q.
[EXPlace's official environment](https://github.com/lamda-bbo/EXPlace/blob/d5d534a5fbdbc1ab4aa0bd450761277b437d0967/src/place_env/place_env.py)
and [RTL-MP](https://vlsicad.ucsd.edu/Publications/Conferences/389/c389.pdf)
motivate structural engineering proxies, not their universal correctness.
The four complete-layout rules remain our explicit adaptations; only matched
native final-feedback comparisons can establish their value here.
