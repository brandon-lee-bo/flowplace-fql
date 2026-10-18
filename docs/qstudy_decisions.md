# Q study: evidence and decisions

Status: development remains in progress; no stable PPA advantage is established.
This page summarizes verified reports, not a new analysis or candidate selection.
The whole-layout action is a terminal contextual-bandit adaptation of FQL;
there is no Bellman bootstrap. See the [literature](qstudy_literature.md),
[plan](qstudy_plan.md) and [full progress record](qstudy_progress.md).

## Hypotheses and decisions

| Hypothesis | Completed evidence | Decision / remaining gate |
| --- | --- | --- |
| Better regression, capacity or physical features provide useful Q | The 42-task screen, pin-feature and direct-scalar controls completed. WL can outperform a macro-HPWL prior, but timing/local ordering remains weak; fitting TRAIN well does not ensure held-family prediction. | Retain the simple anchor; no capacity/feature grid. Judge actual actor outcomes separately. |
| Executed coordinates repair timing prediction | Six paired tasks completed. bp_fe WNS improves modestly, but TNS rank falls .118→−.009; bp_be TNS MAE worsens112.98→135.89ns. | Do not promote this preprocessing change. It does not isolate legalization or backend snapping as the cause. |
| Ensemble disagreement supplies useful conservatism | Saved beta0/.5/1 scores give mixed ranking/selection results; no nonzero beta consistently improves the measured regret. | Keep beta0. This heuristic is neither CQL nor a calibrated confidence bound. |
| TRAIN-time conservatism fixes optimistic Q | COM-inspired12 fits versus12 unchanged controls: WL CV rho .561→.502, family accuracy .547→.503 and top regret .0144→.0193, despite slightly lower MAE. | Close this fixed regularizer test; no COM actor or beta search. Historical development16 does not overturn grouped CV. |
| Labels from current behavior flow improve Q | All80 native labels and36 source tasks completed. On the same current-flow16, legacy O64/T64 WL rho is .500/−.346; T64 worsens every critic seed. Base-only sensitivity does not rescue ranking. | Source substitution alone is negative. The declared O64/T64 actual-policy test remains pending; it tests physical consequences, not promotion. [Source results](qstudy_source_results.md). |
| Q gradients improve nearby layouts | Both designs completed52/52 native probes. FE directions mostly fail; BE signs, ties and mean gains are mixed. Four bases and one critic seed per design. | Do not strengthen Q from these probes. They do not validate source64/bounded critics or identify a legalizer mechanism. |
| Timing/congestion objectives improve policy tradeoffs | Nine bounded-objective tasks/36 fits completed. Timing/congestion CV MAE exceeds their own constant baselines; local ordering is weak. Bounded-WL24/24 native outcomes are complete below; timing/congestion arms remain unmeasured. | Complete the declared matrix. WL improves versus the existing BC reference with timing/area tradeoffs; this is not a controlled bounded-formula effect. No alpha/grid expansion or automatic online extension. [Protocol](qstudy_bounded_objective_protocol.md). |
| Q-based actor learning beats imitation and generate-and-rank | Offline GRT means and identical-initial-Q selection are complete below. Fixed-first4 offline DRT also completed: FQL has worse mean WL, timing, power and area than BC, despite better WL in two of three seeds. | No stable offline advantage. Ranking ability is not evidence of correct actor gradients. Complete the declared main online/final-route matrices before selection. |
| One-step distillation is limited by training duration | Saved same-noise raw RMS to flow is .1000 for BC and .1009 for FQL; both have worse mean WL than flow in all three seeds. This is clipped export distance, not training MSE. | Use the already declared BC1000→1600→2200 continuation first: teacher stays fixed and Q weight is zero. No additional duration GPU/native jobs are launched. |

## Completed native development comparison

All values below are GRT-stage measurements on bp_fe, with the same actual
eight generation-noise arrays and three independent training seeds. Policy
means average all eight outputs within each seed; WL uncertainty is sample SD
across those three means. Other entries are means; full SDs and paired deltas
are in [progress](qstudy_progress.md#matched-offline-results-after-missing-only-recovery).

| Unselected policy | WL (m), mean ± seed SD | WNS (ns) | TNS (ns) | H overflow | V overflow | Power (W) | Std-cell area (um2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current behavior flow | 2.279086 ± .013390 | −.406557 | −206.695 | 22.125 | 7.125 | .169797 | 60997.7 |
| Matched BC | 2.314028 ± .012141 | −.403605 | −217.399 | 7.667 | 1.375 | .170010 | 61200.2 |
| Legacy offline FQL | 2.307890 ± .003804 | −.407202 | −217.652 | 2.625 | .667 | .170140 | 61198.8 |
| Bounded-WL offline actor | 2.304274 ± .014205 | −.418562 | −217.922 | 2.000 | .792 | .170021 | 61525.2 |

Bounded-WL report1797061 verifies24/24 successes: WL−BC is −9753.75 ±2064.471um
(paired-seed SD), lower in all three seeds. WNS worsens, TNS slightly worsens,
and mean WL remains above current flow2.279086m. BC is an existing measured
reference without a bounded zero-Q update-equivalence certificate; the gain
cannot be attributed solely to the bounded formula. The overall report is
partial because timing/congestion are unmeasured, not because WL is incomplete.

For selection, the same frozen initial per-seed Q scores every generator's
eight raw proposals. Each seed contributes **one selected output**, so its SD
answers a different question from policy-mean SD. This uses existing measured
pools, with no additional layout, training or backend calls.

| Fixed8 maximum-Q selector | WL (m), mean ± seed SD | WNS (ns) | TNS (ns) | H overflow | V overflow | Power (W) | Std-cell area (um2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current flow + Q | 2.221280 ± .027090 | −.392523 | −172.297 | .667 | 0 | .169844 | 60753.7 |
| BC + Q | 2.275971 ± .015563 | −.335976 | −182.738 | 31.333 | 2.667 | .169928 | 60456.9 |
| FQL + Q | 2.240982 ± .045602 | −.370599 | −149.911 | 10.000 | 1.000 | .170013 | 60420.5 |

These are selected-versus-selected comparisons, not best-sample equivalents
of unselected policy means. The measured-label WL oracle is a separate
eight-backend-label reference, not a free deployment rule. Candidate count and
native calls match; inference costs differ: current flow10 steps, actor1 step.
The released official20-step flow has one checkpoint/eight noise outputs,
not three independent training replicates; its separate reference is in
[progress](qstudy_progress.md#complete-direct-flow-references-and-fixed-conservative-critic-contrast).

Evidence: baseline1796858 (`ee8d29d7…`), matched round0 report1797018
(`dc19bc3b…`), fixed-Q report1797084
(`fixed_initial_q_selection_v1/20261018T111526_522211_job1797084`). All72
flow/BC/FQL native outcomes succeeded with zero integrity errors. GRT success
does not establish final DRC0 or timing closure. These inspected noises,
historical validation sets and local probes are development evidence.

## Fixed-first4 offline detailed routing

The48-task plan has24 completed round0 layouts and24 round2 sources still missing.
Each arm averages fixed first4 noises within each of three training seeds first; all12 layouts are DRC0.

| Round0 policy | DRT WL (m) | WNS (ns) | TNS (ns) | Power (W) | Std-cell area (um2) | DRC0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched BC | 1.979399833 | −.579834833 | −360.588083 | .173064250 | 61050.3083 | 12/12 |
| Legacy offline FQL | 1.985800583 | −.605670667 | −372.022917 | .173642083 | 61327.8417 | 12/12 |

FQL−BC WL is +6400.75um (paired-seed SD26202.405um): two of three seeds improve, but the aggregate worsens. DRC0 is not timing closure.
GRT all8 and DRT first4 have different denominators and cannot isolate a routing-stage effect. Online round2 and fresh confirmation remain pending.
Evidence: report1797327, snapshot `development_final_route_v1/development_report/snapshots/20261018T114051_961307_job1797327`.

## Stopping and confirmation gate

Finish the already-declared main, source and bounded physical matrices and
their applicable final-route reports. Close negative predictive branches;
do not open new parameter sweeps or online variants. Main BC already restores
the parent actor and optimizer for two further600-step distillation rounds.
Compare its teacher distance and native outcomes before adding another duration
experiment. Each round resets its named random generators, so this is practical
continued training, not an uninterrupted2000-step equivalence test. FQL also
updates Q in these rounds and cannot isolate duration alone. A lower loss
without physical benefit does not justify more duration variants.

Then select **one configuration and original round, retaining all three
seeds**, with native policy means, failures and explicit secondary-metric
tolerances. If no new mechanism has a credible matched physical tradeoff,
the existing frozen-flow legacy anchor versus matched BC is a valid negative
confirmation comparison. Lock rules and sources before the new noise stream;
use the [fresh protocol](qstudy_fresh_confirmation_protocol.md). No fresh
candidate has been selected by this page.

Final delivery should consolidate the mechanism decisions, all-seven-metric
tables, offline/online curves, failures and unique physical-call/training/
inference budgets. Keep selected outputs separate from policy means and
shared-prefix reuse separate from new measurements. Fresh bp_fe confirmation
does not establish cross-design robustness; existing bp_be results remain a
limited second-design diagnostic.
