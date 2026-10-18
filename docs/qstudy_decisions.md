# Q study: evidence and decisions

Status: all declared development matrices are complete; evidence preservation
and fresh confirmation remain pending. No stable PPA advantage is established.
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
| Labels from current behavior flow improve Q | All80 native labels,36 source tasks and48 source-actor outcomes completed. On the same current-flow16, legacy O64/T64 WL rho is .500/−.346. T64 policy mean WL is .129% below O64 but .291% above BC; only one of three seeds improves in either comparison. | This64-label/1000-update source recipe has no stable Q-guided gain. Both actors fail the predeclared primary gate; no source promotion or stronger Q optimization follows. [Source results](qstudy_source_results.md). |
| Q gradients improve nearby layouts | Both designs completed52/52 native probes. FE directions mostly fail; BE signs, ties and mean gains are mixed. Four bases and one critic seed per design. | Do not strengthen Q from these probes. They do not validate source64/bounded critics or identify a legalizer mechanism. |
| Timing/congestion objectives improve policy tradeoffs | Nine bounded-objective tasks/36 fits and all96 native outcomes completed (72 new bounded,24 reused BC). Timing WL is1.157% below descriptive BC, all3 seeds lower, but TNS/H overflow exceed declared tolerances. Against bounded WL, its own measured timing utility worsens in all3 seeds. | Close the declared matrix without promotion: WL misses1%, congestion misses1%/all-seed consistency, timing fails secondary metrics. BC has no bounded zero-Q equivalence certificate. No alpha/grid expansion or automatic online extension. [Results and protocol](qstudy_bounded_objective_protocol.md). |
| Q-based actor learning beats imitation and generate-and-rank | All main GRT streams completed. Frozen FQL2 lowers WL0.804% versus BC2, two of three seeds; updated FQL2 lowers0.628%, two of three, with excessive V overflow. All48 fixed-first4 DRT requests are complete and DRC0; round2 FQL mean WL is0.114483% higher than BC, with improved secondary means. | Neither online branch passes the primary gate; no new promotion. Ranking ability does not establish useful actor gradients. Close provenance/cost/preservation before fresh selection. [Online results and curves](qstudy_online_results.md). |
| One-step distillation is limited by training duration | Saved BC1000→1600→2200 continuation is complete: clipped raw RMS to flow falls .100037→.092280→.084646 in all three seeds, but native mean WL is2314028→2322383→2312632um. Final WL improves over1000 updates in only one seed and remains worse than flow in all three. | More continuation reduces the measured approximation gap without stable PPA improvement; close this duration diagnostic. No extra duration GPU/native jobs. Parent Adam state continues, named RNG resets each round; this is not uninterrupted2200-step training or convergence evidence. [Saved BC result](qstudy_progress.md#saved-bc-continuation-result). |

The declared actors retain alpha10. During offline actor training the separately
fitted critic stays frozen; online FQL uses one critic update per actor update,
with an additional flow update only in the explicitly updated-flow arm. Loss
units and gradient contributions have been audited, but neither alpha nor the
update ratio has been demonstrated optimal. Weak measured Q directions do not
justify amplifying Q through an additional alpha grid. Equal alpha across
different utility transforms also does not establish equal gradient pressure.
Power and area remain audited/reporting metrics, not trained reward terms.
Geometry and direction evidence shows changes through the placement pipeline,
but does not identify legalization as the causal explanation for poor Q.
The second-design bp_be screen/probes support limited mechanism diagnosis;
they do not establish multi-seed cross-design policy improvement.

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
| Bounded timing offline actor | 2.287263 ± .014787 | −.415364 | −231.041 | 11.333 | 2.000 | .169886 | 61027.3 |
| Bounded timing+congestion actor | 2.303493 ± .020889 | −.414849 | −222.394 | 4.083 | 1.167 | .169842 | 61011.4 |

Final bounded report1796866 verifies96/96 successes with no missing, failed
or invalid requests, including24 reused BC references. Bounded WL−BC is −9753.75 ±2064.471um
(paired-seed SD), lower in all three seeds. WNS worsens, TNS slightly worsens,
and mean WL remains above current flow2.279086m. BC is an existing measured
reference without a bounded zero-Q update-equivalence certificate; the gain
cannot be attributed solely to the bounded formula. Timing's mean WL reduction
of1.157% passes the numerical primary threshold, but TNS violation increases
13.642ns (allowed10.870ns) and H overflow increases3.667 (allowed1).
Even with a future matched-control certificate, those secondary failures would
remain. Against bounded WL, timing improves WL0.738% but worsens TNS/H/V and
its own measured utility in all three seeds. Timing/congestion Q on their own
actor proposals has mean rho−.500/−.579; this is conditional development
evidence, not a universal claim about those objectives. Full uncertainties and
source hashes are in the [bounded result](qstudy_bounded_objective_protocol.md).

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

## Completed fixed-first4 detailed routing

Final report1797474 closes48/48 requests across BC/FQL and rounds0/2;
all48 are DRC0. Round2 FQL−BC mean DRT WL is+2265.75um (paired-seed
SD12100.81um), or+.114483%, failing the final-route WL non-worsening gate.
Mean WNS/TNS/power/area improve within the declared tolerances; negative
slacks remain. Round0 FQL−BC mean WL was+6400.75um. Both rounds retain
all three seeds and fixed first4 noises without selecting favorable outputs.
See the [complete final-route table](qstudy_online_results.md#completed-final-detailed-routing)
for means, uncertainties, failure denominators and exact report SHA.
GRT all8 and DRT first4 have different denominators and cannot isolate a
routing-stage effect. Fresh confirmation remains pending.

## Stopping and confirmation gate

The main/source/bounded GRT matrices and saved BC continuation are complete.
The declared detailed-route and call reports are also complete; finish cost
review and preservation closure.
Close negative predictive branches; do not open new parameter sweeps or online
variants. Main BC completed parent-optimizer continuation through2200 actor
updates. Its measured raw coordinate gap falls without stable physical gain.
Named random generators reset each round, so this is practical continuation,
not uninterrupted2200-step equivalence or convergence evidence. FQL also
updates Q in these rounds and cannot isolate duration alone. A lower loss
without physical benefit does not justify more duration variants.

Saved-only actor-loss diagnostic1798143 closes the gradient-pressure and
round0 Q-versus-physical-gain checks using complete native measurements.
The actual alpha10 parameter-gradient Q/distillation norm ratio rises from
.362 at round0 to .716 at round2. Of23 round0 pairs Q predicts to improve,
11 have worse measured utility. This motivates closing blind alpha sweeps
while Q directions remain unreliable, rather than assuming a stronger Q term
will help. Analytical alpha3/30 rescaling is not policy training; no update
ratio ablation or optimal-coefficient claim is made. Full units, uncertainties,
pairing and limits are in the [actor-loss diagnostics](qstudy_online_results.md#actor-loss-diagnostics-from-saved-states).

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
