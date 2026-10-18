# Q study execution record

This is an execution log, not a final result table. The frozen protocol is in
`qstudy_plan.md`; literature and source controls are documented separately.
Data and checkpoints live under `/fact_data/bohanli/flowplace_fql/qstudy_v1`.
Old Table 2 reproduction jobs remain held.

## 2026-10-18: behavior-flow labels and matched controls

The current FQL behavior flow can generate complete layouts without an actor
update. We now sample that flow, retain raw and legalized coordinates, execute
the fixed eight-thread backend, and use native GRT measurements as Q labels.
This is offline Q data collection. It becomes online learning only when a
policy's own new measured samples enter replay before its next update.

The released FlowPlace sampler and the current FQL behavior flow differ in
checkpoint, integration/projection and perturbation recipe. Treat their data
as different sources, rather than assuming identical support.

| Work | Slurm jobs | Declared scope | Interpretation |
| --- | --- | --- | --- |
| Main critic screen | 1795664; report 1795751 | Two designs, seven critic variants, three critic seeds, grouped TRAIN CV and fixed-budget refit | Prediction diagnostics; not placement gains |
| Per-net pin-span feature control | 1795807; final report 1795873 | Same folds and 2x128 hidden trunk; extra per-net inputs | Feature hypothesis, including additional input parameters |
| Direct timing-utility scalar control | 1795825 | Three critic seeds, same multi-head comparator folds and minibatch RNG | Separates objective representation from head count |
| Current-flow source labels | export 1795763; backend 1795769; collect 1795775 | Seed-0 common behavior flow; 32 TRAIN + 8 validation families, base and fixed-RMS local variant | 80 requested native labels, not 80 independent circuits or training seeds |
| Equal-label source controls | plan 1795783; check 1795814; assemble 1795840; fits 1795841; report 1795842 | O64 / T64 / M64 and base-only O32 / T32 / M32; legacy and residual critics, three critic seeds | Same old/new validation candidates; exact original 256 BC targets unchanged |
| Matched BC and legacy FQL | controller 1795764 | Three independent actor seeds; fixed common initialization; 1000 offline actor updates, two online rounds of 8 labels and 600 updates | Behavior flow frozen; actual self-collected feedback required for online updates |
| Local Q-direction checks | exports 1795767 / 1795768; backend 1795818 / 1795819; reports 1795844 / 1795845 | Four common-actor neighborhoods per design; two radii, both signs, matched random direction | Checks real optimization direction and legalizer changes; diagnostic labels stay outside replay |

The main matrix finished all 42 tasks; report 1795751 verified all CV/refit
artifacts with zero integrity errors. The pin overlay finished 18/18 and was
verified by report 1795873. The initially submitted final pin report 1795843
used a file where its CLI required a directory; this reporting-only failure was
recovered without any model or backend rerun.

Main matrix results do not establish a reliable timing
predictor. For example, on bp_fe the larger critic nearly interpolates TRAIN
WNS while grouped held-out WNS remains weak. Per-net pin-span features also
fail to consistently repair timing: their bp_fe OOF WNS Spearman is about
0.040, versus 0.008 for the shared-head Huber model, while TNS worsens. On bp_be
the same pin features improve WL ordering but do not yield a uniform metric
gain. Comparisons use complete matched seeds and retained reports.

Direct scalar timing-utility fitting also finished all three seeds. Its mean
within-fold Spearman is about 0.195 versus 0.204 for the matched multi-head
utility. MAE is slightly better (0.1075 versus 0.1099 in fold-specific utility
units), but local pair accuracy is worse (0.512 versus 0.526). These utilities
must not be pooled across folds with different TRAIN scales.

Report 1795874 checks standalone physical priors using the exact same folds.
On bp_fe, smooth-HPWL-only OOF rank correlation is 0.282, below legacy Q's
0.560. On bp_be it is 0.202, below the residual critic's 0.330, but its WL MAE
is better than that residual critic's. bp_be's TRAIN-fitted HPWL slope is
negative while raw HPWL is positively correlated on the already-observed
historical validation subset. The macro-only prior is not uniformly reliable
across these observed subsets. With only 32 TRAIN and eight historical
validation families on bp_be, limited coverage or sampling variation can also
explain this reversal; it does not establish a sampler-specific domain shift.
This motivates source and direction checks, not a fresh confirmation claim.

First current-flow backend samples succeeded with all seven native metrics
and legal actual macro geometry. Local variants can improve WL while worsening
slack; raw perturbation size can also substantially differ from the executed
coordinate change. These observations motivate preserving multiple metrics and
the complete raw-to-executed provenance. They are not aggregate efficacy claims.

## Coordinate coverage before looking at current-flow labels

Coordinate-only audit 1795926 passed eight invariance/provenance tests and
compared the sealed teacher export against the old audited dataset. It loaded
neither native outcomes nor metric arrays and made no inference, legalization,
or backend calls. Its immutable report is
`qstudy_v1/source_geometry_v1/snapshots/20261018T082630_job1795926_301245/report.json`.

| TRAIN source | Rows / families | Mean raw-to-legalized RMS | Raw layouts with overlaps | Raw layouts outside bounds |
| --- | --- | --- | --- | --- |
| Released sampler data | 384 / 192 | 0.08070 | 49.0% | 49.7% |
| Current behavior flow | 64 / 32 | 0.12570 | 96.9% | 82.8% |

All retained final normalized layouts are legal under the cache geometry
check; this is not a detailed-route DRC check. The raw-to-legalized displacement
uses movable normalized coordinate entries, with macro identity preserved.

The old TRAIN leave-family-out nearest-neighbor raw-coordinate RMS has mean
0.34943 and p95 0.54610. Current-flow TRAIN layouts have nearest-old-TRAIN mean
0.52628; 40.6% exceed that TRAIN-only p95 threshold. The executed-coordinate
comparison is similar. Old historical validation is closer to old TRAIN than
fresh teacher validation is. This is direct evidence that the retained sources
cover different coordinate regions, and that the current final legalizer
intervenes more strongly. It does not show which source produces better PPA,
prove Q failure, or isolate checkpoint, sampler, and perturbation effects.
Unequal family counts and dependent local variants also limit interpretation.

## Interpretation and next gates

- Source-study validation is reported separately on released-FlowPlace and
  current-flow candidates. Base-only controls address the explicit perturbation
  difference. Neither experiment isolates checkpoint from sampler effects.
- Three critic seeds on one seed-0 teacher distribution do not establish
  robustness across behavior-flow training seeds.
- Before a source-specific critic enters a matched actor experiment, convert
  its Q to common physical utility scales. Different TRAIN means otherwise
  change effective distillation strength and multi-metric preferences.
- Ranking accuracy alone cannot justify gradient optimization. Use the native
  positive/negative/random direction measurements, then matched actor outcomes.
- The flow-updated online contrast must start from the exact same offline
  checkpoint and initial collection labels. Shared-prefix imports must preserve
  all states and native evidence, and count reused measurements only once.
- GRT outcomes are not final detailed-route results. Missing runs remain
  missing; physical failures keep their declared penalty without fabricated PPA.

All jobs above use scavenger Slurm queues. Controllers preserve checkpoint and
request identities and recover only their declared work. Source-code snapshots,
dataset hashes and native manifests remain the authority for individual runs.

The stable tested study code has a local Git snapshot `5084276`; it was not
pushed as part of this execution step. Report metadata defines gradient signs:
the logged cosine compares the gradients of `alpha*distillation` and `+Q`.
Because actor loss subtracts Q, their plain gradient improvement directions
have the opposite cosine. This is not the cosine of Adam's actual parameter
step, and no Q direction exists for the q_weight=0 BC control.
