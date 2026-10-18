# Q study execution record

This is an execution log, not a final result table. The frozen protocol is in
`qstudy_plan.md`; literature and source controls are documented separately.
Data and checkpoints live under `/fact_data/bohanli/flowplace_fql/qstudy_v1`.
Old Table 2 reproduction jobs remain held.

## 2026-10-18: measured behavior-flow source control completed

All80 teacher-source requests completed with genuine native GRT results,
including the three preempted requests recovered without replacing their
inputs. The64 TRAIN and16 validation labels each contain all seven physical
metrics. Source fitting1795841 and report1795842 completed all36 tasks with
zero integrity errors. See `qstudy_source_results.md` for the immutable report,
matched three-seed tables, base-only sensitivity and limits.

The source result is negative for the current WL critics. On identical
teacher16 validation layouts, legacy O64 Spearman is0.500 +/-0.091, whereas
T64 is-0.346 +/-0.086 and M64 is-0.520 +/-0.093; substituting T64 worsens
ranking in all three seeds. Residual critics also worsen. This does not
invalidate measuring behavior-flow layouts, but it rules out claiming that
this64-label source substitution alone has improved Q.

Preparation1796553 completed, producing O64/T64 legacy configs with common
physical Q scaling and offline-only semantics. Controller1796559 is submitted
for a bounded actual-effect test: three seeds per source, common initialization,
alpha10, frozen flow,1000 updates and eight development layouts per seed.
Backend concurrency is2, with the same eight-CPU/24GB physical workers.
No M64 actor, lower alpha or source-online run is launched. These are tests
of the physical consequence of a negative predictor result, not a promoted
method. A same-input certificate is needed before reusing BC as a controlled
comparison; O64-versus-T64 does not depend on that certificate.

A prospective shared utility helper is being implemented for later bounded
multiobjective experiments. Existing linear WL actors, backend outcomes,
labels and running controllers remain unchanged. No multiobjective policy
result exists yet.

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
| Online flow-update factor | launch 1796053; reports 1796054 / 1796059 | Same completed legacy round-0 states and measurements; enable flow updates only for subsequent rounds | Inherited prefix is not a new training run or new physical measurement |
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

All six matched offline actor streams completed their fixed 1000 updates.
Their first 96 collect/evaluation backend requests were submitted as 1796007;
they are still development work, not completed final PPA. The flow-update
launcher waits for every declared legacy round-0 native outcome, verifies the
shared prefix, then starts the unchanged two-round controller. Its 20 software
tests passed in job 1796031, including a tiny real round-1 update, interrupted
transaction recovery, and prevention of controller launch without the final
verified import gate. The implementation is locally committed as `e621640`.

At round 1 the two legacy branches have identical initial measured feedback,
critic states and critic minibatch streams. Updating the independent flow can
change the teacher and actor, while Q's tensors remain paired until new
policy-specific feedback arrives. At round 2 the collected layouts can differ;
that comparison estimates the adaptive total effect of flow updates. All
shared physical-run identities remain visible and are counted only once.

Several scavenger backend attempts were preempted on blue-space before writing
terminal native results. Missing-only recovery coordinators are queued as
1796060 (teacher), 1796061 (bp_fe directions), and 1796062 (bp_be directions).
They wait for the original arrays, preserve completed outcomes, and recover
only interrupted missing indices using the same frozen eight-thread worker;
recovery workers exclude blue-space after repeated interruptions there. The
new coordinator's nine mock safety tests passed in 1796052. Teacher collection
1795775 and the direction reports now depend on verified recovery completion,
so an interrupted original array will not strand later source fitting.
Recovery metadata is in `qstudy_v1/recovery_v1/submission_20261018_1635.json`.

## Matched development final routing

Before reading any new final-route measurements, metadata-only job 1796084
declared 48 tasks: BC and frozen-flow legacy FQL, actor seeds 0/1/2, offline
round 0 and online round 2, and the first four of eight fixed evaluation noise
indices. This subset is fixed by index, not selected by Q or measured PPA.
The native flow is the same eight-thread OpenROAD context with detailed-route
iteration cap 20. This is separate from historical Table 2 cap-4 work, which
remains paused.

Controller 1796110 consumes the declared source GRT outcomes as they arrive,
with at most two detailed-route workers. Source sealing job 1796111 follows
the matched online controller 1795764; audited final report 1796112 follows
both sealing and routing. The controller's historical `full_route/summary`
text describes a single seed and is not the study's statistical report. Use
the `qstudy_final_route` development report, which averages noise within each
actor seed and then computes mean/sample SD across the three seed means.

The helper's ten tests passed in 1796078. They cover exact source/native
bindings, refusal to run a missing geometry audit inside the reporter,
configuration/noise mismatches, and retaining missing/physical failures in
the 48-task denominator. No complete-policy mean is reported unless all
declared samples for all three seeds are available. DRT rWL, WNS, TNS, power,
area and DRC are retained; GRT overflow is not relabeled as final routed
congestion. These are development results, not fresh confirmation evidence.
Plan and submission records live in `qstudy_v1/development_final_route_v1`.

### Completed offline first4 DRT evidence

Source resolution1797326 and audited report1797327 completed. The immutable
snapshot is
`qstudy_v1/development_final_route_v1/development_report/snapshots/20261018T114051_961307_job1797327`;
the source tables are `policy_groups.csv` and `matched_deltas.csv`. Of48
planned tasks,24 round0 layouts completed and24 round2 sources remain missing.
Each offline policy has all12 declared layouts: three independent training
seeds, each with the fixed first four evaluation-noise indices. All24 detailed
routes completed with DRC0; no candidate was selected by quality.

| Offline round0 policy | DRT WL (um), mean +/- seed SD | WNS (ns) | TNS (ns) | Power (W) | Std-cell area (um2) | DRC0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched BC | 1979399.833 +/-13918.662 | −.579834833 | −360.588083 | .173064250 | 61050.3083 | 12/12 |
| Legacy frozen-flow FQL | 1985800.583 +/-12414.537 | −.605670667 | −372.022917 | .173642083 | 61327.8417 | 12/12 |

The report first averages four noises within each training seed, then reports
the mean and sample SD across three seed means. Other table entries above
are means. Matched FQL-minus-BC WL is +6400.75um with paired-seed SD26202.405um;
seed deltas are −10918.25, −6424.75 and +36545.25um. Thus two of three seeds
improve WL, while the aggregate WL is worse. Aggregate WNS/TNS, power and area
also do not favor FQL. This is a negative/mixed offline development result,
not evidence of a stable PPA advantage. DRC0 does not imply timing closure:
both WNS and TNS remain negative.

These DRT policy means use first4, whereas the earlier GRT policy means use
all8 evaluation noises. They are not the same denominator and do not isolate
the causal effect of routing. No GRT congestion value is substituted for a
final routed congestion metric. Missing round2 layouts retain their24-task
denominator with unavailable PPA. Online final-route results remain pending;
fresh confirmation also remains pending and requires its own prospective lock.

## Direct current-flow evaluation control

A separate eval-only exporter has been tested for the current behavior flow:
three declared common-flow checkpoints, eight fixed evaluation noises each,
24 GRT requests. It will read the actual saved noise arrays from matched BC
and FQL round-0 requests and verify their hashes, rather than regenerate noise
from its integer seed. CPU and CUDA generators need not produce the same
array from the same integer seed. All samples are evaluation-only, with no
perturbations or eligibility for TRAIN or online replay.

This control measures the common behavior flow before further policy updates;
it is not the released twenty-step hard-constrained FlowPlace sampler or the
historical post-training four-noise flow probe. The common-flow checkpoints
differ across the three seeds. With flow frozen, repeating the same noises in
later rounds would generate the same layouts and does not justify new physical
calls. Its eleven software tests passed in job 1796142. Metadata declaration
1796149 verified the actual source arrays and sealed the 24-request plan;
GPU export 1796150 is queued. Metadata-only native dispatch preparation
1796201 depends on that export. It does not submit physical workers: native
measurements still require review of the completed export and exact dispatch.
Code is locally committed as `3b8e3fd`.

## Source-to-actor preparation and candidate selection

The source-critic configuration adapter passed eight tests in job 1796162
and was committed as `9dec8c0`. A subsequent check against real declared paths
found a byte-identical archived common-checkpoint alias: comparing full path
dictionaries incorrectly rejected it. A narrow fix binds both files and
compares content SHA, retaining both paths. All nine regression tests passed
in 1796193; the fix is committed as `c8613da`. No production source-actor
configurations have been prepared or submitted. Legacy source critics remain
offline-only until source-scaled
online physical-failure targets have a separately tested version.

The current direction probes use original bp_fe384 and bp_be64 critics. Their
results diagnose those critics, not new O64/T64 refits or a timing utility.
Three actor seeds also do not create three independent teacher-data sources.
The clean first source contrast remains O64 versus T64 with the same WL critic,
original behavior targets, frozen flow and common physical utility. M64 tests
a separate coverage hypothesis. These choices do not select whichever source
happens to win on historical validation. Existing BC384 comparisons retain
their descriptive/uncontrolled status when source dataset bindings differ;
no historical artifact is rewritten to make a controlled comparison appear.

The read-only eight-candidate selection reporter passed twelve tests in
1796191 and is committed as `983b7a6`. It compares fixed index 0, executed
macro-HPWL selection, recorded current-Q selection, measured GRT-WL oracle
and common-utility oracle. Policy means remain separate. It requires complete
audited pools, retains physical failures, reports all seven native metrics,
and aggregates across three actor seeds. It performs no new inference or EDA.
Current-Q selection compares generator-plus-scorer systems because online
FQL updates its critic while BC's critic remains frozen. Imported branch
round-zero exports are outside this reporter's standard-export scope and are
withheld rather than silently treated as ordinary pools.

Offline-only diagnostic report 1796184 completed for six round-zero actor
streams, with zero integrity-invalid rounds. Its immutable snapshot is
`offline_actor_diagnostics_v1/20261018T085649.517223Z_job1796184`.
Only 10 of 96 native outcomes were present at capture; this is not a physical
performance comparison. Across the three FQL seeds, mean logged Q/distillation
action-gradient norm ratios are 0.2664, 0.2333 and 0.2552; parameter-gradient
ratios are 0.4576, 0.3187 and 0.3106. Alpha is 10 and Q's weight is 1.
Q therefore supplies nonzero gradients but is smaller in norm at these logged
minibatches. This does not prove a correct physical direction or mean that
increasing Q strength will help. Logged action-gradient cosines are near zero;
the parameter-level means have positive sign, with the actor-objective sign
interpretation stated above. These are unpreconditioned diagnostics, not Adam
update vectors. No alpha or reward preference is selected from an unmeasured
surrogate gain.

Stored-array movement diagnostic 1796341 compares eight actual matched noises
per actor seed. It verifies that BC and FQL retained exactly the same initial
critic state, and performs no model forward calls or EDA. Mean raw FQL-minus-BC
movable-coordinate RMS is 0.02179, 0.01550 and 0.02342 across seeds; mean
executed RMS is 0.02602, 0.02691 and 0.04180. No pair collapses to identical
executed coordinates, and no raw/executed change cosine is negative. However,
mean change cosines are only 0.487, 0.506 and 0.538, and one executed movement
is 4.60 times its raw movement. This refutes complete erasure in these 24
pairs, while showing substantial geometric redirection or amplification.
Recorded Q generally favors FQL, but the fixed native snapshot has no complete
BC/FQL evaluation pair and cannot establish physical improvement. Evidence:
`qstudy_v1/offline_actor_diagnostics_v1/surrogate_movement_v1/movement_summary.json`.

A released-FlowPlace control is declared separately: eight actual matched
noise arrays, public checkpoint, official twenty-step hard sampling, then the
same final legalizer and physical context. One fixed public checkpoint is not
three independent training seeds, so this control has eight unique requests
rather than duplicating them across three seed directories. Seven tests passed
in 1796284; metadata declaration 1796300 completed with zero backend calls.
GPU export 1796301 and metadata-only native-plan preparation 1796309 are
queued. Export code is locally committed as `0c6f6dc`. Physical dispatch still
requires checking the completed export. This does not resume paused Table 2
work or put evaluation samples into Q training.

At 17:05 HKT the original teacher-label array 1795769 was terminal: 77 genuine
successes and three preempted requests (35, 50 and 64), with no confirmed
physical failures. Recovery controller 1796060 submitted only missing work;
the fixed collector and source-critic dependencies remain in place. Pending
or preempted requests are not assigned failure utility or fabricated metrics.

## Stored disagreement-penalty diagnostic

Read-only aggregation 1796329 and presentation 1796348 completed. The report
verified all 42 original screen tasks and their 168 CV/refit artifacts; 36
multi-metric tasks have saved beta diagnostics, while six legacy tasks are
explicitly not applicable. No model was refitted and no inference or physical
measurement was added. Evidence is under
`qstudy_v1/uncertainty_diagnostic_v1/report_job1796329/`.

The paired comparison uses beta 0, 0.5 and 1 on the same fitted model and
candidate pool. CV-fold statistics are averaged within each critic seed and
then summarized over three independent critic seeds. The six mechanisms are
a fixed descriptive set, not six independent replications. Historical
validation remains development data.

The penalty is not a stable improvement. On bp_fe, residual-critic WL beta
0.5 changes CV local-pair accuracy by -0.00006 +/- 0.01395 and reduces mean
physical pool regret from 0.086966 m to 0.073652 m, but only one of three
critic seeds changes its selected candidate. For the same critic's
timing-plus-congestion objective, beta 1 slightly increases rank correlation
yet worsens normalized selection regret in all three seeds. On bp_be,
residual-critic WL beta 0.5 improves local-pair accuracy by 0.04040 +/- 0.01750
while physical regret worsens from 0.163861 m to 0.169803 m. Across all six
mechanisms, neither nonzero beta improves historical-validation WL regret.

Keep beta 0 as the anchor. These observations separate ranking from candidate
selection; they neither select an alpha nor establish physical gradient
quality. The post-training mean-minus-disagreement score is not CQL or
COM-style conservative training and has no calibrated confidence guarantee.

## Partial native direction evidence and execution priority

Read-only snapshot 1796374 completed in 23 seconds. Its immutable manifest is
`qstudy_v1/probe_partial_snapshots/20261018T091049Z_job1796374/manifest.json`.
The snapshot accounts for 32/52 bp_fe and 19/52 bp_be measurements, all genuine
successes. It retains every predeclared cell and missing request; only measured
baseline/plus/minus trios enter differences. This is one original critic seed
at four common-actor neighborhoods, not a three-training-seed policy test.

At this incomplete snapshot, bp_fe legacy-Q plus/minus directional signs agree
with the backend in 0/2 complete trios at each radius. Mean positive-direction
utility change versus the common baseline is -0.00679 at RMS 0.005 and -0.01962
at RMS 0.02. The residual critic has 1/1 agreement at the small radius but 0/2
at the larger radius. bp_be has only one or two complete trios per cell, with
some identical-execution ties at small radius. Missingness is unbalanced across
base/radius, and random-direction cells have different completion coverage;
do not select a critic/radius or compare unpaired partial averages as winners.

This raises a real local-direction concern for the old bp_fe critics. Defer
the proposed stronger-Q alpha-3 trial; increasing a surrogate gradient that
is physically wrong can worsen exploitation. Complete the source comparison
and evaluate source-specific directions or controlled actor outcomes before
interpreting stronger extraction. Original384 probes do not validate O64/T64
critics, and these diagnostic labels remain outside TRAIN/replay.

To prioritize the three missing teacher labels, native array 1796007 received
a temporary scheduler-only throttle change 4 to 2. Running workers and their
eight-CPU/24-GiB physical context were untouched. Restore job 1796382 is bound
to `afterany:1796060`; it restores throttle 4 once teacher recovery terminates.
Before/after evidence is `qstudy_v1/scheduler/teacher_priority_throttle_20261018.json`.
Slurm returned warnings for completed array children, but `scontrol` verified
the new throttle. No other project or paused Table2 job was changed.

The cluster has `JobAcctGatherType=(null)` and no MaxRSS for the queried worker
steps. OpenROAD's own stage-memory lines do not measure Python and archive
peaks, so they do not justify reducing the whole-worker memory request.

## Matched critic-input diagnostic

Before fitting the six tasks, the executed-input diagnostic passed ten Slurm
tests (1796401), including paired minibatches, normalization bindings and exact
checkpoint/RNG continuation. Independent source review found no blocker.
The comparison reuses the completed residual-critic screen on bp_fe384 and
bp_be64, with three critic seeds, three grouped TRAIN folds and one full-TRAIN
refit per seed. Both sides use 2,000 updates, the same labels and family splits.
No new layout generation or physical evaluation is added.

The intervention selects final-legalizer, pre-DEF coordinates rather than raw
proposals. Coordinate-derived features, feature normalization and the
TRAIN-fitted smooth-HPWL residual baseline change together; this is a test of
the input preprocessing pipeline, not coordinates with all priors held fixed.
WNS/TNS are primary and are predicted as positive violation magnitudes.
Report both timing heads and all three seeds, alongside WL and congestion.
Repeated executed coordinates remain in their original family folds and their
cross-fold counts must be disclosed. Historical validation is development
evidence. Better prediction would neither establish backend-actual geometry
nor validate gradients through the nondifferentiable legalizer.

Declaration 1796428 completed with no physical calls. Plan SHA is
`6cc2bf11e7e2a196643518ece38daa103758658e1c0d5fa54a399dcf060fa43b`;
the six-task training array is 1796431, concurrency two, two CPUs/6 GiB each.
The declared successful-row counts are 384/16 for bp_fe and 64/16 for bp_be
(TRAIN/historical validation), with no excluded rows. Unique executed TRAIN
layouts number 382 and 63 respectively; bp_be validation has 14 unique layouts
among 16 logical records. Exact cross-fold overlap is recorded per fit, and
these duplicates are not additional independent evidence.

## Teacher-label collection completed

All 80 declared current-flow requests now have genuine successful native GRT
outcomes: 64 TRAIN and 16 validation, with no confirmed physical/geometry
failures. Recovery controller 1796060 completed its guarded verification with
`all_native_outcomes_verified,count80`. The final missing request 35 completed
as native job 1796414; request 50 reused its already-completed native evidence
instead of repeating physical evaluation. Collector 1795775 and source
assembler 1795840 subsequently completed, releasing the predeclared 36-task
critic array 1795841. This completes data acquisition, not Q/actor efficacy.

Restore job 1796382 completed and `scontrol` confirmed native array 1796007 is
back at throttle four. The eight-CPU physical context and completed measurements
were unchanged by this temporary scheduling priority.

A separate [failure-utility review](qstudy_failure_utility_review.md) identifies
a future multiobjective issue: a real successful TRAIN layout has linear
timing/congestion utility -4.5715, below the generic failure utility -4. Current
launched WL-only actors are unaffected. Before launching a congestion actor or
utility selector, prospectively fix success/failure ordering consistently in
actor, critic and reporting; changing presentation alone is insufficient. Raw
native metrics and stored legacy WL rewards remain unchanged.

## Executed-input predictive result

All six tasks in array 1796431 completed (30--36 seconds each). Reporting job
1796472 verified all 24 paired CV/refit artifacts, with zero integrity errors.
Reporter tests passed in 1796464 (nine tests); source commit is `08d8076`.
Immutable JSON and paired fold/seed CSVs are under
`qstudy_v1/executed_report_v1/`. No new inference, fit or physical measurement
was performed by the reporter. Every fold has zero exact pre-DEF executed
coordinate collisions across its training and target sets, despite duplicates
within the original logical dataset.

Primary timing results below are grouped TRAIN-CV scores, first averaging the
three folds within each critic seed, then reporting three-seed mean/sample SD.
WNS/TNS are positive violation costs; MAE is in ns. Rho is Spearman correlation.

| Dataset / cost | Raw MAE | Executed-input MAE | Raw rho | Executed-input rho |
| --- | ---: | ---: | ---: | ---: |
| bp_fe384 / WNS | .04834 +/- .00086 | .04556 +/- .00063 | .009 +/- .012 | .085 +/- .009 |
| bp_fe384 / TNS | 52.60 +/- .91 | 52.00 +/- .81 | .118 +/- .021 | -.009 +/- .023 |
| bp_be64 / WNS | .07681 +/- .00216 | .08011 +/- .00265 | .131 +/- .045 | .108 +/- .039 |
| bp_be64 / TNS | 112.98 +/- 1.74 | 135.89 +/- 3.67 | .218 +/- .042 | .188 +/- .077 |

The intervention does not consistently improve timing prediction. bp_fe WNS
improves modestly, but its TNS rank worsens; bp_be timing errors worsen.
bp_be TNS within-family pair accuracy also falls from .612 to .384. WL is
essentially unchanged on bp_fe, while bp_be WL rho falls from .522 to .281.
Even bp_fe WNS has worse top-choice regret (.1097 to .1229 ns), despite its
lower MAE; bp_fe TNS regret grows from 98.07 to 130.24 ns. Three critic seeds
measure initialization/optimizer variation on shared folds, not independent
data-sampling uncertainty.
All five heads, paired seed deltas and undefined-statistic accounting remain
available in the full report. The 16-row historical refit development results
are separate and do not override the grouped-CV conclusion.

Do not promote executed inputs to the actor from this result. Retain raw
proposal inputs as the anchor while the source-label study proceeds. This
does not establish that legalization is irrelevant, that backend snapping is
responsible, or that either representation supplies physically valid actor
gradients. Features, normalization and the analytic prior changed together;
the conclusion is limited to this fixed residual-critic preprocessing test.

## Behavior-flow feedback and prospective objective experiment

The user's suggested data chain is now executed: the current behavior flow
generated 40 base layouts, with 40 additional local variants. All 80 have
verified native backend successes and seven measured metrics. The 64 TRAIN
and 16 development-validation labels remain separate. This is offline data
acquisition, rather than an online policy update by itself.

The completed equal-budget source comparison is negative; see
[source results](qstudy_source_results.md). For the legacy critic, three-seed
mean rank correlation on the same 16 current-flow validation candidates is
+0.500 for O64, versus -0.346 for T64. The residual critic also fails to improve
this ranking. The source, sampler and perturbation recipe change together,
and all current-flow labels originate from common teacher seed 0. This does
not invalidate the data chain, or establish the effect on actor PPA.

Preparation 1796553 verified the complete source study, common warmups,
TRAIN-only scales and identical original 256 BC targets. Controller 1796559
now requests the predeclared O64/T64 legacy actor comparison: three independent
actor seeds, alpha 10, frozen flow, 1,000 offline updates and eight common
actual evaluation noises per seed, giving 48 native requests. It is an
explicit bounded negative-hypothesis test, with no selected best seed or
automatic online extension. Q magnitudes are corrected to the O64 TRAIN WL
reference. The frozen generic actor report's per-source utility must not be
compared across sources; recompute common utility from native outcomes.

CPU verification 1796608 independently certified that existing round-0 BC
actor/noise/native outcomes provide the same effective control: identical
common flow, actor initialization and random stream; no critic or flow
updates; finite zero-weight Q gradients. This is an effective-control
certificate, not an unperformed bitwise GPU rerun. Only original coordinates,
noises and native PPA can be reused; old predicted Q and source-scaled utility
are not reusable common-reference scores.

Current behavior-flow export 1796150 and public FlowPlace export 1796301 both
completed. Final CPU audit 1796702 verified all 32 layout/noise/graph bindings,
fixed nodes, shared actual noises and the public hard-sampling trajectory.
It submitted exactly 24 current-flow and eight public-flow native evaluations
as 1796709 and 1796710, each at concurrency two and eight CPUs/24 GiB per
worker. They are development baselines, not TRAIN data or resumed Table 2.

The [bounded-objective protocol](qstudy_bounded_objective_protocol.md) fixes
the prospective comparison of WL, WL plus timing, and WL plus timing and
congestion. Success cost B=2C/(1+C) and confirmed-failure cost 4 prevent a
finite successful target from ranking below a physical failure. This changes
risk preferences and gradient scale, so raw physical metrics and C must also
be reported. Missing/infrastructure outcomes remain missing. Native rewards
are unchanged. The critic regresses B directly; actor Q is minus mean
predicted B, with no second TRAIN-mean normalization.

Helper tests 1796716 passed 11 cases, including ordinary gradients, failure
semantics and finite extreme-value arithmetic. Screen tests 1796707 passed
12 cases, including exact optimizer/RNG continuation and TRAIN-only folds.
Independent source review found no current-scope blocker. Code commit
e95cf48 is pushed. Declaration 1796732 completed; the fixed nine-task array
1796737 requests three objectives times three critic seeds, each with three
grouped CV fits and one full-TRAIN refit. No objective actor has launched yet.
The three objectives share splits and minibatch streams on the complete
384-row TRAIN data; that matching does not automatically extend to datasets
with objective-dependent missing labels.

The source actors' first two pending GPU jobs, 1796604 and 1796605, received
a scheduler-only time-limit change from three hours to 20 minutes. The six
previous matched 1,000-update actors took 3m33s--6m08s. `scontrol` verified
both new limits; hardware, code, checkpoints, updates and random streams are
unchanged. All L20 GPUs were allocated at the scheduling inspection, so this
does not guarantee an earlier start. The frozen worker script still declares
three hours for future submissions and must not be edited during active runs.

## Bounded-objective screen and offline policy launch

All nine tasks in 1796737 completed, covering 36 grouped-CV/refit models.
Read-only report 1796778 verified every fit with zero integrity errors; its
immutable JSON is
`qstudy_v1/bounded_screen_report_v1/20261018T095653437846Z_job1796778/report.json`
(SHA `d66aadbbfe466ac7e66385ce66223b2bbde1e716ce9eecd5458810e4a6b2b68f`).
Reporter tests 1796768 passed eight cases. Three-seed mean/sample SD below first
average the three grouped TRAIN-CV folds within each critic seed. Every objective
has a different scalar target; these are prediction diagnostics, not comparable
cross-objective physical utilities.

| Target | B MAE | Its constant-TRAIN baseline MAE | Spearman rho | Within-family pair accuracy |
| --- | ---: | ---: | ---: | ---: |
| WL | .01321 +/- .00029 | .01524 | .561 +/- .018 | .547 +/- .040 |
| WL + timing | .03480 +/- .00068 | .02728 | .197 +/- .024 | .523 +/- .003 |
| WL + timing + congestion | .07231 +/- .00184 | .06357 | .277 +/- .026 | .525 +/- .029 |

Timing and congestion do not beat their own constant baseline in MAE, and local
ordering remains weak. Historical 16-row development results are kept separate;
none establishes correct physical actor gradients. The same alpha10 also does
not equalize Q-gradient pressure across objectives. Power and area remain
reported metrics, not optimized objectives; their TRAIN variation audit is in
[the objective protocol](qstudy_bounded_objective_protocol.md).

The new offline-only actor adapter retains the exact frozen train/export
function code in a private namespace, with setup and reward overridden. Q is
minus mean predicted B; no second TRAIN-mean normalization is performed.
Tests 1796790 passed six cases, including exact model/optimizer/RNG continuation,
private signal state, immutable full-TRAIN metadata and evaluation noise.
Independent review found no current-scope blocker. Actual CPU integration
1796829 checked all 36 fits and nine refits: 18 saved TRAIN/validation prediction
comparisons were bitwise identical, frozen-input gradients were finite/nonzero,
and a real common GraphPolicy teacher10/actor gradient check passed. Its report
SHA is `798ee7f510879b5320bc9d1ee5b4d50b2f32a74a672b5c468e5fc09919747328`.
This validates implementation, not PPA improvement.

Preparation 1796833 published three immutable configs under
`qstudy_v1/bounded_actors_v1/`. Their SHA values are WL
`806c0093a235042e9e32c234fb2cdeb9a1b5fdc3b0d8046fa2ba8da660d40cc6`,
timing `0b02b8bec3d6fcb30091f4aaf4eef56b3de58c662c5c902c5d189624a26a24e2`,
and congestion `b05f3219672094d5e930c97f667bbae25adc576acc53854c137fadd249e07a98`.
They share full-TRAIN physical scales, three independent common warmups, the
same original 256 BC targets, alpha10, frozen flow and 1,000 actor updates.
GPU jobs 1796839--1796847 form two dependency chains, limiting concurrency to
two. Each uses the same L20 request and eight actual noise4600000 evaluations.
No collect trajectories or online rounds are enabled. These are bounded actual
negative-hypothesis tests, not automatic method promotion.

Native finish job 1796852 waits for all nine successful GPU jobs and refuses
missing/changed export markers before invoking the frozen backend controller.
The declared budget is 72 native requests at concurrency two, eight CPUs and
24GiB each. Its outer wrapper clears inherited SLURM_RESTART_COUNT before new
child submissions; this prevents a controller restart count from being confused
with a child's own requeue count. Its script and submission are recorded under
`bounded_actors_v1/native_finish/`. Original physical metrics and rewards remain
unchanged.

Current/public flow native-finish controllers 1796791/1796792 wait for original
arrays1796709/1796710 before verifying and recovering only missing outcomes.
The new read-only baseline reporter passed 13 tests in1796826 and independent
review; current24 and public8 keep distinct uncertainty units. A source-actor
common-reference report1796793 waits for controller1796559 and recomputes
utility under O64 TRAIN scales; it does not compare source-normalized scores.

The matched controller1795764 stopped after array1796007 ended: 25 requeued
children hit the duplicate-live-attempt guard. Their controller had exported
SLURM_RESTART_COUNT=1 into initial children; the first real requeue also had1,
so the frozen same-job restart test could not distinguish them. This is an
infrastructure recovery problem, not25 physical failures. The affected attempts
and manifests are being audited before same-job recovery;71 successful children
will not be recomputed. Downstream reports triggered by afterany may be partial
and cannot be treated as completed online-policy results.

[Fresh confirmation](qstudy_fresh_confirmation_protocol.md) will use separately
locked evaluation inputs only after development selection. The current actor
noises, source labels, probes and historical validation are development data.

Bounded-actor native reporter final tests1796867 passed11 cases and independent
review. Actual read-only contract1796865 used the final reporter SHA
`82d9a5721619cc8a7fdac04aaf53a60138043ed5d7c610e56eef81cf1da01a40`:
it verified14 existing BC successes, retained10 missing BC requests and all72
not-yet-exported bounded requests without integrity errors. This is a partial
reporting check, not policy efficacy. Formal report1796866 waits for native
finish1796852. Reused BC is explicitly a descriptive unchanged-layout/native
reference: no new bounded zero-Q update-equivalence certificate is claimed.
Three bounded arms are the matched objective intervention; each is evaluated
under all three common-TRAIN-scale utilities, and saved Q is calibrated only
against its own objective. Flow-baseline report1796858 waits for both existing
native-finish controllers. Source and full request denominators are retained
when reports are partial.

## Diagnosed main-array recovery

Readonly audit1796838 strictly validated71 existing native successes and pinned
25 interrupted attempts, original request/stream/matrix, unfinished manifests
and the guard-only failure stdout. These are real scheduler FAILED states,
not physical-failure labels. Most old Slurm task records had already expired
from slurmctld (MinJobAge300s), so another same-job requeue was unavailable.

A new narrow adapter keeps the original physical evaluator bytecode and sample
flock. In private globals only, it admits FAILED for exactly the25 independently
SHA-bound original jobs; real observed states remain FAILED. Unknown/live,
mixed-state or new unaudited FAILED attempts are refused. Actual loaded stream
SHA, index-to-sample mapping and request SHA are checked before any no-attempt
or completed-outcome path; evidence is checked again inside the existing lock.
Completed native results are still recovered through the frozen evaluator.
The source module, prior attempts, physical configuration and71 successes are
unchanged. Independent review identified and fixed an initial stream-binding
gap before any physical dispatch. Final19 tests and actual96-request readonly
preflight passed in1796901.

Frozen adapter SHA is
`9020a52eddf7d576ecde6a77309c15c3b3d9b14ae1949143b8d6c94ddc58a816`;
formal plan `qstudy_v1/main_guard_recovery_v2/plan.json` has SHA
`e9041c438c605ee458c6694830a7edf33f7cd9b7b6bdfb9c56fcb224141615d5`.
Older preflight plans remain preserved and are not the production declaration.
Array1796906 dispatches exactly25 missing indices, concurrency4,8CPUs/24GiB,
no automatic requeue, excluding natural-selection/enterprise/blue-space.
Validation1796908 runs after the array. These are recovery attempts, not25 new
logical training examples or a scientific change in the feedback budget.

Main resume1796907 waits for successful recovery and independently checks all96
native outcomes before opening the same original controller state. Its outer
wrapper clears inherited SLURM_RESTART_COUNT on every entry, including future
requeues. The same frozen actor, source hashes, two online rounds, GPU
concurrency2 and backend concurrency4 remain declared. Report1796909 waits for
that resumed controller; source-resolution1796913 and final-route report1796914
refresh earlier partial route evidence. If recovery is interrupted again,
ordinary missing-only diagnosis continues; no interruption is a PPA label.

bp_be original probe array1795819 is now terminal:41 successful jobs and11
preemptions, with no FAILED worker. Existing missing-only coordinator1796062
was released by that dependency and will recover the missing requests without
repeating completed labels. bp_fe probe recovery1796061 continues separately.

## First restored native result and complete public-flow reference

The first narrow recovery task1796906_0 completed successfully. Read-only audit
1796929 validated72 native outcomes, including that restored request, and found
the original71 outcome hashes and all25 original attempt hashes unchanged.
The recovery admission records the genuine old scheduler state FAILED; the new
physical attempt records restart0 and eight threads. This is recovery of an
existing logical request, not an extra training example. An earlier external
audit1796921 failed because the manifest stores num_cores as the string "8";
the corrected read-only audit converts this metadata to an integer. Neither
audit invoked EDA or changed physical outputs.

The first source actor1796604 completed its1000 updates and eight evaluation
exports in5m54s; seed1 actor1796605 finished in5m51s. Pending source actor1796924
received a scheduler-only time-limit reduction from3h to20m, consistent with
these measured runtimes, and excludes natural-selection/enterprise. The
immutable worker, GPU type, configuration, updates and numerical settings are
unchanged. Bounded-objective actor1796839 has started; no measured bounded
policy gain is available yet.

Public-flow finish1796792 completed all8 requested native outcomes. Baseline
report1796919 initially failed at a reporting-only contract check: it rebuilt
six dispatch fields but compared them with the original full15-field records.
Commit4fb344f reconstructs all15 fields and preserves strict ordered equality.
Missing/invalid/not-exported requests now also make PPA success-conditional,
with an explicit successful-sample denominator. All17 tests passed in1796942,
including the actual32 frozen export/dispatch records; independent review
passed. No export, checkpoint or native measurement was changed or repeated.

Corrected read-only report1796944 completed without integrity errors:
`qstudy_v1/flow_baseline_report_contract_v2/20261018T103032.512619Z_job1796944`.
At that immutable snapshot the current behavior-flow reference had16/24 native
successes and8 unresolved requests, so its three-seed mean is withheld.
The public released checkpoint had8/8 successes, no confirmed failure and no
unresolved request. Its descriptive means and sample SDs across eight actual
generation noises are:

| Public released flow, one checkpoint | Mean | Noise sample SD |
| --- | ---: | ---: |
| GRT wirelength (um) | 2403073.375 | 88253.010 |
| WNS (ns) | -0.41083725 | 0.05117051 |
| TNS (ns) | -216.567625 | 25.832686 |
| Horizontal overflow sum | 76.875 | 96.416415 |
| Vertical overflow sum | 15.25 | 23.013971 |
| Power (W) | 0.169719 | 0.001412872 |
| Standard-cell area (um2) | 60188.8625 | 396.965662 |

These are development GRT outcomes from one public checkpoint, not three
independent public training seeds, final detailed-route results, or a resumed
historical Table2 reproduction. They do not yet establish a matched policy
improvement. The full report1796858 remains dependent on completion of both
baseline finish controllers.

## First bounded actor contract and separate confirmation exporter

Bounded WL actor seed0 completed1000 updates in1796839. Independent read-only
audit1796925 verified207 actor state tensors changed, while the behavior flow
and critic remained identical to their bound common/refit checkpoints and
their optimizer step counts stayed0. All eight actual evaluation noises are
byte-identical to matched BC. Exported requests passed the native validator;
the audit invoked no EDA and made no optimizer update.

Across the ten logged gradient diagnostics, mean Q/distillation gradient-norm
ratio was0.12865 in action space and0.27512 over actor parameters. These are
training-contract and scale diagnostics, not evidence of a physical improvement.
Other objective/seed actors and their native comparisons remain in progress.
The audit is retained at
`qstudy_v1/bounded_actor_post_gpu_audit_v1/job_1796925/report.json`, SHA
`3c9c3db69c9ea143065e8bc4a1de12e6f91e45417fdcce3b274b971f33d753b4`.

The separately reviewed fresh exporter passed16 CPU contract tests in1796984.
It provides validation/declaration without model/noise/backend execution and
immutable evaluation-only exports for frozen actors, current behavior flow and
the released official flow. Original training configs and round identities are
preserved. Fresh selection, noise85000000 generation and physical dispatch
remain unopened until development review and the prospective selection lock.
See [the confirmation protocol](qstudy_fresh_confirmation_protocol.md).

## Complete direct-flow references and fixed conservative-critic contrast

Native-finish1796791 recovered only the two interrupted current-flow requests.
Final read-only report1796858 verified32/32 native successes, zero missing,
invalid or confirmed-failure cases. All32 retain the same eight actual noises
and fixed physical revision. Its immutable report is
`qstudy_v1/flow_baseline_native_report_v1/20261018T104500.097363Z_job1796858/report.json`,
SHA `ee8d29d7996616f0949bfed10ed83f673e0400e6f57c85b7d5a107d93597b406`.

| GRT metric | Current behavior flow | Public official flow |
| --- | ---: | ---: |
| Wirelength (um) | 2279086.375 +/-13389.904 | 2403073.375 +/-88253.010 |
| WNS (ns) | -0.40655675 +/-0.01713645 | -0.41083725 +/-0.05117051 |
| TNS (ns) | -206.694546 +/-22.047881 | -216.567625 +/-25.832686 |
| Horizontal overflow sum | 22.125 +/-16.891658 | 76.875 +/-96.416415 |
| Vertical overflow sum | 7.125 +/-4.160829 | 15.25 +/-23.013971 |
| Power (W) | 0.169797 +/-0.000302041 | 0.169719 +/-0.001412872 |
| Standard-cell area (um2) | 60997.745833 +/-262.000356 | 60188.8625 +/-396.965662 |

Current-flow uncertainty is sample SD across three equally weighted independent
training-seed means, each using eight outputs. Public-flow uncertainty is the
descriptive noise SD of eight outputs from ONE released checkpoint. These
different uncertainty units must not be combined as if there were three public
training seeds. Checkpoints and sampling paths also differ. These results are
references, not a controlled explanation of why a method improved; final DRT,
DRC and stable Q-related gains are not established by this report.

The fixed COM-inspired WL contrast passed independent review and11 tests before
formal dispatch. CPU preparation1797008 verified all12 existing supervised WL
control fits and froze `qstudy_v1/com_wl_v1/plan.json`, SHA
`cd4e2fa088c91a8b769d864a374376bff3f03a321fce8814b322e900c0b09d5b`.
Array1797016 has three seed tasks with three family-CV fits plus refit each:
12 new fits,2 CPUs/6GiB per task, concurrency3, no new actor or backend call.
The source is committed as23f87e2. Its fixed loss and coordinate constraints
are specified prospectively in [the literature note](qstudy_literature.md).
It tests training-time conservatism separately from the earlier negative
post-hoc ensemble-beta screen; no automatic hyperparameter expansion is opened.

## Matched offline results after missing-only recovery

All25 recovery tasks1796906 completed with genuine native success. Final
audit1796908 verified the complete96-request round0 inventory. The original71
outcome hashes and25 interrupted-attempt hashes remain unchanged; the recovery
metadata summary SHA is
`282ec512b9bc8befd5c3d44c12740c8715815e7d7f3993d1fa40a0f40b6e82f8`.
The96 requests contain48 training-eligible collection layouts and48
evaluation-only layouts. Evaluation labels never enter replay. Main resume
1796907 has no remaining dependency and awaits ordinary CPU scheduling before
continuing the original two matched online rounds.

Separate read-only round0 report1797018 is complete: six policy/seed rounds,
96 verified native outcomes, zero invalid cases. Its snapshot is
`qstudy_v1/matched_offline_r0_report_v1/20261018T104921.160264Z_job1797018/report.json`,
SHA `dc19bc3b64d3ea7888cffb2b70e555f183e3a32f3cb7885b89b2e76e7340c7d6`.
The following are evaluation-only means and sample SD across three independent
training-seed means, each using all eight common generation noises:

| GRT metric | Matched BC | Legacy offline FQL |
| --- | ---: | ---: |
| Wirelength (um) | 2314028.167 +/-12141.004 | 2307889.792 +/-3804.304 |
| WNS (ns) | -0.40360479 +/-0.01692463 | -0.40720167 +/-0.00749839 |
| TNS (ns) | -217.399438 +/-16.505006 | -217.651750 +/-0.675703 |
| Horizontal overflow sum | 7.666667 +/-2.504163 | 2.625000 +/-1.887459 |
| Vertical overflow sum | 1.375000 +/-0.450694 | 0.666667 +/-0.072169 |
| Power (W) | 0.170009542 +/-0.000150770 | 0.170139583 +/-0.000453631 |
| Standard-cell area (um2) | 61200.241667 +/-347.024987 | 61198.829167 +/-393.850309 |

The paired FQL-minus-BC wirelength difference is -6138.375 +/-15933.804um;
seed0/1 improve and seed2 worsens. Congestion decreases, while mean signed
timing slack is slightly worse. This is a small, unstable offline policy-mean
gain, not a stable multiobjective improvement or final detailed-route claim.

Read-only fixed-eight-candidate analysis1797038 also completed, using only
those existing evaluation outcomes. Its snapshot is
`qstudy_v1/matched_offline_r0_selection_v1/20261018T105429_998452_job1797038`.
No model inference or physical call was added. Each rule selects one output
per seed from the same eight candidates; uncertainty is across those three
selected outputs, not across24 candidates:

| Fixed selection rule | BC GRT WL (um) | FQL GRT WL (um) |
| --- | ---: | ---: |
| Noise index0 | 2373611.667 +/-37852.973 | 2334025.333 +/-38214.956 |
| Minimum executed macro HPWL | 2343312.333 +/-37946.858 | 2342670.000 +/-22715.448 |
| Maximum recorded Q | 2275970.667 +/-15562.540 | 2240981.667 +/-45602.133 |
| Oracle minimum measured GRT WL | 2198936.667 +/-29275.152 | 2203561.667 +/-38291.581 |

Q selection helps these development pools; its paired FQL-minus-BC wirelength
difference is -34989 +/-30581.283um, with WNS worse and TNS better on average.
The oracle inspects all eight real backend labels and is not a free deployable
selector. Selected-layout results and policy means answer different questions.
These inspected outcomes cannot become fresh confirmation later.

COM array1797016 completed all three seed tasks in3m06s, producing12 fits.
Its separate paired reporter passed11 tests in1797030 and independent review;
it retains the full COM binding and validates the12 original controls without
changing their verifier. Predictive accuracy and regularization pressure will
be reported separately from any actor or physical-direction claim.

## Conservative-critic result and online continuation

Paired COM report1797042 completed with12 new verified fits,12 unchanged
supervised controls and zero integrity errors. Immutable snapshot:
`qstudy_v1/com_wl_report_v1/20261018T105624786102Z_job1797042/report.json`,
SHA `ef82ee31e3fea8f4abd14fb1221a5d3246f7369e76cb09ed14430f0b5751bacf`.
The reporter and tests are committed in95a281e. Grouped-family CV metrics are
first averaged within each seed, then over three independent critic seeds:

| Bounded-WL CV diagnostic | Supervised control | COM-inspired fixed regularizer |
| --- | ---: | ---: |
| MAE | 0.013213147 | 0.013064363 |
| Spearman rank correlation | 0.561028977 | 0.502050403 |
| Family-pair accuracy | 0.547205688 | 0.503251764 |
| Top-candidate regret | 0.014411220 | 0.019259676 |

Paired MAE change is -0.000148784 +/-0.000201476. However, rank correlation
decreases in all three seeds, family-pair accuracy decreases in all three,
and top regret increases in all three. Smaller raw-input gradients and lower
regression error alone do not validate the direction that an actor will take.
The previously inspected historical16 favor COM on MAE/rank but do not reverse
this grouped-CV decision. No COM actor, new physical call or beta grid is
promoted from this result; the negative predictive result remains retained.
All per100/final regularization-pressure logs were present and validated.

Main controller1796907 started and printed the strict96 verification before
resuming round1; no round0 backend requests were repeated. Scheduler-only
adjustments reduce its resumable outer window from12h to4h and allow the
lightweight controller on blue-space. Its script still excludes blue-space
for physical children and clears inherited restart counts on every entry.
These metadata changes are recorded in
`qstudy_v1/main_resume_scheduler_adjustment_v1.json`; source, scientific
budgets and original checkpoint/state identities remain unchanged.

A separate development control is being prepared to compare the same fixed
eight-candidate Q-selection rule for current behavior flow, BC and legacy
FQL. It may only rescore saved proposals with the same frozen initial critic,
using CPU Slurm and no optimizer/layout/backend call. CPU recomputation of
the existing BC/FQL scores is required to check CUDA-score selection agreement.
This control tests whether actor learning adds value beyond generate-and-rank;
it neither changes the fresh selection nor opens its reserved noise stream.

The three bounded-WL actors are now complete. Controller1797058 starts their
already declared24 native evaluations while the other six objective actors
continue training. This changes scheduling order only: no new layout, training
step or logical physical-request budget is added. The final72-request finisher
1796852 keeps all remaining GPU success dependencies and additionally waits
for1797058 to become terminal, then validates/reuses completed WL outcomes.
Its same frozen controller and backend remain in use. Submission metadata is
`bounded_actors_v1/native_wl_early_v1/submission.json`; read-only snapshot job
1797061 will report any complete WL comparison while explicitly retaining the
other objectives as unresolved. Final reporting still waits for all72 requests.

### Completed bounded-WL native comparison

Controller1797058 and report1797061 completed all24 bounded-WL evaluations:
three training seeds × eight actual development noises, all successful, with
no missing or invalid WL outcomes. The existing BC reference also has24/24
successful outcomes on the same noises. The report's overall `partial` status
reflects unmeasured timing/congestion objective arms, not incomplete WL data.
The immutable report is
`qstudy_v1/bounded_wl_early_report_v1/20261018T114552.392983Z_job1797061/report.json`
(SHA `0b7656e0507e60cb2b6980cd199f56b18595275ff018288d025438b5c5d4fc2b`);
native means and paired differences come from `seed_aggregates.csv` and
`matched_contrasts.csv`, without any new analysis or model/backend call.

| GRT policy mean | WL (m) | WNS (ns) | TNS (ns) | H overflow | V overflow | Power (W) | Std-cell area (um2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing BC reference | 2.314028167 | −.403604792 | −217.399438 | 7.666667 | 1.375000 | .170009542 | 61200.2417 |
| Bounded-WL offline actor | 2.304274417 | −.418561833 | −217.921733 | 2.000000 | .791667 | .170021208 | 61525.1708 |

Means first average eight outputs within each seed, then average the three
training seeds. Paired WL−BC is −9753.75 +/-2064.471um (sample SD of three
seed deltas); the individual WL deltas are −9257.875, −7982.375 and −12021um.
All three improve WL. WNS worsens in all three seeds; mean TNS slightly
worsens. Both mean overflows decrease, whereas mean power and area increase.
Bounded-WL still has worse mean WL than current flow's same-noise2.279086m.

The report admits a paired native reference, but explicitly rejects controlled
effective-update eligibility: there is no bounded zero-Q update-equivalence
certificate for the reused BC. This is descriptive measured evidence, not an
isolated causal effect of the bounded utility formula. Timing/congestion
objective policies remain unmeasured here. Finish their already-declared
matrix before judging multiobjective tradeoffs; no alpha5/grid expansion or
new online branch follows from this early WL result.

## Completed local physical-direction checks

Recoveries1796061/1796062 and reports1795844/1795845 completed all52 native
requests on each design. Both inventories match the saved outcome hashes;
there are no missing labels, physical failures or invented infrastructure
penalties. FE evidence SHA is
`11397335826d98d073f22b282ab2591b766f84b6fe6766687129a07b55023c2b`;
BE is `9631813760830ee7e682a82a8f119ad24936681840b4f772e65a796010ebf475`.

Each row below describes four starting-layout neighborhoods of ONE common
actor and ONE critic seed. Signs compare the predicted positive direction to
its negative direction; improvement measures positive direction versus the
unmodified baseline. These are different tests. Utility uses each design's
frozen TRAIN-WL normalization; values are not per-layout percentage changes.

| Design / critic | Raw RMS radius | Correct non-tied directional signs | Ties | Mean positive-direction utility change |
| --- | ---: | ---: | ---: | ---: |
| bp_fe / legacy | .005 | 1/4 | 0 | -.005220 |
| bp_fe / legacy | .020 | 1/4 | 0 | -.009149 |
| bp_fe / Huber residual | .005 | 2/4 | 0 | -.002974 |
| bp_fe / Huber residual | .020 | 1/4 | 0 | -.008555 |
| bp_be / legacy | .005 | 2/2 | 2 | +.001671 |
| bp_be / legacy | .020 | 3/4 | 0 | -.001511 |
| bp_be / Huber residual | .005 | 1/1 | 3 | 0 |
| bp_be / Huber residual | .020 | 3/4 | 0 | -.000961 |

Matched random positive directions average +.005744/-.001666 on FE and
+.002625/+.002813 on BE at radii .005/.020. Random references reused under
two critic normalizations are not independent repeats. At BE residual .005,
every positive-direction execution equals its baseline: its lone correct
non-tied sign only reflects a worse negative-direction layout. At .020 both
BE critics improve three bases, but the fourth makes the mean negative.
This is not evidence for generally useful ascent gradients. It also does not
identify the legalizer as the cause: the experiment tests the whole raw-action
to executed-layout to GRT mapping. These old critics do not validate new
bounded/source critics. An actor alpha reduction remains conditional on actual
favorable policy evidence; no alpha grid is launched from these probes.

The first fixed-Q selection report job1797079 failed only while writing its
score CSV because flow rows already contained a seed field. Its incomplete
snapshot is retained. A narrow explicit identity-merge fix passed13 tests in
1797081; rerun1797084 repeats CPU critic scoring only, with no additional
layout, native evaluation, or label. No incomplete report is presented as a
completed result.

## Same frozen Q selects from Flow, BC and FQL

Read-only CPU critic job1797084 completed in35s. Its immutable snapshot is
`qstudy_v1/fixed_initial_q_selection_v1/20261018T111526_522211_job1797084`.
All72 saved raw proposals and their already measured native outcomes are
accounted for. There was no new policy inference, optimization, layout or EDA
call. All six BC/FQL pools have identical CPU-versus-recorded-CUDA argmax and
full ordering; maximum absolute score difference is1.1921e-7. All generators
are selected using the same CPU implementation and same initial Q per seed.

The selector chooses exactly one of eight existing candidates in each seed.
Numbers below are means and sample SD over those three selected outputs, not
over24 supposedly independent training runs. These are development results.

| Native GRT metric | Behavior flow + Q selection | BC + Q selection | FQL + Q selection |
| --- | ---: | ---: | ---: |
| WL (m) | 2.221280 +/- .027090 | 2.275971 +/- .015563 | 2.240982 +/- .045602 |
| WNS (ns) | -.392523 +/- .020164 | -.335976 +/- .022515 | -.370599 +/- .038509 |
| TNS (ns) | -172.297 +/- 78.022 | -182.738 +/- 11.407 | -149.911 +/- 17.771 |
| Horizontal overflow sum | .667 +/- .577 | 31.333 +/- 38.083 | 10.000 +/- 14.731 |
| Vertical overflow sum | 0 +/- 0 | 2.667 +/- 2.887 | 1.000 +/- 1.732 |
| Power (W) | .1698443 +/- .0011625 | .1699283 +/- .0008473 | .1700127 +/- .0004417 |
| Standard-cell area (um2) | 60753.7 +/- 1074.1 | 60456.9 +/- 1255.8 | 60420.5 +/- 1082.3 |

For comparison, means over all eight generated outputs per seed are
2.279086/2.314028/2.307890m for flow/BC/FQL. The flow's unselected congestion
means are worse, so its lower mean WL alone is not an overall PPA advantage.
The measured-label oracle WL references are2.183941/2.198937/2.203562m;
they inspect all eight backend labels and are not free deployable selectors.

This control supports separating candidate ranking from gradient-based actor
learning. The flow + Q selector is competitive here, and Q-based actor
improvement is not established by Q's ability to select candidates. Current
flow and the one-step actors use different inference costs (10 versus1 network
steps per candidate); the comparison matches candidate count and native calls,
not FLOPs. No fresh winner or new actor hyperparameter is selected from this
single small inspected comparison.

## Saved-proposal teacher versus one-step gap

Read-only CPU Slurm1797105 examined the same72 saved proposals, actual noises
and measured outcomes used above: behavior flow, BC and FQL, three independent
training seeds and eight development noises per seed. It loaded no policy,
performed no inference, optimization or legalization, and added no physical
calls. The pinned graph cache supplies the authoritative movable-node mask;
all11 macros in this design are movable. Report SHA is
`11b08043df5cccde5910da0810ed1c78dbe3363923f81c0ef763f67f8f467cc9`,
at `qstudy_v1/distillation_gap_v1/20261018T112053Z_job1797105/report.json`.

| Paired distance to the same-noise behavior flow | BC | FQL |
| --- | ---: | ---: |
| Saved pre-legalization coordinate RMS | .100037 +/- .004179 | .100850 +/- .003584 |
| Saved external-legalized coordinate RMS | .143660 +/- .010784 | .137539 +/- .009395 |
| Mean measured GRT WL minus flow (um) | +34941.79 +/-25530.76 | +28803.42 +/-9603.54 |

Uncertainty is the sample SD of three per-training-seed means. Coordinates
are normalized to[-1,1]; raw RMS .10 is about5% of each corresponding canvas
axis. Saved raw actor exports are already clipped, whereas the training MSE
uses unclipped predictions, so this distance is not the training loss itself.
No BC or FQL proposal exactly matches its paired flow output. Legalization
increases the pair's RMS in18/24 cases for each actor, but this observation
does not isolate a causal effect of legalization or prove under-training.

Both one-step arms have worse average WL than the same-noise flow in all three
training seeds. Other measured changes remain mixed: BC-minus-flow means are
WNS +.002952ns, TNS -10.70489ns, horizontal/vertical overflow -14.4583/-5.75,
power +.00021254W and standard-cell area +202.496um2. FQL-minus-flow means are
WNS -.000645ns, TNS -10.95720ns, horizontal/vertical overflow -19.5/-6.4583,
power +.00034258W and area +201.083um2. All72 outcomes are successful.
The comparison motivates separating teacher-to-one-step approximation from
Q-based policy improvement. It does not establish that training longer will
improve either distillation or physical quality. A possible matched-duration
diagnostic must preserve the actual1000-step prefix, frozen Q/flow, optimizer
and random states; it must not be disguised as an online update.

On reviewing overlap with declared work, no additional duration production
is launched. Main frozen-flow BC already continues1000→1600→2200 actor updates,
preserving parent actor/Adam state and never updating its teacher or zero-weight
Q. Its same-eight-noise evaluations can test practical continued distillation
without new physical calls. Round-specific generator resets mean it is not
identical to uninterrupted2000-step round0 training. FQL's online rounds also
update Q from real feedback, so their trajectory is not a pure duration effect.
Inspect these existing BC curves before considering a separate duration test.
# Storage-only interruption recovery

The development pipelines encountered a per-user Ceph EDQUOT event. The
[recovery record](qstudy_quota_recovery.md) preserves exact cache migrations,
checkpoint audits, native-attempt accounting and replacement job identities.
Original model/optimizer/RNG states and scientific settings remain fixed;
infrastructure errors are not physical-failure labels. Main controller1797468
has revalidated all96 original round0 native outcomes and opened its original
96-request round1 feedback/evaluation matrix. Source and updated-flow missing
requests are recovering as1797488/1797489. Final detailed routing and reports
continue from the original plans. These are pending measurements, not a new
claim of PPA improvement.

## Independently verified recovery and saved-report figures

Read-only Slurm1797536 rechecked the behavior-flow data chain and the first
recovered native outcomes. Its immutable audit is
`qstudy_v1/recovery_teacher_audit_v1/20261018T122110Z_job1797536.json`, SHA256
`e4add24c43bf3eb515abd3e2338f25bdaa58e44183cfa491be865706bb539f70`.
At that snapshot, eight source-actor and four updated-flow recovery outcomes
passed the unchanged native checker. All81 original request/evidence bindings,
the15 protected successful source outcomes and six unsuccessful EDQUOT
transactions remained unchanged. The audit made no model or EDA calls.

The same audit independently confirms40 direct behavior-flow samples plus40
local variants, all80 native successes with seven metrics, and32/8 complete
families giving64 TRAIN/16 development-validation labels. The identical256
behavior targets remain separate. Valid data provenance does not reverse the
negative held-family Q-ranking result in [the source comparison](qstudy_source_results.md).

All six quota-interrupted GPU replacements completed successfully. The bounded
finisher1796852 verified all nine exported policies and opened its remaining
48 native requests, reusing the24 completed WL outcomes. Main round1,
source-actor and updated-flow recovery arrays are still in progress; no
partial physical matrix is presented as a completed policy result.

`analysis/qstudy_policy_curves.py` adds saved-report-only figures of all seven
GRT metrics across original training rounds. It recomputes each seed mean
from all eight native outcome rows, checks the common actual noise identities,
and reports the sample SD of the three training-seed means. Partial reports,
missing seeds, duplicate rounds, nonfinite metrics, conditional PPA and changed
saved means are rejected. Nine tests passed in Slurm1797551. A real completed
round0 report rendered successfully in1797553, with no inference, training or
backend calls; the PNG was visually checked. This checks the renderer, not
completion of the pending offline-to-online curves.

The prospective fresh protocol now also has a parser-reviewed
[launch checklist](qstudy_fresh_launch_checklist.md). No fresh selection or
85000000 noise arrays have been created. The default still requires completed
development review before fixing the candidate and starting its80 GRT/40 DRT
confirmation requests.

## Frozen-rule fresh acceptance reducer

`fql/qstudy_fresh_acceptance.py` now applies the exact previously committed
prospective rules to the final saved GRT/DRT reports. It performs no model,
training, backend or candidate-selection calls. All locked streams retain
their full request denominators, shared actual-noise and selection bindings;
candidate/BC means are recomputed per training seed. Unknown outcomes remain
inconclusive, confirmed failures never receive fabricated PPA, and a BC
physical failure cannot establish superiority from conditional WL. The
public reference remains one checkpoint rather than three trained replicas.

Independent production-schema review caught and fixed an implementation
error before use: native metric dictionaries retain extra parser/runtime
metadata. The reducer requires and validates the seven/six named numeric
metrics while accepting those auxiliary fields. It also keeps report outputs
outside immutable export/DRT evidence roots. Slurm1797591 passed all15 tests,
including native metadata, threshold boundaries, full-denominator failures,
shared identities and pinned-file tampering. Module SHA256 is
`774568d403ffc9a8b0b5fc3e4623df87e8253b9e42e5cd29793750e2882ec0be`.

The [launch checklist](qstudy_fresh_launch_checklist.md) includes the exact
saved-report reduction command. This is implementation and fixture evidence,
not a fresh physical result: development matrices are still running, no
candidate has been selected and no85000000 noise has been generated.

## Common TRAIN utility reference prepared

Slurm job 1797614 extracted the existing WL, timing and congestion utility
specs fitted on all 384 training rows. All nine original objective/seed
summaries were completed TRAIN-only refits; each objective's three saved specs
matched exactly, and all objectives shared the exact reference and scales.
The current fresh reporter successfully read the standalone inventory back
without refitting any scale or loading model/data tensors.

The immutable reference is
`qstudy_v1/fresh_common_utility_reference_v1/common_train_utilities.json`, SHA256
`55de0bcbbc9129384d01879f7d42eed1c59366f8f6488604214ac9cf1cb8500d`.
Its `completed.json`, SHA256
`9a06489efc8396d31207792dc604ef7e83c4d833d471342fc31a2c1dfec6175f`,
binds all nine source files and the extraction script. The scales remain
2368912.3724 um WL, .398183349 ns WNS violation, 206.822615 ns TNS violation,
80.5364583 horizontal overflow and 28.15625 vertical overflow. The already
declared weights remain 1/0/0/0/0, 1/.15/.35/0/0 and 1/.15/.35/.1/.1.
These normalized bounded utilities supplement the complete native metrics;
they do not redefine the predeclared physical-WL primary endpoint. No fresh
selection, noise, fitting, inference or backend measurement occurred.

## Complete behavior-flow source actor comparison

Source controller1797490 and strict reports1797504/1797505 completed. All48
new O64/T64 native outcomes succeeded; with the24 specifically certified
reused BC outcomes, all72 comparisons are valid, with identical saved noise
arrays and zero integrity errors. Independent read-only recomputation agrees
with all three seed means, sample SDs and paired native-metric differences.
The [source results](qstudy_source_results.md#completed-actor-results) retain
both immutable report paths and exact hashes, full seven-metric statistics
and the per-seed WL changes.

T64 mean WL is2.320758m versus O64 2.323754m and BC2.314028m. The .129% source
reduction is not stable: two seeds worsen and one improves; T64 is .291%
worse than BC overall. TNS and vertical overflow improve versus O64, while
WNS, horizontal overflow and area worsen. Neither source actor satisfies the
locked primary rule. This closes the declared source actor test without
promotion, a source-online extension or a new tuning grid. The valid data
pipeline does not itself establish a better critic or policy. These are GRT
development measurements, not final-route DRC results or fresh confirmation.

## Saved development-noise provenance

Read-only Slurm1797625 independently checked43 sealed request manifests and
652 rows, including the saved float32 noise bytes. Its immutable report is
`/fact_home/bohanli/flowplace_fql_recovery/development_noise_inventory_v1/snapshot_v1/report.json`,
SHA256 `2934f9cf99781c8d8534845f26a93dd70ceca975aefcc4edd0c7cdc52611b0d2`.
The five inspected noise pools contain4 arrays for900000,4 per design for
4300000,48 across the4500000 seed/round family,8 shared arrays for4600000,
and40 teacher-source arrays for5300000. Original900000 requests lacked a
recorded array fingerprint; saved NPZ contents supplied it without recreating
the RNG stream. Every newer recorded noise fingerprint matched saved bytes.

The652 rows are this explicit provenance scope, not a physical-call budget
or exhaustive historical-copy inventory. Unexported round2 streams remain
listed as not yet sealed and need their eventual strict report bindings.
No noise was generated and no model or EDA calls were made. None of these
inspected development pools is eligible to become fresh confirmation.

## Final saved-report postprocessing scheduled

Slurm1797689 waits for the complete three-arm report1797506, then renders
the existing seven-metric policy curves. Its wrapper accepts only the exact
snapshot named by that job's stdout, checks the completion/report hash and
all27 original arm/seed/round identities, and delegates full native-row and
noise validation to the already tested renderer. It performs no model or EDA
calls. Submission and wrapper bindings are retained at
`/scratch/bohanli/flowplace_graph/final_development_curves_v1/submission.json`;
wrapper SHA256 is `d95fa63351280ea78921851fa510170c39070b44517973f9d51c526a956da2c5`.
The output root is `qstudy_v1/actors_v1/flowfactor_curves_v1`.

Read-only metadata job1797663 resolved three older diagnostic snapshots from
their original job journals, verified the recorded hashes, and checked all
five final-report stdout paths against Slurm. The closure plan and Python3.9-
compatible resolver are now copied and pinned in
`qstudy_v1/development_closure_inventory_v1/`. Their SHA256 values are
`bc035b69d1d0afaf3dd1ad19207a5b77b337dd9fae7c70d753c292113da2ae40`
and `2bafd76d8ca74171213a85caea85a2deef8183c7a99507148bc8638e74a73b67`.
Closure metadata job1797740 waits for reports1797472/1797506/1796866/1797474/
1797369 and the existing saved-BC diagnostic1797476. It inventories their
saved metadata without selecting a candidate, changing any source, creating
noise, loading models or invoking the backend. An afterany dependency can
produce diagnostic metadata for incomplete work; the root still must inspect
each report's native success, full denominators, provenance and scientific
control eligibility. This submission is not proof that development or fresh
confirmation is complete.

## Cost snapshot and environment preflight scheduled

Scheduling-only journal `logs/qstudy_native_throttle_20261018T1312.json`
records active native arrays1797489/1797522 increasing from two to three
simultaneous workers after checking spare CPU/memory. Completed-child warnings
produced scontrol exit1; authoritative squeue confirmed both `%3` limits.
No physical input, eight-thread worker, metric, repair or sample budget changed.
Later controller arrays retain their original concurrency settings.

GPU metadata Slurm1797843 completed in7seconds on deep-space. It verified
NVIDIA L20, torch2.2.1+cu121, NumPy1.23.5 and CUDA12.1 under the existing fresh
precision/determinism contract, without loading a model, generating noise or
running EDA. The immutable record is
`/fact_home/bohanli/flowplace_fql_recovery/fresh_hardware_preflight_v1/job1797843.json`,
SHA256 `ab2fd3b6f39536f9868ce647c3fb76c697a5f1076899b610cc593f0a059fe95f`.
This establishes the allocated preflight environment, not candidate selection
or equivalence of earlier GPU numerical runs.

Read-only cost job1797857 now waits for the final development reports. Its
reviewed helper snapshot/submission is in
`/fact_home/bohanli/flowplace_fql_recovery/qstudy_cost_tool_v1/`.
The reducer SHA256 is
`35027ff88e2d486d7484cf132522ca16ebf3db42d55253fc183e10abc534d9ee`;
the explicit scope SHA256 is
`185d8698aec958ca2bb4533400dc8a7f5a304a3f7e530cab3fb42b1c571e3316`.
It binds inventory1797369's exact stdout snapshot, builds a Qstudy-only job
whitelist and invokes the existing allocation collector with `--jobs-file`.
Both failed and recovered GPU attempts are retained; unrelated projects and
paused Table2 are excluded. Allocated resource-hours are not utilization.
Saved native stage times and committed optimizer budgets remain separate from
last-successful-attempt training duration. Isolated inference latency and
unobserved shared/public pretraining costs remain unknown. A later post-fresh
cost snapshot needs explicit fresh/preservation job IDs; this scheduled
development snapshot is not an exhaustive final total.

## Completed bounded-objective native comparison

Report1796866 completed the full96-request matrix:72 new bounded actor
measurements and24 unchanged BC references, all successful with zero missing,
failed or invalid requests. Its immutable report is
`qstudy_v1/bounded_actor_native_report_v1/20261018T132750.979106Z_job1796866/report.json`,
SHA256 `0ce81b4ef635d0d6def02d751a809a3459bc1ed23d6003dd02167bae63d4ddfb`.
The [full result](qstudy_bounded_objective_protocol.md#completed-development-result)
reports all seven metrics, three-seed sample SDs, paired differences and
control limitations.

No bounded arm qualifies for promotion. Bounded WL lowers mean WL0.422%
against the existing BC reference, below the locked1% threshold. Timing lowers
WL1.157%, with all three seeds lower, but TNS violation increases13.642ns
against an allowed10.870ns and H overflow increases3.667 against an allowed1.
Timing+congestion lowers WL0.455% with two seeds worse. The existing BC has no
bounded zero-Q effective-update certificate, so these BC comparisons remain
descriptive; timing's secondary-metric failures independently remain.

Against the bounded-WL actor, timing lowers WL0.738% but worsens TNS/H/V and
its own measured timing utility in all three seeds. Timing and congestion
critics also have negative mean ranking on their own24 actor proposals.
These are conditional development results, not a general rejection of
multiobjective rewards. This closes the bounded branch without more alpha,
capacity, objective-weight or online searches. Main matched online and
final-route comparisons remain pending.

## Round2 execution and storage preservation follow-up

All nine declared round2 actor-training and request-export jobs completed.
No additional policy training was added. The temporary GPU recovery monitor
v5 was stopped after checking all nine immutable completions; its final
session timestamp is2026-10-18T13:49:17.666000Z. Native arrays1797894 and1797903
continue the already declared updated-flow and matched-control evaluations.
Scheduling-only journals `logs/qstudy_native_throttle_20261018T1339_flowr2.json`
and `logs/qstudy_native_throttle_20261018T1347_mainr2.json` record increases
from2 to4 and4 to8 simultaneous workers respectively. Each worker still uses
eight threads, the same physical inputs/settings and the same sample budget.
Authoritative scheduler inspection confirmed the limits; the main-array
scontrol return included warnings for completed children.

The new compact native-preservation module was tested on one actual GRT
outcome and one actual detailed-route outcome in Slurm1797908. Full decoding
and per-file SHA checks passed for118 files:59,739,779 source bytes became
7,492,013 archive bytes. This was a scratch-only two-outcome pilot, not the
complete development archive or an admission to durable storage. Its stage
archive SHA256 is
`fda372b02e545f37104a1b05572977ba2b3f721d87e2cf82ee747158b978d9a9`.
The full development payload must still be measured and independently admitted;
the pilot compression ratio is not a budget estimate for that larger scope.

Storage-only Slurm1798008 compacted15 closed historical graph-policy archives
outside the original74-run repacking plan. All15 decoded inventories and exact
original compressed SHAs had passed probe1797951. The measured archive change
is949,457,982 to294,712,113 bytes, releasing654,745,869 bytes. Separate audit
1798022 rechecked compact and original-path hashes, full decoded file contents,
cache-link targets, original native manifests/outcomes/inventories and the
unchanged original74-run plan/manifest. All checks passed. The new completion
manifest is
`/fact_data/bohanli/flowplace_fql/graph_o2o_archive_repack15_v1/manifest.json`,
SHA256 `111797d8af6a8b3e92ada66842d449b68d88b0e914b2b05a037d361f97af1494`.
Independent verification is
`/fact_home/bohanli/flowplace_fql_recovery/storage_candidate_audit_v1/verification_job1798022.json`,
SHA256 `ac3b68ce187db64d625384c83b7e7404d3af535315b1461c952d77eb9090018d`.

The original compressed paths remain readable as exact-byte scratch cache
links; durable compact archives and the pinned-encoder restore procedure
preserve their contents. The separate admission adapter checks measured sizes
and SHAs, reserves128MiB and refuses unknown destination artifacts. It changes
neither the global EDA storage guard nor metrics, models, original provenance
or held Table2 files. Observed account headroom rose from294,242,503 to
948,183,854 bytes; the difference from archive-byte savings reflects metadata
and concurrent account activity, so these are reported separately.

## Saved BC continuation result

Read-only Slurm1797476 completed the predeclared BC1000→1600→2200 diagnostic:
all72 saved evaluations succeeded, all nine seed/round streams were eligible,
and no integrity errors, new model forwards, training updates, replay rows or
backend calls occurred. The immutable report is
`qstudy_v1/bc_distillation_rounds_v1/20261018T141052Z_job1797476/report.json`,
SHA256 `19172e727ea1a2943b62d84b5a08c0239d54e51218b7e13feff9ce60996e442d`.
Its round0 geometry exactly reproduces report1797105.

| Cumulative actor updates | Clipped raw RMS to teacher | Saved executed RMS to teacher |
| --- | ---: | ---: |
| 1000 | .100037 ± .004179 | .143660 ± .010784 |
| 1600 | .092280 ± .004426 | .130776 ± .014759 |
| 2200 | .084646 ± .003619 | .131449 ± .010207 |

These are means and sample SDs of three training-seed means, each using the
same eight actual evaluation noises. Raw distance decreases in all three
seeds at both continuations. Executed distance is lower than round0 in all
three final seed means, but its aggregate slightly rebounds after1600 updates.

| Updates | GRT WL (um) | WNS (ns) | TNS (ns) | H overflow | V overflow | Power (W) | Std-cell area (um2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 2314028.167 | −.403605 | −217.399438 | 7.666667 | 1.375000 | .170009542 | 61200.241667 |
| 1600 | 2322382.667 | −.415692 | −220.214458 | 11.208333 | 1.458333 | .170145417 | 61317.437500 |
| 2200 | 2312631.750 | −.428621 | −224.646958 | 35.083333 | 6.500000 | .169987208 | 61108.983333 |

Closer teacher coordinates do not produce stable physical improvement here:
mean WL is nonmonotonic, final WL improves over1000 updates in only one seed,
and BC remains worse than its paired teacher in all three final seed means.
Final WNS/TNS and H/V overflow worsen; power and area are slightly lower.
These successful GRT outcomes do not establish final DRC0 or timing closure.

Saved Adam counters verify actual1000/1600/2200 updates with parent optimizer
state, unchanged common flow/initial Q, and empty flow/Q optimizer states.
Feedback remains provenance-bound in the online parent chain, but Q weight0
and disabled flow updates make these BC rounds continued teacher distillation.
Named generators reset each round; this is not uninterrupted2200-step training.
Saved raw coordinates are clipped, while training MSE uses unclipped outputs.
The stochastic logged MSE endpoints also do not decrease in every seed.
Neither convergence nor legalization as the cause of the physical gap follows.
Close this diagnostic without launching the optional duration study.

## Declared development matrices completed, 2026-10-18 15:17 UTC

The earlier pending-status notes above record their observation times. Final
main, source, bounded and flow-update GRT reports are now complete. Frozen
FQL round2 lowers mean GRT WL0.804% versus matched BC, improving two of three
seed means; updated flow lowers0.628%, also two of three, and fails vertical
overflow tolerance. Neither meets the prospective primary rule. Source and
bounded branches also fail their promotion rules, so none replaces the legacy
anchor. See the [decisions](qstudy_decisions.md) for each hypothesis and limit.

Final detailed-route report1797474 completes48/48 requests, all DRC0. On the
fixed first4 noises, round2 frozen FQL mean WL is0.114483% higher than BC;
mean WNS/TNS/power/area improve within the declared tolerances. It fails the
final-route WL gate, and negative slacks remain. The [complete result table](qstudy_online_results.md#completed-final-detailed-routing)
retains all three seeds, per-metric variation, paired differences and report
identity. GRT all8 versus DRT first4 does not isolate a routing-stage effect.

Final call inventory1797369 accounts648 new successful logical GRT outcomes,
48 inherited representations,720 observed GRT attempts and48 successful DRT
requests. The [cost inventory](qstudy_cost_inventory.md) separates retries,
640 prior unique labels and scoped development allocations of369.720833
CPU-hours/3.376111 GPU-hours. The saved-accounting alias correction changes
coverage display, not allocation deduplication or resource totals.

Independent requirement review
`/fact_home/bohanli/flowplace_fql_recovery/goal_requirement_crosswalk_v2/development_addendum.json`
(SHA256 `83845afefc560172da037e73fc0014f62240c8e24a03aebd50d3e3de448cc0ec`)
supports closing the declared scientific exploration without blind parameter
expansion. Alpha3/30 remain analytic gradient diagnostics, not trained
policies; update-ratio optimality, a causal legalizer explanation, multi-seed
cross-design gains and isolated inference latency are unproved. Durable
preservation and unchanged-rule fresh confirmation are still required;
this section does not claim the Goal is complete or create fresh noise.
## 2026-10-18 16:02 UTC: preservation admission and reward-literature follow-up

The completed development proof stage1798247 and independent read-only
review1798270 passed. After reviewing the exact stage and budget, root
submitted preservation admission1798471: COMPLETED0:0, elapsed56s. The
durable manifest is
`/fact_data/bohanli/flowplace_fql/qstudy_v1/native_evidence_preservation_v1/batch_development_zstd19/manifest.json`,
SHA256 `df630cc6c4d190f30beff99f139cb4e3d8c31256bad345d533354f6d119fd2ed`,
status `native_evidence_durably_preserved_and_fully_decoded`.
It binds648 successful new GRT records,48 completed DRT records and772
native-attempt references;47,204 proof files/8,911,516,419 source bytes are
represented by a200,764,767-byte archive. The archive is selected proof,
not a complete EDA resume backup. Original HOME80 evidence, model/optimizer
weights, and unfinished-run scratch remain preserved. Admission changed no
global EDA guard and made no EDA call. Immediate post-copy quota remaining
was733,967,387 bytes. The256MiB future-proof reserve is a spending cap, not
proof of future sufficiency. Independent agent metadata review checked the
durable references/status/counts without repeating the archive decode.

The user then requested learning from rewards in other offline RL methods.
Primary papers and official code were checked for FQL, IQL, AWAC, TD3+BC,
COptiDICE and related placement work. The
[reward-literature addendum](qstudy_literature.md#10-2026-10-18其他-offline-rl-怎样使用-reward)
separates environment reward from actor/critic regularizers, and proposes
measured-quality weighting, separate constraint budgets, and matched
GRT/DRT label-fidelity diagnostics. These are new research suggestions,
not completed training or proof of an improved method. Previously observed
development DRT remains development evidence and has not entered Q/replay.
No new candidate selection, fresh noise, or confirmation measurement was
created. Review this user-directed extension before locking final scope;
the completed original matrices and their negative/mixed conclusions remain
unchanged. Admission1798471 must be included in the later cost addendum.

## 2026-10-18 16:19 UTC: reward-informed paired flow control

Completed a saved-only, strict same-layout stage-fidelity audit1798483,
CPU2/8GiB/32s. All48 planned GRT/DRT pairs resolve to the same source macro
geometry. WL pooled/within-stratum rank correlations are0.9954/0.9833, and
23/24 same-noise BC/FQL contrasts agree in direction. WNS is0.5855/0.3667,
with16/24 direction agreement. Stage absolute metrics remain different;
these are dependent development observations, not48 independent trials.
No DRT labels entered training. See [full audit](qstudy_stage_fidelity.md).
Its exact400-row measured dataset audit also confirms384TRAIN/16validation,
192/8 nonoverlapping families and three matching independent common sources.

The fixed [quality-weighted flow protocol](qstudy_quality_weighted_flow_protocol.md)
was written before training. Both arms use the same384 measured TRAIN raw
targets, not the separate256 mixed/unmeasured behavior targets. Rank weights
use TRAIN GRT WL only, retain all384 layouts, have mean1 and max/min ratio4;
effective sample size332.171. Targets remain raw; their labels describe the
unchanged legalization/DEF/native-snapping execution. No critic, one-step actor,
Q gradient, validation-based choice or online replay is added to this control.

Ten meaningful CPU tests1798486 pass, including exact interrupted-resume
parameters/Adam/RNG and paired data/noise/time streams. Root static review and
an independent data-audit review found no blocker. Initial CPU proposal1798485
captured the earlier validated source just before additional input-identity
guards were added. It is retained as prepared-only and never trained. After
the final source freeze and tests, CPU1798487 prepared the active proposal:
`/scratch/bohanli/flowplace_graph/qstudy_quality_weighted_flow_v2/plan.json`,
SHA256 `a7de7c7968ac13092a9b854ff723eb1aba8b5656df39d36b57439ebc30147640`.
No dataset, hyperparameter or scientific protocol changed between proposals.

Root submitted six fixed1000-step flow trainings1798489–1798494, two GPU
dependency lanes at most: uniform seeds0/1/2 and quality_rank seeds0/1/2.
Each requests one L20,2CPUs,20GiB,20min with checkpoint/restart support;
natural-selection, enterprise and blue-space are excluded. Final full and
compact weights stay in scratch. The saved submission manifest records
all commands and dependencies. Planned development evaluation is48 new
eight-thread GRT measurements, using the existing actual development noises;
none has been submitted at this entry. No fresh noise was created. The
extension must be closed and cost/preservation supplemented before fresh
confirmation or a claim of a stable advantage.

## 2026-10-18 16:58 UTC: expert-rule matrix and new final labels

The user explicitly requested rewards beyond GRT, including placement
expertise. The [fixed eight-objective protocol](qstudy_rule_reward_protocol.md)
retains FQL, the same teacher and one-step actor, the same384 TRAIN proposals
and executed-label mapping, the same legalizer, and three independent training
seeds. It contrasts GRT-WL, macro-HPWL, four individual expert terms, their
HPWL composite and their GRT composite. Rules encourage edge clearance,
group compactness, shorter structural dataflow connections, or fewer narrow
facing channels. They are testable engineering proxies, not proven PPA
improvements. I/O rules are excluded because the cached construction is not
verified. The [primary-source review](qstudy_rule_reward_literature.md)
distinguishes these adaptations from EXPlace, RTL-MP and constrained offline RL.

The corrected TRAIN geometry/source audit1798614 is
`/fact_home/bohanli/flowplace_fql_recovery/qstudy_rule_audit_v3/20261018T164331.707916Z_job1798614/report.json`,
SHA256 `8d9e263be3cf306bdb4ddbdb29e09247f7eda920c4be826b83e05d7cd18c1588`.
Earlier audit continuation inventories used the config-parent nickname
`bp_fe_top` instead of actual result nickname`bp_fe`; preserve them as
superseded inventory reports, not evidence that source checkpoints are absent.
Geometry-rule formulas did not change. Six audit tests1798582 passed.

Critic preparation1798619 and array1798626 completed all24 objective/seed
tasks and96 fits. Thirteen screen tests1798615 passed. Saved-only report
1798710 (seven seconds,2CPUs/8GiB) verified96 fits, zero integrity errors and
all12 cross-objective fold/seed scale/RNG endpoints. Report:
`/fact_home/bohanli/flowplace_fql_recovery/qstudy_rule_reward_results_v1/20261018T165404.256751Z_job1798710/report.json`,
SHA256 `3270821e2e0f416ed10c2df2a4fb87ae2f49500bcda05da5dd86eb47efecab16`.
CV Spearman means are .561(GRT), .788(macroHPWL), .553(edge), .786(group),
.706(dataflow), .583(channel), .605(HPWL composite), .477(GRT composite).
Targets differ, so these numbers neither rank physical policy quality nor
filter the fixed eight actors. Group bounding boxes share their exact largest
value in351/384 layouts (91.4%); this rule has limited variation. All384 raw
proposals are moved by execution; mean movable-coordinate RMS is .08070.
These limitations remain visible rather than being used to change the recipe.

Actor adapter tests1798613 passed all eight checks, including actual tiny
interrupted/resumed actor updates and unchanged teacher/critic weights.
Independent static review passed. Preparation1798689 completed, checking all
24 full-TRAIN refits and the original256 behavior targets separately from
the384 measured labels. Twenty-four fixed offline actor/export jobs
1798712–1798735 were submitted in two dependency lanes on einstein, each one
L20/2CPUs/20GiB/20min. No online round or backend call is included in this
stage. Submission/source records are in
`/scratch/bohanli/flowplace_graph/qstudy_rule_reward_actors_v1/`.
All192 development GRT measurements will be dispatched only after the24
exports complete; same-noise contrasts use the new GRT and HPWL controls.
Old BC is not automatically an equivalent control.

Separately, new TRAIN64 detailed-route plan1798620 fixes32 original TRAIN
families, two variants each, selected by SHA within four GRT-quality strata.
It retains four family-CV folds. No old validation16, development DRT48 or
fresh layout enters this collection. All64 original native GRT sources and
the pinned container match; no archive restore was needed at preparation.
Plan`/scratch/bohanli/flowplace_graph/qstudy_train_drt64_v1/plan.json`, SHA256
`18b109579d8076ee7f65afb84481b7c270a7edc101fdbb67d28216802c7c101f`.
Fifteen collector tests1798616 passed. Array1798635 uses eight CPUs/12GiB,
two workers, unchanged detailed-route repair cap20. First attempts genuinely
entered detailed_route, then were preempted before a resumable DRT database
was written. Preserve infrastructure outcomes and recover exact inputs;
do not replace layouts or assign them physical-failure labels. This collection
will provide final WL/WNS/TNS/power/area/DRC labels for a separate matched
final-feedback reward study, not evidence of improvement on its own.

The auxiliary weighted-flow control completed all six trainings and exports.
Three quality-rank jobs1798492–1798494 failed before training because the
exact software-build contract differed on another node; retries
1798496–1798498 on einstein completed unchanged. All failures and retries
remain in costs. Controller1798585 dispatches48 native GRT requests in
array1798587, at concurrency4. Its pending time limit was reduced to2h before
startup, without changing physical parameters. No weighted-policy result is
claimed yet. This WL-only FM control does not replace the expert/final-PPA
reward experiments. All new evidence still requires preservation and a cost
addendum before fresh confirmation.

## 2026-10-18 17:20 UTC: final-feedback protocol and execution adapters

The [final-feedback protocol](qstudy_final_feedback_protocol.md), SHA256
`ecb148af55e0e491f4016a7e14ed900cc4e295c087263aa1c194c0f738b28e20`,
was frozen before reading new TRAIN64 PPA values. It compares six objectives
on identical labels: GRT WL; detailed-route WL; .75WL+.25TNS-violation;
.75WL+.25power; .50WL+.25TNS+.15power+.10area; and .80WL+.05 for each
of the four fixed expert costs. Terms use fit-TRAIN feasible means before
the same bounded cost2C/(1+C), and Q stays in negative bounded-cost units.
WNS is separately reported, avoiding two correlated timing penalties.
Completed DRC-positive layouts retain measured PPA but receive common B4
failure utility in all six targets. Unresolved infrastructure is NA, not B4.
No native feedback is added as a critic input.

The screen runs four family-CV folds and one explicitly in-sample64-row refit
per target/seed:18 tasks and90 fits. No historical16 validation layouts,
old48 development DRT results or fresh layouts enter training. The actor
keeps the original256 BC targets distinct from64 critic labels and the
original384-row physical context. It retains common frozen teachers,
Euler10 distillation,1000 actor updates and alpha10, with offline round0
only. The declared physical comparison is144 GRT layouts, then the same
first four noises for all18 policies:72 cap20 detailed routes. This is a
fixed six-way contrast, not selection on CV errors or favorable layouts.

Meaningful software tests passed: final screen14 tests1798802, final actor
13 tests1798778, saved-only power QA15 tests1798808, and route metadata
10 tests1798814. Reciprocal independent source reviews found no blocker.
Strict admission requires a complete audited64-label collection and saved
power/condition QA before preparing the screen. Power QA compares captured
image/tool/config/clock semantics, native raw/text power and available
internal/switching/leakage components. Present-day uncaptured library hashes
are not retroactive runtime attestations. No VCD/SAIF annotation is established;
power is reported as a tool-default-activity estimate, not measured workload
or silicon power. No actual final-feedback screen, actor or72-route evaluation
has been submitted at this entry.

Collector array1798635 has genuine completed detailed routes and genuine
preempted allocations. Bounded recovery1798748 is limited to indices0/1/2,
partition/QOS dev, afterany of the original array, max two workers, with the
same eight threads/12GiB/cap20 and exact source inputs. It is not a large dev
sweep. Later preemptions remain visible; do not classify them as bad layouts.

Expert actor jobs1798712–1798735 were all still unallocated when a scheduling
revision was recorded in
`qstudy_rule_reward_actors_v1/pending_scheduler_move_deep_space_v2.json`.
All24 were held, moved together from einstein to the same L20 node deep-space,
given12GiB rather than20GiB host RAM, then released with the original two
dependency lanes and scientific settings unchanged. The original queue's
backfill estimates were17–19 hours away. Probe1798767 verified identical
Torch2.2.1+cu121/git/CUDA12.1/NumPy1.23.5; destination Python is3.9.21,
versus3.9.25 on einstein. This actor protocol does not pin the exact Python
patch version. Host RAM is a resource request, not a measured peak claim.
The first migration attempt used rejected memory syntax before changing any
job; its complete command/error record remains as v1. No job was restarted.

Native GRT controller1798773 depends on successful completion of both final
actor dependency lanes1798734/1798735. All24 exports must already exist,
so the generic controller dispatches backend work only, not its old actor
entry. Fixed192 requests, GPU concurrency allowance2 and backend concurrency4
do not alter the predeclared policy/noise matrix. The saved-only expert actor
reporter passed14 tests1798754 and is ready after native outcomes close.

Cost metadata v5 preserves all116 explicit unique parent jobs, failed software
attempts included, and does not count sbatch admission-only1798740 as an
allocation. Its SHA256 is
`815ff21f6767ad87b61aa97426e4bf37ea45d99a951b08042e265381eab4f2ec`.
Many jobs remain pending/active; there is no final cost or preservation claim
for this extension and fresh confirmation remains on hold.

## 2026-10-18 17:44 UTC: weighted-flow result and overlapped native scheduling

Saved-only reader1798853 completed all48 new uniform/quality-rank GRT outcomes,
plus24 exactly bound existing common-flow references. Report:
`/scratch/bohanli/flowplace_graph/qstudy_quality_weighted_results_v1/20261018T173934.466674Z_job1798853/report.json`,
SHA256 `2b512668a08485e335b65d90d1db009b2691368832ddbb99f31610dad9d7a967`.
All three arms have24/24 successful requests, with no missing/invalid/native
failures. These are fixed eight development noises per independent seed,
not24 independent training repetitions. Values below are three-seed means;
WL uncertainty is sample SD of the three seed means.

| Multi-step flow | GRT WL (m), mean ± seed SD | WNS (ns) | TNS (ns) | H overflow | V overflow | Power (W) | Std-cell area (um2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform measured-source continuation | 2.356046 ± .011625 | −.413601 | −224.454 | 63.250 | 13.583 | .169646 | 60421.9 |
| Quality-rank continuation | 2.331569 ± .006352 | −.403168 | −200.282 | 48.000 | 9.125 | .169510 | 60698.8 |
| Unchanged common behavior flow | 2.279086 ± .013390 | −.406557 | −206.695 | 22.125 | 7.125 | .169797 | 60997.7 |

Weighting versus uniform lowers mean WL1.038887%; per-seed paired WL changes
are−27292.375/−18346.625/−27791.000um. Mean WNS/TNS and both overflows improve;
area increases0.458%, within the prespecified1% tolerance. The complete48
development comparison passes its declared engineering gates. It does not
beat common-flow WL: the weighted mean remains about2.30% higher. This is
evidence about quality weighting under fixed source adaptation, without Q
optimization or a one-step actor. It is not a new FQL win, final DRC proof or
statistical-significance claim. Continue the protocol's24 detailed routes,
both arms/all three seeds/indices0..3, cap20; no temperature or source sweep.

Two reporting-only faults were fixed without changing frozen producers,
checkpoints, declarations or native data.1798825 exposed extra observer imports
in the producer's dynamic source inventory; isolated replay now verifies the
exact captured working sources and snapshots, while binding observer sources
separately.1798844 then exposed a literal-vs-canonical seed0 checkpoint alias;
the actual path/inode/SHA are identical. The reader now resolves and rehashes
that identity, still rejecting a different equal-byte file or a retargeted
symlink.18 targeted tests1798850 and independent reviews pass. All failed
attempts remain in accounting. No EDA was repeated to repair reporting.

Scheduling-only revision overlaps the expert-rule actors with their192 native
GRT measurements. Per-arm controllers1798855–1798862 each wait for all three
corresponding rule-specific exports, use explicit round0 and two backend
workers, and form two afterok dependency lanes. The original combined1798773
was held before submissions, made dependent on both lane tails1798861/1798862
and final actors1798734/1798735, then released. It will validate/reuse those
same outcomes. Total maximum GRT-worker concurrency stays4; there are no new
layouts, seeds or backend settings. Journal:
`qstudy_rule_reward_actors_v1/native_pipeline_submissions_v2.json`.
The first scheduling attempt was refused because completed Slurm job IDs had
expired as new dependency prerequisites. No first-attempt controller allocated;
the v2 journal verifies completed actor accounting plus export-marker hashes
before omitting only already-satisfied dependencies. The refused admission-only
test is not an allocated job. Frozen scientific sources remain unchanged.

TRAIN64 recovery coordinator1798873 is queued afterany original1798635 and
bounded dev0121798748. It audits all64 exact inputs and retries only missing,
confirmed infrastructure-interrupted rows, never completed routes or closed
native failures. Three generations at most, two8CPU/12GiB/cap20 workers.
18 software tests1798869 pass, including real compressed-array accounting,
live-writer exclusion, source/spec tampering and interrupted-submit adoption.
Redundant late worker dependencies are omitted after the coordinator's terminal
gates, avoiding the same expired-ID rejection; gates and physical budget stay
unchanged. This changes recovery scheduling, not final-feedback training labels.

## 2026-10-18 17:52 UTC: conditional weighted-flow detailed routing launched

The weighted-flow GRT gate passed before routing was admitted. The new metadata
adapter passed nine final-source fixtures1798945 and root/independent review;
the earlier nine-fixture1798931 result is retained as a separate software cost.
It binds the actual six trained flow checkpoints and Euler10 exports, rather
than assuming a one-step actor chain. It changes neither the original native
route worker nor its final-result inspector.

Plan1798965 fixes24 development routes: uniform/quality-rank, all three training
seeds, indices0..3, cap20, eight backend threads and at most two workers.
`/scratch/bohanli/flowplace_graph/qstudy_quality_weighted_route_v1/full_route_plan.json`
has SHA256 `167954ce9d9e8b223749e29c8bf1b7e52b81f71291e53b6037825393d40d9eae`.
Resolution1798975 independently revalidated all48 GRT outcomes and sealed24
source bindings with actual request, noise, raw/executed coordinates and native
source identities; all24 resolved. Its report SHA256 is
`621fd191c1f0c1ff08d44d5d39ec268013d5d0ab2e133db8990763f30e6d713b`.
Unchanged native controller1798984 is submitted; saved-result report1798989
depends on its success. Nonzero DRC retains measured PPA, and missing/infra
outcomes remain nonnumeric. No training/replay or fresh confirmation consumes
these development layouts.

Expert-rule saved-result job1798978 is also queued after combined native
controller1798773. It will report the full192-request matrix with all eight
objectives and three seeds; queued reporting is not evidence of physical gains.
Cost inventory v8 preserves all140 v7 parents and adds exactly two native GRT
arrays1798864/1798865 and two weighted-route software tests1798931/1798945,
for144 unique parents. SHA256 is
`58bb61393cfd861256a7eaf75518ef3777faf9b483e02a4d3eb35dd7bfa0ed69`.
The newer plan/resolve/controllers/report jobs still need the next cost append.

## 2026-10-18 18:15 UTC: broader rewards remain in flight; final-label QA queued

The eight frozen expert-rule objectives retain matched three-seed one-step
FQL actors, teacher, training streams and native evaluation. At this snapshot
16/24 actors have completed, two are running and six remain dependency-pending.
The first two native GRT arrays have40 completed tasks; this is not yet the
complete192-request result and no rule is selected from partial outcomes.
TRAIN64 final routing has19 completed tasks and six infrastructure-preempted
tasks; the previously queued bounded recovery chain retains all64 identities.
The weighted-flow native controller has completed three of its24 declared
routes, including a twice-preempted request recovered without changing inputs.
No partial native means are used to change weights or training budgets.

After root and independent schema review, external handoff1799254 is queued
after recovery1798873. It requires that exact coordinator to prove64/64 completed
original-TRAIN results and a complete hashed collector report, saves its handoff
receipt, then invokes unchanged `analysis.qstudy_final_power_audit`. It neither
trains nor runs EDA. A successful scheduler dependency is not enough: unresolved
recovery is rejected, and a handoff is not power admission. The six final-feedback
objectives still require actual native power QA and every fold's variation gate.
Metadata handoff1799280 is subsequently queued after QA1799254. It accepts only
that job's verified64-row admission snapshot and then invokes unchanged
`fql.qstudy_final_feedback_screen prepare`. Its output is the fixed eighteen-task,
ninety-fit plan only if the original admission and variance checks pass; no
training, fallback target or physical job is submitted by this handoff.

Read-only preservation audit1799010 and a scratch-only exact compression
pilot1799170 distinguish inference state from full resumable checkpoints.
The failed first pilot1799109 is retained in costs; no approximate tensor
conversion was used. Closed-cache audit1799239, restricted to the fixed211
old-run inventory, found no safely reclaimable large duplicate directory or
prearchive. All201 closed native runs already link their raw caches to scratch;
the remaining durable archives are protected. Global `/fact_data/bohanli`
headroom is705,807,110 bytes, below the intended8GiB reserve. It has not been
restored. New extension outputs remain on scratch; no bulk preservation or
complete durability claim is made and no unique artifact was deleted.

Cost inventory v10 preserves the previous152 entries and adds exactly seven
proven parent jobs, for159. Its SHA256 is
`459c6929b1cde932c38f926b1134a0af42021bb3fda9fdc76062ccbce5660da6`.
It includes the failed compression pilot and QA handoff, without treating array
task aliases as extra parents. Ongoing native submissions still require later
appends; this is not a final extension cost total. Fresh confirmation stays held.
