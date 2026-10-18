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
