# Behavior-flow labels: measured source control

The behavior flow can generate layouts that can be measured by the real
backend and used to train Q. That data path has completed. It does not by
itself imply that the resulting Q generalizes or supplies useful gradients.

The current seed0 behavior flow generated40 independent layout bases, each
paired with one local perturbation. All80 requests completed with genuine
OpenROAD global-route measurements and all seven physical metrics. There
are64 TRAIN labels and16 validation labels, grouped by32/8 layout families.
These are40 direct flow samples plus40 local variants, not80 independent
flow samples. Three critic seeds do not create three independent teacher
distributions. The released20-step sampler and the current10-step behavior
flow differ in both checkpoint and sampling procedure.

All36 declared source tasks completed, with three grouped CV fits and one
full-TRAIN refit per task. The immutable report is:

`/fact_data/bohanli/flowplace_fql/qstudy_v1/source_controls_v1/report/20261018T093324Z_3422023`

Its summary SHA256 is
`858083a57a38dee6636f682277b2e89b7d634d7372361f85965a7427b541ed6d`.
There are zero integrity errors. Each comparison keeps the identical256 BC
targets. O/T/M mean old labels/current-flow labels/half of each, at64 TRAIN
labels. Both validation pools are identical across arms and are reported
separately.

| Critic | Source | WL Spearman, old16 | WL Spearman, current-flow16 | WL MAE, current-flow16 (um) |
| --- | --- | --- | --- | --- |
| Legacy | O64 | 0.760 +/- 0.027 | 0.500 +/- 0.091 | 89,693 |
| Legacy | T64 | 0.575 +/- 0.047 | -0.346 +/- 0.086 | 117,509 |
| Legacy | M64 | 0.839 +/- 0.093 | -0.520 +/- 0.093 | 172,341 |
| Huber + HPWL residual | O64 | 0.778 +/- 0.031 | 0.310 +/- 0.198 | 99,086 |
| Huber + HPWL residual | T64 | 0.458 +/- 0.076 | -0.204 +/- 0.012 | 117,749 |
| Huber + HPWL residual | M64 | 0.789 +/- 0.062 | -0.383 +/- 0.054 | 175,391 |

Values are mean and sample SD over critic seeds0/1/2 on the same development
layouts. Their variation is training randomness, not uncertainty over new
datasets. Replacing O64 with T64 reduces WL rank correlation on teacher16
in every seed for both critic variants. There is no winning-source claim.

The base-only32-label sensitivity also has weak current-flow validation
ranking: T32 legacy -0.056 +/- 0.084 and residual -0.118 +/- 0.090 over16
validation rows. Restricting validation to the eight direct-flow bases does
not rescue T32 ranking (legacy -0.040 +/- 0.096, residual -0.119 +/- 0.082).
Thus the different perturbation procedures alone do not explain the result;
the smaller32-label sensitivity is a separate budget, not a matched64-label
treatment effect.

Other metrics are mixed. The residual critic's teacher16 horizontal-overflow
correlation improves from -0.159 to0.371 when using T64, but TNS correlation
falls from0.390 to0.143 and its MAE rises from40.8 to115.7ns. WNS ranking stays
near zero. Legacy timing/congestion predictions are absent, not zero.

Within-source grouped CV likewise shows weak T64 WL learning. Its legacy
OOF correlation is -0.170 +/- 0.017 and residual is -0.039 +/- 0.056. O64
old-source OOF correlations are0.344 +/- 0.022 and0.286 +/- 0.017. These
OOF target pools differ; this is supporting diagnosis, not a paired source
treatment comparison.

## Actor contrast: preparation and comparison

The dated review in `qstudy_source_actor_review_20261018.json` selects only
O64 versus T64 with the same legacy critic. This is a bounded negative-
hypothesis test, not promotion. Both use alpha10, source-corrected Q scaling
to a common O64 physical utility, the same three common initializations,
frozen behavior flow,1000 actor updates, and all eight predetermined
development noises per training seed. Legacy source actors are initially
offline-only because their frozen online implementation has source-dependent
failure scaling. Neither actor prediction nor a lower loss establishes PPA
improvement; every requested layout needs measured backend evaluation.

The original bp_fe384/bp_be64 direction probes completed52/52 native requests
on each design, with negative/mixed direction evidence. They do not establish
useful directions for these source-specific64-label refits. No M64 actor,
stronger Q weight, new training-source mixture or online source promotion
follows automatically.
Existing matched BC results may be reused only after certifying effective
BC training inputs and streams, rather than equating different dataset file
hashes. All current layout pools remain development evidence.

The primary physical comparison is O64 versus T64 across all requested
evaluation layouts, first reporting confirmed failure rate and then per-seed
mean GRT WL, with three-seed mean/sample SD. Missing infrastructure outcomes
remain unresolved, not physical failures. If physical failures occur, do not
declare an improvement from successful-layout WL alone. Timing, horizontal
and vertical overflow, power and area accompany WL; unavailable metrics stay
missing. Compare the actual saved noise arrays/hashes before pairing samples.
There are24 requested layouts per arm (3 seeds x8 noises), with no best-layout
selection, early checkpoint selection or favorable-seed exclusion.

This contrast changes the whole source/neighborhood recipe: old Gaussian
perturbations and current-flow fixed-RMS perturbations differ. It cannot
isolate the teacher checkpoint or integration procedure by itself.

## Completed actor results

All 48 requested O64/T64 outcomes completed successfully. Source controller
1797490 and both strict native reporters completed with zero integrity errors.
The common-reference report reuses 24 original BC round0 outcomes under the
specific effective-update certificate; it does not rerun BC or claim a new
bitwise GPU repeat. All nine arm/seed streams and three paired contrasts are
complete. Each pair shares the actual saved noise bytes, seed and index.

Authoritative snapshots under `qstudy_v1/source_actors_v1/`:

- Generic native report:
  `report/20261018T124252.590592Z_job1797504/report.json`, SHA256
  `e0148130b9befdbb53df98102f365fdb6573b788039595e1b1617075aaf4a7fe`.
- Common-reference paired report:
  `common_reference_report/20261018T124308.027194Z_job1797505/report.json`, SHA256
  `5ffe75e58bc1e18fa33a50a1b503377bce7b051aee0dcaac899c4e421adc4c98`.

All values below are mean +/- sample SD of three training-seed means, with
eight predeclared outputs per seed. These are seven **global-route** metrics,
not final detailed-route/DRC measurements.

| Policy | WL (m) | WNS (ns) | TNS (ns) | H overflow | V overflow | Power (W) | Standard-cell area (um2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BC, no Q | 2.314028 +/- .012141 | -.403605 +/- .016925 | -217.399 +/-16.505 | 7.667 +/-2.504 | 1.375 +/-.451 | .170010 +/-.000151 | 61200.24 +/-347.02 |
| FQL, old-source O64 | 2.323754 +/- .006085 | -.415913 +/- .011548 | -217.693 +/-4.139 | 8.375 +/-2.929 | 2.042 +/-.520 | .170364 +/-.000293 | 61269.83 +/-182.19 |
| FQL, current-flow T64 | 2.320758 +/- .030722 | -.419403 +/- .004590 | -210.515 +/-14.279 | 9.583 +/-10.573 | 1.250 +/-.650 | .170362 +/-.000782 | 61387.77 +/-195.08 |

Paired WL changes (m), target minus reference; negative is shorter:

| Training seed | O64 minus BC | T64 minus BC | T64 minus O64 |
| --- | ---: | ---: | ---: |
| 0 | +.000210 | +.010395 | +.010185 |
| 1 | +.000463 | -.037157 | -.037620 |
| 2 | +.028505 | +.046952 | +.018447 |
| Mean +/- sample SD | +.009726 +/-.016264 | +.006730 +/-.042174 | -.002996 +/-.030268 |

T64 mean WL is .129% lower than O64 but .291% higher than matched BC; only one
of three seeds is better in either comparison. O64 is worse than BC in all
three seeds. T64 improves mean TNS and vertical overflow versus O64, while
mean WNS, horizontal overflow and area worsen; power is almost unchanged.
Both source actors fail the predeclared primary improvement rule, so this
comparison does not support promotion or stronger Q optimization.

The data path works, but this 64-label, 1000-update recipe has not established
a better Q-guided policy. The result does not rule out behavior-flow labels
generally, different coverage or larger data budgets. It also does not turn
the small mean T64-versus-O64 WL change into a stable benefit. The comparison
remains inspected development evidence; no fresh selection or confirmation
noise was created from it. Other previously declared development matrices
continue before the final candidate review.
