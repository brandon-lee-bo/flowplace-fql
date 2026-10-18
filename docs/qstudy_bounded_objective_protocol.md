# Prospective bounded multiobjective comparison

This is a new objective version. It leaves existing WL-only policy runs,
native backend rewards, original layouts and measurements unchanged. The
first question is whether different measured objectives within one scalar
critic family produce useful physical policy tradeoffs. It is not a claim
that scalar heads beat native-metric heads.

## Measured objective

Convert native slack to nonnegative violation costs and retain native signs
in physical reports. For costs c = (WL, WNS violation, TNS violation,
horizontal overflow, vertical overflow), define:

    C = sum_k w_k * c_k / S_ref,k
    successful terminal cost B = 2*C/(1+C)
    confirmed physical/geometry failure cost B = 4
    utility U = -B

All reference scales are positive. The three fixed preferences are:

| Objective | WL | WNS | TNS | H overflow | V overflow |
| --- | --- | --- | --- | --- | --- |
| WL | 1 | 0 | 0 | 0 | 0 |
| Timing | 1 | 0.15 | 0.35 | 0 | 0 |
| Timing and congestion | 1 | 0.15 | 0.35 | 0.10 | 0.10 |

Every genuine finite successful layout has B at most2, so it cannot rank
below the confirmed-failure target4. Inactive missing metrics are ignored;
missing active metrics and infrastructure interruptions remain invalid/missing,
with no invented PPA. Derivation must be numerically safe for extreme finite
inputs. Native `outcome.reward` keeps its legacy WL-only meaning; B is a
separate derived target.

B preserves ordering among deterministic successful candidates for fixed
weights and scales. It changes policy risk preferences: minimizing mean B
is not equivalent to minimizing mean C, because it compresses large successful
costs. Reports must retain mean C, B, raw physical means/tails and failures.
The guarantee is about measured targets, not the correctness of Q predictions.

## Critic comparison

Use the existing audited bp_fe384 TRAIN data. There are nine tasks: three
objectives times independent critic seeds0/1/2. Every task has three grouped
family-CV fits and one full-TRAIN refit. All objective variants use the same
family splits, paired minibatch streams,2000 updates, batch64, AdamW learning
rate3e-4/weight decay1e-4 and two independent2x128 MLP members. Inputs remain
the original proposal coordinates and17 pre-backend physical features.

Each CV fit derives reference scales from that fold's TRAIN metrics only.
The three objectives share those scales within that fold. Feature and target
standardizers also use that fold's TRAIN only. Full refits use full384 TRAIN
scales; these are fixed for all later actors and online rounds. Do not pool
CV predictions or utility scores across folds with different scales.

Train a scalar terminal-cost head directly on B with Huber loss. Actor Q is
minus the mean predicted B in native B units; do not divide by its TRAIN mean
a second time. Do not apply the rational transform to unrestricted predicted
linear costs, which can be negative. An ordinary unconstrained head can still
predict outside the measured-success interval; record that explicitly rather
than clipping predictions and silently creating zero-gradient regions.

Check held-family MAE, ranking, family-local pairs and top-choice regret
against a constant TRAIN-mean B predictor. Store the original seven metrics
alongside predictions, precise splits, utility specifications, all normalization
bindings, optimizer/RNG state and native provenance. Check finite predictions
and action gradients. Three critic seeds measure training variation, not
independent data-sampling uncertainty.

## Bounded actual-policy test

After implementation/target checks and the complete critic report, a fixed
three-objective by three-seed offline policy experiment is an informative
negative-hypothesis test even if timing prediction remains weak. It must be
identified as such, rather than promoted based on a favorable validation cell.
Do not add source/head-count/alpha grids. Use common independent warmups,
frozen flow, the same256 behavior targets, alpha10,1000 actor updates and
eight actual shared development noise arrays per seed. This requests72 GRT
measurements, with the existing fixed physical flow and resources.

Compare all requested layouts, not only the best candidate. First report
failures, then per-seed and three-seed physical summaries for WL, WNS, TNS,
both overflows, power, area and actual macro legality. Compare measured B and
linear C under each declared preference on every arm. Failures enter B as4;
unresolved infrastructure results do not. Prediction errors and Q-versus-
distillation gradient contributions accompany the physical results.

The bounded transform changes gradient magnitude, and its derivative depends
on C. Fixed alpha10 does not imply equal measured Q gradient pressure across
objectives. A null result cannot establish that timing preferences are useless.
No online extension or fresh confirmation follows automatically. Only a
credible measured tradeoff, acceptable failures and a fixed method justify
the subsequent matched own-feedback online test. Negative or inconclusive
results can close this bounded objective experiment honestly.

Power and area remain measured report outcomes in this bounded experiment.
The existing bp_fe384 audit finds power 0.168985 +/- 0.001481 W and standard
cell area 60052.2 +/- 655.8 square micrometers (sample SD across TRAIN rows).
Their relative variation is narrow but not zero or inherently unimportant.
Keep the first objective comparison focused on WL/timing/congestion while
checking whether these outcomes deteriorate; do not add two more arbitrary
weights merely to consume every available label. These are GRT-stage values,
not the later final-route power/area reference.

## Completed development result

The saved-only report job1796866 completed on2026-10-18. Its artifact is
`/fact_data/bohanli/flowplace_fql/qstudy_v1/bounded_actor_native_report_v1/20261018T132750.979106Z_job1796866/report.json`,
SHA256 `0ce81b4ef635d0d6def02d751a809a3459bc1ed23d6003dd02167bae63d4ddfb`.
The three bounded arms each have24/24 successful GRT outcomes. The unchanged
existing BC reference also has24/24:72 new bounded measurements plus24 reused
reference measurements, not96 new measurements. There are no missing,
unexported, invalid, geometry-failed or physically failed requests. Every
seed uses the same eight saved actual development noise arrays; all24 pairs
are admitted for each contrast. These are complete GRT-stage summaries,
without final detailed-route or DRC-zero claims.

Each entry below is the equally weighted mean of three training-seed means
plus/minus their sample SD. Each seed mean covers all eight requested outputs;
the SD is not over24 independent training runs.

| Native metric | Existing BC reference | Bounded WL | Bounded timing | Bounded timing + congestion |
| --- | --- | --- | --- | --- |
| Routed WL (m), lower | 2.31403 ± 0.01214 | 2.30427 ± 0.01421 | 2.28726 ± 0.01479 | 2.30349 ± 0.02089 |
| WNS (ns), closer to0 | −0.40360 ± 0.01692 | −0.41856 ± 0.01218 | −0.41536 ± 0.00347 | −0.41485 ± 0.01291 |
| TNS (ns), closer to0 | −217.40 ± 16.51 | −217.92 ± 10.34 | −231.04 ± 16.41 | −222.39 ± 14.37 |
| Horizontal overflow, lower | 7.667 ± 2.504 | 2.000 ± 1.317 | 11.333 ± 3.403 | 4.083 ± 1.371 |
| Vertical overflow, lower | 1.375 ± 0.451 | 0.792 ± 0.382 | 2.000 ± 1.750 | 1.167 ± 0.473 |
| Power (W), lower | 0.170010 ± 0.000151 | 0.170021 ± 0.000150 | 0.169886 ± 0.000282 | 0.169842 ± 0.000277 |
| Standard-cell area (µm²), lower | 61,200.2 ± 347.0 | 61,525.2 ± 330.8 | 61,027.3 ± 46.2 | 61,011.4 ± 46.6 |

Paired WL differences are candidate minus reference, inµm; negative is better.
The final column is the mean and sample SD of the three paired seed differences,
which must not be inferred by subtracting the marginal SDs above.

| Contrast | Seed0 | Seed1 | Seed2 | Paired mean ± SD |
| --- | --- | --- | --- | --- |
| Bounded WL − BC | −9,257.875 | −7,982.375 | −12,021.000 | −9,753.750 ± 2,064.471 |
| Bounded timing − BC | −13,268.875 | −49,379.625 | −17,645.750 | −26,764.750 ± 19,706.945 |
| Bounded timing + congestion − BC | +7,118.375 | −42,104.500 | +3,382.000 | −10,534.708 ± 27,403.995 |
| Bounded timing − bounded WL | −4,011.000 | −41,397.250 | −5,624.750 | −17,011.000 ± 21,134.520 |

The within-bounded contrasts are controlled objective interventions under this
protocol. Comparisons with existing BC remain descriptive paired physical
references: the report explicitly sets
`controlled_effective_update_eligible=false` for those contrasts. The earlier
source-actor BC certificate does not establish bounded zero-Q update
equivalence. Reusing the same actual noises and native BC layouts establishes
pairing, but does not extend that certificate.

The timing arm has shorter WL in all three seeds than bounded WL, with a0.738%
mean reduction. It nevertheless increases TNS violation by13.120ns, horizontal
overflow by9.333 and vertical overflow by1.208 relative to bounded WL. Under
the same fixed timing utility specification, its measured mean B is1.202607,
versus1.198283 for bounded WL and1.197221 for BC; lower B is better. Adding
congestion terms reduces both overflows relative to the timing arm, but gives
back16,230.042µm of mean WL and still has worse measured congestion utility
than bounded WL. Thus the experiment shows physical tradeoffs, without reliable
optimization of the intended timing/congestion utility. Its saved own-objective
Q/native-utility rank correlations average−0.500 for timing and−0.579 for
congestion, versus+0.611 for WL. These small, policy-conditional diagnostics
identify prediction limitations; they do not prove a unique cause of the result.

Against the locked development-promotion rules in
`fql/qstudy_confirmation_rules_v1.json` (SHA256
`b6553120d9a4f410740d9f45ee3214dfa62b48f27b0a75a8c4cfe262f9b940ab`):

| Arm versus descriptive BC | Mean WL reduction | All three seeds lower WL? | Numerical gate outcome |
| --- | --- | --- | --- |
| Bounded WL | 0.422% | Yes | Below the required1% primary threshold |
| Bounded timing | 1.157% | Yes | Primary passes numerically, but TNS and horizontal-overflow tolerances fail |
| Bounded timing + congestion | 0.455% | No | Primary threshold and three-seed direction fail |

For timing, TNS violation increases13.642ns versus the allowed10.870ns
(`max(1ns,5% of BC)`); horizontal overflow increases3.667 versus the allowed1.
Mean WNS, vertical overflow, power and area remain within their respective
locked tolerances. The secondary failures remain even if a bounded BC
equivalence certificate were later established. No bounded arm is eligible for
promotion from this completed experiment. This is a development conclusion,
not a fresh-confirmation result, and it creates no automatic new controls,
online runs or candidate selection.
