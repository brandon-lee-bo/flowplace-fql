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
