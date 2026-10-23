# Placement versions: measured development results

Updated 2026-10-23. These experiments clarify what changes when replacing the
teacher or the action granularity. They do **not establish stable overall PPA
improvement**. All new results below use `bp_fe` (11 movable macros) and global
routing (GRT). They are not detailed-route/Table 2 results or independent
confirmation experiments.

## What the versions implement

| Version | Teacher | Student action | Critic target |
| --- | --- | --- | --- |
| FlowPlace reference | Released weights and original constrained 20-step sampler | No distilled student | No RL critic |
| V1.0 matched pilot | Public initialization, 300 FM warmup updates, then frozen Euler10 | All macro coordinates in one forward | Measured terminal reward regression |
| V1.1 matched pilot | Frozen released weights and original constrained 20-step sampler | All macro coordinates in one forward | Same terminal reward regression |
| V2_4 prototype | Frozen released weights with **modified prefix conditioning** | Current macro's two coordinates in one forward | Recorded-return MC warmup, then Bellman updates |

The V1.1 teacher is the exact released sampler. V2 changes its conditioning and
mask handling to hold the supplied prefix fixed; it is an experimental adapter,
not an unchanged FlowPlace baseline. The student observes the same complete
latent tensor as the conditional teacher, but distills only the current macro.
During offline training the prefixes come from recorded layouts. During rollout
they come from the student's own previous decisions, including projection.

Whole-layout V1 is a single-action contextual bandit: Q regresses a measured
terminal score without bootstrapping. V2 is an 11-decision finite-horizon MDP.
"One-step policy" means one forward per **action**; a V2 layout still needs 11
actor calls. Neither definition implies that weights update during evaluation.

## V1.0: Q guidance is not consistently better than matched BC

Each seed has eight fixed, paired evaluation noises per arm. All 48 student
evaluation requests succeeded. BC and FQL share the common initialization,
teacher, training data, update budget and actual evaluation noises. BC has
`q_weight=0`; FQL has `q_weight=1`, with the same alpha=10.

| Independent training seed | BC mean rWL (µm) | FQL mean rWL (µm) | Mean paired improvement |
| --- | ---: | ---: | ---: |
| 0 | 2,357,146.750 | 2,366,186.750 | −0.3794% |
| 1 | 2,351,137.875 | 2,304,150.000 | +1.8668% |
| 2 | 2,325,744.375 | 2,351,225.875 | −1.1462% |

Improvement here is the mean of eight `(BC − FQL) / BC × 100%` values, then
averaged over the three seeds: **+0.1138%, sample SD 1.5659 percentage points**.
Only one training seed improves. This does not support a stable wirelength gain.
The separate public FlowPlace20 reference has mean rWL 2,334,605.75 µm over
eight measured layouts. That reference was measured once and reused explicitly;
it is not three independent public-reference measurements.

## Online controls: extra optimization must be separated from new feedback

The V1.0 online stage updates **both actor and Q**, while the teacher remains
frozen. Reward is `−measured GRT rWL / 2341121.25`; the denominator is fixed from
the original TRAIN64. New feedback means actual collector layouts evaluated by
the backend, not the critic's own predictions or evaluation layouts.

Each R1 arm starts from the same seed-0 R0 FQL parent. Continuing for 600 actor
updates gives the following means over the same eight evaluation noises:

| Continuation | Mean rWL (µm) |
| --- | ---: |
| R0 FQL parent | 2,366,186.750 |
| New online feedback, Q updated | 2,360,075.500 |
| Offline data only, equal update count | 2,313,338.625 |
| Online continuation, Q frozen | 2,324,305.000 |

Online improves slightly relative to the parent, but is **2.007% worse than the
equal-update offline control** by mean paired improvement. At R2, online mean
rWL is 2,332,933.625 µm versus 2,295,893.375 µm for the equal-update offline
control. Thus these runs do not establish that new feedback causes improvement.

Holding the same eight feedback labels fixed and taking 150/600/1200 updates
gives mean rWL 2,345,002.000 / 2,360,075.500 / 2,348,034.375 µm. More updates
are not monotonically better. These are development observations from one
training seed, not a selected optimum or a confirmed stopping rule.

In `versions_whole_train`, replay trains Q with balanced old/new minibatches;
the actor distills fresh samples from the frozen teacher while maximizing Q.
The prepared behavior minibatch and its RNG draws are historical bookkeeping:
there is no flow loss/update in these frozen-teacher branches. New successful
layouts do not automatically become targets for a newly trained behavior flow.

## V1.1: implementation and training completed; matched PPA remains incomplete

The public teacher passed exact, same-noise equivalence checks against an
independently loaded original sampler. Both seed-0 BC/FQL training arms have
completed. Each changed 207 actor and 16 critic parameter tensors; the teacher
changed **zero**. The flow optimizer is retained for schema compatibility and
has no update state.

This verifies the intended training path. The matched independent backend
comparison is not complete, so there is **no V1.1 quality-gain claim**. Public
sampling runs at B=1 within each minibatch, which also makes distillation more
expensive than the old vectorized Euler10 teacher.

## V2: sequential placement works as a prototype, with important limitations

V2 places macros in descending-area order, with node ID breaking ties. The
state contains the graph, known coordinates, a placed mask and the current ID;
unknown final coordinates are hidden. The environment projects each proposal
to a feasible position without moving the already placed prefix. No feasible
position is a recorded failure, not a silently discarded sample.

Offline TRAIN64 layouts become 704 transitions. They still contain **64 backend
labels**, not 704 independent measurements. Intermediate rewards are zero;
the last reward is the measured layout utility. MC initialization follows the
recorded continuation. Subsequent Bellman targets use the current student's
next proposal, gamma=1 and a target critic updated with tau=.005.

The original execution padding=0 variant passed the native geometry/import
gate for only **BC 2/8 and FQL 1/8**. A separately declared padding=.001
execution variant passed **8/8 in both arms**; only this variant received the
following complete GRT comparison. It does not replace the failed original
gate results.

| Metric | BC | FQL | FQL relative to BC |
| --- | ---: | ---: | --- |
| rWL (µm) ↓ | 2,177,983.375 | 2,164,043.125 | 0.640% lower by ratio of means; 6/8 paired wins |
| WNS (ns) ↑ | −0.4767135 | −0.4572176 | Better |
| TNS (ns) ↑ | −243.615125 | −254.722875 | Worse |
| Power (W) ↓ | 0.168946375 | 0.168289875 | Lower |
| Standard-cell area (µm²) ↓ | 61,575.925 | 61,695.650 | Higher |
| Horizontal overflow sum ↓ | 3.500 | 0.750 | Lower |
| Vertical overflow sum ↓ | 1.875 | 4.375 | Higher |

There is one training seed and eight paired noise streams, not eight independent
trained models. All 16 logical GRT evaluations completed; a preempted attempt
required one declared retry. Results retain every predetermined pair.

Q is fitted on recorded identity-projection actions, but actor optimization and
Bellman bootstrapping query clamped proposals before environment projection.
Learning the value of these unseen proposals and new student prefixes is not
guaranteed. The final FQL critic has some terminal ranking signal but very weak
calibration: its predicted reward standard deviation is only about 0.0624% /
0.0755% of actual label standard deviation on the BC/FQL cohorts. This does not
establish a useful local Q gradient. Sequential HPWL-based online adaptation
has not been implemented or evaluated here.

V1 and V2 differ in model, action dimensions, training exposure, Q initialization
and execution. The cross-version results are system comparisons, not an
equal-compute ablation proving that sequential actions alone caused a gain.

## Evidence and reproduction

[The saved numerical snapshot](data/placement_versions_20261023.json) includes
all seven metric means, seed-level statistics, native failure denominators,
paired summaries and source-report SHA256 digests. It is a compact extraction,
not a substitute for the large source artifacts or their full audit.

[Reproduction and parameter guide](placement_versions_reproduction.md) describes
the entrypoints, data requirements and budgets. The historical graph/FQL,
reward-screening and detailed-route studies remain separately documented in
[the original experiment record](graph_fql_offline_to_online.md) and
[Q-study execution guide](qstudy_reproduction.md); their results must not be
relabeled as these matched pilots.
