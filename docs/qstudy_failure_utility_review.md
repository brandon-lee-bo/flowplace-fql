# Failure utility under planned multiobjective rewards

Read-only review, 2026-10-18. **Current declared WL-only actors are unaffected. A future congestion-objective actor or utility-based candidate selector needs an explicit failure-ordering decision before launch.** This note changes no source, plan, physical metric, or stored reward.

## Definition and current scope

The screening utilities use costs in this order: routed WL, `max(-WNS,0)`, `max(-TNS,0)`, horizontal overflow, vertical overflow. Each cost is divided by its mean measured TRAIN magnitude; this objective scale is separate from the regression target standard deviation ([cost conversion and scales](../fql/qstudy_models.py#L37), [TRAIN statistics](../fql/qstudy_models.py#L101)). The measured utility is

```text
U_success = -sum_j(weight_j * cost_j / TRAIN_mean_cost_j).
```

Weights are not normalized to sum to one. The predeclared screening definitions are WL `[1,0,0,0,0]`, timing `[1,.15,.35,0,0]`, and timing/congestion `[1,.15,.35,.10,.10]` ([objective declarations](../fql/qstudy_screen.py#L23), [predicted and observed utility](../fql/qstudy_models.py#L298)). The last two have mean successful TRAIN costs of approximately 1.5 and 1.7 respectively; those means do not bound individual costs.

The existing `actors_v1/{bc_frozen,legacy_frozen,legacy_updated}/config.json` all declare WL-only weights, `uncertainty_beta=0`, and failure utility `-4`; BC has `q_weight=0`, the legacy actors `q_weight=1`. No launched multiobjective actor is established by the screening objectives. The legacy actor also explicitly rejects non-WL weights ([actor reward dispatch](../fql/qstudy_actor.py#L75)).

The native backend's stored `outcome.reward` remains **negative TRAIN-normalized WL on success and `-4` on confirmed physical/geometry failure**, irrespective of a prospective multimetric utility ([backend scale validation](../fql/qstudy_backend.py#L79), [reward integrity check](../fql/qstudy_backend.py#L165), [reward writing](../fql/qstudy_backend.py#L257)). Multimetric utility must be derived separately from preserved native metrics; it must not relabel that legacy field. Online multimetric critic updates currently regress confirmed failures toward scalar utility `-4`, without inventing a PPA vector ([failure-target loss](../fql/qstudy_actor.py#L241)).

## A concrete successful TRAIN layout scores below failure

Frozen evidence: `/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/bp_fe384/rows.jsonl`, line 120. Identity: `design=bp_fe`, `split=train`, `family=59`, `variant=1`, `source_row=178`, `collection_kind=additional_fixed_offline_behavior`, **`status=success`**. Its native outcome binding has SHA-256 `7300861c714687016654b2a6da18cb48b2febdba992974417459f49c3aa5e3df`; executed-action SHA-256 is `d3d47abffccbccef40b483f84f411767e1ad28e68727dd4a281fd33de6c110d8`.

| Metric | This native successful row | Frozen bp_fe384 TRAIN objective scale |
| --- | ---: | ---: |
| rWL (µm) | 2,495,337 | 2,368,912.3723958335 |
| WNS (ns; native signed value) | -0.489377 | 0.3981833489583333 |
| TNS (ns; native signed value) | -244.545 | 206.82261484375 |
| Horizontal overflow sum | 624 | 80.53645833333333 |
| Vertical overflow sum | 604 | 28.15625 |

The TRAIN scales are derived from the frozen manifest's mean magnitudes. All these TRAIN slack labels are negative, so their mean violation costs are the absolute native means.

```text
U_success = -(2495337 / 2368912.3723958335
              + .15 * .489377 / .3981833489583333
              + .35 * 244.545 / 206.82261484375
              + .10 * 624 / 80.53645833333333
              + .10 * 604 / 28.15625)
          = -4.5715348 (rounded).

U_confirmed_failure = -4 > U_success.
```

Thus the planned linear timing/congestion utility can prefer an actual failure to a successful layout **even with a perfect critic**. This is a demonstrated counterexample, not a count or estimate of how many successful rows are affected. Congestion tails relative to their small TRAIN means cause the crossing; the weights' sum alone is not a safety guarantee.

Evidence hashes:

```text
bp_fe384/dataset.npz:  0daecf483b64283af46e3aec101161912ebff2984a493d0778fba5b2ebba99e4
bp_fe384/rows.jsonl:   dfaa97f0b59602212e6eace0d702e5a9c1ab21508fe223c2ad9b98553a1a3f03
bp_fe384/manifest.json:e5d374f919c3fe3b8e94898f43d41a101a21ff40b943746716e4b2f86670275b
```

## What remains safe on known TRAIN support

For current WL-only bp_fe384, the largest observed TRAIN WL divided by its mean is 1.149209, so every successful TRAIN reward exceeds `-1.149209`, safely above `-4`.

For timing-only `[1,.15,.35,0,0]`, adding the separate worst observed TRAIN ratios gives a conservative upper bound on any row's cost. The per-metric maxima need not occur in the same layout; the bound still holds.

| Frozen TRAIN dataset | Upper bound on timing-only successful cost |
| --- | ---: |
| bp_fe384 | 1.900165 |
| bp_fe64 | 1.820255 |
| bp_be64 | 1.764120 |

All known successful TRAIN rows therefore outrank `-4` under these timing-only definitions. This establishes a property of the existing TRAIN support, **not** a bound on unseen layouts, future online data, or extrapolated critic outputs. Linear normalized costs are unbounded; even WL-only cannot guarantee failure-last ordering everywhere with a finite failure constant.

## Gate before future multiobjective actor or candidate selection

Two alternatives need an explicit prospective declaration; this review does not choose between them:

1. **Bounded successful utility.** Define a monotone bounded transformation and a known successful-utility range strictly above the common failure utility. Declare its effects on metric tradeoffs, actor gradients, critic success/failure targets, and reporting. Dividing weights by their sum, or selecting a more negative constant from the largest observed TRAIN cost, does not by itself bound unseen costs. Do not tune the new definition on evaluation outcomes.
2. **Strict feasibility ordering.** Rank every confirmed physical/geometry failure below every measured success, then optimize the declared PPA utility among successes. A failure-last candidate selector alone does not fix an actor trained with the existing crossing scalar objective: actor/critic feasibility treatment must be separately defined and consistent, rather than claiming the reporting rule repairs training.

Before either is promoted, bind its TRAIN reference scales and weights, failure semantics, actor extraction objective, critic target and measured selection rule in the same prospective experiment. Source-specific scales must not silently change the physical preferences of a source comparison. Retain the native WL-only `outcome.reward`, native seven metrics, and failure status unchanged.

Only native-confirmed physical or geometry failures receive the physical failure utility. An unavailable active metric yields missing utility, not a successful zero or a fabricated failure ([observed-utility mask](../fql/qstudy_models.py#L310)); infrastructure failures must remain unresolved/missing rather than acquiring PPA or a physical-failure label ([replay admission](../fql/qstudy_actor.py#L109)). Report missing cases and failures explicitly. No new fit, inference, placement generation, EDA run, or objective implementation was performed for this note.
