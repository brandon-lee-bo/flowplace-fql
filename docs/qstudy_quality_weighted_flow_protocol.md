# Measured-quality weighted flow: fixed development control

Declared 2026-10-18 UTC, before new flow training or backend evaluation.
This is an auxiliary control for the FQL study, not a replacement claim for
offline RL, IQL or AWAC. Those methods motivate favoring measured high-quality
actions without asking an actor to climb an inaccurate Q surface. Here we test
that idea with the existing flow-matching distribution model.

## Hypothesis and controls

The saved actor diagnostic found that 11 of 23 layouts which the critic scored
as improvements actually worsened the measured utility. Quality-weighted flow
matching removes that explicit critic-gradient optimization route. It does not
guarantee generalization, feasibility, or improved downstream performance.

Train exactly two arms for bp_fe, each with independent training seeds 0/1/2:

- `uniform`: continue flow matching on measured TRAIN layouts, all weights 1.
- `quality_rank`: the same training, with fixed bounded weights derived from
  the native TRAIN GRT wirelength labels.

For each seed, both arms load precisely the same `common['flow']` state, reset
Adam identically, and use the same sampled row indices, noise and time values.
Uniform versus the unchanged common flow diagnoses adaptation to the measured
layout source and additional updates. Quality-rank versus uniform isolates
weighting. Historical FQL remains an anchor, but its different one-step output
and training budget prevent a clean causal comparison to these multi-step flows.

## Frozen inputs and weighting

Use only `train.raw_action` and its row-matched native GRT labels from:

`/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/bp_fe384/dataset.npz`

Dataset SHA256:
`0daecf483b64283af46e3aec101161912ebff2984a493d0778fba5b2ebba99e4`.

There are 384 measured TRAIN records from 192 families, including a base and
perturbation per family, with 382 distinct executed proposals. These are from
the older public FlowPlace20-hard collection. They are not the newer current
behavior-flow dataset. Labels describe the externally legalized, DEF-rounded,
native-snapped execution, conditional on this unchanged postprocessing. Keep
raw training targets intact; do not clip them and retain the original labels.

The separate `behavior_action` array contains 256 mixed measured/unmeasured
targets and must not be paired positionally with the 384 labels. The 16
historical validation records, all final DRT labels and all evaluation outputs
are excluded from training and weighting. Report physical failures and missing
active labels; exclude them identically from both arms, never turn an
infrastructure failure into a low-quality numeric label.

Let `u_i = -WL_i / mean_TRAIN(WL)`. Rank these utilities in ascending order with
zero-based average ranks for exact ties, and let `p_i = rank_i / (N-1)`.
Set `w_tilde_i = exp(log(4) * (p_i - 0.5))`, then divide by the complete TRAIN
mean of `w_tilde`. Higher utility receives more weight; the largest/smallest
weight ratio is at most 4. If all utilities tie, all weights are 1. No
temperature search, best-only filtering, or weighted resampling is performed.
Rank weights are our explicit adaptation, not the original IQL/AWAC objective.

The common checkpoint paths and hashes are the three entries in
`/fact_data/bohanli/flowplace_fql/qstudy_v1/actors_v1/bc_frozen/config.json`.
Each must be checked independently; seed0 must not stand in for seeds1/2.
Bind the config, dataset, row provenance, source code and checkpoints before
training. Save the exact weights and eligible TRAIN indices.

## Fixed optimization and generation

Use the existing GATv2/self-attention/MLP flow network and linear noise-to-layout
flow-matching path. Average squared velocity error over movable coordinates
within each example, multiply by its detached weight, then average over the
uniformly sampled batch. Do not use the existing batch-reduced flow loss as
though it still contained individual example losses.

- 1,000 updates per arm/seed; batch16; Adam learning rate `1e-5`.
- Gradient norm clip1; identical paired RNG streams; no Q or one-step actor.
- Fixed last checkpoint; no checkpoint selection by validation or backend.
- Resume preserves optimizer, RNG and step; interrupted attempts remain logged.
- Generate with 10 Euler steps, fixed-port restoration, final clipping and the
  existing external legalizer. No intermediate hard constraints are added.

Train and evaluate through Slurm only. Weights and datasets stay on bulk or
scratch; use scratch for this extension because the durable bulk quota is
tight. Preserve selected final artifacts and their bindings before cleanup.
This storage choice does not relax the existing global 8GiB backend reserve.

## Evaluation, limits and stopping rule

Use the actual eight saved development noise arrays already used by the common
flow and BC/FQL controls, with identical arrays across arms and seeds. These
are development data, not the reserved fresh-confirmation stream. Evaluate
all eight outputs from all six trained flows: 48 new eight-thread GRT runs,
with a maximum of four concurrent workers and no policy/layout selection.
The unchanged common flow's existing 24 measurements may be reused only after
the complete checkpoint, sampler, noise and physical-context bindings match.

Report all seven GRT metrics, physical/infra status separately, raw-to-executed
displacement, duplicate execution counts and three seed means. Standard
deviations describe variation across seed means, not independent training runs
created by treating generated samples as seeds. Preserve measured TRAIN source
provenance and account for all new training and backend calls.

For promotion, require at least 1% mean GRT wirelength reduction versus the
uniform arm and the same improvement direction for all three seed means, with
no added physical failures. Use the existing secondary operational tolerances:
WNS deterioration at most0.02ns; TNS violation-magnitude increase at most the
larger of1ns or5%; each overflow increase at most the larger of1 or5%; power
and area increases at most1%. These are engineering gates, not significance
tests. Missing evidence is inconclusive.

If this fixed recipe fails, close it without a temperature/source sweep. If
it passes, evaluate the predeclared first four outputs per seed of both arms
with the existing detailed-route procedure (24 additional runs), preserving
the same repair cap and reporting DRC. A later independent fresh confirmation
is still necessary before promoting a stable method. No DRT result in this
development control enters TRAIN or online replay.
