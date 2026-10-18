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
| Per-net pin-span feature control | 1795807; reports 1795829 / 1795843 | Same folds and 2x128 hidden trunk; extra per-net inputs | Feature hypothesis, including additional input parameters |
| Direct timing-utility scalar control | 1795825 | Three critic seeds, same multi-head comparator folds and minibatch RNG | Separates objective representation from head count |
| Current-flow source labels | export 1795763; backend 1795769; collect 1795775 | Seed-0 common behavior flow; 32 TRAIN + 8 validation families, base and fixed-RMS local variant | 80 requested native labels, not 80 independent circuits or training seeds |
| Equal-label source controls | plan 1795783; check 1795814; assemble 1795840; fits 1795841; report 1795842 | O64 / T64 / M64 and base-only O32 / T32 / M32; legacy and residual critics, three critic seeds | Same old/new validation candidates; exact original 256 BC targets unchanged |
| Matched BC and legacy FQL | controller 1795764 | Three independent actor seeds; fixed common initialization; 1000 offline actor updates, two online rounds of 8 labels and 600 updates | Behavior flow frozen; actual self-collected feedback required for online updates |
| Local Q-direction checks | exports 1795767 / 1795768; backend 1795818 / 1795819; reports 1795844 / 1795845 | Four common-actor neighborhoods per design; two radii, both signs, matched random direction | Checks real optimization direction and legalizer changes; diagnostic labels stay outside replay |

Main matrix results available so far do not establish a reliable timing
predictor. For example, on bp_fe the larger critic nearly interpolates TRAIN
WNS while grouped held-out WNS remains weak. Per-net pin-span features also
fail to consistently repair timing: their bp_fe OOF WNS Spearman is about
0.040, versus 0.008 for the shared-head Huber model, while TNS worsens. On bp_be
the same pin features improve WL ordering but do not yield a uniform metric
gain. Final comparisons must use complete matched seeds and retained reports.

First current-flow backend samples succeeded with all seven native metrics
and legal actual macro geometry. Local variants can improve WL while worsening
slack; raw perturbation size can also substantially differ from the executed
coordinate change. These observations motivate preserving multiple metrics and
the complete raw-to-executed provenance. They are not aggregate efficacy claims.

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
