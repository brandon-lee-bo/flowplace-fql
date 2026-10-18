# Q / actor objective study, 2026-10-18

This continues the closed graph-FQL pilot. Table2 jobs stay held. The terminal
action is a complete layout: no Bellman bootstrap is introduced. The objective
is measured physical improvement, not a lower fitted loss or larger predicted Q.

## Frozen context and evidence

Use the audited bp_fe64, bp_fe384 and bp_be64 TRAIN/validation families. Native
GRT outcomes supply wirelength, signed WNS/TNS, H/V overflow and available
power/area. Only TRAIN fits normalization or model weights. Historical online
or evaluation labels are diagnostic evidence, not silently added offline TRAIN.
Raw proposal, legalized layout, rounded DEF and actual backend geometry retain
separate identities. Missing labels are missing, not zeros.

Initial experiments keep the ORFS container, Nangate45 technology, clock,
placement density, routing settings and number of backend threads unchanged.
GRT is the label stage; final DRT metrics are reported separately. H/V overflow
is OpenROAD's measure and must not be called the paper's Innovus rO-H/rO-V.

## Hypotheses and first matrix

| ID | Change from multi_mse | Question |
| --- | --- | --- |
| legacy | existing twin scalar reward MLP, MSE | reproduce the current critic under identical splits |
| multi_mse | separate five cost heads, twin 2x128 MLP | can metric-specific supervision improve WL and enable interpretable tradeoffs? |
| multi_huber | Huber regression | do extreme timing/congestion labels dominate squared loss? |
| huber_rank | add 0.1 pairwise ranking | does ordering improve, including within layout families? |
| huber_local | add 0.1 same-family difference loss | do local effects generalize better than absolute labels? |
| huber_residual | add TRAIN-fitted smooth-HPWL residual baseline | does a physical prior improve extrapolation? |
| huber_large | 3x256 rather than 2x128 | is baseline capacity limiting generalization? |

Start with seeds 0/1/2, bp_fe384 and bp_be64. bp_fe64 is the label-count control
after the runner is validated. Match training budget; record all attempted
models. Each fit sees TRAIN only. A deterministic family-grouped three-fold
cross-validation inside TRAIN checks whether gains depend on the small fixed
validation set. Refit on all TRAIN for fixed validation predictions. Checkpoint
choice is fixed update budget in this screening, not best TRAIN loss. Report
per-metric MAE, normalized MAE, Spearman/Kendall, top-choice regret and within-
family pair accuracy. Report constant/missing metrics explicitly.

The shared five costs are WL, max(-WNS,0), max(-TNS,0), horizontal overflow,
vertical overflow. Regression target z-scores and objective magnitude scales
are different quantities, both TRAIN-only. Initial utility controls:

- WL only: weights [1,0,0,0,0].
- WL + timing: [1,0.15,0.35,0,0].
- WL + timing + congestion: [1,0.15,0.35,0.10,0.10].

Weights define explicit preferences, not a universal optimum. Reuse metric
predictions to inspect Pareto tradeoffs before new actor/EDA runs. Power/area
stay audit-only until their label stage and usable variation are established.

## From prediction to actual actions

Screening does not establish actor improvement. Advance only a small number
of distinct promising mechanisms. For each, measure predicted and real changes
along equal-RMS positive/negative Q directions and a matched random direction
from fresh fixed-noise layouts. Retain all legalizer jumps and failures. These
new diagnostic labels stay separate from final evaluation and initial TRAIN.

Actor controls share the same behavior flow and initialization. Compare BC,
old scalar FQL, and selected metric critics; use alpha=10 as the reference and
probe alpha=3/30 only when measured gradient norms justify it. Record Q and
distillation losses and parameter/input gradient contributions. Treat ensemble
mean-minus-standard-deviation as a heuristic uncertainty penalty, not CQL.
Add a reward-weighted behavior-learning/candidate-ranking control if gradients
fail while ranking succeeds. COM-style pessimism remains a separate hypothesis.

Freeze behavior flow for the first actor contrast so Q effects are isolated.
Then compare frozen vs updated behavior flow with matched online samples,
critic updates and actor updates. Online means the policy's own new measured
layouts enter replay before a subsequent parameter update. Start at two rounds
of eight samples; extend based on observed learning curves rather than assuming
more labels help. Retain matched BC and equal-budget best-measured-candidate
controls; never compare a selected singleton with an unselected policy mean.

After validation-based choices are frozen, use new evaluation noise seeds and
three independent train seeds. Record all failures and full requested counts.
Final DRT evaluates WL, WNS, TNS, DRC, power and area, plus GRT congestion as a
separate stage. Extend to another/larger design after a reproducible mechanism
is found; absence of improvement is a valid outcome, not a reason to stop audit.

## Execution and deliverables

All training/inference/audits/EDA run under Slurm. Parameter batches use
scavenger queues and resumable checkpoints. Code/configs stay in this repo;
datasets/models/results go to /fact_data or /scratch. Do not touch other
projects' jobs. Worker identity, config/data/source hashes, actual update count,
optimizer/RNG state, backend count, wall time and failure class are recorded.

This is a staged protocol, not a claim all experiments above have run. Matrix
and outcome manifests must distinguish planned/submitted/running/completed/
failed. Literature mechanisms and our adaptations are documented separately in
qstudy_literature.md. New observations may refine later stages with an explicit
dated change before seeing their results.
