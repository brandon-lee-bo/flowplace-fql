# Expert-rule reward contrasts

Development protocol, fixed before these reward policies are trained or
evaluated. The older weighted-flow WL experiment is an auxiliary control;
these experiments retain the FQL one-step actor and behavior-flow teacher.

## Fixed comparisons

| Arm | Base cost | Additional cost |
| --- | --- | --- |
| `grt_wl` | measured GRT routed wirelength | none |
| `macro_hpwl` | executed-layout macro-net HPWL | none |
| `hpwl_edge` | macro-net HPWL | mean nearest-edge clearance |
| `hpwl_group` | macro-net HPWL | fixed cached group bounding-box area |
| `hpwl_dataflow` | macro-net HPWL | fixed dataflow-weighted center distance |
| `hpwl_channel` | macro-net HPWL | narrow facing-channel risk |
| `hpwl_expert_composite` | macro-net HPWL | equal mean of the four rules above |
| `grt_expert_composite` | measured GRT routed wirelength | the same four-rule mean |

All rule definitions and coordinate units are those in the final bound
`analysis/qstudy_rule_audit.py`. They are complete-layout adaptations inspired
by EXPlace and routing practice, not its exact sequential reward. The channel
threshold is two intervals of the released 128-point legalization lattice.
I/O obstruction is excluded: its historical cache construction is unverified.
No static net weight is interpreted as STA criticality. Nonoverlap, bounds and
fixed-node preservation remain hard execution checks, not variable rewards on
already legal layouts. A cached cluster is not asserted to be a semantic RTL
module. These rules are hypotheses, including the possibility that they hurt.

For each fit, normalize every active nonnegative cost by its mean on that
fit's TRAIN rows. Let `b` be the normalized base and `e` an individual
normalized rule or the equal mean of the four normalized rules. Use `C=b` for
base-only controls and `C=(b+0.25e)/1.25` for the other arms. Fit the scalar
target `B=2C/(1+C)`, with `Q=-mean(predicted B)`. There is no further mean-B
renormalization. This keeps a common successful-target range and mean-cost
reference; it does not guarantee equal actor gradients. Report distillation
and Q gradient norms separately. Constant or unavailable required terms make
an arm inadmissible rather than silently changing its formula.

## Data, model and training

- Use exactly the original measured 384 TRAIN rows /192 paired families,
  with source, raw coordinates, executed coordinates and outcome bindings.
  Rules and native outcomes refer to executed layouts; critic inputs remain
  raw proposals, including the original legalization difference.
- All eight arms use the same three existing family-CV splits, three model
  seeds, two-member critics with two hidden layers of width128, Huber
  regression and 2000 updates. Refit on all384 rows at the same fixed last
  update. Fit input/target normalizers and cost scales on fold TRAIN only.
  Historical16 validation rows may be reported as development diagnostics;
  they do not select rules, scales, weights or checkpoints.
- The actor uses the existing common initialization for each of three seeds,
  its frozen Euler10 behavior teacher, the same paired update streams,
  1000 updates and loss `10*coordinate_distillation - Q`. No action pruning,
  network change, legalizer change or online replay enters this first contrast.
  Separate online trials are considered only with an explicit collection plan.
- Keep all predeclared goals despite negative TRAIN associations. This is a
  fixed small set, not a weight grid chosen against development backend results.

## Physical evaluation and interpretation

Use the same eight saved development noises per training seed, unchanged
final legalization/DEF conversion, eight-thread native GRT flow and all seven
native metrics. Verify actual noise hashes across arms. Report each seed's
mean and the three-seed sample standard deviation, paired layout differences,
physical failures, infrastructure failures and missing requests. A high rule
score or Q prediction does not establish better engineering outcomes.

The initial eight arms require at most192 GRT requests. Completed, exactly
bound matching controls can be reused only if their source, training targets,
training stream, sampler and physical settings all match. Any final detailed
route comparison uses a predeclared subset consistently across compared arms;
GRT congestion is a separate diagnostic if final DRT has no equivalent output.
No single best sample substitutes for a policy mean.

New final-route TRAIN64 collection is a separate, matched-label-fidelity
experiment. The existing48 final-route development observations remain
evaluation only. Future final-feedback objectives will compare GRT and DRT
on exactly that new TRAIN subset, with family folds and TRAIN-only scales,
then test a hybrid with one predeclared expert prior. Final confirmation noises
remain unused until the development program is closed.
