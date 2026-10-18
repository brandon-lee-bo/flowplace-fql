# From behavior-flow labels to matched actors

This is a development protocol, written before the current-flow source study
finishes. It does not select a winning source or authorize a result claim.
The source study changes Q labels while keeping the original 256 behavior
targets fixed. Its native request budget is separate from later online replay.

## Gate before another actor experiment

1. Complete the teacher's 80 declared requests, including ordinary recovery of
   preempted attempts. Retain failed requests and their actual status.
2. Verify all six assembled source datasets and all 36 critic fits. Read Q
   scores separately on old historical validation and fresh teacher validation;
   inspect base-only sensitivity and family-local comparisons.
3. Examine the already-declared positive/negative/random direction probes.
   Useful ordering alone is not evidence that an actor can follow Q gradients.
4. Record the proposed source/critic contrast and its reason in a dated review
   note. Preserve negative prediction results rather than choose a favorable
   metric after looking at physical actor outcomes.

The clean first source contrast is O64 versus T64 using the same critic
variant. M64 tests coverage at the same label budget. A second critic variant
is an additional factor, not a replacement baseline chosen per source.
Start with one justified mechanism, then expand if its physical results warrant
it. The current T64 teacher is common seed 0 only; three critic/actor seeds do
not create three independent teacher-source distributions.

## Common physical utility and distillation strength

The legacy critic for source j predicts approximately -WL/S_j, where S_j is
that source's successful initial TRAIN mean WL. Merely retaining alpha=10 for
every source changes the effective physical Q strength. Fix a reference S_ref
from the original O64 TRAIN data, and use:

    actor_loss_j = 10 * distillation - (S_j / S_ref) * mean(Q_j)

This makes the physical target -WL/S_ref for every source. The input/target
normalizers and Q regression remain fitted on that source's TRAIN only. Q
outputs include their stored reward normalization; do not multiply by their
reward standard deviation a second time.

For the multi-metric critic, its predicted costs are already restored to
native units. Fix reference preferences w_ref and reference TRAIN cost scales
S_ref,k. With each critic's stored objective scales S_j,k, use:

    w_j,k = w_ref,k * S_j,k / S_ref,k
    Q_j = -sum_k(w_j,k * predicted_cost_j,k / S_j,k)

Thus Q_j uses the same physical utility for all sources. A single alpha cannot
correct changed relative weights between several metrics. Keep reference
scales separate from fitted neural target z-scores. Failure penalties also
need common-utility semantics if any failure enters replay; the current frozen
actor setup must not be handed NaN initial labels without an explicit tested
failure-handling adaptation.

## Matched execution and interpretation

Share the three audited common flow/actor initializations, actor minibatch/noise
streams, learning rates, update budgets, distillation coefficient, uniform
prior, 10-step teacher, final legalizer, fixed-node handling and physical
backend. Freeze flow initially. The independent one-step actor retains the
released graph backbone; the source contrast does not redesign that backbone.

Retain a matched BC control and current FQL reference. Online comparisons use
each policy's own measured collection layouts, the same initial/online replay
ratio and the same requested label budget. Their subsequent layouts may
differ: later-round differences are adaptive total effects, not matched
per-layout reward comparisons. Validation and direction probes stay out of
replay. A frozen-versus-updated-flow contrast reuses the exact completed round-0
prefix and counts inherited measurements once.

Compare full requested policy samples and independent training-seed means,
including failures and every recorded native metric. Report a best measured
candidate only against another equally selected candidate with the same
measurement budget. A lower actor loss, higher predicted Q, lower validation
MAE, or more source labels alone cannot establish better physical placement.

The first actor/DRT results are development evidence. After choosing a method,
lock new evaluation noise and run a fresh confirmation with multiple training
seeds. GRT congestion and final DRT PPA remain distinct measurement stages.
