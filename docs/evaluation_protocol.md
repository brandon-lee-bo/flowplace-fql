# Evaluation protocol (fixed before real training results)

## Scope and claims

Table 2 reproduction has two distinct tracks: rerun the authors' released DEFs
through their released OpenROAD flow, and generate new layouts with their released
checkpoint before running that same backend. The initial target is the FlowPlace
row on all six circuits, not a retraining/reimplementation of every competitor.
Tool-container version differences and every compatibility patch are recorded.
Fresh paired seeds do not claim to reproduce the paper's unspecified RNG history.

The first FQL experiment is a single-circuit, terminal-action adaptation with a
frozen flow teacher. A fixed graph is implicit context; the entire macro layout
is one action. A flow integration step is not an environment step. No claim of
cross-circuit generalization or full FQL benchmark reproduction is made.

## Splits and selection

Noise families, including all perturbations derived from each teacher sample,
are split together. Only training rewards determine reward normalization.
Critic and actor optimization use training families only. Validation is measured
with the external legalizer before hyperparameter selection. `select_methods.py`
locks one alpha per algorithm family based on pooled mean validation reward over
all training seeds; it never chooses a favorable training seed. The illegal-layout
penalty is part of that reward. Heldout evaluation requires this selection file.
If subsequent debugging uses heldout findings, those findings become exploratory
and a new independent final split is required for a confirmatory claim.

## Controls and costs

Compare original FlowPlace, FlowPlace with one additional legalization pass,
one-step distillation, and FQL-style guidance. Controls using the direct analytic
proxy distinguish the value of learning Q from merely accessing the objective.
All student raw proposals receive the same final legalizer; its changes and
failures are recorded. Q accuracy is checked on student-chosen actions, not just
offline validation examples. Predicted Q is never reported as measured reward.

Report teacher generation, student inference, legalization, data collection,
training, and backend costs separately. Do not compare a warmed actor's compute
against a teacher's compilation-inclusive first call as an end-to-end speedup.
Backend candidates use indices declared in the corresponding frozen plan, rather than the
best layout selected after viewing outcomes. The initial proxy transfer uses the
first three heldout families; the later GRT actor amendment below reserves different IDs.
Repeated training seeds and paired
noise samples are different sources of variation, not interchangeable replicates.

## Metrics

Geometry is independently checked for finiteness, overlap and canvas boundaries.
The inexpensive reward is explicitly the released cache's **macro-only hypernet
HPWL**, including clock nets. It omits standard cells and fixed I/O terminals.
In bp_fe the cache contains only 11 macros and 15 hypernets. Improvement in this
proxy does not establish routed wirelength, timing, power, or area improvement.
Final Table 2 metrics require completed detailed routing and the documented
report stage. Global-route labels, if used later, are a separate intermediate
fidelity and must never be substituted silently for final routed results.

## Backend experiment decisions after proxy results

The initial macro-proxy heldout results are already inspected. Their subsequent
GRT evaluation is explicitly an exploratory cross-fidelity check, not selection
using a still-unseen GRT test set. All 12 validation-selected method/seed groups
are evaluated on the same first three heldout noise families (36 requests).
Do not choose a visually good layout or a favorable training seed for this check.

For the separate GRT-reward experiment, retain the predeclared 96-label family
split. Check critic validation error against a training-mean constant predictor
and report rank correlation and reward variation before expanding policy search.
A fixed-circuit bandit also needs a best-observed-training-layout baseline:
select the minimum measured GRT wirelength among training labels only, reuse
that exact legal layout, and charge its entire data collection budget. Reusing
one deterministic backend result for several noise indices does not create
independent experimental replicates. This control tests whether learning a
policy adds value beyond storing the best layout already evaluated.

## GRT actor amendment before actor results

The timestamped [v1 physical-failure policy](../fql/grt_actor_physical_policy_v1.json)
keeps validation families 192–195 and reserves heldout families 244–247, replacing
the historical future-actor default 240–243. This change avoids proxy/sampler GRT
exploration at 240–242 and is based only on disjoint IDs, not measured quality.
Earlier plans remain unchanged. Macro diagnostics already examined all heldout
noise families, so this isolates GRT exploration inputs rather than making a
claim that these noises have never been inspected in any experiment.

Every alpha must succeed on all 3 seeds × 4 validation families to be eligible
for mean-rWL ranking. Verified physical failures retain the complete denominator;
incomplete configurations and incomplete BC comparisons have no numerical mean.
If no alpha is eligible, record that outcome and do not execute a heldout sweep.
Heldout evaluates the sealed choice without reselection. Unknown, infrastructure,
and parser failures remain errors requiring diagnosis. See the implementation
details in [the integration design](fql_integration_design.md). Twelve successes
are an eligibility rule for this pilot, not proof of population-level reliability.
