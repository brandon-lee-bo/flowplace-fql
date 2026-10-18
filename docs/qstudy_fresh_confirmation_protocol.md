# Fresh confirmation gate

This protocol reserves a new evaluation stream. It does not select a method,
launch evaluation, or resume the paused Table 2 task. Development includes the
existing validation sets, noise 4600000, source controls and bounded-objective
experiments. None of those outcomes counts as fresh confirmation.
The selection record must inventory every previously inspected outcome,
including the earlier 900000 evaluations, candidate pools, direction probes
and teacher-source noise 5300000; none may be relabeled fresh.

Before generating or evaluating fresh layouts, publish an immutable selection
record with the complete development reports, the scientific reason for the
choice, one candidate configuration and all three selected policy checkpoints.
Lock its training/online round, code closure, objective version, TRAIN reference
scales, all baselines, hardware/software and physical-flow revisions. A negative
development result is valid: the existing FQL anchor can be confirmed without
claiming that a new mechanism improved it. Do not choose a best training seed.
The record must also fix the primary physical endpoint, acceptable changes in
the other metrics, failure accounting and the rule for calling a result an
improvement or a tradeoff. Choose those rules from development evidence before
fresh measurements, rather than selecting a favorable endpoint afterwards.
Pin the design and graph/cache identity as well. Confirmation on bp_fe alone
supports a claim about that design, not cross-design generalization.

Use CUDA noise seed 85000000 and eight actual saved noise arrays, shared across
every method and all three independent policy-training seeds. Preserve and
verify the actual arrays; an integer seed alone is insufficient. Do not feed
these layouts or their labels back into any model or choose checkpoints with
them. Fix every candidate-generation/selection rule before export; report all
requested outputs, not just a favorable selected layout.
These are new generation-noise observations for the three locked trained
policies, not three new training replicates. Existing exporters are bound to
development noise 4600000: use a separately reviewed evaluation-only exporter
with checkpoint/source bindings, rather than modifying frozen training configs.

Include the matched BC control and current behavior flow. Include the released
FlowPlace checkpoint with its official hard-constraint sampler as a separate
reference. That single checkpoint has eight unique noise outputs; copying it
into three directories would not create three independent training seeds.
Explain any effective reuse of BC checkpoints and retain its certificate.
New native measurements are required for the new actual noise inputs.
If the candidate is a new source or bounded-objective method, also include the
existing legacy FQL anchor. Lock matched offline/online rounds, warmup and flow
sources, training labels and added-backend-call budgets, and disclose any
remaining differences. The legacy anchor is unnecessary as a duplicate arm
when it is itself the selected candidate.

Use the same eight-thread GRT evaluator and unchanged native metrics. Report
unresolved requests, confirmed failures, macro legality, all seven physical
metrics, per-training-seed means and their sample SD. Recompute any declared
utility from native outcomes under one locked TRAIN reference; never compare
scores normalized by different source means. Failed runs have no invented PPA,
and infrastructure interruptions receive no physical-failure target.
Keep the complete eight-request denominator per training seed. Report PPA
means as success-conditional, with missing requests and failure rates beside
them; incomplete requests do not constitute a complete policy mean. Public
checkpoint comparisons are descriptive and never create three independent
public training replicates.

For final detailed routing, preselect noise indices 0--3 in every stream before
inspecting fresh GRT results. Use the same locked route/repair settings and cap
20 for all methods. Report actual routed length, signed WNS/TNS, DRC, power,
area and failures. Retain nonzero DRC counts rather than repairing only one
method until it succeeds. GRT congestion and final-route metrics describe
different stages; do not fabricate final congestion from GRT summaries.
Record logical requests and unique physical calls separately. Only certified
identical inputs and evaluator settings may reuse measurements; reused calls
are not new independent evidence.

Any additional adaptation after fresh results creates a new development
iteration and requires a separately locked confirmation stream. Three training
seeds and eight noises still give limited uncertainty evidence; distinguish
training variation from within-policy output variation.

The evaluation-only entry point is `fql/qstudy_fresh_eval.py`. Its `validate`
and `declare` stages require a selection JSON and its SHA256, but create no
noise arrays, load no policy and invoke no backend. `export` additionally takes
one declared stream and training seed and requires a Slurm CUDA allocation.
It preserves the selected original training round separately from the fresh
native request's logical round0. No critic or optimizer is constructed.

The selection file binds original configs, completed checkpoint/replay chains,
the code closure, graph cache, geometry inputs, common native evaluation data,
and hardware/precision settings. Shared actual CUDA noise bytes are published
once under a lock. Completed request exports are immutable and revalidated
on resume, including public-flow trajectories and preselected DRT indices.
The public checkpoint uses its official20-step hard-constraint sampler;
current behavior flow uses10 Euler steps, and one-step actors use one forward
pass. All receive the same saved noise arrays and final external legalizer.

Sixteen CPU contract tests passed in Slurm1796984. These validate software
contracts with synthetic fixtures, not a selected scientific comparison.
The final selection record still requires review of actual method identities,
matched offline/online budgets, common utility scales and acceptance rules.
No real fresh noise, layout or physical outcome has been generated yet.

The separately reviewed native finish adapter is `fql/qstudy_fresh_native.py`.
Its `prepare` stage requires the immutable selection and every completed export
inventory. Submit its `controller` through `qstudy_fresh_native.slurm` with the
absolute plan path and plan SHA256. Only its eight-thread Slurm worker calls
the unchanged native evaluator. Completed measurements are verified and reused;
genuine infrastructure interruptions are retried without inventing PPA labels.
The read-only `fql/qstudy_fresh_report.py` reports all declared requests, saved
actual-noise pairing, physical failures, unresolved requests and all seven GRT
metrics. Sixteen contract tests passed in Slurm1797068. These modules do not
select a candidate or generate fresh noise.

Detailed routing uses the separate `fql/qstudy_fresh_route.py` adapter and
`qstudy_fresh_route.slurm`. Its `prepare` and `controller` accept the selection
and sealed plan bindings respectively. It predeclares all locked streams and
indices0--3, then continues only their verified saved GRT inputs using the exact
existing `graph_o2o_full_route` worker, eight threads and cap20. Its reporter is
`fql/qstudy_fresh_route_report.py`; it preserves nonzero DRC and distinguishes
upstream confirmed failures from unresolved routing/tool attempts. Thirteen
contract tests passed in Slurm1797107. No actual fresh routing has started.

The metadata-only `fql/qstudy_fresh_lock.py template` prints a request template;
`build --request PATH --sha256 SHA` requires the root's explicit scientific
selection, original checkpoint rounds, prospective acceptance rules and complete
development inventory. It hashes/copies metadata and validates the selection,
without declaring an evaluation or loading tensors/noise. Five contract tests
passed in1797112, including prospective binding of the native and detailed-route
code closure. Software validation does not replace review of the selected
scientific comparison and matched training budgets. The production builder has
not yet been invoked.
