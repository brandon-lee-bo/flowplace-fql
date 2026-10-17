# Minimal FlowPlace / FQL integration

Status: the current graph implementation and actual offline-to-online results are described in [graph_fql_offline_to_online.md](graph_fql_offline_to_online.md), with its method below under "Graph FQL offline-to-online pilot". The earlier MLP designs and gates remain historical records. Read against FlowPlace `da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547` and FQL `e8cd16eb490332924dfa2492097219f181765933` on 2026-10-16/17.

## Fresh-flow follow-up, 2026-10-17 evening

The user authorized a careful new attempt after pausing Table 2. The earlier
cached-teacher design below remains as historical context. The new bounded
adaptation is implemented in `fql/fresh_flow.py`, `run_fresh_flow.py`, and
`train_physical_actor.py`:

1. Train a behavior velocity network on offline raw layout proposals with
   Gaussian-to-action conditional flow matching. Integrate for 10 Euler steps;
   there is no intermediate hard projection in this learned behavior model.
2. Train the one-step policy on freshly sampled Gaussian noise throughout,
   matching the current behavior flow for exactly the same noise. Freeze the
   teacher targets. Actor Q gradients do not cross the flow or hard legalizer.
3. For terminal placement, the critic fits measured final rewards directly.
   A low-dimensional physical-feature ridge critic is diagnosed using the
   existing real GRT labels, with penalty selection on TRAIN-family CV. Raw
   features permit actor gradients; executed-layout features are score-only.
4. From identical BC checkpoints, compare a fixed 1500-step BC continuation
   and one fixed alpha=1 GRT-Q continuation at three training seeds and two
   predeclared validation noises. All twelve layouts receive the same GRT
   backend; no candidate is selected by Q and no heldout work is launched.

This fixes the missing fresh learned-flow component of the earlier experiment.
It remains a single-design terminal-action adaptation, with MLP networks and a
fixed circuit context. It does **not** replace the graph-conditioned velocity
network of the released FlowPlace checkpoint, establish cross-design
generalization, or reproduce the original multi-step FQL benchmark. Improved
distillation is a combined change of teacher/prior/training procedure, not an
isolated ablation proving a single cause. Source snapshots and predeclared
budgets are in `fresh_flow_plan_v2.json` and `physical_actor_plan_v1.json`.

The next general implementation, if measured physical outcomes justify it,
would condition both behavior flow and one-step policy on the same netlist
graph. FlowPlace's GAT/attention blocks can then encode circuit structure while
the one-step head directly outputs all macro coordinates. Final hard
legalization remains an explicit external stage. Reliable Q on actor-chosen
proposals, timing tradeoffs, the fixed best-TRAIN layout, and additional circuits
must be checked before a larger comparison. These are future requirements,
not claims that this small MLP pilot already satisfies them.

## What transfers from the papers

[FQL](https://arxiv.org/pdf/2502.02538), Eqs. 5, 7, 9 and Algorithm 1: a behavior flow teacher, a critic, and a separate stochastic one-step student. The student maximizes Q while matching the teacher **for the same noise**. The deployed policy is the student. Teacher outputs are stopped-gradient targets; RL gradients do not pass through the flow integration. The original method also trains the flow teacher by behavior cloning and uses bootstrapped Q learning for multi-step environments.

[FlowPlace](https://arxiv.org/pdf/2604.23658v1), Secs. 3.2–3.3: graph-conditioned velocity prediction over all macro coordinates, with uniform source positions and hard legalization during sampling. The graph contains sizes and pin offsets. Table 2 is downstream OpenROAD PPA, not macro HPWL. Preserve the provided checkpoint, prior, sampling procedure and backend configuration for the reproduction.

## First experiment: fixed-circuit terminal action

Use one circuit as a constant context `s`. One action specifies all movable macro center coordinates. The environment legalizes this proposal, evaluates it, and terminates. This is a contextual bandit, equivalent to horizon-one RL; it is **not** the sequential macro-placement MDP used by MaskPlace. Flow integration time must not be treated as an environment transition unless we deliberately define and collect such an MDP.

The minimal adaptation freezes the pretrained FlowPlace teacher and caches `(noise, teacher_output)` pairs. A fixed-circuit MLP student maps noise directly to coordinates. This deliberately avoids cross-design generalization claims. It is a **FQL-style bandit adaptation**, not an exact reproduction of the original FQL benchmark algorithm: the teacher is frozen, its prior is uniform, the circuit is fixed, and there is no Bellman bootstrap.

Use the frozen teacher to generate multiple seed layouts. Add a separately labeled mixture of small perturbations and legal re-projections to increase quality diversity. Do not regenerate the netlist from each proposed layout: the circuit stays identical. Store failed candidates rather than silently removing their cost. For cached distillation use the actual initial random positions, not a newly sampled noise unrelated to that layout.

## Losses and interface

Let `L(a)` be the deterministic legalizer plus coordinate/DEF conversion and `r(L(a))` the measured reward. Learn:

```text
critic_target = observed terminal reward
critic_loss = mean((Q_j(s, raw_action) - reward(legalized_action))**2)
student_action = clip(student(s, noise), -1, 1), with fixed coordinates restored
distill_loss = mean_over_movable_coordinates((student_raw - teacher(noise))**2)
actor_loss = alpha * distill_loss - mean_j(Q_j(s, student_action))
```

The critic input must consistently be the **raw proposal**, with reward belonging to its actual legalized realization. Training Q on legal coordinates but querying arbitrary illegal actor coordinates creates a new distribution shift. Alternative designs may train a critic on legal coordinates, but then must explain how actor gradients cross legalization; a straight-through estimator would be a separate method change.

Match the official [FQL implementation](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py): coordinate-mean squared distillation, ensemble-mean actor Q, detached teacher, and optional detached absolute-Q normalization. A sum over 2N coordinates changes effective alpha with circuit size. Terminal examples have mask zero; Q target networks are unnecessary here. Use Adam at 3e-4 initially, log gradient norms, and share actor initialization across alpha variants.

NPZ interchange:

| Key | Shape | Meaning |
|---|---|---|
| `noise` | N × D (or N × V × 2) | Actual FlowPlace initial noise; D = 2V |
| `teacher_action` | N × D | Frozen teacher output for exactly this noise |
| `raw_action` | M × D | Proposal before deterministic legalization |
| `action` | M × D | The corresponding executed, legalized placement |
| `reward` | M | Evaluated reward belonging to `action` |
| `split` | N | 0 train, 1 validation, 2 heldout noise |
| `critic_split` | M | Same roles for reward records |
| `movable_mask` | D | Optional coordinate mask; fixed entries never change |
| `fixed_action` | D | Required when there are fixed coordinates |

Keep candidate IDs, parent noise/seed IDs, checkpoint SHA, graph hash, legalizer options, DEF hash, evaluator commit, timing, legality and metric provenance in a neighboring JSON/JSONL manifest. Split perturbations of the same parent together; a random row split alone may leak near-identical layouts. A split of noises within one design tests stochastic generalization, **not** generalization to new circuits.

## Rewards: what can be concluded

Start with exact macro HPWL after legalization, normalized by a fixed training-set reference. This is an inexpensive plumbing/optimization experiment. It cannot demonstrate improved routing, timing, power, or Table 2. Use the upstream hypernet/pin-offset semantics, not blindly a sum of graph-edge Manhattan distances. If only graph edge lengths are available, label them a separate proxy.

For the real objective, generate additional rewards using the same full backend as the reproduced baseline. A useful first target is normalized routed wirelength with timing treated as a reported constraint; do not combine WNS, TNS, area and power with arbitrary weights and advertise a universal improvement. Preserve raw metrics and failure status. Model selection must use validation measurements; heldout backend runs are evaluated only after configuration selection.

Because HPWL is directly computable, compare learned-Q improvement with direct HPWL-gradient/local-search optimization. If the critic merely relearns a cheap analytic objective, it needs a convincing speed/quality or backend-transfer advantage. Few full PPA examples in a high-dimensional action space may be insufficient: report heldout critic error, rank correlation and actor-selected calibration error rather than assuming accurate extrapolation.

## Necessary comparisons

1. Frozen FlowPlace, its original sampling and legalization, one sample per noise.
2. Distillation-only student, same training noise and initialization as FQL variants.
3. FQL-style student, small predeclared alpha sweep (e.g. 1, 10, 100), identical teacher and critic dataset.
4. Frozen teacher best-of-K, selected by the same available metric and charged for all K evaluations.
5. Matched-budget perturbation/local search, and analytic HPWL guidance when optimizing HPWL.

Use at least three training seeds when affordable, paired heldout noise seeds across methods, mean/median/spread, legality rate, legalization displacement, raw/realized reward, predicted Q, and full generation plus legalization latency. Report one-time data generation and training costs separately. Count backend evaluations, wall time and CPU/GPU resource limits. Do not compare an adapted on-design FQL policy to zero-shot FlowPlace without also stating and matching adaptation/search budget.

## Source-level pitfalls to preserve or test

* [`models.py:194–246`](https://github.com/lamda-bbo/flowplace/blob/da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547/flow_matching/models.py#L194) internally samples noise; it has no external-noise argument. A recording wrapper must capture the actual noise without changing the sampler. The hard-constrained sampler supports B=1 and normally compiles the legalizer.
* That sampler's line 240 interpolates from the **current** `x`; the paper Sec. 3.3 shows interpolation from initial `x0`. Record the difference; do not silently change it during reproduction. The implementation is the first operational reference.
* [`hard_constraint_legalizer.py`](https://github.com/lamda-bbo/flowplace/blob/da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547/flow_matching/hard_constraint_legalizer.py#L99) returns the original position when no valid grid point exists. Its computed `movable_sorted` is not actually used by `self.order`. Therefore independently check bounds, overlap and fixed-pin preservation; do not infer legality from the function name.
* Student inference uses one final projection while the teacher projects repeatedly. This difference is part of the deployed student and must be evaluated, not hidden. Distillation-only establishes how much quality is lost just by collapsing the generator.
* Keep fixed port coordinates unchanged and use normalized **center** coordinates, not bottom-left DEF locations. Validate conversion against the provided DEF, accounting for macro dimensions, units and orientations.
* Do not backpropagate through legalizer sorts/argmax or claim that `no_grad` projection provides useful actor gradients. A reward-trained raw-action critic is the explicit differentiable surrogate.

## Decision after the first run

If the baseline backend completes, first verify reference DEF and generated DEF on one design. If student distillation collapses, fix it before adding Q; if Q improves predicted scores but worsens realized scores, constrain/retrain the critic and stop escalating the alpha sweep. If HPWL improves but PPA worsens, that is a useful negative result supporting backend-aware rewards. Expand to other Table 2 designs only after coordinate, legality and metric provenance checks pass.

## Implemented optional controls and latency

`fql/train.py --analytic-hpwl-cache CACHE --analytic-alphas ...` adds a same-student analytic control. It shares the exact warmup, initialization and optimization batches with the learned-Q variants but minimizes distillation plus normalized differentiable macro HPWL. This answers whether a learned critic improves on an available cheap objective. Its analytic gradient acts before legalization; the critic instead learns the composed raw-proposal-to-legalized-reward function. A difference can therefore reflect learning the legalizer as well as the objective, and is not a pure value-network architecture comparison.

`fql/coordinate_control.py` starts from each teacher layout paired to the recorded noise and performs coordinate Adam with HPWL plus optional displacement anchoring. It is charged for teacher sampling and all test-time iterations. It outputs proposals for the same final legalizer. One deterministic `seed_control` is recorded, not duplicate seed copies. Validation chooses steps/lr/anchor settings; final test uses those frozen settings.

`fql/latency.py` measures warm single-layout actor latency and batched throughput independently with explicit CUDA synchronization. Training writes these measurements automatically. Neither metric includes legalization, transfer, noise generation or downstream evaluation, so end-to-end speed claims require those additional costs. The coordinate optimizer separately records batched runtime including its Adam setup and logging; this is not an actor-only latency comparison.

## Why a fixed-circuit MLP may fail to distill, and falsifiable tests

The student is not unconditional: it maps the recorded noise to a layout under a constant circuit context. For one fixed circuit, omission of a graph input does not remove varying information. A sufficiently expressive MLP can represent a deterministic noise-to-layout map. The practical concerns are sample complexity and the teacher's nonsmooth mapping: greedy grid selection can abruptly change macro choices as noise changes, and a small finite teacher cache may not cover those regions. Conditional-mean predictions can then average distinct layouts into overlapping coordinates.

Do not conflate these issues with arbitrary permutation symmetry. Macros have identities, different sizes and connections; matching or permuting them freely changes the circuit. Graph relabeling requires permuting the graph, noise, features, masks and output together. Further, the released greedy legalizer follows input macro order, so even an equivariant graph backbone does not make its legalized teacher fully permutation equivariant. A graph student might need an order/identity feature to reproduce that teacher. Reflection/rotation augmentation is also not automatically valid with fixed pins, orientations and unequal canvas axes.

Run the following in order before replacing the MLP:

1. **Fit versus generalization:** compare training and validation distillation MSE; `diagnose_distillation.py` also compares with the training-mean layout and nearest-training-noise baseline. Large training error suggests capacity/optimization trouble; near-zero train error and mean-baseline-level validation error indicate insufficient coverage or an unsuitable inductive bias.
2. **Diversity versus actual quality:** compare teacher/student per-coordinate standard deviation, unique legalized-layout counts, legalization displacement and actual reward. Low diversity alone is not failure for a single fixed optimization instance, but mode averaging with large legalization corrections is a mechanism to investigate.
3. **Noise sensitivity:** generate a small Slurm test set of `(z, z+epsilon)` pairs for several epsilon values, holding the circuit and legalizer fixed. Compare teacher and student output distances and count which macro grid choices change. This directly tests whether legalizer discontinuities dominate the mapping. Keep these new pairs outside the critic/test data unless explicitly starting a new experiment.
4. **Data scaling:** repeat the BC-only student at increasing counts of independent teacher noise families, with a fixed heldout-noise set and identical evaluation budget. Do not obtain more data by duplicating examples or replacing their noise with unrelated samples. A learning curve distinguishes missing data from a fundamental representation issue.
5. **Controlled architectural alternative:** after a clear BC gap exists, compare a residual MLP (`z + delta(z)`) and a graph-conditioned one-step student initialized from the FlowPlace backbone, under the same data/parameters/runtime budget as appropriate. Both remain coupled to the same noise; neither justifies arbitrary Hungarian matching of macro coordinates. Q optimization should wait until the corresponding BC-only baseline is characterized.

## Budgeted real-backend relabeling interface

`fql/relabel_backend.py` reads the original NPZ plus `prepare_backend_tasks.py`'s JSONL mapping from original critic row indices to completed backend artifacts. It does not launch or retry backend runs. Each row binds the source dataset hash, canonical little-endian float32 action hash, and exported DEF hash to its successful backend manifest. The original predeclared `selection_plan.json` is checked automatically when present, so an unsuccessful requested row cannot disappear from the mapping. A separate subset opt-in permits exclusion of unrequested rows for budget reasons; it never permits dropping a failed requested row. Invalid/missing labels leave an explicit audit and fail.

All teacher/noise/split rows are preserved exactly. Only selected critic-indexed arrays are subset, and original indices and group/parent links remain. The objective is negative measured global-route wirelength, normalized by the mean over the retained **training** critic rows only; validation and heldout labels do not affect scaling. WNS, TNS, power and area remain separate measured diagnostics. Original proxy rewards/stage and full backend provenance remain available. This is a real-backend intermediate-stage experiment, not final detailed-route Table 2 evaluation.

The expected initial budget is 32 training, 8 validation and 8 heldout noise families, two predeclared variants per family (96 backend requests). Hold the row plan fixed before obtaining outcomes. Backend runtime and failures belong in the experimental record. A backend-trained critic must be calibrated against backend rewards rather than macro HPWL. Any analytic HPWL control in this experiment needs its own explicitly retained proxy normalization and should be described as testing transfer from a different objective.

## Implemented teacher-sensitivity diagnostic

`fql/probe_teacher_sensitivity.py` takes the first eight training noises from the deterministic real dataset. One preseeded Gaussian direction per point is normalized to unit RMS over movable coordinates. It applies both signs at epsilon `1e-5`, `1e-3`, and `1e-2`, clips noise to its original support, and records the actual perturbation after clipping. The same model instance evaluates these pairs with hard projection enabled and disabled; checkpoint, initial seed, 20 sampling steps, current-state interpolation and output clipping are unchanged. No extra final legalization is applied. Neither unprojected output legality nor quality is assumed.

Each original noise is evaluated twice per mode, requiring bit-identical output and full trajectories. Hard-mode originals must also match the dataset's cached teacher bit for bit, so incompatible arithmetic, source drift or numerical nondeterminism fail the run. The artifact contains all input/output arrays, per-step trajectories, final output MSE, maximum coordinate change, maximum macro displacement and output/input RMS ratios. This is a paired local finite-difference probe, not a proof of global discontinuity or an independent-seed quality experiment. Stronger sensitivity with hard projection would support one mechanism behind limited-data distillation difficulty; it would not rule out model capacity, sample coverage or other sources of error.

Measured result: Slurm `1784060` completed successfully in 76 seconds, using the pinned shared environment (`torch 2.2.1+cu121`, NVIDIA L20). All 16 repeat comparisons and eight cached-teacher comparisons were exact. At epsilon `1e-5`, hard projection produced mean output MSE `0.313862`, versus `2.68377e-6` without hard projection; all 16 hard perturbations had a maximum coordinate change exceeding 0.01, versus none without projection. At epsilon `1e-3`/`1e-2`, the corresponding mean MSE values were `0.325414`/`0.423858` with projection and `3.29651e-6`/`4.92576e-6` without. This identifies a substantial local sensitivity mechanism in the actual teacher mapping, beyond repeated-run randomness. The no-hard ablation is not automatically a better placement generator; evaluating its legalized quality is a separate experiment. Increasing paired teacher samples or replacing the student architecture should not be assumed to solve such sensitivity without a measured learning curve.

## Locked small-data GRT experiment: gate before actor expansion

The initial real-label budget is 64 train, 16 validation and 16 heldout critic rows from 32/8/8 noise families. There are only 32 independent training families, not 64 independent designs, and only one fixed 22-dimensional action space. The legalizer's nonsmooth response and the measured teacher sensitivity make extrapolation a serious concern. A larger critic/actor sweep must not be justified merely because labels exist.

The critic gate is fixed **before observing those labels**: three seeds `0,1,2`, two critics per seed, hidden widths `64,64`, 1,000 Adam steps, batch size 64, learning rate `3e-4`, weight decay `1e-4`, gradient norm cap 10. This is deliberately a different capacity from the initial macro-proxy `256,256,256` critics. `fql/critic_gate.py` fits training rows only and reports train/validation MAE, RMSE, Spearman/Pearson, prediction/label variation and the training-mean constant baseline. No heldout predictions, losses or aggregate label values enter the gate. The advisory continuation rule is that at least two of the three seeds beat the constant validation MAE and have positive validation Spearman. This is a sanity gate, not a significance test or an automatic authorization to spend more backend budget. Poor support calibration should stop expansion rather than trigger an unlimited search for a positive result.

Every gate saves its exact critic checkpoint. If actor work proceeds, `train.py --pretrained-critic-dir GATE` reuses that critic, validates its dataset/seed/dimension, and keeps it frozen. Actor and critic widths are separate checkpoint fields and may differ. GRT datasets automatically suppress heldout critic and actor diagnostics during training; raw candidate arrays are only measured after the selection protocol permits it.

An essential baseline is **best observed training layout**: choose the lowest GRT rWL among the 64 legal training actions, with source-row tie-breaking. This baseline requires no learned policy for a fixed-circuit optimization task. `critic_gate.py` preserves its exact executed action, source row, full original DEF/metrics/manifest and hashes. Charge the entire eligible training-label search and teacher/data-generation budget. It is one measured fixed layout, not an independent observation for each evaluation noise or seed. Never relegalize it and reuse the old backend metric without verifying that the executed DEF is still identical.

If the critic gate justifies a bounded actor experiment, the proposed next budget is validation noise indices **192–195** × three seeds **0,1,2** × `[BC, FQL alpha=1, FQL alpha=10, FQL alpha=100]` = **48 new candidate GRT runs**. Alpha=1 was added before observing GRT validation labels because the original proxy and GRT rewards both use training-mean normalization and excluding stronger guidance would narrow the original sweep prematurely. Freeze these methods, seeds and both validation/heldout index lists before requesting candidate outcomes. Select one alpha across all seeds by mean validation rWL, preserving BC and fixed training baselines. Then heldout noise indices **240–243** × three seeds × `[BC, selected FQL]` = **24 new runs**. Teacher-relegalized variant-0 labels (four per phase) and one best-TRAIN label may be reused only for exact executed-DEF/backend matches; reuse does not create independent replicates. This is a proposed next stage, not an already launched experiment. The existing macro-selected candidate transfer experiment remains separately labeled exploratory and cannot select the GRT-trained method.

`fql/aggregate_backend_candidates.py` checks every frozen `method/index` task, source action artifact/hash, rounded-DEF geometry audit, exact backend input DEF, stage, implementation provenance and success status. Missing/failed requests remain explicit and cause failure; it never generates a winner comparison from successful survivors only. Results preserve seed-level means and per-noise paired comparisons, with timing/power reported separately. Macro-trained Q values are not calibrated against routed labels. GRT Q calibration additionally requires a bound source dataset manifest and the exact training-only normalization used by its critic.

`fql/seal_grt_selection.py` accepts only complete GRT validation reports with a predeclared heldout index list, identical seed/noise budgets, BC, teacher and a single best-TRAIN reference. It locks one alpha globally across seeds, records whether it even beats BC/best-TRAIN, and does not launch heldout work. Heldout aggregation requires this sealed selection and matching backend provenance; exploratory transfer results are rejected as a selection source. A complete negative result remains valid evidence: lower validation rWL than another alpha does not imply improvement over simple training-layout reuse or better overall PPA.

## Pre-actor amendment: bounded physical-failure eligibility and GRT input isolation

The historical strict plan above remains recorded. Before generating any real-GRT actor results, the experiment adopts an **explicit optional** policy, frozen in [`grt_actor_physical_policy_v1.json`](../fql/grt_actor_physical_policy_v1.json), with its original UTC freeze time and SHA bound into every candidate plan. Default aggregation remains strict; the existing exploratory-only mode remains separate. The policy is accepted only for bound real-GRT actor training, validation and sealed heldout artifacts, never for the proxy-selected exploratory comparisons or critic-label collection.

Validation remains seeds **0,1,2** × noises **192–195** × BC/FQL **alpha 1,10,100**: 48 new runs, four exact teacher reuses and one fixed best-TRAIN reference. Every predeclared case must have an independently verified terminal outcome. An alpha is eligible only when **all 12 of its requested validation cases succeed**. Choose the eligible alpha with minimum mean rWL across the three seed means, with the original lexical tie-break. A configuration with any verified physical failure retains all requested cases and its failure fraction, but has no numerical configuration mean. BC failure makes `validation_beats_BC=null`, not true. If every alpha is ineligible, seal a `no_eligible_configuration` stop with **no executable heldout plan**. This is a small-budget feasibility condition, not evidence of a population zero failure rate.

Only the independently classified CTS `RSZ-0060` maximum-buffer-count error is currently admissible, using the unchanged exact action/DEF/audit/task/manifest/raw-log bindings and terminal Slurm evidence. Unknown, infrastructure, preemption, missing and parser failures remain hard errors. No failure may be silently removed, replaced or retried to obtain a preferred placement outcome. Training relabeling and the critic gate stay strict. At heldout, report the locked alpha even if physical failures occur; preserve denominators, suppress incomplete configuration means, and never reselect alpha.

The **future actor heldout IDs change to 244–247**, within the already budgeted eight heldout label families. This isolates them from proxy/sampler GRT exploration using 240–242; it is an ID-overlap amendment, not a quality-based selection. Earlier plans and artifacts using 240–243 remain unchanged. Prior macro diagnostics inspected all heldout noises, so this amendment isolates **GRT cross-fidelity exploratory inputs only**, not all prior information. The exporter retains its historical default; new actor commands must explicitly pass `--heldout-indices 244,245,246,247`, and the new policy rejects the old IDs.

Use `prepare_grt_candidate_tasks.py --physical-failure-policy fql/grt_actor_physical_policy_v1.json --heldout-indices 244,245,246,247` for both phases, retaining the same policy file and source bindings. Aggregate with `--allow-preregistered-physical-failures`; this flag cannot be combined with `--allow-exploratory-physical-failures`. Seal validation through the existing `seal_grt_selection.py` command. A selected decision carries the policy SHA and locked methods into heldout; a stop decision is rejected by heldout export. Successful references are reused once with their original provenance, never reinterpreted as new failed actor runs.

Execute only `pending_tasks.jsonl`, but pass the **full `tasks.jsonl` to terminal-failure classification and aggregation**: the physical-failure sidecar's task-manifest SHA must bind the exact full requested comparison, including reused references. Binding a sidecar to the pending-only manifest will correctly fail aggregation. Complete individual seeds may still have diagnostic means/pairs even when their alpha is ineligible; these are not a substitute for the suppressed whole-alpha mean.


## Graph FQL offline-to-online pilot (2026-10-18 HKT)

This is a new experiment under `/fact_data/bohanli/flowplace_fql/graph_o2o_v1`.
The old Table 2 tasks remain paused. It supersedes neither the old ridge-Q/MLP
results nor their negative calibration findings. Official FlowPlace is pinned at
`da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547`; FQL at
`e8cd16eb490332924dfa2492097219f181765933`.

The environment is a fixed-netlist terminal placement task. A raw bounded action
contains all 11 macro centers. Its reward is obtained after the released
legalizer, DEF rounding, native track snapping, standard-cell placement, CTS,
and global routing. Flow integration steps are not environment transitions.
The terminal Bellman mask is zero, so target-network bootstrapping vanishes.
The measured Q target is `-GRT_rWL_um / 2341121.25`, with that TRAIN-only scale
frozen throughout online learning. A declared utility of -4 is reserved for
verified terminal physical/geometry failures, with no invented PPA values.
Unknown tool, parser, storage and infrastructure failures do not receive it.

`fql/graph_o2o_models.py` uses independent BC-flow and one-step actor policies,
each retaining FlowPlace's 6,285,058-parameter GATv2/global-attention backbone.
The BC flow uses flow-matching regression and ten Euler integration steps for
distillation; the one-step actor minimizes `10 * distill_MSE - mean(twin_Q)`.
Its matched BC control sets the Q coefficient to zero. Twin critics are separate
128-wide feature MLPs with TRAIN-standardized coordinates and 17 physical
features. Uniform [-1,1] noise, graph weights, fixed-graph critics and the
external legalizer are explicit placement adaptations of official FQL, not an
unchanged run of its Gaussian/MLP implementation. No derivative through the
hard legalizer is asserted: Q learns the reward of the entire execution map.

The initial critic dataset contains 64 measured TRAIN and 16 validation rows.
BC-flow training also uses 192 unlabeled TRAIN teacher layouts. Heldout labels
are not used. Common warmup is 300 flow steps, 800 actor-distillation steps and
2,000 critic steps. Both arms then receive 1,000 offline updates. Two subsequent
online rounds each use 600 updates per arm, incorporating their own newly
measured collection actions. Q and BC-flow batches mix online data at 50% when
available, matching the official code's optional balanced_sampling=1 mode;
the FQL paper instead defaults to unbalanced append-and-sample replay. Failed actions train Q but are not imitation targets. Learning rates
are 1e-5 for each graph policy and 3e-4 for twin critics; batch size 16.
These are a bounded pilot configuration, not a claim of convergence.

Each stage emits eight fresh collection and four fixed evaluation noises per
arm. The original FlowPlace also emits the four paired evaluation noises at
round zero. All predeclared samples count; no Q/noise selection is performed.
Only collection samples enter arm-local replay. After online round two, each
arm has actually trained on 16 new measurements; its final eight collection
samples are retained for diagnostics/possible continuation, not retroactively
claimed as training data. The complete three-stage pilot requests 76 new GRT
outcomes. Four noises are not four independent training seeds.

`graph_o2o_controller.py` journals Slurm handles and resumes only after actual
scheduler evidence. It first tests one physical sample, then uses cpu-scavenger
arrays with eight CPUs per task and concurrency four; later GPU updates use
GPU scavenger. `graph_o2o_backend.py` binds actions, checkpoints, DEFs, measured
metrics and the offline physical-flow revision. `graph_o2o_collect.py` builds
replay and complete-denominator summaries. Physical metrics are reparsed from
raw output; fixed evaluation samples never enter replay. The official native
track snapping can move macros: read-only ODB audits preserve actual macro
boxes and check overlap/bounds at zero DBU tolerance. The offline TRAIN and
validation audit passed 80/80 saved GRT layouts (job 1792376).

Durable compression preserves every results/objects file, with a full decoded
per-file SHA audit before replacing intermediate directories with scratch cache
links. Logs, reports and original physical manifests remain durable. Archive
checks passed in job 1792365; that synthetic storage fixture is not a physical
measurement. Job 1792367 separately validated the real saved-ODB geometry audit.

Full-route comparison is separately frozen in `full_route_plan.json`: four
paired evaluation noises for original FlowPlace, BC offline, FQL offline, BC
online round two and FQL online round two, plus one best-observed TRAIN reference
(21 total requested layouts). All use the same source GRT checkpoint, tool,
Nangate45 files and common detailed-route cap20. This deliberately differs from
the historical released cap4 and must not be mixed with those Table 2 rows.
Native DRC counts, canonical raw JSON PPA, final macro preservation, failures,
and nonzero DRC remain explicit. The best-TRAIN reference is source row12,
GRT rWL 2,159,262 um, selected solely among the 64 TRAIN labels before new
online outcomes. It is one fixed reference, not four independent replicates.

The original graph pilot actually completed all76 measured GRT requests, both
online updates and21 cap20 final routes, all DRC0. Its final online FQL mean
rWL is1.30% worse than matched BC. Three predeclared offline training seeds
also completed native final evaluation: mean rWL is0.405% lower for FQL, with
one seed worse and worse average WNS/TNS. The complete multiseed online and
second-circuit comparisons remain in progress; see the current experiment
record for exact denominators and source evidence. Lower losses or these
partial comparisons do not establish completion or a stable RL advantage.
