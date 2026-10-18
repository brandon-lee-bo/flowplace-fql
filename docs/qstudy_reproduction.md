# Q study: inputs and execution entries

The [decision table](qstudy_decisions.md) and [online results](qstudy_online_results.md)
describe measured outcomes. The [literature notes](qstudy_literature.md)
separate referenced algorithms from implemented adaptations. This server-side
guide does not launch work or resume the paused Table2 controllers.

## Required artifacts

Run from `/fact_home/bohanli/RL_Placement/experiments/flowplace_fql` with the
repository `.venv`, pinned upstream sources, public checkpoint, benchmark
cache and OpenROAD container restored as described in the root and backend
READMEs. Git contains code and small result artifacts, not training data,
weights or native EDA outputs.

The study root is `/fact_data/bohanli/flowplace_fql/qstudy_v1` (`BASE` below).
`data_audit/{bp_fe64,bp_fe384,bp_be64}/dataset.npz` and their manifests bind
the audited TRAIN/validation data. Actor configs bind exact cache,
common-initialization and critic checkpoint paths and SHAs. Some original
paths are verified scratch cache links backed by durable archives; restore
their contents before rerunning a verifier or trainer.

Training resume requires the same data, source, optimizer and RNG contract.
Changing a frozen configuration requires a new experiment directory and run
prefix. The current study uses fixed budgets and every predetermined noise,
not a best checkpoint. Computation runs through Slurm.

## Execution map

Wrappers set the project directory and interpreter. Check each module's
`main()` or run its `--help` through a small CPU Slurm allocation for arguments.

| Stage | Entry | Inputs and outputs |
| --- | --- | --- |
| Independent Q fits | `fql/qstudy_screen.slurm` → `fql.qstudy_screen` | Audited dataset, variant, seed,2000 steps,3 family folds; fold/refit checkpoints and metrics. `fql/qstudy_report.py` validates the declared matrix. |
| Behavior-flow labels | `fql/qstudy_teacher.slurm` → `fql.qstudy_teacher` | Fixed common teacher,40 noise families and one local variant each; saved noise/raw/executed proposals and backend requests. Measurement is a separate step. |
| Source comparison | `fql/qstudy_source.slurm` → `fql.qstudy_source` | `plan → check → assemble → train → report`; original/current-flow labels, frozen splits,36 tasks. See the [source protocol](qstudy_behavior_source.md). |
| Bounded objectives / conservatism | `fql/qstudy_bounded_screen.slurm`, `fql/qstudy_com_screen.slurm` | TRAIN-only scales and family folds; separate bounded and COM-inspired fits. Report modules do not train actors. |
| Main actor configuration | `fql/qstudy_cpu_stage.slurm fql.qstudy_config` | Completed critic refits and common initializations → immutable actor `config.json`. Source/bounded actors use their own audited adapters. |
| Actor train/export | `fql/qstudy_actor.slurm` → `fql.qstudy_actor` | `--experiment`, `--seed`, `--round`, `--stage train/export/all`; checkpoints, learning curves, actual noise and layout requests. Round0 adds1000 actor updates; rounds1/2 add600 each. |
| Offline-to-online controller | `fql/qstudy_controller.slurm` → `fql.qstudy_controller` | Explicit experiment list and concurrency; train/export → collect measurements → update replay → continue. Genuine interruptions use the recorded resume/recovery contracts. |
| Native GRT | `fql/qstudy_backend.slurm` → `fql.qstudy_backend`; batch wrapper dispatches through `fql.qstudy_controller` | Bound sample request and fixed physical setup → native outcome, seven metrics and geometry proof. Final round is evaluation-only. |
| Read-only actor report | `fql/qstudy_readonly_report.slurm -m fql.qstudy_actor_report` | Saved configs/checkpoints/requests/native results → new immutable JSON/CSV snapshot; no model forward, training or EDA. |
| Detailed routing | `fql/qstudy_final_route.py` and `fql/graph_o2o_route_controller.slurm` | Explicit request/plan, fixed first4 noises and repair cap20 → WL/WNS/TNS/DRC/power/area. GRT overflow remains separate. |
| Fresh confirmation | [Launch checklist](qstudy_fresh_launch_checklist.md) | Reviewed selection and full development inventory → new shared noise → inference-only exports → GRT/DRT → saved-report acceptance reducer. Fresh labels never enter replay. |
| Durable native proof | `fql/qstudy_preserve_native.slurm` → `fql.qstudy_preserve_native` | SHA-bound plan → scratch stage/full decode → separately reviewed capacity admission. Original weights and referenced historical home evidence remain retained. |

Main directories are `actors_v1/{bc_frozen,legacy_frozen,legacy_updated}`.
The updated arm imports exact frozen-FQL round0; its48 inherited requests
are not new physical calls. `flow_eval_v1` is the common10-step teacher;
`public_flow_eval_v1` is the separate released20-step hard-constraint sampler.
See [cost definitions](qstudy_cost_inventory.md) for logical requests versus
real attempts, and immutable reports for actual success/failure counts.

## Expert-rule and final-feedback extensions

These are separate experiments from the historical online matrix above.
Their fixed definitions, data denominators and stage-specific metrics are in
the [expert-rule protocol](qstudy_rule_reward_protocol.md) and
[final-feedback protocol](qstudy_final_feedback_protocol.md). A prepared
configuration or completed software test does not establish native PPA gains.

| Stage | Entry | Contract |
| --- | --- | --- |
| Eight expert-rule critic objectives | `fql/qstudy_rule_reward_screen.slurm` → `fql.qstudy_rule_reward_screen prepare/run` | Fixed384-row TRAIN, three family folds plus refit, three critic seeds; TRAIN-only utility scales. |
| Expert-rule actor/export | `fql/qstudy_rule_reward_actor.slurm` → `fql.qstudy_rule_reward_actor prepare/run` | Eight objectives, three actor seeds, fixed1000 updates and eight predetermined native layouts each;192 GRT requests. |
| Expert-rule result reduction | `analysis/qstudy_rule_reward_actor_results.slurm` | Validate every planned request and preserve incomplete/failure denominators; seven native GRT metrics and paired comparisons. |
| New final-route TRAIN labels | `fql/qstudy_train_final_route.slurm` and `fql/qstudy_train_drt_recovery.slurm` | Fixed32 families/two variants each; original GRT→cap20 DRT, explicit infrastructure recovery, no replacement of difficult samples. |
| Power/label admission | `analysis/qstudy_final_power_audit.slurm` | Complete TRAIN64 outcomes, fixed activity assumptions, per-fold and full-refit variation checks; reject unavailable/constant required targets. |
| Six final-feedback critic objectives | `fql/qstudy_final_feedback_screen.slurm` → `fql.qstudy_final_feedback_screen prepare/run` | Externally pinned complete collector and power-audit reports; four family folds plus refit, three seeds. |
| Final-feedback actor/export | `fql/qstudy_final_feedback_actor.slurm` → `fql.qstudy_final_feedback_actor prepare/run` | Six objectives, three actor seeds, fixed1000 updates;144 GRT requests with the same physical pipeline. |
| Final-feedback detailed routes/report | `fql.qstudy_final_feedback_route plan/resolve`, `fql/graph_o2o_route_controller.slurm`, `analysis/qstudy_final_feedback_actor_results.slurm` | Resolve exact GRT inputs for the fixed first four noises per arm/seed,72 DRT requests; all six final metrics plus separate GRT overflow diagnostics. |

Immutable launch receipts under
`/fact_home/bohanli/flowplace_fql_recovery/` record the actual submitted
dependencies, hashes and recovery attempts. Scientific arrays remain on
scavenger. Short development fixtures and bounded interface smoke checks use
the PDF's dev workflow; they are not substitute policy evaluations.

## Measured scalar continuation adapters

Scalar expert/final objectives use the declared bounded cost B and
`Q=-mean(predicted B)`, without a second mean(B) rescaling. These adapters
preserve the actual source checkpoint, optimizer/RNG state and fixed root
teacher/normalizers. They do not select a reward, collection budget, update
budget or winning checkpoint automatically.

| Capability | Entry | Inputs and current scope |
| --- | --- | --- |
| Scalar offline actor export | `fql/qstudy_scalar_online_export.slurm` | `prepare --request ... --request-sha256 ... --output ...`, then `export --experiment ... --config-sha256 ... --seed ...`; explicit three-seed collect/evaluation streams, root actor identity and complete raw/legalized/DEF lineage. |
| First measured FQL continuation | `fql/qstudy_scalar_online_train.slurm` | `--request ... --request-sha256 ...`; own C0 feedback only, externally bound final DRT/QA when required by the objective; actual actor/critic Adam and named RNG continue. |
| Genuine BC export/first continuation | `fql/qstudy_scalar_online_bc_export.slurm`, `fql/qstudy_scalar_online_bc_train.slurm` | Original BC actor/teacher prefix; scalar critic provenance attached separately and frozen. BC training uses Q weight0 and no critic/teacher updates. |
| Actual FQL T1 export | `fql/qstudy_scalar_online_child_export.slurm` | Same explicit prepare/export interface, real completed first-training request/checkpoint; logical round1, no relabeling as offline round0. |
| Second measured FQL continuation | `fql/qstudy_scalar_online_second_train.slurm` | Explicit cycle2 request; independently admitted C0(root actor)+C1(T1 actor), complete cumulative replay, original TRAIN/teacher/normalizers, actual T1 Adam/RNG and audited cumulative counts. Supports updated-Q or frozen-Q settings. |
| Actual BC T1 export | `fql/qstudy_scalar_online_bc_child_export.slurm` | Genuine completed BC continuation, with its original BC actor prefix and separately attached frozen scalar critic; logical round1 collect/evaluation, unchanged physical process. |
| Second measured BC continuation | `fql/qstudy_scalar_online_bc_second_train.slurm` | Same complete C0+C1 admission and cumulative actor-state accounting; Q weight0, no critic/teacher updates, and bitwise checks of their weights, optimizers and unused RNG streams. |
| Actual FQL T2 evaluation export | `fql/qstudy_scalar_online_second_export.slurm` | Explicit three completed second-training requests/checkpoints and evaluation streams only. Loads actual T2 actor and T2 critic, retains the complete source history, and excludes every exported row from training. |
| Actual BC T2 evaluation export | `fql/qstudy_scalar_online_bc_second_export.slurm` | Same evaluation-only contract for completed genuine BC T2 parents, with original BC prefix and separate frozen scalar refit explicit throughout. |

The completed three-seed interface checks use one actual update per segment
and one native collect per seed/segment; they validate implementation, not
scientific online quality or the earlier600-update schedule. Their results
must not be combined with the historical matrix as an additional independent
policy-quality run. The BC T1 exporter, BC second trainer and FQL T2 exporter
passed12,14 and12 CPU fixtures respectively. The BC T2 exporter passed another
12. Actual graph integration is a separate check: the BC T1 and FQL T2
three-seed exports completed with six and three legal DEFs respectively,
without training or backend calls. The BC second continuation subsequently
completed in all three seeds (1800263), each admitting its complete C0+C1 and
retaining cumulative actor Adam1002/Q0/teacher0. Actual BC T2 prepare1800268
and three-seed export1800269 also completed: three legal evaluation-only DEFs,
with the same actual77108000 noise arrays as FQL T2, no replay eligibility and
no new backend calls. Its independent saved-metadata audit is recovery
`qstudy_scalar_online_bc_t2_eval_smoke_v1/actual_final_metadata_audit_v1.json`,
SHA256 `63f24c09f056cf8d1c002886c1dcc5835b5d39f6bf7132d9dd0af86e28abae91`.
A scientific online trial needs its own explicit matched collection/update
plan and all actual feedback before training can start; these integrations
do not supply600-update policy-quality results.

## Recompute a report without rerunning policies

This reads the completed main matrix and writes a new project-scratch report.
It does not overwrite the authoritative snapshot. Use the exact timestamp/job
path printed by this job.

```bash
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
BASE=/fact_data/bohanli/flowplace_fql/qstudy_v1

sbatch fql/qstudy_readonly_report.slurm -m fql.qstudy_actor_report \
  --experiments "$BASE/actors_v1/bc_frozen" "$BASE/actors_v1/legacy_frozen" \
  --rounds 2 \
  --output /scratch/bohanli/flowplace_graph/qstudy_report_reproduction_v1
```

Add `"$BASE/actors_v1/legacy_updated"` to the same `--experiments` list to
include the adaptive branch. The resulting360 outcome representations
include inherited results. Review native failure counts and control-eligibility
flags: successful aggregation does not establish improvement, final DRC0 or
equal training compute.

## Interpretation

This adapts FQL's behavior flow, same-noise one-step distillation and Q
optimization to a terminal whole-layout action, using FlowPlace's graph
backbone/public initialization. CQL/Cal-QL/IQL/TD3+BC and PPO placers are
mechanism references, not newly reproduced competitive baselines. Tested
source, ranking, conservative and multiobjective changes have not established
stable PPA gains. See the [contribution boundary](qstudy_literature.md#7-新贡献需要满足什么).
