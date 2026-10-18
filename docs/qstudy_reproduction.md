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
