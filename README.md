# FlowPlace + FQL investigation

The current Q/reward study is **in progress**, with **no stable PPA advantage
established**. The [compact evidence/decision table](docs/qstudy_decisions.md)
separates completed diagnostics from the pending main/source/bounded physical
policy matrices and fresh confirmation. The study tests critic inputs, losses,
physical objectives and behavior-flow data sources. Read the [current plan](docs/qstudy_plan.md),
[measured progress](docs/qstudy_progress.md),
[behavior-flow label protocol](docs/qstudy_behavior_source.md) and
[source-to-actor protocol](docs/qstudy_source_actor_protocol.md).
The completed [source-control result](docs/qstudy_source_results.md) is negative
for current-flow WL prediction; its fixed-budget physical policy test is
separate. The [bounded multiobjective comparison](docs/qstudy_bounded_objective_protocol.md)
has completed its predictive screen; its fixed-budget physical policy test is
queued. The [fresh confirmation protocol](docs/qstudy_fresh_confirmation_protocol.md)
reserves new generation noises after development selection is locked.
These new experiments have not established a stable PPA improvement. The old
Table 2 reproduction remains paused. Results below describe completed earlier
phases, rather than completion of the current study.

This study implements **graph-based FlowPlace + Flow Q-learning (FQL), with
actual offline-to-online training and native downstream evaluation**. The
earlier evaluation phase is complete, including final native evidence
preservation. All three bp_fe
training seeds completed offline training, two
measured-feedback updates and native final routing:63/63 predeclared requests
passed, all with DRC0. Online FQL mean final wirelength is0.308% lower than
matched BC, but is worse in two of three seeds and has greater between-seed
variation. The expanded384-label and second-circuit bp_be models have also
completed offline and both measured-feedback online updates. All eight new
online arm/stage pairs have independent actual parent/child tensor and replay
checks. Expanded final GRT wirelength is1.255% lower than matched BC, but
timing is worse; identical-proposal Q diagnostics do not show better critic
generalization. Expanded21/21 predeclared final routes and both best400 controls
have completed, all DRC0, with the full native audit and durable evidence
archive verified. Its final FQL wirelength is1.388% lower than matched BC,
with slightly better WNS but worse TNS; this is one training seed.
bp_be's76/76 GRT and23/23 final-route requests also succeeded, all final
DRC0. Its online FQL final wirelength is0.754% below matched BC, with16.121%
less TNS magnitude, but0.868% worse wirelength than its own offline stage.
Independent audit1793511 reloaded all six parent/child checkpoint pairs and
reparsed all23 final DEFs; native preservation1794901 completed full decoding
of2249 files. Consolidation1794697 covers all three studies; the bounded
geometry review records zero macro overlap and out-of-core counts for all109
logical final requests, including the expanded best400 controls. The
current Chinese [experiment record](docs/graph_fql_offline_to_online.md),
[completion audit](docs/completion_audit.md) and [run ledger](docs/run_ledger.md)
report the completed results, provenance and limitations. No stable RL PPA
advantage is claimed.

An earlier bounded MLP follow-up is complete: fresh learned behavior-flow
distillation, a physical-feature critic and12/12 paired GRT outcomes. Its about
0.9% mean wirelength gain over matched BC is an exploratory single-circuit
historical result, not the current graph-policy result. See the last section
of `docs/results_report.md`,
`fql/fresh_flow_plan_v2.json`, `fql/physical_actor_plan_v1.json`, and
`logs/fql_fresh_completed_v1.json`.
The existing Table 2 work covers FlowPlace and DREAMPlace on six OpenROAD
designs with a common physical-design backend; its remaining jobs and automatic
continuation are paused. Do not resume them without user direction. See the
[current execution plan](docs/pending_pipeline.md),
[Table 2 protocol](docs/table2_protocol.md), and
[DREAMPlace public-implementation review](docs/table2_dreamplace_protocol.md).
This work has not yet established paper-matching numbers.

The earlier FQL research pilot ran on 2026-10-16/17 (Hong Kong).
Macro-proxy and real-GRT pilot experiments are complete. The macro-reward FQL
gain did not transfer to the tested backend cases; the real-GRT critic failed
its predefined validation gate, so conditional actor expansion was not launched.
Those pilots established neither paper-matching Table 2 numbers nor a routed
PPA advantage from FQL.

Read the [results report](docs/results_report.md) for measured outcomes,
the [run ledger](docs/run_ledger.md) for Slurm jobs/failures, and the
[evaluation protocol](docs/evaluation_protocol.md) for what each result means.
The concise Chinese [research interpretation](docs/research_interpretation.md)
explains what was adapted from FQL and which hypotheses remain unproven.
The [six-circuit comparison](analysis/artifacts/table2_released_v1/comparison.md)
retains five completed fullflows and the released swerv DEF's overlap failure.

## GitHub checkout

Repository: https://github.com/brandon-lee-bo/flowplace-fql

```sh
git clone https://github.com/brandon-lee-bo/flowplace-fql.git
cd flowplace-fql
```

The repository includes experiment code, configuration, Slurm scripts, research
records and the existing local commit history. Python environments, model
weights, datasets, downloaded upstream sources and large native EDA artifacts
remain on the server and are excluded from Git. This is a source-code snapshot,
not a self-contained copy of the experiment data. The current study is documented
in [the experiment record](docs/graph_fql_offline_to_online.md).

For a fresh checkout, restore the official sources at the revisions recorded in
`manifest.json`:

```sh
git clone https://github.com/lamda-bbo/flowplace.git upstream/flowplace
git -C upstream/flowplace checkout --detach da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547
git -C upstream/flowplace rev-parse HEAD > upstream/flowplace/SOURCE_COMMIT
git clone https://github.com/seohongpark/fql.git upstream/fql
git -C upstream/fql checkout --detach e8cd16eb490332924dfa2492097219f181765933
git -C upstream/fql rev-parse HEAD > upstream/fql/SOURCE_COMMIT
```

Release checkpoints and physical-design inputs must be restored separately;
see [the baseline adapter](baseline/README.md) and
[the backend instructions](backend/README.md). Existing job scripts contain
FactLab paths, resource requests and references to server-local artifacts. Review
these before submitting from another checkout; use Slurm for computation.
Historical Table 2 controllers remain paused.

## Goal

Integrate the actual FQL flow teacher, one-step actor and critic with FlowPlace's
graph backbone; train offline, collect actual policy feedback, update replay
and continue learning. Compare measured layout quality against original
FlowPlace, matched BC and best-measured-layout controls, including independent
seeds, more real training labels and a second circuit. Use one OpenROAD/Nangate45
flow, preserve failures and report final PPA and DRC. All substantive computation
runs via FactLab Slurm with requested resource limits.

The earlier FlowPlace/DREAMPlace Table2 reproduction remains recorded and
paused following the user's2026-10-17 21:39 HKT instruction. Its old pending
tasks must not be resumed as part of this goal.

## Layout

- `manifest.json`: pinned upstream revisions and scope.
- `upstream/`: untouched official source archives (`SOURCE_COMMIT`).
- `baseline/`: strictly loaded pretrained FlowPlace, data conversion, samples.
- `backend/`: released OpenROAD flow, tool container, DEF export and evaluation.
- `fql/`: terminal contextual-bandit adaptation, one-step distillation baseline.
- `analysis/`: independent legality and macro-only HPWL validation.
- `collect_dataset.py`: paired teacher/noise and perturbed-action collection,
  grouped train/validation/test splits, explicit reward provenance.
- `docs/`: integration rationale, cluster rules and experiment decisions.
- `logs/`: installation failures and runtime logs are retained.

## Evidence boundaries

FlowPlace Table 2 reports full routed physical metrics. A macro-only HPWL
improvement is not a timing, power, or area improvement. Whole-layout
single-action FQL is a contextual bandit and does not test long-horizon
Bellman credit assignment. Author-provided DEF replay validates a backend,
while fresh generation additionally validates the model pipeline. Keep these
levels separate in any report.

The original source and released configuration can differ from paper text;
preserve and document such differences rather than silently alter a baseline.
Run-to-run sample variability, legality, and cost must be reported alongside
any selected best sample. Test seeds cannot select hyperparameters.
