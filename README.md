# FlowPlace + FQL investigation

The active goal is **graph-based FlowPlace + Flow Q-learning (FQL), with actual
offline-to-online training and native downstream evaluation**. It remains in
progress. On bp_fe, seed0 completed offline training, two measured-feedback
updates and final routing. All three predeclared training seeds have audited
offline final PPA: FQL mean wirelength is0.405% lower than matched BC, with one
seed worse and worse mean timing. Full independent-seed online results, the
expanded-data policy and bp_be training/evaluation remain incomplete. The
current Chinese [experiment record](docs/graph_fql_offline_to_online.md),
[completion audit](docs/completion_audit.md) and [run ledger](docs/run_ledger.md)
separate measured results from pending work. No stable RL PPA advantage is claimed.

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
Exact Table 2 numbers and a routed PPA improvement from FQL have not been established.

Read the [results report](docs/results_report.md) for measured outcomes,
the [run ledger](docs/run_ledger.md) for Slurm jobs/failures, and the
[evaluation protocol](docs/evaluation_protocol.md) for what each result means.
The concise Chinese [research interpretation](docs/research_interpretation.md)
explains what was adapted from FQL and which hypotheses remain unproven.
The [six-circuit comparison](analysis/artifacts/table2_released_v1/comparison.md)
retains five completed fullflows and the released swerv DEF's overlap failure.

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
