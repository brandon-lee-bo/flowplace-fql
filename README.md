# FlowPlace + FQL investigation

This is a live research experiment, started 2026-10-16 (Hong Kong).
The macro-proxy experiments and independent audits are complete. Exact Table 2
numbers and a routed PPA improvement from FQL have not been established.

Read the evolving [results report](docs/results_report.md) for measured outcomes,
the [run ledger](docs/run_ledger.md) for Slurm jobs/failures, and the
[evaluation protocol](docs/evaluation_protocol.md) for what each result means.

## Goal

Reproduce the released FlowPlace OpenROAD evaluation, generate our own
placements with the pinned pretrained model, then test whether FQL-style
value optimization improves generated placement quality. All substantive
computation runs via FactLab Slurm with requested resource limits.

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
