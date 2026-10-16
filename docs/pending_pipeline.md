# Overnight pipeline handoff

Snapshot: 2026-10-17, before the 96-label collection finishes. This is an
operational checklist, not a result report. See `results_report.md` for evidence.
All computation uses FactLab Slurm. Do not change unrelated jobs.

## Already queued

- GRT labels: `1784047` (96 tasks, concurrency 2) → audit `1784160` → relabel
  `1784119` → critic gate `1784156`. Relabel waits for both the array and audit.
  First two Slurm failures were postprocessing failures, recovered without
  rerouting; original failures and parser-recovery evidence remain preserved.
- Exploratory proxy transfer: `1784109` → strict audit `1784161` and terminal
  classification `1784450` → aggregate `1784481` → plots `1784485`.
  Known physical CTS failure is retained; strict audit is expected to flag it.
- Sampler transfer: `1784158` (after proxy array) → classification `1784451`
  → aggregate `1784482` → plots `1784486`.
- Released fullflows: `1783836` → final audit `1784328` → six-design summary
  `1784311`. Released swerv fails raw macro overlap, so the audit/summary must
  preserve an incomplete comparison rather than claim six successful circuits.
- Separate diagnostics: released bp_fe cap20 DRT `1784154`; deterministic
  generated swerv fullflow `1784543`. Neither replaces the original table row.

## Conditional real-GRT actor experiment

Inspect `runs/bp_fe_grt_gate_v1/report.json` after completion. The advisory gate
requires at least 2/3 seeds to beat the TRAIN-mean constant in validation MAE and
have positive validation Spearman correlation. Only eight validation families
are available; this does not guarantee actor-selected actions will be accurate.
If weak, diagnose/report the negative pilot instead of blindly expanding sweeps.

If supported, train the predeclared three-seed BC/FQL experiment using
`datasets/bp_fe_grt_v1/dataset.npz` and the gate's pretrained critics. Use
256×3 actors, 64×64 critics, warmup1000/actor2000, batch128, lr0.0003,
alphas1/10/100, `--seal-heldout`; `--critic-steps 1` is a parser placeholder,
not a request to retrain loaded critics. Allocate one GPU via Slurm.

Both candidate export phases must explicitly pass:

```text
--physical-failure-policy fql/grt_actor_physical_policy_v1.json
--heldout-indices 244,245,246,247
```

Validation uses 192–195: 48 new backend tasks plus five exact reused references.
Execute only `pending_tasks.jsonl`. Audit and classify with **full `tasks.jsonl`**;
failure evidence hashes bind that complete manifest. Aggregate with
`--allow-preregistered-physical-failures`. Keep all fixed cases and use the
configuration-level summary, never a successful-seed-only mean.

Run `fql/seal_grt_selection.py --validation-report REPORT/report.json --output
runs/bp_fe_grt_actor_v1/grt_selection_v1.json` through `run_cpu.slurm`.
`no_eligible_configuration` means stop, with no heldout plan. `selected` means
export the locked alpha and BC on 244–247 (24 new tasks + five reused references),
passing `--selection` and the same policy. Never reselect after heldout outcomes.

## Finish

Inspect actual aggregates/plots and resolve genuine software or infrastructure
failures without weakening physical constraints. Update results, complete a
Slurm resource-accounting snapshot including failed attempts, save final code,
and assess whether learned Q/policy adds value over direct objectives and the
best measured TRAIN layout. Mark the goal complete only after required results
and final interpretation are delivered.
