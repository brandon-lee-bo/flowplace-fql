# Overnight pipeline handoff

Snapshot:2026-10-17 about02:45HKT. All computation uses FactLab Slurm.
Do not change unrelated jobs. See results_report.md for scientific interpretation.

## Completed

- GRT96 labels1784047 → audit1784160 → relabel1784119 → gate1784156 completed.
  All96 labels verified; first two parser-only failures recovered without reroute.
- **Real-GRT critic gate failed0/3. Do not launch the conditional actor or48+24
  candidate backend expansion.** All validation MAEs exceed the TRAIN-mean
  baseline0.03913. No heldout scoring or tuning. This is the predeclared pilot
  stop decision, not proof that all FQL approaches are ineffective.
- Proxy36 outcomes:35success+1verified CTS failure. Classifier1784450,
  aggregate1784481 and plots1784485 complete and checked. Strict all-success
  audit1784161 correctly failed. CSV retains36 cases; failed seed/config means
  null. Actual GRT does not support the macro-reward FQL improvement.
- Released bp_fe cap20 DRT1784154 completed, audit1785142 passed,DRC1904→192.
  Generated cap20 example reached0DRC atiteration9. Neither changes cap4 rows.
- Generated swerv1784543 completed fullflow+audit:rWL4.163184Mum,DRC7282.
  Independent fixed seed300; released swerv remains a failed Table2 row due to
  macro overlaps already present in its raw DEF.

## Pending physical results

- Sampler1784158 continues8new tasks/11full records with3reused references.
  Task0/raw1785399 was PREEMPTED after1m02 inCTS, noGRT. Original attempt and
  preemption evidence preserved. SameDEF/flow fresh retry1785454 completed159s
  innewrun bp_fe_sampler_transfer_v1_n00_i0240_retry1,eda8CPU12GiB/no-requeue.
  Audit1785462 passed real raw-report checks and6negative binding tests.
- Original downstream classifier1784451/aggregate1784482/plots1784486 were
  canceled while pending. Classifier replacement1785468 is queued with explicit
  infrastructure-retry map at backend/assets/sampler_infrastructure_retry_map_v1.json;
  original tasks/selection/11-case denominator stay unchanged. fql_design is
  finishing aggregate support; root must schedule aggregate/plot replacements
  after1785468 and inspect actual outputs. No new physical samples are added.
- Original bp1783836_2 was PREEMPTED after59m02. Its unstarted automatic requeue
  was held/canceled; original work and manifest unchanged.
- bp recovery1785220 started02:31:09 oneda8CPU/14GiB/2h30/no-requeue, adjusted
  while pending from24GiB/6h based on original observed memory/time. Startup
  actualimage/archive/patch/current-flow and5GRT checkpoint-copy checks passed.
  It resumes originalcap4; runtime is resumedDRT/fill/finish,not freshfullflow.
- Six-design summary1785253 waits for recovery and originalattempt audit1784328
  (failed for interruptedbp and invalidswerv). It uses table2_retry_map_v1.json
  and validates recovery evidence; swerv failure stays. Do not rerun merely
  because completeness exit code is nonzero. Old summary1784311 canceled pending.

## Remaining completion work

Finish sampler recovery/aggregation and recoveredbp/Table2 summary; resolve real
software/infrastructure issues without weakening physical constraints. Complete
results/research interpretation, final Slurm accounting including all attempts,
final code/doc commit and a whole-goal completion audit. Do not mark complete
while these required physical results remain pending.

The conditional GRT actor workflow/frozen failure policy remain implemented and
software-tested in fql/README.md and docs/evaluation_protocol.md, but unexecuted.
