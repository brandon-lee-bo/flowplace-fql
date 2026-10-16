# Whole-goal completion audit

Final review:2026-10-17, Hong Kong. The requested overnight investigation is
complete; positive FQL PPA improvement and exact numerical Table2 reproduction
were hypotheses to test, not presumed outcomes. All calculations used Slurm.

| Original requirement | Verified end state | Evidence |
|---|---|---|
| Understand and pin FlowPlace/FQL | Official source revisions retained; terminal-bandit adaptation and deviations explained | manifest.json; docs/fql_integration_design.md; docs/research_interpretation.md |
| Run a credible generation/backend baseline | Strict checkpoint load; deterministic repeated/compiled outputs; DEF geometry and full backend verified | baseline/artifacts/; backend/results/; docs/run_ledger.md |
| Attempt FlowPlace Table2 | All6releasedDEF outcomes retained:5completed fullflows and1raw-overlap failure; same-input bp preemption recovery bound to original checkpoints | analysis/artifacts/table2_released_v1/report.json and comparison.md |
| Implement a minimal FQL experiment | Whole-layout student, frozen teacher, terminal critic, matched BC/direct-analytic/coordinate controls;3trainingseeds | fql/; runs/bp_fe_macro_deterministic_v1/ |
| Validate actual quality and legality | Independent macro checks;36proxy-transfer outcomes35success+1CTSfailure;11sampler outcomes10success+1CTSfailure; all planned denominators preserved | analysis/artifacts/proxy_transfer_grt_v1/; analysis/artifacts/sampler_transfer_grt_recovered_v1/ |
| Adapt based on evidence | 96realGRTlabels verified; critic gate0/3failed, so conditional actor/48+24expansion intentionally not run | runs/bp_fe_grt_gate_v1/report.json; docs/evaluation_protocol.md |
| Explain mechanisms and fairness | Hard-projection sensitivity, teacher BC learnability, Euler variant, fixed-layout controls, generation+legalization latency; scope limits explicit | docs/results_report.md |
| Preserve reproducibility and failure evidence | Pinned tools, configs, task/selection/data hashes, original failures/retry maps, source snapshots for later runs; original macro startup-code-hash gap disclosed | backend/results/; datasets/; docs/run_ledger.md |
| Quantify cost and obey scheduling | Final allocation snapshot includes failed/preempted attempts; no unrelated jobs touched; no experiment jobs remain queued/running | analysis/artifacts/cost_final_v1/report.json; docs/cluster_execution.md |
| Deliver code, results and judgment | Code committed; measured tables/plots and concise scientific recommendation documented | README.md; docs/results_report.md; docs/research_interpretation.md |

Independent final reviews checked both actual GRT reports and rendered figures,
the six-design recovery summary, and final Slurm accounting. Software regression
1785504 verified that changing a copied metric after its audit is rejected;
the original files stayed unchanged. Other recorded failures were not relabeled
as successes or removed from comparison denominators.

Remaining ideas in research_interpretation.md are proposals for a new experiment,
not unexecuted requirements of this completed pilot. In particular, do not launch
the conditional actor merely because its implementation exists: its predeclared
critic gate failed. There are no unresolved infrastructure/parser errors in the
final scientific result sets.
