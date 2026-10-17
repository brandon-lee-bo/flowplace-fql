# Whole-goal completion audit

## Current graph Flow-Q offline-to-online goal — active, 2026-10-18 HKT

The current objective is the full user instruction in
`/fact_home/bohanli/.codex1/attachments/5c8e6639-aa3b-43c9-8bbf-97b9419b1108/pasted-text-1.txt`.
The historical completion statement below belongs to an earlier, smaller pilot;
it does not establish completion of this graph-policy offline-to-online goal.

| Current requirement | Actual evidence and scope | State |
| --- | --- | --- |
| Check both official methods and reuse the FlowPlace graph backbone | Pinned official revisions and explicit adaptations in `fql_integration_design.md`; `graph_o2o_models.py` retains GATv2/attention for the behavior flow and independent one-step actor | Implemented; fixed-circuit terminal adaptation, not an unchanged multi-step FQL benchmark |
| Define real actions, reward, data and termination | Raw proposal predicts the measured result after legalization and native backend import; terminal mask is zero; 64 initial TRAIN and 16 validation labels; family splits and TRAIN-only normalization checked | Verified for the original circuit; second-design scale still awaits its complete 80-label collection |
| Actually train offline and then collect feedback and update both policy and Q | All three bp_fe seeds completed offline and two measured-feedback online updates; seed2 round2 training1794430 COMPLETED8m45; changed flow/actor/critic weights recorded | Actual bp_fe training complete; full multi-seed final PPA and bp_be training remain in progress |
| Pursue downstream quality rather than only lower loss | Original final PPA and fixed-replay, best-measured-layout and larger-feedback controls retained; current seed0 online FQL does not beat matched BC | Actual negative result; expanded 384-label policy has not trained yet |
| Compare methods fairly and check seed/circuit variation | All three predeclared offline seeds0/1/2 have native-audited final PPA:12 BC and12 FQL layouts, all DRC0. Offline mean FQL rWL is0.405% lower, with one seed worse and worse mean timing. Frozen FlowPlace layouts are not counted as independent trainings; bp_be import probes completed | Multiple-seed online final results and actual second-design RL/PPA are incomplete |
| Use one physical flow and report legality, DRC and final PPA honestly | Native OpenROAD/Nangate45, eight threads and cap20; original 21 requested routes completed with DRC0; source/final DEF and native DRC independently audited | Complete for the original cohort; not complete for all current experiments |
| Investigate failures and useful placement-specific improvements | Q transfer and local-gradient diagnostics, added-feedback experiment, and actual GRT-to-DRT fidelity checks distinguish critic error from reward-stage transfer | Real evidence exists; larger-data and second-design checks remain open; no positive RL claim |
| Preserve provenance, report results and obey FactLab rules | Config/source/data/weight/job bindings, immutable failed attempts and route evidence retained; heavy work runs through Slurm; `graph_fql_offline_to_online.md` contains the current Notion-ready record | Ongoing; live jobs and scratch assets still require final reconciliation and evidence preservation |

Current authoritative handles include the independent-seed controllers
1792766/1792794, 320-label expansion1792874 and second-design labels1793035.
Feedback-channel training1794210, eight new GRT results1794211, routing1794212
and final report1794213 have completed. All20 diagnostic layouts have DRC0;
adding behavior data alone worsens final rWL by1.108%, adding Q data alone
by0.070%, and adding both by1.549%, relative to the same-parent fixed-replay
control. These are four already-observed evaluation noises from one parent,
not independent multi-seed algorithm validation.
Offline three-seed native-evidence report1794338 completed without reading
online metrics or changing selection. Coverage routing1792875 and report1792565 have
completed all20 requested layouts with DRC0; native evidence audit1794208
also passed. The added-feedback FQL still underperforms BC on final rWL.
Other handles' later terminal status must be read from
Slurm, not inferred from this dated document. Second-design training and the
expanded-data policy are queued behind real label collection.

The goal is **not complete**. Completion requires actual downstream results
for the pending comparisons and circuits, a consolidated scientific judgment,
and a final audit against the full current objective. More code, queued jobs,
passing helper checks or one favorable layout do not satisfy those items.

## Historical pilot audit — 2026-10-17, not the current completion claim

The following original record is retained as historical evidence. Its statements
about no remaining jobs and completed scope are specific to that earlier pilot.

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
