# Whole-goal completion audit

## Current graph Flow-Q offline-to-online goal — active, 2026-10-18 HKT

The current objective is the full user instruction in
`/fact_home/bohanli/.codex1/attachments/5c8e6639-aa3b-43c9-8bbf-97b9419b1108/pasted-text-1.txt`.
The historical completion statement below belongs to an earlier, smaller pilot;
it does not establish completion of this graph-policy offline-to-online goal.

| Current requirement | Actual evidence and scope | State |
| --- | --- | --- |
| Check both official methods and reuse the FlowPlace graph backbone | Pinned official revisions and explicit adaptations in `fql_integration_design.md`; `graph_o2o_models.py` retains GATv2/attention for the behavior flow and independent one-step actor | Implemented; fixed-circuit terminal adaptation, not an unchanged multi-step FQL benchmark |
| Define real actions, reward, data and termination | Raw proposal predicts the measured result after legalization and native backend import; terminal mask is zero; 64 initial TRAIN and 16 validation labels; family splits and TRAIN-only normalization checked. Second-design80/80 outcomes independently audited by1793202; bp_be reward scale3537708.3125 uses successful TRAIN only | Verified for both circuits' initial datasets; second-design online feedback remains in progress |
| Actually train offline and then collect feedback and update both policy and Q | All three bp_fe seeds completed offline and two measured-feedback online updates; changed flow/actor/critic weights recorded. bp_be offline1793203 COMPLETED23m41; first28/28 fixed GRT requests successful, first actual online1794723 live with own8-row replay per arm | Actual bp_fe training/final PPA and bp_be offline complete; bp_be actual online updates remain in progress |
| Pursue downstream quality rather than only lower loss | Original final PPA and fixed-replay, best-measured-layout and larger-feedback controls retained; current seed0 online FQL does not beat matched BC. All320 added offline scores independently audited;384-label offline1792852 COMPLETED19m04, first28/28 GRT outcomes successful, own8-row replay per arm verified; first online1794707 COMPLETED8m52, next24-request feedback array1794728 live | Actual original negative result; expanded second online update and downstream outcomes remain incomplete |
| Compare methods fairly and check seed/circuit variation | Complete three-seed native aggregate1792811:63/63 logical requests successful, all DRC0. Online FQL mean final rWL0.308% lower than BC, but worse in two of three seeds and greater between-seed variation. Frozen FlowPlace is not independently retrained | Multiple-seed main result complete without stable RL superiority; actual second-design RL/PPA remains incomplete |
| Use one physical flow and report legality, DRC and final PPA honestly | Native OpenROAD/Nangate45, eight threads and cap20; all63 main requests independently audited against actual source/final DEF and native DRC; expanded/bp_be outcomes pending | Complete for all main three-seed cohorts; not complete for all current experiments |
| Investigate failures and useful placement-specific improvements | Q transfer and local-gradient diagnostics, added-feedback experiment, and actual GRT-to-DRT fidelity checks distinguish critic error from reward-stage transfer | Real evidence exists; larger-data and second-design checks remain open; no positive RL claim |
| Preserve provenance, report results and obey FactLab rules | Config/source/data/weight/job bindings, immutable failed attempts and route evidence retained; heavy work runs through Slurm; `graph_fql_offline_to_online.md` contains the current Notion-ready record | Ongoing; live jobs and scratch assets still require final reconciliation and evidence preservation |

At06:05 HKT, all three bp_fe policy controllers, all76 GRT requests per
seed and all63 predeclared final-route requests have completed. Native
aggregate1792811 confirms online FQL mean final rWL2.031444667m versus
BC2.037714333m, only0.30768% lower, with greater between-seed variation and
two of three seeds worse. The plotted complete result was visually inspected.
There is no stable RL-superiority or comprehensive-PPA claim. Seed1/2 native
evidence is durably preserved by1794588/1794612 with full per-file decoding
checks; these are not full OpenDB/SPEF resume archives. Scoped Slurm accounting
1794610 includes all recorded requeue/preemption allocations of main training
and GRT, not a whole-project cost or independent per-arm GPU measurement.
Expanded-data training and bp_be offline-to-online training/final PPA still
prevent a whole-goal completion claim.

At06:13 HKT, bp_be offline1793203 has actually completed both arms and
exported28 requests. Its controller1793204 is live and the first real GRT
1794621 is running; actual second-design online parameter updates have not
yet happened. Both completed offline inference states were durably preserved
by1794623, with954 tensors and all non-optimizer fields exactly checked.
Expanded-label BC offline substage has completed, but FQL/export and all
expanded online/final-PPA results are still pending. Its validation Q MAE is
6.08% lower on the same16 rows, with no improved Spearman ranking; this does
not substitute for the predeclared fixed-cohort or downstream comparisons.

At06:20 HKT, expanded offline1792852 has completed both arms and all28
exports. Both corresponding inference states were exactly preserved by
1794636:101336045bytes and954 tensors, with optimizer states remaining in
the original scratch files. The first actual GRT candidate succeeded in both
new campaigns; existing arrays1794630/1794639 now collect the remaining27
requests each. Only their scavenger array throttle changed to8, retaining
eight threads,24GiB, fixed action lists and all learning/physical settings.
No new-campaign online parameter update or final policy comparison is yet
claimed. Fixed-cohort diagnostic source hashes still match their frozen plan.

At06:37 HKT, expanded first28/28 real GRT requests have succeeded and the
fixed replay contains exactly eight own collection outcomes per arm. Actual
online1794707 is running on an allocated L20: BC completed600 updates with
changed flow/actor/critic weights; FQL remains live. Snapshot1794720 completed
and independently rebound/copy-verified91 small request/action/outcome/replay
files,411529bytes. Its manifest SHA is
`fd32191c9b0f73d3119efcf93e079ece04ad064fd8747b858fd83bf0de1db5cb`.
This proves measured feedback was supplied to actual online training, not
completion of the whole round or new-campaign final PPA. The bp_be first
feedback array remains live, without restarting any observed job.

At06:39 HKT, bp_be initial feedback array1794630 has authoritatively
completed every child and controller1793204 submitted actual online1794723,
now RUNNING on L20. Independent replay snapshot1794724 completed: all28
actual outcomes successful, eight own collection labels per arm and no
evaluation/cross-arm feedback;91 copied small files420757bytes. Both new
campaigns now actually train on measured feedback; their final updates,
fixed-cohort diagnostic and full downstream outcomes remain incomplete.

At06:41 HKT, expanded first online1794707 has completed both600-step
updates and all24 fixed exports. Both392-row critics and both policy/flow
networks changed; existing controller1792853 started array1794728.
Its scheduler-only throttle2→8 retains the actual default8CPU/12GiB per
task, fixed actions and physical flow. Preservation1794732 completed4s,
with101336173bytes and954 exact tensors/all other non-optimizer fields.
No new-campaign complete downstream comparison, final model improvement or
whole-goal completion follows from these training and preservation checks.

Current authoritative handles include the independent-seed controllers
1792766/1792794, 320-label expansion1792874 and second-design labels1793035.
At05:49 HKT,1792766 and its route controller1792778 have completed all76 GRT
and21 detailed-route requests for seed1, all21 reporting DRC0. The complete
three-seed native aggregate is still waiting for seed2. Second-design labels
1793035 and audit1793202 have completed; actual offline training1793203 and
the23-request route controller1793205 are now live. The frozen second-design
dataset binding is
`da826d9d5d614bc5b7f54248506b03b6696e9f47c540a1b7745a8b724a55226e`.
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
expanded-data policy have both started after their complete label audits.
At05:58 HKT all320 added labels are successful and independently audited;
the expanded training binding is
`05b87166dada1545529d54d23556142706ca878d60a75e3e630cbb88cd889224`.

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
