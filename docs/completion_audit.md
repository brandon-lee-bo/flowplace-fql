# Whole-goal completion audit

## Current graph Flow-Q offline-to-online goal — active, 2026-10-18 HKT

The current objective is the full user instruction in
`/fact_home/bohanli/.codex1/attachments/5c8e6639-aa3b-43c9-8bbf-97b9419b1108/pasted-text-1.txt`.
The historical completion statement below belongs to an earlier, smaller pilot;
it does not establish completion of this graph-policy offline-to-online goal.

| Current requirement | Actual evidence and scope | State |
| --- | --- | --- |
| Check both official methods and reuse the FlowPlace graph backbone | Pinned official revisions and explicit adaptations in `fql_integration_design.md`; `graph_o2o_models.py` retains GATv2/attention for the behavior flow and independent one-step actor | Implemented; fixed-circuit terminal adaptation, not an unchanged multi-step FQL benchmark |
| Define real actions, reward, data and termination | Raw proposal predicts the measured result after legalization and native backend import; terminal mask is zero; 64 initial TRAIN and16 validation labels; family splits and TRAIN-only normalization checked. Second-design80/80 outcomes audited by1793202; reward scale3537708.3125 uses successful TRAIN only. Actual used feedback preserved and rebound by1794812/1794876,160 logical observations in total | Verified for both initial datasets and all five completed final online replay scopes; final extra collection is not counted as used training data |
| Actually train offline and then collect feedback and update both policy and Q | All three bp_fe seeds, expanded384 bp_fe and bp_be completed offline and two measured-feedback updates. Independent actual parent/child audits1794837/1794882 verify all eight new online arm/stages and exact replay use; each phase changes207/207/16 flow/actor/critic tensors | Actual offline-to-online training verified in all registered studies; final new downstream outcomes remain incomplete |
| Pursue downstream quality rather than only lower loss | Original final PPA and fixed-replay, best-measured-layout and larger-feedback controls retained. Expanded all76 GRT requests and audit1792866 completed: final FQL wirelength1.255% below BC but timing worse. Fixed-cohort1792947 and deduplicated624-prediction audit1794868 show40 unique proposals, no improved Q ranking | Actual measured main negative/stability result and expanded GRT signal; expanded final PPA and bp_be final scoring/PPA remain incomplete |
| Compare methods fairly and check seed/circuit variation | Complete three-seed native aggregate1792811:63/63 logical requests successful, all DRC0. Online FQL mean final rWL0.308% lower than BC, but worse in two of three seeds and greater between-seed variation. Frozen FlowPlace is not independently retrained | Multiple-seed main result complete without stable RL superiority; actual second-design RL/PPA remains incomplete |
| Use one physical flow and report legality, DRC and final PPA honestly | Native OpenROAD/Nangate45, eight threads and cap20; all63 main requests independently audited against actual source/final DEF and native DRC; expanded/bp_be outcomes pending | Complete for all main three-seed cohorts; not complete for all current experiments |
| Investigate failures and useful placement-specific improvements | Q transfer/local-gradient diagnostics, added-feedback and2×2 data-path controls, native GRT-to-DRT fidelity and larger-label same-proposal comparisons distinguish critic error from reward-stage transfer | Actual mechanism evidence; larger-data final PPA and second-design effects remain open; no stable RL or comprehensive PPA claim |
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

At06:50 HKT, bp_be first online1794723 COMPLETED11m15 and exported24
fixed requests. Both arms updated600 steps on64 original+8 own labels;
flow/actor/critic changed207/207/16 parameter tensors in each record.
Preservation1794767 completed5s:101328287bytes,954 exact tensors and all
other non-optimizer fields; manifest SHA
`3c792f8a9428c0abd6f4c5cec0d4354f15f1b6f2e6066dcea76aac6bb4e4a995`.
Only the new scoring array1794768 throttle changed4→8, retaining actual
8CPU/24GiB and all physical/learning settings. Final online update, full
new policy PPA and scientific consolidation still remain incomplete.

At07:00 HKT, expanded second online1794779 actually COMPLETED8m55:
both arms trained600 updates on384 original+16 own measurements, with
207/207/16 changed flow/actor/critic parameter tensors. Final24-request
scoring1794808 is live, with only its array throttle2→8 changed; actual
8CPU/12GiB and scientific settings remain unchanged. Exact inference-state
preservation1794813 completed4s,101335083bytes and954 tensors/all other
non-optimizer fields. Manifest SHA
`416aa2b5324944c2ae53000ea892343e42de6038832a7f3df192b09ada4ebd88`.
Used-feedback preservation1794812 completed17s and fully decoded all5647
files for128 actual logical training observations across main seeds0/1/2
and expanded384 seed0. It preserves input DEF/actions/native GRT evidence
without evaluation feedback or source mutation; this is not a full resume
archive or new detailed-route result. Its manifest SHA is
`34001417cc81a29d22c175917882cd036f36372089d29004ba1bbf97ea9a1721`.
bp_be final update, both new-campaign final PPA and consolidated judgment
still prevent whole-goal completion.

At07:04 HKT, independent actual-tensor audit1794837 COMPLETED9s for all
six newly completed arm/stage pairs: expanded384 rounds1/2 and bp_be round1.
It reread each real parent/child checkpoint, recomputed exact parameter
changes, checked600 iterations/config and checkpoint-bound replay SHA, and
confirmed the three actual training jobs COMPLETED0:0. Every stage has
207/207/16 changed flow/actor/critic tensors; this is stronger than reading
completion records alone. Evidence SHA
`f316963c999014863699c9dd5c6f42a96042a38ee06aea2eec231d4346bb194c`,
`graph_o2o_actual_online_audit_v1/phase_01/job_1794837/evidence.json`.
Final layout quality and bp_be round2 remain unverified and incomplete.

At07:07 HKT, bp_be first-online feedback1794768 has actually completed
all24 GRT observations successfully. Its controller started second online
1794848 at07:06:43, with actual L20 allocation and16 own observations per
arm, replay SHA
`a7a2a0aa10fc4bbe1ffc1fe3ab2583328f1e6405e5282e01bcc448a13117cc83`.
The first-online FQL GRT mean rWL is only0.072% lower than BC while WNS/TNS
are worse; this does not establish a final or comprehensive PPA benefit.
Cross-budget frozen-request metadata also confirms all36 BC proposals and
executed coordinates match across64/384-label seed0 runs, while all36 FQL
proposals differ. The BC runs are paired controls, not independent seeds;
this metadata-only identity check does not prove final physical equality.

At07:20 HKT, expanded all76 GRT observations and actual-training comparison
1792866 have completed. Final FQL mean GRT wirelength2.2952435m is1.255%
below paired BC2.324416m, with worse WNS/TNS. Fixed-Q audit1792947 and
independent recomputation1794868 cover624 predictions on52 logical cases,
40 unique raw proposals after removing cross-budget duplicate BC. On the
same40 proposals, final original/expanded Q MAE is0.045450/0.049214 and
Spearman0.0792/0.0236; the declared four-case primary remains unchanged.
Neither critic accuracy nor stable policy benefit is established. First
summary1794859 failed only while copying dependency evidence because of a
shadowed path variable; failure and source/plan retained, repaired1794866
passed. Plot-only readability revision1794868 passed with exactly identical
scientific values, and both final figures were visually inspected.

bp_be final online1794848 actually COMPLETED10m35, with16 own observations
per arm. Independent actual-tensor/replay audit1794882 passed both phases;
inference-state backup1794875 preserved101328287bytes and954 exact tensors.
Used-feedback backup1794876 preserved all32 actual logical collect outcomes,
1452files/510468889 source bytes, fully decoded and source-checked. All five
completed final online replay scopes now have durable native feedback and
both model arms retained. Optimizer/full OpenDB/SPEF remain at source; no
claim of a complete training/EDA resume archive is made.

Current live handles were actually polled: bp_be controller1793204 and its
24-request final GRT array1794869; native-route controllers1792865/1793205
with16/21 and9/23 requests complete, respectively. Four physical route
workers and eight GRT array children were actually live. Only array1794869
throttle changed4→8 after observing100 eligible idle scavenger threads;
actual8CPU/24GiB per task, frozen actions/config/backend unchanged. Final
reports1792868/1793511 and consolidation1794697 retain pending dependencies.
Native-evidence retention1794899/1794901 is registered after the corresponding
final audits, not counted as completed. Actual1794900 rejects the existing
same-implementation GRT-only report as final PPA before any archival work.
Previous prompt-only response is not counted as scientific progress; this
continuation re-read the full objective and moved actual audit/preservation
and documentation state forward. No jobs were restarted because a poll
expired, and the paused old Table2 jobs were not resumed.

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
