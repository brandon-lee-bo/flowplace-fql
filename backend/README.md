# FlowPlace Table 2 backend reproduction

Scope: evaluate official released macro placements with the author's released Nangate45 OpenROAD flow. This is a backend reproduction first; evaluating a released DEF does **not** reproduce model generation or training.

## Inputs and provenance

- Paper: https://arxiv.org/pdf/2604.23658v1, Table 2 (page 6). Rounded FlowPlace values are transcribed in `assets/paper_table2_flowplace.json`.
- Official release: https://github.com/lamda-bbo/flowplace/releases/tag/v1.0.0
- Download manifest and SHA256: `assets/download_manifest.json`.
- Original release archives remain under `assets/`; extracted flow is `thirdparty/OpenROAD-flow-scripts`; released DEFs are `releases/flowplace_DEFS/openroad`.
- Tool image: `docker://openroad/orfs:26Q4-151-g8b5e62fbb`; Docker multiarchitecture index digest `sha256:d7ee20aa21a71ded1ddc7f87b53d48ea69238befad28c1b2c8ff9e92e9771699`. Build job records actual executable version and image SHA256. Image lives under `/fact_data/bohanli/flowplace_fql_backend/containers/`.
- This image is newer than the paper. The exact author's OpenROAD version is not documented in the release archive; agreement with the paper cannot be assumed.

## Scheduling and isolation

All nontrivial container build and evaluation work uses Slurm. The original host container pull was stopped after the explicit scheduling instruction; no evaluation was run outside Slurm. Jobs use only user-owned directories and do not alter global packages. `scripts/build_container.slurm` requests cpu-scavenger/4 CPU/12GB. `scripts/evaluate.slurm` requests eda/8 CPU/24GB, restricts OpenROAD and numerical threads, and retains intermediates.

```sh
sbatch scripts/build_container.slurm
sbatch scripts/evaluate.slurm --design bp_fe --run-id released_bp_fe_01
```

Use unique run IDs; the wrapper refuses to overwrite existing run directories. To evaluate another DEF pass `--def-path /absolute/file.def`. The wrapper imports the unmodified official evaluator but passes this backend directory as root. It writes `manifest.json`, `metrics.json`, all stage logs/reports/ODBs, and fails if final PPA metrics are incomplete.

## Known fidelity concerns

1. Paper section 4.1 describes timing-driven and congestion-driven global placement. Released `config_eval.mk` for bp_fe explicitly sets `GPL_TIMING_DRIVEN=0` and `GPL_ROUTABILITY_DRIVEN=0`. Initial reproduction preserves the released configuration, records this discrepancy, and does not silently change settings to chase published results.
2. `SKIP_DP_CHECK=1` is in the author's config. Backend completion must not be claimed as proof of zero placement overlap. Record macro geometry and detailed-route DRC separately.
3. Power is the OpenROAD/OpenSTA report under the released timing/activity assumptions; no vector-based power simulation is present in this flow.
4. The import step reads only macro placements from the DEF, then rebuilds standard-cell placement from included synthesized Verilog and SDC. It is unnecessary to reuse the DEF's standard-cell placement.
5. bp_fe: 11 SRAM macros, DEF database units 2000 per micron, die 800 by 600 microns, nominal core [5,5,795,595]. Macro orientations are mixed N/FS and affect pin positions.
6. The released platform config limits detailed routing to iterations 0–4 (`-droute_end_iter 4`). The first full bp_fe replay retains 1,904 DRC violations. A completed flow with metrics is **not** a DRC-clean implementation. Keep this limit fixed in comparisons; do not tune it to recover the published numbers.
7. The author's evaluator takes the first logged global-route wirelength, before incremental repair. Cheap-label `rWL_um` instead reads the final global-route JSON value, and is explicitly an intermediate-stage metric. Detailed-route wirelength remains the Table 2 comparison metric.
8. The released PDN stage emits `PDN-0113: Cannot find shapes to connect to on metal4`, catches the error, and continues in both successful bp_fe fullflows and the released GRT validation. Completed signal-routing/PPA results therefore do not establish a valid power-distribution network or complete signoff. This common limitation is distinct from a terminal CTS failure.
9. The released `swerv_wrapper.def` has three overlapping macro rectangles under the released LEF dimensions, already present before OpenROAD snapping. Task1783836_4 stops at `MPL-0041` while placing DCCM bank3 beside bank1 (3.42um horizontal overlap). Read-only geometry audit1784533 finds no raw macro overlaps in the other five released DEFs. Hash-bound evidence is in `results/released_def_geometry/` and the failed swerv run's `failure_analysis.json`. This failed row is retained; no overlap check or coordinate is relaxed to force completion.

## Failures retained

- Docker daemon unavailable; native OpenROAD absent. Singularity 4.5 is installed and successfully executes an Alpine test.
- Git through inherited proxy failed; direct Python Requests downloads (`trust_env=False`) succeeded.
- Job 1783808 failed immediately: Singularity CE does not support the Apptainer-style `--mksquashfs-args` flag. Job 1783809 uses the supported pull command within Slurm CPU restrictions.

### First evaluated attempts

- Container job 1783809 completed in 5m33s. Image SHA256: `85f0ce74d79a79c1190632f775f0ed818c8de6e0fafd87717ae3d4839207ea4c`. OpenROAD's version string in this image is `unknown`; image/binary hashes are therefore the precise reproducibility anchors.
- Evaluation 1783818 (`released_bp_fe_01`) imported the DEF successfully, then failed because sourcing the image's env.sh also exported FLOW_HOME pointing at the image's newer flow. No PPA result was produced. The wrapper now clears inherited flow path variables so the author's released scripts and PDK are used.
- Evaluation 1783828 (`released_bp_fe_02`) is the corrected run; monitor Slurm and its per-stage files, not the existence of a metrics file alone.

### Macro coordinate handoff

`export_macro_def.py` takes the generator's `macro_physical.json` mapping numeric cache node IDs to `[lower_left_x, lower_left_y, width, height]` in cache units. It verifies IDs against template component order, matches **every** macro dimension against the released LEFs, recovers site scaling and row origin, follows the author's round/unscale/shift step, and checks bounds/overlap. Orientations remain those of the released template. The original pre-generation DEF is absent, so unchanged orientation relative to that original is an assumption rather than independently proven.

A roundtrip test converts the released bp_fe DEF back into cache coordinates and exports it again. The result is byte-identical (SHA256 `9fb14ce342b5011a324afcb537451b49f37c7e908c4c508bed4b0d867dd24710`), verifies all 11 macros, and has no macro overlap or row-bound violations. Test artifacts are retained in `results/export_roundtrip`.

- Job 1783828 reached and completed global routing, but the released flow has a Tcl scope bug: `write_ref_sdc.tcl` is sourced inside `global_route_helper` and accesses local `$env(RESULTS_DIR)` rather than global `$::env(RESULTS_DIR)`. Patch `patches/001-global-env-scope.patch` makes only that qualification change. The script still restores the original active clock after writing its optional reference SDC. Job 1783834 (`released_bp_fe_03`) is the fresh full rerun. The original archive and original file are preserved; patched-file hashes are recorded.

## Global-routing labels (distinct from final Table 2)

`label_grt.slurm` uses the same floorplan, placement, CTS, and global-routing stages as the full backend. A separate Tcl wrapper sources the author's routing script then calls its report function while global-routing parasitics remain live. It stops before detailed routing and extraction. The fullflow default is unchanged.

```sh
sbatch scripts/label_grt.slurm --design bp_fe --run-id grt_released_bp_fe_01
# Batch: one JSON object per line, with design, absolute def_path, unique run_id.
sbatch --array=0-95%2 scripts/label_grt_batch.slurm /absolute/tasks.jsonl
```

The batch script **is not submitted automatically**. Each task requests 8 CPUs/12GB, max 2 concurrent (as submitted above), and disables automatic requeue to prevent ambiguous duplicate artifacts. Failed tasks remain explicit and can be resubmitted under new run IDs.

For the fixed 96-label collection, root later increased array1784047's scheduling throttle from 2 to 4 after checking cluster capacity and quota. Per-task CPU/memory and every physical setting stayed unchanged. Final audit1784160 validates all 96 requested labels with zero missing/error cases and one common physical-flow revision; the two original parser failures remain preserved as described below.

New GRT runs default to the existing home-backed `results` directory and retain all intermediates. Ceph quota checks found approximately 573 GB of user headroom on `/fact_home/bohanli` versus 59 GB on `/fact_data/bohanli`, despite much more aggregate filesystem free space. The default therefore stays on home storage. Each new run records current quota usage and requires at least 8 GiB headroom. Optional `--storage bulk` creates a compatibility symlink under `results` to `/fact_data/bohanli/flowplace_fql_backend/grt_results`; it is not the batch default, and no existing run is moved. At approximately 350 MB per measured bp_fe run, a 96-run batch is expected to use about 34 GB with all intermediates retained.

For final fullflow comparisons on a selected subset, `scripts/evaluate_batch.slurm` accepts the same JSONL schema and array override. It requests 8 CPUs/24GB per task, max 2 concurrent, and retains the original detailed-route iteration limit. This helper is also not submitted automatically.

Per-task reusable outputs:

- `results/<run_id>/metrics.json`: rWL in microns, WNS/TNS in ns, reported power/area, horizontal/vertical congested edge counts, sums and maxima of positive demand-minus-capacity. Congestion sums are **not percentages** and are **not detailed-route DRC**.
- `results/<run_id>/manifest.json`: input hash, scheduling ID, tool/config provenance, stage runtimes, completion flag.
- `results/<run_id>/orfs_work/logs/nangate45/<design_nickname>/eval/5_grt_label.json`: raw OpenROAD metrics.
- `results/<run_id>/orfs_work/reports/nangate45/<design_nickname>/eval/5_fql_global_route_label.rpt`: timing/power report.
- `.../reports/.../fql_grt_marker_status.json` and, when markers exist, `fql_final_congestion.rpt`: final global-routing overflow evidence. Legacy positive cases also retain `congestion_post_recover_power.rpt`; see the parser correction below.

Required fields are checked; an absent metric causes failure rather than being replaced by an invented zero. A completed cheap-label run is explicitly marked `stage=global_route_only` and `not_final_table2=true`. Validation against the matching fullflow GRT checkpoint is required before launching an offline label dataset.

## First full replay and cheap-label validation

Slurm job **1783834**, run `results/released_bp_fe_03`, completed all released flow stages in 815.84 seconds using 8 CPUs. Detailed routing used 630.61 seconds. Results versus the rounded paper row:

| Metric | Released-flow replay | Published Table 2 |
| --- | ---: | ---: |
| Detailed-route wirelength (million um) | 2.169485 | 2.33 |
| WNS (ns) | -0.57361 | -0.56 |
| TNS (ns) | -359.085 | -124.48 |
| Reported power (W) | 0.171 | 0.21 |
| Standard-cell area (um²) | 59,268.3 | 72,407 |
| Detailed-route DRC | 1,904 | not reported |

This validates a working evaluation pipeline, **not numeric reproduction of the paper**. Original tool provenance is incomplete and the published/released placement settings differ; both facts are retained rather than compensated by parameter tuning. The full final JSON reports total instance area 220,243 um², macro area 160,975 um², and standard-cell area 59,268.3 um². Adding macro area does not explain the 13,138.7 um² standard-cell-area gap to the paper. Power is reported analysis under released assumptions, not hardware measurement.

`python3 scripts/audit_fullflow.py results/<run_id>` independently checks the author's final metrics against raw route/finish JSON and the detailed DRC report, accounting for the text report's three-significant-digit power rounding. It saves `fullflow_audit.json`, per-stage times/memory, and DRC type counts without changing the physical flow.

Slurm job **1783972**, run `results/grt_released_bp_fe_01`, completed the same DEF through global route in **141.22 seconds**: macro import 1.67, floorplan/place/CTS 98.91, global route and reporting 39.74 seconds. Its CTS ODB, global-route ODB, and final congestion report are **byte-identical** to the fullflow checkpoints; all shared physical metrics match. See `fullflow_comparison.json`. Extra report warnings and wall-time metrics differ, as expected.

Validated cheap labels: rWL 2,491,854 um; WNS -0.405417 ns; TNS -226.623 ns; reported power 0.167680 W; standard-cell area 59,268.3 um²; horizontal/vertical overflow sums 836/844 over 231/240 edges. The author's first-log GRT wirelength is 2,487,041 um, explaining that parser discrepancy without a layout difference. At this measured rate, 96 labels at two concurrent 8-CPU tasks take about 1.9 hours of execution, excluding queue time and variable layout difficulty.

## First newly generated layout through the full backend

Job **1783970**, `results/generated_bp_fe_seed300_01`, evaluates the independent original-GPU-mode sample saved at `../baseline/artifacts/pretrained_original/bp_fe/seed_300`. It is distinct from the later deterministic stream. Coordinate export and all 11 macro geometry checks passed. Fullflow took 461.80 seconds and reported detailed-route wirelength 2.152282 million um, WNS -0.669114 ns, TNS -419.931 ns, power 0.175 W, standard-cell area 60,285.2 um², and 12 DRC violations. Relative to the released DEF, this single sample has 0.79% lower wirelength but worse timing and higher power/area; it is not a FQL improvement result and is not DRC clean. See `released_comparison.json` and the raw retained reports.

All six original-mode stream placements were also exported and geometry-checked successfully under `results/official_stream_seed300_exports`; no fullflow jobs were automatically submitted for those six exports.

The released `swerv_wrapper` DEF fails before routing because its raw macro coordinates contain three overlaps, independently verified against the released LEF in `results/released_def_geometry/swerv_wrapper.json`. The original failed run and hash-bound `failure_analysis.json` remain at `results/released_swerv_wrapper_job1783836_4`. A separate generated control uses the already-saved deterministic seed300 swerv placement; export verifies all 28 macros with no overlaps or row-bound violations. Job1784543, `results/generated_swerv_deterministic_seed300_job1784543`, completed the same unmodified author flow in 3,146.42 seconds with final numerical audit passing: detailed-route wirelength 4.163184 million um, WNS -0.588714ns, TNS -681.557ns, reported power 0.249W, standard-cell area 182,038um², and 7,282 routing violations. Thus the flow can evaluate a legal generated swerv placement, but this control is neither DRC-clean nor a substitute for the failed released-DEF Table2 row.

## GRT parser correction during the 96-label collection

Array 1784047 initially exposed a reporting edge case: tasks 0 and 1 completed routing and timing analysis, but parser v1 failed because no congestion report existed. Missing files are **not** a valid zero label. Pending tasks were briefly held while the saved final ODBs were inspected under Slurm. Job 1784105 established directly that both final databases have no `Global route` marker category; their zero overflow labels are therefore backed by database evidence. Existing positive case row0003 had exactly 23 final markers, and its PPA and H/V overflow values agree exactly with the original parser.

`write_grt_marker_status.tcl` now reads the final marker category/count and exports the marker report. These are report-only operations, executed after the unchanged global-routing optimization; no placement, routing, or optimization command was added or changed. `grt_metrics.py` checks finite PPA values, final checkpoint presence, the exact marker count, and each `usage - capacity = overflow` entry. It never substitutes an earlier `congestion.rpt` for missing final evidence. The first full task under this revision (task7/row0010) completed successfully in 147.72 seconds with live final marker evidence.

Recovered results keep original failed manifests as `manifest.before_parser_v2.json`, preserve their original Slurm job IDs and terminal status, and record postprocessing recovery events. No routing computation was repeated. The first six tasks executed the timing-only reporting wrapper; remaining tasks use the marker-reporting wrapper. Physical-flow identity is recorded separately from reporting and parser revisions, so this distinction remains explicit. The archived `assets/global_route_label_v1.tcl` matches the recorded v1 hash. Tasks0/1 predated runtime wrapper hashing; their reconstructed revision is marked retrospective, rather than represented as a runtime capture.

To audit every requested completed label, validate raw reports again and normalize missing metadata:

```sh
python3 scripts/audit_grt_tasks.py ../datasets/bp_fe_grt_request_v1/tasks.jsonl \
  --stamp-complete --output results/bp_fe_grt_v1_audit.json
```

The audit distinguishes not-started/running/failed tasks from validated labels, checks input DEF hashes against the requested manifest, and reports physical-flow/reporting/parser revision counts. Final completeness requires all requested labels and one common physical-flow revision. Root owns collection/relabel jobs; the backend has not submitted the separate 36-layout proxy collection itself.

The proxy candidate `bp_fe_proxy_transfer_v1_m02_i0242` failed during CTS hold repair with terminal `RSZ-0060: Max buffer count reached` after inserting 8,208 hold buffers. This is a physical-flow failure, not a parser failure or preemption. No GRT metric or penalty value is invented. `physical_failure.json` binds the original failed manifest, DEF, action/task, audit, and error log by hash, preserves Slurm terminal evidence, and leaves the original manifest unchanged. Only this specific proven failure class is currently recognized by `scripts/classify_physical_failure.py`; exploratory summaries must report feasibility alongside any conditional PPA statistics. Strict all-success audits remain unsatisfied when such a failure occurs.

Sampler array1784158 task0 (`bp_fe_sampler_transfer_v1_n00_i0240`, raw job1785399) was preempted after 62 seconds during CTS setup repair. Its incomplete manifest, work directory and hash-bound `preemption.json` remain unchanged. Same-input retry1785454 runs under separate ID `bp_fe_sampler_transfer_v1_n00_i0240_retry1` and completed GRT in 159.02 seconds on eda with 8 CPUs/12GB; no sample or physical parameter changed. The frozen `assets/sampler_infrastructure_retry_map_v1.json` binds the original task manifest, interrupted attempt, Slurm evidence and new manifest/metrics hashes. Optional `--infrastructure-retry-map` on the terminal classifier and GRT audit preserves the original logical task and denominator while explicitly resolving its successful retry. It rejects an unbound replacement, changed input/flow, incomplete retry or a physical/parser failure; an infrastructure interruption is never treated as a physical outcome. `verify_infrastructure_retry.py` reparses the retry's raw metrics and exercises rejection of corrupted bindings without modifying either attempt.

Retry integrity job1785462 passed its raw-metric comparisons and all six binding-rejection checks. Final sampler classification1785468 accounts for all 11 logical outcomes: ten successful evaluations and one terminal CTS failure for `fixed_zero_anchor` (`n06_i0240`, raw job1785470), which inserted 8,221 hold buffers before `RSZ-0060`. The failed fixed control is counted once, has no invented PPA values, and is not retried. `results/sampler_transfer_recovered_v1_classification.json` records the one successful infrastructure recovery separately from this physical failure.

## Separate detailed-route iteration diagnostic

Authorized diagnostic array1784131 uses only the two already-evaluated bp_fe layouts and their saved, hash-verified global-route ODB/SDC checkpoints. It changes only `DETAILED_ROUTE_ARGS` from cap4 to cap20, then uses the same fill/final-report steps. No placement, CTS, or global routing is repeated; baseline directories are not overwritten. Resources are 8 CPUs/12GB/2 hours and one diagnostic task at a time. The released-DEF task was preempted after 5m04s on cpu-scavenger, with no final diagnostic metrics; its explicit interruption record is retained. Replacement array1784154 runs only that case on eda after the first array ends, preserving single-task concurrency.

The generated-layout diagnostic completed in 357.58 seconds for DRT plus finish. Its DRC sequence was `10543,3210,2623,97,12,11,11,2,2,0`: first five iterations match the cap4 baseline, and iteration9 clears the reported routing violations. At the same final-report point, wirelength changes from 2,152,282 to 2,152,303 um; WNS from -0.669114 to -0.669164 ns; TNS from -419.931 to -419.959 ns. Power 0.175W and standard-cell area 60,285.2um² are unchanged. Original and copied GRT checkpoint hashes remain unchanged. This demonstrates premature routing termination for this particular sample; it is not a complete signoff or a replacement for the released-flow baseline. Results: `results/diagnostic_drt20_generated_bp_fe_seed300_01_job1784131`.

The released-layout cap20 diagnostic completed in 4,781.01 seconds (job1784154; numerical audit1785142 passed). Its DRC sequence was `14171,5167,4719,2016,1904,1830,1820,1793,1793,1654,1092,996,482,412,340,257,257,257,210,210,192`. The remaining 192 violations comprise 170 shorts, 19 metal-spacing violations and 3 cut-spacing violations. At the same final-report point, wirelength changes from 2,169,485 to 2,169,684um; WNS stays -0.57361ns, TNS changes -359.085 to -360.475ns, and power/area stay 0.171W/59,268.3um². Original and copied GRT hashes remain unchanged. More iterations reduce violations substantially but do not make this released placement DRC-clean at cap20, and do not explain the paper's large TNS/area difference.

## Infrastructure recovery of released bp

The original released-bp attempt (array1783836 task2, raw job1784277) was preempted after 59m02s during detailed routing. Its original manifest and entire work directory are unchanged; `preemption.json` binds the original manifest and the Slurm PREEMPTED record. The held automatic requeue was canceled before it restarted. Recovery1785220 uses new run `released_bp_resume_job1785220`, copies the completed GRT ODB/SDC/guide and logs with SHA checks, then runs only the original cap4 detailed route and finish on eda with requeue disabled. This is infrastructure recovery of the same DEF, not a new placement or a tuned flow. Runtime explicitly measures the resumed work and excludes earlier pre-route computation and interrupted DRT work. The recovered row must retain its interrupted predecessor in summaries.

At recovery startup, the runner copies/hashes its source, validates the actual image/archive/patch against the original recorded hashes, and records current executed-flow file hashes checked against the archive except for the recorded Tcl patch. The interrupted attempt did not capture a startup flow-tree snapshot; this limitation remains explicit and is not replaced by a retrospective claim.

While recovery1785220 was still pending, its scheduling limit was reduced from the submitted 6 hours to 2 hours 30 minutes to permit earlier backfill. The interrupted original detailed-route log had reached iteration3 at 90% after approximately 36 minutes of routing, providing substantial but non-guaranteed margin for the unchanged cap4 rerun and final reporting. The job was temporarily held to prevent a startup race, updated, and released; its 8 CPUs, 24GB memory, eda partition and physical settings were unchanged. `assets/resume_schedule_1785220.json` preserves the before/after scheduler evidence and original-log hash; the batch source retains its original 6-hour submission line with an annotation describing the subsequent scheduler update.

A subsequent pending-only resource adjustment reduced memory from 24GB to 14GB after finding that the eda node had approximately 14.9GiB of schedulable memory. The interrupted detailed-route log's maximum reported memory was 9,786.64MB; a 14,336MiB allocation gives at least 46.5% margin even conservatively interpreting the log's MB as MiB. Startup hashes the 1.50GB image before launching routing, and the flow archive is 30.49MB. This observed margin does not guarantee the future peak. `assets/resume_memory_1785220.json` records the evidence, rejected size-suffix command and successful integer-MiB update. CPU count, partition, wall-time limit and physical inputs/settings stayed unchanged. The submitted batch's original 24GB line remains intact and annotated.
