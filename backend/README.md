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

New GRT runs default to the existing home-backed `results` directory and retain all intermediates. Ceph quota checks found approximately 573 GB of user headroom on `/fact_home/bohanli` versus 59 GB on `/fact_data/bohanli`, despite much more aggregate filesystem free space. The default therefore stays on home storage. Each new run records current quota usage and requires at least 8 GiB headroom. Optional `--storage bulk` creates a compatibility symlink under `results` to `/fact_data/bohanli/flowplace_fql_backend/grt_results`; it is not the batch default, and no existing run is moved. The measured 96-run bp_fe batch uses approximately 34 GB with all intermediates retained.

For final fullflow comparisons on a selected subset, `scripts/evaluate_batch.slurm` accepts the same JSONL schema and array override. It requests 8 CPUs/24GB per task, max 2 concurrent, and retains the original detailed-route iteration limit. This helper is also not submitted automatically.

Per-task reusable outputs:

- `results/<run_id>/metrics.json`: rWL in microns, WNS/TNS in ns, reported power/area, horizontal/vertical congested edge counts, sums and maxima of positive demand-minus-capacity. Congestion sums are **not percentages** and are **not detailed-route DRC**.
- `results/<run_id>/manifest.json`: input hash, scheduling ID, tool/config provenance, stage runtimes, completion flag.
- `results/<run_id>/orfs_work/logs/nangate45/<design_nickname>/eval/5_grt_label.json`: raw OpenROAD metrics.
- `results/<run_id>/orfs_work/reports/nangate45/<design_nickname>/eval/5_fql_global_route_label.rpt`: timing/power report.
- `.../reports/.../congestion_post_recover_power.rpt`: final global-routing overflow regions.

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

Slurm job **1783972**, run `results/grt_released_bp_fe_01`, completed the same DEF through global route in **141.22 seconds**: macro import 1.67, floorplan/place/CTS 98.91, global route and reporting 39.74 seconds. Its CTS ODB, global-route ODB, and final congestion report are **byte-identical** to the fullflow checkpoints; all shared physical metrics match. See `fullflow_comparison.json`. Extra report warnings and wall-time metrics differ, as expected.

Validated cheap labels: rWL 2,491,854 um; WNS -0.405417 ns; TNS -226.623 ns; reported power 0.167680 W; standard-cell area 59,268.3 um²; horizontal/vertical overflow sums 836/844 over 231/240 edges. The author's first-log GRT wirelength is 2,487,041 um, explaining that parser discrepancy without a layout difference. At this measured rate, 96 labels at two concurrent 8-CPU tasks take about 1.9 hours of execution, excluding queue time and variable layout difficulty.

## First newly generated layout through the full backend

Job **1783970**, `results/generated_bp_fe_seed300_01`, evaluates the independent original-GPU-mode sample saved at `../baseline/artifacts/pretrained_original/bp_fe/seed_300`. It is distinct from the later deterministic stream. Coordinate export and all 11 macro geometry checks passed. Fullflow took 461.80 seconds and reported detailed-route wirelength 2.152282 million um, WNS -0.669114 ns, TNS -419.931 ns, power 0.175 W, standard-cell area 60,285.2 um², and 12 DRC violations. Relative to the released DEF, this single sample has 0.79% lower wirelength but worse timing and higher power/area; it is not a FQL improvement result and is not DRC clean. See `released_comparison.json` and the raw retained reports.

All six original-mode stream placements were also exported and geometry-checked successfully under `results/official_stream_seed300_exports`; no fullflow jobs were automatically submitted for those six exports.
