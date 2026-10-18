# FlowPlace Table 2: reproduction protocol audit

Audit date: 2026-10-17. The user clarified that **Table 2**, not Table 3, is
the intended baseline. This document records a read-only source/configuration
audit; it does not claim new experimental results. The earlier
`iccad/table3_protocol.md` concerns a different experiment and remains separate.

## Target and exact published values

The immediate target is now **FlowPlace and DREAMPlace on all six OpenROAD
designs**. DREAMPlace generation and its public-implementation limitations are
specified separately in `table2_dreamplace_protocol.md`; both methods must use
the same frozen physical backend. The FlowPlace-specific tracks below remain:
There are two distinct tracks: (A) replay the authors' released DEFs through
their released backend; (B) generate new DEFs using the released model and
sampler, then evaluate the same backend. Track A tests the backend without
model randomness; track B tests the complete available inference pipeline.
Neither requires retraining the model or the other Table 2 competitors.

Paper Table 2 (arXiv v1, PDF p. 6; `artifacts/flowplace.txt:346`):

| Design | rWL (million µm) | WNS (ns) | TNS (ns) | Power (W) | Cell area (µm²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ariane133 | 7.62 | -1.10 | -2874.29 | 0.44 | 384905 |
| ariane136 | 8.14 | -1.37 | -3845.98 | 0.50 | 397798 |
| bp | 10.03 | -4.74 | -23.87 | 0.53 | 531359 |
| bp_be | 3.10 | -1.19 | -316.39 | 0.20 | 121846 |
| bp_fe | 2.33 | -0.56 | -124.48 | 0.21 | 72407 |
| swerv_wrapper | 4.54 | -0.66 | -586.54 | 0.27 | 227567 |

`backend/assets/paper_table2_flowplace.json` matches these transcribed values.
They are rounded published values, not raw run data. The FlowPlace average
ranks are rWL **2.83**, WNS **2.33**, TNS **1.83**, power **2.00**, area **2.17**.
Ranking is per design among the five methods, averaged over six designs;
smaller rank is better. Raw rWL/power/area are minimized; signed WNS/TNS are
maximized toward zero. The paper's rounded power values obscure ties, and
its exact tie-handling/raw power precision is not stated. A FlowPlace-only
reproduction cannot independently reproduce five-method ranks. Comparing new
FlowPlace results to other methods' printed values would mix backend versions
and should not be presented as a controlled ranking experiment.

The paper does not specify per-design random-seed counts, whether printed
values are selected samples or averages, confidence intervals, or the exact
OpenROAD executable revision. Do not infer a best-of-N selection rule.

## Correct tools and metric stages

Table 2 uses **Nangate45 and the authors' OpenROAD full flow**: global and
detailed placement, placement optimization, CTS, global routing, detailed
routing, and final reports. Evidence: PDF pp. 4–5, text lines 194–210 and
244–260; upstream README lines 99–110. **Innovus EarlyGlobalRoute is the
ICCAD Table 1/Table 3 backend**, not a Table 2 requirement.

For OpenROAD graphs, the upstream README mentions DREAMPlace for converting
generated macro coordinates to DEF. The released evaluator then imports macro
locations and rebuilds standard-cell placement through OpenROAD; it does not
use DREAMPlace standard-cell placement as the Table 2 implementation flow.
The local coordinate exporter reproduces that mapping without a DREAMPlace
run. Its ID/size/scaling/orientation checks and bp_fe byte-identical roundtrip
are documented at `backend/README.md:49`. This adapter should remain an
explicit conversion deviation, with per-design mapping verification retained.

The evaluator (`upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py`)
prefers `config_eval.mk` (line 23), reuses the released `1_synth.v` and
`1_synth.sdc` (line 82), runs `make run_mp` (line 432), then
`make run_wo_synth` (line 482). This starts from released synthesis artifacts,
not a fresh synthesis with the installed Yosys version.

| Reported quantity | Required provenance |
| --- | --- |
| rWL | Detailed-route wirelength, normally `5_2_route.json` / route log; convert µm to million µm |
| WNS, TNS | Final `6_report.json` / `6_finish.rpt`, setup timing in ns |
| Power | Final reported total power under released analysis/activity assumptions, W |
| Cell area | Final standard-cell area (`finish__design__instance__area__stdcell`), µm² |
| DRC | Final detailed-route violations, supplementary physical-validity result |

The author's parser first reads JSON but can overwrite timing/power with text
report values (lines 213–279); retain raw JSON and text precision, and use the
existing independent fullflow audit. Its JSON WNS key is setup `ws`; all
published slack values here are negative, but a future positive worst slack
must not be silently labeled negative slack without declaring the convention.
GRT wirelength, macro HPWL, congestion labels and intermediate timing are not
substitutes for these Table 2 final-stage values. Adding macro area to standard
cell area is not the published cell-area interpretation used by the evaluator.

## Sampling and checkpoint recipe

Pinned source: `upstream/flowplace/SOURCE_COMMIT` =
`da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547`. Public archive:
`mask.61.train_mask_flowplace_uniform.61.tar.gz`; archive SHA256 is in
`baseline/assets/manifest.json`. The loaded `best.ckpt` SHA256 is
`0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359`.

Use the released model with uniform prior, 20 flow steps, hard constraints on,
regularity weight 0.1 and distance weight 1.0. Source:
`upstream/flowplace/scripts/test_flow_matching.sh:4` and
`flow_matching/configs/config_eval.yaml:16` (seed 300). For Table 2 set
`task=openroad-s0`, `num_output_samples=6`: the loop indexes six graph entries,
not six samples per graph (`flow_matching/eval.py:200`).

The parsing notebook's graph order is **ariane133, ariane136, bp_be, bp_fe,
bp, swerv_wrapper** (`benchmark_utils/parse_openroad.ipynb:461`), different
from paper table column order. Global RNG order therefore matters. The local
`baseline/run_official_stream.py` seeds before one model initialization and
samples all designs in this order, storing the actual initial noise and
trajectories. It reconstructs the released evaluation RNG convention but does
not establish the paper's undocumented original RNG history.

Two existing six-design streams have independent overlap/boundary checks:

- `baseline/artifacts/official_stream_seed300`: original GPU mode, job 1783973.
- `baseline/artifacts/official_stream_deterministic_seed300`: explicit
  deterministic runtime, job 1784006, CuBLAS workspace `:4096:8` and
  `torch.use_deterministic_algorithms(True)`.

Both used NVIDIA L20 / Torch 2.2.1+cu121. The original-mode stream already has
exported DEFs at `backend/results/official_stream_seed300_exports/`; their
existence alone is not backend completion. Per-design seed resets and global
RNG sampling are separate protocols. The known original-mode GPU variability
and hard-projection sensitivity are documented in `baseline/README.md:38`.
Choose one declared stream for a complete track, retain it unchanged, and do
not pick different favorable streams per circuit.

## Released flow versus paper and local implementation

The public flow archive SHA256 is
`4a0e7841a9ea6f580b3253bf7e57f696dde73f0e03545e2d4026c6229cba631f`;
the released DEF archive SHA256 is
`a881c7b71a6b4c49d3e1a52418274244cd277edfa01397d4a830a4d69ca2251e`.
URLs and sizes are recorded in `backend/assets/download_manifest.json`.

Material limitations already established by source and logs:

1. All six `config_eval.mk` files explicitly set `GPL_TIMING_DRIVEN=0`,
   `GPL_ROUTABILITY_DRIVEN=0`, `SKIP_DP_CHECK=1`. The paper describes timing-
   and congestion-driven global placement. Preserve the released recipe as
   the primary reproduction; any paper-text interpretation is a separate,
   predeclared diagnostic. BP design directories are `black_parrot`,
   `bp_be_top`, `bp_fe_top`, with nicknames `bp`, `bp_be`, `bp_fe`.
2. The released Nangate45 platform caps detailed routing at iteration 4
   (`platforms/nangate45/config.mk:112`). Fullflow completion can retain DRC.
   Existing cap20 diagnostics are separate and do not replace cap4 results.
3. The installed image is newer than the paper:
   `openroad/orfs:26Q4-151-g8b5e62fbb`; image SHA256
   `85f0ce74d79a79c1190632f775f0ed818c8de6e0fafd87717ae3d4839207ea4c`.
   Its OpenROAD version prints `unknown`; binary/image hashes are the anchors.
   It supplies executables while the author's scripts/PDK are used, after
   clearing inherited flow path variables (`backend/scripts/evaluate.slurm:20`).
4. One compatibility patch qualifies Tcl `$env(RESULTS_DIR)` as
   `$::env(RESULTS_DIR)` in `write_ref_sdc.tcl`, avoiding a procedure-scope
   failure. Preserve the archived original and patch hash.
5. Released bp_fe runs retain a caught `PDN-0113` error. Even a 0 signal-route
   DRC count is not proof of complete power-network validity or signoff.
6. Published swerv DEF has three raw macro overlaps verified independently;
   the released-DEF flow fails during macro import. Keep the failure rather
   than silently legalizing it or replacing it with a new generated sample.

Prior results in `analysis/artifacts/table2_released_v1/comparison.md` are
five completed released-DEF fullflows and the swerv failure, with material
TNS/area differences from the paper. These are backend execution results,
not an exact numerical reproduction. Separate generated bp_fe/swerv controls
already exist but do not constitute a consistent six-design generated track.

## Next execution and reporting criteria

First verify provenance of existing inputs and retain the released-DEF track.
Then complete one consistently sampled six-design generated track through the
same fullflow and independently audit all five PPA metrics and DRC. Reuse
already completed exact-input/exact-flow computations only with hash-bound
evidence; different RNG streams are not interchangeable. Preserve failures
and never average only successful designs while claiming a full-table mean.
Report per-design raw values and differences from the rounded paper values,
with tool/config deviations alongside them. Backend diagnosis should test
specific hypotheses on fixed inputs, not tune settings to recover table values.

All heavy work requires Slurm. The older storage paragraph at
`backend/README.md:71` predates the written-policy audit and is superseded by
`docs/cluster_execution.md:27`: new datasets/weights belong in `/fact_data` or
`/scratch`, temporary files in `/scratch`. Existing large files require a
planned move with provenance preservation. New scratch-backed runs also need
the corresponding container bind and TMPDIR configuration; the historical
`evaluate.slurm` binds only the project and backend `/fact_data` directory.
