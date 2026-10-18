# DREAMPlace Table 2 input adaptation (2026-10-17)

This is an input/protocol audit and a candidate smoke implementation, not a
claim that the paper's DREAMPlace baseline has been reconstructed exactly.
No placement was executed during this read-only audit.

## Available inputs

- Six candidate configs: `upstream/flowplace/benchmark_utils/openroad/dmp_config/or_cases/{bp_fe,bp,ariane133,ariane136,bp_be,swerv_wrapper}.json`.
- Matching geometry/timing libraries: `backend/thirdparty/OpenROAD-flow-scripts/platforms/nangate45/{lef,lib}`. Replace only the `benchmarks/or_cases/nangate45` prefix in config paths.
- Published synthesized netlists and constraints: `backend/thirdparty/OpenROAD-flow-scripts/results/nangate45/<design>/eval/{1_synth.v,1_synth.sdc}` for all six designs.
- Full placed DEF templates: `backend/releases/flowplace_DEFS/openroad/<design>.def`.
- Original `benchmarks/or_cases/<design>/replace.def` referenced by configs is **not present in the public release/local input inventory**. The cache notebook consumes pre-extracted macro graph data, not a full standard-cell netlist/DEF generator.
- Installed DREAMPlace 4.1.0 CPU: `iccad/assets/dmp_install_cpu`; interpreter `iccad/assets/dmp_env/bin/python`; source `iccad/assets/DREAMPlace-4.1.0`, commit `5d13c9001a3bc900dca1e108e633d5dd45b00701`. Prior ICCAD compilation and execution succeeded. CPU execution is an explicit difference from the paper's GPU setting.

| Design alias | ORFS config directory | Published netlist bytes | Released DEF bytes |
|---|---|---:|---:|
| bp_fe | bp_fe_top | 5,365,796 | 9,065,852 |
| bp | black_parrot | 46,244,141 | 168,048,256 |
| ariane133 | ariane133 | 28,825,970 | 54,883,674 |
| ariane136 | ariane136 | 29,519,261 | 56,145,284 |
| bp_be | bp_be_top | 8,905,285 | 15,077,138 |
| swerv_wrapper | swerv_wrapper | 18,258,186 | 26,481,903 |

A small bp_fe text audit expands the 10 Yosys modules from `bp_fe_top` and finds
33,199 leaf instances, exactly matching every released DEF `(instance, master)`
pair; no missing/extra/changed masters, and all 11 fakeram instances agree.
This is an inventory check, **not** a full pin-connectivity equivalence proof.
The other five designs require the same scheduled check before optimization.

## Avoiding a FlowPlace warm start

Use the released DEF only as a source of connectivity, instance identity,
geometry, die/rows, IO positions and orientations. Keep the original immutable.
In a derived DEF, make every true fakeram macro movable (`FIXED` to `PLACED`),
then overwrite the positions of **all** movable macros and standard cells with
a neutral center placeholder. Preserve fixed IO/nonmacro obstacles, orientations,
constraints and every non-component section. DREAMPlace's original
`random_center_init_flag=1`, with its own declared `random_seed`, replaces every
movable position with Gaussian noise around the layout center in `BasicPlace`.
Record that initial vector to demonstrate that FlowPlace coordinates were not
used as an optimization warm start.

The released config sets `macro_place_flag=0`; it is not a documented standalone
DREAMPlace macro-baseline recipe. The candidate sets it to `1` and retains
`use_bb=1`, activating the original 4.1.0 mixed-size first stage, original macro
legalization, and second stage with macros fixed. This is a documented
derivation, not an assertion about unavailable author settings. No replacement
optimizer, overlap repair or stronger custom legalizer is introduced.

Author halo values are ariane133=60, ariane136=60, bp=40, bp_be=22, bp_fe=20,
swerv_wrapper=20 in both dimensions. The 4.1.0 implementation adds these values
directly to already scaled database coordinates. They must **not** be silently
interpreted as microns or conflated with ORFS macro halos. Root/protocol review
must freeze this choice before submission. The smoke runner preserves these
values unless its explicit `--halo` argument is supplied, recording the change.

Several released keys do not exist in the installed standard 4.1.0 schema:
`ref_path`, `sol_file_format`, `remap_flag`, `macro_criterion`,
`macro_top_k_criterion_k`, `l_overlap_penalty`, `macros2place_each_step`, and
`radius_ratio_lb` (only some designs use each). A candidate must record these as
unapplied rather than claim their semantics were reproduced. The standard
4.1.0 macro mask must exactly match the intended fakeram instance set.

## Identical downstream evaluation

The released ORFS `run_macroplace.tcl` loads `MACRO_DEF` and exports only
`write_macro_placement .../macro_out`. `run_wo_synth` starts again from the
published `1_synth.v/.sdc`, and `scripts/macro_place_util.tcl` sources that macro
placement before standard-cell placement. Therefore DREAMPlace's standard-cell
positions are **not** the final evaluated positions: only its macro result is
handed off, followed by the same full ORFS flow used for FlowPlace.

After DREAMPlace completes, independently check macro count/identity,
row-consistent orientation, rectangle non-overlap and bounds, then make a separate handoff DEF
with exactly those macro positions marked FIXED. Preserve the original DMP
output as well. Verify the OpenROAD macro import actually preserves/snap-adjusts
those coordinates using the same acceptance rule as FlowPlace; report failures
rather than adding a different legalizer. For a fair pair, both methods must
use the same frozen ORFS recipe, clock/RC libraries, and tool image; the GPL
diagnostic must remain separate until a final protocol is chosen.

The candidate runner saves the original and derived DEFs, author/derived/effective
configs, unapplied custom keys, source/install/input hashes, initial coordinates,
before/after the original macro legalizer, the final DEF, geometry audits and
macro coordinates. Full PPA must still come from the separate original ORFS
evaluator; DREAMPlace HPWL is not Table 2 routed wirelength.

## First scheduled smoke and orientation correction

Slurm job `1787984`, run `dmp_public41_bp_fe_smoke_v1`, completed the stock
placement and both stock macro-legalization passes. It produced 11 macros with
zero overlaps, zero die/core violations, unchanged connectivity, and all 2,511
fixed IOs unmoved. The wrapper then failed because its original guard required
unchanged macro orientations. The failed run and its manifest remain intact.

Seven of the 11 macros changed between N and FS. This is stock behavior:
`place_io/src/place_io.cpp:60–98` applies final coordinates to every non-FIXED
node and aligns its vertical orientation to the destination row using
`PlaceDB.h::getRowIndex`. N and FS have the same width/height, while a macro pin
offset changes from `(x, y)` to `(x, height-y)` under the vertical flip.
Restoring the original orientation would change the algorithm's final result.

The revised wrapper audits each macro's before/after orientation and the actual
row, then rereads the final DEF with a fresh stock LEF/DEF database and checks
every connected macro pin offset against that transformation. This is read-only
validation after placement, not another optimization. `rawdb.apply` updates
orientation without updating existing raw offsets. In addition,
`PyPlaceDB.cpp::convertOrient` always converts its Python-side pin offsets back
to N orientation, while preserving the orientation label. The first revised
audit in job `1787991` incorrectly applied the physical h-y transformation to
those normalized Python arrays and failed. This second wrapper-audit failure
is also preserved, not accepted as a result. The corrected audit compares
Python N-normalized arrays unchanged, and separately compares actual physical
pin offsets from a freshly read **raw C++** database with the exact integer-DBU
h-y transformation. Raw node coordinates, dimensions and orientation must also
match the final DEF. This avoids treating normalized Python offsets as physical
pin locations; no placement or orientation is repaired by the audit.
The downstream OpenROAD flow reads the final DEF afresh and exports orientation
in `macro_out` (`N` → `R0`, `FS` → `MX`), so orientation is part of the handoff.
The handoff DEF changes only macro status to FIXED, preserving final coordinates
and orientation exactly.

CPU/GPU paths can differ numerically even with the same seed, and CPU wall time
cannot substantiate the paper's GPU speedup. Missing original `replace.def`,
standalone baseline settings, and exact legalizer correspondence remain explicit
limitations; neither legal geometry nor a matching version removes them.

Job `1787996`, `dmp_public41_bp_fe_smoke_v3`, passed the corrected audit:
11 macros, seven stock N/FS flips, no overlap or die/core violations, and 2,163
macro pin connections with **zero** normalized-Python offset difference and
**zero DBU** physical-raw offset difference. Its final DEF hash equals the v2
final DEF hash, confirming that this change repaired the audit, not the
placement. The original two failure manifests remain untouched. The validated
handoff is `/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2/dmp_public41_bp_fe_smoke_v3/bp_fe.handoff.def`,
SHA256 `d18f962aab7a7280ca4e8c0cb6566aa2c5ddeb5fc5d94494c7957eb2ff9cda58`.
This is a successful **CPU generation smoke**, not a Table 2 PPA result.

The GPU candidate adds `--gpu --install <tested-install>` and a separate
`run_dreamplace_table2_gpu.slurm`, preserving CPU defaults. Its build directory
comes from `DMP_GPU_BUILD`; a success marker, build provenance and matching
installation hashes are mandatory. It verifies the actual imported configure
path, compiled CUDA support, one scheduler-visible GPU and runtime CUDA
availability; it refuses a CPU fallback. Device/PyTorch/CUDA identities and
build metadata hashes are saved. No GPU candidate is accepted until its own
single-design smoke and the same geometry/connectivity/pin audits pass.
When the GPU build succeeds through a compatibility retry, the runner follows
`build_completion.json` to that retry, preserving the original failed-build
exit code and recording the successful retry's header patch, compatibility
description, configure/loaded-library files and four operator-test logs. It
requires the successful attempt's exit code to be zero and verifies both the
complete installed file inventory and hashes. Runtime GPU name/capability and
PyTorch/CUDA versions must match the tested build cohort.

## Validated H20 smoke and frozen generation cohort

The GPU build used job `1787994`'s isolated installation, compilation retry
`1787998`, and validation continuation `1788005`. All four upstream operator
checks (HPWL, weighted-average wirelength, electric potential and pin position)
passed their CPU/CUDA paths. Two CUB namespace qualifications and one test-only
`uint8`→`bool` mask fixture adaptation are retained as explicit patches; installed
placement algorithms and numerical expectations/tolerances were not changed.

The first full GPU startup, job `1788006`, exposed a separate missing `cffi`
import. Its failure is retained. The GPU wrapper now appends the same explicit,
read-only system package directories used by the successful CPU environment,
while keeping pinned packages first and `PYTHONNOUSERSITE=1`. It records the
actual imported cffi/cairocffi paths and hashes.

GPU smoke `1788008`, run `dmp_public41_gpu_bp_fe_smoke_v2`, passed all generation
audits on an actual `cuda:0` placement tensor: 11 macros, no overlap or die/core
violations, and 2,163 connected macro pins with zero raw-DBU transform error.
Its handoff SHA256 is
`ace3ac1988bcd2eb072d192135ee416e5a8edaf7e07682dbd17c5e32c85e13ff`.
The frozen six-design protocol is in
`backend/plans/table2_dreamplace_h20_v1`; task SHA256
`16fe0490d16c08ae22d6c16318c1f19da6b3d508b0176d9ebcf86a8f0d6470d1`.
Job `1788011` ran its remaining five designs serially on one H20, reusing the
predeclared bp_fe result. No seed/halo search or OpenROAD PPA is part of that
cohort. Its final handoff inventory retains all six designs, including explicit
failures if any.

## Abacus row-boundary failure and disclosed implementation correction

The original H20 cohort completed with five passing designs and one retained
failure, `ariane133`. Stock final macro legalization passed, stock greedy
standard-cell legalization placed all cells, and its whole-layout legality
check passed. The following Abacus call aborted at
`abacus_legalize_cpu.h:94`, while sorting rectangles assigned to a row.
This was not a failure of the macro rectangle audit.

Observation-only job `1788018` replayed exactly the original configuration and
saved the unmodified inputs immediately before the abort. It did not replace
an operator, suppress the assertion, change precision, or introduce RNG calls.
The raw float32 snapshot establishes the following values:

| Value | Standard cell 31897 | Macro 167689 |
|---|---:|---:|
| Lower y | 847.368408203125 | 854.73681640625 |
| Height | 7.36842155456543 | 213.6842041015625 |
| Upper y | 854.73681640625 | 1068.4210205078125 |
| Relevant row | Center maps to 115 | Lower edge should map to 116 |

The two rectangles touch exactly in y: the gap is zero. The macro's lower
edge divided by float32 row height `7.3684210777282715` is
`115.99999237060547`; the stock lower-bin expression truncates that to row 115.
It therefore inserts a macro into the preceding row and finds an apparent
horizontal containment with a cell that is actually below it.

The isolated correction replaces only
`std::max((y[node_id] - yl) / bin_size_y, (T)0)` with
`std::max(floorDiv(y[node_id] - yl, bin_size_y), (T)0)`.
This uses the library's existing tolerance-aware `floorDiv`, consistent with
its other row-index utilities; no new tolerance is selected. Abacus remains
enabled. Its final site-alignment loop can touch macro coordinates, so simply
skipping Abacus would not establish strict equivalence.

CPU build/validation job `1788043` created a separate installation at
`/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/abacus_floor_div_job_1788043`.
Only its Abacus shared object differs from the validated parent. All other
installation files remain byte-identical, and the parent's complete hash
inventory was verified unchanged. Original compiler/linker flags and the
unchanged utility archive were reused and recorded. The correction patch SHA256
is `3b8600c884155025dd50e06690a2530838db823e8d2feb635e9f5174a9d7bfc1`.

The two-node fixture extracted from the actual failure aborts with the original
operator. The corrected operator leaves its coordinates unchanged. The complete
retained Abacus input then replays successfully: stock whole-layout legality
passes before and after, and all 132 macros have zero coordinate displacement
during this replay. These facts are recorded in `operator_patch_validation.json`
and `patched_validation/validation.json`; the four unchanged parent numerical
operator tests are inherited evidence, not claimed as rerun tests.

Uniform corrected cohort `1788062` runs all six designs anew, with six new run
IDs, one H20, seed 1000 and unchanged original halos. It uses frozen plan
`backend/plans/table2_dreamplace_h20_abacusfix_v1`, task SHA256
`60422f1bb11a6c7db07b13670b8262da2eb805f3a385c01798e3441da411be29`.
It neither reuses the five unpatched successes nor selects a different seed.
The correction and its validation hashes are required in every run manifest.
This is a disclosed implementation correction, not an unmodified-public-binary
claim or proof of the unpublished author's exact baseline configuration.

The corrected cohort completed with exit code 0 in 6m19s; all six generation
audits passed. Macro counts were 11/24/132/136/10/28 in the frozen design order,
with zero overlap, die-boundary or core-boundary violations and zero physical
pin-offset error in DBU. The full six-entry handoff file is
`/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2/cohort_1788062/handoffs.jsonl`
(SHA256 `2ceded44a75b3e31d7fc5a9ed5af23a405e5f278214d229dc14525ceec8b9515`).
Its `cohort_manifest.json` SHA256 is
`1fcdc500a0f21059486cb19ae757bb86cc81f73db461ed0da1a332dabb4fb45c`.
These are generation results; downstream PPA is evaluated separately through
the same frozen OpenROAD flow as FlowPlace.
