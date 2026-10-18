# FlowPlace ICCAD reproduction (started 2026-10-17)

This is separate from the earlier OpenROAD/Table 2 pilot. User requires the
paper's tool chain: DREAMPlace 4.1.0 standard-cell placement, then Cadence
Innovus EarlyGlobalRoute, with routed wirelength, H/V overflow, WNS and TNS.
OpenROAD results are not substitutes for these measurements.

## Scope and fidelity

- Paper: https://arxiv.org/pdf/2604.23658v1, Section 4, Table 1 and Table 3.
- Table 1 is the eight-design ICCAD comparison. Table 2 is the six-design
  OpenROAD comparison. Table 3 compares two FlowPlace training datasets on
  ICCAD, reporting mean per-design ranks, **not absolute PPA**.
- Upstream: https://github.com/lamda-bbo/flowplace, pinned source already in
  `../upstream/flowplace`. The README says the ICCAD commercial-tool
  evaluation scripts cannot be shared. Exact reproduction of undocumented
  routing, RC, clock, timing-view and reporting settings is not established.
- First validate the backend with the author's released DEFs; then evaluate
  fresh model-generated macro placements. Released-DEF replay alone does not
  reproduce training or inference.
- Reconstructing Table 3 additionally requires both random-synthetic and
  mask-guided training runs under matched model/training/sampling budgets.
  The public release lists only the mask-guided checkpoint and dataset.
  Do not report Table 3 reproduced from a single released checkpoint.
- The first DREAMPlace build is the **CPU implementation of tag 4.1.0** in
  Slurm CPU allocations. This is an explicit implementation difference; no paper GPU
  runtime comparison is allowed. Build and configuration logs remain intact.
- Initial DREAMPlace parameters use its pinned ICCAD example, with input/output
  paths and CPU/thread settings changed. These parameters are a documented
  reconstruction, not claimed to be FlowPlace's unpublished evaluation config.
  No tuning to match published numbers.

## Sources

- DREAMPlace tag 4.1.0 resolves to `5d13c9001a3bc900dca1e108e633d5dd45b00701`.
  The GitHub release API currently returns 404 for this tag, but the git tag
  exists. Submodules must match their pinned commits before compilation.
- ICCAD input archive: the Google Drive link in the FlowPlace README,
  file ID `1xeauwLR9lOxnYvsK2JGPSY0INQh8VuE4`, name `iccad2015.ot.tar.gz`.
- Technology LEF: FlowPlace release v1.0.0 `tech.lef`.
- Macro DEFs: existing original v1.0.0 release in
  `../backend/releases/flowplace_DEFS/iccad2015/`.
- Source downloads, hashes, extracted inputs, build artifacts and raw logs
  live under `assets/` and `logs/`; they are not committed.

## Tool availability

Slurm `eda` runs on enterprise. Installed executable:
`/fact_data/softwares/cadence/INNOVUS251/bin/innovus`.
Version query: **Innovus v25.11-s102_1**, built 2025-08-27.

Two real scheduled startup probes failed before the Tcl script ran:

| Job | Result |
| --- | --- |
| 1787652 | Read-only enterprise/tool-path probe completed |
| 1787653 | No license environment; LMC-01902, license search path absent |
| 1787655 | Shared Cadence license.dat supplied; IMPLIC-4, no valid product license |
| 1787656 | Input extraction and SHA256 manifest job |

The shared installation's pre-existing Oct 12 Innovus logs show the same
IMPLIC-4 result, including with the shared license file and campus Synopsys
server in the search path. Do not modify shared installations or license files.
User has been asked for the group's supported Innovus environment/license setup.

After locating the lab PDF, scheduled retries1787702/1787703 checked normal
Bash/Zsh login environments, shared-file normal/exclusive search, and the full
`setup_eda.env` initialization. All startup attempts still fail before Tcl.
The admin script's configured server is reachable but reports a Synopsys
`snpslmd` daemon; the combined Innovus search still reports IMPLIC-4. See
`status_2026-10-17.md` for the exact scope and retained logs. This does not prove
that no other supported Cadence configuration exists in the lab.

**Subsequent alternative verified:** through the user's VPN/SSH forwarding,
SCUT host SMW8 successfully starts `/SM01/eda/cadence/INNOVUS191/bin/innovus`,
version19.12-s087_1. The normal `invs`19.1 license checkout succeeds; a2-thread
no-design Tcl probe executes all the way to completion (0tool errors/warnings).
Required EGR, timing, congestion, import and unit commands exist. No real design
has yet been imported/routed there. The user's SCUT scheduling/resource policy
must be established before full-circuit evaluation; FactLab Slurm rules are not
automatically a statement of SCUT policy. Full evidence/scope is in the status
file. This supplies a working base license but does not erase version/config
differences or establish routed PPA.

## Measurement contract

1. Keep released and newly generated layouts distinct. Preserve macro names,
   coordinates, sizes, orientations and FIXED status through DREAMPlace.
2. Validate placement bounds, overlaps and component identity before and after
   standard-cell placement; keep failures rather than silently legalizing the
   model output with an unrelated macro placer.
3. Innovus must consume the resulting standard-cell placement with the matching
   ICCAD LEF, Verilog, early/late Liberty and SDC. The author's routing tech LEF
   is an additional input, not a replacement for macro/cell geometry.
4. Use **EarlyGlobalRoute**, not OpenROAD GRT or Innovus detailed routing, for
   this comparison. Retain raw wirelength and H/V overflow reports, timing
   views, units, RC setup, clock treatment, and tool version.
5. Table 1 units: rWL = million microns; WNS = ns; TNS = 100,000 ns; rO-H/rO-V
   = percentages. Keep raw units in machine-readable results and convert only
   in comparison tables. Never substitute macro HPWL for rWL.
6. A missing timing result is missing, never zero. Absence of a valid Innovus
   run means no ICCAD PPA result. EarlyGlobalRoute does not establish DRC-clean
   detailed routing or signoff.
7. Hash source/config/input/output files and tie every measurement to the
   Slurm job and exact DEF. Keep an explicit difference list from the paper.

All builds, inference and EDA evaluation must use Slurm. Do not use other
projects' allocations or cancel their jobs.

## Audited released-DEF geometry

Job **1787667** completed a check of all eight actual inputs, with LEF sizes,
DEF units and orientation transforms. In this task the model places the 512
largest instances; these are FIXED in released DEFs, and can include CORE-class
cells. Other BLOCK-class cells may remain movable. The audit therefore checks
the actual FIXED set, not all LEF BLOCK instances.

| Design | Fixed instances | Positive rectangle-overlap pairs | Outside DIEAREA |
| --- | ---: | ---: | ---: |
| superblue1 | 512 | 3 | 0 |
| superblue3 | 512 | 15 | 0 |
| superblue4 | 512 | 9 | 0 |
| superblue5 | 512 | 0 | 0 |
| superblue7 | 512 | 0 | 0 |
| superblue10 | 512 | 0 | 1 |
| superblue16 | 512 | 5 | 0 |
| superblue18 | 512 | 0 | 0 |

The eight component-name/master multisets match the benchmark inputs; DEF
component ordering differs. Actual evidence and hashes are in
`assets/geometry_1787667/*.json`. Example: a superblue1 overlap measures
0.22 by 32.49 microns; superblue10's fixed block ends at x=8263.58 microns
while its DEF die ends at 8263.48. These are input-file geometric findings,
not Innovus observations or proof about every output the method can generate.
No coordinate has been repaired. The three passing released designs are
eligible for backend smoke runs; superblue18 is the first, chosen before PPA.

Initial job1787658 checked all LEF BLOCK instances, which is a different scope;
its source/results remain but do not define the table above. Job1787660 exposed
decimal DBU coordinates in the original DEF; the parser was corrected to accept
them without rounding. The corrected complete audit is1787667.

## New official-model generation

Job **1787679** generated superblue18 from the original input netlist using the
released checkpoint and released 20-step hard-guided uniform sampler (seed300).
All512 macros pass the independent normalized-coordinate check and the
post-export DEF rectangle/boundary check. SHA256 and the saved noise, trajectory,
graph, macro names and input hashes are in
`assets/generated_superblue18_job1787679/`.

Sampling plus first-use setup took9.123 seconds on the allocated GPU; this is
not steady-state latency or end-to-end PPA runtime. Wrapper compilation was
disabled, original tensor operations and sampler retained. PYTHONHASHSEED=0
stabilizes graph-order tie-breaking. This is a single-design sample, not the
paper's undisclosed random stream or the Table3 training-data ablation.

## Build attempts

1787657 was moved while pending from eda to cpu-scavenger with matching QOS;
before/after Slurm records are retained. It failed on missing Boost headers on
quantum.1787659 found missing Boost graph/regex libraries.1787666 then exposed
missing Python development headers. Dependencies were staged under this
experiment's assets, never installed system-wide. Python3-devel comes from the
Rocky9 repository RPM; Boost headers/libraries and Python runtime/config header
come from the existing local system.1787680 retries with explicit dependency
paths and the same original DREAMPlace source. Failures and build logs remain.

`scripts/innovus_egr.tcl` is an **unexecuted reconstruction draft**, using
early/late libraries, the supplied SDC in ps/fF, ideal-clock preCTS timing and
70 C (the nominal temperature in the supplied libraries). Its RC corner uses
technology-LEF defaults. These choices are visible assumptions, not recovered
author settings. The exact RC extraction setup, reporting aggregation and
compatibility with Innovus25.1 must be verified after license startup succeeds.
No results or successful routing claims are based on this draft.

Build1787680 reached the real placement/timing extension libraries but failed
in the old OpenTimer doctest harness: modern glibc's SIGSTKSZ is not a constant
array bound. The recorded patch `patches/001-doctest-glibc-signals.patch` disables
doctest's POSIX crash-signal handler **only on its two test executables**. It
does not modify the DREAMPlace or OpenTimer numerical algorithms. Build
**1787684 completed** with that test-only compatibility patch. Dependent
run1787681 was canceled automatically after the failed build and did no work;
1787685 is the actual initial released-superblue18 CPU run.

The first CPU run snapshots its driver, audit script, input hashes, source
commit and config. An additional installed-file hash manifest was added after
that process started and will apply to later runs; do not claim it was captured
at startup for1787685. The original build provenance and test-only patch are
retained separately.

## Standard-cell placement and export verification

Both superblue18 inputs finished the DREAMPlace 4.1.0 CPU placement executable:

| Input | Placement job | Placement-process time | Output directory |
| --- | --- | ---: | --- |
| Author's released DEF | 1787685 | 417.94 s | `assets/runs/released_superblue18_cpu_job1787685` |
| Fresh official-model sample, seed300 | 1787687 | 320.64 s | `assets/runs/generated_superblue18_seed300_cpu_v1` |

These are backend wall times, not paper GPU latency or full PPA runtime. The
released run reported an overflow-plateau divergence and used DREAMPlace's
built-in handling before legalization; its original warning is retained.
Neither run has been evaluated by Innovus.

Two independent verification issues were diagnosed without modifying either DEF:

1. Released-output reload check1787686 failed using float32: scaling DBU by
   1/380 introduced site-alignment roundoff. Check1787694 uses float64 for
   **reading/checking only**, with the identical DEF and original checker
   tolerance. It passes; maximum row and site residuals are both zero.
   Placement itself was still the original float32 run. The false result and
   full diagnostic log remain alongside the float64 result.
2. The fresh run's placement executable exited0, but our initial exact-dictionary
   comparison rejected decimal serialization noise in the author's generated
   DEF, e.g.7471999.999999999 versus7472000 DBU. All512 fixed-instance identities,
   statuses and orientations match; maximum coordinate difference is
   1.862645149230957e-9 DBU. Post-check1787696 verifies the unchanged files with
   absolute tolerance1e-6 DBU and zero relative tolerance. This tolerance is
   one millionth of a DBU; a regression check rejects a one-DBU displacement.
   The failed original manifest is preserved, and the successful independent
   check is `output_verification.json`. No placement was rerun or repaired.

Dependent check1787688 was canceled after the original fresh-run verifier
failure. Check1787695 stopped at that same incomplete-manifest guard, before
loading the layout. Check1787697 follows the successful independent export
verification and checks the fresh layout with float64. It **passed**, with
zero maximum row/site residuals,767556 movable instances and512 fixed instances.
Both final DEFs are now ready for a licensed Innovus backend smoke test.

An all-cell pass here means DREAMPlace's movable-cell boundary, row/site,
overlap and fence-region checks, supplemented by the separate strict fixed-macro
geometry audit. It does not establish routed DRC cleanliness or signoff timing.

### Commands for the prepared pipeline

From this experiment root (`experiments/flowplace_fql`):

```bash
sbatch iccad/scripts/probe_innovus.slurm
sbatch iccad/scripts/generate_iccad.slurm
sbatch iccad/scripts/run_dreamplace.slurm --run-id YOUR_NEW_RUN_ID
sbatch iccad/scripts/check_dmp_output.slurm iccad/assets/runs/YOUR_NEW_RUN_ID --dtype float64
```

The legality check must run after the placement finishes (or use an `afterok`
Slurm dependency). Run IDs must be new; existing artifacts are not overwritten.
Generation and placement are separate: pass `--def-path` to place a fresh DEF;
otherwise the wrapper uses the released DEF. Inspect the scripts for resource
requests. Do not submit the commercial evaluation draft until supported license
startup works and the reconstructed RC/timing settings have been validated.
