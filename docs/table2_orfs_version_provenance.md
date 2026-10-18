# Table 2: released flow source and historical OpenROAD candidate

Audited 2026-10-17. This records source evidence and a predeclared compatibility
candidate. It does **not** identify the executable used to produce the paper's
numbers. No version is selected by its PPA results.

## Exact source matches

The released `backend/thirdparty/OpenROAD-flow-scripts` archive contains 746
files with the same 2026-05-07 archive timestamp; that timestamp cannot establish
the underlying OpenROAD/ORFS version. Three selected scripts were compared
byte-for-byte with official Git history:

| Released path | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/load.tcl` | 3928 | `7c7b2d1999558f1e5188a3b330f48be4d2516716172857a84438074bd189b79b` |
| `scripts/util.tcl` | 5228 | `285367675fe7f2ae936055a6c076c4b57874c641ffa560f0430257229cda97f9` |
| `scripts/variables.yaml` | 17397 | `3337f37e43cad9d74fa209d428c516bf528c56fdf664d2767b09abf11b2b1fc3` |

All three match **both** official ORFS commits below:

- [8ea7b363a790a0ba91a0fc63b9917427ea04c95d](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/commit/8ea7b363a790a0ba91a0fc63b9917427ea04c95d),
  committed 2024-10-30 23:35:30 UTC, a feature-branch merge.
- [18371323ef27794effe878dc6872d637b0f83bc5](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/commit/18371323ef27794effe878dc6872d637b0f83bc5),
  committed 2024-10-31 22:39:13 UTC, the historical master head returned by
  `commits?sha=master&until=2024-10-31T23:59:59Z&per_page=1`.

The second commit's exact files are
[load.tcl](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/18371323ef27794effe878dc6872d637b0f83bc5/flow/scripts/load.tcl),
[util.tcl](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/18371323ef27794effe878dc6872d637b0f83bc5/flow/scripts/util.tcl), and
[variables.yaml](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/18371323ef27794effe878dc6872d637b0f83bc5/flow/scripts/variables.yaml).

At both snapshots, the `tools/OpenROAD` gitlink is
[`cd519bb5e861ec4d5d00931c85650c179a864e34`](https://github.com/The-OpenROAD-Project/OpenROAD/commit/cd519bb5e861ec4d5d00931c85650c179a864e34),
committed 2024-10-27 17:04:59 UTC. The Yosys gitlink is
`e97731b9dda91fa5fa53ed87df7c34163ba59a41`; Yosys is not being changed for
the proposed existing-netlist backend diagnostic.

This supports a late-October 2024 **source lineage**. It does not establish
that the other 743 files are identical, that the author checked out either
whole commit, or that the author built the corresponding OpenROAD gitlink.
Some released files have local modifications. The archive does not identify
the executable used for the paper's Table 2 evaluation. It does contain older
`metadata-base-ok.json` reference runs with several different executable
versions; those must not be mistaken for a Table 2 executable identity.

### Embedded historical metadata, inspected 2026-10-17 20:15 HKT

The existing six `metadata-base-ok.json` files record `base` reference runs,
not the `eval` runs requested by the released evaluator:

| Design | Recorded reference date | Recorded OpenROAD version |
| --- | --- | --- |
| ariane133 | 2023-04-01 | v2.0-7484-ga8187d5b7 |
| ariane136 | 2024-10-24 | v2.0-16728-ga30d14f65 |
| bp | 2024-08-24 | v2.0-15311-g42ba30d73 |
| bp_be | 2024-10-15 | v2.0-16535-g199588e84 |
| bp_fe | 2024-10-02 | v2.0-16219-g56c3c86ba |
| swerv_wrapper | 2024-09-27 | v2.0-16089-gd06039e7d |

All six released synthesis Verilog headers instead identify Yosys
`0.45+145`, commit `59404f8ce`. A synthesis-tool identity does not establish
the downstream placement/routing executable. These mixed historical records
support investigating historical source provenance, but do not justify using
a different binary per circuit, or treating a version yielding closer PPA as
the authors' version. Paths, small metadata hashes, and headers are recorded
in `backend/references/released_flow_embedded_versions_v1.json`.

An independent source-build probe (1788539), followed by a scheduled build
(1788546), targets the already identified ORFS gitlink `cd519bb5...`.
This choice precedes any PPA run using that binary. The original twelve-row
cohort keeps its frozen December 2024 executable; any new-binary experiment
must be a separately declared fixed-layout diagnostic.

The first two compile attempts against modern container dependencies exposed
spdlog/fmt and Boost.Asio incompatibilities. Their failed records are retained.
The historical dependency installer specifies spdlog 1.8.1, Boost 1.80.0 and
OR-Tools 9.10.4067. These are now configured in isolated scratch prefixes;
the Boost archive passes the installer's MD5 and both downloads have recorded
SHA256 values. Build 1788558 uses the same pinned algorithm source without
patching those incompatibilities into OpenROAD. After correcting only the
installation RPATH cache to prefer that SDK, build 1788558 attempt_1 completed
with a working published executable. Its SHA256 is
`eadf0dd0baea50e510e77bf207e29ed1168ca473ddc1c6dd248a5a77573272cb`;
the reported version is the complete pinned Git revision. Import smoke
1788561 also completed: the released bp_fe DEF and platform LEFs produce
33,199 instances, including 11 macros, with 2,000 DBU per micron.
This establishes startup and input compatibility, not a PPA result or the
authors' exact executable identity. Full setup records, retained failures,
runtime-SDK restoration plan and fixed-layout controls are described in
`pending_pipeline.md`.

Small metadata and source snapshots are retained in
`/scratch/bohanli/flowplace_legalizer_audit/orfs_history/`: `load_commits.json`,
`util.tcl_candidate_commits.json`, `variables.yaml_candidate_commits.json`,
`util_variables_history_matches.json`, the two commit-prefixed copies of all
three scripts and `*_tools.json`, `oct31_mainline_head.json`, and
`openroad_gitlink_commit.json`.

## Available prebuilt releases and predeclared candidate

The official ORFS [prebuilt instructions](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/master/docs/user/BuildWithPrebuilt.md)
refer users to [Precision Innovations OpenROAD releases](https://github.com/Precision-Innovations/OpenROAD/releases).
The release-list API returned these adjacent surviving 2024 releases around
the matched source date:

| Release | Ubuntu 22.04 asset | Bytes | Relation to matched source |
| --- | --- | ---: | --- |
| [2024-10-07](https://github.com/Precision-Innovations/OpenROAD/releases/tag/2024-10-07) | `openroad_2.0_amd64-ubuntu22.04-2024-10-07.deb` | 55,885,384 | Earlier; may lack interfaces used by the later flow |
| [2024-12-14](https://github.com/Precision-Innovations/OpenROAD/releases/tag/2024-12-14) | `openroad_2.0-17598-ga008522d8_amd64-ubuntu-22.04.deb` | 56,971,996 | Closest surviving later prebuilt release found |

No release between those dates is present in the retrieved release list.
An exact `cd519bb5...` prebuilt artifact was not found. Queries of the official
`openroad/orfs` Docker tags for `24Q4`, `24Q3`, and `v2.0` returned zero matches;
this is a bounded search result, not proof that no historical image exists.

**Candidate chosen before any PPA experiment: the 2024-12-14 Ubuntu 22.04
package.** It follows the matching source period and avoids starting with an
older executable that could lack required commands. This is a compatibility
diagnostic, not an assertion of an exact paper reproduction.

Frozen metadata: release ID `190767528`, asset ID `213317854`, release name
`2.0-17598-ga008522d8`, published 2024-12-14 12:47:55 UTC. GitHub's asset
`digest` is null, so no publisher SHA256 is available from this metadata. The
downloaded bytes and extracted executable must each receive a locally
computed SHA256, and the actual executable must report its version.
The package's abbreviated source revision resolves in official OpenROAD to
[`a008522d88b669ac4c985609533cf5a3d2649222`](https://github.com/The-OpenROAD-Project/OpenROAD/commit/a008522d88b669ac4c985609533cf5a3d2649222),
committed 2024-12-11 15:56:17 UTC. Its metadata is preserved as
`openroad_release_a008522d8_commit.json` beside the other source evidence.

Download URL:
<https://github.com/Precision-Innovations/OpenROAD/releases/download/2024-12-14/openroad_2.0-17598-ga008522d8_amd64-ubuntu-22.04.deb>

The API metadata snapshots are `precision_releases_page1.json` and
`prebuilt_candidate1.json` in the scratch evidence directory above. A
subsequent API page had a partial-read error; page 1 already contains all
listed 2024 releases and extends through 2023-10-05.

## Isolated compatibility probe

`backend/scripts/probe_openroad_20241214.slurm` requests cpu-scavenger,
1 CPU, 2 GiB, 30 minutes, no requeue. It downloads only this selected package,
checks its frozen byte count, computes package/executable SHA256, uses
`dpkg-deb --extract` into a job-specific `/fact_data` directory without
installation, and runs `ldd` and `-version` inside the existing Ubuntu 22.04
ORFS SIF. Temporary files use `/scratch`. It does not modify shared flow,
install dependencies, run a design, or evaluate PPA.

Each probe writes `manifest.json`, its script snapshot and individual logs to
`/fact_data/bohanli/flowplace_fql_backend/tools/openroad_20241214/probe_JOBID/`.
The manifest reports missing dependencies or a version mismatch explicitly.
The existing SIF's previously recorded SHA is labeled as such; this small
probe does not recompute the large SIF hash. Fullflow execution and a fresh
tool identity freeze remain the parent task's separate decision.

Probe job **1787995** completed download, package extraction, `ldd` and
`-version` in 2m12s. Both executable probes returned zero; no missing shared
library was reported. Actual version: **`v2.0-17598-ga008522d8`**. The original
probe compared this against a string missing its leading `v`, so its original
Slurm state remains **FAILED, exit 1:0** and its manifest remains unchanged.
`version_prefix_reconciliation.json` independently binds the original manifest
and command-log hashes, records this narrow reporting correction, and accepts
the observed full version. The future probe script now expects the correct
full string; the exact submitted script is preserved in the original output.
No re-download or design run was performed for this correction.

- Package SHA256: `40ed178396b0276a5d5dfbbe695c9de9aac9088157a6655be02b39a0cef07207`.
- Extracted executable: `/fact_data/bohanli/flowplace_fql_backend/tools/openroad_20241214/probe_1787995/extracted/usr/bin/openroad`.
- Executable size: 83,704,232 bytes; SHA256: `93fe61454a52b927602fd6c798c4402214c19e80e7d614636345f0f427ea2f46`.

These checks establish that the executable loads and reports the expected
identity inside the existing SIF. Flow-script/API compatibility and PPA still
require the separately frozen single-design diagnostic.

## Metric compatibility checks for the fixed-input diagnostic

The parent submitted `table2_legacy_tool_v1_bp_fe`, Slurm array `1787997`,
using the same released `bp_fe.def`, existing flow patches, GPL timing and
routability switches both zero, and eight threads. Its startup manifest
independently confirms the exact executable SHA and version above. Both old
and new floorplan JSON record `1ns` time and `1um` distance units. The same
released `final_report.tcl` performs extraction/read-SPEF and final metrics
after detailed routing, so the requested evaluation stage is unchanged.

One concrete parser difference must be checked when the old final reports
arrive. OpenROAD `a008522d8` pins OpenSTA
`aa598a2f14c5c142e90391a69988523505e7db3d`; its
[Search.tcl](https://github.com/The-OpenROAD-Project/OpenSTA/blob/aa598a2f14c5c142e90391a69988523505e7db3d/search/Search.tcl#L691)
prints `tns NUMBER` and `wns NUMBER`. The new-tool retained `6_finish.rpt`
prints `tns max NUMBER` and `wns max NUMBER`. The released author Python
parser initially reads precise JSON, then overwrites WNS/TNS if the older
text patterns match. Thus text display rounding can cause the existing
strict JSON audit to fail despite successful physical execution. This is a
source-confirmed compatibility risk, **not yet a claim about this run's final
results**. The running runner, parser, and auditors are kept unchanged.

If the final reports confirm this exact issue, retain the original manifest,
author `metrics.json`, and failed audit. A separate postprocessing record
must bind the raw final reports, verify successful physical-stage returns
and unchanged inputs, and record raw-JSON canonical values alongside author
text values and their exact text tokens. Do not rerun placement/routing for
a formatting correction or loosen all metric tolerances. Any future cohort
should retain both author and canonical values consistently across methods.

The old OpenSTA [Power.tcl](https://github.com/The-OpenROAD-Project/OpenSTA/blob/aa598a2f14c5c142e90391a69988523505e7db3d/power/Power.tcl#L68)
explicitly labels the total-power table in Watts. The Liberty/platform
metadata `power_units: 1nW` therefore does not imply an extra `1e-9` multiplier
on the reported total-power value. Both text and JSON must still be checked
against the actual final report.
