# Predeclared global-placement switch diagnostic

Date: 2026-10-17. These paired runs investigate the discrepancy between the
paper's timing/congestion-driven global-placement description and the released
per-design `GPL_TIMING_DRIVEN=0`, `GPL_ROUTABILITY_DRIVEN=0` configurations.
They are diagnostics, not replacement primary rows or alternative seeds.

The two switches were changed together to 1. The released macro DEF, flow
source, current OpenROAD binary, synthesis seed, clock, density, 8 threads,
and detailed-route iteration cap 4 stayed unchanged. The bp_be confirmation
was submitted before bp_fe final metrics were available. Thus the effect
cannot be attributed separately to timing and congestion optimization.

| Design | GPL timing/routability | rWL (million µm) | WNS (ns) | TNS (ns) | Power (W, author parser) | Std-cell area (µm²) | Route DRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bp_fe | 0/0 | 2.169485 | -0.573610 | -359.0850 | 0.171 | 59268.3 | 1904 |
| bp_fe | 1/1 | 1.916897 | -0.304722 | -74.3821 | 0.167 | 60382.5 | 1200 |
| bp_be | 0/0 | 3.044539 | -1.203400 | -820.4050 | 0.159 | 102423 | 247 |
| bp_be | 1/1 | 2.797797 | -0.982794 | -259.5340 | 0.154 | 100423 | 6 |

Both switch diagnostics completed the full flow and passed the independent
raw-report audit. They improve wirelength and timing on these two layouts,
but do not explain the paper's larger area and power values. This is not
evidence that these are the authors' actual switches. Their PPA must remain
separate from the unmodified released recipe. DRC counts remain nonzero;
successful fullflow is not signoff cleanliness.

Evidence:

- Baselines: `backend/results/released_bp_fe_03` and
  `backend/results/released_bp_be_job1783836_3`.
- bp_fe diagnostic: job `1787852_0`,
  `backend/results/table2_paper_gpl_diag_v1_bp_fe`.
- bp_be diagnostic: job `1787985_0`,
  `backend/results/table2_paper_gpl_diag_v1_bp_be`.
- Frozen plans: `backend/plans/table2_paper_gpl_diagnostic_v1` and
  `backend/plans/table2_paper_gpl_bp_be_diagnostic_v1`.
- Each result preserves `metrics.json`, `fullflow_audit.json`, the raw final
  timing/power and detailed-route reports, and its executed make commands.

The next independent control changes only OpenROAD to a public binary near
the released flow's source era. Its selection rule was frozen before PPA;
see `table2_orfs_version_provenance.md`.

## Source-era tool × GPL interaction control

After selecting the source-era binary independently of PPA, one additional
fixed-layout bp_fe control was declared: old OpenROAD with GPL 1/1. This fills
the missing cell of a 2×2 tool-era × GPL comparison. It retains the same
released bp_fe DEF, source, libraries, SDC, eight threads, and route cap as
the three existing controls. No new generation seed or layout is introduced.
Changing both switches together still cannot isolate their individual effects.

Frozen task SHA256:
`c901180018cb40eff90174253cb5b4ad9ceb4a7b983317aa880cdda568c9fab9`.
Plan: `backend/plans/table2_source_era_paper_gpl_bp_fe_v1/tasks.jsonl`.
Physical job: `1788249_0`; run ID:
`table2_source_era_paper_gpl_bp_fe_v1`. At declaration this result was pending.
Its complete result and any failure must remain a diagnostic. The primary
12-row recipe stays at released GPL 0/0 regardless of numerical closeness.

The fourth cell completed in job 1788249_0 (19m42s, exit 0) and passed the
independent final raw-report audit. The full fixed-layout bp_fe comparison,
using final JSON precision uniformly, is:

| OpenROAD tool era | GPL timing/routability | rWL (million µm) | WNS (ns) | TNS (ns) | Power (W) | Std-cell area (µm²) | Route DRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New 26Q4 image | 0/0 | 2.169485 | -0.573610 | -359.0850 | 0.170899 | 59268.3 | 1904 |
| New 26Q4 image | 1/1 | 1.916897 | -0.304722 | -74.3821 | 0.166701 | 60382.5 | 1200 |
| Source-era v2.0-17598 | 0/0 | 2.256702 | -0.440867 | -100.9740 | 0.206998 | 71858.8 | 6352 |
| Source-era v2.0-17598 | 1/1 | 2.367180 | -0.301039 | -64.0233 | 0.208768 | 73774.3 | 3743 |
| Paper FlowPlace | unknown actual flags | 2.33 | -0.56 | -124.48 | 0.21 | 72407 | not reported |

This supports two bounded observations: tool era strongly affects area and
power on this layout; GPL1/1 improves slack in both tool versions, but its
wirelength effect changes sign. It does not identify the authors' actual
binary or flags, and no cell matches every paper metric. This one-layout,
single-run diagnostic is not a general claim about optimizer quality or a
reason to replace selected primary rows. All four runs retain nonzero DRC.
