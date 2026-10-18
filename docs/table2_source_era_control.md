# OpenROAD source-era binary control

On 2026-10-17 the same released bp_fe DEF was evaluated with two OpenROAD
binaries in the identical Ubuntu 22.04 container. The released flow, synthesis
seed, SDC, libraries, RC rules, GPL0/0 switches, route cap 4, and eight threads
were preserved. Source-era selection was based on matching the released scripts
to late-2024 ORFS, before the older binary's PPA was available. This does not
identify the authors' exact binary.

| Run | rWL (million µm) | WNS (ns) | TNS (ns) | Total power (W) | Std-cell area (µm²) | Route DRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New OpenROAD, released bp_fe | 2.169485 | -0.573610 | -359.085 | 0.170899 | 59268.3 | 1904 |
| OpenROAD v2.0-17598-ga008522d8, identical DEF | 2.256702 | -0.440867 | -100.974 | 0.206998 | 71858.8 | 6352 |
| Paper FlowPlace bp_fe | 2.33 | -0.56 | -124.48 | 0.21 | 72407 | not reported |

Values above use final JSON, without fitting or scaling to the paper. The
source-era run narrows the area/power discrepancy substantially. Differences
remain in wirelength, timing, and route DRC. The older binary is therefore a
source-supported common backend candidate, not proof of exact reproduction.

The old-tool physical flow completed both make stages with return code zero.
Its first independent audit marked it failed because the author parser replaced
full-precision JSON WNS/TNS with two-decimal text values. Original metrics,
manifest, failed audit, and Slurm FAILED state remain intact. Independent
postprocessing job1788017 verified every changed value against its precise
text token and raw report, plus original input/source/tool identities:

`/fact_data/bohanli/flowplace_fql_backend/table2_reports/reconciliation_1787997_v1`

This reconciliation is separate from the formal generated-layout cohort. Future
formal tasks retain `author_metrics.json` unchanged and use uniformly extracted
final JSON in `metrics.json`, with text-rounding evidence checked again by the
aggregate verifier.

Formal FlowPlace six-design plan was frozen before this control's final PPA:
`backend/plans/table2_source_era_flowplace_v1`, SHA256
`d9a04b4fa367940aab26679dee053821f618694d23cdaacdf31ab010cc1e9b9b`.
Physical array1788030 uses two concurrent eight-core tasks. Archive array1788033
uses aftercorr dependencies to losslessly preserve each completed run's bulky
intermediates on fact_data while retaining logs, reports, metrics and provenance
as ordinary persistent files. Both methods will use the same backend protocol.
