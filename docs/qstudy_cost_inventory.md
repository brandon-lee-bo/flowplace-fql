# Q-study physical-call inventory

This is a **non-final snapshot**, observed during active work at
2026-10-18 11:36:51–11:36:54 UTC. CPU Slurm job **1797308** completed
successfully in 3 seconds, including five bookkeeping tests. The inventory
read saved request/outcome bindings and native manifest/metric metadata;
it did not run training, inference, EDA, archive decoding, or a new full
geometry audit. A submitted job ID is never counted as physical success.

## Planned current-phase GRT requests

| Branch | Logical requests | Reused requests | New request budget |
| --- | ---: | ---: | ---: |
| Main matched BC/FQL, 2 arms × 3 seeds | 240 | 0 | 240 |
| Updated-flow FQL, 3 seeds | 120 | 48 inherited main round-0 results | 72 |
| Original/current-flow label-source actors | 48 | 0 | 48 |
| Bounded-objective actors | 72 | 0 | 72 |
| Current behavior-flow baseline | 24 | 0 | 24 |
| Released FlowPlace baseline | 8 | 0 | 8 |
| Current-flow source labels | 80 | 0 | 80 |
| Local direction probes, bp_fe/bp_be | 52 + 52 | 0 | 104 |
| **Total** | **696** | **48** | **648** |

The frozen actor exporter uses eight collection and eight evaluation
requests in each non-final round, then **eight evaluation-only requests
in the final round**. Thus main is `2 × 3 × (16 + 16 + 8) = 240` and
updated-flow is `3 × (16 + 16 + 8) = 120`. The earlier 288/144 estimates
incorrectly assigned collection requests to the final round. Configurations
with explicit `online_rounds: 0` have eight evaluation-only requests per seed.
This correction changes bookkeeping only; actual scientific/export budgets
were not changed.

## Observed physical work at this snapshot

| Quantity | Count | Meaning |
| --- | ---: | --- |
| Unique observed current-phase GRT physical directories/manifests | 403 | Deduplicated native manifest paths and physical directory IDs |
| Saved successful terminal native manifests | 335 | Completed physical runs supported by saved bindings/metadata |
| Observed nonterminal manifests | 68 | Interrupted historical attempts and currently active attempts; **not 68 physical failures** |
| Prior unique completed native runs | 640 | Historical data acquisition, separate from current-phase calls |

Logical requests, physical attempts, and terminal successes are distinct.
A retry can create another observed physical directory without adding a
logical request. Conversely, the updated-flow branch reuses 48 main results
without spending 48 additional calls. Missing outcomes, unknown attempts,
and unfinished manifests do not receive invented PPA values or failure rewards.
The 335 successes are a count at observation time, not the final cost or a
claim that the remaining planned requests have run.

## Prior data and permitted uses

The **640 prior unique completed native runs** comprise:

- **480 original labels:** bp_fe384 has 384 TRAIN + 16 historical-validation
  rows; bp_be64 has 64 TRAIN + 16 historical-validation rows. Together these
  are 448 TRAIN and 32 historical-validation labels.
- **160 historical online-feedback labels**, kept separate from initial TRAIN.

The bp_fe64 view (64 TRAIN + 16 historical validation) is a subset of the
bp_fe384 pool. Its 80 rows must **not** be added again as 80 unique calls.
Likewise, the 256 original behavior targets describe a training pool, not
256 newly measured backend calls in this phase.

Current-flow source acquisition contributes 64 TRAIN + 16 validation labels
and no online-replay eligibility. The two 52-request direction sets are
diagnostic only. Actor development-evaluation results are not initial TRAIN
labels. No fresh-confirmation evaluation is included in this inventory.

## Separate development final-route budget

The declared DRT plan contains **48 requests**, separate from all GRT totals:
24 round-0 requests completed; 24 round-2 requests had not started at this
snapshot. DRT reports and metadata-only aggregation do not add physical calls.
GRT PPA/congestion labels are not final-route DRC or signoff results.

## Reproducible evidence

- Immutable report:
  `/fact_data/bohanli/flowplace_fql/qstudy_v1/call_inventory_v1/20261018T113654Z_job1797308/report.json`
- Report SHA256:
  `3ba2db5fc899705f840603088506ff047f6b52de395c91061030082a1f50979b`
- Metadata helper:
  `/fact_data/bohanli/flowplace_fql/qstudy_v1/call_inventory_v1/inventory_v2.py`
- Helper SHA256:
  `daba0311ede970671aace98d90469d1582722d864618a5f61c4b86e3e15dccfd`

The immutable report retains individual request/outcome/config/task/native
fingerprints and the exact source copy. Reads span a short active-work
interval, rather than a globally atomic filesystem snapshot. Later results
must be reported as a new dated snapshot. The earlier job1797301 snapshot
with wrong final-round denominators is retained as superseded bookkeeping
evidence, not silently overwritten.
