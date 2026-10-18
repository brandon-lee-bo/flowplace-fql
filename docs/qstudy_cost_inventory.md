# Q-study physical-call inventory

This is the **completed development-call inventory**, captured by CPU Slurm
job **1797369** at 2026-10-18 15:16:34 UTC, with five bookkeeping tests.
Its saved-metadata reduction performs no training, inference or EDA. Fresh
confirmation, evidence preservation and their later bookkeeping are separate
from this development snapshot. A submitted job ID is never physical success.

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

## Completed development calls and observed attempts

| Quantity | Count | Meaning |
| --- | ---: | --- |
| New logical GRT requests and successful outcomes | 648/648 | Exact request/outcome bindings; no missing or invalid logical results |
| Unique observed current-phase GRT physical directories/manifests | 720 | Deduplicated physical attempts, including retries |
| Saved successful terminal native manifests | 648 | Completed physical runs supported by saved bindings/metadata |
| Observed nonterminal manifests | 66 | Interrupted attempts; **not 66 physical failures** |
| Unsuccessful terminal infrastructure attempts | 6 | Retained separately from successful retried logical outcomes |
| Detailed-route logical requests and successes | 48/48 | Separate physical stage; all48 final DRC0 |
| Prior unique completed native runs | 640 | Historical data acquisition, separate from current-phase calls |

Logical requests, physical attempts, and terminal successes are distinct.
A retry can create another observed physical directory without adding a
logical request. Conversely, the updated-flow branch reuses 48 main results
without spending 48 additional calls. Missing outcomes, unknown attempts,
and unfinished manifests do not receive invented PPA values or failure rewards.
The72 extra GRT attempts are not72 failed layouts. Reused logical results,
GRT attempts, detailed routing and historical calls are not added as though
they represented the same denominator.

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

The declared DRT plan contains **48 completed requests**, separate from all
GRT totals: BC/FQL, rounds0/2, three seeds and fixed first4 noises. Every run
used8 threads and repair cap20; all48 are DRC0. The cap does not mean every
run used20 repairs. DRT reports and metadata aggregation add no physical calls.
GRT PPA/congestion labels are not final-route DRC or signoff results.

## Training and allocation costs

The explicit development job manifest requests945 IDs. Its accounting
snapshot contains1056 allocation attempts:856 completed,83 preempted and117
failed, reserving **369.720833 CPU-hours and3.376111 GPU-hours**. These are
allocated resources, not utilization, energy, isolated inference latency or
an exhaustive accounting of all historical Goal activity. Distinct requeue
attempts are summed once each; batch/extern steps and overlapping array
parent/child query rows are not counted again. Parallel elapsed times cannot
be summed into experiment wall-clock time.

The training inventory has42 actor-stream records:39 newly completed streams
and3 inherited round0 streams. New streams commit31,800 configured actor
updates, excluding common imported warmup. Online work commits7,200 critic
and3,600 flow updates. The420 completed critic CV/refit leaves commit840,000
steps; they are not420 independent method comparisons. Updates lost to a
preemption/rollback cannot be reconstructed from checkpoint counters, while
their allocations remain in the resource totals.49 export markers do not
provide a complete measured forward-call or latency count.

An independent saved-accounting audit finds945/945 requested IDs covered.
The original collector displayed764 IDs as unobserved because numeric array
child IDs differ from formatted `parent_index` IDs. Checking both aliases
resolves all764; **allocation deduplication and CPU/GPU totals were already
correct**. The narrow coverage-display fix passes the end-to-end Slurm test
1798261, including two distinct requeue attempts and duplicate query rows.
The old cost snapshot remains unchanged, with the audit as an addendum.

## Reproducible evidence

- Immutable report:
  `/fact_data/bohanli/flowplace_fql/qstudy_v1/call_inventory_v1/20261018T151634Z_job1797369/report.json`
- Report SHA256:
  `50464ffa22d4e6baa5f933057f0a2b70ad0c3f7d825f2cc56f8917b3e6681416`
- Metadata helper:
  `/fact_data/bohanli/flowplace_fql/qstudy_v1/call_inventory_v1/inventory_v2.py`
- Helper SHA256:
  `daba0311ede970671aace98d90469d1582722d864618a5f61c4b86e3e15dccfd`

Additional immutable evidence:

- Development training-cost metadata:
  `/fact_home/bohanli/flowplace_fql_recovery/qstudy_cost_snapshots_v1/20261018T151723Z_job1797857/final/metadata.json`,
  SHA256 `b5d0bb69adf63b3cf5d4e471209ea072966ee18da50d7a5e75d5f96cda4a5b6e`.
- Its `allocation_accounting/report.json` in the same snapshot root:
  SHA256 `6176e10a8e7023831b5869bd3168699576e0f263cd215310802ca9bf71131bf2`.
- Independent call/cost check:
  `/fact_home/bohanli/flowplace_fql_recovery/final_call_cost_readonly_v1/check_job1798242.json`,
  SHA256 `0b4b045e104647406e250f98ea2705719de51e5918b549ece4bc9b86e87d2da5`.
- Array-alias coverage correction:
  `/fact_home/bohanli/flowplace_fql_recovery/final_call_cost_readonly_v1/allocation_alias_job1798245.json`,
  SHA256 `3cfb5e9c1edffdade8881fa3edfa24088897f6fa48a87d2431be63a0c65847dd`.

Reports retain individual request/outcome/config/task/native fingerprints
and exact source copies. The non-final1797308 and incorrect-denominator
1797301 inventories remain superseded evidence. Later confirmation and
preservation costs require a new dated snapshot and explicit additional-job
manifest; they do not silently replace or enlarge this development scope.
