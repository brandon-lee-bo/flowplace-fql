# FlowPlace + FQL investigation

This repository studies whether Q-guided policy learning can improve macro
placement beyond a distilled FlowPlace policy. It contains the baseline adapter,
whole-layout and sequential research prototypes, matched controls, and native
OpenROAD evaluation code. **No stable overall PPA advantage has been established.**

## Latest version comparison — 2026-10-23

| Version | Main change | Current evidence |
| --- | --- | --- |
| V1.0 | Frozen Euler10 teacher; whole-layout one-step actor with terminal Q | Three development training seeds: mean paired rWL improvement +0.114%, SD 1.566 percentage points; only one seed improves |
| V1.1 | Replace teacher with frozen public FlowPlace weights and exact constrained20 sampler | Sampler equivalence and both seed-0 training arms complete; matched backend comparison incomplete |
| V2_4 | Prefix-conditioned teacher, one macro per action, graph critics and Bellman updates | One seed; padding=.001 execution variant lowers mean GRT rWL 0.640% versus matched BC, with worse TNS, area and vertical overflow |

The V1.0 online controls do not isolate a benefit from new feedback: equal-update
training on offline data alone outperforms the online branch in the measured
R1/R2 comparisons. V2's original padding=0 execution has low native acceptance
(BC2/8, FQL1/8); its padding=.001 results are a separately declared variant.
These are fixed-design GRT experiments, not final detailed-route/Table2 claims.

- [Measured results, controls and limitations](docs/placement_versions.md)
- [Training parameters, data requirements and runnable entrypoints](docs/placement_versions_reproduction.md)
- [Frozen released FlowPlace teacher contract](docs/released_flowplace_teacher.md)
- [Machine-readable result snapshot and evidence digests](docs/data/placement_versions_20261023.json)
- [Fresh-run configuration examples](examples/placement_versions)

Publication verification passed all57 CPU Slurm tests, including real public
weights, exact original-sampler output equality, teacher freezing, actor/Q
updates, sequential masking and exact checkpoint recovery. See the reproduction
guide for the test command. These checks establish code behavior, not quality gains.

## Earlier studies

Earlier graph offline-to-online runs, expert/final-feedback reward screening,
scalar controls and final-routing studies are separate protocols. Their budgets,
reward scales, teachers and physical stages must not be mixed with this version
comparison. The existing records retain their original conclusions and provenance:

- [Graph/FQL experiment record](docs/graph_fql_offline_to_online.md),
  [completion audit](docs/completion_audit.md) and [run ledger](docs/run_ledger.md).
- [Q-study execution guide](docs/qstudy_reproduction.md),
  [decision table](docs/qstudy_decisions.md) and [online comparisons](docs/qstudy_online_results.md).
- [Measured Euler10/actor inference costs](docs/qstudy_cost_inventory.md#completed-graph-inference-measurement).
  Those timings are not a public constrained20 speed comparison.
- [FlowPlace/DREAMPlace Table2 protocol](docs/table2_protocol.md) and
  [pending pipeline](docs/pending_pipeline.md). Historical controllers remain paused.

## GitHub checkout

Repository: https://github.com/brandon-lee-bo/flowplace-fql

```sh
git clone https://github.com/brandon-lee-bo/flowplace-fql.git
cd flowplace-fql
```

The repository includes experiment code, configuration, Slurm scripts, research
records and the existing local commit history. Python environments, model
weights, datasets, downloaded upstream sources and large native EDA artifacts
remain on the server and are excluded from Git. This is a source-code snapshot,
not a self-contained copy of the experiment data. The latest version-development results are documented
in [the version comparison](docs/placement_versions.md).

For a fresh checkout, restore the official sources at the revisions recorded in
`manifest.json`:

```sh
git clone https://github.com/lamda-bbo/flowplace.git upstream/flowplace
git -C upstream/flowplace checkout --detach da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547
git -C upstream/flowplace rev-parse HEAD > upstream/flowplace/SOURCE_COMMIT
git clone https://github.com/seohongpark/fql.git upstream/fql
git -C upstream/fql checkout --detach e8cd16eb490332924dfa2492097219f181765933
git -C upstream/fql rev-parse HEAD > upstream/fql/SOURCE_COMMIT
```

Release checkpoints and physical-design inputs must be restored separately;
see [the baseline adapter](baseline/README.md) and
[the backend instructions](backend/README.md). Existing job scripts contain
FactLab paths, resource requests and references to server-local artifacts. Review
these before submitting from another checkout; use Slurm for computation.
Historical Table 2 controllers remain paused.

## Code map and evidence boundaries

`baseline/` wraps the pinned upstream FlowPlace model. `backend/` exports macros
and runs the native physical-design flow. `analysis/` validates geometry and
metrics. `fql/versions_whole_train.py` implements matched whole-layout training;
`fql/sequential_{env,teacher,models,train}.py` implements the sequential prototype.
The older graph/Q-study entrypoints remain available for their separate protocols.

Whole-layout V1 is a single-action contextual bandit; V2 makes sequential
placement decisions. Macro-only HPWL, GRT and detailed-route metrics describe
different evaluation levels. Record legality, fixed sample denominators, seeds,
failures and cost alongside quality. Reward is a chosen measured objective;
Q is its learned estimate and can be wrong. Neither a higher predicted Q nor a
lower wirelength alone establishes better overall PPA.

Model weights, datasets, downloaded upstream sources, runtime logs, containers
and large EDA artifacts are excluded from Git. Full historical reproduction
requires their separately preserved provenance. Computation uses finite Slurm
allocations under [the cluster execution policy](docs/cluster_execution.md).
