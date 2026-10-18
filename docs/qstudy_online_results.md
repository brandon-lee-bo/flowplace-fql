# Matched offline-to-online development results

The full declared GRT comparison has completed. Frozen-flow FQL has a small
mean advantage over matched BC after two measured-feedback rounds, but neither
FQL branch meets the prospectively declared stable-improvement rule. Final
detailed routing and fresh confirmation remain pending. This is development
evidence on `bp_fe`, not completion of the paused FlowPlace Table2 reproduction.

## What was compared

All three independent training seeds0,1,2 start from their matched common
initialization. Round0 adds1000 actor updates. Round0 and round1 each export8
collection layouts per policy/seed; rounds1 and2 respectively consume those
measured feedback sets and each add600 updates. Round2 only exports evaluation
layouts. The final actor therefore
has2200 updates in this study, **excluding** the shared800-update actor warmup;
the shared300-update flow warmup is also a separate cost. Every round is
evaluated on the same eight saved development-noise arrays, without putting
evaluation labels into replay.

| Arm | Actor objective | Critic/teacher continuation | Feedback budget per seed by round2 |
| --- | --- | --- | --- |
| Matched BC |10×teacher-coordinate MSE; Q weight0 | Both frozen |16 measured layouts, unused for training |
| Frozen-flow FQL |10×teacher-coordinate MSE − Q | Critic updated; teacher frozen |16 own measured layouts added to replay |
| Updated-flow FQL | Same actor objective | Critic and teacher updated |16 own measured layouts added to replay |

The updated branch imports the exact completed frozen-FQL round0 prefix,
including optimizer/RNG state. Its48 round0 native requests are inherited
references, not48 new physical calls. It adds1200 teacher updates during
online continuation. Its comparison with frozen BC/FQL describes the total
adaptive procedure, including changed teacher and replay; it does not isolate
a Q-only or teacher-only causal effect. Equal collection budgets do not imply
equal training compute. The main frozen BC/FQL control checks pass for all
three seeds and rounds.

These are terminal whole-layout contextual-bandit updates with measured
feedback, rather than a long-horizon Bellman experiment. A policy emits one
complete layout; it is not the stepwise macro-placement policy used in PPO
placers. Current behavior flow takes10 integration steps, whereas the separate
released FlowPlace reference uses its20-step hard-constraint sampler.

## All seven GRT metrics

Each entry is the equally weighted mean of three training-seed means; each
seed mean includes all eight predeclared outputs. `±` below is the sample SD
of the three seed means for WL. Full SDs for all seven metrics, individual
requests and paired differences remain in the immutable JSON/CSV reports.
WL is converted from native µm to m. Signed WNS/TNS are ns; less negative is
better. Overflow columns retain native sum units. GRT does not measure final
DRC, and no final-route congestion is inferred from these values.

![Development policy means and sample SD across three training seeds](../analysis/artifacts/qstudy_development_online_v1/policy_curves.png)

The [underlying curve data](../analysis/artifacts/qstudy_development_online_v1/curves.csv)
and [per-seed values](../analysis/artifacts/qstudy_development_online_v1/per_seed.csv)
are copied from completed read-only render1797689 with SHA-verified provenance.

| Method | Round | rWL(m)↓ | WNS(ns)↑ | TNS(ns)↑ | H overflow↓ | V overflow↓ | Power(W)↓ | Cell area(µm²)↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC |0 |2.314028±.012141 |−.403605 |−217.399 |7.667 |1.375 |.170010 |61200.24 |
| BC |1 |2.322383±.026714 |−.415692 |−220.214 |11.208 |1.458 |.170145 |61317.44 |
| BC |2 |2.312632±.023697 |−.428621 |−224.647 |35.083 |6.500 |.169987 |61108.98 |
| Frozen FQL |0 |2.307890±.003804 |−.407202 |−217.652 |2.625 |.667 |.170140 |61198.83 |
| Frozen FQL |1 |2.300366±.005708 |−.413557 |−212.942 |15.417 |1.792 |.169970 |61302.47 |
| Frozen FQL |2 |2.294036±.033791 |−.407682 |−213.833 |27.375 |5.500 |.169744 |60975.11 |
| Updated FQL |0, inherited |2.307890±.003804 |−.407202 |−217.652 |2.625 |.667 |.170140 |61198.83 |
| Updated FQL |1 |2.341368±.031897 |−.423541 |−215.375 |25.750 |7.083 |.170793 |61510.90 |
| Updated FQL |2 |2.298117±.050592 |−.403953 |−215.564 |20.417 |11.542 |.170038 |61281.28 |

The matched-only report verified all240 native outcomes across18 original
arm/seed/round streams. The three-arm report verified360 outcome
representations across27 streams, including the48 inherited updated-round0
references. These representation counts are not the total unique physical
call budget: the final fixed-scope inventory separately accounts reuse and
real retry attempts. Both reports have status `complete`, zero invalid
streams and no physical-failure conditioning.

## Paired conclusions

| Comparison | Mean WL change | Paired-seed WL differences(µm), seeds0/1/2 | Primary gate |
| --- | --- | --- | --- |
| Frozen FQL0 − BC0 |−0.265% |−10382.000 / −19520.750 / +11487.625 |Fails magnitude and all-seed consistency |
| Frozen FQL1 − BC1 |−0.948% |−9187.750 / −54872.625 / −1991.125 |Below1% |
| Frozen FQL2 − BC2 |−0.804% |−65782.125 / +37212.375 / −27217.000 |Fails magnitude and all-seed consistency |
| Updated FQL2 − BC2 |−0.628% |+10619.625 / −46483.250 / −7681.625 |Fails magnitude and all-seed consistency |

Frozen FQL2−BC2 has paired mean WL delta−18595.583µm, sample SD52035.693µm.
Its mean WNS improves.020939ns, TNS improves10.814296ns, H/V overflow fall
7.708333/1.000000, power falls.000243375W and area falls133.875µm². These
favorable mean secondary differences do not erase the failed primary rule or
the between-seed variation. Its own mean WL falls about.600% from round0 to2;
this does not establish stable improvement over the matched continuation.

Updated FQL2−BC2 has paired WL delta−14515.083±29158.304µm. Its mean vertical
overflow increases5.041667, exceeding the operational tolerance1. Its mean
WL is4080.500µm higher than frozen FQL2, with larger between-seed SD. These
observations do not support promoting this adaptive branch. They do not prove
that teacher adaptation generally harms placement or identify a single causal
mechanism: replay, critic and teacher all evolve together.

The declared source64 and bounded-objective branches are already closed
without promotion; see [source results](qstudy_source_results.md) and
[bounded results](qstudy_bounded_objective_protocol.md). No new method has
passed the promotion rule. The prospective default remains frozen legacy
round2 versus BC round2; final selection is still contingent on reviewing
the complete declared routing, provenance, cost and preservation records.
It is a valid negative-confirmation candidate, not a claim that it is best.
Round1 is reported without selecting its more favorable seed pattern after
looking at these results. In addition to missing1%, frozen FQL1 increases H
overflow by4.208333 versus BC1, exceeding the tolerance1.

The initial common behavior flow has development mean WL2.279086m. Both
round2 actor means remain higher; beating BC by a small amount is not
superiority to the teacher. Comparisons use policy means rather than a best
layout, and one-step versus10/20-step generation has different inference
budgets. Isolated inference latency has not been measured.

## Immutable evidence

Matched report, Slurm1797472:
`/fact_data/bohanli/flowplace_fql/qstudy_v1/actors_v1/matched_report_v1/20261018T140923.407736Z_job1797472/report.json`,
SHA256 `3785ad38689db6be99797f1efe18eb1c1d81985cfacd6f1218a814a49b30a1f9`.

Three-arm report, Slurm1797506:
`/fact_data/bohanli/flowplace_fql/qstudy_v1/actors_v1/flowfactor_report_v1/20261018T140923.407752Z_job1797506/report.json`,
SHA256 `774612c45e3ab008c83f1e064a66e98a59e6cb82c859dcb7796d3a851a00d5ad`.
Each sibling `completed.json` binds its exact report and CSV artifacts.

Prospective thresholds are the unchanged
[confirmation rules](../fql/qstudy_confirmation_rules_v1.json), SHA256
`b6553120d9a4f410740d9f45ee3214dfa62b48f27b0a75a8c4cfe262f9b940ab`.
They are practical study thresholds, not field standards or significance
tests. The [fresh protocol](qstudy_fresh_confirmation_protocol.md) reserves
new actual noise inputs and forbids tuning or replay with their outcomes.
