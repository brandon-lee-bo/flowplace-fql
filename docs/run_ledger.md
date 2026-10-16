# Run ledger — 2026-10-16/17, Hong Kong

All substantial work is submitted through Slurm. Earlier failed runs are retained.
This ledger records key jobs; individual outputs/manifests contain exact arguments.

| Job | Purpose | Evidence / outcome |
|---|---|---|
| 1783809 | Pinned OpenROAD container build | Completed; image SHA in backend manifests |
| 1783818 | First released bp_fe backend | Failed inherited FLOW_HOME; environment isolated afterward |
| 1783828 | Second backend attempt | Reached GRT, failed author's Tcl env scope; one-token patch saved |
| 1783834 | Released bp_fe full backend | Completed in 815.84 s; 1904 DRCs, not a clean route |
| 1783836 | Remaining five released Table 2 circuits | Array, at most two concurrent |
| 1783831 / 1783893 / 1784011 | FQL software smoke/tests | Passed; synthetic checks, not placement quality evidence |
| 1783842 / 1783969 | Independent metrics / pinned-environment tests | Passed; latter 9 checks |
| 1783894 | Initial wheel install | Failed conflicting setuptools versions; explicitly pinned 75.3.4 |
| 1783934 | Dependency installation | Expanded most packages; final step preceded cuDNN download completion |
| 1783962 | Completed pinned environment | pip check and torch/PyG imports passed |
| 1783946 | First generated bp_fe + adapter check | Layout legal; compile/eager comparison failed, investigated below |
| 1783965 | Planned eager six-circuit generation | Cancelled before execution when equivalence was unresolved |
| 1783973 | Released-mode six-circuit generation | All six macro layouts legal; original nondeterministic GPU mode |
| 1783970 | First generated bp_fe full backend | Completed in 461.80 s; 12 DRCs, not a clean route |
| 1783972 | GRT-only evaluator validation | 141.22 s; final ODB and physical metrics match full-flow GRT |
| 1783987 | Controlled sampler diagnosis | Network forward varies; identical legalizer input gives exact repeats |
| 1783995 | Deterministic execution check | Three seeds: repeated and compiled/eager outputs exactly equal |
| 1784006 | Deterministic baseline + six circuits | Adapter checks passed; all layouts geometrically legal |
| 1783988 | Initial dataset collector launch | Import-name collision before collection; package/import order fixed |
| 1784007 | Deterministic bp_fe dataset | 288 teacher families, 864 critic rows, 100% legal; 3m32s |
| 1784018 | Three-seed macro-reward training | BC, FQL alpha 1/10/100, direct analytic student controls |
| 1784019 | Predeclared backend-label export | 96 DEFs: 32/8/8 families per split, variants 0 and 1 |
| 1784031 | Coordinate controls, validation selection, heldout proxy evaluation | Depends on training |
| 1784047 | Real GRT labels | 96 tasks, maximum two concurrent, 8 CPUs / 12 GiB each |
| 1784060 | Local teacher sensitivity probe | Exact repeats; intermediate hard projection strongly amplifies small input changes |
| 1784068 / 1784070 / 1784092 | Independent dataset/evaluation audit and plots | Passed; 288 families, 864 labels, all selected heldout geometries and HPWL checked |
| 1784098 | Predeclared proxy-to-GRT candidate export | 36 candidates: all 12 selected method/seed combinations × first 3 heldout families |
| 1784102 / 1784104 / 1784108 / 1784112 | No-hard teacher plus final projection diagnostic | First failed during imports; GPU data completed on retry, CPU plot recovery; first 16 TRAIN families only |
| 1784105 | Recover absent congestion reports | Read final ODB without rerouting; missing category explicitly proved zero markers |
| 1784109 | Proxy-to-GRT cross-fidelity evaluation | 36 predeclared tasks, one concurrent; separate from 96-label array |
| 1784119 | Strict GRT dataset relabeling | Queued after all 1784047 tasks end; requires all 96 verified metrics, including repaired failures |
| 1784125 | Experimental standard Euler sampler | 16 TRAIN noises; exact same forward arithmetic verified; official code unchanged |
| 1784133 | Fixed anchor controls | One zero-center and one TRAIN-mean layout; both legal; no replicated noise measurements |
| 1784131 | Deeper detailed-route diagnostic | Two fixed bp_fe examples, cap20, one concurrent; reuse GRT, no new placement |
| 1784135 | GRT workflow software tests | 11 passed, including critic loading and heldout isolation; synthetic software checks |
| 1784147 | Resource accounting snapshot | Captured documented/local-log job allocation time; failed jobs included |
| 1784154 | Released bp_fe deeper-DRT retry | First diagnostic was Slurm PREEMPTED; retry uses eda after the original array, no overwrite |
| 1784155 | Sampler/anchor backend export | 8 new candidates plus 3 exact teacher results to reuse; all DEFs geometrically legal |
| 1784156 | Three-seed GRT critic gate | Queued after verified relabeling; fixed 64×64 network, 1000 steps, validation only |
| 1784157 | Validation-selection sealing check | Passed; rejects changed heldout plan |
| 1784158 | Sampler/anchor backend evaluation | 8 new tasks, one concurrent, after proxy-transfer array |
| 1784160 / 1784161 | Final GRT/proxy provenance audits | Queued after corresponding arrays, require all requested metrics valid |
| 1784185 / 1784188 | Euler teacher BC learnability diagnostic | First failed before generation on optional mask fields; retry completed in 3m14s, three seeds, identical original hard-BC layouts verified |
| 1784189 / 1784190 | Fixed-control pairing and GRT exporter tests | Both passed; fixed controls never become noise replicates; reuse/selection/checkpoint bindings verified |
| 1784231 / 1784232 | Initial proxy and sampler aggregation schedule | Cancelled while pending after a real CTS failure required explicit physical-failure outcome reporting; no metrics discarded |
| 1784233 / 1784234 | Initial GRT plots schedule | Cancelled while pending with parent aggregations; replacements will preserve incomplete methods without success-only means |
| 1784311 | Six-design Table 2 summary | Queued after fullflow array; retain any failures and report every declared circuit |
| 1784327 / 1784328 | Fullflow numerical consistency audits | Already-completed examples and all six released runs; summary additionally waits for final audit |
| 1784344 | Generation-plus-legalization latency | Completed in 45s; 16 fixed noises ×3 rounds ×7 paths, original saved final coordinates reproduced exactly |
| 1784384 | Physical CTS failure evidence | Binds retained RSZ-0060 log, original manifest, frozen task, DEF and action; does not modify failure or rerun layout |
| 1784418 | Ariane136 fullflow raw-report audit | Queued after completed route; no optimization |
| 1784450 / 1784451 | Final exploratory outcome classification | Depend on proxy/sampler arrays; definite physical failure is an outcome, while unknown/infra/parser failures remain unresolved |
| 1784472 | Physical-failure reporting tests | Five passed; real retained failure recognized, corrupt/unknown failures rejected, denominators and seed/fixed-control rules verified |
| 1784481 / 1784482 | Exploratory GRT outcome reports | After final outcome classification; preserve physical failures and suppress incomplete-method/configuration means |
| 1784485 / 1784486 | Failure-aware GRT plots | After verified outcome reports; show failure counts, all candidate CSV rows, and no incomplete-method mean or pooled correlation |
| 1784533 | Released DEF geometry audit | All six checked against released LEFs; swerv has three raw macro overlaps, other five have none; no coordinates changed |
| 1784534 | bp_be fullflow raw-report audit | Completed and passed; DRC247 retained separately from flow completion |
| 1784543 | Generated deterministic swerv fullflow | Independent official-checkpoint baseline after released input failed macro overlap; original failed Table 2 row remains unchanged |

## Interpretation fixes

The initial compile/eager discrepancy did **not** establish a compiler error.
A controlled comparison found GPU network prediction variation; subsequent
discrete legalization amplifies small numerical differences. Setting
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`
removed the difference on the tested seeds. Formal data uses this deterministic
mode; original-mode artifacts remain separately labeled.

Author Nangate45 config limits detailed routing to iterations 0–4. Finishing the
flow therefore does not imply zero DRCs. The paper's Table 2 values are not yet
numerically reproduced. Area differences cannot be explained by simply adding
macro area: the author's parser explicitly uses standard-cell area.

Global-route labels use the final stage JSON wirelength. The author's full-flow
parser selects the first GRT total from its log; these are different report
points. Final DRT wirelength and GRT wirelength are kept separate throughout.

Ceph quota checks found about 534 GiB free in the user's home quota, versus about
55 GiB under fact_data. New GRT results remain in home, with all intermediates
retained; no active run is moved and other project data is untouched.

The first two tasks in array 1784047 completed global routing but failed during
postprocessing because no congestion report file was emitted. Pending tasks were
held while the backend parser was fixed. Missing reports must not be interpreted
as zero congestion without evidence from the final saved database. Existing
physical results and original failed Slurm statuses are retained for recovery.
