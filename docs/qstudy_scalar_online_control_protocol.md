# Matched scalar GRT offline-to-online control

This is a development experiment on `bp_fe`, with three independently trained
seeds, a frozen behavior flow, and two measured continuation cycles. It does
not select a new reward, replace the final-PPA study, or use the reserved fresh
noise stream. Existing scalar adapters and physical workers remain unchanged.

The actual completed `rule_grt_wl` offline actor is the FQL T0. Its scalar
critic predicts the fixed TRAIN-normalized bounded GRT wirelength cost B, and
Q is exactly minus the ensemble mean B. The genuine BC T0 retains its own
original actor, actor Adam, teacher, and RNG. The BC provider separately binds
the same scalar refit and an empty critic Adam; it never rewrites the old BC
checkpoint or claims the BC actor was trained with that scalar critic.

| Branch | Online alpha | Q weight | Critic updates per cycle | Actor updates per cycle |
| --- | --- | --- | --- | --- |
| genuine BC | 10 | 0 | 0 | 600 |
| updated Q, alpha10 | 10 | 1 | 600 | 600 |
| frozen Q, alpha10 | 10 | 1 | 0 | 600 |
| updated Q, alpha30 | 30 | 1 | 600 | 600 |

The three FQL branches start from the **same real T0 checkpoint and the same
C0 requests/outcomes**, not copies renamed as three different actors. Their
first training requests/settings/output directories define the fork. BC has
its own genuine T0 and own C0. Each completed T1 then exports its own C1 and
continues its own complete actor/critic/Adam/RNG prefix into T2. The original
T0 is only the frozen teacher/TRAIN/schema/normalizer anchor in cycle2.

The offline prefix is the existing 1000-step alpha10 training. Alpha30 is an
online intervention from this shared prefix, not an offline-alpha30 result.
All flows, TRAIN scales, normalizer buffers, initial scalar critic, actor and
critic learning rates, and backend settings are inherited unchanged. Actor
and critic clipping is 1, Huber delta is 1, minibatch is 16. Updated critics
sample eight offline rows and eight own online rows per update. Cycle1 sees
eight C0 labels; cycle2 samples uniformly from the sixteen own C0+C1 labels.
Success and confirmed GRT geometry/physical failure share the same scalar
Huber regression, with B4 for confirmed failure. Infrastructure/unresolved
labels are never filled with zero. The native outcome's GRT reward is retained
unchanged; derived B is a separate training target.

The immutable manifest declares CUDA evaluation seed81400000/count8 at T0,
T1, and T2 for every actor seed and method. C0 uses seed81000000+100*seed;
C1 uses seed81001000+100*seed. Common collection noise across methods pairs
the input noise, not their generated actions or measured feedback. Collection
and evaluation are disjoint; evaluation never enters replay. Seed85000000 is
reserved for a later locked fresh confirmation. Actual saved noise hashes,
not seed integers alone, must match at the stage gates.

There are 384 distinct GRT requests: two T0 sources × three seeds ×16
collect/evaluation requests =96, four T1 branches × three seeds ×16 =192,
and four T2 branches × three seeds ×8 evaluation-only requests =96. Initial
FQL C0/E0 are physically measured once and referenced by all three branches.
There are 72 fixed DRT requests: the first four evaluation indices for the
two T0 sources and four T2 branches, each across three actor seeds. T1 has
no DRT. All use the unchanged eight-thread, cap20, scratch physical flow.
These are requested-layout budgets: actual attempts/preemptions are retained
separately, and upstream physical failures may have no DRT invocation.

Before any formal submission, root reviews the exact manifest/source closure
and the independent checkpoint certificate. Before any T1 child export, the
actual alpha10/alpha30 T1 critic tensors, critic Adam and Q-generator states
must be identical for each seed, with 600 actor and critic updates each.
This tests that alpha changed actor extraction while the initial online-Q
fit used the same data/update stream. The frozen-Q and BC prefixes are
separately validated by their existing adapters. T2 must have actor Adam2200;
updated branches have critic Adam1200, frozen-Q/BC have critic Adam0, and
teacher updates remain zero. No checkpoint or noise is chosen using Q/PPA.

The prospective manifest includes the complete four-arm topology. The first
reviewed driver implements only explicit `prepare0`, `export0`, and `native0`
stages; later train/gate/child/T2/DRT glue is not yet implemented. There is no
long-running new controller. Root advances each stage after actual closure.
The stage0 helper, wrappers, and this protocol are themselves bound and copied
into the immutable stage0 source closure. Later stages must use new independent
helper files and a stage addendum referencing the original manifest SHA. They
must not edit these files or rewrite the original manifest to extend it.

Stage0 prepares use two CPUs/8 GiB/10 minutes; exports use two CPUs/20 GiB/
one `nvidia_l20` GPU/15 minutes; GRT uses eight CPUs/24 GiB/two hours. Export
concurrency is two, GRT concurrency is four. No TimeMin shortening is enabled.

`tools/qstudy_scalar_online_control.py` only creates immutable requests,
invokes already validated entry points, journals scheduler submissions, and
dispatches missing work. It preserves true shared-root/child/second-cycle
identities. Slurm PREEMPTED/NODE_FAIL/BOOT_FAIL may recover missing tasks at
most three generations. FAILED/TIMEOUT/unknown outcomes stop for diagnosis;
they do not become bad-PPA labels or authorize a new sample. All attempts,
submission ambiguity, source SHA and original terminal states remain in the
journal. No historical Table2 task or unrelated project is touched.

R2 is the predeclared primary endpoint; R0 and R1 are also reported in full,
without choosing the best round after seeing results. Primary comparisons are
updatedQ10 versus same-parent frozenQ10, alpha30
versus updatedQ10, and each FQL branch versus genuine BC. Within-seed fixed
noise means precede three-seed mean/sample SD. Report all seven GRT metrics,
six final DRT metrics, DRC and full failure/missing denominators, as well as
gradient contributions/Q calibration. GRT all-eight and DRT first-four
differences are not a stage-change causal estimate. Positive results do not
establish a globally optimal alpha or timing closure; negative results remain
part of this fixed control. No automatic winner promotion is implemented.
Training diagnostics are predeclared every 100 updates. Any later gradient
probe must preserve the named actor/Q training streams, using separately bound
diagnostic RNG or saved-state restoration; this stage0 driver performs no such
probe or training update.
