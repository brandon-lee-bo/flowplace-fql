# Q-study storage interruption and recovery

The 2026-10-18 development runs encountered `OSError(122, Disk quota
exceeded)`. Ceph's per-user `/fact_data/bohanli` limit was 3,298,534,883,328
bytes; its recorded usage was 3,298,758,739,671 bytes. This was an
infrastructure interruption, not a measured physical-design failure. Original
failed attempts, scientific configurations and completed results are retained.
The paused Table 2 work and other projects were not resumed or modified.

Two reproducible caches were relocated under Slurm:

| Job | Cache | Verification and result |
| --- | --- | --- |
| 1797453 | Failed Singularity build `build-temp-2063460906` | All 26,407 entries checked before removal; regular-file bytes, links and modes matched; 1,267,631,177 logical file bytes moved to scratch. |
| 1797455 | Pinned ORFS container image | Exact 1,499,717,632 bytes retained; SHA256 `85f0ce74d79a79c1190632f775f0ed818c8de6e0fafd87717ae3d4839207ea4c` verified before and after migration. |

Both original cache paths now resolve through symlinks to
`/scratch/bohanli/flowplace_qstudy_recovery_20261018`. Copy verification finished
before removing each original cache. Scientific datasets, weights and native
evidence remain in durable storage. The image's declared path and bytes are
unchanged; new native manifests truthfully record its new `resolved_path`.
This is a storage change, not a new tool version or a physical-flow change.
Migration scripts and full entry/image journals are in that scratch recovery
directory. Measured quota usage after migration was 3,295,996,336,494 bytes,
leaving about 2.54 GB under the enforced limit.

The six interrupted GPU checkpoints passed read-only CPU audit 1797458:
complete tensor loading, finite model/optimizer values, actual Adam counters,
global/CUDA/named RNG state, source hashes, config bindings and parent/replay
identities. Audit report
`/fact_home/bohanli/flowplace_fql_recovery/edquot_gpu_v1/report_1797458.json`
has SHA256 `4e99e51b91b6a4a0b8baa816555b650b08aa58d924fc7bf71f388e7ddbf3c5d8`.

| Interrupted job | Replacement | Stream | Committed local step |
| --- | --- | --- | --- |
| 1797370 | 1797462 | Legacy frozen, seed1, online round1 | 101/600 |
| 1797371 | 1797463 | Legacy frozen, seed2, online round1 | 1/600 |
| 1796844 | 1797464 | Bounded timing, seed2, offline | 701/1000 |
| 1796845 | 1797465 | Bounded congestion, seed0, offline | 801/1000 |
| 1796846 | 1797466 | Bounded congestion, seed1, offline | 1/1000 |
| 1796847 | 1797467 | Bounded congestion, seed2, offline | 1/1000 |

These use the original workers and scientific settings, with an explicit
20-minute scheduler time limit. The dependency chain permits two replacement
GPU workers at a time. A corrupt, uncommitted congestion `resume.pt.tmp` is
never a resume source. Its valid `resume.pt` contains the complete step801
state. The original log's later step851 may therefore be replayed; existing
reporters keep the last record per repeated step. No log history was edited.

Main controller 1797468 waits for the two main replacement exports before
reopening the original controller state. The existing immutable round0 native
checks remain required. Bounded finisher 1796852 now waits for all four bounded
replacement jobs and still verifies every original nine-policy export.
Detailed-route controller 1797460 resumes the original 48-request plan, keeping
the 24 completed round0 routes and waiting for round2 sources. Its concurrency
remains two. The final call inventory 1797369 is temporarily held so its
dependencies can include the replacement controllers rather than the failed
ones.

Backend recovery passed 21 contract tests in1797484, an actual81-request
metadata audit/plan in1797485 and independent implementation review. The plan
`qstudy_v1/quota_recovery_v1/plan.json` has SHA256
`74ccbb3e660849dcd44dc187088c9445ed2fc05ebe419e60d5cfd87078de4902`.
The exact frozen evaluator bytecode runs in a private namespace. Only six
SHA-bound unsuccessful EDQUOT attempts are omitted from its attempt iterator,
under the original sample lock. All later attempts remain visible to the
original native verification and duplicate-work guards. The other75 requests
never wrote attempt metadata or invoked the runner. No terminal unsuccessful
transaction is converted into a success, including source index16, which had
saved numerical metrics before its quota exception. Original failed states,
logs, native manifests and metrics remain unchanged.

| Work | Recovery array | Resume controller | Unchanged logical requests |
| --- | --- | --- | --- |
| Source actors | 1797488, indices15--47, concurrency4 | 1797490 | 33 interrupted;15 earlier source outcomes preserved |
| Updated-flow online round1 | 1797489, indices0--47, concurrency2 | 1797491 | 48 interrupted |

Both arrays use8 CPUs/24GiB per native worker. Controllers resume the same
states only after these exports/outcomes finish, so completed stages bypass
the diagnosed old FAILED jobs without changing the controller's restart rules.
The updated-flow controller then continues its original round2 plan.

The final call inventory1797369 was rebound to all five replacement/current
controllers and released. New main/source/paired/flow reports are1797472,
1797504,1797505,1797506; route source resolution/report are1797473/1797474.
Saved-only BC1000/1600/2200 analysis1797476 waits for the main report and makes
no new model inference or physical calls. Recovery attempts do not enlarge
the logical training or evaluation denominator. Actual native invocations,
including the six incomplete infrastructure attempts and their retries,
remain separately counted. No missing or infrastructure-failed label is
replaced by zero or a physical-failure target.
