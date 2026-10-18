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

Backend recovery is being independently audited for the 81 interrupted source
and updated-flow requests. Recovery attempts do not enlarge the logical
training or evaluation denominator. Actual native invocations, including
incomplete infrastructure attempts, remain separately counted. No missing or
infrastructure-failed label is replaced by zero or a physical-failure target.
