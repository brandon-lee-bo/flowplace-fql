# Table 2 physical-artifact storage

The durable copy of each physical intermediate remains on `/fact_data`: first
as ordinary files, and, after an explicitly authorized archive commit, as a
verified `artifacts.tar.zst`. `/scratch` is only a convenience cache. No placement,
routing, timing, or metric extraction parameter is changed by this operation.

## Bounded inspection on 2026-10-17

| Completed diagnostic | `results` bytes | `objects` bytes | Files |
| --- | ---: | ---: | ---: |
| `table2_paper_gpl_diag_v1_bp_fe` | 901,133,572 | 32,627 | 40 + 4 |
| `table2_paper_gpl_diag_v1_bp_be` | 1,386,399,905 | 32,628 | 40 + 4 |

The bp_fe logs and reports together occupy about 1.5 MB. Its largest files are
stage ODBs; several stage ODBs have equal lengths, but no deduplication or equality
assumption is used. All files in the two selected trees are archived, including
small files and empty directories. This avoids a suffix filter accidentally
omitting an input needed to reopen a saved stage.

The ongoing `table2_legacy_tool_v1_bp_fe` was observed as `running` and is excluded.
At that inspection, historical diagnostics were retained for a non-destructive
compression measurement. Subsequent verified commits are recorded below.
DREAMPlace installations, generated DEFs, original release files, tool images,
and all other trees are outside this helper's scope.

## Evidence readers and links

`audit_fullflow.py` reads root manifests/metrics, `orfs_work/logs`, and
`orfs_work/reports`. `compare_table2_method.py` and the two generated/methods
summarizers consume those reports and provenance. Their `inside()` check resolves
paths and rejects evidence outside the physical run directory. Therefore those
evidence trees, including input DEF snapshots in `provenance`, must stay on bulk.
The currently audited report paths do not refer to `results` or `objects`.

Only `orfs_work/results` and `orfs_work/objects` may become cache links. Interactive
OpenROAD stage reopening can follow those links while the cache exists. Existing
Table 2 runs are never resumed by `run_table2.py`; it refuses existing run IDs.
If scratch is purged, metric audits still work, but stage reopening requires a
restore. A missing scratch cache does not mean the archived physical result is
lost. The immutable original run manifest is not edited: storage history is in
the separate archive directory's `archive_manifest.json` and `commit_state.json`.

## Helper and gates

`backend/scripts/archive_table2_artifacts.py` has three separate actions:

1. `create`: require completed/verified run and the exact Slurm task
   `COMPLETED/0:0`; reject symlinks, symlink ancestors, special files and hardlinks
   in both bounded trees; hash all source files; stream a PAX tar through fixed
   `zstd -3`; decode and independently hash every archive member; rehash the
   original trees and control JSONs; finalize the archive and its manifest.
   Source directories are unchanged. A partial or failed attempt remains visible.
2. `verify`: verify archive/inventory SHA256 and repeat full decoded member
   identity verification without requiring the original directories or cache.
3. `commit`: additionally require `--confirm-remove-bulk-backup`. Verify the
   archive, completed run and unchanged sources; copy to a new scratch directory;
   hash-verify the copy; flush copied files; rename both original bulk directories
   to unique backups in the same run; publish and verify the cache links; then
   remove only those two backups. The durable archive is already complete before
   the removal phase starts. Actual commits have a separate `commit_state.json`.

The exact task is derived from `slurm_array_job_id` and `slurm_array_task_id` when
present, so a completed array element can be archived while its siblings run.
Archive staging checks bulk filesystem free space and Ceph quota, conservatively
reserving source size plus compression overhead plus 8 GiB. Scratch staging checks
both filesystem free space and the `/scratch/bohanli` Ceph quota, retaining 8 GiB.
An archive-root lock prevents concurrent archive/commit jobs, but it does not
reserve space against independent fullflow producers. Keep archive jobs serial
and account for concurrent fullflow growth when choosing producer concurrency.

`commit_state.json` records backup paths before each rename. Ordinary Python
errors during publication attempt rollback before any backup removal. A job kill
can leave a recorded backup or published cache link; inspect this state before
any retry. Commit refuses a preexisting commit state and never overwrites an
existing cache. Do not rerun or manually remove leftover backups blindly.

## Compression probe and resources

`archive_table2_artifacts.slurm` requests 2 CPUs, 2 GiB and 10 minutes on
`cpu-scavenger`. GNU tar 1.34 and zstd 1.5.5 are available; Python's streaming tar
writer preserves the tree without materializing an uncompressed archive.

Probe 1788012 stopped at the scheduler gate before compression: querying an old
finished job with `squeue -j` returned 1 on the compute node. Its log is retained.
The helper now uses exact `sacct` completion evidence. Retry 1788014 writes:

```
/fact_data/bohanli/flowplace_fql_backend/table2_archives/
  table2_paper_gpl_diag_v1_bp_fe/zstd3_probe_v2/
```

The probe does not save bulk space because the uncompressed source is retained.
Use its measured archive ratio, not a promised 50%, for formal-run budgeting.
Stage new formal runs in bulk with the existing runner; only after fullflow audit
and the reviewed archive commit should their uncompressed bulk copy be released.

## Commands and recovery

Create and verify use the wrapper. Pass an absolute run path and a new archive
directory under the fixed durable archive root:

```sh
sbatch backend/scripts/archive_table2_artifacts.slurm create \
  --run /fact_data/bohanli/flowplace_fql_backend/table2_results/RUN_ID \
  --archive-dir /fact_data/bohanli/flowplace_fql_backend/table2_archives/RUN_ID/ATTEMPT
sbatch backend/scripts/archive_table2_artifacts.slurm verify \
  --archive-dir /fact_data/bohanli/flowplace_fql_backend/table2_archives/RUN_ID/ATTEMPT
```

Commit is deliberately separate and applies only to completed, verified runs
within this project's artifact-storage scope:

```sh
sbatch backend/scripts/archive_table2_artifacts.slurm commit \
  --archive-dir /fact_data/bohanli/flowplace_fql_backend/table2_archives/RUN_ID/ATTEMPT \
  --scratch-dir /scratch/bohanli/flowplace_table2/artifact_cache/RUN_ID/ATTEMPT \
  --confirm-remove-bulk-backup
```

To restore, first run `verify` above and compare the archive SHA256 with
`archive_manifest.json`. In a Slurm allocation, create a **new empty directory**
with sufficient quota, then use:

```sh
tar --zstd --extract --preserve-permissions --no-same-owner --keep-old-files \
  --file /fact_data/bohanli/flowplace_fql_backend/table2_archives/RUN_ID/ATTEMPT/artifacts.tar.zst \
  --directory /NEW_EMPTY_RESTORE_DIRECTORY
```

This recreates `orfs_work/results` and `orfs_work/objects` underneath that
directory. The saved per-file `source_inventory.json` contains SHA256 and length
for checking the restored files. Restore to a new location and verify it before
repointing any cache links; never extract through old links or over existing
results. Archive and restore are storage operations, not new PPA experiments.

## Later commits and scratch-first compression

The verified bp_fe diagnostic archive was committed by job 1788129. The bp_be
diagnostic was archived and committed by jobs 1788131/1788132. Their complete
durable archives retain all intermediate files; source logs, metrics, reports,
and manifests stay in place. These storage operations do not add successful
rows to the formal comparison.

The completed formal FlowPlace bp run passed its physical audit, but archive
job 1788074 refused creation because its roughly 7.6 GiB of intermediates plus
the 8 GiB reserve exceeded available quota. Its dependent commit 1788075 was
cancelled; the source was untouched. No physical experiment is repeated for
this storage failure.

`archive_table2_staged.py` is a separate, versioned variant; the helper used by
already submitted jobs is unchanged. It retains the original completion gate,
inventory, full decoded archive verification, unchanged-source checks, and
commit procedure. The only storage-policy change is to compress in a new
scratch staging directory first. After full verification, it admits a durable
copy only if the **actual compressed size plus 8 GiB** fits both filesystem and
quota headroom. It flushes and SHA256-verifies that copy before publishing the
durable archive or removing the scratch compressed temporary file. Failed
attempts retain their evidence and source; commits remain separate.

The new helper SHA256 is
`7fe87e6e021d8da8f4ad680515640d0af716f731d7fce18650dbf4a63f50fcbf`.
The first bp recovery create/commit jobs 1788200/1788201 completed with exit
code 0. The 8,183,407,597-byte source was retained in a 1,252,945,345-byte durable
archive; the scratch copy was verified before the redundant bulk directories
were removed. `backend/plans/table2_background_v1/bp_archive_recovery.json`
records the arguments and the verified archive/commit-state identities.

The new plan-bound driver `archive_table2_staged_task.py` uses this helper with
the same six-design task, manifest, and source-job checks as its predecessor.
Its SHA256 is
`e5f974f6642dd568c130cf166650ceb904748a795080b12bf45014aa21ade5b4`.
Only pending storage tasks were replaced: array 1788218 covers DREAMPlace
indices 1–5, and 1788219 covers original FlowPlace indices 3 and 5. Each remains
conditional on its exact successful source task. Submission does not establish
success; the journal is
`backend/plans/table2_background_v1/staged_archive_arrays.json`.

## Superseded infrastructure interruptions

`archive_interrupted_table2_staged.py` is a separate helper for stopped,
superseded infrastructure attempts. It does **not** use the successful-run
classification. Its gates require an unchanged frozen retry binding, equal
old/new tasks except run ID, the exact source job in PREEMPTED/NODE_FAIL/
BOOT_FAIL, and no final metrics or completed final-report JSON. Active jobs,
application failures, and successful results are ineligible. All original
manifests, logs, reports and failure classifications remain unchanged.

Two large PREEMPTED attempts were selected by artifact size, independently of
PPA. Verified durable archives now preserve all 6,809,579,940 source bytes:

| Interrupted run | Verified commit job | Source bytes | Compressed bytes |
| --- | --- | ---: | ---: |
| Original FlowPlace bp (`1788030_1`) | 1788246 | 2,749,925,034 | 351,359,757 |
| Original FlowPlace ariane133 (`1788030_2`) | 1788239 | 4,059,654,906 | 744,413,106 |

The first bp commit, 1788237, exhausted its 2 GiB allocation while copying to
scratch. Before recovery, its state was inspected: `copying`, empty backup-path
map, both original source directories still ordinary directories. Nothing had
been published or removed. Recovery 1788246 used a new archive attempt and cache
directory with 8 GiB, independently reverified the durable archive and unchanged
sources, then committed. The failed state and partial cache are retained. The
ariane archive and commit used 12 GiB and completed normally.

Archive copy page cache contributes to Slurm memory accounting. Remaining
large archive jobs therefore request 12 GiB (20 GiB for bp) and exclude
`enterprise`, where primary physical runs execute. These are storage resource
changes, not physical-flow changes. The new automatic physical retries retain
their original input hashes and continue independently.

The complete journal is
`backend/plans/table2_background_v1/interrupted_artifact_preservation_v1.json`.
The durable attempts are respectively `interrupted_preservation_recover_v2`
and `interrupted_preservation_v1` under each original run's archive directory.
They remain explicitly classified as
`superseded_infrastructure_interruption_not_a_success` and add no successful
rows to the Table 2 comparison.

The same infrastructure-only gate subsequently preserved interrupted
DREAMPlace bp and ariane133 attempts, jobs `1788077_1`/`1788077_2`, after
identical-input replacements were frozen. Create/commit jobs
1788266/1788267 and 1788268/1788269 completed with exit code 0. Their durable
archives contain respectively 3,624,978,015 source bytes in 486,362,130
compressed bytes and 1,332,090,552 source bytes in 214,773,507 compressed bytes.
The journal is `dreamplace_interrupted_preservation_v1.json` in the same plan
directory. These attempts remain PREEMPTED, not successful physical results.

## 容器缓存按实验室规则迁移（2026-10-17）

可重新下载的 `singularity-cache` 已由Slurm1788367迁移到
`/scratch/bohanli/flowplace_table2/container_cache_relocated_v1`，原路径保留链接。
复制前后逐文件SHA256、大小、mtime和权限一致，重复源副本确认后移除，
释放3,044,661,364 bytes持久额度。正式SIF、工具二进制、源码及全部实验结果
均不在此次迁移范围。当前物理wrapper使用各自scratch TMPDIR，不使用旧缓存
作为运行根目录。完整清单见 `backend/plans/table2_background_v1/container_cache_relocation_v1.json`。
先前1788364仅因wrapper路径拼接错误而在执行前退出2，没有读取或修改缓存；
错误wrapper及修正记录均保留。

FP ariane133重试完成后的归档1788346/1788347也已校验并提交：源
4,522,215,640 bytes压缩为809,900,824 bytes，原结果大目录转为已验证缓存链接，
持久压缩包和文件清单均保留，报告/manifest/指标未修改。
