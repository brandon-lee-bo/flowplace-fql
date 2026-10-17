# FactLab execution

User explicitly requires Slurm scheduling (2026-10-17 local time).
Inspected `/etc/slurm/slurm.conf`, `cli_filter.lua`, `task_prolog.sh`,
`scontrol show partition`, and user QOS association. The initial experiment
used those configurations; the written policy was located and read later on
2026-10-17: `/fact_home/bohanli/CPU_EMBODIED/Fact-Lab-Compute-Rules.pdf`
(99 pages, title "Fact Lab Computing Cluster - A Practical Introduction for
New Lab Members"). Do not imply the PDF was read before the initial runs.

## Written policy verified on 2026-10-17

- Page8: `eda0` = `enterprise` = `acf3109.ece.ust.hk`.
- Page26: eda has no GPU or time limit; stated **per-user** limit is64cores /512GB.
  Blank caps in locally visible QOS records do not supersede this written limit.
- Pages27/29: lightweight EDA work using at most50% of the EDA node may run
  directly through RDP without Slurm. This is an explicit documented exception;
  the user's instruction to use Slurm still governs this project's experiments.
- Pages17/28: heavy tasks and all GPU workloads require Slurm. Large parallel
  sweeps belong on scavenger queues, with checkpoint/restart support.
- Page41: Cadence is under `/fact_data/softwares/cadence/`; the slide says the
  license is configured and instructs checking `CDS_LIC_FILE`. Our current
  interactive shell has no value for this variable; the tested shared-file
  startup failure does not prove the lab lacks a working supported license.
- Page42: the50 shared seats and exit-when-finished instruction specifically
  describe **Synopsys**, not a confirmed Cadence license-seat count.
- Pages21/22/93: datasets and weights belong in `/fact_data` or `/scratch`, not
  `/fact_home`; temporary files belong in `/scratch`, not `/tmp`. Avoid broad
  recursive directory searches and mass metadata operations on shared storage.
  Use the known PDF path directly for future rule checks. Existing experiment
  large assets in `/fact_home` predate this policy read and need a planned move
  with path/provenance preservation before further large-data expansion.

The PDF text extraction is `/scratch/bohanli/factlab_rules_read_20261017.txt`.

## Project execution convention and historical records

All training, inference, physical evaluation, and substantial build work must
run through `sbatch`/`srun`, with CPU, RAM, time and GPU requests explicit.
Use dev for short GPU development (maximum 12h), eda for EDA backend work,
or matching cpu-scavenger/gpu-scavenger partition+QOS for preemptible jobs.
Honor Slurm CUDA_VISIBLE_DEVICES and container containment variables; do not
override them to select unallocated GPUs. Set numerical library threads to
SLURM_CPUS_PER_TASK. Never cancel another project's jobs or touch its files.
Record job IDs, scripts, logs, exit codes and elapsed times for our jobs.

Lightweight source inspection, editing, downloads, package preparation and
scheduler queries are performed from the interactive session. Initial host
GPU query found all eight H20 GPUs busy; no inference/training was run there.

At about 02:05 HKT, `sinfo` showed idle CPU capacity on several nodes and
`sacctmgr` confirmed the user's matching cpu-scavenger association. The existing
96-label array 1784047 was changed from throttle2 to throttle4 with `scontrol`;
per-task 8-CPU/12GB requests, labels, physical parameters, and other arrays stayed
unchanged. The update returned exit1 for already-finished array children, while
`squeue` authoritatively confirmed the remaining range as `[56-95%4]`.
No work was restarted as a result of that scheduler-command warning.

Recovery1785220 was submitted to eda with8CPUs/24GiB/6h and no automatic requeue.
While still pending, its limit was reduced to2h30 and memory to14GiB after
checking the interrupted route: it reached iteration3 at90% after about36min,
with reported peak9786.64MB. Each update used a temporary hold to avoid changing
a started allocation. The eda node then had only about14.9GiB of schedulable
memory despite idle CPUs; the14GiB request retained at least46.5% margin over
the observed peak, and Slurm started it at02:31:09 HKT. This is an observed
resource estimate, not a guarantee of its future peak. The original submitted
batch settings and before/after scheduler records are retained in
`backend/assets/resume_{schedule,memory}_1785220.json`. Physical inputs and
optimization parameters were not changed.

After dependencies cleared, short CPU aggregation/relabeling/gate/plot jobs
1784481,1784485,1784119,1784156 were still pending for eda capacity. They were
moved, before startup, to cpu-scavenger with its matching QOS; their2CPUs,
8GiB,30min, commands and dependencies stayed unchanged. Before/after scheduler
records are in `logs/scheduler_migration_*.json` or the corresponding text
files. This frees the eda allocation for long physical implementation work;
these short jobs remain Slurm-scheduled and preemptible.

## Closed pilot archive storage, 2026-10-18

`graph_o2o_archive_repack_v1/manifest.json` records74 completed legacy archives
stored as a smaller durable long-window representation, with exact original
compressed bytes retained as verified scratch cache links. Every original
outcome, manifest and archive SHA remains unchanged. No physical evaluation was
rerun. Each durable `grt_intermediates.long28.tar.zst` contains the identical tar
stream; regenerating it with the pinned original zstd encoder reproduces the
original compressed file byte for byte. An independent restore test passed.

If an original archive cache is evicted, restore it under Slurm using its
`legacy_archive_repack.json` before any consumer requiring the old archive:

```bash
sbatch fql/graph_o2o_archive_repack.slurm --stage restore \
  --output /fact_data/bohanli/flowplace_fql/graph_o2o_archive_repack_v1 \
  --record /fact_data/bohanli/flowplace_fql_backend/grt_results/RUN/legacy_archive_repack.json
```

The restore verifies the durable archive, pinned encoder binary and original
archive SHA. Ordinary GRT/DRT intermediate scratch caches can separately be
rebuilt from that archive and checked against the preserved per-file inventory.

## Explicit scratch backend storage, 2026-10-18

New graph-policy experiments may explicitly request `--storage scratch` for
physical outputs under `/scratch/bohanli/flowplace_graph/physical_results`.
The controller flag is `--backend-storage scratch`; route plans use
`backend_storage: scratch`. Existing bulk defaults and completed results remain
unchanged. Scratch has no redundancy: it is not durable bulk storage, and
required final evidence must be preserved and verified before cleanup.

Storage allocation enforces the approved user quota, the8GiB reserve and refusal
to overwrite existing run names. Compression retains the complete intermediate
tar stream with per-file verification. Interrupted physical workers and their
source/binding records remain available; successful EDA is recovered rather than
repeated if only reporting or archive work was interrupted.

Actual fixed-input equivalence checks passed under Slurm:

- GRT worker1792770 / checker1792774: identical DEF, physical flow revision,
  eight threads and all seven recorded PPA/congestion scalars.
- Final-route worker1792772 / checker1792773: identical source GRT database,
  cap20, six final PPA/DRC scalars and exact macro preservation. DRC remains0.

Both checks fully decoded and verified the archives. Evidence lives in
`/scratch/bohanli/flowplace_graph/graph_o2o_storage_validation_v1/` as
`validation.json` and `full_route_validation.json`; their source/outcome hashes
are recorded. This is storage equivalence for two fixed cases, not evidence of
policy quality or general reproducibility across arbitrary layouts.

## Repeated scavenger preemption, 2026-10-18 03:54 HKT

Observed `natural-selection` repeatedly preempt the fixed-label controller
(six automatic reentries) and the same seed1 layouts before their physical
calculation finished. New `graph_o2o_backend.slurm` and
`graph_o2o_full_route.slurm` submissions exclude that node. Specific pending
controllers/reports were held briefly, given the same exclusion, then released
with their dependencies preserved. Running allocations were not interrupted;
no historical Table2 hold was released. This changes scheduling only, not the
eight-thread physical evaluator, actions, flow settings or route repair cap.
Source backups and scheduler before/after evidence are in
`graph_o2o_replication_v1/scheduler_node_exclusion_20261018_0354.json`.

The already-active320-label array1792796 received an explicit scheduler-only
throttle increase4→8 after checking available scavenger CPU capacity. Slurm
returned warnings for completed children, while authoritative `squeue`
confirmed the remaining range as `%8`. Its controller configuration remains4;
any later recovery array may therefore use4. Exact output and immutable
physical-parameter declaration are recorded in
`graph_o2o_offline_expand_v1/active_array_throttle_20261018_0356.json`.

## Enterprise capacity reservation, 2026-10-18 04:12–04:17 HKT

The new fixed80-label array1793079 received a scheduler-only throttle override
2→4 after observing366 idle scavenger CPUs. Completed children produced the
expected `scontrol` warnings; `squeue` confirmed the remaining `%4` range.
The immutable label plan and its recovery controller retain their original2.
Evidence: `graph_o2o_second_design_v1/initial_label_throttle_20261018_0410.json`.

Future generic GRT/DRT batch scripts now exclude both `natural-selection` and
`enterprise`; the same exclusion was applied to current arrays1792796,
1793079,1793161,1793123. Existing running allocations were not interrupted.
This keeps heavy scavenger sweeps away from the EDA node's written per-user
64CPU/512GiB limit, which also applies to allocations via cpu-scavenger.
Small existing route controllers may finish there. Script backups and exact
before/after scheduler records are in
`graph_o2o_second_design_policy_v1/heavy_scavenger_eda_node_exclusion_20261018.json`.
No tool, placement, physical optimization, eight-thread count or repair cap
changed. Old Table2 holds remain unchanged.

A single bp_be import feasibility DRT probe1793145 was actually PREEMPTED on
quantum after9m30 and no longer appeared in squeue. Recovery1793520 explicitly
uses partition/QOS eda, eight CPUs/24GiB,2h30 and the same source/spec/cap20.
At submission the user's enterprise allocations were only three one-CPU
controllers, so this request totals11CPUs/30GiB, below the written per-user
cap. This one diagnostic is not a large sweep. Original attempts remain
available, and unknown/interrupted work is never assigned physical PPA or
failure utility. Evidence:
`graph_o2o_second_design_v1/import_drt_probe_recovery_20261018.json`.

## Bounded EDA-node scavenger workers 2026-10-18 04:33 HKT

Four already-pending DRT workers1793559,1793694,1793691,1793718 received an
explicit `ReqNodeList=enterprise` and `ExcNodeList=natural-selection` while
retaining partition/QOS cpu-scavenger. Their original controllers, eight
threads,12GiB,2h30, physical inputs and cap20 stayed unchanged. This is a
bounded scheduling exception to the generic enterprise exclusion, following
repeated real preemptions on GPU/dev nodes; it is not a new physical sweep.

At declaration the user had21CPUs on enterprise, including two eight-CPU
same-input provenance repeats1794022/1794023 in EDA. The four scavenger workers
add at most32, making53CPUs; an authoritative running-job query subsequently
confirmed53, below the written64CPU limit. Other heavy arrays retain their
enterprise exclusion. The source snapshots, exact pending-job updates and
allocation evidence are in `graph_o2o_report_schema_repair_v1/` as
`bounded_enterprise_scavenger_migration.json` and
`enterprise_running_20261018_0434.json`. Check current allocation before any
further migration; do not treat a previous allocation snapshot as a reservation.

After both EDA provenance repeats completed, the eleven still-pending children
of seed1 round1 array1793161 were also given enterprise placement. Its existing
array throttle remains2, so this reserves at most16 additional CPUs, not88.
The running allocation at declaration was37CPUs; four DRT workers plus the
two possible GRT workers and the five small controllers therefore total53.
Other array children already running elsewhere were not interrupted. The exact
per-child scheduler changes and unchanged throttle are in
`seed1_r1_bounded_enterprise_remainder.json` in the same evidence directory.

## Seven preempted second-design labels, 2026-10-18 04:44 HKT

After the four bounded DRT workers completed, seven confirmed-pending,
previously preempted bp_be children1793079_9/13/15/17/20/21/22 received
`ReqNodeList=enterprise`, `ExcNodeList=natural-selection`. Their existing
cpu-scavenger partition/QOS, array throttle4, eight CPUs/24GiB/2h and physical
inputs stayed unchanged. Running children elsewhere were not interrupted.
Five one-CPU controllers plus at most two seed1 workers and four bp_be workers
bound enterprise use to53CPUs, below64, with memory below512GiB. Other heavy
arrays retain enterprise exclusion. Migration itself does not prove startup;
the seven children were still pending in the later scheduler snapshot.
Before/after evidence:
`graph_o2o_second_design_v1/initial_label_bounded_enterprise_requeues_20261018_0445.json`.

## Existing-label concurrency increase, 2026-10-18 05:20 HKT

After observing136 idle scavenger CPUs outside enterprise/natural-selection,
the existing independent-seed GRT arrays1794203/1794327 received a scheduler-only
throttle2→4, and the existing bp_be80-label array1793079 received4→6. Commands
warned about already-finished children; subsequent squeue output confirmed the
new pending-array throttles. Inputs, candidate counts, eight-thread physical
workers, tool configuration, repair cap and recovery-controller defaults did
not change. No running worker was restarted and no Table2 hold was released.

Only the same seven previously declared bp_be children may use enterprise;
other generic workers still exclude it. With at most six eight-CPU workers
and a conservative eight one-CPU controller allowance, the declared bound is
56CPUs, below64; current enterprise allocation at declaration was four CPUs.
Memory also stays below512GiB. Exact resource query and per-array before/after
records: `graph_o2o_replication_v1/active_array_throttles_20261018_0520.json`.

## Fixed recovery and seed2 second-round arrays 2026-10-18 05:33–05:34 HKT

The original320-label array terminated with261 actual outcomes and59 missing
requests. Its live controller submitted only those59 as1794475; successful
requests were not repeated. Increased this recovery array's throttle4→8 after
checking scavenger capacity. The seed2 second-round24-request array1794486
received2→4 after observing134 idle CPUs outside enterprise/natural-selection.
Both changes preserve the task manifests, worker eight threads/12GiB,
physical parameters and both node exclusions. No running worker was restarted
and no old Table2 hold changed. Exact scheduler/resource records are
`graph_o2o_offline_expand_v1/recovery_array_throttle_20261018_0533.json` and
`graph_o2o_replication_v1/seed2_round2_throttle_20261018_0534.json`.
