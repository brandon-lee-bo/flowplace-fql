# Placement versions: parameters and reproduction

These are fixed-design research prototypes, not a packaged cross-circuit placer.
Start with [the measured results and limitations](placement_versions.md).

## Data and environment

Restore the pinned upstream revisions, public checkpoint and graph cache using
[the root README](../README.md), [baseline instructions](../baseline/README.md)
and [backend instructions](../backend/README.md). Install the environment from
`requirements.txt`, including the compatible PyG extensions. A Git checkout
does not contain model weights, datasets, EDA tools or physical artifacts.

V1's matched entrypoint is specifically for `bp_fe` and reads these fixed paths
under the repository root:

- `datasets/bp_fe_grt_v1/dataset.npz` and its strict manifest: 64 TRAIN and 16
  validation layouts, with group-level separation. A further 16 held-out layouts
  are excluded from training.
- `datasets/bp_fe_macro_deterministic_v1/dataset.npz`: additional unlabeled TRAIN
  behavior samples used by the V1.0 flow warmup only.
- `baseline/runtime/benchmark_cache/bp_fe.pt` and the public checkpoint at the
  location defined in `baseline/api.py`.

V2 uses the same measured-layout dataset, `backend_labels.jsonl` and the
label-bound DEF, metrics and manifest files. Its loader verifies their SHA256,
split, design and exact executed actions before constructing trajectories.
Relocating files without restoring their declared provenance paths is not
enough. Rebuild a separately bound dataset for a new location; do not patch an
old manifest or reuse an old run identity.

The offline reward is always `−GRT rWL / 2341121.25` for these matched pilots.
It is a measured metric transformed into a reward. Q is the learned prediction
of that reward (or return), not an independently measured "Q metric". GRT is
global routing, before detailed routing; its power, timing, area and congestion
numbers are stage-specific. Standard-cell area is not die area. Overflow sum
is not an overflow ratio. WNS/TNS improve toward zero, while rWL/power/area and
overflow improve downward. These experiments do not evaluate PDN/IR-drop quality.

## Actual training budgets

| Parameter | V1.0 | V1.1 | V2_4 |
| --- | ---: | ---: | ---: |
| Flow sampling steps per complete target | 10 | 20 | 20 |
| Flow weight updates | 300 before freezing | 0 | 0 |
| Common actor BC updates | 800 | 800 | 1000 |
| Common Q updates | 2000 terminal regression | 2000 terminal regression | 1000 recorded-return MC |
| Additional actor/Q updates per BC/FQL arm | 1000 each | 1000 each | 1000 each, Bellman Q |
| Actor and branch-Q batch | 16 | 16 | 2 |
| Common Q batch | 64 (hardcoded) | 64 (hardcoded) | 2 |
| Common actor alpha | 1 (hardcoded) | 1 (hardcoded) | 10 |
| Branch alpha / FQL Q weight | 10 / 1 | 10 / 1 | 10 / 1 |
| Matched BC Q weight | 0 | 0 | 0 |
| Actor optimizer / learning rate | Adam / 1e−5 | Adam / 1e−5 | Adam / 1e−5 |
| Q optimizer / learning rate / weight decay | AdamW / 3e−4 / 1e−4 | Same | Same |
| Gradient norm clip | 1 | 1 | 1 |
| Q heads / aggregation | 2 / mean | 2 / mean | 2 / mean |
| Q hidden width | 128 | 128 | 128 |
| Gamma / target tau | Terminal; no bootstrap | Terminal; no bootstrap | 1 / .005 |

Noise is uniform on `[-1,1]`, matching this FlowPlace setup. This is not a
Gaussian-policy experiment. Each teacher target is freshly sampled with the
same noise supplied to its student; teacher gradients are disabled. V1 uses
`alpha * whole-layout MSE − mean Q`. V2 distills the unbounded current-macro
prediction, and queries Q with its `[-1,1]`-clamped proposal. Projection is an
environment operation, outside that actor loss.

An update is one minibatch optimizer step. It is neither an ODE sampling step,
an environment decision nor a new backend measurement. Repeated samples and
two Q heads do not increase the number of independent labels.

One final V1 actor inherits 1800 updates × batch16 = 28,800 complete-layout
training targets, or 633,600 scalar coordinate supervision terms (11×2 per
target). One final V2 actor inherits 2000 updates × batch2 = 4000 current-macro
targets, or 8000 scalar coordinate terms. Both losses average their terms, so
these counts are not loss/gradient multipliers or independent information counts.

Across both arms, count the shared common stage once: V1 needs 2800 actor
updates and 44,800 targets; V2 needs 3000 updates and 6000 targets. Euler10
samples the V1.0 batch together; the public20 and conditional20 wrappers loop
over individual samples. A network-call count alone cannot establish equal
compute or wall-clock speed.

V1.0 online R1 adds eight measured collection layouts per arm and 600 actor/Q
updates, batch16 with eight original and eight online draws. R2 retains the
cumulative online replay. Evaluation noises/layouts do not enter replay. The
matched teacher stays frozen. The published V2 entrypoint trains offline and
exports rollouts; it does not implement sequential online fine-tuning.

## Fresh training entrypoints

The three [example configurations](../examples/placement_versions) reproduce
the declared numerical budgets. They omit historical source bindings and
template paths, use a relative dataset path, and correct the old V1.0 template's
misleading public20 description. They are **fresh-run templates**, not exact
historical config files. Choose a new output directory and unique run prefix.
No training starts merely by copying a config.

Submit from the repository root. The portable entrypoint uses the submit
directory and supports `PLACEMENT_PYTHON` and `PLACEMENT_TRAIN_SCRATCH` overrides.
Partition/GPU requests are FactLab-specific and must match the target cluster.

```bash
mkdir -p logs runs/my_v11
cp examples/placement_versions/v11_matched.json runs/my_v11/config.json
# Set run_id_prefix in the copied config to a new, unique identifier.
sbatch tools/train_placement_version.slurm v1 \
  --experiment "$PWD/runs/my_v11" --seed 0 --round 0
```

For V1.0 use `v10_matched.json` with the same entrypoint. For V2:

```bash
sbatch tools/train_placement_version.slurm v2 train \
  --config examples/placement_versions/v2_sequential.json \
  --output "$PWD/runs/my_v2" --seed 0
```

V1 training exports requests after the training stages. V2 uses a separate
`export` command on the completed checkpoint; its original execution padding
remains zero:

```bash
sbatch tools/train_placement_version.slurm v2 export \
  --checkpoint "$PWD/runs/my_v2/seed_0/fql/checkpoint.pt" \
  --output "$PWD/runs/my_v2_fql_export" --count 8 --noise-seed 3900000
```

Neither command executes OpenROAD or manufactures feedback. Run the separately
configured native backend using the bound request
and common physical protocol. The historical backend, online-control and
recovery orchestration depends on server-local evidence; this publication does
not turn those controllers into a turnkey public workflow. Existing source-bound
`.slurm` files retain their historical paths/bytes; prefer the portable entrypoint
above for new runs. Resume old runs only from their matching archived source.

## Verification

After restoring pinned upstream sources, fixture tests need the Python
environment but not the large experiment dataset. Opt-in real sampler tests
also need the public checkpoint and `bp_fe` graph cache.

```bash
mkdir -p logs
sbatch tools/verify_placement_versions.slurm
sbatch --export=ALL,FQL_REAL_CPU_SAMPLER=1,FQL_REAL_SEQUENTIAL_TEACHER=1 \
  tools/verify_placement_versions.slurm
```

Publication validation: CPU Slurm **1849715**, 2 CPUs / 8 GiB, completed in
34 seconds with exit 0. **All 57 tests passed**, including both actual-weight
sampler checks; none were skipped. Contracts cover exact public-sampler output,
fixed prefixes, hidden future-coordinate exclusion, actor/Q updates, frozen
teacher weights, true terminal masks, MC versus Bellman updates, split leakage,
checkpoint/source identity and exact optimizer/RNG/target recovery. This is code
correctness evidence, not another performance experiment.

The [numerical snapshot](data/placement_versions_20261023.json) can be regenerated
by a read-only JSON extraction when the saved source reports are available:

```bash
python -m tools.export_placement_snapshot \
  --evidence-root /path/to/versions_goal_20261022 \
  --output /path/to/new_snapshot.json
```

The exporter records source-report digests; it does not replace the original
physical audit, fetch missing artifacts, or reconstruct unavailable measurements.
