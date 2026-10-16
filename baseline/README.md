# FlowPlace pretrained baseline adapter

Source: official `lamda-bbo/flowplace`, commit pinned in `../upstream/flowplace/SOURCE_COMMIT`; release assets v1.0.0, URLs and SHA-256 in `assets/manifest.json`. Upstream source is unmodified.

The runner builds the OpenROAD graph using the official `PlaceDBInstanceNangate45`, applies the same port-mask flattening and removal of already-normalized chip size as the official validation loader, strictly loads the checkpoint, and calls the upstream `reverse_samples` sampler. Defaults match `scripts/test_flow_matching.sh`: uniform prior, 20 steps, hard constraints, regularity weight 0.1 and distance weight 1.0. Each run saves explicit initial noise, normalized centers, full sampling trajectory, macro-ID keyed lower-left coordinates, and configuration.

```bash
sbatch baseline/run_pretrained.slurm --designs bp_fe --seeds 300 --output baseline/artifacts/pretrained_original
```

All substantial generation and training must use a Slurm allocation with declared GPU, CPUs and memory. Login-node work is limited to source inspection, downloads and lightweight setup.

## Reusable interface

`api.py`: `load_problem`, `load_model`, `generate(model, problem, noise=...)`, `legalize(raw_centers, problem)`, `legality`, `macro_net_hpwl`, and `save_sample`.

An action has shape `[number_of_macros, 2]`, containing macro centers normalized to the rectangular canvas `[-1,1] x [-1,1]`. The fixed circuit graph and sizes are the context. Explicit noise can be supplied for paired teacher/student evaluation. FQL-generated raw actions should be evaluated both before and after the same legalizer; legality is independently checked, not assumed from legalizer success.

## Coordinate exports

- `sample.pt`: normalized centers, initial noise, sizes, names.
- `macro_physical.json`: `[lower_left_x, lower_left_y, width, height]` in benchmark-cache coordinate units, relative to the canvas. The backend must verify the DREAMPlace scaling/shift against DEF units and die/core origin.
- `<design>.pt`: exact official `convert_to_openroad` grid representation, including subtracting one grid unit. The upstream DREAMPlace apply routine reverses that offset, adds optional halo, rounds and unscales.
- `metrics.json`: strict geometric overlap/boundary checks plus macro-net HPWL in cache units. This is NOT routed wirelength and excludes standard-cell and port nets absent from the released macro graph.

## Explicit deviations and pitfalls

1. Strict checkpoint loading replaces the upstream helper that silently catches `load_state_dict` errors. No random-weight fallback is permitted.
2. The direct wrapper avoids GUI plotting and unrelated downstream conversion. Generation timings include legalizer setup and JIT/compile overhead and are labeled accordingly.
3. Seeds are individually set per design, with noise saved for exact reuse. This differs from consuming one global RNG stream through all six designs and must not be treated as the paper's exact random sample.
4. The released sampling update interpolates using current `x`, while the paper's stated interpolation may suggest original noise. We preserve released code for the baseline; no silent paper-correcting patch.
5. The legalizer computes an area-sorted order but uses the unsorted movable sequence. If no grid candidate is feasible, it retains the original, possibly illegal position. Every result is independently checked.
6. The source flash-attention implementation disables the math fallback. GPU execution is needed unless an explicit compatibility patch is recorded. `--disable-compile` disables only the wrapper's `torch.compile` invocation, leaving eager tensor operations unchanged, when compilation is unsupported or excessively costly.
7. The macro net pin offsets are relative to macro centers, verified in `benchmark_utils/openroad/problem_instance.py` where node size / 2 is subtracted.

8. Released `node_id_to_name` values are integer DREAMPlace node IDs, not human-readable macro names. For bp_fe they match the released DEF component ordinal (33188 through 33198). Backend conversion must verify this mapping and LEF sizes for every circuit. bp_fe cache units are 0.19 microns with core-row origin (5.13, 5.60) microns, inferred from DEF rows and 4157-site width; backend validates this before use.

## Observed reproducibility findings (2026-10-17)

The original GPU sampler is **not a deterministic function of the saved noise alone** with default Torch settings on the tested NVIDIA L20 / Torch 2.2.1 environment. A 20-step repeated run with the same model object and input noise changed final coordinates by up to 1.59 normalized units; a repeated neural forward differed by 0.0009765625. Small prediction differences can change discrete legalizer choices and then propagate through later flow steps. Both compiled and eager runs showed this variability. A fixed legalizer input, in contrast, gave exactly equal compiled/eager/repeated results in the diagnostic.

Artifacts:

- `artifacts/adapter_validation_failed.json` and `smoke-1783946.log`: initial failed cross-run comparison, retained rather than overwritten.
- `artifacts/sampler_diagnostic/report.json`, job 1783987: same-model repeated compiled/eager trials, full trajectories, fixed-input legalizer isolation.
- `artifacts/deterministic_diagnostic/report.json`, job 1783995: `CUBLAS_WORKSPACE_CONFIG=:4096:8` plus `torch.use_deterministic_algorithms(True)` yielded exact compiled-repeat and compiled/eager equality for seeds 300, 301 and 302. All nine placements were independently legal. This is evidence for these tested inputs/platform, not a universal hardware equivalence claim.

Use the explicit `--deterministic` flag and export the CuBLAS setting before the Python process starts for reproducible scientific comparisons. Preserve the original-mode generation separately as the released-inference attempt. Runtime increases from roughly 0.36 seconds to 0.66 seconds per bp_fe generation in this environment after warmup; first-run setup is separately recorded. No algorithmic update to the released sampler was made.

`artifacts/official_stream_seed300/summary.json` contains successful original-mode generation of all six OpenROAD circuits with zero independent overlap/boundary violations. The `official_stream_deterministic_seed300` counterpart records the explicit deterministic-runtime variant. These are **macro-generation** results, not Table 2 backend PPA; the backend experiment owns DEF conversion, global/detailed route and PPA reporting.

## Experimental Euler BC learnability diagnostic

`diagnose_euler_bc.py` and `diagnose_euler_bc.slurm` implement a bounded, exploratory comparison on the same 288 original noises and 192/48/48 family split. The experimental target is the **raw standard-Euler teacher output clipped to [-1,1]**; the original target was the **official hard-projected output**. These target semantics differ. Both use an MLP with three 256-wide hidden layers, seeds 0/1/2, 1,000 warmup steps followed by a fresh Adam for 2,000 BC steps, batch size 128, learning rate 3e-4, gradient cap 10, and the original per-seed NumPy batch RNG offsets 20000/30000. There is no Q training, alpha selection, or backend evaluation. Original hard BC predictions are reused without retraining.

Completed artifact: `artifacts/euler_bc_learnability_v2/`, Slurm **1784188**, NVIDIA L20, elapsed allocation 3m14s. The first 16 Euler outputs exactly match the previous bounded sampler diagnostic. All original hard BC validation/heldout final coordinates match the saved evaluator outputs exactly. The initial job 1784185 failed in three seconds before generation/training because the original dataset omits all-movable mask arrays; v2 mirrors `PlacementData.load` defaults and independently verifies them against the graph. The failed v1 startup snapshot and logs remain intact.

For the 48 previously inspected heldout noises, all three seeds give:

| Target / predictor | Unclipped network MSE | MSE / TRAIN-mean predictor error on these noises |
|---|---:|---:|
| Original hard BC, seeds 0/1/2 | 0.532984 / 0.505501 / 0.546878 | 1.74781 / 1.65769 / 1.79337 |
| Experimental Euler raw BC, seeds 0/1/2 | 0.035882 / 0.038050 / 0.037113 | 0.16154 / 0.17130 / 0.16709 |
| Euler `noise + mean_TRAIN(target - noise)` | 0.045247 | 0.20371 |

`raw_error` measures the unbounded network output against the specified target; `clipped_error` measures the actual clipped proposal. The earlier `diagnose_distillation.py` uses clipped proposals. Its hard seed-0 MSE **0.48722043765** is reproduced exactly here; the corresponding unbounded MSE **0.53298384271** is a different measurement of the same checkpoint, not a warmup/final-checkpoint mismatch. Both error definitions and normalization by the fixed TRAIN target variance are saved.

After one identical final legalizer, the Euler teacher has mean macro HPWL **71,430.18**, while Euler BC has seed-mean **70,857.70**, with standard deviation **745.69 across the three seed means**. Each teacher/student heldout set is 48/48 geometrically legal with 48 distinct layouts. Final across-noise coordinate standard-deviation RMS is **0.52844** for the teacher and **0.50947 / 0.50682 / 0.50412** for the students. The identity-plus-TRAIN-residual baseline gives macro HPWL **75,143.76**. These are macro-only proxy measurements, not PPA. In particular, the fixed Euler TRAIN-mean layout has still lower HPWL **48,497.07**, but is exactly one unique layout: its repeated rows only align prediction errors to the noise list and are not independent measurements.

The result supports improved empirical learnability of this raw Euler target under the matched BC budget; it does not prove a superior FlowPlace sampler, better FQL, or better physical implementation. Training MSE remains much lower than validation/heldout MSE. The same heldout split had already informed earlier exploration, so these comparisons are explicitly exploratory.

The artifact includes all teacher unclipped/clipped outputs and trajectories, student checkpoints and initial states, geometry/HPWL/diversity for each split, exact original-output hashes, and PNG/PDF figures. `source_snapshot/` contains startup copies of all local executed Python dependencies and upstream Python/config files. The separately copied `fql/train.py` comparison reference changed during the run; the diagnostic never imports or executes it, and its startup copy remains available. `interpretation_notes.json` records this and the fixed-control/error-definition caveats without changing the original result report.
