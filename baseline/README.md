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
