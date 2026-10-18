# FlowPlace Table 3: source audit and reproduction protocol

Audit date: 2026-10-17. Sources are the local arXiv v1 paper and the unmodified
upstream source snapshot pinned by `upstream/flowplace/SOURCE_COMMIT` to
`da4a10fc1a68c66dfa0ec31dd10f75b75e2e5547`. This is a protocol audit, not an
experimental result. No jobs were launched by this audit.

## What Table 3 actually compares

Table 3 is a **training-dataset ablation of FlowPlace**. Train FlowPlace on
random synthetic data and on mask-guided synthetic data, then evaluate the
two models on the eight ICCAD2015 designs. It is not a FlowPlace-versus-
ChipDiffusion model comparison, and one released FlowPlace checkpoint is
insufficient. Evidence: paper section 4.2, PDF p. 5, text lines 329–338;
Table 3 caption, PDF p. 6, text lines 376–384.

| Training data | rWL | WNS | TNS | rO-H | rO-V |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random synthetic (paper) | 1.75 | 1.75 | 1.88 | 1.88 | 1.88 |
| Mask-guided synthetic (paper) | 1.25 | 1.25 | 1.12 | 1.12 | 1.12 |

Each cell is the **average of per-design ranks across the two alternatives**.
For rWL and congestion, smaller raw values are better; for signed WNS/TNS,
larger values (closer to zero) are better. Better gets rank 1, worse rank 2;
average over `superblue1,3,4,5,7,10,16,18`. With no ties, the mask model wins
6/8 designs for rWL and WNS and 7/8 for the other metrics. This win-count
interpretation is conditional because the paper does not state its tie rule.
Record exact ties and, for our reconstruction, use midrank 1.5 for both.
Preserve raw precision and do not rank rounded display values.

The paper does not supply Table 3's per-design raw metrics, repetition counts,
uncertainty, training seeds, or the rule for combining multiple layout samples.
It does not provide a complete training-hyperparameter appendix. The values
below come from released code/configuration and are not all established as the
exact settings used to produce Table 3.

## Required backend and stage

Use FlowPlace macro output → **DREAMPlace 4.1.0 standard-cell placement** →
**Cadence Innovus EarlyGlobalRoute and timing/congestion reports**, consistently
for both branches. Evidence: PDF p. 4, text lines 196–210; upstream README
lines 93–103. OpenROAD full-flow evaluation belongs to Table 2 and cannot be
substituted. DREAMPlace HPWL and OpenTimer timing are useful diagnostics but
are not the Table 3 routed/timing measurements.

Authors explicitly withhold their ICCAD commercial-tool evaluation scripts
(`upstream/flowplace/README.md:103`). Therefore tool-identical reconstruction
is possible, but exact Innovus settings, RC setup, corner choices and report
definitions are not established by the public source. Keep these assumptions
explicit and frozen across both branches. Validate metric units and ensure
rWL, overflow and timing refer to the same layout and routing stage.

## Released training and inference recipe

The official entry point is `train_graph_with_test.py`, whose Hydra config is
`config_combined.yaml` (line 42), **not** `config_graph.yaml`.

| Item | Official current script/config | Evidence under `upstream/flowplace/` |
| --- | --- | --- |
| Mask training set | `mask.61`; 40,000 train + 2,000 validation | `scripts/train_flow_matching.sh:3`; `data-gen/configs/config.yaml:8` |
| Seed | 61 | `flow_matching/configs/config_combined.yaml:18` |
| Updates | 1,000,000 | `scripts/train_flow_matching.sh:7` overrides combined config |
| Train/validation batch size | 64 / 64 | `flow_matching/configs/config_combined.yaml:63` |
| Optimizer | Adam, learning rate 0.0003 | `flow_matching/train_graph_with_test.py:150`; combined config line 82 |
| Model | att-gnn, GAT configuration, large widths | combined config lines 3–5; `configs/model/size/large.yaml` |
| Prior | uniform during training and evaluation | training script line 8; evaluation script line 6 |
| Augmentation | false | combined config line 83 |
| Loss/checkpoint interval | 100,000 updates | training script lines 4–5 |
| Test interval | 50,000 updates, 8 ICCAD examples | training script lines 6, 9–10 |
| Final sampling | 20 steps, hard constraints enabled, `w_reg=0.1`, `w_dist=1.0` | `scripts/test_flow_matching.sh:7` |
| Evaluation seed | 300 | `flow_matching/configs/config_eval.yaml:16` |

**Batch semantics matter.** `GraphDataLoader.get_batch` randomly selects one
graph and replicates its target layout 64 times; different noise and flow
times are then sampled for these copies. It is not a batch of 64 distinct
circuits (`flow_matching/utils.py:924`; `models.py:156`). The loss samples
continuous uniform time internally, ignoring the discrete time argument
constructed by the outer training loop.

`best.ckpt` is selected by the loss of one randomly selected validation graph
expanded to a batch, evaluated at `print_every`; it is not selected by ICCAD
PPA. See training lines 200–225, `utils.py:78`, and `utils.py:929`. The test
report consumes random numbers but is not the checkpoint criterion. Changing
validation/test/display frequency can change later random draws. The training
code resets `best_loss` on resume; a faithful resumable wrapper should preserve
the historical best and RNG state, with any corrections logged.

`num_output_samples=8` in the test script means loop over the **eight graph
entries**, one generation each; it does not mean eight samples per design
(`flow_matching/eval.py:200`). A new design-specific seed policy is an explicit
reconstruction choice, not the original global-RNG evaluation stream.

## Random-data branch: what is known and what is missing

`data-gen/configs/versions/v1.yaml` is a plausible public random-data recipe:
it selects `algorithm: v2`, whose implementation places random non-overlapping
rectangles by rejection sampling and then generates pin/net connections
(`data-gen/algorithms.py:235`). Despite their names, **config `v1` selects class
V2**; config `v2` increases maximum instance count to 1,600 and changes geometry
scales, so these should not be interchanged.

The mask recipe selects `algorithm: mask`, uses a 128×128 grid and boundary
weighting (`versions/mask.yaml:7`). Both v1 and mask cap candidate instances at
400 and share size, terminal and edge distributions, but their configured
target densities differ: random v1 is 0.75–0.90, mask is 0.65–0.80. Also the
random implementation adds attempted rectangle area even when all rejection
attempts fail (`algorithms.py:250`); mask adds area only after successful
placement (`mask_guided_data_generation.py:210`). Thus the comparison of the
published recipes is a dataset-family comparison, not a pure isolated toggle
of the boundary score. Preserve these settings for a reconstruction and report
actual generated macro count/density/edge statistics; do not silently fix the
random generator or equalize density while claiming the released recipe.

Neither Table 3 nor the README identifies which `versions/*.yaml` produced
the exact random dataset used for that table. Match to the original
ChipDiffusion dataset or obtain author confirmation before calling v1 exact.
If unavailable, train both current released recipes with the same architecture,
prior, step budget, validation rule and backend, and label the result a
documented reconstruction of Table 3's comparison.

Data generation currently uses CPU multiprocessing (`generate_parallel.py:22,
70`), despite the paper's description of GPU-parallel mask computation. Cap
workers to the Slurm allocation; do not run the script's unrestricted `nproc`
default on a login node. Put data, checkpoints and temporary artifacts in
FactLab-approved `/fact_data` or `/scratch` paths.

## Released checkpoint provenance caveat

The local release manifest contains the mask-named checkpoint archive and
benchmark cache (`baseline/assets/manifest.json:7`). The archived model's
`config.yaml` inside
`baseline/runtime/logs/debug/mask.61.train_mask_flowplace_uniform.61/` states:

- `task: v3.61`, `test_task: ibm.cluster512.v1`, although the method/directory
  name says mask;
- 1,000,000 steps, seed 61, batch sizes 64, uniform prior, 40,000/2,000 examples;
- print/evaluation every 1,000 and test every 2,000, unlike today's script.

These are real provenance inconsistencies, not evidence that the checkpoint
is wrong. The released checkpoint is useful for backend verification and
reproducing released inference, but it cannot establish a controlled two-branch
training-data ablation against a newly trained random model by itself. Preserve
the archive hash and do not rewrite its metadata to match the README.

## Execution order and completion criteria

1. Validate the shared DREAMPlace → Innovus path on an unchanged released
   superblue18 DEF, retaining hashes, import warnings, placement invariants,
   EGR output and timing units. This isolates backend discrepancies before
   retraining. Inspect all error/constraint reports, not just process exit code.
2. Freeze the two dataset recipes and explicit unknowns. Verify counts,
   geometry/net statistics and train/validation split; pilot resource use.
3. Train both branches under the same declared protocol, saving resolved
   configs, source/data hashes, RNG state, training step and checkpoint choice.
   Released weights remain a separate reference track.
4. Generate both branches for all eight circuits under one declared seed and
   inference-selection policy, retaining legal and failed outcomes alike.
   Use identical 512-instance selection/mapping and legalizer settings.
5. Run the same backend for all 16 layouts (more if a predeclared multi-seed
   analysis is used). Keep missing or failed metrics explicit; do not compute
   an apparently complete eight-design rank from a favorable subset.
6. Publish the 16 raw metric rows, per-design ranks and final averages alongside
   paper ranks and protocol deviations. Report whether the qualitative dataset
   advantage reproduced; do not tune the backend to hit paper numbers.

Repeated seeds are useful supplementary robustness tests, but the paper does
not specify them. Keep supplementary seed statistics separate from the primary
declared Table 3 reconstruction. Flow-Q changes belong to a later experiment,
after this baseline is stable.
