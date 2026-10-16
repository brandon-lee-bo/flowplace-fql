# FQL-style terminal placement experiment

This is a minimal **fixed-circuit contextual-bandit adaptation**, not a reproduction of FQL's locomotion benchmarks. FlowPlace is a frozen, cached teacher. The critic predicts the measured terminal reward after external legalization. There is no Bellman bootstrap and no claim of generalization to unseen circuits. See `../docs/fql_integration_design.md` for design decisions and limitations.

The three independently parameterized networks are a one-step MLP actor and two critics. Coordinate-mean squared distillation and ensemble-mean Q follow [official FQL](https://github.com/seohongpark/fql/blob/e8cd16eb490332924dfa2492097219f181765933/agents/fql.py). The default source noise here is the recorded uniform FlowPlace prior. Unlike full FQL, the critic is pretrained and frozen during student optimization; this is valid for fixed terminal rewards but does not solve a multi-step MDP.

## Input contract

NPZ requires `noise[N,D]`, `teacher_action[N,D]`, `raw_action[M,D]`, `action[M,D]`, `reward[M]`, `split[N]`, `critic_split[M]`. Coordinate arrays may initially be `[N,V,2]`; they are flattened internally. All movable coordinate values must be normalized to `[-1,1]` before collection. `raw_action` is the bounded proposal submitted to the legalizer; `action` is the actual legalized realization whose reward was measured. Both are retained. Rewards are not renormalized by this code.

Split values are 0=train, 1=validation, 2=heldout. `critic_parent_index[M]` can link perturbations to their teacher noise family, and enforces matching splits; `group_id[N]` and `critic_group_id[M]` additionally reject groups spread across splits. Use explicit group splits for research. `--allow-random-split` is provided only for truly independent records; it cannot prevent family leakage on its own.

Optional `movable_mask[D]` and `fixed_action[D]` restore fixed coordinates before Q evaluation and inference; fixed positions are excluded from distillation loss. If ports are omitted entirely, all coordinates can be movable.

## Running

Run inside Slurm, with the same Python environment as FlowPlace. From the workspace root:

```sh
experiments/flowplace_fql/.venv/bin/python experiments/flowplace_fql/fql/train.py \
  --dataset /path/to/dataset.npz --output /path/to/new_run \
  --seeds 0,1,2 --alphas 1,10,100 --device cuda
```

Each seed fits the same terminal critic, warms up one actor with cached paired-noise distillation, then starts every variant from that identical warmup. `distill_only` and the FQL variants get equal additional actor steps, identical minibatch order and a reset Adam optimizer. FQL variants use `alpha * coordinate_mean_MSE - mean_Q`; BC uses MSE only. Critic and actor gradients are clipped at norm 10. Optional `--normalize-q` divides Q loss by detached mean absolute Q, matching the original implementation's option.

The script uses fixed final iterations. It does not select checkpoints or alpha from heldout rewards. Validation logs show distillation error only; downstream selection and final evaluation are separate. Hyperparameter exploration must be charged and recorded.

Output paths:

```text
run.json
seed_0/
  training.jsonl
  critic_diagnostics.json
  summary.json
  distill_only/{checkpoint.pt,proposals.npz,diagnostics.json}
  fql_alpha_1/{checkpoint.pt,proposals.npz,diagnostics.json}
  ...
summary.json
```

`proposals.npz` includes `noise`, `split`, `teacher_action`, `network_action` (unclipped), `raw_action` (clipped and fixed coordinates restored), and `predicted_q`. **Run `raw_action` through the same external legalizer and measure true reward before drawing conclusions.** No field pretends the predicted Q is an evaluated reward. Critic diagnostics contain MAE, RMSE, Spearman/Pearson correlation, and ensemble disagreement on all three explicit splits.

New arbitrary noise can be supplied to `predict.py --checkpoint ... --noise ... --output ...`. To load in another script:

```python
from fql.core import OneStepActor
checkpoint = torch.load(path, map_location=device)
actor = OneStepActor(checkpoint['dim'], checkpoint['widths'],
                     checkpoint['movable_mask'], checkpoint['fixed_action'])
actor.load_state_dict(checkpoint['actor_state'])
actor.to(device).eval()
with torch.no_grad():
    raw_action = actor(noise, clip=True)
# legalize and evaluate externally
```

## Software verification

`test_core.py` checks gradient direction, dimension-independent loss scaling, fixed coordinates, split leakage rejection, and measured critic calibration. `make_synthetic_smoke.py` creates a tiny mathematical fixture; its artifacts must never be reported as chip-placement evidence. A full tiny smoke, submitted through Slurm, is:

```sh
python experiments/flowplace_fql/fql/test_core.py
python experiments/flowplace_fql/fql/make_synthetic_smoke.py --output /tmp/fql_synthetic.npz
python experiments/flowplace_fql/fql/train.py --dataset /tmp/fql_synthetic.npz \
  --output /tmp/fql_smoke_new --seeds 0 --alphas 1 --widths 16,16 \
  --critic-steps 10 --warmup-steps 10 --actor-steps 10 --log-every 5 --device cpu
```

Training refuses non-Slurm execution by default. The bounded `--allow-local-smoke` escape applies only to tiny software checks; cluster rules still take precedence. Do not run real dataset collection or training on a login node.

Verified software run: Slurm job `1783831`, completed `0:0` in 11 seconds, using the existing read-only Argus Python environment (`torch 2.13.0+cu130`, `numpy 2.5.2`, CPU). All four correctness tests, the tiny synthetic BC/FQL training, checkpoint reload, and arbitrary-noise inference passed. Log: `../logs/fql_unit_1783831.log`. Artifacts: `/fact_data/bohanli/flowplace_fql/fql_software_smoke_1783831`. These are software checks, not a reproduction of placement performance or of the pinned FlowPlace environment.

The expanded smoke passed as Slurm job `1783893` (19 seconds, `0:0`): seven core tests including exact HPWL/finite-difference gradients and batch-independent coordinate optimization, synthetic analytic-actor/coordinate controls, latency and distillation diagnostics. Latest job `1784011` (19 seconds, `0:0`, same Argus CPU environment) additionally passed six backend-relabel tests, including missing/failed requests, source hashes, mixed stages, train-only normalization and frozen-plan completeness. All 13 tests passed, followed by the full synthetic smoke under the new deterministic training defaults. Its log is `../logs/fql_unit_1784011.log`; artifacts remain in `/fact_data/bohanli/flowplace_fql/fql_software_smoke_1784011`. No backend performance was simulated or inferred from these fixtures.

## Optional analytic controls

The current FQL implementation is unchanged. `train.py --analytic-hpwl-cache CACHE.pt --analytic-alphas 1,10,100` additionally creates `analytic_hpwl_alpha_*` variants using exactly the same actor initialization, paired teacher data, Adam reset, training steps and minibatches. Their objective is `alpha * distillation_MSE + prelegal_macro_HPWL / training_reference`. This replaces the learned Q gradient with an exact HPWL subgradient, not with a differentiable legalizer. Final proposals go through the same evaluator. The analytic loss therefore lacks the learned critic's model of legalizer displacement or failure penalties; this limitation must accompany comparisons. The reward scale defaults to `reward_scale_train_only` in the dataset's neighboring `manifest.json`, or can be explicitly supplied with `--reward-scale`.

An additional nonlearned control starts from each paired teacher layout and directly optimizes its coordinates:

```sh
python experiments/flowplace_fql/fql/coordinate_control.py \
  --dataset DATA/dataset.npz --cache CACHE.pt --training-dir TRAIN \
  --reference-checkpoint TRAIN/seed_0/distill_only/checkpoint.pt \
  --steps 100 --lr 0.01 --anchor-weights 0,1,10 --device cuda
```

It writes `TRAIN/seed_control/coordinate_hpwl_steps_100_anchor_*/proposals.npz`, discoverable by the shared root evaluator. It uses zero critic gradients: the supplied critic provides diagnostic Q estimates only. These are deterministic per-layout controls, **not three independent training seeds**. Their cost includes the teacher generation, all test-time gradient steps, and final legalization. Batched optimization timing is recorded explicitly and must not be called single-layout inference latency. No final legalization is performed inside the optimizer; shared evaluation supplies it exactly once.

Both controls use pin offsets and true hypernet bounding boxes under the same macro-cache convention as `analysis/placement_metrics.py`. They exclude standard cells, explicit fixed I/O positions and routed PPA. `analytic.py` computes the exact piecewise-linear HPWL with differentiable scatter min/max; ties use PyTorch's subgradient convention. Geometry is still checked externally.

## Actor latency and distillation diagnostics

Training now records `actor_latency.json` for every method: warmed batch=1 latency and batches=32/128 throughput, with explicit CUDA synchronization around each timed call. Inputs reside on device before timing. Clipping/fixed-coordinate restoration are included; loading, noise sampling, transfer, legalization and metric evaluation are excluded. Defaults are 10 warmup calls and 50 repeats; use `--latency-repeats 0` to skip. Do not compare this actor-only time with a teacher time that includes legalization without separately accounting for the difference.

Existing checkpoints can be measured independently:

```sh
python experiments/flowplace_fql/fql/latency.py --checkpoint CHECKPOINT \
  --noise DATA/dataset.npz --output actor_latency_new.json --device cuda
```

`diagnose_distillation.py --dataset DATA/dataset.npz --proposals PROPOSALS --output NEW.json` compares student MSE to a training-mean layout and nearest-training-noise baseline, and records output variation and unique layouts. This checks whether the cached-teacher student generalizes beyond lookup or mean collapse; it does not evaluate legality or PPA. Run all these experiment-wide commands under Slurm.

## Replacing proxy rewards with real backend labels

`relabel_backend.py` consumes the original NPZ and the JSONL request list produced by `prepare_backend_tasks.py`; it launches no backend jobs. Run this only after every predeclared request has a successful `metrics.json` and `manifest.json`:

```sh
python experiments/flowplace_fql/fql/relabel_backend.py \
  --dataset DATA/dataset.npz --labels-jsonl BACKEND_PLAN/tasks.jsonl \
  --output /fact_data/bohanli/flowplace_fql/datasets/bp_fe_grt_v1 \
  --allow-unlabeled-subset --require-row-hashes
```

Use a Slurm job/allocation. The explicit subset flag allows only unrequested rows excluded by the labeling budget. Missing or failed **requested** rows always fail, even with that flag. The tool automatically checks `selection_plan.json` beside the JSONL, or accepts `--selection-plan PATH`; removing a failed row from the requests then also fails. Duplicate rows, illegal source layouts, source dataset/action/DEF hash mismatches, mixed backend stages or mixed OpenROAD/configuration provenance fail and leave `audit.json`. Output directories cannot be reused.

All original `noise`, `teacher_action`, teacher splits and group identities remain unchanged. Only critic-indexed fields are subset; `source_row_index` preserves their original indices. The new reward is exactly `-rWL_um / mean(training rWL_um)` using retained critic rows with split 0. All three critic splits must remain nonempty. Raw prelegal actions, executed legal actions, original `proxy_reward`, original `proxy_reward_stage`, raw rWL/WNS/TNS/power/area, availability flags and hashes are retained. WNS/TNS and power are diagnostics only; they never enter this reward. `backend_labels.jsonl` retains complete request records, backend metrics and manifests. Unknown row-indexed NPZ fields require explicit `--extra-critic-keys` or `--extra-teacher-keys` to prevent accidental misalignment.

The new stage is **global_route_only**, explicitly not final detailed-route Table 2 PPA. A macro-HPWL evaluator can still inspect proposals but cannot calibrate a critic trained on global-route rWL; Q-versus-measured comparisons require matching backend labels and this training reference. If adding an analytic HPWL control to the GRT experiment, explicitly use the original proxy training normalization with `--reward-scale`, and describe HPWL as a different objective; silently normalizing macro HPWL by the GRT scale changes its relative weight.

Training defaults now enable deterministic PyTorch algorithms, configure cuBLAS before CUDA initialization, disable cuDNN benchmarking/TF32, and record these settings in `run.json`. Unsupported deterministic operations raise instead of silently switching algorithms. Reproducibility refers to the same seed, software and hardware; it does not promise identical outputs across PyTorch versions or GPU types.
