#!/bin/bash
#SBATCH --job-name=flowplace_fql_unit
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:05:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/fql_unit_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
FQL_TEST_PYTHON=${FQL_TEST_PYTHON:-/fact_home/bohanli/.conda/envs/argus/bin/python}
FQL_TEST_OUTPUT=/fact_data/bohanli/flowplace_fql/fql_software_smoke_${SLURM_JOB_ID}
mkdir -p "$FQL_TEST_OUTPUT"
"$FQL_TEST_PYTHON" -c 'import torch,numpy; print("SOFTWARE TEST ONLY", torch.__version__, numpy.__version__)'
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/test_core.py
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/test_relabel_backend.py
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/make_synthetic_smoke.py --output "$FQL_TEST_OUTPUT/dataset.npz" --cache-output "$FQL_TEST_OUTPUT/toy_cache.pt"
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/train.py \
  --dataset "$FQL_TEST_OUTPUT/dataset.npz" --output "$FQL_TEST_OUTPUT/training" \
  --seeds 0 --alphas 1 --widths 16,16 --critic-steps 10 --warmup-steps 10 \
  --actor-steps 10 --log-every 5 --device cpu \
  --analytic-hpwl-cache "$FQL_TEST_OUTPUT/toy_cache.pt" --analytic-alphas 1 --reward-scale 20 \
  --latency-warmup 1 --latency-repeats 2 --latency-batches 1,2
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/predict.py \
  --checkpoint "$FQL_TEST_OUTPUT/training/seed_0/fql_alpha_1/checkpoint.pt" \
  --noise "$FQL_TEST_OUTPUT/dataset.npz" --output "$FQL_TEST_OUTPUT/arbitrary_noise_predictions.npz"
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/coordinate_control.py \
  --dataset "$FQL_TEST_OUTPUT/dataset.npz" --cache "$FQL_TEST_OUTPUT/toy_cache.pt" \
  --reference-checkpoint "$FQL_TEST_OUTPUT/training/seed_0/distill_only/checkpoint.pt" \
  --training-dir "$FQL_TEST_OUTPUT/training" --reward-scale 20 --steps 5 --anchor-weights 0,1
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/latency.py \
  --checkpoint "$FQL_TEST_OUTPUT/training/seed_0/distill_only/checkpoint.pt" \
  --noise "$FQL_TEST_OUTPUT/dataset.npz" --output "$FQL_TEST_OUTPUT/standalone_latency.json" \
  --warmup 1 --repeats 2 --batch-sizes 1,2
"$FQL_TEST_PYTHON" experiments/flowplace_fql/fql/diagnose_distillation.py \
  --dataset "$FQL_TEST_OUTPUT/dataset.npz" \
  --proposals "$FQL_TEST_OUTPUT/training/seed_0/distill_only/proposals.npz" \
  --output "$FQL_TEST_OUTPUT/distillation_diagnostics.json"
