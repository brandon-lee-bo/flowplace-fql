#!/bin/bash
#SBATCH --job-name=fql-macro-eval
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/proxy_eval_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
data_dir=datasets/bp_fe_macro_deterministic_v1
training_dir=runs/bp_fe_macro_deterministic_v1
.venv/bin/python fql/coordinate_control.py --dataset "$data_dir/dataset.npz" \
 --cache baseline/runtime/benchmark_cache/bp_fe.pt \
 --reference-checkpoint "$training_dir/seed_0/distill_only/checkpoint.pt" \
 --training-dir "$training_dir" --threads 4 --device cuda
.venv/bin/python fql/diagnose_distillation.py --dataset "$data_dir/dataset.npz" \
 --proposals "$training_dir/seed_0/distill_only/proposals.npz" --output "$training_dir/distillation_diagnostic.json"
.venv/bin/python evaluate_candidates.py --dataset-dir "$data_dir" --training-dir "$training_dir" --phase validation
.venv/bin/python select_methods.py --training-dir "$training_dir"
.venv/bin/python evaluate_candidates.py --dataset-dir "$data_dir" --training-dir "$training_dir" \
 --phase heldout --selection "$training_dir/selection.json"
