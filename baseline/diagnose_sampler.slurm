#!/bin/bash
#SBATCH --job-name=flowplace-diagnostic
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/baseline/artifacts/diagnostic-%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
.venv/bin/python baseline/diagnose_sampler.py
