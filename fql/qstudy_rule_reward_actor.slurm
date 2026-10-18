#!/bin/bash
#SBATCH --job-name=qstudy_rule_actor
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --exclude=natural-selection,enterprise,blue-space
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_rule_actor_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=disabled MPLBACKEND=Agg
export TMPDIR="/scratch/bohanli/flowplace_graph/qstudy_rule_actor_tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -m fql.qstudy_rule_reward_actor "$@"
