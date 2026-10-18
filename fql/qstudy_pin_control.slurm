#!/bin/bash
#SBATCH --job-name=qstudy_pin_control
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_pin_%A_%a.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled MPLBACKEND=Agg
export TMPDIR="/scratch/bohanli/flowplace_graph/qstudy_pin/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
task_index="${SLURM_ARRAY_TASK_ID:?array index required}"
if (( task_index < 0 || task_index > 5 )); then exit 2; fi
task_seed="$((task_index % 3))"
task_dataset=bp_fe384
if (( task_index >= 3 )); then task_dataset=bp_be64; fi
exec .venv/bin/python -u -m fql.qstudy_pin_control \
  --dataset "/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/$task_dataset/dataset.npz" \
  --output "/fact_data/bohanli/flowplace_fql/qstudy_v1/pin_control_v1/$task_dataset/pin_huber/seed_$task_seed" \
  --seed "$task_seed" "$@"
