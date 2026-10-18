#!/bin/bash
#SBATCH --job-name=flowplace-iccad-dmp
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/dmp_%j.log
set -euo pipefail
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_install_cpu"
exec "$ROOT/iccad/assets/dmp_env/bin/python" iccad/scripts/run_dreamplace.py \
  --run-id "released_superblue18_cpu_job${SLURM_JOB_ID}" "$@"
