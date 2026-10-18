#!/bin/bash
#SBATCH --job-name=flowplace_table2_dmp
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_dmp_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="/scratch/bohanli/flowplace_table2/dreamplace_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_install_cpu"
exec "$ROOT/iccad/assets/dmp_env/bin/python" backend/scripts/run_dreamplace_table2.py "$@"
