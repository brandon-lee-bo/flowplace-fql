#!/bin/bash
#SBATCH --job-name=flowplace-dmp-audit
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/dmp_audit_%j.log
set -euo pipefail
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_install_cpu"
exec "$ROOT/iccad/assets/dmp_env/bin/python" iccad/scripts/check_dmp_output.py "$@"
