#!/bin/bash
#SBATCH --job-name=qstudy_controller_resume
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise,blue-space
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/qstudy_controller_resume_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBATCH_EXCLUDE=natural-selection,enterprise,blue-space
# The controller's restart count must never become a new child's count.
unset SLURM_RESTART_COUNT
exec .venv/bin/python -u -m fql.qstudy_controller "$@"
