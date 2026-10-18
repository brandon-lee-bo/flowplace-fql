#!/bin/bash
#SBATCH --job-name=table2_finish_reports
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_finish_reports_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
: "${1:?Runner SHA256}"
RUNNER=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/analysis/finish_table2_reports.py
test "$(sha256sum "$RUNNER" | cut -d ' ' -f1)" = "$1"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec python3 "$RUNNER"
