#!/bin/bash
#SBATCH --job-name=table2_exact_report
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_exact_report_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
: "${1:?Report runner SHA256}"
RUNNER=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/analysis/write_table2_exact_import_report_capaware.py
test "$(sha256sum "$RUNNER" | cut -d ' ' -f1)" = "$1"
shift
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 "$RUNNER" "$@"
