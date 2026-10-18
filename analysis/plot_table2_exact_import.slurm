#!/bin/bash
#SBATCH --job-name=table2_exact_plots
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_exact_plots_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
: "${1:?Summary JSON}"
: "${2:?Summary SHA256}"
: "${3:?New output directory}"
: "${4:?Plot runner SHA256}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
RUNNER=$ROOT/analysis/plot_table2_exact_import.py
test "$(sha256sum "$RUNNER" | cut -d ' ' -f1)" = "$4"
export MPLCONFIGDIR=/scratch/bohanli/flowplace_table2/mpl_${SLURM_JOB_ID}
export TMPDIR=/scratch/bohanli/flowplace_table2/plot_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
exec python3 "$RUNNER" --summary "$1" --summary-sha256 "$2" --output "$3"
