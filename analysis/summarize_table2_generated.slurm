#!/bin/bash
# Submit with --dependency=afterany:<six-design-array-job> and frozen CLI inputs.
#SBATCH --job-name=table2_summary
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/analysis/table2_summary_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="/scratch/bohanli/table2_summary_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$ROOT"
exec python3 analysis/summarize_table2_generated.py "$@"
