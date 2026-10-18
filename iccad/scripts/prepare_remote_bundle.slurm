#!/bin/bash
#SBATCH --job-name=flowplace-scut-bundle
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/scut_bundle_%j.log
set -euo pipefail
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="/scratch/bohanli/flowplace_iccad/bundle_tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT
python3 iccad/scripts/prepare_remote_bundle.py "$@"
