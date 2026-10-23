#!/bin/bash
#SBATCH --job-name=placement_versions_tests
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --exclude=natural-selection,enterprise,blue-space,infinite-frontier
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/placement_versions_tests_%j.log
set -euo pipefail
# Submit from the repository root after restoring the pinned upstream checkout.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="${PLACEMENT_TEST_SCRATCH:-/scratch/$USER/placement-tests}/$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"
exec "${PLACEMENT_PYTHON:-.venv/bin/python}" -u -m unittest -v \
  fql.test_released_flowplace_teacher fql.test_qstudy_actor \
  fql.test_sequential_env fql.test_sequential_models \
  fql.test_sequential_teacher fql.test_sequential_train
