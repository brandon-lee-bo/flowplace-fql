#!/bin/bash
#SBATCH --job-name=placement_version
#SBATCH --partition=dev
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --no-requeue
#SBATCH --output=logs/placement_version_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
version="${1:?First argument must be v1 or v2}"
shift
case "$version" in
  v1) module=fql.versions_whole_train ;;
  v2) module=fql.sequential_train ;;
  *) echo 'First argument must be v1 or v2' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=disabled MPLBACKEND=Agg
export TMPDIR="${PLACEMENT_TRAIN_SCRATCH:-/scratch/$USER/placement-training}/$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"
exec "${PLACEMENT_PYTHON:-.venv/bin/python}" -u -m "$module" "$@"
