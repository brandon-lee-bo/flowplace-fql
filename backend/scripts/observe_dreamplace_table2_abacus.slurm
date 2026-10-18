#!/bin/bash
# Observation-only reproduction of the stock Abacus assertion.
#SBATCH --job-name=table2-dmp-abacus-observe
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_dmp_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
: "${SLURM_CPUS_PER_TASK:?Missing allocated CPU count}"
: "${DMP_GPU_BUILD:?Set to the successful audited GPU build directory}"
test -f "$DMP_GPU_BUILD/BUILD_AND_OPERATOR_TESTS_PASSED"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="/scratch/bohanli/flowplace_table2/dreamplace_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$DMP_GPU_BUILD/install:$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_env/lib/python3.9/site-packages:/usr/lib64/python3.9/site-packages:/usr/lib/python3.9/site-packages"
export CUDA_HOME="${DMP_GPU_CUDA_ROOT:-/usr/local/cuda-12.8}"
exec "$DMP_GPU_BUILD/env/bin/python" backend/scripts/observe_dreamplace_table2_abacus.py --gpu --install "$DMP_GPU_BUILD/install" "$@"
