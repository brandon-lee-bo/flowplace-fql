#!/bin/bash
# Requires an independently validated build; does not download or compile.
#SBATCH --job-name=table2-dmp410-gpu
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_dmp_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
: "${SLURM_CPUS_PER_TASK:?Missing allocated CPU count}"
: "${DMP_GPU_BUILD:?Set to the successful audited GPU build directory}"
test -f "$DMP_GPU_BUILD/BUILD_AND_OPERATOR_TESTS_PASSED"
test -x "$DMP_GPU_BUILD/env/bin/python"
test -f "$DMP_GPU_BUILD/install/dreamplace/configure.py"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="/scratch/bohanli/flowplace_table2/dreamplace_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export PYTHONNOUSERSITE=1
# Match the already-working CPU environment's explicit system dependency
# fallback (not user-site). cffi is installed there and imported by drawing
# modules even though plot_flag=0. Pinned package directories stay first.
export PYTHONPATH="$DMP_GPU_BUILD/install:$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_env/lib/python3.9/site-packages:/usr/lib64/python3.9/site-packages:/usr/lib/python3.9/site-packages"
export CUDA_HOME="${DMP_GPU_CUDA_ROOT:-/usr/local/cuda-12.8}"
exec "$DMP_GPU_BUILD/env/bin/python" backend/scripts/run_dreamplace_table2.py --gpu --install "$DMP_GPU_BUILD/install" "$@"
