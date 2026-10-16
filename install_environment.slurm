#!/bin/bash
#SBATCH --job-name=flowplace-env
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/install_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
.venv/bin/python -m pip install --no-index --no-deps artifacts/wheels/*.whl
.venv/bin/python -m pip install --no-index --no-build-isolation --find-links artifacts/wheels -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > artifacts/environment.freeze.txt
.venv/bin/python -c 'import torch, torch_geometric; print({"torch":torch.__version__,"torch_geometric":torch_geometric.__version__})'
