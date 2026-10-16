#!/bin/bash
#SBATCH --job-name=classify_failures
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/classify_failures_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
python3 scripts/classify_failure_batch.py "$@"
