#!/bin/bash
#SBATCH --job-name=grt_final_audit
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/grt_final_audit_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
: "${1:?Pass task manifest}" "${2:?Pass output JSON}"
python3 scripts/audit_grt_tasks.py "$1" --stamp-complete --require-complete --output "$2"
