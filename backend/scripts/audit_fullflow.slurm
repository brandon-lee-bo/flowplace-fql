#!/bin/bash
#SBATCH --job-name=fullflow_audit
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/fullflow_audit_%j.log
set -uo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
status=0
for run in "$@"; do
  python3 scripts/audit_fullflow.py "$run" || status=1
done
exit "$status"
