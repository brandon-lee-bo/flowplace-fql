#!/bin/bash
#SBATCH --job-name=flowplace-iccad-audit
#SBATCH --partition=eda
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/audit_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cp iccad/scripts/audit_inputs.py "iccad/assets/audit_inputs_${SLURM_JOB_ID}.py"
python3 iccad/scripts/audit_inputs.py --output "iccad/assets/geometry_${SLURM_JOB_ID}" "$@"
