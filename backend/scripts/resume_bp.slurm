#!/bin/bash
#SBATCH --job-name=resume_bp_cap4
#SBATCH --partition=eda
#SBATCH --qos=eda
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
# Job1785220 was submitted with the 6h line above, then changed while pending to
# 02:30:00 via scontrol for backfill. The spooled script is unchanged; see
# assets/resume_schedule_1785220.json for before/after scheduler evidence.
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/resume_bp_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
run_id=released_bp_resume_job${SLURM_JOB_ID}
singularity exec --cleanenv --bind /fact_home/bohanli/RL_Placement,/fact_data/bohanli/flowplace_fql_backend \
 --env "NUM_CORES=$SLURM_CPUS_PER_TASK,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,SLURM_JOB_ID=$SLURM_JOB_ID" \
 "$IMAGE" bash -c 'set -euo pipefail; source /OpenROAD-flow-scripts/env.sh; unset FLOW_HOME DESIGN_HOME PLATFORM_HOME WORK_HOME SCRIPTS_DIR UTILS_DIR; export OPENROAD_EXE=$(command -v openroad); export YOSYS_EXE=$(command -v yosys); export KLAYOUT_CMD=$(command -v klayout); exec python3 scripts/resume_fullflow.py "$@"' bash \
 --source-run released_bp_job1783836_2 --run-id "$run_id"
python3 scripts/audit_fullflow.py "results/$run_id"
