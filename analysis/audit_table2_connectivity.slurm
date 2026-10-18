#!/bin/bash
#SBATCH --job-name=table2_connectivity
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_connectivity_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
: "${1:?Design}"
: "${2:?New output directory}"
: "${3:?Frozen runner SHA256}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
RUNNER=$ROOT/analysis/audit_table2_connectivity.py
test "$(sha256sum "$RUNNER" | cut -d ' ' -f1)" = "$3"
export TMPDIR=/scratch/bohanli/flowplace_table2/connectivity_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export SINGULARITY_TMPDIR="$TMPDIR" APPTAINER_TMPDIR="$TMPDIR"
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
OPENROAD=/fact_data/bohanli/flowplace_fql_backend/tools/openroad_20241214/probe_1787995/extracted/usr/bin/openroad
singularity exec --cleanenv \
  --bind /fact_home/bohanli/RL_Placement,/fact_data/bohanli,/scratch/bohanli \
  --env "SLURM_JOB_ID=$SLURM_JOB_ID,TMPDIR=$TMPDIR,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1" \
  "$IMAGE" python3 "$RUNNER" --design "$1" --output "$2" --openroad "$OPENROAD"
