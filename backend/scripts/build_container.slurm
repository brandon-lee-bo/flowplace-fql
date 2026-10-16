#!/bin/bash
#SBATCH --job-name=flowplace_orfs_image
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/container_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export SINGULARITY_CACHEDIR=/fact_data/bohanli/flowplace_fql_backend/singularity-cache
export SINGULARITY_TMPDIR=/fact_data/bohanli/flowplace_fql_backend/singularity-tmp
singularity pull /fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif docker://openroad/orfs:26Q4-151-g8b5e62fbb
singularity exec /fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif bash -lc 'command -v openroad; openroad -version; command -v yosys; command -v klayout; ls /OpenROAD-flow-scripts/tools/install'
sha256sum /fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
