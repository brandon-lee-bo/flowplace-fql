#!/bin/bash
#SBATCH --job-name=table2_historical_runtime_path
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_runtime_path_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
TASK=/scratch/bohanli/flowplace_table2/openroad_cd519_source_v1
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
RECORD=/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1/runtime_path_${SLURM_JOB_ID}
mkdir "$RECORD"
export TMPDIR="$TASK/tmp" SINGULARITY_TMPDIR="$TASK/tmp" APPTAINER_TMPDIR="$TASK/tmp"
cp "$TASK/build_make/CMakeCache.txt" "$RECORD/cache_before.txt"
singularity exec --cleanenv --bind /fact_data/bohanli,/scratch/bohanli \
 --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1 "$IMAGE" \
 cmake -S "$TASK/source" -B "$TASK/build_make" \
 -DCMAKE_INSTALL_RPATH:STRING="$TASK/dependencies/or-tools-9.10.4067/lib" \
 > "$RECORD/reconfigure.log" 2>&1
cp "$TASK/build_make/CMakeCache.txt" "$RECORD/cache_after.txt"
python3 - "$ROOT" "$RECORD" <<'PY'
import json,pathlib,subprocess,sys,time
root,record=map(pathlib.Path,sys.argv[1:])
journal={'time':time.time(),'record':str(record),'reason':'Historical source appends /opt/or-tools to installation RPATH; supply the matching SDK directory first through the CMake cache. No source or algorithm patch.','commands':[]}
dest=root/'backend/plans/table2_background_v1/openroad_runtime_path_recovery_v1.json'
for command in [['scontrol','requeue','1788558'],
 ['scontrol','update','JobId=1788561','Dependency=afterok:1788558'],
 ['scontrol','update','JobId=1788559','Dependency=afterok:1788558:1788561:1788517:1788399'],
 ['scontrol','update','JobId=1788555','Dependency=afterok:1788558:1788517:1788399']]:
 r=subprocess.run(command,capture_output=True,text=True)
 journal['commands'].append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 dest.write_text(json.dumps(journal,indent=2)+'\n')
 assert r.returncode==0,journal
print(json.dumps(journal,indent=2))
PY
