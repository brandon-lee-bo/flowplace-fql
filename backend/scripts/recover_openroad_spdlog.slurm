#!/bin/bash
#SBATCH --job-name=table2_openroad_historical_spdlog
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_spdlog_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
TASK=/scratch/bohanli/flowplace_table2/openroad_cd519_source_v1
RECORD=/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1/spdlog_${SLURM_JOB_ID}
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
mkdir -p "$RECORD" "$TASK/dependencies" "$TASK/tmp"
cp "$ROOT/backend/scripts/recover_openroad_spdlog.slurm" "$RECORD/submitted.slurm"
trap 'code=$?; printf "%s\n" "$code" > "$RECORD/exit_code.txt"' EXIT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
git -c http.version=HTTP/1.1 clone --depth=1 --branch v1.8.1 \
 https://github.com/gabime/spdlog.git "$TASK/dependencies/spdlog-1.8.1" \
 > "$RECORD/fetch.log" 2>&1
git -C "$TASK/dependencies/spdlog-1.8.1" rev-parse HEAD > "$RECORD/source_revision.txt"
git -C "$TASK/dependencies/spdlog-1.8.1" diff --exit-code > "$RECORD/source.patch"
export TMPDIR="$TASK/tmp" SINGULARITY_TMPDIR="$TASK/tmp" APPTAINER_TMPDIR="$TASK/tmp"
prefix=(singularity exec --cleanenv --bind /fact_data/bohanli,/scratch/bohanli
 --env OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=1 "$IMAGE")
"${prefix[@]}" cmake -S "$TASK/dependencies/spdlog-1.8.1" -B "$TASK/dependencies/spdlog-build" \
 -G 'Unix Makefiles' -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc \
 -DCMAKE_INSTALL_PREFIX="$TASK/dependencies/install" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DSPDLOG_BUILD_EXAMPLE=OFF -DSPDLOG_BUILD_TESTS=OFF \
 > "$RECORD/configure.log" 2>&1
"${prefix[@]}" cmake --build "$TASK/dependencies/spdlog-build" --parallel 2 > "$RECORD/build.log" 2>&1
"${prefix[@]}" cmake --install "$TASK/dependencies/spdlog-build" > "$RECORD/install.log" 2>&1
# The failed OpenROAD build has terminated before this job starts. Preserve its
# complete cache, then change only the discovered dependency's package path.
cp "$TASK/build_make/CMakeCache.txt" "$RECORD/openroad_cache_before.txt"
"${prefix[@]}" cmake -S "$TASK/source" -B "$TASK/build_make" \
 -Dspdlog_DIR="$TASK/dependencies/install/lib/cmake/spdlog" \
 > "$RECORD/openroad_reconfigure.log" 2>&1
cp "$TASK/build_make/CMakeCache.txt" "$RECORD/openroad_cache_after.txt"
python3 - "$RECORD" "$ROOT" "$TASK" <<'PY'
import hashlib,json,pathlib,subprocess,sys,time
record,root,task=map(pathlib.Path,sys.argv[1:])
cache=(record/'openroad_cache_after.txt').read_text()
expected=str(task/'dependencies/install/lib/cmake/spdlog')
assert any(line.startswith('spdlog_DIR:') and line.split('=',1)[1]==expected
           for line in cache.splitlines())
commands=[['scontrol','requeue','1788546'],
 ['scontrol','update','JobId=1788548','Dependency=afterok:1788546:1788517:1788399'],
 ['scontrol','release','1788548']]
journal={'time':time.time(),'dependency':'spdlog v1.8.1, as pinned by the historical DependencyInstaller.sh',
 'source_revision':(record/'source_revision.txt').read_text().strip(),
 'no_algorithm_source_changes':True,'commands':[]}
dest=root/'backend/plans/table2_background_v1/openroad_spdlog_recovery_v1.json'
for command in commands:
 result=subprocess.run(command,capture_output=True,text=True)
 journal['commands'].append({'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
 dest.write_text(json.dumps(journal,indent=2)+'\n')
 assert result.returncode==0,journal
print(json.dumps(journal,indent=2))
PY
