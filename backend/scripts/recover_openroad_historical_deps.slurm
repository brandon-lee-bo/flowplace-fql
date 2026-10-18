#!/bin/bash
#SBATCH --job-name=table2_openroad_historical_deps
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_historical_deps_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
TASK=/scratch/bohanli/flowplace_table2/openroad_cd519_source_v1
DEPS="$TASK/dependencies"
RECORD=/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1/historical_deps_${SLURM_JOB_ID}
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
mkdir -p "$RECORD" "$DEPS" "$TASK/tmp"
cp "$ROOT/backend/scripts/recover_openroad_historical_deps.slurm" "$RECORD/submitted.slurm"
trap 'code=$?; printf "%s\n" "$code" > "$RECORD/exit_code.txt"' EXIT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
curl --fail --location --continue-at - --retry 3 --connect-timeout 20 --max-time 1800 \
 https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
 -o "$DEPS/boost_1_80_0.tar.gz" > "$RECORD/boost_download.log" 2>&1
printf '077f074743ea7b0cb49c6ed43953ae95  %s\n' "$DEPS/boost_1_80_0.tar.gz" | md5sum -c - > "$RECORD/boost_checksum.log"
curl --fail --location --continue-at - --retry 3 --connect-timeout 20 --max-time 1800 \
 https://github.com/google/or-tools/releases/download/v9.10/or-tools_amd64_ubuntu-22.04_cpp_v9.10.4067.tar.gz \
 -o "$DEPS/or-tools_9.10.4067.tar.gz" > "$RECORD/ortools_download.log" 2>&1
sha256sum "$DEPS/boost_1_80_0.tar.gz" "$DEPS/or-tools_9.10.4067.tar.gz" > "$RECORD/download_sha256.txt"
tar -xf "$DEPS/boost_1_80_0.tar.gz" -C "$DEPS"
mkdir "$DEPS/or-tools-9.10.4067"
tar --strip-components=1 -xf "$DEPS/or-tools_9.10.4067.tar.gz" -C "$DEPS/or-tools-9.10.4067"
export TMPDIR="$TASK/tmp" SINGULARITY_TMPDIR="$TASK/tmp" APPTAINER_TMPDIR="$TASK/tmp"
prefix=(singularity exec --cleanenv --bind /fact_data/bohanli,/scratch/bohanli
 --env OMP_NUM_THREADS=8,OPENBLAS_NUM_THREADS=1 "$IMAGE")
"${prefix[@]}" bash -c 'cd "$1"; ./bootstrap.sh --prefix="$2" --with-libraries=iostreams,serialization,system,thread,test; ./b2 -j8 link=static runtime-link=shared threading=multi cxxflags=-fPIC install' \
 bash "$DEPS/boost_1_80_0" "$DEPS/boost-install" > "$RECORD/boost_build_install.log" 2>&1
# Preserve the complete failed cache/build; configure a fresh directory so no
# package imported from the modern OR-Tools SDK can survive in the new graph.
mv "$TASK/build_make" "$TASK/build_make_modern_deps_failed_v1"
"${prefix[@]}" cmake -S "$TASK/source" -B "$TASK/build_make" -G 'Unix Makefiles' \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TASK/install" \
 -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc \
 -DBUILD_GUI=OFF -DBUILD_PYTHON=OFF -DENABLE_TESTS=OFF -DBUILD_MAN=OFF \
 -Dspdlog_DIR:PATH="$DEPS/install/lib/cmake/spdlog" \
 -DBoost_DIR:PATH="$DEPS/boost-install/lib/cmake/Boost-1.80.0" \
 -DBoost_USE_STATIC_LIBS=ON -Dortools_DIR:PATH="$DEPS/or-tools-9.10.4067/lib/cmake/ortools" \
 -DCMAKE_PREFIX_PATH="$DEPS/boost-install;$DEPS/or-tools-9.10.4067;$DEPS/install" \
 -DCMAKE_IGNORE_PREFIX_PATH=/opt/or-tools > "$RECORD/openroad_configure.log" 2>&1
cp "$TASK/build_make/CMakeCache.txt" "$RECORD/openroad_cache.txt"
python3 - "$ROOT" "$RECORD" "$TASK" <<'PY'
import json,pathlib,subprocess,sys,time
root,record,task=map(pathlib.Path,sys.argv[1:])
cache=(record/'openroad_cache.txt').read_text()
for key,expected in [('Boost_DIR',task/'dependencies/boost-install/lib/cmake/Boost-1.80.0'),
 ('ortools_DIR',task/'dependencies/or-tools-9.10.4067/lib/cmake/ortools'),
 ('spdlog_DIR',task/'dependencies/install/lib/cmake/spdlog')]:
 assert any(line.startswith(key+':') and line.split('=',1)[1]==str(expected) for line in cache.splitlines()),key
journal={'time':time.time(),'dependency_versions':{'boost':'1.80.0','spdlog':'1.8.1','ortools':'9.10.4067'},
 'basis':'Versions specified by the pinned OpenROAD DependencyInstaller.sh, not PPA results',
 'record':str(record),'no_algorithm_source_changes':True,
 'runtime_sdk':str(task/'dependencies/or-tools-9.10.4067'),
 'runtime_sdk_retention':'Keep scratch SDK while binary is used; source archives and hashes retained. Preserve durable SDK before declaring a reusable installed tool.',
 'commands':[]}
dest=root/'backend/plans/table2_background_v1/openroad_historical_deps_recovery_v1.json'
for command in [['scontrol','requeue','1788546'],
 ['scontrol','update','JobId=1788551','Dependency=afterok:1788546:1788517:1788399']]:
 r=subprocess.run(command,capture_output=True,text=True)
 journal['commands'].append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 dest.write_text(json.dumps(journal,indent=2)+'\n')
 assert r.returncode==0,journal
print(json.dumps(journal,indent=2))
PY
