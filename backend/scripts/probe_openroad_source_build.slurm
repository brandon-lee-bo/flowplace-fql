#!/bin/bash
#SBATCH --job-name=table2_source_build_probe
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_source_build_probe_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
TASK=/scratch/bohanli/flowplace_table2/openroad_cd519_source_v1
RECORD=/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1/probe_${SLURM_JOB_ID}
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
REV=cd519bb5e861ec4d5d00931c85650c179a864e34
test ! -e "$TASK"
mkdir -p "$TASK/tmp" "$RECORD"
cp "$ROOT/backend/scripts/probe_openroad_source_build.slurm" "$RECORD/submitted.slurm"
export TMPDIR="$TASK/tmp" SINGULARITY_TMPDIR="$TASK/tmp" APPTAINER_TMPDIR="$TASK/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
trap 'code=$?; printf "%s\n" "$code" > "$RECORD/exit_code.txt"' EXIT
git init "$TASK/source" > "$RECORD/git_init.log" 2>&1
git -C "$TASK/source" remote add origin https://github.com/The-OpenROAD-Project/OpenROAD.git
timeout 900 git -C "$TASK/source" fetch --depth=1 origin "$REV" > "$RECORD/git_fetch.log" 2>&1
git -C "$TASK/source" checkout --detach FETCH_HEAD > "$RECORD/git_checkout.log" 2>&1
test "$(git -C "$TASK/source" rev-parse HEAD)" = "$REV"
timeout 900 git -C "$TASK/source" submodule update --init --recursive --depth=1 > "$RECORD/git_submodules.log" 2>&1
git -C "$TASK/source" submodule status --recursive > "$RECORD/submodule_status.txt"
git -C "$TASK/source" status --short > "$RECORD/source_status.txt"
cp "$TASK/source/CMakeLists.txt" "$RECORD/CMakeLists.txt"
cp "$TASK/source/.gitmodules" "$RECORD/gitmodules.txt"
singularity exec --cleanenv --bind /fact_data/bohanli,/scratch/bohanli \
    --env OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=1 "$IMAGE" \
    bash -c 'cat /etc/os-release; command -v cmake gcc g++ ninja swig; cmake --version; gcc --version; ls /usr/local; ls /opt; dpkg-query -W libboost-dev libeigen3-dev libspdlog-dev libtcl8.6 libtcl8.6-dev swig libreadline-dev libzstd-dev 2>/dev/null || true' \
    > "$RECORD/environment.log" 2>&1
set +e
singularity exec --cleanenv --bind /fact_data/bohanli,/scratch/bohanli \
    --env OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=1 "$IMAGE" \
    cmake -S "$TASK/source" -B "$TASK/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TASK/install" \
    -DENABLE_GUI=OFF -DENABLE_TESTS=OFF > "$RECORD/configure.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RECORD/configure_exit_code.txt"
python3 - "$RECORD" "$TASK" "$REV" "$SLURM_JOB_ID" <<'PY'
import hashlib, json, pathlib, socket, sys, time
out, task = map(pathlib.Path, sys.argv[1:3])
manifest = {'schema_version': 1, 'revision': sys.argv[3], 'job': sys.argv[4],
    'host': socket.gethostname(), 'scratch_tree': str(task), 'time': time.time(),
    'configure_exit_code': int((out/'configure_exit_code.txt').read_text()),
    'purpose': 'Build feasibility of the source-era ORFS gitlink; not proof of author binary identity; no PPA execution.',
    'records_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
PY
exit "$status"
