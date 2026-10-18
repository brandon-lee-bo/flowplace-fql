#!/bin/bash
#SBATCH --job-name=table2_openroad_cd519_build
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=quantum,infinite-frontier
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_cd519_build_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_CPUS_PER_TASK:?Missing allocation}"
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
TASK=/scratch/bohanli/flowplace_table2/openroad_cd519_source_v1
RECORD=/fact_data/bohanli/flowplace_fql_backend/tools/openroad_cd519_source_v1/build_${SLURM_JOB_ID}/attempt_${SLURM_RESTART_COUNT:-0}
IMAGE=/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif
REV=cd519bb5e861ec4d5d00931c85650c179a864e34
mkdir -p "$RECORD" "$TASK/tmp" "$TASK/cache"
cp "$ROOT/backend/scripts/build_openroad_cd519.slurm" "$RECORD/submitted.slurm"
trap 'code=$?; printf "%s\n" "$code" > "$RECORD/exit_code.txt"' EXIT
test "$(git -C "$TASK/source" rev-parse HEAD)" = "$REV"
git -C "$TASK/source" submodule status --recursive > "$RECORD/submodule_status.txt"
if rg -q '^[-+U]' "$RECORD/submodule_status.txt"; then
    echo 'Source probe did not finish pinned submodules; inspect its preserved logs.' >&2
    exit 2
fi
git -C "$TASK/source" diff --exit-code > "$RECORD/source.patch"
export TMPDIR="$TASK/tmp" SINGULARITY_TMPDIR="$TASK/tmp" APPTAINER_TMPDIR="$TASK/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
prefix=(singularity exec --cleanenv --bind /fact_data/bohanli,/scratch/bohanli
        --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK,OPENBLAS_NUM_THREADS=1,CCACHE_DIR=$TASK/cache,CCACHE_MAXSIZE=2G,XDG_CACHE_HOME=$TASK/cache"
        "$IMAGE")
child=''
trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; exit 143' TERM INT
"${prefix[@]}" cmake -S "$TASK/source" -B "$TASK/build_make" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TASK/install" \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DBUILD_GUI=OFF -DBUILD_PYTHON=OFF -DENABLE_TESTS=OFF -DBUILD_MAN=OFF \
    > "$RECORD/configure.log" 2>&1 &
child=$!
wait "$child"
child=''
"${prefix[@]}" cmake --build "$TASK/build_make" --parallel "$SLURM_CPUS_PER_TASK" \
    > "$RECORD/build.log" 2>&1 &
child=$!
wait "$child"
child=''
"${prefix[@]}" cmake --install "$TASK/build_make" > "$RECORD/install.log" 2>&1 &
child=$!
wait "$child"
child=''
test -x "$TASK/install/bin/openroad"
"${prefix[@]}" "$TASK/install/bin/openroad" -version > "$RECORD/version.log" 2>&1
"${prefix[@]}" ldd "$TASK/install/bin/openroad" > "$RECORD/ldd.log" 2>&1
if rg -q 'not found' "$RECORD/ldd.log"; then exit 2; fi
python3 - "$RECORD" "$TASK" "$REV" "$SLURM_JOB_ID" <<'PY'
import hashlib, json, os, pathlib, shutil, socket, sys, time
out, task = map(pathlib.Path, sys.argv[1:3])
quota = int(os.getxattr('/fact_data/bohanli', 'ceph.quota.max_bytes')) - int(os.getxattr('/fact_data/bohanli', 'ceph.dir.rbytes'))
binary = task/'install/bin/openroad'
assert quota >= 8*1024**3+binary.stat().st_size, 'Preserve 8 GiB physical-flow reserve'
dest = out/'openroad'
shutil.copy2(binary, dest)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2), b''): h.update(b)
 return h.hexdigest()
assert sha(binary)==sha(dest)
manifest = {'schema_version':1, 'revision':sys.argv[3], 'job':sys.argv[4],
 'host':socket.gethostname(), 'time':time.time(), 'binary':str(dest), 'binary_sha256':sha(dest),
 'reported_version':(out/'version.log').read_text().strip(),
 'scratch_build':str(task/'build_make'), 'no_algorithm_source_patch':True,
 'purpose':'Source-supported historical-tool diagnostic; author executable identity remains unconfirmed.',
 'records_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p!=dest}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
PY
"${prefix[@]}" "$RECORD/openroad" -version > "$RECORD/published_version.log" 2>&1
cmp "$RECORD/version.log" "$RECORD/published_version.log"
