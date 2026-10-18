#!/bin/bash
# Validation-only continuation: build/install and HPWL already succeeded.
#SBATCH --job-name=table2-dmp410-gpu-validate
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/dmp_gpu_validate_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
test "$SLURM_CPUS_PER_TASK" = 8
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
RUN=/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_1787994
PREVIOUS="$RUN/retries/job_1787998"
SOURCE=/scratch/bohanli/flowplace_table2/dmp_gpu_build_1787994/source
INSTALL="$RUN/install"
RETRY="$RUN/retries/job_${SLURM_JOB_ID}"
PY="$RUN/env/bin/python"
test "$(cat "$PREVIOUS/exit_code.txt")" = 1
test ! -e "$RUN/BUILD_AND_OPERATOR_TESTS_PASSED"
test ! -e "$RETRY"
mkdir "$RETRY"
trap 'status=$?; printf "%s\n" "$status" > "$RETRY/exit_code.txt"' EXIT
cp "$ROOT/backend/scripts/validate_dreamplace_table2_gpu.slurm" "$RETRY/submitted_script.slurm"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$INSTALL:$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_env/lib/python3.9/site-packages"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_HOME=/usr/local/cuda-12.8
export TMPDIR=/scratch/bohanli/flowplace_table2/dmp_gpu_build_1787994/tmp
"$PY" - "$SOURCE" "$PREVIOUS" "$RETRY" <<'PY'
import difflib, hashlib, json, os, shutil, sys
from pathlib import Path
import torch
source, previous, retry = map(Path, sys.argv[1:])
assert torch.__version__ == '2.2.1+cu121' and torch.version.cuda == '12.1'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == 'NVIDIA H20' and torch.cuda.get_device_capability(0) == (9, 0)
hpwl = (previous / 'test_hpwl.log').read_text()
assert '\nOK\n' in hpwl and 'hpwl_value cuda =' in hpwl
files = ['compatibility.json', 'cccl_namespace.patch', 'configure.log', 'CMakeCache.txt',
         'installed_configure.json', 'loaded_operator_modules.json', 'loaded_cuda_libraries.txt', 'test_hpwl.log']
for name in files:
    shutil.copyfile(previous / name, retry / name)
upstream = source / 'unittest/ops/weighted_average_wirelength_unittest.py'
before = upstream.read_text()
old = 'pin_mask = np.zeros(len(pin2net_map), dtype=np.uint8)'
new = 'pin_mask = np.zeros(len(pin2net_map), dtype=np.bool_)'
assert before.count(old) == 1
after = before.replace(old, new)
adapted = retry / 'weighted_average_wirelength_unittest.py'
adapted.write_text(after)
(retry / 'test_fixture.patch').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), after.splitlines(True), fromfile=str(upstream), tofile=str(adapted))))
(retry / 'validation_continuation.json').write_text(json.dumps({
    'build_retry_job': 1787998, 'validation_job': os.environ['SLURM_JOB_ID'],
    'reused_successful_artifacts': {name: {'source': str(previous / name),
        'sha256': hashlib.sha256((previous / name).read_bytes()).hexdigest()} for name in files},
    'fixture_original_sha256': hashlib.sha256(before.encode()).hexdigest(),
    'fixture_adapted_sha256': hashlib.sha256(after.encode()).hexdigest(),
    'reason': 'Torch 2.2 masked_fill requires bool; actual stock BasicPlace already creates a bool pin_mask by comparison.',
    'scope': 'One test fixture dtype only. Installed operator code, numerical expectations and tolerances unchanged.'}, indent=2) + '\n')
PY
cd "$RETRY"
"$PY" "$RETRY/weighted_average_wirelength_unittest.py" > "$RETRY/test_weighted_average_wirelength.log" 2>&1
for op in electric_potential pin_pos; do
    "$PY" "$SOURCE/unittest/ops/${op}_unittest.py" > "$RETRY/test_${op}.log" 2>&1
done
"$PY" - "$INSTALL" "$RUN" "$RETRY" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
install, run, retry = map(Path, sys.argv[1:])
hashes = {str(p.relative_to(install)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(install.rglob('*')) if p.is_file() and p.suffix in ('.py', '.so', '.json')}
(run / 'installation_file_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
(run / 'build_completion.json').write_text(json.dumps({
    'initial_job': 1787994, 'build_retry_job': 1787998,
    'successful_retry_job': os.environ['SLURM_JOB_ID'], 'retry_dir': str(retry),
    'compatibility_patch_sha256': hashlib.sha256((retry / 'cccl_namespace.patch').read_bytes()).hexdigest(),
    'test_fixture_patch_sha256': hashlib.sha256((retry / 'test_fixture.patch').read_bytes()).hexdigest(),
    'scope': 'Build/install and HPWL reused from 1787998; remaining three operator tests completed here. No Table 2 placement.'}, indent=2) + '\n')
(run / 'BUILD_AND_OPERATOR_TESTS_PASSED').write_text('Successful validation continuation ' + os.environ['SLURM_JOB_ID'] + '; no Table 2 placement executed.\n')
PY
printf '%s\n' "GPU_INSTALL=$INSTALL" "GPU_PYTHON=$PY" "VALIDATION=$RETRY"
