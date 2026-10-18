#!/bin/bash
# Incremental compatibility retry for build 1787994; no placement changes.
#SBATCH --job-name=table2-dmp410-gpu-resume
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_h20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/dmp_gpu_resume_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
test "$SLURM_CPUS_PER_TASK" = 8
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
RUN=/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_1787994
SCRATCH=/scratch/bohanli/flowplace_table2/dmp_gpu_build_1787994
SOURCE="$SCRATCH/source"
BUILD="$SCRATCH/build"
INSTALL="$RUN/install"
RETRY="$RUN/retries/job_${SLURM_JOB_ID}"
OVERLAY="$SCRATCH/cccl_namespace_overlay"
PY="$RUN/env/bin/python"
test "$(cat "$RUN/exit_code.txt")" = 2
test ! -e "$RUN/BUILD_AND_OPERATOR_TESTS_PASSED"
test ! -e "$RETRY"
test ! -e "$OVERLAY"
mkdir -p "$RETRY"
trap 'status=$?; printf "%s\n" "$status" > "$RETRY/exit_code.txt"' EXIT
cp "$ROOT/backend/scripts/resume_dreamplace_table2_gpu.slurm" "$RETRY/submitted_script.slurm"
cp "$BUILD/CMakeCache.txt" "$RETRY/CMakeCache.before.txt"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$INSTALL:$ROOT/.venv/lib/python3.9/site-packages:$ROOT/iccad/assets/dmp_env/lib/python3.9/site-packages"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="$SCRATCH/tmp"
"$PY" - "$RETRY" "$OVERLAY" <<'PY'
import difflib, hashlib, json, os, sys
from pathlib import Path
import torch
retry, overlay = map(Path, sys.argv[1:])
assert torch.__version__ == '2.2.1+cu121'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (9, 0)
assert torch.cuda.get_device_name(0) == 'NVIDIA H20'
fixes = [
    ('cub/device/dispatch/dispatch_histogram.cuh',
     'auto minus_one = cuda::proclaim_return_type<int>',
     'auto minus_one = ::cuda::proclaim_return_type<int>',
     'https://github.com/NVIDIA/cccl/pull/2940/files'),
    ('cub/util_vsmem.cuh', 'cuda::discard_memory(thread_ptr, line_size);',
     '::cuda::discard_memory(thread_ptr, line_size);',
     'https://github.com/NVIDIA/cccl/pull/3733/files'),
]
records, patch = [], []
for relative, old, new, upstream in fixes:
    original = Path('/usr/local/cuda-12.8/include') / relative
    source = original.read_text()
    assert source.count(old) == 1 and new not in source, relative
    changed = source.replace(old, new)
    target = overlay / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(changed)
    patch.extend(difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
                                    fromfile=str(original), tofile=str(target)))
    records.append({'original': str(original), 'overlay': str(target),
                    'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'overlay_sha256': hashlib.sha256(changed.encode()).hexdigest(),
                    'upstream_fix': upstream})
(retry / 'cccl_namespace.patch').write_text(''.join(patch))
(retry / 'compatibility.json').write_text(json.dumps({
    'parent_build_job': 1787994, 'retry_job': os.environ['SLURM_JOB_ID'],
    'reason': 'CUB unqualified cuda namespace resolves to DreamPlace::cuda.',
    'scope': 'Two upstream namespace qualification fixes only; system toolkit and DREAMPlace source unchanged.',
    'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
    'torch': torch.__version__, 'torch_compiled_cuda': torch.version.cuda,
    'headers': records}, indent=2) + '\n')
PY
# Preserve the first failure logs. Only CUDA objects affected by the include
# option rebuild; existing successful CPU targets remain reusable.
cmake -S "$SOURCE" -B "$BUILD" \
    -DCUDA_NVCC_FLAGS="--std=c++17;-I$OVERLAY" > "$RETRY/configure.log" 2>&1
cmake --build "$BUILD" --target pin_pos_cuda_segment -j 8 > "$RETRY/cuda_probe_build.log" 2>&1
cmake --build "$BUILD" -j 8 > "$RETRY/build.log" 2>&1
cmake --install "$BUILD" > "$RETRY/install.log" 2>&1
cp "$BUILD/CMakeCache.txt" "$BUILD/install_manifest.txt" "$RETRY/"
cd "$RETRY"
"$PY" - "$INSTALL" <<'PY'
import importlib, json, sys
from pathlib import Path
import torch
from dreamplace import configure
root = Path(sys.argv[1]).resolve()
assert Path(configure.__file__).resolve().is_relative_to(root)
assert configure.compile_configurations['CUDA_FOUND'] == 'TRUE'
assert torch.cuda.is_available()
Path('installed_configure.json').write_text(json.dumps(configure.compile_configurations, indent=2) + '\n')
modules = {}
for name in ('hpwl', 'weighted_average_wirelength', 'electric_potential'):
    module = importlib.import_module(f'dreamplace.ops.{name}.{name}_cuda')
    assert Path(module.__file__).resolve().is_relative_to(root)
    modules[name] = module.__file__
Path('loaded_operator_modules.json').write_text(json.dumps(modules, indent=2) + '\n')
Path('loaded_cuda_libraries.txt').write_text('\n'.join(
    line for line in Path('/proc/self/maps').read_text().splitlines()
    if any(token in line.lower() for token in ('cuda', 'cublas', 'cufft', 'curand', 'cusolver', 'cusparse'))
) + '\n')
PY
# Include pin_pos because it was the first target blocked by the CUB headers.
for op in hpwl weighted_average_wirelength electric_potential pin_pos; do
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
    'initial_job': 1787994, 'successful_retry_job': os.environ['SLURM_JOB_ID'],
    'retry_dir': str(retry), 'compatibility_patch_sha256': hashlib.sha256((retry / 'cccl_namespace.patch').read_bytes()).hexdigest(),
    'scope': 'Installation and four small upstream numerical operator tests; no Table 2 placement.'}, indent=2) + '\n')
(run / 'BUILD_AND_OPERATOR_TESTS_PASSED').write_text('Successful retry ' + os.environ['SLURM_JOB_ID'] + '; no Table 2 placement executed.\n')
PY
printf '%s\n' "GPU_INSTALL=$INSTALL" "GPU_PYTHON=$PY" "RETRY=$RETRY"
