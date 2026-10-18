#!/bin/bash
# Prepared build candidate only. Submit explicitly; no installation downloads.
#SBATCH --job-name=table2-dmp410-gpu-build
#SBATCH --partition=gpu-scavenger
#SBATCH --qos=gpu-scavenger
#SBATCH --gres=gpu:nvidia_l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/dmp_gpu_build_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script using sbatch}"
: "${SLURM_CPUS_PER_TASK:?Missing Slurm CPU allocation}"
if (( SLURM_CPUS_PER_TASK < 8 || SLURM_CPUS_PER_TASK > 16 )); then
    echo 'This candidate expects 8 to 16 allocated CPUs.' >&2
    exit 2
fi
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
SOURCE_INPUT="$ROOT/iccad/assets/DREAMPlace-4.1.0"
DEPS="$ROOT/iccad/assets/deps"
BASE_PACKAGES="$ROOT/.venv/lib/python3.9/site-packages"
EXTRA_PACKAGES="$ROOT/iccad/assets/dmp_env/lib/python3.9/site-packages"
CUDA_ROOT="${DMP_GPU_CUDA_ROOT:-/usr/local/cuda-12.8}"
RUN="/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_${SLURM_JOB_ID}"
SCRATCH="/scratch/bohanli/flowplace_table2/dmp_gpu_build_${SLURM_JOB_ID}"
SOURCE="$SCRATCH/source"
BUILD="$SCRATCH/build"
INSTALL="$RUN/install"
ENV_DIR="$RUN/env"
if [[ -e "$RUN" || -L "$RUN" || -e "$SCRATCH" || -L "$SCRATCH" ]]; then
    echo 'Refusing to overwrite an existing build attempt.' >&2
    exit 2
fi
test -x "$CUDA_ROOT/bin/nvcc"
test -f "$CUDA_ROOT/include/cuda_runtime.h"
test -f "$DEPS/rpmroot/usr/include/python3.9/Python.h"
test -f "$DEPS/lib/libpython3.9.so.1.0"
cd "$ROOT"
# A small read-only quota check runs inside this allocation before creating data.
"$ROOT/.venv/bin/python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'backend/scripts')
from result_storage import storage_status
status = storage_status(Path('/fact_data/bohanli'), Path('/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build'))
for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
    if status.get(key) is not None and status[key] < 4 * 1024**3:
        raise SystemExit('Less than 4 GiB bulk headroom: ' + key)
print(status)
PY
mkdir -p "$RUN" "$SCRATCH"
trap 'status=$?; printf "%s\n" "$status" > "$RUN/exit_code.txt"' EXIT
cp "$ROOT/backend/scripts/build_dreamplace_table2_gpu.slurm" "$RUN/submitted_script.slurm"
git -C "$SOURCE_INPUT" rev-parse HEAD > "$RUN/source_commit.txt"
test "$(cat "$RUN/source_commit.txt")" = 5d13c9001a3bc900dca1e108e633d5dd45b00701
git -C "$SOURCE_INPUT" submodule status --recursive > "$RUN/submodules.txt"
if rg -q '^[-+U]' "$RUN/submodules.txt"; then
    echo 'Submodule revisions differ from the pinned checkout.' >&2
    exit 2
fi
git -C "$SOURCE_INPUT" diff --submodule=diff > "$RUN/source.patch"
# The existing CPU build has a documented OpenTimer doctest-only CMake fix.
# Capture nested working-tree patches explicitly; do not modify the source.
git -C "$SOURCE_INPUT" submodule foreach --recursive 'git diff -- .; git status --short' > "$RUN/submodule_worktree_patches.txt"
# OpenTimer writes some build outputs into its source directory: isolate it too.
rsync -a --exclude=.git --exclude=__pycache__ "$SOURCE_INPUT/" "$SOURCE/"
"$ROOT/.venv/bin/python" - "$SOURCE" "$RUN/source_file_hashes.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
out = {}
for path in sorted(root.rglob('*')):
    if path.is_file():
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        out[str(path.relative_to(root))] = h.hexdigest()
Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + '\n')
PY
# New interpreter environment; reuse existing pinned packages read-only.
# No pip install, upgrade, download, or mutation of the working CPU environment.
"$ROOT/.venv/bin/python" -m venv --without-pip "$ENV_DIR"
PY="$ENV_DIR/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$INSTALL:$BASE_PACKAGES:$EXTRA_PACKAGES"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export TMPDIR="$SCRATCH/tmp"
mkdir "$TMPDIR"
"$CUDA_ROOT/bin/nvcc" --version > "$RUN/nvcc.txt"
gcc --version > "$RUN/gcc.txt"
cmake --version > "$RUN/cmake.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$RUN/nvidia_smi.csv"
"$PY" - "$RUN" <<'PY'
import importlib.metadata, json, os, platform, re, sys
from pathlib import Path
import torch
out = Path(sys.argv[1])
assert torch.__version__ == '2.2.1+cu121', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert not torch._C._GLIBCXX_USE_CXX11_ABI
assert torch.cuda.is_available(), 'Stock DREAMPlace silently selects CPU when CUDA is unavailable.'
assert torch.cuda.device_count() == 1, 'Expected one scheduler-visible GPU.'
toolkit = re.search(r'release\s+(\d+)\.(\d+)', (out / 'nvcc.txt').read_text())
assert toolkit and toolkit.group(1) == '12', 'This candidate is limited to CUDA 12.x.'
major, minor = torch.cuda.get_device_capability(0)
device = {'torch': torch.__version__, 'torch_compiled_cuda': torch.version.cuda,
          'nvcc_cuda': '.'.join(toolkit.groups()),
          'torch_file': torch.__file__, 'python': sys.version, 'platform': platform.platform(),
          'gpu': torch.cuda.get_device_name(0), 'capability': [major, minor],
          'cuda_home': os.environ['CUDA_HOME'], 'ld_library_path': os.getenv('LD_LIBRARY_PATH', ''),
          'abi': int(torch._C._GLIBCXX_USE_CXX11_ABI), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
          'scope': 'Build compatibility attempt; CUDA toolkit minor version may differ from the PyTorch wheel.'}
(out / 'environment.json').write_text(json.dumps(device, indent=2) + '\n')
# DREAMPlace uses the legacy FindCUDA dotted notation, not modern integer 89.
(out / 'cuda_arch.txt').write_text(f'{major}.{minor}\n')
packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
(out / 'python_packages.json').write_text(json.dumps(packages, indent=2) + '\n')
PY
ARCH=$(cat "$RUN/cuda_arch.txt")
cmake -S "$SOURCE" -B "$BUILD" \
    -DPython_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY" \
    -DPython_INCLUDE_DIR="$DEPS/rpmroot/usr/include/python3.9" \
    -DPYTHON_INCLUDE_DIR="$DEPS/rpmroot/usr/include/python3.9" \
    -DPython_LIBRARY="$DEPS/lib/libpython3.9.so.1.0" \
    -DPYTHON_LIBRARY="$DEPS/lib/libpython3.9.so.1.0" \
    -DBoost_INCLUDE_DIR="$DEPS/include" -DBOOST_LIBRARYDIR="$DEPS/lib" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_ABI=0 \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" -DCUDA_NVCC_EXECUTABLE="$CUDA_ROOT/bin/nvcc" \
    -DCUDA_HOST_COMPILER=/usr/bin/gcc -DCUDA_NVCC_FLAGS=--std=c++17 \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" > "$RUN/configure.log" 2>&1
"$PY" - "$BUILD/dreamplace/configure.py" <<'PY'
import runpy, sys
cfg = runpy.run_path(sys.argv[1])['compile_configurations']
assert cfg['CUDA_FOUND'] == 'TRUE', cfg
print(cfg)
PY
# Compile one real CUDA operator first to expose toolchain/header problems early.
# This target is reused by the subsequent full build.
cmake --build "$BUILD" --target weighted_average_wirelength_cuda -j "$SLURM_CPUS_PER_TASK" > "$RUN/cuda_probe_build.log" 2>&1
cmake --build "$BUILD" -j "$SLURM_CPUS_PER_TASK" > "$RUN/build.log" 2>&1
cmake --install "$BUILD" > "$RUN/install.log" 2>&1
cp "$BUILD/CMakeCache.txt" "$BUILD/install_manifest.txt" "$RUN/"
cd "$RUN"
"$PY" - "$INSTALL" <<'PY'
import importlib, json, sys
from pathlib import Path
import torch
from dreamplace import configure
assert Path(configure.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert configure.compile_configurations['CUDA_FOUND'] == 'TRUE'
assert torch.cuda.is_available()
Path('installed_configure.json').write_text(json.dumps(configure.compile_configurations, indent=2) + '\n')
modules = {}
for name in ('hpwl', 'weighted_average_wirelength', 'electric_potential'):
    module = importlib.import_module(f'dreamplace.ops.{name}.{name}_cuda')
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
    modules[name] = module.__file__
Path('loaded_operator_modules.json').write_text(json.dumps(modules, indent=2) + '\n')
# Wheel RPATH and extension RPATH may resolve CUDA shared libraries differently.
# Record what was actually loaded; do not change LD_LIBRARY_PATH to force it.
Path('loaded_cuda_libraries.txt').write_text('\n'.join(
    line for line in Path('/proc/self/maps').read_text().splitlines()
    if any(token in line.lower() for token in ('cuda', 'cublas', 'cufft', 'curand', 'cusolver', 'cusparse'))
) + '\n')
PY
# Tiny upstream numerical operator checks, including the CUDA paths. No design
# placement or OpenROAD job is part of this installation attempt.
for op in hpwl weighted_average_wirelength electric_potential; do
    "$PY" "$SOURCE/unittest/ops/${op}_unittest.py" > "$RUN/test_${op}.log" 2>&1
done
"$PY" - "$INSTALL" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py', '.so', '.json')}
Path('installation_file_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
Path('BUILD_AND_OPERATOR_TESTS_PASSED').write_text('No Table 2 placement has been executed by this build job.\n')
PY
printf '%s\n' "GPU_INSTALL=$INSTALL" "GPU_PYTHON=$PY" "GPU_ARCH=$ARCH" "BUILD_SCRATCH=$SCRATCH"
