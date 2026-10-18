#!/bin/bash
#SBATCH --job-name=flowplace-dmp410-build
#SBATCH --partition=eda
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/iccad/logs/build_dmp_%j.log
set -euo pipefail
ROOT=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql
cd "$ROOT"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
SOURCE="$ROOT/iccad/assets/DREAMPlace-4.1.0"
RUN="$ROOT/iccad/assets/build_${SLURM_JOB_ID}"
mkdir "$RUN"
cp "$ROOT/iccad/scripts/build_dreamplace.slurm" "$RUN/submitted_script.slurm"
git -C "$SOURCE" rev-parse HEAD > "$RUN/source_commit.txt"
git -C "$SOURCE" submodule status --recursive > "$RUN/submodules.txt"
if grep -E '^[-+U]' "$RUN/submodules.txt"; then
    echo 'Submodules must match the pinned source before building'; exit 1
fi
git -C "$SOURCE" diff --submodule=diff > "$RUN/source.patch"
"$ROOT/.venv/bin/python" -m venv --system-site-packages "$ROOT/iccad/assets/dmp_env"
PY="$ROOT/iccad/assets/dmp_env/bin/python"
"$PY" -m pip install --disable-pip-version-check pyunpack patool cairocffi pkgconfig > "$RUN/pip.log" 2>&1
# The separate venv exposes the existing pinned FlowPlace dependencies explicitly.
export PYTHONPATH="$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pip freeze > "$RUN/pip_freeze.txt"
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch._C._GLIBCXX_USE_CXX11_ABI)'
# Use only a Slurm CPU allocation. This validates the original 4.1.0 CPU flow;
# runtime must not be compared with the paper GPU runtime.
cmake -S "$SOURCE" -B "$ROOT/iccad/assets/dmp_build_cpu" \
    -DPython_EXECUTABLE="$PY" \
    -DPython_INCLUDE_DIR="$ROOT/iccad/assets/deps/rpmroot/usr/include/python3.9" \
    -DPython_LIBRARY="$ROOT/iccad/assets/deps/lib/libpython3.9.so.1.0" \
    -DPYTHON_EXECUTABLE="$PY" \
    -DPYTHON_INCLUDE_DIR="$ROOT/iccad/assets/deps/rpmroot/usr/include/python3.9" \
    -DPYTHON_LIBRARY="$ROOT/iccad/assets/deps/lib/libpython3.9.so.1.0" \
    -DCMAKE_INSTALL_PREFIX="$ROOT/iccad/assets/dmp_install_cpu" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_ABI=0 \
    -DBoost_INCLUDE_DIR="$ROOT/iccad/assets/deps/include" \
    -DBOOST_LIBRARYDIR="$ROOT/iccad/assets/deps/lib" > "$RUN/configure.log" 2>&1
cmake --build "$ROOT/iccad/assets/dmp_build_cpu" -j "$SLURM_CPUS_PER_TASK" > "$RUN/build.log" 2>&1
cmake --install "$ROOT/iccad/assets/dmp_build_cpu" > "$RUN/install.log" 2>&1
echo 'DREAMPLACE_BUILD_COMPLETED'
