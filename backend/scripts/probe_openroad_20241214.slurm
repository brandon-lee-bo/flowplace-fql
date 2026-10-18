#!/bin/bash
#SBATCH --job-name=flowplace_openroad_20241214_probe
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/openroad_20241214_probe_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
export TMPDIR="/scratch/bohanli/openroad_20241214_probe/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export SINGULARITY_TMPDIR="$TMPDIR"
export APPTAINER_TMPDIR="$TMPDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python3 - "$SLURM_JOB_ID" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import traceback

job = sys.argv[1]
base = pathlib.Path('/fact_data/bohanli/flowplace_fql_backend/tools/openroad_20241214')
out = base / ('probe_' + job)
out.mkdir(parents=True, exist_ok=False)
source = pathlib.Path('/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/scripts/probe_openroad_20241214.slurm')
image = pathlib.Path('/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif')
url = 'https://github.com/Precision-Innovations/OpenROAD/releases/download/2024-12-14/openroad_2.0-17598-ga008522d8_amd64-ubuntu-22.04.deb'
expected_size = 56971996
expected_version = 'v2.0-17598-ga008522d8'
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest = {
    'schema_version': 1, 'job_id': job, 'host': socket.gethostname(),
    'time_start': now(), 'status': 'started', 'success': False,
    'purpose': 'Isolated download and dependency/version probe; no design execution.',
    'release_url': 'https://github.com/Precision-Innovations/OpenROAD/releases/tag/2024-12-14',
    'asset_url': url, 'asset_id': 213317854, 'expected_asset_size_bytes': expected_size,
    'expected_version': expected_version, 'publisher_sha256': None,
    'publisher_sha256_note': 'GitHub release API digest is null; local hash is computed after download.',
    'tool_image': str(image), 'tool_image_size_bytes': image.stat().st_size,
    'tool_image_sha256_previously_recorded': '85f0ce74d79a79c1190632f775f0ed818c8de6e0fafd87717ae3d4839207ea4c',
    'tool_image_hash_recomputed_in_this_probe': False,
    'source_sha256_start': digest(source), 'commands': [],
}
shutil.copyfile(source, out / source.name)
def save():
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
def run(name, command, timeout=120, required=True):
    before = now()
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    log = out / (name + '.log')
    log.write_text(result.stdout)
    manifest['commands'].append({'name': name, 'command': command, 'time_start': before,
                                 'time_end': now(), 'returncode': result.returncode,
                                 'log': str(log), 'log_sha256': digest(log)})
    save()
    print(name, 'returncode=', result.returncode, flush=True)
    if required and result.returncode:
        raise RuntimeError(name + ' failed; see ' + str(log))
    return result
save()
try:
    deb = out / url.rsplit('/', 1)[1]
    partial = deb.with_suffix(deb.suffix + '.partial')
    run('download', ['curl', '--fail', '--location', '--retry', '3', '--retry-delay', '3',
                     '--connect-timeout', '30', '--max-time', '600', '--output', str(partial), url], timeout=900)
    if partial.stat().st_size != expected_size:
        raise RuntimeError('Release asset size differs from frozen metadata')
    partial.rename(deb)
    manifest['deb'] = {'path': str(deb), 'size_bytes': deb.stat().st_size, 'sha256': digest(deb)}
    run('package_info', ['dpkg-deb', '--info', str(deb)])
    root = out / 'extracted'
    run('extract', ['dpkg-deb', '--extract', str(deb), str(root)])
    binary = root / 'usr/bin/openroad'
    if not binary.is_file():
        raise RuntimeError('Expected extracted usr/bin/openroad is absent; no alternative guessed')
    manifest['binary'] = {'path': str(binary), 'size_bytes': binary.stat().st_size,
                          'sha256': digest(binary)}
    prefix = ['singularity', 'exec', '--cleanenv', '--bind', '/fact_data/bohanli,/scratch/bohanli',
              '--env', 'OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1', str(image)]
    run('container_os', prefix + ['cat', '/etc/os-release'])
    deps = run('ldd', prefix + ['ldd', str(binary)], required=False)
    missing = [line.strip() for line in deps.stdout.splitlines() if 'not found' in line]
    manifest['missing_dependencies'] = missing
    version = run('version', prefix + [str(binary), '-version'], required=False)
    manifest['reported_version'] = version.stdout.strip()
    manifest['version_matches_release'] = version.stdout.strip() == expected_version
    manifest['success'] = deps.returncode == 0 and not missing and version.returncode == 0 and manifest['version_matches_release']
    manifest['status'] = 'compatible' if manifest['success'] else 'dependency_or_version_probe_failed'
except Exception as exc:
    manifest['status'] = 'failed'
    manifest['error'] = str(exc)
    (out / 'error.log').write_text(traceback.format_exc())
finally:
    manifest['source_sha256_end'] = digest(source)
    manifest['source_unchanged'] = manifest['source_sha256_start'] == manifest['source_sha256_end']
    if not manifest['source_unchanged']:
        manifest['success'] = False
        manifest['status'] = 'source_changed'
    manifest['time_end'] = now()
    save()
    print(json.dumps({'output': str(out), 'status': manifest['status'], 'success': manifest['success']}, indent=2), flush=True)
sys.exit(0 if manifest['success'] else 1)
PY
