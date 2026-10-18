#!/bin/bash
#SBATCH --job-name=t2_keep_bp_archive_recovery
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_keep_bp_archive_recovery_%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 - <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
scripts=Path('/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/scripts')
helper=scripts/'archive_interrupted_table2_staged.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='69c9bdd66c95889f4317d7207e475f222f6fde26b266130753422c8134b0ec1a'
sys.path.insert(0,str(scripts));import archive_interrupted_table2_staged as archive
parent=archive.ARCHIVE_ROOT/'table2_source_era_flowplace_v1_bp'
source=parent/'interrupted_preservation_v1'
destination=parent/'interrupted_preservation_recover_v2'
with (archive.ARCHIVE_ROOT/'.archive.lock').open('a') as lock:
    archive.acquire_lock(lock)
    state=json.loads((source/'commit_state.json').read_text())
    assert state['status']=='copying' and state['backup_paths']=={} and state['job_id']=='1788237'
    record,entries=archive.load_verified(source)
    run=Path(record['run'])
    assert all((run/relative).is_dir() and not (run/relative).is_symlink() for relative in archive.TREES)
    assert archive.content_identity(archive.inventory(run))==archive.content_identity(entries)
    eligibility=record['source_eligibility']
    archive.RETRY_BINDING=Path(eligibility['retry_binding'])
    archive.RETRY_BINDING_SHA256=eligibility['retry_binding_sha256']
    assert archive.interrupted_run(run)==eligibility
    destination.mkdir(exist_ok=False)
    for name in ['archive_manifest.json','source_inventory.json','archive_interrupted_table2_staged.py','archive_interrupted_table2_staged.slurm']:
        shutil.copyfile(source/name,destination/name)
    os.link(source/'artifacts.tar.zst',destination/'artifacts.tar.zst')
    assert archive.sha(destination/'artifacts.tar.zst')==record['archive_sha256']
    shutil.copyfile('/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/plans/table2_background_v1/recover_interrupted_bp_archive.slurm',destination/'recovery.slurm')
    archive.save(destination/'recovery_origin.json',{'source':str(source),'failed_commit_job':'1788237','failure':'OUT_OF_MEMORY before any source rename','original_commit_state_sha256':archive.sha(source/'commit_state.json'),'source_originals_unchanged':True,'durable_archive_reused_by_hardlink':True,'recovery_job':os.environ['SLURM_JOB_ID'],'memory_GiB':8,'old_partial_scratch_cache_retained':state['scratch']})
    archive.sync_directory(destination)
PY
exec python3 /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/scripts/archive_interrupted_table2_staged.py commit \
  --archive-dir /fact_data/bohanli/flowplace_fql_backend/table2_archives/table2_source_era_flowplace_v1_bp/interrupted_preservation_recover_v2 \
  --scratch-dir /scratch/bohanli/flowplace_table2/artifact_cache/table2_source_era_flowplace_v1_bp/interrupted_preservation_recover_v2 \
  --retry-binding /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/plans/table2_source_era_flowplace_retry2/retry_binding_1.json \
  --retry-binding-sha256 bbd1cbfa8b0741e0a61cd55cc976701459fc0c282d0860204bd4dbba911499fa \
  --threads "$SLURM_CPUS_PER_TASK" --confirm-remove-bulk-backup
