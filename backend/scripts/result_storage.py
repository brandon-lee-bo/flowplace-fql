"""Allocate new GRT runs with Ceph quota checks; existing runs are never moved."""
import os
import re
import shutil
from pathlib import Path

BULK_ROOT = Path('/fact_data/bohanli/flowplace_fql_backend/grt_results')
SCRATCH_ROOT = Path('/scratch/bohanli/flowplace_graph/physical_results')
MIN_HEADROOM = 8 * 1024**3


def storage_status(quota_root, result_root):
    status = {'result_root': str(result_root), 'quota_root': str(quota_root),
              'minimum_headroom_bytes': MIN_HEADROOM,
              'filesystem_free_bytes': shutil.disk_usage(quota_root).free}
    for key, attr in [('quota_max_bytes', 'ceph.quota.max_bytes'),
                      ('quota_used_bytes', 'ceph.dir.rbytes')]:
        try:
            status[key] = int(os.getxattr(quota_root, attr))
        except OSError as exc:
            status[key] = None
            status[key + '_read_error'] = str(exc)
    if status.get('quota_max_bytes') and status.get('quota_used_bytes') is not None:
        status['quota_remaining_bytes'] = status['quota_max_bytes'] - status['quota_used_bytes']
    return status


def create_grt_run_dir(base, run_id, storage='home'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run_id must be a single alphanumeric/underscore/dot/hyphen name')
    alias = base / 'results' / run_id
    if os.path.lexists(alias):
        raise FileExistsError(f'Refusing to overwrite existing result: {alias}')
    roots = {'home': (base / 'results', Path('/fact_home/bohanli')),
             'bulk': (BULK_ROOT, Path('/fact_data/bohanli')),
             'scratch': (SCRATCH_ROOT, Path('/scratch/bohanli'))}
    if storage not in roots:
        raise ValueError('storage must be home, bulk or scratch')
    result_root, quota_root = roots[storage]
    status = storage_status(quota_root, result_root)
    status['storage_policy'] = storage
    if storage == 'scratch':
        status['storage_notice'] = 'Approved shared scratch, without bulk redundancy; preserve verified experiment evidence before cleanup'
    for key in ['filesystem_free_bytes', 'quota_remaining_bytes']:
        if key in status and status[key] < MIN_HEADROOM:
            raise RuntimeError(f'{storage} storage headroom below 8 GiB: {key}={status[key]}; no run created')
    result_root.mkdir(parents=True, exist_ok=True)
    physical = result_root / run_id
    physical.mkdir(exist_ok=False)
    if storage == 'home':
        status.update({'result_alias': str(alias), 'physical_result_dir': str(physical)})
        return alias, status
    try:
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(physical, target_is_directory=True)
    except BaseException:
        physical.rmdir()  # Only the just-created empty directory, never an existing run.
        raise
    status.update({'result_alias': str(alias), 'physical_result_dir': str(physical)})
    return alias, status
