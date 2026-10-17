"""Compare the same DEF under bulk and scratch storage using actual GRT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .graph_o2o_archive import atomic_json, sha, verify_archive
from .graph_o2o_backend import ROOT, load_requests, validate_request
from .graph_o2o_collect import audit_outcome, METRICS


def run(exp):
    exp = Path(exp)
    cfg = json.loads((exp/'config.json').read_text())
    requests, _ = load_requests(exp, 0, 0)
    if len(requests) != 1:
        raise ValueError('Storage check requires exactly one fixed request')
    row = requests[0]
    validate_request(row)
    reference_path = Path(row['reference_outcome_path'])
    if sha(reference_path) != row['reference_outcome_sha256']:
        raise ValueError('Original measured reference changed')
    reference = json.loads(reference_path.read_text())
    outcome = audit_outcome(row, cfg, reference['reward_scale'])
    if outcome['status'] != reference['status'] or outcome['status'] != 'success':
        raise ValueError('Storage check did not produce two successful physical outcomes')
    current = json.loads((Path(outcome['physical_run'])/'manifest.json').read_text())
    old = json.loads((Path(reference['physical_run'])/'manifest.json').read_text())
    for key in ['def_sha256', 'num_cores', 'physical_flow_revision', 'physical_flow_components',
                'tool_image_sha256', 'openroad_sha256', 'design_config_sha256']:
        if current[key] != old[key]:
            raise ValueError('Storage check changed physical input/flow: '+key)
    if current['storage']['storage_policy'] != 'scratch' or outcome['requested_storage_policy'] != 'scratch':
        raise ValueError('New physical output did not actually use scratch')
    if any(outcome['metrics'][k] != reference['metrics'][k] for k in METRICS):
        raise ValueError('Actual PPA/congestion differs under the storage change')
    archive = outcome['archive']
    if archive['storage_policy'] != 'scratch' or archive['status'] != 'verified_scratch_archive_and_cache':
        raise ValueError('Archive storage provenance is incorrect')
    inventory_path = Path(outcome['physical_run'])/'grt_intermediates_inventory.json'
    verify_archive(archive['archive_path'], json.loads(inventory_path.read_text()))
    # Refuse a second allocation with the real, already-used result name.
    sys.path.insert(0, str(ROOT/'backend/scripts'))
    from result_storage import create_grt_run_dir
    try:
        create_grt_run_dir(ROOT/'backend', Path(outcome['physical_run']).name, 'scratch')
    except FileExistsError:
        overwrite_rejected = True
    else:
        raise ValueError('Storage allocator overwrote an existing physical result')
    output = exp/'validation.json'
    atomic_json(output, dict(status='fixed_DEF_storage_equivalence_passed', slurm_job_id=os.environ['SLURM_JOB_ID'],
        source_sha256=sha(__file__), config_sha256=sha(exp/'config.json'),
        reference_outcome_path=str(reference_path), reference_outcome_sha256=sha(reference_path),
        new_outcome_path=outcome['outcome_path'], new_outcome_sha256=outcome['outcome_sha256'],
        identical_metrics={k: outcome['metrics'][k] for k in METRICS},
        unchanged_physical_flow_revision=current['physical_flow_revision'],
        decoded_archive_fully_verified=True, existing_run_overwrite_rejected=overwrite_rejected,
        scope='One exact fixed DEF; storage equivalence, not policy/multiseed performance',
        scratch_storage_not_bulk_redundancy=True))
    print(json.dumps(dict(status='passed', evidence=str(output))), flush=True)


def check_full_route(exp):
    exp = Path(exp)
    source = json.loads((exp/'full_route_reference.json').read_text())
    path = Path(source['path'])
    if sha(path) != source['sha256']:
        raise ValueError('Original detailed-route reference changed')
    old = json.loads(path.read_text())
    actual_path = exp/'full_route/outcome.json'
    actual = json.loads(actual_path.read_text())
    if actual['status'] != old['status'] or actual['status'] != 'completed':
        raise ValueError('Detailed-route storage comparison is incomplete')
    a = json.loads((Path(actual['physical_run'])/'manifest.json').read_text())
    b = json.loads((Path(old['physical_run'])/'manifest.json').read_text())
    if (sha(Path(actual['physical_run'])/'manifest.json') != actual['physical_manifest_sha256'] or
            sha(Path(old['physical_run'])/'manifest.json') != old['physical_manifest_sha256']):
        raise ValueError('Detailed-route manifests changed')
    keys = ['DRT_WL', 'WNS', 'TNS', 'Power', 'StdCellArea', 'DRC']
    if any(actual['metrics'][k] != old['metrics'][k] for k in keys):
        raise ValueError('Actual final PPA/DRC differs under the storage change')
    for key in ['source_grt_manifest_sha256', 'source_geometry_sha256', 'num_cores',
                'physical_flow_components', 'physical_flow_revision', 'route_end_iteration']:
        if a[key] != b[key]:
            raise ValueError('Detailed-route inputs or physical parameters changed: '+key)
    if a['storage']['storage_policy'] != 'scratch' or a['route_end_iteration'] != 20:
        raise ValueError('Detailed route did not use scratch and common cap20')
    if not actual['final_macros_match_audited_grt']:
        raise ValueError('Detailed route changed macro locations')
    archive = actual['archive']
    verify_archive(archive['archive_path'], json.loads(
        (Path(actual['physical_run'])/'grt_intermediates_inventory.json').read_text()))
    atomic_json(exp/'full_route_validation.json', dict(status='fixed_GRT_detailed_route_storage_equivalence_passed',
        source_sha256=sha(__file__), reference_outcome_path=str(path), reference_outcome_sha256=sha(path),
        actual_outcome_path=str(actual_path), actual_outcome_sha256=sha(actual_path),
        identical_final_metrics={k: actual['metrics'][k] for k in keys},
        decoded_archive_fully_verified=True, common_cap=20, slurm_job_id=os.environ['SLURM_JOB_ID'],
        scope='One identical source GRT database; no policy/multiseed performance claim'))
    print(json.dumps(dict(status='detailed_route_storage_check_passed')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--full-route', action='store_true')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Verify archives through Slurm')
    (check_full_route if args.full_route else run)(args.experiment)
