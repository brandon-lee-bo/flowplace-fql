"""Measure one predeclared graph FQL action and preserve all physical outcomes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

from .graph_o2o_archive import archive_run, atomic_json, sha
from .graph_o2o_geometry import audit as audit_backend_geometry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend/scripts'))
from grt_metrics import parse_grt_metrics
REFERENCE = ROOT/'backend/results/bp_fe_grt_v1_row0000/manifest.json'


def load_requests(experiment, seed, round_number):
    experiment = Path(experiment).resolve()
    folder = experiment/f'seed_{seed}'/f'round_{round_number}'/'requests'
    task_path = folder/'tasks.jsonl'
    marker = json.loads((folder/'completed.json').read_text())
    if marker['tasks_sha256'] != sha(task_path) or marker['config_sha256'] != sha(experiment/'config.json'):
        raise ValueError('Changed task manifest/configuration')
    rows = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    if len(rows) != marker['count'] or len({r['run_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid request count or duplicate logical candidate')
    return rows, task_path


def validate_request(row):
    sample = Path(row['sample_dir'])
    if json.loads((sample/'request.json').read_text()) != row:
        raise ValueError('Request sidecar differs from frozen array manifest')
    with np.load(sample/'actions.npz', allow_pickle=False) as arrays:
        for key, inline in [('raw_action', 'raw_action'), ('action', 'executed_action')]:
            values = arrays[key]
            if not np.isfinite(values).all() or not np.array_equal(values, np.asarray(row[inline], dtype=values.dtype)):
                raise ValueError('Recorded raw/executed action mismatch')
        actual = hashlib.sha256(np.asarray(arrays['action'], dtype='<f4').reshape(-1).tobytes()).hexdigest()
    if actual != row['action_sha256'] or sha(row['source_checkpoint']) != row['checkpoint_sha256']:
        raise ValueError('Action/checkpoint identity mismatch')
    if row['terminal_mask'] != 0 or row['training_feedback_eligible'] != (row['role'] == 'collect'):
        raise ValueError('Invalid terminal or replay eligibility semantics')
    if 'def_path' in row:
        path = Path(row['def_path'])
        audit = json.loads(path.with_suffix('.audit.json').read_text())
        if sha(path) != row['def_sha256'] or audit['output_sha256'] != row['def_sha256']:
            raise ValueError('DEF/export audit binding mismatch')
        if bool(audit['legal_no_halo']) != bool(row['rounded_geometry']):
            raise ValueError('Rounded geometry metadata differs')
    return sample


def verify_physical(run, row, successful):
    manifest = json.loads((run/'manifest.json').read_text())
    reference = json.loads(REFERENCE.read_text())
    if manifest.get('success') != successful or not manifest.get('time_end'):
        raise ValueError('Physical run is not the claimed terminal outcome')
    if manifest['def_sha256'] != row['def_sha256'] or sha(manifest['def_path']) != row['def_sha256']:
        raise ValueError('Physical run evaluated a different DEF')
    for key in ['physical_flow_revision', 'physical_flow_components']:
        if manifest[key] != reference[key]:
            raise ValueError('Physical evaluator differs from the offline labels: ' + key)
    if int(manifest['num_cores']) != 8:
        raise ValueError('Backend thread count differs from fixed evaluator')
    result = dict(physical_run=str(run), physical_manifest_sha256=sha(run/'manifest.json'),
                  physical_flow_revision=manifest['physical_flow_revision'],
                  physical_slurm_job_id=manifest['slurm_job_id'], stage='global_route_only',
                  not_final_table2=True, detailed_route_DRC=None,
                  result_storage_policy=manifest['storage']['storage_policy'])
    if successful:
        saved = json.loads((run/'metrics.json').read_text())
        runtime = saved['runtime_seconds']
        if not math.isfinite(runtime) or not 0 < runtime <= manifest['time_end']-manifest['time_start']+1:
            raise ValueError('Invalid saved evaluation runtime')
        metrics = parse_grt_metrics(run, runtime)
        if metrics != saved:
            raise ValueError('Reparsed physical metrics differ from saved measurement')
        result.update(status='success', metrics=metrics, metrics_sha256=sha(run/'metrics.json'))
    else:
        log = run/'floorplan_place_cts.log'
        text = log.read_text() if log.exists() else ''
        if ('floorplan_place_cts exited 2' in manifest.get('failure', '') and
            '[ERROR RSZ-0060] Max buffer count reached.' in text and
            'do-4_1_cts] Error 1' in text and 'do-cts] Error 2' in text and
            not (run/'metrics.json').exists()):
            result.update(status='physical_failure', failure_class='cts_hold_repair_max_buffer_count',
                          metrics=None, failure_log_sha256=sha(log),
                          evidence='Terminal runner manifest and propagated OpenROAD RSZ-0060 / make CTS failure; no numerical PPA assigned')
        else:
            raise RuntimeError('Unrecognized physical/infrastructure/parser failure; do not invent a reward')
    return result


def measure(args):
    rows, tasks = load_requests(args.experiment, args.seed, args.round)
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    row = rows[index]
    sample = validate_request(row)
    if (sample/'outcome.json').exists():
        outcome = json.loads((sample/'outcome.json').read_text())
        if outcome['request_sha256'] != sha(sample/'request.json'):
            raise ValueError('Existing outcome does not bind this request')
        print(json.dumps(dict(status='already_measured', outcome=str(sample/'outcome.json'))), flush=True)
        return
    cfg = json.loads((Path(args.experiment)/'config.json').read_text())
    dataset_manifest = json.loads((ROOT/'datasets/bp_fe_grt_v1/manifest.json').read_text())
    base = dict(version=1, run_id=row['run_id'], request_sha256=sha(sample/'request.json'),
                tasks_sha256=sha(tasks), seed=row['seed'], round=row['round'], arm=row['arm'],
                role=row['role'], index=row['index'], action_sha256=row['action_sha256'],
                raw_action=row['raw_action'], executed_action=row['executed_action'],
                training_feedback_eligible=row['training_feedback_eligible'],
                reward_scale=dataset_manifest['reward_scale_train_only'], terminal_mask=0,
                requested_storage_policy=args.storage,
                slurm_job_id=os.environ['SLURM_JOB_ID'], created_unix=time.time(),
                predicted_q=row['predicted_q'], checkpoint_sha256=row['checkpoint_sha256'])
    if row.get('pre_backend_failure'):
        if row['pre_backend_failure'] not in ['normalized_geometry', 'rounded_DEF_geometry']:
            raise ValueError('Unrecognized pre-backend failure')
        if row['pre_backend_failure'] == 'normalized_geometry' and row['geometry']['legal']:
            raise ValueError('Contradictory geometry failure')
        if row['pre_backend_failure'] == 'rounded_DEF_geometry' and row.get('rounded_geometry'):
            raise ValueError('Contradictory rounded-DEF failure')
        outcome = dict(base, status='geometry_failure', failure_class=row['pre_backend_failure'],
                       metrics=None, reward=cfg['physical_failure_reward'], stage='pre_backend',
                       failure_utility_only=True, detailed_route_DRC=None)
        atomic_json(sample/'outcome.json', outcome)
        return
    # Recover an already finished physical attempt before scheduling a duplicate.
    attempts = sample/'backend_attempts'
    attempts.mkdir(exist_ok=True)
    measured = None
    for metadata_path in sorted(attempts.glob('*/attempt.json')):
        metadata = json.loads(metadata_path.read_text())
        run = Path(metadata['physical_run'])
        manifest_path = run/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('time_end'):
                try:
                    measured = verify_physical(run, row, bool(manifest['success']))
                except (RuntimeError, ValueError):
                    continue
                break
    if measured is None:
        run_id = row['run_id'] + '_a' + os.environ['SLURM_JOB_ID']
        run = ROOT/'backend/results'/run_id
        attempt = attempts/os.environ['SLURM_JOB_ID']
        attempt.mkdir(exist_ok=False)
        source = {}
        for name in ['fql/graph_o2o_backend.py', 'fql/graph_o2o_archive.py', 'fql/graph_o2o_label.sh',
                     'fql/graph_o2o_backend.slurm', 'backend/scripts/run_grt_label.py',
                     'backend/scripts/result_storage.py',
                     'backend/scripts/grt_metrics.py', 'backend/scripts/grt_provenance.py',
                     'fql/graph_o2o_geometry.py', 'fql/graph_o2o_geometry.sh', 'fql/graph_o2o_geometry.tcl',
                     'analysis/audit_table2_macro_preservation.py']:
            path = ROOT/name
            dest = attempt/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            source[name] = sha(path)
        atomic_json(attempt/'attempt.json', dict(physical_run=str(run), logical_run_id=row['run_id'],
                    request_sha256=sha(sample/'request.json'), tasks_sha256=sha(tasks), source_sha256=source,
                    slurm_job_id=os.environ['SLURM_JOB_ID'], array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                    array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), started_unix=time.time()))
        with (attempt/'runner.log').open('w') as log:
            process = subprocess.run(['bash', str(ROOT/'fql/graph_o2o_label.sh'), '--design', 'bp_fe',
                '--def-path', row['def_path'], '--run-id', run_id, '--storage', args.storage],
                stdout=log, stderr=subprocess.STDOUT)
        atomic_json(attempt/'process_exit.json', dict(returncode=process.returncode, ended_unix=time.time()))
        if not (run/'manifest.json').exists():
            raise RuntimeError('Evaluator failed before a physical manifest; inspect runner.log, no reward assigned')
        manifest = json.loads((run/'manifest.json').read_text())
        measured = verify_physical(run, row, bool(manifest['success']))
        if measured['status'] == 'success' and process.returncode:
            raise RuntimeError('Success manifest but unsuccessful wrapper; investigate before admission')
    if measured['status'] == 'success':
        geometry = audit_backend_geometry(measured['physical_run'])
        geometry_path = Path(measured['physical_run'])/'backend_macro_audit.json'
        measured.update(backend_macro_geometry_path=str(geometry_path),
                        backend_macro_geometry_sha256=sha(geometry_path),
                        backend_macro_legal=geometry['legal_no_halo'],
                        backend_track_snapped_macro_count=len(geometry['input_to_actual_changes']))
        if not geometry['legal_no_halo']:
            measured.update(status='physical_failure', failure_class='backend_macro_overlap_or_bounds',
                            observed_metrics=measured['metrics'], metrics=None,
                            evidence='Completed GRT but actual saved macro rectangles violate hard geometry constraints')
    # Keep raw measurement before archiving: a storage interruption must not rerun EDA.
    atomic_json(sample/'measurement.json', dict(base, **measured))
    archive = archive_run(measured['physical_run'])
    reward = (-measured['metrics']['rWL_um']/base['reward_scale'] if measured['status'] == 'success'
              else cfg['physical_failure_reward'])
    if not math.isfinite(reward):
        raise ValueError('Nonfinite measured reward')
    outcome = dict(base, **measured, reward=reward, failure_utility_only=measured['status'] != 'success',
                   archive=archive, finished_unix=time.time())
    atomic_json(sample/'outcome.json', outcome)
    print(json.dumps({k: outcome[k] for k in ['arm', 'role', 'round', 'index', 'status', 'reward']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--index', type=int)
    parser.add_argument('--storage', choices=['bulk', 'scratch'], default='bulk')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for EDA evaluation, hashing and compression')
    measure(args)


if __name__ == '__main__':
    main()
