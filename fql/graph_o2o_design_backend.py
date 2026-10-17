"""Measure actual second-design policy actions with pinned physical settings."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .graph_o2o_archive import archive_run, atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests, validate_request
from .graph_o2o_controller import LIVE, RESTARTABLE, states
from .graph_o2o_design_policy_data import measured_context
from .graph_o2o_design_verify import verify
from .graph_o2o_geometry import audit


def measure(args):
    experiment = Path(args.experiment).resolve()
    cfg, binding, verification = measured_context(experiment)
    label_plan = json.loads(Path(cfg['initial_labels']['plan']).read_text())
    rows, tasks = load_requests(experiment, args.seed, args.round)
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    row = rows[index]
    sample = validate_request(row)
    if (row['graph'] != cfg['design'] or row['seed'] != args.seed or row['round'] != args.round or
            row['config_sha256'] != sha(experiment / 'config.json') or
            row['dataset_binding_sha256'] != sha(experiment / 'dataset_binding.json')):
        raise ValueError('Policy request crossed its netlist/data/configuration identity')
    if (sample / 'outcome.json').exists():
        from .graph_o2o_design_collect import audit_outcome
        out = audit_outcome(row, cfg, binding['reward_scale'])
        if out.get('physical_run'):
            verified = verify(out['physical_run'], row, verification)
            if verified['physical_manifest_sha256'] != out['physical_manifest_sha256']:
                raise ValueError('Existing physical outcome changed')
        print(json.dumps(dict(status='already_measured', outcome=str(sample / 'outcome.json'))), flush=True)
        return
    base = dict(version=1, run_id=row['run_id'], request_sha256=sha(sample / 'request.json'),
        tasks_sha256=sha(tasks), seed=row['seed'], round=row['round'], arm=row['arm'],
        role=row['role'], index=row['index'], graph=cfg['design'], action_sha256=row['action_sha256'],
        raw_action=row['raw_action'], executed_action=row['executed_action'],
        training_feedback_eligible=row['training_feedback_eligible'], reward_scale=binding['reward_scale'],
        dataset_binding_sha256=sha(experiment / 'dataset_binding.json'), terminal_mask=0,
        verification_plan_sha256=sha(label_plan['verification_plan']),
        requested_storage_policy='scratch', slurm_job_id=os.environ['SLURM_JOB_ID'],
        created_unix=time.time(), predicted_q=row['predicted_q'], checkpoint_sha256=row['checkpoint_sha256'])
    if row.get('pre_backend_failure'):
        failure = row['pre_backend_failure']
        if failure not in ['normalized_geometry', 'rounded_DEF_geometry']:
            raise ValueError('Unrecognized geometry failure')
        if ((failure == 'normalized_geometry' and row['geometry']['legal']) or
                (failure == 'rounded_DEF_geometry' and row.get('rounded_geometry'))):
            raise ValueError('Contradictory geometry failure')
        atomic_json(sample / 'outcome.json', dict(base, status='geometry_failure', failure_class=failure,
            metrics=None, reward=cfg['physical_failure_reward'], failure_utility_only=True,
            stage='pre_backend', detailed_route_DRC=None, not_final_table2=True))
        return
    attempts = sample / 'backend_attempts'
    attempts.mkdir(exist_ok=True)
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    key = os.environ['SLURM_JOB_ID'] + (f'_restart_{restart}' if restart else '')
    measured = None
    recovered = None
    for metadata_path in sorted(attempts.glob('*/attempt.json')):
        metadata = json.loads(metadata_path.read_text())
        if metadata['request_sha256'] != base['request_sha256']:
            raise ValueError('Previous attempt measured a different policy action')
        run = Path(metadata['physical_run'])
        manifest_path = run / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()).get('time_end'):
            measured = verify(run, row, verification)
            recovered = str(metadata_path)
            break
        if metadata['job_id'] == os.environ['SLURM_JOB_ID'] and metadata['restart_count'] < restart:
            continue
        observed = states(metadata['job_id'])
        if any(state in LIVE for state in observed) or not all(state in RESTARTABLE for state in observed):
            raise RuntimeError('Do not duplicate an unresolved backend attempt: ' + str(observed))
    if measured is None:
        name = row['run_id'] + '_a' + key
        run = ROOT / 'backend/results' / name
        attempt = attempts / key
        attempt.mkdir(exist_ok=False)
        implementation = json.loads((experiment / 'implementation_plan.json').read_text())
        for source in implementation['source_sha256']:
            dest = attempt / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, dest)
        atomic_json(attempt / 'attempt.json', dict(physical_run=str(run), request_sha256=base['request_sha256'],
            tasks_sha256=sha(tasks), job_id=os.environ['SLURM_JOB_ID'], restart_count=restart,
            implementation_plan_sha256=sha(experiment / 'implementation_plan.json'),
            source_sha256=implementation['source_sha256'], started_unix=time.time()))
        with (attempt / 'runner.log').open('w') as log:
            process = subprocess.run(['bash', str(ROOT / 'fql/graph_o2o_label.sh'),
                '--design', cfg['design'], '--def-path', row['def_path'], '--run-id', name,
                '--storage', 'scratch'], stdout=log, stderr=subprocess.STDOUT)
        atomic_json(attempt / 'process_exit.json', dict(returncode=process.returncode, ended_unix=time.time()))
        measured = verify(run, row, verification)
        if measured['status'] == 'success' and process.returncode:
            raise RuntimeError('Wrapper exit contradicts successful physical manifest')
    if measured['status'] == 'success':
        geometry = audit(measured['physical_run'])
        geometry_path = Path(measured['physical_run']) / 'backend_macro_audit.json'
        measured.update(backend_macro_geometry_path=str(geometry_path),
            backend_macro_geometry_sha256=sha(geometry_path), backend_macro_legal=geometry['legal_no_halo'],
            backend_track_snapped_macro_count=len(geometry['input_to_actual_changes']))
        if not geometry['legal_no_halo']:
            measured.update(status='physical_failure', failure_class='backend_macro_overlap_or_bounds',
                observed_metrics=measured['metrics'], metrics=None)
    atomic_json(sample / 'measurement.json', dict(base, **measured, recovered_completed_attempt=recovered))
    archive = archive_run(measured['physical_run'])
    reward = (-measured['metrics']['rWL_um'] / binding['reward_scale']
              if measured['status'] == 'success' else cfg['physical_failure_reward'])
    if not math.isfinite(reward):
        raise ValueError('Nonfinite measured reward')
    outcome = dict(base, **measured, archive=archive, reward=reward,
        failure_utility_only=measured['status'] != 'success', recovered_completed_attempt=recovered,
        finished_unix=time.time())
    atomic_json(sample / 'outcome.json', outcome)
    from .graph_o2o_design_collect import audit_outcome
    audit_outcome(row, cfg, binding['reward_scale'])
    print(json.dumps({key: outcome[key] for key in ['arm', 'role', 'round', 'index', 'status', 'reward']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--index', type=int)
    parser.add_argument('--storage', choices=['scratch'], default='scratch')
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or int(os.environ['SLURM_CPUS_PER_TASK']) != 8:
        raise RuntimeError('Run the physical evaluator through Slurm with eight threads')
    measure(args)
