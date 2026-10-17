"""Recover fixed import probes with an explicitly design-specific verifier.

Original probe plans, failed worker logs, actions and terminal physical files
are immutable. Completed successful EDA is audited/archived, never rerun.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .graph_o2o_archive import archive_run, atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_controller import states, LIVE, RESTARTABLE
from .graph_o2o_design_verify import context, verify
from .graph_o2o_geometry import audit


def run(args):
    repair_path = Path(args.verification_plan).resolve()
    repair = json.loads(repair_path.read_text())
    plan_path = Path(repair['original_probe_plan'])
    plan = json.loads(plan_path.read_text())
    if sha(plan_path) != repair['original_probe_plan_sha256']:
        raise ValueError('Original fixed-action probe plan changed')
    context(repair)
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Original probe implementation changed: ' + name)
    design_path = Path(plan['design_plan'])
    design = json.loads(design_path.read_text())
    if sha(design_path) != plan['design_plan_sha256'] or design['design'] != repair['design']:
        raise ValueError('Original design selection changed')
    audit_path = Path(design['audit_output']) / 'evidence.json'
    evidence = json.loads(audit_path.read_text())
    if (evidence['plan_sha256'] != plan['design_plan_sha256'] or
            evidence['status'] != 'behavior_data_audited_no_real_rewards_yet'):
        raise ValueError('Original data audit changed')
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    family = plan['TRAIN_families'][index]
    item = evidence['import_probes'][index]
    probe_path = Path(item['path'])
    row = json.loads(probe_path.read_text())
    if (sha(probe_path) != item['sha256'] or row != item['record'] or
            row['source_family'] != family or row['real_backend_reward'] is not None):
        raise ValueError('Original fixed probe action changed')
    output = Path(plan['output']) / f'family_{family:04d}'
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(plan_sha256=sha(plan_path), behavior_audit_sha256=sha(audit_path),
        probe_path=str(probe_path), probe_sha256=sha(probe_path), design=design['design'],
        source_family=family, source_row_index=row['source_row_index'],
        stage='global_route_only', not_final_table2=True, storage='scratch', threads=8,
        TRAIN_only=True, policy_feedback=False, reward=None, reward_scale=None,
        note='Real PPA/import feasibility probe; no RL training or final detailed-route DRC claim')
    outcome_path = output / 'outcome.json'
    if outcome_path.exists():
        out = json.loads(outcome_path.read_text())
        if out['spec'] != spec or out['verification_plan_sha256'] != sha(repair_path):
            raise ValueError('Existing recovered result has different bindings')
        if out.get('physical_run'):
            verify(out['physical_run'], row, repair)
        print(json.dumps(dict(status='already_recovered', family=family)), flush=True)
        return
    if not row['geometry']['legal'] or not row.get('rounded_DEF_legal'):
        atomic_json(outcome_path, dict(spec=spec, status='geometry_failure', metrics=None,
            failure_class='normalized_geometry' if not row['geometry']['legal'] else 'rounded_DEF_geometry',
            verification_plan_sha256=sha(repair_path), slurm_job_id=os.environ['SLURM_JOB_ID']))
        return
    if sha(row['def_path']) != row['def_sha256']:
        raise ValueError('The exact fixed DEF changed')
    attempts = output / 'attempts'
    attempts.mkdir(exist_ok=True)
    restart = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    key = os.environ['SLURM_JOB_ID'] + (f'_restart_{restart}' if restart else '')
    measured = None
    recovered_from = None
    for path in sorted(attempts.glob('*/attempt.json')):
        previous = json.loads(path.read_text())
        if previous['spec'] != spec:
            raise ValueError('Historical probe attempt has a different action/spec')
        physical = Path(previous['physical_run'])
        manifest_path = physical / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()).get('time_end'):
            measured = verify(physical, row, repair)
            recovered_from = str(path)
            break
        if previous['job_id'] == os.environ['SLURM_JOB_ID'] and previous['restart_count'] < restart:
            continue
        observed = states(previous['job_id'])
        allowed = RESTARTABLE | {'CANCELLED'}
        if any(state in LIVE for state in observed) or not all(state in allowed for state in observed):
            raise RuntimeError('Do not duplicate a live/unresolved previous attempt: ' + str(observed))
        if 'CANCELLED' in observed and previous['job_id'] not in repair['cancelled_pending_job_ids']:
            raise RuntimeError('Cancellation lacks the explicit pending-only scheduler recovery record')
    if measured is None:
        name = f"go2o_{design['design']}_import_f{family:04d}_a{key}"
        physical = ROOT / 'backend/results' / name
        attempt = attempts / key
        attempt.mkdir(exist_ok=False)
        for source in repair['source_sha256']:
            dest = attempt / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, dest)
        atomic_json(attempt / 'attempt.json', dict(spec=spec, physical_run=str(physical),
            job_id=os.environ['SLURM_JOB_ID'], restart_count=restart, started_unix=time.time(),
            verification_plan_sha256=sha(repair_path)))
        with (attempt / 'runner.log').open('w') as log:
            process = subprocess.run(['bash', str(ROOT / 'fql/graph_o2o_label.sh'),
                '--design', design['design'], '--def-path', row['def_path'],
                '--run-id', name, '--storage', 'scratch'], stdout=log, stderr=subprocess.STDOUT)
        atomic_json(attempt / 'process_exit.json', dict(returncode=process.returncode, ended_unix=time.time()))
        measured = verify(physical, row, repair)
        if measured['status'] == 'success' and process.returncode:
            raise RuntimeError('Wrapper exit contradicts successful terminal evidence')
    if measured['status'] == 'success':
        geometry = audit(measured['physical_run'])
        geometry_path = Path(measured['physical_run']) / 'backend_macro_audit.json'
        measured.update(backend_geometry_path=str(geometry_path), backend_geometry_sha256=sha(geometry_path),
                        backend_macro_legal=geometry['legal_no_halo'])
        if not geometry['legal_no_halo']:
            measured.update(status='physical_failure', failure_class='backend_macro_overlap_or_bounds',
                            observed_metrics=measured['metrics'], metrics=None)
    binding = dict(verification_plan_path=str(repair_path), verification_plan_sha256=sha(repair_path),
                   recovered_completed_attempt=recovered_from,
                   reran_completed_EDA=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output / 'measurement.json', dict(spec=spec, **measured, **binding))
    archive = archive_run(measured['physical_run'])
    atomic_json(outcome_path, dict(spec=spec, **measured, **binding, archive=archive, finished_unix=time.time()))
    print(json.dumps(dict(status=measured['status'], family=family, recovered_from=recovered_from,
                         reward=None, metrics=measured['metrics'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verification-plan', required=True)
    p.add_argument('--index', type=int)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or int(os.environ['SLURM_CPUS_PER_TASK']) != 8:
        raise RuntimeError('Use Slurm with eight CPU threads')
    run(args)
