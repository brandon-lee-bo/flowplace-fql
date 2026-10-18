"""Pinned eight-thread GRT evaluation for Q-study bp_fe/bp_be requests.

This worker reuses the existing physical runner and native validators, never
rewards infrastructure failures, and recovers completed physical measurements
before attempting another run. Raw multi-objective labels are authoritative;
the legacy reward field is only negative TRAIN-normalized GRT wirelength.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .graph_o2o_archive import archive_run, atomic_json, sha
from .graph_o2o_backend import ROOT, REFERENCE, load_requests, validate_request, verify_physical
from .graph_o2o_controller import LIVE, RESTARTABLE, states
from .graph_o2o_design_verify import context as verification_context, verify as verify_design
from .graph_o2o_geometry import audit as audit_geometry
from .qstudy_data import action_sha

BP_BE_PLAN = Path('/fact_data/bohanli/flowplace_fql/graph_o2o_second_design_v1/import_verification_recovery_plan.json')
BP_BE_PLAN_SHA = '76452fbad3fc4ae55d466ac87ac5163e44340d83b2fdde837f553835336018f5'
PHYSICAL_REVISIONS = {
    'bp_fe': '9250cb616501862e16a17fde5c8f408b7e0d51e99d1ee5f4ff8d4d27a9e4fd3e',
    'bp_be': 'd9b3186697294ae7976850ff5b057962fe5639bf68a15ad896f0615d856b080f',
}
PREFLIGHT_ROOT = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/preflight')


def read_json(path):
    return json.loads(Path(path).read_text())


def archive_measurement(experiment,row,measured):
    """Retain one immutable old-namespace preflight without widening archiver.

    A real smoke request was already executing with a qs1_ name when the
    production exporter was fixed to go2o_. Never rerun that physical result
    merely to change storage naming. Only descendants of the declared Q-study
    preflight directory receive this noncompressing retention exception.
    """
    experiment=Path(experiment).resolve()
    run=Path(measured['physical_run']).resolve()
    if (experiment.is_relative_to(PREFLIGHT_ROOT.resolve()) and
            row['run_id'].startswith('qs1_') and
            run.name.startswith(row['run_id']+'_a')):
        return dict(status='retained_preflight_legacy_namespace',physical_run=str(run),
            physical_manifest_sha256=measured['physical_manifest_sha256'],
            intermediates_compressed=False,intermediates_retained=True,
            approved_scope=str(PREFLIGHT_ROOT),
            note='Preflight-only old namespace; all native verification and PPA are unchanged')
    return archive_run(run)


def context(experiment):
    experiment = Path(experiment).resolve()
    cfg = read_json(experiment/'config.json')
    design = cfg['design']
    if design not in PHYSICAL_REVISIONS:
        raise ValueError('This worker is pinned only for bp_fe and bp_be')
    dataset = Path(cfg['dataset_path']).resolve()
    manifest_path = dataset.with_name('manifest.json')
    manifest = read_json(manifest_path)
    if (manifest['design'] != design or manifest['dataset_sha256'] != sha(dataset) or
            (cfg.get('dataset_sha256') and cfg['dataset_sha256'] != sha(dataset)) or
            manifest.get('stage') != 'global_route_only' or
            not manifest.get('no_test_or_heldout_exported') or manifest.get('historical_online_included')):
        raise ValueError('Wrong design, stage or changed initial TRAIN dataset')
    if set(manifest['physical_flow_revisions']) != {PHYSICAL_REVISIONS[design]}:
        raise ValueError('Q-study dataset differs from the pinned physical evaluator')
    scale = float(cfg.get('reward_scale', manifest['summary']['train']['metrics']['rWL_um']['mean']))
    if not math.isfinite(scale) or scale <= 0 or cfg['physical_failure_reward'] != -4:
        raise ValueError('Invalid fixed reward scale or undeclared failure utility')
    if int(cfg.get('backend_threads',8)) != 8:
        raise ValueError('The fixed evaluator requires eight threads')
    verification = None
    if design == 'bp_fe':
        reference = read_json(REFERENCE)
        if reference['physical_flow_revision'] != PHYSICAL_REVISIONS[design]:
            raise ValueError('Pinned bp_fe physical reference changed')
        reference_binding = dict(path=str(REFERENCE),sha256=sha(REFERENCE))
    else:
        plan_path = Path(cfg.get('backend_verification_plan',str(BP_BE_PLAN)))
        digest = cfg.get('backend_verification_plan_sha256',BP_BE_PLAN_SHA)
        if sha(plan_path) != digest:
            raise ValueError('Pinned bp_be verification plan changed')
        verification = read_json(plan_path)
        _,revision = verification_context(verification)
        if verification['design'] != design or revision != PHYSICAL_REVISIONS[design]:
            raise ValueError('Wrong bp_be physical verification context')
        reference_binding = dict(path=str(plan_path),sha256=digest)
    binding = dict(design=design,dataset_path=str(dataset),dataset_sha256=sha(dataset),
        dataset_manifest_sha256=sha(manifest_path),physical_flow_revision=PHYSICAL_REVISIONS[design],
        physical_reference=reference_binding,reward_scale=scale,
        config_sha256=sha(experiment/'config.json'))
    return cfg,verification,binding


def checked_request(row,args,cfg,binding):
    sample = validate_request(row)
    if (row['graph'] != cfg['design'] or row['seed'] != args.seed or row['round'] != args.round or
            row['config_sha256'] != binding['config_sha256']):
        raise ValueError('Request changed its design/config/seed/round identity')
    if action_sha(row['raw_action']) != row['raw_action_sha256']:
        raise ValueError('Raw proposal fingerprint changed')
    with np.load(sample/'actions.npz',allow_pickle=False) as arrays:
        noise = arrays['noise']
        if not np.isfinite(noise).all() or action_sha(noise) != row['noise_sha256']:
            raise ValueError('Noise identity changed')
        if noise.shape != arrays['raw_action'].shape or arrays['raw_action'].shape != arrays['action'].shape:
            raise ValueError('Noise/raw/executed coordinate shapes differ')
    if row.get('dataset_sha256',binding['dataset_sha256']) != binding['dataset_sha256']:
        raise ValueError('Request belongs to a different offline dataset')
    return sample


def verify_run(run,row,cfg,verification):
    manifest_path = Path(run)/'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('No physical manifest exists; no PPA/reward assigned')
    manifest = read_json(manifest_path)
    if cfg['design'] == 'bp_fe':
        if manifest.get('design') != 'bp_fe':
            raise ValueError('Backend run evaluated another circuit')
        measured = verify_physical(Path(run),row,bool(manifest.get('success')))
    else:
        measured = verify_design(run,row,verification)
    if measured['physical_flow_revision'] != PHYSICAL_REVISIONS[cfg['design']]:
        raise ValueError('Backend physical revision mismatch')
    if measured['status'] == 'success':
        geometry = audit_geometry(run)
        geometry_path = Path(run)/'backend_macro_audit.json'
        measured.update(backend_macro_geometry_path=str(geometry_path),
            backend_macro_geometry_sha256=sha(geometry_path),backend_macro_legal=geometry['legal_no_halo'],
            backend_track_snapped_macro_count=len(geometry['input_to_actual_changes']))
        if not geometry['legal_no_halo']:
            measured.update(status='physical_failure',failure_class='backend_macro_overlap_or_bounds',
                observed_metrics=measured['metrics'],metrics=None,
                evidence='Actual saved GRT macro rectangles violate overlap/boundary constraints')
    return measured


def checked_existing(out,row,sample,tasks,binding,cfg,verification):
    expected = dict(request_sha256=sha(sample/'request.json'),tasks_sha256=sha(tasks),
        config_sha256=binding['config_sha256'],dataset_sha256=binding['dataset_sha256'],
        raw_action_sha256=row['raw_action_sha256'],action_sha256=row['action_sha256'],
        noise_sha256=row['noise_sha256'],checkpoint_sha256=row['checkpoint_sha256'],terminal_mask=0)
    if any(out.get(k) != v for k,v in expected.items()):
        raise ValueError('Existing outcome no longer binds its immutable input')
    if out.get('physical_run'):
        verified = verify_run(out['physical_run'],row,cfg,verification)
        for key in ['status','metrics','physical_manifest_sha256','physical_flow_revision']:
            if verified[key] != out[key]:
                raise ValueError('Existing outcome no longer matches its native physical result: '+key)
    elif out['status'] != 'geometry_failure' or out['metrics'] is not None:
        raise ValueError('Outcome without physical run is not an admitted geometry failure')
    expected_reward = (-out['metrics']['rWL_um']/binding['reward_scale'] if out['status']=='success'
                       else cfg['physical_failure_reward'])
    if not math.isclose(out['reward'],expected_reward,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Outcome reward disagrees with native measurement/failure utility')


def measure(args):
    cfg,verification,binding = context(args.experiment)
    rows,tasks = load_requests(args.experiment,args.seed,args.round)
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(rows):
        raise IndexError('Slurm task is outside the immutable request manifest')
    row = rows[index]
    sample = checked_request(row,args,cfg,binding)
    with (sample/'.qstudy_backend.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (sample/'outcome.json').exists():
            out=read_json(sample/'outcome.json')
            checked_existing(out,row,sample,tasks,binding,cfg,verification)
            print(json.dumps(dict(status='already_measured',outcome=str(sample/'outcome.json'))),flush=True)
            return
        storage=args.storage or cfg.get('backend_storage','scratch')
        base=dict(version=1,run_id=row['run_id'],request_sha256=sha(sample/'request.json'),
            tasks_sha256=sha(tasks),config_sha256=binding['config_sha256'],
            dataset_sha256=binding['dataset_sha256'],dataset_manifest_sha256=binding['dataset_manifest_sha256'],
            seed=args.seed,round=args.round,arm=row['arm'],role=row['role'],index=row['index'],graph=cfg['design'],
            action_sha256=row['action_sha256'],raw_action_sha256=row['raw_action_sha256'],noise_sha256=row['noise_sha256'],
            raw_action=row['raw_action'],executed_action=row['executed_action'],
            training_feedback_eligible=row['training_feedback_eligible'],terminal_mask=0,
            reward_scale=binding['reward_scale'],physical_reference=binding['physical_reference'],
            requested_storage_policy=storage,slurm_job_id=os.environ['SLURM_JOB_ID'],created_unix=time.time(),
            predicted_q=row.get('predicted_q'),checkpoint_sha256=row['checkpoint_sha256'],
            reward_note='Legacy negative TRAIN-normalized GRT WL only; multiobjective targets derive from preserved raw metrics')
        if row.get('pre_backend_failure'):
            failure=row['pre_backend_failure']
            if failure not in ['normalized_geometry','rounded_DEF_geometry']:
                raise ValueError('Unsupported pre-backend failure')
            if ((failure=='normalized_geometry' and row['geometry']['legal']) or
                    (failure=='rounded_DEF_geometry' and row.get('rounded_geometry'))):
                raise ValueError('Contradictory geometry failure')
            atomic_json(sample/'outcome.json',dict(base,status='geometry_failure',failure_class=failure,
                metrics=None,reward=cfg['physical_failure_reward'],failure_utility_only=True,
                stage='pre_backend',detailed_route_DRC=None,not_final_table2=True))
            return
        attempts=sample/'backend_attempts';attempts.mkdir(exist_ok=True)
        restart=int(os.environ.get('SLURM_RESTART_COUNT','0'))
        key=os.environ['SLURM_JOB_ID']+(f'_restart_{restart}' if restart else '')
        measured=None;recovered=None
        for metadata_path in sorted(attempts.glob('*/attempt.json')):
            metadata=read_json(metadata_path)
            if metadata['request_sha256'] != base['request_sha256'] or metadata['config_sha256'] != binding['config_sha256']:
                raise ValueError('Prior attempt belongs to another request/configuration')
            run=Path(metadata['physical_run']);mp=run/'manifest.json'
            if mp.exists() and read_json(mp).get('time_end'):
                # Terminal unknown failures must be diagnosed, not silently assigned failure PPA.
                measured=verify_run(run,row,cfg,verification)
                recovered=str(metadata_path);break
            if metadata['job_id']==os.environ['SLURM_JOB_ID'] and metadata['restart_count']<restart:
                continue
            observed=states(metadata['job_id'])
            if not observed or any(state in LIVE for state in observed) or not all(state in RESTARTABLE for state in observed):
                raise RuntimeError('Do not duplicate an unresolved/live physical attempt: '+str(observed))
        if measured is None:
            name=row['run_id']+'_a'+key
            run=ROOT/'backend/results'/name
            attempt=attempts/key;attempt.mkdir(exist_ok=False)
            source_names=['fql/qstudy_backend.py','fql/qstudy_backend.slurm','fql/qstudy_data.py',
                'fql/graph_o2o_backend.py','fql/graph_o2o_design_verify.py','fql/graph_o2o_archive.py',
                'fql/graph_o2o_label.sh','fql/graph_o2o_geometry.py','fql/graph_o2o_geometry.sh',
                'fql/graph_o2o_geometry.tcl','backend/scripts/run_grt_label.py',
                'backend/scripts/result_storage.py','backend/scripts/grt_metrics.py',
                'backend/scripts/grt_provenance.py','analysis/audit_table2_macro_preservation.py']
            source={}
            for relative in source_names:
                dest=attempt/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,dest);source[relative]=sha(ROOT/relative)
            atomic_json(attempt/'attempt.json',dict(physical_run=str(run),logical_run_id=row['run_id'],
                request_sha256=base['request_sha256'],tasks_sha256=sha(tasks),config_sha256=binding['config_sha256'],
                dataset_sha256=binding['dataset_sha256'],source_sha256=source,job_id=os.environ['SLURM_JOB_ID'],
                restart_count=restart,array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),started_unix=time.time()))
            with (attempt/'runner.log').open('w') as log:
                process=subprocess.run(['bash',str(ROOT/'fql/graph_o2o_label.sh'),'--design',cfg['design'],
                    '--def-path',row['def_path'],'--run-id',name,'--storage',storage],
                    stdout=log,stderr=subprocess.STDOUT)
            atomic_json(attempt/'process_exit.json',dict(returncode=process.returncode,ended_unix=time.time()))
            measured=verify_run(run,row,cfg,verification)
            if measured['status']=='success' and process.returncode:
                raise RuntimeError('Successful physical manifest contradicts failed runner; no admission')
        # This durable small record is written before archiving can fail.
        atomic_json(sample/'measurement.json',dict(base,**measured,recovered_completed_attempt=recovered))
        archive=archive_measurement(args.experiment,row,measured)
        reward=(-measured['metrics']['rWL_um']/binding['reward_scale'] if measured['status']=='success'
                else cfg['physical_failure_reward'])
        if not math.isfinite(reward):
            raise ValueError('Nonfinite measured reward')
        outcome=dict(base,**measured,archive=archive,reward=reward,
            failure_utility_only=measured['status']!='success',recovered_completed_attempt=recovered,
            finished_unix=time.time())
        atomic_json(sample/'outcome.json',outcome)
        checked_existing(outcome,row,sample,tasks,binding,cfg,verification)
        print(json.dumps({k:outcome[k] for k in ['arm','role','round','index','status','reward']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--round',type=int,default=0)
    parser.add_argument('--index',type=int)
    parser.add_argument('--storage',choices=['bulk','scratch'])
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=8:
        raise RuntimeError('The physical evaluator requires Slurm with exactly eight allocated CPUs')
    measure(args)
