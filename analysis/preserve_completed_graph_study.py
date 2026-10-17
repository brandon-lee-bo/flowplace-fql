"""Resolve and preserve final native evidence only after its audited producer.

The registration pins the expected study, scope, producer and implementations.
Actual output hashes are frozen only when those completed results exist. No
model, layout, checkpoint selection or physical computation is changed here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from fql.graph_o2o_archive import atomic_json, sha
from fql.graph_o2o_final_evidence import audit_final_outcome, LEGACY_REPEAT_PLAN
from fql.graph_o2o_preserve_native_evidence import preserve

ROOT = Path(__file__).resolve().parents[1]


def completed_report(plan, report):
    if plan['kind'] == 'expanded384':
        if (report['status'] != 'expanded_offline_to_online_accounted'
                or report['report_source_sha256'] != plan['report_source_sha256']
                or not report['actual_native_route_audits']
                or not report['final_route_reports']):
            raise ValueError('Expanded GRT-only/incomplete report is not final PPA')
        binding = report['final_route_reports']['expanded384']
        if binding['path'] != plan['summary'] or sha(binding['path']) != binding['sha256']:
            raise ValueError('Expanded report does not bind the declared route summary')
    elif plan['kind'] == 'second_design':
        if (report['status'] != 'second_fixed_design_training_online_and_final_PPA_audited'
                or report['source_sha256'] != plan['report_source_sha256']
                or report['design'] != plan['design']
                or report['config_sha256'] != plan['config_sha256']
                or report['requested_final_routes'] != plan['requested']
                or report['final_route_plan_sha256'] != plan['route_plan_sha256']
                or report['slurm_job_id'] != plan['producer_job_id']):
            raise ValueError('Second-design report has the wrong completed study scope')
        if len(report['final_layouts']) != plan['requested']:
            raise ValueError('Missing final layout evidence')
        for row in report['final_layouts']:
            if sha(row['outcome_path']) != row['outcome_sha256']:
                raise ValueError('Audited final layout changed')
    else:
        raise ValueError('Unknown registered completed study')


def run(plan_path, reject_grt=None):
    plan = json.loads(plan_path.read_text())
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Registered preservation dependency changed: ' + name)
    if reject_grt is not None:
        if plan['kind'] != 'expanded384':
            raise ValueError('GRT-only rejection check requires expanded study')
        report = json.loads(reject_grt.read_text())
        if (report['status'] != 'expanded_offline_to_online_accounted'
                or report['report_source_sha256'] != plan['report_source_sha256']):
            raise ValueError('Rejection check needs the actual same-implementation GRT report')
        try:
            completed_report(plan, report)
        except ValueError as error:
            print(json.dumps(dict(status='actual_GRT_only_report_rejected', reason=str(error),
                input_sha256=sha(reject_grt), plan_sha256=sha(plan_path))), flush=True)
            return
        raise ValueError('GRT-only evidence was accepted as final PPA')

    producer = plan['producer_job_id']
    result = subprocess.run(['sacct', '-j', producer, '-X', '-n', '-P', '-o',
                             'JobIDRaw,State,ExitCode'], text=True, capture_output=True, check=True)
    states = [line.split('|') for line in result.stdout.splitlines() if line]
    if not any(row[:3] == [producer, 'COMPLETED', '0:0'] for row in states):
        raise ValueError('Actual final-audit producer has not successfully completed')
    exp = Path(plan['experiment'])
    if (sha(exp / 'config.json') != plan['config_sha256']
            or sha(exp / 'full_route_plan.json') != plan['route_plan_sha256']):
        raise ValueError('Registered study or route plan changed')
    report_path = Path(plan['report'])
    report = json.loads(report_path.read_text())
    completed_report(plan, report)
    summary_path = Path(plan['summary'])
    summary = json.loads(summary_path.read_text())
    if (summary['status'] != 'all_requested_outcomes_accounted'
            or summary['requested'] != plan['requested']
            or len(summary['records']) != plan['requested']
            or summary['plan_sha256'] != plan['route_plan_sha256']):
        raise ValueError('Incomplete or wrong final-route denominator')
    folder = plan_path.parent
    extra = {plan_path, report_path}
    for name in plan['report_files']:
        extra.add(report_path.parent / name)
    for row in summary['records']:
        extra.add(Path(row['task']['output']) / 'outcome.json')
    control_audits = []
    if plan['kind'] == 'expanded384':
        reference = report['best_measured_reference_reports']['expanded384']
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('Audited best400 reference changed')
        extra.add(Path(reference['path']))
        search = json.loads(Path(reference['path']).read_text())
        if search != reference['report'] or set(search['outcomes']) != {'bc', 'fql'}:
            raise ValueError('Supplemental control denominator changed')
        for arm, row in search['outcomes'].items():
            checked = audit_final_outcome(row['outcome'], design=plan['design'], route_cap=20,
                                          legacy_plan=LEGACY_REPEAT_PLAN)
            control_audits.append(dict(arm=arm, audit=checked))
            for physical, final in [(row['outcome']['physical_run'], True),
                                    (checked['source_run'], False)]:
                run_path = Path(physical)
                for item in run_path.iterdir():
                    if item.is_file() and item.suffix in {'.json', '.log'}:
                        extra.add(item)
                manifest = json.loads((run_path / 'manifest.json').read_text())
                imported = Path(manifest['def_path'])
                if sha(imported) != manifest['def_sha256']:
                    raise ValueError('Supplemental control import changed')
                extra.add(imported)
                for name in ['actions.npz', 'sample.json', 'request.json', 'outcome.json']:
                    if (imported.parent / name).is_file():
                        extra.add(imported.parent / name)
                for kind in ['logs', 'reports']:
                    directory = run_path / 'orfs_work' / kind / 'nangate45' / plan['design'] / 'eval'
                    for item in directory.iterdir():
                        if item.is_file() and item.suffix in {'.json', '.log', '.rpt'}:
                            extra.add(item)
                if final:
                    extra.add(run_path / 'orfs_work/results/nangate45' / plan['design'] / 'eval/6_final.def')
    controls_path = folder / 'supplemental_control_audits.json'
    controls = dict(count=len(control_audits), records=control_audits,
                    source_report_sha256=sha(report_path), new_EDA_runs=0)
    freeze(controls_path, controls)
    extra.add(controls_path)
    resolved = dict(version=1, purpose=plan['scope'],
        registration_sha256=sha(plan_path), producer_job_id=producer,
        source_sha256=plan['source_sha256'],
        summaries=[dict(path=str(summary_path), sha256=sha(summary_path),
                        requested=plan['requested'], design=plan['design'])],
        requested_logical_results=plan['requested'], output=plan['output'],
        extra_files=[dict(path=str(p), sha256=sha(p)) for p in sorted(extra)],
        supplemental_native_controls=len(control_audits), new_EDA_runs=0,
        full_EDA_intermediate_backup=False)
    resolved_path = folder / 'resolved_plan.json'
    freeze(resolved_path, resolved)
    preserve(resolved_path)
    manifest_path = Path(plan['output']) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['status'] != 'native_evidence_durably_preserved_and_fully_decoded'
            or manifest['requested_logical_results'] != plan['requested']
            or manifest['plan_sha256'] != sha(resolved_path)):
        raise ValueError('Durable native preservation does not match resolved scope')
    atomic_json(folder / 'completion.json', dict(
        status='completed_study_native_evidence_preserved', registration_sha256=sha(plan_path),
        producer_job_id=producer, final_report_sha256=sha(report_path),
        requested_route_results=plan['requested'], supplemental_native_controls=len(control_audits),
        preservation_manifest=str(manifest_path), preservation_manifest_sha256=sha(manifest_path),
        slurm_job_id=os.environ['SLURM_JOB_ID'], new_EDA_runs=0, goal_completed=False))


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Previously resolved evidence binding changed: ' + str(path))
    else:
        atomic_json(path, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reject-grt', type=Path)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for actual native auditing and compression')
    run(args.plan.resolve(), args.reject_grt)
