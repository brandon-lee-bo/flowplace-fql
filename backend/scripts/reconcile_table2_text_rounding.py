#!/usr/bin/env python3
"""Audit a terminal Table 2 run whose only failed checks are text-rounded timing.

Writes new evidence and canonical raw-JSON metrics. Never modifies the original
manifest, author metrics, audit, or physical results; never executes OpenROAD.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'analysis/summarize_table2_generated.py'
spec = importlib.util.spec_from_file_location('frozen_table2_audit', HELPER)
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)
require, digest, read_json, inside = AUDIT.require, AUDIT.digest, AUDIT.read_json, AUDIT.inside


def decimal_display_check(raw_token, text_token):
    """Two printed representations must have overlapping rounding intervals."""
    raw, displayed = Decimal(str(raw_token)), Decimal(text_token)
    require(raw.is_finite() and displayed.is_finite(), 'Nonfinite printed value')
    if raw == 0:
        require(displayed == 0, 'An exact raw zero cannot explain nonzero text by display rounding')
    raw_quantum = Decimal(1).scaleb(raw.as_tuple().exponent)
    text_quantum = Decimal(1).scaleb(displayed.as_tuple().exponent)
    difference = abs(raw - displayed)
    bound = (raw_quantum + text_quantum) / 2
    require(difference <= bound, 'Text and JSON are incompatible at their printed precision')
    return {'json_number_token': str(raw), 'text_number_token': text_token,
            'absolute_difference': str(difference), 'json_print_quantum': str(raw_quantum),
            'text_print_quantum': str(text_quantum), 'maximum_rounding_interval_distance': str(bound),
            'rounding_intervals_overlap': True}


def stage_success(manifest):
    stages = manifest.get('stages', [])
    require(len(stages) == 2, 'Expected exactly the two released physical make calls')
    for stage, target in zip(stages, ('run_mp', 'run_wo_synth')):
        command = stage.get('command', [])
        require(command[:2] == ['make', target], 'Unexpected physical stage command')
        require(stage.get('status') == 'completed' and stage.get('returncode') == 0
                and stage.get('time_end'), 'A physical make call failed or is unfinished')
    return [{'target': stage['command'][1], 'returncode': stage['returncode'],
             'wall_seconds': stage.get('wall_seconds')} for stage in stages]


def scheduler_evidence(manifest):
    job = str(manifest['slurm_job_id'])
    require(re.fullmatch(r'\d+', job), 'Unexpected original Slurm job identifier')
    command = ['sacct', '-j', job, '--format=JobIDRaw,JobID,State,ExitCode,Elapsed', '--parsable2']
    result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
    array_id = str(manifest['slurm_array_job_id']) + '_' + str(manifest['slurm_array_task_id'])
    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter='|'))
    matches = [row for row in rows if row.get('JobID') == array_id
               or (row.get('JobIDRaw') == job and '.' not in row.get('JobID', ''))]
    require(len(matches) == 1, 'Cannot identify one original terminal Slurm task')
    row = matches[0]
    require(row['State'] == 'FAILED' and row['ExitCode'] == '1:0',
            'Original task must be terminal FAILED 1:0 from its audit, not interrupted')
    return command, result.stdout, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--tasks-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'Run this provenance audit through Slurm')
    run = args.run.resolve()
    output = args.output.resolve()
    require(output.is_relative_to(Path('/fact_data/bohanli/flowplace_fql_backend')),
            'Use a new output directory below the backend /fact_data root')
    require(not output.exists(), 'Reconciliation output already exists; original evidence is never overwritten')
    require(digest(args.tasks) == args.tasks_sha256, 'Frozen task manifest differs')
    manifest = read_json(run / 'manifest.json')
    require(manifest.get('status') == 'failed' and manifest.get('time_end'), 'Run is not terminal failed')
    require(manifest.get('evaluation_complete') is True and manifest.get('success') is True,
            'Author evaluation did not successfully produce finite metrics')
    require(not manifest.get('verification_complete'), 'Run already claims successful verification')
    require(manifest.get('failure') == "RuntimeError('independent_audit failed; see retained log.')",
            'Failure is not the expected strict metric audit failure')
    require(manifest.get('independent_audit', {}).get('returncode') == 1,
            'Independent audit did not return the expected failed check status')
    physical_stages = stage_success(manifest)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    index = manifest['task_index']
    task = tasks[index]
    require(manifest.get('task_manifest_sha256') == args.tasks_sha256 and manifest.get('task') == task,
            'Run does not match the frozen task row')
    require(manifest['design'] == task['design'] and manifest['run_id'] == task['run_id'] == run.name,
            'Run/design identity differs')
    require(manifest['def_sha256'] == task['def_sha256']
            and digest(task['def_path']) == task['def_sha256']
            and digest(inside(run, manifest['executed_def_path'])) == task['def_sha256'],
            'Original or executed DEF differs from the frozen input')
    require(Path(manifest['def_path']).resolve() == Path(task['def_path']).resolve(), 'DEF path differs')
    require(digest(HELPER) == digest(run / 'provenance' / HELPER.name), 'Runtime audit helper changed')
    source_audit = AUDIT.verify_sources(run, manifest, args.tasks_sha256)
    tool = manifest['tools']['openroad']
    require(tool['sha256'] == manifest['openroad_sha256'] == task['openroad_sha256']
            and digest(tool['resolved_path']) == task['openroad_sha256'], 'OpenROAD binary identity differs')
    require(Path(tool['resolved_path']).resolve() == Path(task['openroad_exe']).resolve(), 'OpenROAD path differs')
    require(tool['version_returncode'] == 0 and tool['version_stdout'] == task['tool_version'],
            'Runtime OpenROAD version differs')
    require(manifest['tool_image_sha256'] == task['tool_image_sha256'], 'Runtime SIF identity differs')
    for filename in ('audit_fullflow.py', 'compare_table2_method.py'):
        require(digest(ROOT / 'backend/scripts' / filename) == digest(run / 'provenance' / filename),
                'Verification script changed after the original run: ' + filename)
    audit = read_json(run / 'fullflow_audit.json')
    author = read_json(run / 'metrics.json')
    require(audit.get('run_id') == run.name and audit.get('stage') == 'finish_after_detailed_route'
            and audit.get('all_checks_pass') is False, 'Missing original failed final audit')
    failed_checks = {key for key, value in audit['checks'].items() if value.get('agrees') is not True}
    require(bool(failed_checks) and failed_checks.issubset({'WNS', 'TNS'}),
            'Failed audit contains a discrepancy beyond text-rounded WNS/TNS')
    reports = {}
    report_paths = {}
    for relative, expected_sha in audit['source_report_sha256'].items():
        path = inside(run, relative)
        require(digest(path) == expected_sha, 'A final JSON report changed since the original audit')
        require(path.name not in reports, 'Duplicate final report identity')
        reports[path.name] = json.loads(path.read_text(), parse_float=Decimal, parse_int=Decimal)
        report_paths[path.name] = path
    require(set(reports) == {'5_2_route.json', '6_report.json'}, 'Missing route or finish JSON report')
    route, finish = reports['5_2_route.json'], reports['6_report.json']
    raw_decimal = {
        'DRT_WL': route['detailedroute__route__wirelength'],
        'DRC': route['detailedroute__route__drc_errors'],
        'WNS': min(Decimal(0), finish['finish__timing__setup__ws']),
        'TNS': finish['finish__timing__setup__tns'],
        'Power': finish['finish__power__total'],
        'StdCellArea': finish['finish__design__instance__area__stdcell'],
    }
    require(all(value.is_finite() for value in raw_decimal.values()), 'Nonfinite final raw metric')
    require(all(raw_decimal[key] >= 0 for key in ('DRT_WL', 'DRC', 'Power', 'StdCellArea'))
            and raw_decimal['TNS'] <= 0 and raw_decimal['DRC'] == int(raw_decimal['DRC']),
            'Invalid sign or integer count in canonical metrics')
    raw = {key: float(value) for key, value in raw_decimal.items()}
    raw['DRC'] = int(raw_decimal['DRC'])
    logs = report_paths['6_report.json'].parent
    reports_dir = run / 'orfs_work/reports' / logs.relative_to(run / 'orfs_work/logs')
    text_path = reports_dir / '6_finish.rpt'
    text = text_path.read_text()
    tokens = {}
    for key in ('WNS', 'TNS'):
        values = re.findall(r'^' + key.lower() + r'\s+(-?\d+(?:\.\d*)?)\s*$', text, re.M)
        require(len(values) == 1, 'Expected exactly one old-format timing text token: ' + key)
        tokens[key] = values[0]
    power_section = text.split('finish report_power', 1)
    require(len(power_section) == 2 and '(Watts)' in power_section[1], 'Power text unit is not explicitly Watts')
    totals = [line.split() for line in power_section[1].splitlines()
              if line.strip().startswith('Total') and '%' in line]
    require(len(totals) == 1 and len(totals[0]) >= 5, 'Ambiguous total-power text row')
    tokens['Power'] = totals[0][4]
    comparisons = {}
    for key, canonical in raw.items():
        value = author[key]
        checked = audit['checks'][key]
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                'Nonfinite or invalid author metric: ' + key)
        require(checked['author_value'] == value and checked['raw_json_value'] == canonical,
                'Original audit is not bound to author/raw metrics: ' + key)
        if key in tokens:
            require(value == float(tokens[key]), 'Author metric differs from its exact text token: ' + key)
            precision = decimal_display_check(raw_decimal[key], tokens[key])
            comparisons[key] = {'author_value': value, 'canonical_raw_json_value': canonical,
                                'source': '6_finish.rpt text token', **precision}
        else:
            require(value == canonical and checked.get('agrees') is True,
                    'Non-text metric differs from canonical JSON: ' + key)
            comparisons[key] = {'author_value': value, 'canonical_raw_json_value': canonical,
                                'source': 'raw JSON', 'exact_match': True}
    drc_path = reports_dir / '5_route_drc.rpt'
    counts = Counter(re.findall(r'^violation type:\s*(.+)$', drc_path.read_text(), re.M)) if drc_path.exists() else Counter()
    require(sum(counts.values()) == raw['DRC'], 'Detailed DRC text count differs from raw route count')
    require(audit['checks']['drc_report_count'].get('agrees') is True
            and audit['checks']['drc_report_count']['observed'] == raw['DRC'], 'Original DRC audit differs')
    scheduler_command, scheduler_text, scheduler_row = scheduler_evidence(manifest)
    original_files = [run / 'manifest.json', run / 'metrics.json', run / 'fullflow_audit.json',
                      run / 'independent_audit.log', *report_paths.values(), text_path]
    if drc_path.exists():
        original_files.append(drc_path)
    original_hashes = {str(path): digest(path) for path in original_files}
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'snapshots'
    snapshots.mkdir()
    for path in original_files:
        shutil.copyfile(path, snapshots / path.name)
    shutil.copyfile(__file__, snapshots / Path(__file__).name)
    shutil.copyfile(HELPER, snapshots / HELPER.name)
    shutil.copyfile(args.tasks, snapshots / 'tasks.jsonl')
    (output / 'scheduler_original.txt').write_text(scheduler_text)
    (output / 'canonical_metrics.json').write_text(json.dumps(raw, indent=2, allow_nan=False) + '\n')
    require(all(digest(path) == value for path, value in original_hashes.items()),
            'Original evidence changed while reconciling')
    result = {
        'schema_version': 1, 'status': 'verified_text_rounding_reconciliation',
        'created_unix': time.time(), 'reconciliation_slurm_job_id': os.environ['SLURM_JOB_ID'],
        'original_run': str(run), 'run_id': run.name, 'design': manifest['design'],
        'original_manifest_status': manifest['status'], 'original_scheduler': scheduler_row,
        'original_artifacts_unchanged': True, 'physical_execution_complete': True,
        'physical_stages': physical_stages, 'failed_original_audit_checks': sorted(failed_checks),
        'canonical_metric_policy': 'Use bound final detailed-route/finish raw JSON; WNS is min(0, setup worst slack).',
        'author_metrics_preserved': author, 'canonical_metrics': raw, 'metric_comparisons': comparisons,
        'DRC': raw['DRC'], 'route_drc_clean': raw['DRC'] == 0, 'drc_type_counts': dict(counts),
        'method': task['method'], 'diagnostic': bool(task.get('diagnostic')), 'recipe_label': manifest['recipe_label'],
        'task_manifest_sha256': args.tasks_sha256, 'task_index': index, 'def_sha256': task['def_sha256'],
        'source_audit': source_audit, 'runtime_openroad_identity': tool,
        'tool_image_sha256_recorded_at_runtime': manifest['tool_image_sha256'],
        'tool_image_rehashed_by_reconciliation': False,
        'original_evidence_sha256': original_hashes,
        'canonical_metrics_sha256': digest(output / 'canonical_metrics.json'),
        'script_sha256': digest(__file__), 'audit_helper_sha256': digest(HELPER),
        'scheduler_command': scheduler_command, 'scheduler_snapshot_sha256': digest(output / 'scheduler_original.txt'),
        'limitations': ['Original run retains FAILED state; this is a separate postprocessing correction.',
                        'This diagnostic is not automatically eligible for a formal twelve-row cohort.',
                        'Completed physical flow does not imply zero DRC or exact paper reproduction.',
                        'SIF identity was validated against the frozen task/runtime record, not rehashed here.'],
    }
    (output / 'reconciliation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(output), 'canonical_metrics': raw}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
