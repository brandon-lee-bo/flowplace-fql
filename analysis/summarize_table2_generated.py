#!/usr/bin/env python3
"""Audit all six frozen generated-DEF Table 2 tasks without dropping failures.

Optional scheduler JSON: {"by_run_id": {"run_id": {"state": "PENDING",
"job_id": "123_0"}}}, or {"by_job_id": {"123_0": {"state": "FAILED"}}}.
Alternatively pass sacct --parsable2 output with JobIDRaw|State|ExitCode headers
and --array-job to map task indices before a run manifest exists.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports')
RUN_ROOT = Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')
DESIGNS = ('ariane133', 'ariane136', 'bp', 'bp_be', 'bp_fe', 'swerv_wrapper')
COLUMNS = (('rWL_million_um', 'DRT_WL', 1e-6), ('WNS_ns', 'WNS', 1),
           ('TNS_ns', 'TNS', 1), ('Power_W', 'Power', 1), ('CellArea_um2', 'StdCellArea', 1))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, path):
    path = Path(path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(resolved.is_relative_to(root.resolve()), 'Evidence path escapes run: ' + str(path))
    return resolved


def scheduler_records(path):
    if not path:
        return {}, {}
    text = path.read_text()
    if text.lstrip().startswith('{'):
        value = json.loads(text)
        return value.get('by_run_id', {}), value.get('by_job_id', {})
    rows = csv.DictReader(text.splitlines(), delimiter='|')
    require({'JobIDRaw', 'State'}.issubset(rows.fieldnames or []), 'sacct snapshot needs JobIDRaw and State headers')
    return {}, {row['JobIDRaw']: {'state': row['State'], 'exit_code': row.get('ExitCode')}
                for row in rows if row.get('JobIDRaw')}


def unfinished_status(manifest, scheduler):
    state = str(scheduler.get('state', '')).split()[0].rstrip('+').upper() if scheduler else ''
    if state in {'PENDING', 'CONFIGURING', 'REQUEUED', 'REQUEUE_HOLD', 'RESV_DEL_HOLD'}:
        return 'pending', 'Scheduler reports ' + state
    if state in {'RUNNING', 'COMPLETING', 'SUSPENDED', 'STAGE_OUT'}:
        return 'running', 'Scheduler reports ' + state
    if state in {'PREEMPTED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'FAILED', 'DEADLINE'}:
        return 'failed_' + state.lower(), manifest.get('failure', 'Scheduler reports ' + state)
    if manifest.get('time_end') and (manifest.get('status') in {'failed', 'interrupted'} or manifest.get('success') is False):
        return 'failed', manifest.get('failure', 'Terminal manifest reports unsuccessful execution')
    return 'incomplete', 'No validated terminal success; ' + ('scheduler reports ' + state if state else 'scheduler state unavailable')


def verify_sources(run, manifest, tasks_sha):
    provenance = run / 'provenance'
    require(digest(provenance / 'tasks.jsonl') == tasks_sha, 'Startup task snapshot differs')
    require(digest(provenance / 'run_table2.py') == manifest['runner_sha256'], 'Runner source snapshot changed')
    evaluator = manifest['evaluator']
    require(digest(inside(run, evaluator['path'])) == evaluator['sha256'], 'Evaluator source snapshot changed')
    starts, ends = [read_json(provenance / ('flow_inputs_' + phase + '.json')) for phase in ('start', 'end')]
    require(manifest.get('source_inputs_unchanged') is True, 'Flow inputs changed during execution')
    for phase, value in [('start', starts), ('end', ends)]:
        entries = value['files']
        canonical = [{'path': item['relative_path'], 'sha256': item['sha256'], 'size_bytes': item['size_bytes']} for item in entries]
        computed = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(computed == value['tree_sha256'] == manifest['flow_inputs_' + phase + '_sha256'], 'Flow source inventory hash differs')
    require(starts['tree_sha256'] == ends['tree_sha256'], 'Start/end source inventories differ')
    for item in starts['files']:
        relative = Path(item['relative_path'])
        if relative.suffix in {'.tcl', '.mk', '.py', '.sh', '.yaml', '.yml', '.sdc'} or relative.name == 'Makefile':
            require(digest(inside(run, provenance / 'flow_snapshot' / relative)) == item['sha256'], 'Executable/config source snapshot changed: ' + str(relative))
    # Shared flow identity excludes design-specific cached synthesis V/SDC.
    shared = [{'path': item['relative_path'], 'sha256': item['sha256']} for item in starts['files']
              if not item['relative_path'].startswith('results/')]
    return {'runner_sha256': manifest['runner_sha256'], 'evaluator_sha256': evaluator['sha256'],
            'flow_source_inventory_sha256': digest(provenance / 'flow_inputs_start.json'),
            'shared_flow_sha256': hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest(),
            'audit_source_copy_sha256': digest(provenance / 'audit_fullflow.py'),
            'image_hash_policy': 'Use recorded runtime identity; do not reread the large SIF during aggregation.'}


def verify_metrics(run, manifest):
    metrics, audit = read_json(run / 'metrics.json'), read_json(run / 'fullflow_audit.json')
    if manifest.get('metric_policy'):
        require(manifest['metric_policy'] == 'table2_raw_final_json_v1_preserve_author_text',
                'Unknown canonical metric policy')
        for key in ('author_metrics', 'canonicalizer', 'metric_canonicalization', 'canonical_metrics'):
            identity = manifest[key]
            require(digest(inside(run, identity['path'])) == identity['sha256'],
                    'Canonical metric evidence changed: ' + key)
        source = inside(run, manifest['canonicalizer']['path'])
        require(source == (run / 'provenance/canonical_table2_metrics.py').resolve(),
                'Canonicalizer must be the captured source')
        spec = importlib.util.spec_from_file_location('verified_table2_canonicalizer', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected, evidence = module.canonicalize(run, read_json(run / 'author_metrics.json'))
        require(metrics == expected and read_json(run / 'metric_canonicalization.json') == evidence,
                'Canonical metrics or text-rounding evidence differ on independent recomputation')
    require(audit.get('run_id') == manifest['run_id'] and audit.get('stage') == 'finish_after_detailed_route'
            and audit.get('all_checks_pass') is True, 'Final fullflow audit missing or unsuccessful')
    reports = {}
    for name, expected_hash in audit['source_report_sha256'].items():
        path = inside(run, name)
        require(digest(path) == expected_hash, 'Raw final report changed after audit')
        require(path.name not in reports, 'Ambiguous final report identity')
        reports[path.name] = read_json(path)
    require(set(reports) == {'6_report.json', '5_2_route.json'}, 'Expected exactly final timing and detailed-route reports')
    finish, route = reports['6_report.json'], reports['5_2_route.json']
    raw = {'DRT_WL': route['detailedroute__route__wirelength'], 'DRC': route['detailedroute__route__drc_errors'],
           'WNS': min(0., finish['finish__timing__setup__ws']), 'TNS': finish['finish__timing__setup__tns'],
           'Power': finish['finish__power__total'], 'StdCellArea': finish['finish__design__instance__area__stdcell']}
    for key, raw_value in raw.items():
        value, checked = metrics[key], audit['checks'][key]
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                and math.isfinite(raw_value), 'Invalid numerical metric ' + key)
        tolerance = 0.500001 * 10 ** (math.floor(math.log10(abs(raw_value))) - 2) if key == 'Power' and raw_value else 1e-9
        require(checked.get('agrees') is True and value == checked['author_value']
                and raw_value == checked['raw_json_value'] and abs(value - raw_value) <= tolerance,
                'Metric differs from independently bound raw report: ' + key)
    require(raw['DRC'] >= 0 and float(raw['DRC']).is_integer(), 'Invalid DRC count')
    require(audit['checks']['drc_report_count'].get('agrees') is True
            and audit['checks']['drc_report_count']['observed'] == raw['DRC'], 'DRC audit count mismatch')
    return metrics, raw, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True, type=Path)
    parser.add_argument('--tasks-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--results-root', type=Path, default=RUN_ROOT)
    parser.add_argument('--scheduler-snapshot', type=Path)
    parser.add_argument('--array-job', help='Array parent ID to resolve pre-manifest scheduler records')
    args = parser.parse_args()
    require('SLURM_JOB_ID' in os.environ, 'Submit aggregation through Slurm')
    require(re.fullmatch(r'[0-9a-f]{64}', args.tasks_sha256), 'A full lowercase SHA256 is required')
    require(digest(args.tasks) == args.tasks_sha256, 'Frozen task manifest hash mismatch')
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    require(len(tasks) == 6 and {row['design'] for row in tasks} == set(DESIGNS), 'Exactly six distinct Table 2 designs are required')
    require(len({row['run_id'] for row in tasks}) == 6, 'Duplicate run IDs')
    require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['run_id']) for row in tasks), 'Unsafe run ID')
    require(all(not row.get('make_overrides') and not row.get('diagnostic') for row in tasks), 'Diagnostic override tasks do not belong in the primary released-recipe table')
    output = args.output.resolve()
    require(output.is_relative_to(REPORT_ROOT) and output != REPORT_ROOT, 'Output must be a unique directory under ' + str(REPORT_ROOT))
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    paper_path = ROOT / 'backend/assets/paper_table2_flowplace.json'
    for path in (Path(__file__), Path(__file__).with_suffix('.slurm'), args.tasks, paper_path):
        shutil.copyfile(path, snapshots / path.name)
    scheduler_by_run, scheduler_by_job = scheduler_records(args.scheduler_snapshot)
    if args.scheduler_snapshot:
        shutil.copyfile(args.scheduler_snapshot, snapshots / ('scheduler' + args.scheduler_snapshot.suffix))
    paper = read_json(paper_path)
    report = {'scope': 'All six frozen FlowPlace generated layouts through released OpenROAD fullflow; no competitor reruns or Table 3 ablation',
              'created_unix': time.time(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'array_job': args.array_job,
              'tasks_path': str(args.tasks.resolve()), 'tasks_sha256': args.tasks_sha256,
              'source_sha256': digest(snapshots / Path(__file__).name), 'paper_values_sha256': digest(paper_path),
              'source_paper': paper['source'], 'scheduler_snapshot_sha256': digest(args.scheduler_snapshot) if args.scheduler_snapshot else None,
              'designs': [], 'limitations': ['Published values are rounded; exact author tool revision and seed history are unavailable.',
                  'Released GPL timing/routability settings differ from paper narrative; original cap4 routing may retain DRC.',
                  'Completed fullflow is not a claim of DRC-clean signoff or numerical reproduction.',
                  'All six outcomes are retained. No successful-subset mean and no five-method average ranking is computed.']}
    indexed = {task['design']: (index, task) for index, task in enumerate(tasks)}
    flat = []
    for design in DESIGNS:
        index, task = indexed[design]
        run = (args.results_root / task['run_id']).resolve()
        row = {'design': design, 'run_id': task['run_id'], 'run': str(run), 'task_index': index,
               'paper_rounded': paper['designs'][design], 'status': 'incomplete'}
        manifest = {}
        try:
            if (run / 'manifest.json').exists():
                manifest = read_json(run / 'manifest.json')
                row['manifest_sha256'] = digest(run / 'manifest.json')
                require(manifest.get('design') == design and manifest.get('run_id') == task['run_id'], 'Run identity mismatch')
                require(manifest.get('task_manifest_sha256') == args.tasks_sha256 and manifest.get('task_index') == index
                        and manifest.get('task') == task, 'Run is not bound to the frozen task')
                require(not manifest.get('diagnostic') and not manifest.get('not_release_recipe')
                        and not manifest.get('make_overrides'), 'Diagnostic recipe is not eligible for the primary comparison')
            job_id = str(manifest.get('slurm_job_id') or (str(args.array_job) + '_' + str(index) if args.array_job else ''))
            array_id = str(args.array_job or manifest.get('slurm_array_job_id', '')) + '_' + str(index)
            scheduler = scheduler_by_run.get(task['run_id'], scheduler_by_job.get(job_id, scheduler_by_job.get(array_id, {})))
            row.update(slurm_job_id=job_id or None, scheduler=scheduler or None)
            complete = (manifest.get('success') is True and manifest.get('evaluation_complete') is True
                        and manifest.get('verification_complete') is True and manifest.get('status') == 'completed' and manifest.get('time_end'))
            if not complete:
                row['status'], row['reason'] = unfinished_status(manifest, scheduler)
            else:
                require(manifest.get('def_sha256') == task['def_sha256'], 'Input DEF differs from frozen task')
                require(Path(manifest['def_path']).resolve() == Path(task['def_path']).resolve(), 'Input DEF path differs')
                require(digest(task['def_path']) == task['def_sha256']
                        and digest(inside(run, manifest['executed_def_path'])) == task['def_sha256'], 'Original or executed DEF changed')
                require(digest(run / 'provenance/paper_table2_flowplace.json') == report['paper_values_sha256'], 'Run and aggregate use different paper references')
                if task.get('geometry_audit'):
                    require(digest(task['geometry_audit']) == task['geometry_audit_sha256'], 'Frozen macro geometry audit changed')
                    geometry = read_json(task['geometry_audit'])
                    require(geometry.get('design') == design and geometry.get('output_sha256') == task['def_sha256']
                            and geometry.get('legal_no_halo') is True, 'Macro geometry evidence does not validate this input')
                row['source_provenance'] = verify_sources(run, manifest, args.tasks_sha256)
                metrics, raw, audit = verify_metrics(run, manifest)
                row.update(status='validated', def_sha256=task['def_sha256'],
                           measured={name: metrics[key] * scale for name, key, scale in COLUMNS},
                           raw_json={name: raw[key] * scale for name, key, scale in COLUMNS}, DRC=int(raw['DRC']),
                           route_drc_clean=raw['DRC'] == 0, runtime_seconds=metrics.get('runtime'),
                           metrics_sha256=digest(run / 'metrics.json'), fullflow_audit_sha256=digest(run / 'fullflow_audit.json'),
                           source_report_sha256=audit['source_report_sha256'],
                           flow_identity={key: manifest[key] for key in ('openroad_sha256', 'tool_image_sha256', 'flow_archive_sha256', 'backend_patch_sha256')})
        except Exception as exc:
            row.update(status='validation_failed', reason=str(exc), error_type=type(exc).__name__)
            row.pop('measured', None)
            row.pop('raw_json', None)
        for name, _, _ in COLUMNS:
            measured = row.get('measured', {}).get(name)
            reference = row['paper_rounded'][name]
            difference = None if measured is None else measured - reference
            flat.append({'design': design, 'run_id': task['run_id'], 'status': row['status'], 'reason': row.get('reason', ''),
                         'metric': name, 'paper_rounded': reference, 'measured': measured,
                         'raw_json_value': row.get('raw_json', {}).get(name), 'difference': difference,
                         'relative_to_abs_paper_percent': None if difference is None or reference == 0 else 100 * difference / abs(reference)})
        report['designs'].append(row)
    successful = [row for row in report['designs'] if row['status'] == 'validated']
    identities = {json.dumps({**row['flow_identity'], 'shared_flow_sha256': row['source_provenance']['shared_flow_sha256'],
                              'evaluator_sha256': row['source_provenance']['evaluator_sha256']}, sort_keys=True) for row in successful}
    report.update(status_counts=dict(Counter(row['status'] for row in report['designs'])), requested_count=6,
                  validated_count=len(successful), same_shared_flow_and_tools=len(identities) == 1 if successful else None,
                  status='complete' if len(successful) == 6 and len(identities) == 1 else 'incomplete',
                  all_route_drc_clean=all(row['route_drc_clean'] for row in successful) if len(successful) == 6 else None)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (output / 'metric_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# FlowPlace Table 2: frozen generated-layout track', '', report['scope'], '',
             f"Validated: {len(successful)}/6. Status: {report['status']}.", '',
             'Values below are measured / rounded paper value. Full precision and signed differences are in metric_comparison.csv.', '',
             '| Design | Status | rWL (million µm) | WNS (ns) | TNS (ns) | Power (W) | Cell area (µm²) | DRC |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in report['designs']:
        cells = []
        for name, _, _ in COLUMNS:
            value = row.get('measured', {}).get(name)
            cells.append(('—' if value is None else format(value, '.8g')) + ' / ' + format(row['paper_rounded'][name], '.8g'))
        lines.append('| ' + ' | '.join([row['design'], row['status'], *cells, str(row.get('DRC', '—'))]) + ' |')
    lines += ['', 'Missing/failed measurements stay blank and are not zero-valued results.', '']
    lines += ['- ' + row['design'] + ': ' + row['reason'].replace('\n', ' ') for row in report['designs'] if row.get('reason')]
    lines += ['', *['- ' + note for note in report['limitations']]]
    if len(identities) > 1:
        lines += ['', '**Shared flow/tool identities differ across successful runs; this is not a uniform comparison.**']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'validated': len(successful), 'output': str(output)}))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
