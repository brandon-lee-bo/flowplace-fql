#!/usr/bin/env python3
"""Audit FlowPlace and DREAMPlace Table 2 runs, retaining all 12 outcomes.

Repeat --tasks and --tasks-sha256 for separately frozen arrays. --array-job, when
used, must likewise have one value per plan (use '-' if unavailable). A selected
method/design can occur only once: report other seeds or recipes separately.
Old method-less tasks require an explicit --legacy-method annotation.
Infrastructure-only retries require an explicit hash-bound retry binding;
superseded attempts remain in the report and cannot have final PPA metrics.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
COMPARATOR_PATH = ROOT / 'backend/scripts/compare_table2_method.py'
_spec = importlib.util.spec_from_file_location('table2_method_comparison', COMPARATOR_PATH)
COMPARE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(COMPARE)
AUDIT = COMPARE.AUDIT


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_recipe_snapshot(run, manifest, recipe):
    """Bind GPL defaults to the captured config, not just mutable manifest text."""
    inventory = AUDIT.read_json(run / 'provenance/flow_inputs_start.json')
    entries = [entry for entry in inventory['files'] if manifest['design_config']
               in (entry.get('path'), entry.get('resolved_path'))]
    AUDIT.require(len(entries) == 1, 'Design config missing/ambiguous in source inventory')
    config = AUDIT.inside(run, Path('provenance/flow_snapshot') / entries[0]['relative_path'])
    AUDIT.require(AUDIT.digest(config) == manifest['design_config_sha256'] == entries[0]['sha256'],
                  'Design config identity differs from source snapshot')
    defaults = {}
    for line in config.read_text().splitlines():
        match = re.match(r'^\s*(?:export\s+)?(GPL_TIMING_DRIVEN|GPL_ROUTABILITY_DRIVEN)\s*[:?]?=\s*([01])\s*(?:#.*)?$', line)
        if match:
            defaults[match[1]] = match[2]
    for key in COMPARE.GPL_KEYS:
        AUDIT.require(key in defaults, 'GPL default cannot be established from design config: ' + key)
        expected = recipe['make_overrides'].get(key, defaults[key])
        AUDIT.require(recipe['effective_gpl'][key] == expected, 'Recorded GPL setting differs from source/command: ' + key)
    recipe['gpl_evidence'] = 'SHA256-verified design config snapshot plus recorded executed make switches'
    return recipe


def evaluate_task(plan, index, task, method, method_evidence, args, schedulers):
    run = (args.results_root / task['run_id']).resolve()
    row = {'method': method, 'method_evidence': method_evidence, 'design': task['design'],
           'run_id': task['run_id'], 'run': str(run), 'task_index': index,
           'task_manifest_sha256': plan['sha256'], 'status': 'incomplete',
           'recipe': {'diagnostic': bool(task.get('diagnostic') or task.get('make_overrides'))}}
    row['generation_protocol'] = {key: task[key] for key in
        ('generation_seed', 'generation_slurm_job', 'rng_mode', 'source_commit',
         'source_runner_sha256', 'installation_file_hashes_sha256', 'baseline_reconstruction') if key in task}
    manifest = {}
    try:
        if (run / 'manifest.json').exists():
            manifest = AUDIT.read_json(run / 'manifest.json')
            row['manifest_sha256'] = AUDIT.digest(run / 'manifest.json')
            AUDIT.require(manifest.get('design') == task['design'] and manifest.get('run_id') == task['run_id'], 'Run identity mismatch')
            AUDIT.require(manifest.get('task_manifest_sha256') == plan['sha256']
                          and manifest.get('task_index') == index and manifest.get('task') == task,
                          'Run is not bound to the selected frozen task')
            actual_method, _ = COMPARE.method_from_task(task, manifest, args.legacy_method)
            AUDIT.require(actual_method == method, 'Method identity changed')
        row['recipe'] = COMPARE.recipe_metadata(manifest, task)
        array = plan.get('array_job') or manifest.get('slurm_array_job_id')
        job_id = str(manifest.get('slurm_job_id') or (str(array) + '_' + str(index) if array else ''))
        array_id = str(array) + '_' + str(index) if array else ''
        by_run, by_job = schedulers
        scheduler = by_run.get(task['run_id'], by_job.get(array_id, by_job.get(job_id, {})))
        row.update(slurm_job_id=job_id or None, scheduler=scheduler or None)
        completed = (manifest.get('success') is True and manifest.get('evaluation_complete') is True
                     and manifest.get('verification_complete') is True and manifest.get('status') == 'completed'
                     and manifest.get('time_end'))
        if not completed:
            row['status'], row['reason'] = AUDIT.unfinished_status(manifest, scheduler)
            return row
        AUDIT.require(manifest.get('def_sha256') == task['def_sha256'], 'DEF hash differs from frozen task')
        AUDIT.require(Path(manifest['def_path']).resolve() == Path(task['def_path']).resolve(), 'DEF path differs from frozen task')
        AUDIT.require(AUDIT.digest(task['def_path']) == task['def_sha256']
                      and AUDIT.digest(AUDIT.inside(run, manifest['executed_def_path'])) == task['def_sha256'],
                      'Original or executed DEF changed')
        if task.get('geometry_audit'):
            AUDIT.require(AUDIT.digest(task['geometry_audit']) == task['geometry_audit_sha256'], 'Geometry audit changed')
            geometry = AUDIT.read_json(task['geometry_audit'])
            AUDIT.require(geometry.get('design') == task['design'] and geometry.get('output_sha256') == task['def_sha256']
                          and geometry.get('legal_no_halo') is True, 'Geometry audit does not validate this DEF')
        row['source_provenance'] = AUDIT.verify_sources(run, manifest, plan['sha256'])
        row['recipe'] = verify_recipe_snapshot(run, manifest, row['recipe'])
        metrics, raw, audit = AUDIT.verify_metrics(run, manifest)
        flow_identity = {key: manifest[key] for key in ('openroad_sha256', 'tool_image_sha256', 'flow_archive_sha256', 'backend_patch_sha256')}
        backend_identity = {**flow_identity,
                            'shared_flow_sha256': row['source_provenance']['shared_flow_sha256'],
                            'evaluator_sha256': row['source_provenance']['evaluator_sha256'],
                            'recipe': {key: row['recipe'][key] for key in ('recipe_label', 'diagnostic', 'not_release_recipe',
                                       'make_overrides', 'effective_gpl', 'num_cores', 'platform', 'variant')}}
        row.update(status='validated', def_sha256=task['def_sha256'],
                   measured={name: metrics[key] * scale for name, key, scale in AUDIT.COLUMNS},
                   raw_json={name: raw[key] * scale for name, key, scale in AUDIT.COLUMNS},
                   DRC=int(raw['DRC']), route_drc_clean=raw['DRC'] == 0, runtime_seconds=metrics.get('runtime'),
                   metrics_sha256=AUDIT.digest(run / 'metrics.json'), fullflow_audit_sha256=AUDIT.digest(run / 'fullflow_audit.json'),
                   source_report_sha256=audit['source_report_sha256'], flow_identity=flow_identity,
                   backend_identity=backend_identity, comparison_group=fingerprint(backend_identity)[:16])
        if task.get('generation_manifest'):
            source = Path(task['generation_manifest'])
            actual_sha = AUDIT.digest(source)
            expected = task.get('generation_manifest_sha256')
            AUDIT.require(not expected or actual_sha == expected, 'Frozen generation manifest changed')
            row['generation_manifest'] = {'path': str(source.resolve()), 'sha256': actual_sha,
                                           'frozen_in_task': bool(expected)}
        # Paper reference files affect comparison, not physical execution. Legacy
        # runs may contain a FlowPlace-only comparator; never reuse its numbers.
        row['paper_comparison_policy'] = 'Recomputed from verified raw metrics and the method-specific aggregate reference; legacy paper_comparison.json ignored.'
    except Exception as exc:
        row.update(status='validation_failed', reason=str(exc), error_type=type(exc).__name__)
        for key in ('measured', 'raw_json', 'comparison_group', 'backend_identity', 'DRC', 'route_drc_clean'):
            row.pop(key, None)
    return row


def markdown_table(rows):
    lines = ['| Method | Design | Status | Backend group | rWL (million µm) | WNS (ns) | TNS (ns) | Power (W) | Cell area (µm²) | DRC |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [('—' if row.get('measured', {}).get(name) is None else format(row['measured'][name], '.8g'))
                  + ' / ' + format(row['paper_rounded'][name], '.8g') for name, _, _ in AUDIT.COLUMNS]
        lines.append('| ' + ' | '.join([row['method'], row['design'], row['status'], row.get('comparison_group', '—'),
                                      *values, str(row.get('DRC', '—'))]) + ' |')
    return lines


def select_infrastructure_retries(candidates, bindings, schedulers, results_root):
    """Replace interrupted attempts only; never choose between measured PPAs."""
    by_run = {item[2]['run_id']: item for values in candidates.values() for item in values}
    removed, retry_history = set(), []
    for path, expected in bindings:
        AUDIT.require(AUDIT.digest(path) == expected, 'Infrastructure retry binding hash mismatch')
        record = AUDIT.read_json(path)
        old_id, new_id = record['old_run_id'], record['new_run_id']
        AUDIT.require(old_id in by_run and new_id in by_run and old_id != new_id,
                      'Retry must bind two supplied frozen attempts')
        old, new = by_run[old_id], by_run[new_id]
        for prefix, item in [('old', old), ('new', new)]:
            plan, index, task, _ = item
            AUDIT.require(record[prefix + '_tasks_sha256'] == plan['sha256']
                          and record[prefix + '_task_index'] == index
                          and Path(record[prefix + '_tasks']).resolve() == Path(plan['path']).resolve(),
                          'Retry binding differs from supplied frozen plan')
        AUDIT.require({k: v for k, v in old[2].items() if k != 'run_id'}
                      == {k: v for k, v in new[2].items() if k != 'run_id'},
                      'Infrastructure retry changed input, recipe, tool, seed or generation')
        AUDIT.require(old_id not in removed, 'Multiple replacements for the same interrupted attempt')
        run = results_root / old_id
        manifest = AUDIT.read_json(run / 'manifest.json')
        AUDIT.require(manifest.get('task') == old[2]
                      and manifest.get('task_manifest_sha256') == old[0]['sha256']
                      and manifest.get('task_index') == old[1], 'Interrupted attempt identity mismatch')
        job = str(manifest['slurm_array_job_id']) + '_' + str(manifest['slurm_array_task_id'])
        AUDIT.require(record['old_job'] == job, 'Retry scheduler identity mismatch')
        by_run_scheduler, by_job = schedulers
        scheduler = by_run_scheduler.get(old_id, by_job.get(job, by_job.get(str(manifest['slurm_job_id']), {})))
        state = str(scheduler.get('state', '')).split()[0].rstrip('+').upper() if scheduler else ''
        AUDIT.require(state in {'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL'},
                      'Only scheduler infrastructure failures may be superseded: ' + job)
        AUDIT.require(not manifest.get('success') and not manifest.get('evaluation_complete')
                      and not manifest.get('verification_complete'), 'Cannot replace a finished physical evaluation')
        AUDIT.require(not any((run / name).exists() for name in
                             ('metrics.json', 'author_metrics.json', 'fullflow_audit.json')),
                      'Cannot replace an attempt that already produced final metrics')
        AUDIT.require(not any((run / 'orfs_work/logs').glob('*/*/*/6_report.json')),
                      'Cannot replace a physical attempt that already wrote its final raw report')
        removed.add(old_id)
        retry_history.append({'binding': str(path.resolve()), 'binding_sha256': expected,
                              'old_run_id': old_id, 'new_run_id': new_id, 'scheduler': scheduler,
                              'old_manifest_sha256': AUDIT.digest(run / 'manifest.json'),
                              'old_manifest': manifest, 'policy': 'Identical task except run_id; no PPA-based selection'})
    selected = {}
    for key, values in candidates.items():
        remaining = [item for item in values if item[2]['run_id'] not in removed]
        AUDIT.require(len(remaining) == 1, 'Duplicate method/design without a complete infrastructure retry chain: ' + str(key))
        selected[key] = remaining[0]
    return selected, retry_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True, action='append', type=Path)
    parser.add_argument('--tasks-sha256', required=True, action='append')
    parser.add_argument('--array-job', action='append')
    parser.add_argument('--legacy-method', choices=COMPARE.METHODS)
    parser.add_argument('--scheduler-snapshot', type=Path)
    parser.add_argument('--infrastructure-retry', action='append', type=Path, default=[])
    parser.add_argument('--infrastructure-retry-sha256', action='append', default=[])
    parser.add_argument('--results-root', type=Path, default=AUDIT.RUN_ROOT)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    AUDIT.require('SLURM_JOB_ID' in os.environ, 'Submit aggregation through Slurm')
    AUDIT.require(len(args.tasks) == len(args.tasks_sha256), 'Each task plan needs its SHA256')
    AUDIT.require(not args.array_job or len(args.array_job) == len(args.tasks), 'One --array-job per plan, or omit all')
    AUDIT.require(len(args.infrastructure_retry) == len(args.infrastructure_retry_sha256),
                  'Each infrastructure retry binding needs its SHA256')
    schedulers = AUDIT.scheduler_records(args.scheduler_snapshot)
    plans, candidates, run_ids = [], {}, set()
    for number, (path, expected) in enumerate(zip(args.tasks, args.tasks_sha256)):
        AUDIT.require(re.fullmatch(r'[0-9a-f]{64}', expected) and AUDIT.digest(path) == expected, 'Frozen task manifest hash mismatch')
        array = (args.array_job or [None] * len(args.tasks))[number]
        plan = {'path': str(path.resolve()), 'sha256': expected, 'array_job': None if array == '-' else array}
        tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        AUDIT.require(tasks, 'Empty task plan')
        for index, task in enumerate(tasks):
            method, evidence = COMPARE.method_from_task(task, legacy_method=args.legacy_method)
            key = (method, task['design'])
            AUDIT.require(task['design'] in AUDIT.DESIGNS, 'Unknown Table 2 design')
            AUDIT.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', task['run_id']) and task['run_id'] not in run_ids, 'Unsafe/duplicate run ID')
            candidates.setdefault(key, []).append((plan, index, task, evidence))
            run_ids.add(task['run_id'])
        plans.append(plan)
    selected, retry_history = select_infrastructure_retries(
        candidates, list(zip(args.infrastructure_retry, args.infrastructure_retry_sha256)), schedulers, args.results_root)
    output = args.output.resolve()
    AUDIT.require(output.is_relative_to(AUDIT.REPORT_ROOT) and output != AUDIT.REPORT_ROOT, 'Output must be unique beneath ' + str(AUDIT.REPORT_ROOT))
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    source_hashes = {}
    for path in (Path(__file__), COMPARATOR_PATH, Path(AUDIT.__file__)):
        shutil.copyfile(path, snapshots / path.name)
        source_hashes[path.name] = AUDIT.digest(path)
    references = {}
    for method in COMPARE.METHODS:
        value, info = COMPARE.load_reference(method)
        shutil.copyfile(info['path'], snapshots / Path(info['path']).name)
        AUDIT.require(AUDIT.digest(snapshots / Path(info['path']).name) == info['sha256'], 'Reference changed during startup snapshot')
        references[method] = (value, info)
    for number, plan in enumerate(plans):
        shutil.copyfile(plan['path'], snapshots / ('tasks_' + str(number) + '.jsonl'))
        AUDIT.require(AUDIT.digest(snapshots / ('tasks_' + str(number) + '.jsonl')) == plan['sha256'], 'Task plan changed during startup snapshot')
    for number, (path, expected) in enumerate(zip(args.infrastructure_retry, args.infrastructure_retry_sha256)):
        target = snapshots / ('infrastructure_retry_' + str(number) + '.json')
        shutil.copyfile(path, target)
        AUDIT.require(AUDIT.digest(target) == expected, 'Retry binding changed during snapshot')
    if args.scheduler_snapshot:
        shutil.copyfile(args.scheduler_snapshot, snapshots / ('scheduler' + args.scheduler_snapshot.suffix))
    report = {'schema_version': 1, 'scope': 'Six designs × two methods, audited OpenROAD fullflow; no Table 3 ablation.',
              'created_unix': time.time(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'plans': plans,
              'source_sha256': source_hashes, 'references': {method: info for method, (_, info) in references.items()},
              'scheduler_snapshot_sha256': AUDIT.digest(args.scheduler_snapshot) if args.scheduler_snapshot else None,
              'legacy_method_annotation': args.legacy_method, 'rows': [],
              'infrastructure_retries': retry_history,
              'limitations': ['Published values are rounded. Tool versions, baseline initialization and exact author seed history may differ.',
                  'Method names in frozen tasks identify the selected upstream generation; this report does not establish paper-identical algorithms.',
                  'GPL switches, source/tool identities and diagnostics define separate comparison groups.',
                  'Completed fullflow does not establish DRC-clean signoff or exact numerical reproduction.',
                  'All 12 outcomes remain present. No successful-subset mean or five-method average rank is computed.']}
    flat = []
    for design in AUDIT.DESIGNS:
        for method in COMPARE.METHODS:
            item = selected.get((method, design))
            if item:
                plan, index, task, evidence = item
                row = evaluate_task(plan, index, task, method, evidence, args, schedulers)
            else:
                row = {'method': method, 'design': design, 'run_id': None, 'status': 'not_planned',
                       'reason': 'No frozen task supplied for this method/design'}
            row['paper_rounded'] = references[method][0]['designs'][design]
            for name, _, _ in AUDIT.COLUMNS:
                measured = row.get('measured', {}).get(name)
                target = row['paper_rounded'][name]
                delta = None if measured is None else measured - target
                flat.append({'method': method, 'design': design, 'run_id': row['run_id'], 'status': row['status'],
                             'reason': row.get('reason', ''), 'comparison_group': row.get('comparison_group'),
                             'diagnostic': row.get('recipe', {}).get('diagnostic'),
                             'effective_gpl': json.dumps(row.get('recipe', {}).get('effective_gpl', {}), sort_keys=True),
                             'metric': name, 'paper_rounded': target, 'measured': measured,
                             'raw_json_value': row.get('raw_json', {}).get(name), 'difference': delta,
                             'relative_to_abs_paper_percent': None if delta is None or target == 0 else 100 * delta / abs(target),
                             'DRC_supplemental': row.get('DRC')})
            report['rows'].append(row)
    validated = [row for row in report['rows'] if row['status'] == 'validated']
    groups = {}
    for row in validated:
        group = groups.setdefault(row['comparison_group'], {'backend_identity': row['backend_identity'], 'members': []})
        group['members'].append({'method': row['method'], 'design': row['design'], 'run_id': row['run_id']})
    pairs = []
    for design in AUDIT.DESIGNS:
        pair = [row for row in report['rows'] if row['design'] == design]
        complete = all(row['status'] == 'validated' for row in pair)
        same = (pair[0]['comparison_group'] == pair[1]['comparison_group']
                and pair[0]['recipe']['design_config_sha256'] == pair[1]['recipe']['design_config_sha256']) if complete else None
        pairs.append({'design': design, 'both_validated': complete, 'same_shared_backend': same})
    ready = len(validated) == 12 and len(groups) == 1 and all(pair['same_shared_backend'] for pair in pairs)
    report.update(status='complete' if ready else 'incomplete', requested_count=12, planned_count=len(selected),
                  validated_count=len(validated), status_counts=dict(Counter(row['status'] for row in report['rows'])),
                  comparison_groups=groups, design_pairs=pairs, comparison_ready=ready,
                  primary_comparison_ready=ready and not any(row['recipe']['diagnostic'] for row in validated),
                  same_shared_backend=ready if len(validated) == 12 else None,
                  all_route_drc_clean=all(row['route_drc_clean'] for row in validated) if len(validated) == 12 else None)
    if ready:
        report['numerical_reproduction_error'] = {}
        for method in COMPARE.METHODS:
            members = [row for row in report['rows'] if row['method'] == method]
            report['numerical_reproduction_error'][method] = {
                name: {'six_design_mean_absolute_percent_error':
                       sum(abs(row['measured'][name] - row['paper_rounded'][name])
                           / abs(row['paper_rounded'][name]) for row in members) / 6 * 100,
                       'six_design_max_absolute_percent_error':
                       max(abs(row['measured'][name] - row['paper_rounded'][name])
                           / abs(row['paper_rounded'][name]) for row in members) * 100}
                for name, _, _ in AUDIT.COLUMNS}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (output / 'metric_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# Table 2: FlowPlace and DREAMPlace', '', report['scope'], '',
             f"Validated {len(validated)}/12; common-backend comparison ready: {ready}; primary comparison ready: {report['primary_comparison_ready']}.", '',
             'Each cell is measured / rounded paper value. Missing or failed runs stay blank; they are not zero-valued results.', '']
    primary = [row for row in report['rows'] if not row.get('recipe', {}).get('diagnostic')]
    diagnostics = [row for row in report['rows'] if row.get('recipe', {}).get('diagnostic')]
    if primary:
        lines += ['## Primary / not yet classified tasks', '', *markdown_table(primary), '']
    if diagnostics:
        lines += ['## Diagnostic recipes (kept separate)', '', *markdown_table(diagnostics), '']
    lines += ['## Backend identities', '']
    for group_id, group in groups.items():
        recipe = group['backend_identity']['recipe']
        lines.append('- ' + group_id + ': ' + json.dumps(recipe, sort_keys=True))
    lines += ['', *['- ' + row['method'] + '/' + row['design'] + ': ' + str(row['reason']).replace('\n', ' ')
                    for row in report['rows'] if row.get('reason')], '', *['- ' + note for note in report['limitations']]]
    if retry_history:
        lines += ['', '## Infrastructure retries', '',
                  'Identical frozen inputs and settings; interrupted attempts and scheduler costs remain in summary.json.', '']
        lines += ['- ' + item['old_run_id'] + ' → ' + item['new_run_id'] + ': '
                  + item['scheduler']['state'] for item in retry_history]
    reconstructions = {row['generation_protocol'].get('baseline_reconstruction')
                       for row in report['rows'] if row.get('generation_protocol')}
    for note in sorted(value for value in reconstructions if value):
        lines += ['', '**Baseline reconstruction limitation:** ' + note]
    if ready:
        lines += ['', '## Numerical distance from the paper', '',
                  'All six designs per method; this measures reproduction error, not ranking against five methods. '
                  'Timing percentages are relative to the absolute published slack and can be large near zero.', '',
                  '| Method | Metric | Mean absolute error (%) | Max absolute error (%) |',
                  '| --- | --- | ---: | ---: |']
        for method, errors in report['numerical_reproduction_error'].items():
            for metric, value in errors.items():
                lines.append('| ' + method + ' | ' + metric + ' | '
                             + format(value['six_design_mean_absolute_percent_error'], '.3f') + ' | '
                             + format(value['six_design_max_absolute_percent_error'], '.3f') + ' |')
    if len(groups) > 1:
        lines += ['', '**Different backend groups are not a matched two-method comparison.**']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'validated': len(validated), 'comparison_ready': ready, 'output': str(output)}))
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
