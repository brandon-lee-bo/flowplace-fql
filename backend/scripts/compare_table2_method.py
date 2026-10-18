#!/usr/bin/env python3
"""Compare one audited Table 2 run with its own method's published values.

This checks metrics against bound raw reports; it does not by itself certify the
whole run or exact reproduction of the paper. The aggregate audits run provenance.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
METHODS = ('FlowPlace', 'DREAMPlace')
GPL_KEYS = ('GPL_TIMING_DRIVEN', 'GPL_ROUTABILITY_DRIVEN')


def load_helper(path):
    spec = importlib.util.spec_from_file_location('table2_generated_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_helper(ROOT / 'analysis/summarize_table2_generated.py')


def method_from_task(task, manifest=None, legacy_method=None):
    manifest = manifest or {}
    declared = [value for value in (task.get('method'), manifest.get('method'),
                manifest.get('task', {}).get('method')) if value is not None]
    AUDIT.require(not declared or len(set(declared)) == 1, 'Conflicting method declarations')
    method = declared[0] if declared else legacy_method
    AUDIT.require(method in METHODS, 'Task must declare method FlowPlace or DREAMPlace')
    return method, 'task_or_manifest' if declared else 'caller_declared_legacy'


def load_reference(method, path=None, expected_sha=None):
    path = Path(path or ROOT / 'backend/references' / ('paper_table2_' + method.lower() + '.json'))
    value = AUDIT.read_json(path)
    actual_sha = AUDIT.digest(path)
    AUDIT.require(not expected_sha or actual_sha == expected_sha, 'Paper reference hash mismatch')
    AUDIT.require(value.get('method') == method, 'Paper reference is for another method')
    AUDIT.require(set(value['designs']) == set(AUDIT.DESIGNS), 'Reference needs all six designs')
    for design in AUDIT.DESIGNS:
        for metric, _, _ in AUDIT.COLUMNS:
            number = value['designs'][design][metric]
            AUDIT.require(isinstance(number, (int, float)) and not isinstance(number, bool)
                          and math.isfinite(number), 'Invalid paper reference number')
    return value, {'path': str(path.resolve()), 'sha256': actual_sha,
                   'source': value['source'], 'page': value.get('page'), 'method': method}


def recipe_metadata(manifest, task=None):
    """Use recorded executed make switches, never infer them from metric quality."""
    task = task or manifest.get('task', {})
    task_overrides = {key: str(value) for key, value in task.get('make_overrides', {}).items()}
    overrides = {key: str(value) for key, value in manifest.get('make_overrides', task_overrides).items()}
    AUDIT.require(overrides == task_overrides, 'Task and executed recipe overrides differ')
    AUDIT.require(set(overrides).issubset(GPL_KEYS)
                  and all(value in ('0', '1') for value in overrides.values()), 'Unsupported recipe overrides')
    exports = manifest.get('design_config_exports', {})
    effective = {key: overrides.get(key, str(exports[key]) if key in exports else None) for key in GPL_KEYS}
    for stage in manifest.get('stages', []):
        command = stage.get('command', [])
        actual = dict(token.split('=', 1) for token in command if isinstance(token, str)
                      and any(token.startswith(key + '=') for key in GPL_KEYS))
        AUDIT.require(all(actual.get(key) == value for key, value in overrides.items()),
                      'Executed stage does not contain its declared overrides')
        AUDIT.require(all(key in overrides or effective[key] == value for key, value in actual.items()),
                      'Executed stage has an undeclared GPL setting')
    diagnostic = bool(overrides or task.get('diagnostic') or manifest.get('diagnostic')
                      or manifest.get('not_release_recipe'))
    return {'recipe_label': manifest.get('recipe_label', task.get('recipe_label', 'unrecorded')),
            'diagnostic': diagnostic, 'not_release_recipe': bool(manifest.get('not_release_recipe') or overrides),
            'make_overrides': overrides, 'effective_gpl': effective,
            'gpl_evidence': 'manifest design_config_exports plus executed command-line overrides',
            'design_config_sha256': manifest.get('design_config_sha256'),
            'num_cores': manifest.get('num_cores'), 'platform': manifest.get('platform', task.get('platform', 'nangate45')),
            'variant': manifest.get('variant', task.get('variant', 'eval'))}


def metric_rows(reference, metrics=None, raw=None):
    result = []
    for name, key, scale in AUDIT.COLUMNS:
        actual = metrics[key] * scale if metrics is not None else None
        raw_value = raw[key] * scale if raw is not None else None
        target = reference[name]
        delta = None if actual is None else actual - target
        result.append({'metric': name, 'paper_rounded': target, 'measured': actual,
                       'raw_json_value': raw_value, 'difference': delta,
                       'relative_to_abs_paper_percent': None if delta is None or target == 0 else 100 * delta / abs(target),
                       'preferred_direction': 'higher' if name in ('WNS_ns', 'TNS_ns') else 'lower'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--legacy-method', choices=METHODS, help='Explicit annotation for an old task without method; never overwrites a declaration')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = AUDIT.read_json(run / 'manifest.json')
    method, method_evidence = method_from_task(manifest.get('task', {}), manifest, args.legacy_method)
    reference, reference_info = load_reference(method, args.reference, args.reference_sha256)
    design = manifest['design']
    AUDIT.require(design in AUDIT.DESIGNS, 'Unknown Table 2 design')
    metrics, raw, audit = AUDIT.verify_metrics(run, manifest)
    report = {'schema_version': 1, 'method': method, 'method_evidence': method_evidence,
              'design': design, 'run_id': manifest['run_id'], 'status': 'validated_metrics',
              'reference': reference_info, 'recipe': recipe_metadata(manifest),
              'metrics': metric_rows(reference['designs'][design], metrics, raw),
              'supplemental': {'DRC': int(raw['DRC']), 'route_drc_clean': raw['DRC'] == 0},
              'provenance': {'manifest_sha256_at_comparison': AUDIT.digest(run / 'manifest.json'),
                             'metrics_sha256': AUDIT.digest(run / 'metrics.json'),
                             'fullflow_audit_sha256': AUDIT.digest(run / 'fullflow_audit.json'),
                             'source_report_sha256': audit['source_report_sha256'],
                             'comparator_sha256': AUDIT.digest(__file__),
                             'audit_helper_sha256': AUDIT.digest(AUDIT.__file__)},
              'limitations': ['Raw metric validation is not full-run provenance validation.',
                             'Published values are rounded; this does not establish numerical reproduction.',
                             'DRC is supplemental. No average rank over unrun competitors is claimed.']}
    output = args.output or run / 'method_paper_comparison.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('| Method / design | Metric | Paper (rounded) | Measured | Delta |')
    print('| --- | --- | ---: | ---: | ---: |')
    for row in report['metrics']:
        print(f"| {method} / {design} | {row['metric']} | {row['paper_rounded']} | {row['measured']} | {row['difference']} |")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
