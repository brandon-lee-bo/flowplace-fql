#!/usr/bin/env python3
"""Validate completed requested GRT labels and optionally normalize metadata."""
import argparse
import collections
import json
from pathlib import Path
from grt_metrics import atomic_json, parse_grt_metrics, reparse, PARSER_VERSION
from grt_provenance import stamp_provenance

parser = argparse.ArgumentParser()
parser.add_argument('tasks', type=Path)
parser.add_argument('--stamp-complete', action='store_true')
parser.add_argument('--output', type=Path)
parser.add_argument('--require-complete', action='store_true')
args = parser.parse_args()
base = Path(__file__).resolve().parents[1]
rows = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
result = {'requested': len(rows), 'validated': [], 'incomplete': [], 'errors': [],
          'physical_revisions': {}, 'reporting_revisions': {}, 'parser_revisions': {}}
groups = {key: collections.Counter() for key in ['physical_revisions', 'reporting_revisions', 'parser_revisions']}
for row in rows:
    work = base / 'results' / row['run_id']
    manifest_path = work / 'manifest.json'
    if not manifest_path.exists():
        result['incomplete'].append({'run_id': row['run_id'], 'state': 'not_started'})
        continue
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('success') or 'time_end' not in manifest:
        result['incomplete'].append({'run_id': row['run_id'], 'state': 'running_or_failed', 'failure': manifest.get('failure')})
        continue
    try:
        if row.get('def_sha256') != manifest['def_sha256']:
            raise RuntimeError('Requested/evaluated DEF hash mismatch')
        metrics = json.loads((work / 'metrics.json').read_text())
        parsed = parse_grt_metrics(work, metrics['runtime_seconds'])
        for key in ['rWL_um', 'WNS_ns', 'TNS_ns', 'Power_W', 'StdCellArea_um2',
                    'horizontal_overflow_sum', 'vertical_overflow_sum',
                    'horizontal_overflow_edges', 'vertical_overflow_edges']:
            if metrics[key] != parsed[key]:
                raise RuntimeError(f'Reparsed metric mismatch: {key}')
        if args.stamp_complete:
            if metrics.get('parser_version') != PARSER_VERSION:
                reparse(work)
                manifest = json.loads(manifest_path.read_text())
                metrics = json.loads((work / 'metrics.json').read_text())
            elif 'physical_flow_revision' not in manifest:
                stamp_provenance(manifest, base, retrospective=True)
                atomic_json(manifest_path, manifest)
        for group, value in [('physical_revisions', manifest.get('physical_flow_revision', 'missing')),
                             ('reporting_revisions', manifest.get('executed_reporting_revision', 'missing')),
                             ('parser_revisions', metrics.get('parser_version', 'legacy_v1'))]:
            groups[group][value] += 1
        result['validated'].append(row['run_id'])
    except Exception as exc:
        result['errors'].append({'run_id': row['run_id'], 'error': repr(exc)})
for key, counts in groups.items():
    result[key] = dict(counts)
result['validated_count'] = len(result['validated'])
result['complete_and_consistent'] = (len(result['validated']) == len(rows) and not result['errors']
                                     and len(result['physical_revisions']) == 1
                                     and 'missing' not in result['physical_revisions'])
if args.output:
    atomic_json(args.output, result)
print(json.dumps({k: v for k, v in result.items() if k not in ['validated', 'incomplete']}, indent=2))
if result['errors'] or (args.require_complete and not result['complete_and_consistent']):
    raise SystemExit(1)
