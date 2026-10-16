#!/usr/bin/env python3
"""Audit completed fullflows against raw final-stage reports; no optimization."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from grt_metrics import atomic_json

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
a = p.parse_args()
run = a.run.resolve()
manifest = json.loads((run / 'manifest.json').read_text())
metrics = json.loads((run / 'metrics.json').read_text())
if not manifest.get('success'):
    raise SystemExit('Fullflow has not completed successfully')
log_dirs = list((run / 'orfs_work/logs').glob('*/*/*'))
log_dirs = [d for d in log_dirs if (d / '6_report.json').exists()]
if len(log_dirs) != 1:
    raise SystemExit('Expected one final report directory')
logs = log_dirs[0]
reports = run / 'orfs_work/reports' / logs.relative_to(run / 'orfs_work/logs')
finish_path, route_path = logs / '6_report.json', logs / '5_2_route.json'
finish = json.loads(finish_path.read_text())
route = json.loads(route_path.read_text())
checks = {}
expected = {
    'DRT_WL': route['detailedroute__route__wirelength'],
    'DRC': route['detailedroute__route__drc_errors'],
    'WNS': min(0.0, finish['finish__timing__setup__ws']),
    'TNS': finish['finish__timing__setup__tns'],
    'StdCellArea': finish['finish__design__instance__area__stdcell'],
    'Power': finish['finish__power__total'],
}
for key, raw in expected.items():
    value = metrics[key]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f'Nonfinite final metric: {key}')
    # The unmodified author's Power parser reads a 3-significant-digit text
    # report, whereas final JSON preserves more digits. Other fields are exact.
    tolerance = (0.500001 * 10 ** (math.floor(math.log10(abs(raw))) - 2)
                 if key == 'Power' and raw != 0 else 1e-9)
    checks[key] = {'author_value': value, 'raw_json_value': raw,
                   'absolute_tolerance': tolerance,
                   'agrees': abs(value - raw) <= tolerance}
drc_path = reports / '5_route_drc.rpt'
counts = Counter(re.findall(r'^violation type:\s*(.+)$', drc_path.read_text(), re.M)) if drc_path.exists() else Counter()
if not drc_path.exists() and expected['DRC'] != 0:
    raise SystemExit('Positive DRC count without detailed violation report')
checks['drc_report_count'] = {'observed': sum(counts.values()), 'expected': expected['DRC'],
                             'agrees': sum(counts.values()) == expected['DRC']}
runtime = {}
for path in sorted(logs.glob('*.log')):
    text = path.read_text()
    elapsed = re.findall(r'Elapsed time: ([0-9:.]+)\[h:\]min:sec', text)
    peak = re.findall(r'Peak memory: (\d+)KB', text)
    if elapsed:
        seconds = 0.0
        for item in elapsed[-1].split(':'):
            seconds = seconds * 60 + float(item)
        runtime[path.name] = {'elapsed_seconds': seconds, 'peak_memory_KB': int(peak[-1]) if peak else None}
area = {k: v for k, v in finish.items() if k in [
    'finish__design__instance__area', 'finish__design__instance__area__stdcell',
    'finish__design__instance__area__macros']}
result = {'run_id': run.name, 'stage': 'finish_after_detailed_route',
          'all_checks_pass': all(c['agrees'] for c in checks.values()),
          'route_drc_clean': expected['DRC'] == 0, 'checks': checks,
          'area_accounting_raw_json': area,
          'source_report_sha256': {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [finish_path, route_path]}}
atomic_json(run / 'stage_runtimes.json', runtime)
atomic_json(run / 'drc_type_counts.json', dict(counts))
atomic_json(run / 'fullflow_audit.json', result)
print(json.dumps(result, indent=2))
if not result['all_checks_pass']:
    raise SystemExit(1)
