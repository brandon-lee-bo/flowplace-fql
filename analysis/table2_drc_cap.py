"""Posthoc audit correction for a proven OpenDB export cap, without changing runs."""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_frozen_metric_auditor', ROOT / 'analysis/summarize_table2_generated.py')
ORIGINAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ORIGINAL)
read_json, digest, inside, require = ORIGINAL.read_json, ORIGINAL.digest, ORIGINAL.inside, ORIGINAL.require
PROOF = ROOT / 'backend/references/openroad_drc_cap_a008522d8/manifest.json'
PROOF_SHA = '63e5b5c52491fd528641e4a49262039f24b514533da5061db49b6d0d321dc447'


def is_cap_audit_failure(manifest, audit):
    return (manifest.get('success') is True and manifest.get('evaluation_complete') is True
            and manifest.get('verification_complete') is False and manifest.get('status') == 'failed'
            and bool(manifest.get('time_end'))
            and manifest.get('failure') == "RuntimeError('independent_audit failed; see retained log.')"
            and manifest.get('independent_audit', {}).get('returncode') == 1
            and audit.get('all_checks_pass') is False
            and {k for k, v in audit['checks'].items() if v.get('agrees') is not True} == {'drc_report_count'}
            and len(manifest.get('stages', [])) == 2
            and all(s.get('status') == 'completed' and s.get('returncode') == 0 and s.get('time_end')
                    for s in manifest['stages'])
            and [s['command'][:2] for s in manifest['stages']] == [['make', 'run_mp'], ['make', 'run_wo_synth']])


def cap_evidence(run, raw, audit):
    require(digest(PROOF) == PROOF_SHA, 'OpenROAD source proof changed')
    proof = read_json(PROOF)
    for name, source in proof['files'].items():
        require(digest(PROOF.parent / name) == source['sha256'], 'Source snapshot changed')
    manifest = read_json(run / 'manifest.json')
    require(manifest['openroad_sha256'] == '93fe61454a52b927602fd6c798c4402214c19e80e7d614636345f0f427ea2f46', 'Cap rule only established for the frozen source-era binary')
    logs = list((run / 'orfs_work/logs').glob('*/*/eval/5_2_route.log'))
    require(len(logs) == 1, 'Expected one terminal native detailed-route log')
    path = logs[0]
    text = path.read_text()
    matches = list(re.finditer(r'\[INFO DRT-0199\]\s+Number of violations = (\d+)\.', text))
    require(matches and int(matches[-1][1]) == raw['DRC'], 'Final native DRC count differs from JSON')
    tail = text[matches[-1].end():]
    require('[INFO DRT-0198] Complete detail routing.' in tail, 'Detailed route did not finish')
    table = tail.split('[INFO DRT-0267]', 1)[0].strip().splitlines()
    require(table[0].startswith('Viol/Layer'), 'Missing final native violation table')
    layers = table[0].split()[1:]
    counts = Counter()
    for line in table[1:]:
        match = re.fullmatch(r'(\D*?\S)\s+(\d+(?:\s+\d+)*)\s*', line)
        require(match is not None, 'Unexpected native violation table row: ' + line)
        name, values = match[1], [int(v) for v in match[2].split()]
        require(len(values) == len(layers), 'Truncated native layer counts')
        if name == 'Short':
            counts['Cut Short'] += sum(v for layer, v in zip(layers, values) if layer.startswith('via'))
            counts['Short'] += sum(v for layer, v in zip(layers, values) if not layer.startswith('via'))
        else:
            counts[name] += sum(values)
    counts = Counter({k: v for k, v in counts.items() if v})
    require(sum(counts.values()) == raw['DRC'], 'Native table does not sum to final DRC count')
    report = run / 'orfs_work/reports' / path.parent.relative_to(run / 'orfs_work/logs') / '5_route_drc.rpt'
    observed = Counter(re.findall(r'^violation type:\s*(.+)$', report.read_text(), re.M))
    expected = Counter({k: min(v, 10000) for k, v in counts.items()})
    require(observed == expected and any(v > 10000 for v in counts.values()), 'Exported type counts do not exactly match the proven 10000-per-type cap')
    require(sum(observed.values()) == audit['checks']['drc_report_count']['observed'], 'Original failed check differs from retained text report')
    return {'policy': 'posthoc_drc_category_cap_a008522d8_v1', 'full_drc_count': raw['DRC'],
            'native_type_counts': dict(counts), 'exported_type_counts': dict(observed),
            'export_limit_per_category': 10000, 'original_audit_stays_failed': True,
            'source_proof_manifest': str(PROOF), 'source_proof_manifest_sha256': PROOF_SHA,
            'native_log': str(path), 'native_log_sha256': digest(path),
            'exported_report': str(report), 'exported_report_sha256': digest(report),
            'scope': 'All PPA values unchanged. Full native DRC stays nonzero; export truncation is not DRC cleaning.'}


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
            and (audit.get('all_checks_pass') is True or is_cap_audit_failure(manifest, audit)), 'Final fullflow audit missing or unsupported failure')
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
    if audit['checks']['drc_report_count'].get('agrees') is True:
        require(audit['checks']['drc_report_count']['observed'] == raw['DRC'], 'DRC audit count mismatch')
    else:
        require(is_cap_audit_failure(manifest, audit), 'Unexpected failed check')
        audit = dict(audit, posthoc_drc_cap_evidence=cap_evidence(run, raw, audit))
    return metrics, raw, audit


