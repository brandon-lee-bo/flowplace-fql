#!/usr/bin/env python3
"""Validate author metrics and return uniform Table 2 raw-JSON PPA values.

API: canonicalize(run: Path, author_result: dict) -> (canonical, evidence).
The caller must first persist author_result verbatim as author_metrics.json.
This module only reads evidence. The caller persists returned values and binds
this module's source hash to its run. This is metric validation, not a proof of
successful physical execution or full-run provenance.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

METRIC_POLICY = 'table2_raw_final_json_v1_preserve_author_text'
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
FIELDS = {
    'DRT_WL': ('5_2_route.json', 'detailedroute__route__wirelength'),
    'DRC': ('5_2_route.json', 'detailedroute__route__drc_errors'),
    'WNS': ('6_report.json', 'finish__timing__setup__ws'),
    'TNS': ('6_report.json', 'finish__timing__setup__tns'),
    'Power': ('6_report.json', 'finish__power__total'),
    'StdCellArea': ('6_report.json', 'finish__design__instance__area__stdcell'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def precision_evidence(raw, token):
    """Validate only the overlap of these two specific printed intervals."""
    raw, displayed = Decimal(raw), Decimal(token)
    require(raw.is_finite() and displayed.is_finite(), 'Nonfinite report token')
    if raw == 0:
        require(displayed == 0, 'A raw zero cannot justify a nonzero text metric')
    json_quantum = Decimal(1).scaleb(raw.as_tuple().exponent)
    text_quantum = Decimal(1).scaleb(displayed.as_tuple().exponent)
    difference = abs(raw - displayed)
    bound = (json_quantum + text_quantum) / 2
    require(difference <= bound, 'Bound text differs from JSON beyond its printed precision')
    return {'raw_json_number_token': str(raw), 'text_number_token': token,
            'absolute_difference': str(difference), 'json_print_quantum': str(json_quantum),
            'text_print_quantum': str(text_quantum), 'rounding_interval_bound': str(bound),
            'rounding_intervals_overlap': True}


def canonicalize(run: Path, author_result: dict):
    run = Path(run).resolve()
    require(isinstance(author_result, dict), 'Author result must be a dictionary')
    captured = {}

    def capture(path):
        path = Path(path).resolve()
        require(path.is_relative_to(run), 'Metric evidence escapes its run directory')
        data = path.read_bytes()
        captured[path] = data
        return data

    author_path = run / 'author_metrics.json'
    author_data = capture(author_path)
    require(json.loads(author_data) == author_result, 'Persisted author result differs from the supplied result')
    finish_paths = list((run / 'orfs_work/logs').glob('*/*/*/6_report.json'))
    require(len(finish_paths) == 1, 'Expected exactly one final report directory')
    logs = finish_paths[0].parent
    report_dir = run / 'orfs_work/reports' / logs.relative_to(run / 'orfs_work/logs')
    report_paths = {name: logs / name for name in ('5_2_route.json', '6_report.json')}
    reports = {name: json.loads(capture(path), parse_float=Decimal, parse_int=Decimal)
               for name, path in report_paths.items()}
    units_path = logs / '2_1_floorplan.json'
    units = json.loads(capture(units_path))
    require(units.get('run__flow__platform__time_units') == '1ns'
            and units.get('run__flow__platform__distance_units') == '1um',
            'Table 2 requires explicitly recorded ns and um report units')
    text_path = report_dir / '6_finish.rpt'
    text = capture(text_path).decode()
    tokens = {}
    for key in ('WNS', 'TNS'):
        values = re.findall(r'^' + key.lower() + r'(?:\s+max)?\s+(' + NUMBER + r')\s*$', text, re.M)
        require(len(values) == 1, 'Expected exactly one setup timing text token: ' + key)
        tokens[key] = values[0]
    power_section = text.split('finish report_power', 1)
    require(len(power_section) == 2 and '(Watts)' in power_section[1],
            'Final total-power text must explicitly use Watts')
    totals = [line.split() for line in power_section[1].splitlines()
              if line.strip().startswith('Total') and '%' in line]
    require(len(totals) == 1 and len(totals[0]) >= 5, 'Ambiguous total-power text row')
    require(re.fullmatch(NUMBER, totals[0][4]), 'Invalid power token')
    tokens['Power'] = totals[0][4]
    canonical = deepcopy(author_result)
    comparisons = {}
    for key, (filename, field) in FIELDS.items():
        require(key in author_result, 'Missing author metric: ' + key)
        value = author_result[key]
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                'Invalid author metric: ' + key)
        source_value = reports[filename][field]
        require(isinstance(source_value, Decimal) and source_value.is_finite(), 'Invalid raw JSON metric: ' + key)
        raw = min(Decimal(0), source_value) if key == 'WNS' else source_value
        require(raw <= 0 if key in ('WNS', 'TNS') else raw >= 0, 'Invalid raw metric sign: ' + key)
        require(key != 'DRC' or raw == raw.to_integral_value(), 'DRC count is not an integer')
        precision = None
        text_semantics = None
        if key in tokens:
            if key == 'WNS' and source_value > 0:
                text_value = Decimal(tokens[key])
                require(text_value >= 0, 'Positive setup slack is inconsistent with a negative WNS token')
                if text_value == 0:
                    # OpenSTA report_wns clamps positive setup worst slack.
                    precision = precision_evidence(Decimal(0), tokens[key])
                    text_semantics = 'report_wns_clamped_to_zero'
                else:
                    # Preserve a possible un-clamped worst-slack report as
                    # such; never compare positive text to canonical zero.
                    precision = precision_evidence(source_value, tokens[key])
                    text_semantics = 'positive_setup_worst_slack_canonicalized_to_wns'
            else:
                precision = precision_evidence(raw, tokens[key])
                text_semantics = 'metric_at_text_display_precision'
        if value == float(raw):
            match = 'raw_json'
        elif key == 'WNS' and source_value > 0 and value == float(source_value):
            # The author's JSON path labels setup worst slack as WNS. Preserve
            # that value separately but use the actual nonpositive WNS metric.
            match = 'raw_setup_worst_slack_clamped_to_wns'
        else:
            require(key in tokens and value == float(tokens[key]),
                    'Author metric matches neither raw JSON nor its bound text token: ' + key)
            match = 'bound_text_display_rounding'
        canonical[key] = int(raw) if key == 'DRC' else float(raw)
        comparisons[key] = {'author_value': value, 'source_json_value': float(source_value),
                            'canonical_value': canonical[key], 'match': match,
                            'raw_report': str(report_paths[filename].relative_to(run)), 'raw_field': field,
                            'text_precision_check': precision,
                            'text_semantics': text_semantics,
                            'wns_clamped': key == 'WNS' and source_value > 0}
    for path, data in captured.items():
        require(path.read_bytes() == data, 'Metric evidence changed while canonicalizing: ' + str(path))
    evidence = {
        'schema_version': 1, 'metric_policy': METRIC_POLICY, 'all_checks_pass': True,
        'author_metrics_path': str(author_path.relative_to(run)), 'author_metrics_sha256': digest_bytes(author_data),
        'source_report_sha256': {str(path.relative_to(run)): digest_bytes(captured[path.resolve()])
                                 for path in report_paths.values()},
        'source_text_path': str(text_path.relative_to(run)), 'source_text_sha256': digest_bytes(captured[text_path.resolve()]),
        'units_source_path': str(units_path.relative_to(run)), 'units_source_sha256': digest_bytes(captured[units_path.resolve()]),
        'units': {'time': 'ns', 'distance': 'um', 'cell_area': 'um2', 'total_power': 'W'},
        'comparisons': comparisons, 'canonical_metric_keys': list(FIELDS),
        'non_ppa_author_fields_preserved': sorted(set(author_result) - set(FIELDS)),
        'canonicalizer_sha256': digest_bytes(Path(__file__).read_bytes()),
        'limitations': ['Metric validation does not certify physical-stage success or full-run provenance.',
                        'Author metrics remain unchanged; canonical values use final raw JSON.',
                        'WNS is min(0, setup worst slack); no paper-value fitting or metric rescaling is performed.'],
    }
    return canonical, evidence
