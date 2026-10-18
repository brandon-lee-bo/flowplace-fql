"""Saved-only TRAIN64 power/analysis-condition audit; never run STA or EDA.

The admission flag checks captured common conditions and available evidence.
Host-mounted library/script hashes inspected now are explicitly present-day
observations, not retroactive runtime attestations or workload activity data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

from backend.scripts.canonical_table2_metrics import canonicalize, precision_evidence
from fql import qstudy_train_final_route as collector
from fql.graph_o2o_archive import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT/'backend/thirdparty/OpenROAD-flow-scripts'
PROTOCOL_SHA = 'ecb148af55e0e491f4016a7e14ed900cc4e295c087263aa1c194c0f738b28e20'
PROTOCOL = ROOT/'docs/qstudy_final_feedback_protocol.md'
SOURCE_NAMES = ('analysis/qstudy_final_power_audit.py',
    'analysis/qstudy_final_power_audit.slurm', 'fql/qstudy_train_final_route.py',
    'fql/graph_o2o_final_evidence.py', 'backend/scripts/canonical_table2_metrics.py',
    'docs/qstudy_final_feedback_protocol.md')
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
ACTIVITY = re.compile(r'\b(?:read_vcd|read_saif|set_power_activity)\b')


class Capture:
    def __init__(self):
        self.files = {}

    def bytes(self, path):
        path = Path(path).absolute()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if str(path) in self.files and self.files[str(path)] != digest:
            raise ValueError('Evidence changed while auditing: '+str(path))
        self.files[str(path)] = digest
        return content

    def text(self, path):
        return self.bytes(path).decode()

    def read(self, path):
        return json.loads(self.text(path))

    def verify(self):
        for path, digest in self.files.items():
            if sha(path) != digest:
                raise ValueError('Evidence changed before report publication: '+path)


def safe_relative(run, name):
    name = PurePosixPath(name)
    if name.is_absolute() or '..' in name.parts or str(name) in ('', '.'):
        raise ValueError('Unsafe saved native report path')
    path = Path(run)/str(name)
    if not path.resolve().is_relative_to(Path(run).resolve()):
        raise ValueError('Saved native report escapes its physical run')
    return path


def decimal_token(value):
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('Nonnumeric saved analysis-condition value') from exc
    if not result.is_finite():
        raise ValueError('Nonfinite analysis-condition value')
    return format(result.normalize(), 'f')


def commands(text):
    """Read numeric generated SDC, without executing Tcl or resolving variables."""
    text = text.replace('\\\n', ' ')
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            yield line


def tokens(line):
    result, current, braces, brackets, quoted = [], [], 0, 0, False
    for char in line:
        if char == '"' and braces == 0:
            quoted = not quoted
        elif not quoted:
            braces += (char == '{') - (char == '}')
            brackets += (char == '[') - (char == ']')
        if braces < 0 or brackets < 0:
            raise ValueError('Malformed saved SDC command')
        if char.isspace() and braces == brackets == 0 and not quoted:
            if current:
                result.append(''.join(current)); current = []
        else:
            current.append(char)
    if braces or brackets or quoted:
        raise ValueError('Unsupported multiline/unbalanced saved SDC')
    if current:
        result.append(''.join(current))
    return result


def clock_signature(text):
    """Compare clock/units semantics, not layout-dependent full SDC text hashes."""
    clocks, units = [], {}
    for line in commands(text):
        t = tokens(line)
        if ACTIVITY.search(line):
            raise ValueError('Explicit activity input violates fixed-default-activity protocol')
        if t[0] == 'create_generated_clock':
            raise ValueError('Generated clock requires an explicitly reviewed semantic parser')
        if t[0] == 'set_units':
            if len(t) % 2 != 1:
                raise ValueError('Malformed saved set_units')
            units.update(zip(t[1::2], t[2::2]))
        if t[0] != 'create_clock':
            continue
        record, i = {}, 1
        while i < len(t) and t[i].startswith('-'):
            key = t[i]; i += 1
            if key == '-add':
                record[key] = True
            elif key in ('-name', '-period', '-waveform') and i < len(t):
                value = t[i]; i += 1
                if key == '-period':
                    value = decimal_token(value)
                    if Decimal(value) <= 0:
                        raise ValueError('Clock period must be positive')
                elif key == '-waveform':
                    value = [decimal_token(x) for x in value.strip('{}').split()]
                    if len(value) != 2:
                        raise ValueError('Unsupported clock waveform')
                record[key] = value
            else:
                raise ValueError('Unsupported numeric saved create_clock option: '+key)
        if '-period' not in record or i >= len(t) or '$' in ' '.join(t[i:]):
            raise ValueError('Saved SDC clock must have numeric period and explicit target')
        record['target'] = ' '.join(t[i:])
        clocks.append(record)
    if not clocks:
        raise ValueError('No explicit numeric clock in final SDC')
    return dict(clocks=sorted(clocks, key=lambda r:json.dumps(r, sort_keys=True)), units=units)


def power_components(text):
    section = text.split('finish report_power', 1)
    if len(section) != 2 or '(Watts)' not in section[1]:
        raise ValueError('Final power text lacks explicit Watts')
    rows = [line.split() for line in section[1].splitlines()
            if line.strip().startswith('Total') and '%' in line]
    if len(rows) != 1 or len(rows[0]) < 5:
        raise ValueError('Ambiguous final power total')
    row = rows[0]
    if not re.fullmatch(NUMBER, row[4]):
        raise ValueError('Invalid final power total token')
    total = Decimal(row[4])
    if not total.is_finite() or total <= 0:
        raise ValueError('Final total power must be finite and positive')
    if not all(re.fullmatch(NUMBER, x) for x in row[1:4]):
        if any(not re.fullmatch(NUMBER, x) and x.upper() not in ('NA','N/A','--','-') for x in row[1:4]):
            raise ValueError('Malformed/nonfinite native power component')
        return dict(total_token=row[4], components=None,
                    component_tokens=None, component_status='not_available_in_saved_text', sum_check=None)
    values = [Decimal(x) for x in row[1:4]]
    if any(not x.is_finite() or x < 0 for x in values):
        raise ValueError('Invalid native power component')
    # Every printed scalar can be rounded by half its own final decimal place.
    bound = sum(Decimal(1).scaleb(x.as_tuple().exponent)/2 for x in values+[total])
    difference = abs(sum(values)-total)
    if difference > bound:
        raise ValueError('Native power components disagree beyond printed rounding')
    return dict(total_token=row[4], component_status='saved_text_present',
        components=dict(zip(('internal_W','switching_W','leakage_W'), map(float, values))),
        component_tokens=dict(zip(('internal_W','switching_W','leakage_W'),row[1:4])),
        sum_check=dict(absolute_difference=str(difference), rounding_bound=str(bound), passed=True))


def json_components(raw, text):
    """Retain available native component fields; absent fields stay NA."""
    keys = dict(internal_W='finish__power__internal', switching_W='finish__power__switching',
                leakage_W='finish__power__leakage')
    values, checks = {}, {}
    for name, key in keys.items():
        value = raw.get(key)
        if value is None:
            values[name] = None; continue
        if isinstance(value,bool) or not isinstance(value,(int,float,Decimal)):
            raise ValueError('Malformed JSON native power component: '+key)
        value = Decimal(str(value))
        if not value.is_finite() or value < 0:
            raise ValueError('Nonfinite/negative JSON native power component: '+key)
        values[name] = str(value)
        if text['component_tokens'] is not None:
            checks[name] = precision_evidence(str(value), text['component_tokens'][name])
    total = Decimal(str(raw['finish__power__total']))
    summed = None
    if all(value is not None for value in values.values()):
        components = [Decimal(value) for value in values.values()]
        bound = sum(Decimal(1).scaleb(x.as_tuple().exponent)/2 for x in components+[total])
        difference = abs(sum(components)-total)
        if difference > bound:
            raise ValueError('JSON native power components disagree beyond report precision')
        summed = dict(absolute_difference=str(difference),rounding_bound=str(bound),passed=True)
    return dict(fields=values,available=sum(value is not None for value in values.values()),
        json_text_precision_checks=checks,sum_check=summed,
        status='all_native_fields_present' if all(value is not None for value in values.values()) else 'partial_or_unavailable_not_zero')


def native_power(run, captured):
    run = Path(run)
    manifest = captured.read(run/'manifest.json')
    metrics = captured.read(run/'metrics.json')
    author = captured.read(run/'author_metrics.json')
    proof = captured.read(run/'metric_canonicalization.json')
    if (manifest['canonical_metrics_sha256'] != sha(run/'metrics.json') or
            manifest['metric_evidence_sha256'] != sha(run/'metric_canonicalization.json')):
        raise ValueError('Native final metric proof binding differs')
    for relative, digest in proof['source_report_sha256'].items():
        path = safe_relative(run, relative)
        captured.bytes(path)
        if sha(path) != digest:
            raise ValueError('Native raw report changed')
    for key in ('source_text', 'units_source', 'author_metrics'):
        path = safe_relative(run, proof[key+'_path'])
        captured.bytes(path)
        if sha(path) != proof[key+'_sha256']:
            raise ValueError('Canonical native evidence changed: '+key)
    canonical, recalculated = canonicalize(run, author)
    if canonical != metrics or recalculated != proof:
        raise ValueError('Native canonicalization cannot be reproduced exactly')
    report_relative = proof['comparisons']['Power']['raw_report']
    raw = captured.read(safe_relative(run, report_relative))
    value = raw['finish__power__total']
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not 0 < value < float('inf'):
        raise ValueError('Raw final power must be finite positive Watts')
    if value != metrics['Power']:
        raise ValueError('Raw final power differs from canonical Power')
    components = power_components(captured.text(safe_relative(run, proof['source_text_path'])))
    precision_evidence(str(value), components['total_token'])
    raw_exact = json.loads(captured.text(safe_relative(run, report_relative)),parse_float=Decimal,parse_int=Decimal)
    native_components = json_components(raw_exact,components)
    base = run/'orfs_work/results/nangate45'/manifest['design']/'eval'
    sdc = base/'6_1_fill.sdc'
    signature = clock_signature(captured.text(sdc))
    units = captured.read(safe_relative(run, proof['units_source_path']))
    common = dict(tool_image_sha256=None, openroad_sha256=manifest['openroad_sha256'],
        design=manifest['design'], design_config_sha256=manifest['design_config_sha256'],
        physical_flow_components=manifest['physical_flow_components'],
        detailed_route_flow_components=manifest['detailed_route_flow_components'],
        physical_flow_revision=manifest['physical_flow_revision'],
        detailed_route_physical_revision=manifest['detailed_route_physical_revision'],
        clock=signature, time_units=units['run__flow__platform__time_units'],
        distance_units=units['run__flow__platform__distance_units'], power_units='W')
    source = captured.read(Path(manifest['source_grt_run'])/'manifest.json')
    common['tool_image_sha256'] = source['tool_image_sha256']
    if common['time_units'] != '1ns' or common['distance_units'] != '1um':
        raise ValueError('Unexpected captured native units')
    return dict(power_W=value, components=components, native_json_components=native_components, common_conditions=common,
        final_sdc=dict(path=str(sdc), sha256=captured.files[str(sdc.absolute())]),
        evidence=proof, native_run=str(run), DRC=metrics['DRC'])


def checked_native_power(outcome, captured):
    """Close the inspector-to-read interval against the immutable outcome."""
    run = Path(outcome['physical_run'])
    value = native_power(run, captured)
    for name, key in (('manifest.json','physical_manifest_sha256'),('metrics.json','metrics_sha256')):
        if captured.files[str((run/name).absolute())] != outcome[key]:
            raise ValueError('Captured power proof differs from its audited outcome: '+name)
    if captured.read(run/'metrics.json') != outcome['metrics']:
        raise ValueError('Captured final native metrics differ from audited outcome')
    return value


def make_assignments(text):
    result = {}
    for line in commands(text):
        match = re.match(r'(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*[:?]?=\s*(.*?)\s*(?:#.*)?$', line)
        if match:
            result[match[1]] = match[2]
    return result


def expand(value, variables):
    expression = re.compile(r'\$\((\w+)\)|\$\{(\w+)\}')
    for _ in range(15):
        new = expression.sub(lambda m:variables.get(m[1] or m[2], m[0]), value)
        if new == value:
            break
        value = new
    if '$' in value:
        raise ValueError('Cannot statically resolve host library declaration')
    return value


def liberty_metadata(text):
    def field(name):
        match = re.search(r'\b'+name+r'\s*:\s*([^;]+);', text)
        return match[1].strip().strip('"') if match else None
    leakage = re.findall(r'\bcell_leakage_power\s*:\s*('+NUMBER+r')\s*;', text)
    internal = len(re.findall(r'\binternal_power\s*\(', text))
    corner = {k:field(k) for k in ('nom_process','nom_temperature','nom_voltage')}
    corner = {k:decimal_token(v) if v is not None else None for k,v in corner.items()}
    return dict(units={k:field(k) for k in ('time_unit','voltage_unit','current_unit','leakage_power_unit')},
        corner=corner,
        internal_power_groups=internal, cell_leakage_values=len(leakage),
        nonzero_cell_leakage_values=sum(Decimal(x)>0 for x in leakage),
        power_model_presence='present' if internal and leakage else 'not_established',
        model_accuracy='not independently validated; presence does not prove workload annotation or model accuracy')


def host_observations(common, captured):
    """Runtime-captured digests remain separate from uncaptured host observations."""
    design_path = None
    # Only the one explicitly recorded design configuration, never a broad scan.
    for path in captured.files:
        if path.endswith('/manifest.json'):
            value = json.loads(Path(path).read_text())
            if value.get('stage') == 'finish_after_detailed_route':
                design_path = Path(value['design_config']); break
    if design_path is None:
        raise ValueError('No captured final design configuration identity')
    platform = FLOW/'platforms/nangate45'
    files = [FLOW/'scripts'/name for name in ('load.tcl','util.tcl','read_liberty.tcl',
        'report_metrics.tcl','final_report.tcl','deleteRoutingObstructions.tcl')]
    files += [platform/'config.mk', platform/'setRC.tcl', design_path]
    derate = platform/'derate.tcl'
    if derate.exists(): files.append(derate)
    texts = {str(path):captured.text(path) for path in files}
    for path, text in texts.items():
        # Ignore comment-only examples; no Tcl is executed by this audit.
        if any(ACTIVITY.search(line) for line in commands(text)):
            raise ValueError('Explicit activity command in current inspected host path: '+path)
    comparisons = {str(design_path):common['design_config_sha256'],
        str(platform/'config.mk'):common['physical_flow_components']['platform_config_sha256']}
    for relative, digest in common['detailed_route_flow_components'].items():
        path = FLOW/relative
        captured.bytes(path)
        comparisons[str(path)] = digest
    for path, digest in comparisons.items():
        if sha(path) != digest:
            raise ValueError('Current host file differs from captured native digest: '+path)
    variables = make_assignments(texts[str(platform/'config.mk')])
    variables.update(make_assignments(texts[str(design_path)]))
    variables['PLATFORM_DIR'] = str(platform)
    if 'LIB_FILES' not in variables:
        raise ValueError('Missing explicit loaded-library declaration')
    libraries = []
    for name in expand(variables['LIB_FILES'], variables).split():
        path = Path(name)
        if not path.is_absolute(): path = FLOW/path
        if not path.resolve().is_relative_to(FLOW.resolve()):
            raise ValueError('Loaded library escapes explicit project flow')
        content = captured.bytes(path)
        if path.suffix == '.gz': content = gzip.decompress(content)
        libraries.append(dict(path=str(path), sha256=captured.files[str(path.absolute())],
            fakeram='fakeram' in path.name.lower(), **liberty_metadata(content.decode())))
    critical = []
    if not libraries or any(x['power_model_presence'] != 'present' for x in libraries):
        critical.append('A declared loaded library has no established internal/leakage power model')
    corners = [x['corner'] for x in libraries]
    if any(c != corners[0] for c in corners[1:]):
        critical.append('Current loaded-library nominal conditions differ')
    if any(value is None for c in corners for value in c.values()):
        critical.append('Current loaded-library nominal voltage/corner is unavailable')
    return dict(status='present_day_observations_only', libraries=libraries, critical_errors=critical,
        files={path:captured.files[path] for path in texts}, captured_digest_matches=comparisons,
        activity='No explicit activity command found in this inspected host source closure; no numeric default attested',
        limitation='Uncaptured library/load/setRC byte identities are not retroactive runtime proof. Fixed-tool default activity estimates are not measured workload power.')


def checked_collector(plan, report, captured):
    if (report.get('version') != collector.VERSION or report.get('status') != 'complete' or
        report.get('requested') != 64 or report.get('counts') != {'completed':64} or
        report.get('plan') != collector.bind(Path(plan['output'])/'plan.json') or
        report.get('source_sha256') != plan['source_sha256'] or
        report.get('source_split') != 'original384TRAIN_only' or
        report.get('policy_evaluation_or_old_DRT_used') is not False or
        report.get('family_folds') != plan['selection']['selected_families'] or
        len(report.get('records', [])) != 64 or
        report.get('DRC_zero') != sum(r.get('DRC_zero',False) for r in report.get('records',[]))):
        raise ValueError('Only the complete pinned original-TRAIN64 collector report is admitted')
    for task, saved in zip(plan['tasks'], report['records']):
        actual = collector.inspect_result(plan, task)
        actual['attempts'] = collector.attempt_evidence(task['output'])
        if actual != saved or actual['state'] != 'completed':
            raise ValueError('Saved collector row differs from actual audited TRAIN result: '+task['key'])
        captured.bytes(actual['outcome']['path'])
        if captured.files[actual['outcome']['path']] != actual['outcome']['sha256']:
            raise ValueError('Collector native outcome changed while auditing')


def collect(plan_path, plan_sha256, report_path, report_sha256):
    captured = Capture()
    plan_path, report_path = Path(plan_path).absolute(), Path(report_path).absolute()
    plan = collector.checked_plan(plan_path, plan_sha256)
    captured.bytes(plan_path); captured.bytes(report_path)
    if captured.files[str(report_path)] != report_sha256:
        raise ValueError('Changed or unpinned complete collector report')
    report = captured.read(report_path)
    checked_collector(plan, report, captured)
    captured.bytes(PROTOCOL)
    if captured.files[str(PROTOCOL)] != PROTOCOL_SHA:
        raise ValueError('Fixed final-feedback protocol changed')
    rows, errors = [], []
    for task, saved in zip(plan['tasks'], report['records']):
        row = dict(index=task['index'], key=task['key'], family=task['family'], fold=task['fold'],
            variant=task['variant'], status='invalid', errors=[])
        try:
            out = captured.read(saved['outcome']['path'])
            power = checked_native_power(out, captured)
            if power['power_W'] != saved['metrics']['Power']:
                raise ValueError('Saved TRAIN row power differs from raw native source')
            row.update(status='verified', **power)
        except Exception as exc:
            row['errors'] = [str(exc)]; errors.append(task['key']+': '+str(exc))
        rows.append(row)
    good = [r for r in rows if r['status']=='verified']
    if good and any(r['common_conditions'] != good[0]['common_conditions'] for r in good[1:]):
        errors.append('Captured image/tool/config/flow/clock/unit conditions differ across TRAIN64')
    host = None
    try:
        if len(good) == 64 and not errors:
            host = host_observations(good[0]['common_conditions'], captured)
            errors.extend('host source/library QA: '+error for error in host['critical_errors'])
    except Exception as exc:
        errors.append('host source/library QA: '+str(exc))
    captured.verify()
    return dict(schema='qstudy_final_power_audit_v1', status='verified' if not errors else 'not_admitted',
        admission_eligible=not errors and len(good)==64, planned=64, verified=len(good), errors=errors,
        plan=dict(path=str(plan_path),sha256=plan_sha256), collector_report=dict(path=str(report_path),sha256=report_sha256),
        protocol=dict(path=str(PROTOCOL),sha256=PROTOCOL_SHA), rows=rows,
        common_captured_conditions=good[0]['common_conditions'] if good else None,
        present_day_host_observations=host, evidence_sha256=captured.files,
        assumptions=dict(power='Fixed-tool default-activity estimate; not measured workload/silicon power',
            numeric_default_activity='not independently established for the frozen binary',
            clock_check='Semantic clock/unit comparison; per-layout SDC byte hashes need not be identical',
            library_provenance='Present-day uncaptured hashes are not retroactive runtime attestations',
            feasibility='DRC>0 remains measured power evidence, receives shared failure utility elsewhere',
            selection='No target weights, families or policy outputs selected by this audit'),
        model_calls=0, backend_calls=0, label_mutations=0)


def checked_report(path, expected_sha256, collector_plan_binding, collector_report_binding):
    """Screen admission: verify the immutable completed QA and unchanged evidence."""
    path = Path(path).absolute()
    if sha(path) != expected_sha256:
        raise ValueError('Changed or unpinned final-power audit report')
    report = collector.read(path)
    completed = collector.read(path.with_name('completed.json'))
    sources = {name:sha(ROOT/name) for name in SOURCE_NAMES}
    if (report.get('schema') != 'qstudy_final_power_audit_v1' or
        report.get('status') != 'verified' or report.get('admission_eligible') is not True or
        report.get('plan') != collector_plan_binding or report.get('collector_report') != collector_report_binding or
        report.get('protocol') != dict(path=str(PROTOCOL),sha256=PROTOCOL_SHA) or
        report.get('planned') != 64 or report.get('verified') != 64 or report.get('errors') != [] or
        len(report.get('rows',[])) != 64 or
        [row.get('index') for row in report['rows']] != list(range(64)) or
        any(row.get('status') != 'verified' for row in report['rows']) or
        report.get('sources') != sources or
        any(report.get(k) != 0 for k in ('model_calls','backend_calls','label_mutations')) or
        completed != dict(status='completed',admission_eligible=True,
            report_sha256=expected_sha256,sources=sources,model_calls=0,backend_calls=0)):
        raise ValueError('Incomplete, inconsistent or mismatched final-power admission proof')
    for name,digest in sources.items():
        if sha(path.parent/'sources'/name) != digest:
            raise ValueError('Final-power audit source snapshot changed')
    for original,digest in report['evidence_sha256'].items():
        if sha(original) != digest:
            raise ValueError('Power evidence changed since admission: '+original)
    report = dict(report)
    report['audit_binding'] = collector.bind(path, expected_sha256)
    report['completion_binding'] = collector.bind(path.with_name('completed.json'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use a bounded CPU Slurm allocation')
    output = args.output.resolve()
    for protected in (args.plan.resolve().parent, args.report.resolve().parent, FLOW.resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Audit output must be separate from all original input trees')
    result = collect(args.plan, args.plan_sha256, args.report, args.report_sha256)
    snapshot = output/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_job'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(parents=True, exist_ok=False)
    sources = {name:sha(ROOT/name) for name in SOURCE_NAMES}
    for name, digest in sources.items():
        target = snapshot/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        if sha(target) != digest: raise ValueError('Changed audit source while snapshotting')
    result.update(sources=sources, slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(snapshot/'report.json', result)
    atomic_json(snapshot/'completed.json', dict(status='completed', admission_eligible=result['admission_eligible'],
        report_sha256=sha(snapshot/'report.json'), sources=sources, model_calls=0, backend_calls=0))
    atomic_json(output/'latest.json', dict(report=collector.bind(snapshot/'report.json'),
        completion=collector.bind(snapshot/'completed.json'), admission_eligible=result['admission_eligible']))
    print(json.dumps(dict(snapshot=str(snapshot), admission_eligible=result['admission_eligible'],
                         verified=result['verified'], errors=result['errors'])), flush=True)


if __name__ == '__main__': main()
