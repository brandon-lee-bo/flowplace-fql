"""Parse final GRT outputs; absent congestion files are never treated as zero."""
import argparse
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from grt_provenance import stamp_provenance

PARSER_VERSION = 'grt_v2_final_odb_markers'


def parse_grt_metrics(work, runtime_seconds):
    matches = list((work / 'orfs_work/logs/nangate45').glob('*/eval/5_grt_label.json'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one final GRT metrics file, found {len(matches)}')
    raw_file = matches[0]
    design = raw_file.parent.parent.name
    report = work / 'orfs_work/reports/nangate45' / design / 'eval'
    odb = work / 'orfs_work/results/nangate45' / design / 'eval/5_1_grt.odb'
    if not odb.is_file() or not (report / '5_fql_global_route_label.rpt').is_file():
        raise RuntimeError('Final ODB or timing report absent: flow completion not established')
    raw = json.loads(raw_file.read_text())
    wanted = {'rWL_um': 'globalroute__global_route__wirelength',
              'TNS_ns': 'globalroute__timing__setup__tns',
              'WNS_ns': 'globalroute__timing__setup__ws',
              'Power_W': 'globalroute__power__total',
              'StdCellArea_um2': 'globalroute__design__instance__area__stdcell'}
    metrics = {'stage': 'global_route_only', 'not_final_table2': True,
               'parser_version': PARSER_VERSION}
    for name, key in wanted.items():
        value = raw.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f'Missing/nonfinite required metric {key}')
        metrics[name] = value
    metrics['worst_slack_ns'] = metrics['WNS_ns']
    metrics['WNS_ns'] = min(0.0, metrics['WNS_ns'])

    status_file = report / 'fql_grt_marker_status.json'
    final_report = report / 'fql_final_congestion.rpt'
    legacy_final = report / 'congestion_post_recover_power.rpt'
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status.get('category_present') is False and status.get('marker_count') == 0:
            entries = []
            chosen_report = None
            metrics['congestion_evidence'] = 'Final OpenDB Global route marker category explicitly absent'
        elif status.get('category_present') is True and final_report.is_file():
            chosen_report = final_report
            entries = parse_overflow_entries(final_report)
            if status.get('marker_count') != len(entries):
                raise RuntimeError(f'Marker count mismatch: OpenDB={status.get("marker_count")}, parsed={len(entries)}')
            metrics['congestion_evidence'] = 'Final OpenDB marker export with exact marker-count validation'
        else:
            raise RuntimeError('Invalid final marker status or missing explicit marker export')
        metrics['congestion_marker_status'] = str(status_file)
        metrics['congestion_marker_evidence_source'] = status['evidence_source']
    elif legacy_final.is_file():
        chosen_report = legacy_final
        entries = parse_overflow_entries(legacy_final)
        if not entries:
            raise RuntimeError('Empty legacy final report requires explicit OpenDB marker evidence')
        metrics['congestion_evidence'] = 'Existing positive final post-recover-power congestion report'
    else:
        raise RuntimeError('No final congestion evidence: recover marker status from saved final ODB; never fall back to initial congestion.rpt')

    metrics['congestion_report'] = str(chosen_report) if chosen_report else None
    for direction in ['Horizontal', 'Vertical']:
        values = [overflow for dr, overflow in entries if dr == direction]
        metrics[f'{direction.lower()}_overflow_edges'] = len(values)
        metrics[f'{direction.lower()}_overflow_sum'] = sum(values)
        metrics[f'{direction.lower()}_overflow_max'] = max(values, default=0)
    metrics['congestion_note'] = 'Positive final routing-edge demand-minus-capacity entries; not a percentage or detailed-route DRC.'
    metrics['runtime_seconds'] = runtime_seconds
    return metrics


def parse_overflow_entries(path):
    text = path.read_text()
    matches = re.findall(r'violation type: (Horizontal|Vertical) congestion.*?capacity:(\d+) usage:(\d+) congestion:(\d+)', text, re.S)
    if len(matches) != len(re.findall(r'violation type:', text)):
        raise RuntimeError(f'Unparsed congestion marker in {path}')
    result = []
    for direction, cap, use, overflow in matches:
        if int(overflow) <= 0 or int(use) - int(cap) != int(overflow):
            raise RuntimeError(f'Inconsistent overflow entry in {path}')
        result.append((direction, int(overflow)))
    return result


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def reparse(work):
    manifest_path = work / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if 'global_route_label' not in manifest.get('stage_wall_seconds', {}):
        raise RuntimeError('GRT execution stage not recorded; refuse postprocessing recovery')
    if hashlib.sha256(Path(manifest['def_path']).read_bytes()).hexdigest() != manifest['def_sha256']:
        raise RuntimeError('Input DEF changed after evaluation')
    runtime = manifest['time_end'] - manifest['time_start']
    metrics = parse_grt_metrics(work, runtime)
    for path in [manifest_path, work / 'metrics.json']:
        backup = path.with_name(path.stem + '.before_parser_v2.json')
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
    old_success = manifest.get('success')
    stamp_provenance(manifest,Path(__file__).resolve().parents[1],retrospective=True)
    manifest['success'] = True
    manifest['parser_version'] = PARSER_VERSION
    manifest['postprocessing_parser_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest.setdefault('postprocessing_recoveries', []).append({
        'time': time.time(), 'previous_success': old_success,
        'reason': ('Parser v1 assumed congestion file existed; recovered final marker evidence from saved ODB without rerunning flow' if not old_success else 'Validated existing result with parser v2 and explicit final congestion evidence; physical flow not rerun'),
        'original_slurm_job_id': manifest['slurm_job_id'],
        'original_slurm_status_preserved': True})
    atomic_json(work / 'metrics.json', metrics)
    atomic_json(manifest_path, manifest)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(reparse(args.run_dir.resolve()), indent=2))
