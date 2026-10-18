#!/usr/bin/env python3
"""Audit the separate uniform exact-import cohort; require strict generator-coordinate preservation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from decimal import Decimal

REPORTS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports')
ARCHIVES = Path('/fact_data/bohanli/flowplace_fql_backend/table2_archives')
POSITION = re.compile(r'\+\s+(FIXED|PLACED|COVER)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def macros(path):
    """Stream only the COMPONENTS section; do not load routed net geometry."""
    units, declared, count, inside, finished = None, None, 0, False, False
    entries, pending = {}, ''
    with path.open() as stream:
        for line in stream:
            if not inside:
                match = re.match(r'\s*UNITS DISTANCE MICRONS\s+(\d+)\s*;', line)
                if match:
                    units = int(match[1])
                match = re.match(r'\s*COMPONENTS\s+(\d+)\s*;', line)
                if match:
                    declared, inside = int(match[1]), True
                continue
            if re.match(r'\s*END COMPONENTS\s*$', line):
                assert not pending.strip(), 'Unterminated component'
                finished = True
                break
            pending += line
            assert len(pending) < 1024 * 1024, 'Unexpected component record size'
            if ';' not in pending:
                continue
            assert pending.count(';') == 1, 'Expected one component record per terminator'
            match = re.fullmatch(r'\s*-\s+(\S+)\s+(\S+)(.*?);\s*', pending, re.S)
            assert match, 'Malformed component'
            name, master, body = match.groups()
            count += 1
            if master.startswith('fakeram45_'):
                name = re.sub(r'\\(.)', r'\1', name)
                assert name not in entries, 'Duplicate macro'
                position = POSITION.search(body)
                assert position, 'Macro lacks placement'
                entries[name] = {'master': master, 'xy_dbu': [int(position[2]), int(position[3])],
                                 'orientation': position[4], 'status': position[1]}
            pending = ''
    assert finished and units and count == declared and entries, 'Incomplete DEF component parse'
    return units, entries


def archive_evidence(run, final, final_sha):
    evidence = []
    for attempt in sorted((ARCHIVES / run.name).glob('*')):
        state, manifest, inventory = [attempt / name for name in
                                      ('commit_state.json', 'archive_manifest.json', 'source_inventory.json')]
        if not all(p.is_file() for p in (state, manifest, inventory)):
            continue
        commit, record = json.loads(state.read_text()), json.loads(manifest.read_text())
        if commit.get('status') != 'committed' or commit.get('run') != str(run):
            continue
        assert sha(manifest) == commit['archive_manifest_sha256']
        assert sha(inventory) == record['source_inventory_sha256']
        assert record['source_control_sha256']['manifest.json'] == sha(run / 'manifest.json')
        source = json.loads(inventory.read_text())
        assert source[final.relative_to(run).as_posix()]['sha256'] == final_sha
        assert (attempt / record['archive']).is_file()
        evidence.append({'archive_manifest': str(manifest), 'archive_manifest_sha256': sha(manifest),
                         'commit_state_sha256': sha(state), 'source_inventory_sha256': sha(inventory),
                         'archive_sha256_recorded': record['archive_sha256']})
    if (run / 'orfs_work/results').is_symlink():
        assert evidence, 'Scratch-linked final DEF lacks committed durable archive evidence'
    return evidence


def geometry_check(run, manifest, units, actual):
    frozen = json.loads((run / 'provenance/flow_inputs_start.json').read_text())
    assert frozen['tree_sha256'] == manifest['flow_inputs_start_sha256']
    files = {entry['relative_path']: entry for entry in frozen['files']}
    bounds = [Decimal(v) * units for v in manifest['design_config_exports']['CORE_AREA'].split()]
    assert len(bounds) == 4
    sizes, lef_sha, boxes = {}, {}, {}
    for name, macro in actual.items():
        master = macro['master']
        if master not in sizes:
            relative = 'platforms/nangate45/lef/' + master + '.lef'
            path = run / 'provenance/flow_snapshot' / relative
            if not path.is_file():
                path = Path(files[relative]['path'])
            digest = sha(path)
            assert digest == files[relative]['sha256'], 'LEF differs from physical-run snapshot'
            match = re.search(r'\bSIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;', path.read_text())
            assert match
            sizes[master] = tuple(Decimal(v) * units for v in match.groups())
            assert all(v == int(v) and v > 0 for v in sizes[master])
            lef_sha[relative] = digest
        width, height = sizes[master]
        if macro['orientation'] in ('E', 'W', 'FE', 'FW'):
            width, height = height, width
        else:
            assert macro['orientation'] in ('N', 'S', 'FN', 'FS')
        x, y = macro['xy_dbu']
        boxes[name] = (x, y, x + int(width), y + int(height))
    overlaps, out_of_core = [], []
    for index, (name, box) in enumerate(boxes.items()):
        x0, y0, x1, y1 = box
        if x0 < bounds[0] or y0 < bounds[1] or x1 > bounds[2] or y1 > bounds[3]:
            out_of_core.append(name)
        for other, b in list(boxes.items())[:index]:
            dx, dy = min(x1, b[2]) - max(x0, b[0]), min(y1, b[3]) - max(y0, b[1])
            if dx > 0 and dy > 0:
                overlaps.append({'a': name, 'b': other, 'overlap_width_dbu': dx, 'overlap_height_dbu': dy})
    return {'lef_sha256': lef_sha, 'core_bounds_dbu': [float(v) for v in bounds],
            'overlaps': overlaps, 'out_of_core': out_of_core, 'zero_tolerance_dbu': True,
            'legal_no_halo': not overlaps and not out_of_core}


def native_macroplacement(run, units, actual):
    """Read post-restoration integer-DBU OpenDB evidence, not pre-restoration MPL logs."""
    manifest = json.loads((run / 'manifest.json').read_text())
    task, correction = manifest['task'], manifest['macro_import']
    assert task['macro_import_policy'] == correction['policy'] == 'preserve_generator_dbu_coordinates_v1'
    assert task['diagnostic'] is True and task['not_primary_table2_result'] is True
    assert task['recipe_label'] == 'released_flow_exact_macro_import_v1'
    assert correction['helper_sha256'] == '3ab91cc16ab2408020998ecaa03ea0e4384759c55f32d9e456e9f6e5f36e88d2'
    assert sha(run / 'provenance/macro_import.py') == correction['helper_sha256']
    assert sha(run / 'provenance/run_table2.py') == '0a8ea3d2605eb4f0bf9b661117ab936ae0ebcce7532a7cee53d82d5dfe9df58a'
    assert sha(run / 'provenance/audit_table2_macro_preservation.py') == correction['parser_sha256'] == '521fe0f12165be6590ca407b4bc52509cf6e014ce77a496e7011c664f3cc37e4'
    assert correction['dbu_per_um'] == units and correction['macro_count'] == len(actual)
    assert correction['input_def_sha256'] == task['def_sha256']
    for key in ('original_tcl', 'executed_tcl'):
        assert Path(correction[key]).resolve().parent == (run / 'provenance').resolve()
        assert sha(correction[key]) == correction[key + '_sha256']
    original, executed = [Path(correction[k]).read_bytes() for k in ('original_tcl', 'executed_tcl')]
    assert executed.startswith(original), 'Author macro commands were altered'
    assert executed.count(b'getPlacementStatus] ne "LOCKED"') == len(actual)
    candidates = list((run / 'orfs_work/logs').glob('*/*/eval/2_3_floorplan_macro.log'))
    assert len(candidates) == 1
    log, placed = candidates[0], {}
    digest = sha(log)
    for line in log.read_text().splitlines():
        if not line.startswith('TABLE2_ODB_EXACT\t'):
            continue
        fields = line.split('\t')
        assert len(fields) == 9, 'Malformed exact-coordinate record'
        _, name, master, orient, status, *coordinates = fields
        assert name not in placed and status == 'LOCKED'
        placed[name] = {'bbox_dbu': [int(v) for v in coordinates], 'master': master,
                        'orientation': {'R0': 'N', 'MX': 'FS'}[orient], 'status': status}
    assert set(placed) == set(actual), 'Post-restoration macro inventory differs'
    disagreements = []
    for name, macro in actual.items():
        native = placed[name]
        if native['bbox_dbu'][:2] != macro['xy_dbu'] or native['orientation'] != macro['orientation'] or native['master'] != macro['master'] or macro['status'] != 'FIXED':
            disagreements.append({'name': name, 'native': native, 'final': macro})
    assert sha(log) == digest
    return {'log': str(log), 'log_sha256': digest, 'macro_count': len(placed),
            'final_matches_native': not disagreements, 'disagreements': disagreements,
            'comparison_tolerance_dbu': 0, 'policy': correction['policy'],
            'scope': 'Exact integer DBU coordinates and LOCKED status printed after uniform OpenDB restoration; final DEF must remain FIXED.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    assert sha(args.summary) == args.summary_sha256
    summary = json.loads(args.summary.read_text())
    assert len(summary['rows']) == 12, 'Keep the full twelve-outcome denominator'
    output = args.output.absolute()
    assert output.is_relative_to(REPORTS) and output != REPORTS and output.resolve() == output
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.summary, output / 'source_summary.json')
    shutil.copyfile(__file__, output / Path(__file__).name)
    results = []
    for row in summary['rows']:
        result = {k: row[k] for k in ('method', 'design', 'run_id', 'status')}
        if row['status'] != 'validated':
            result['macro_check'] = 'not_evaluated_unvalidated_run'
            results.append(result)
            continue
        run = Path(row['run'])
        assert sha(run / 'manifest.json') == row['manifest_sha256']
        manifest = json.loads((run / 'manifest.json').read_text())
        task = manifest['task']
        source = Path(manifest['executed_def_path'])
        assert sha(source) == row['def_sha256'] == task['def_sha256']
        geometry = Path(task['geometry_audit'])
        assert sha(geometry) == task['geometry_audit_sha256']
        audit = json.loads(geometry.read_text())
        assert audit['output_sha256'] == sha(source) and audit['legal_no_halo'] is True
        candidates = list((run / 'orfs_work/results').glob('*/*/eval/6_final.def'))
        assert len(candidates) == 1, 'Expected one final routed DEF'
        final = candidates[0]
        final_sha = sha(final)
        input_units, expected = macros(source)
        final_units, actual = macros(final)
        assert input_units == final_units and set(expected) == set(actual)
        expected_count = audit.get('macro_count', len(audit.get('macros', [])))
        assert len(expected) == expected_count
        changes = []
        for name, original in expected.items():
            assert original['status'] == 'FIXED', 'Handoff macro was not fixed'
            assert original['master'] == actual[name]['master'], 'Macro master changed'
            if {k: original[k] for k in ('xy_dbu', 'orientation')} != {
                    k: actual[name][k] for k in ('xy_dbu', 'orientation')}:
                changes.append({'name': name, 'input': original, 'final': actual[name],
                                'delta_xy_dbu': [actual[name]['xy_dbu'][i] - original['xy_dbu'][i] for i in (0,1)]})
        geometric = geometry_check(run, manifest, final_units, actual)
        native = native_macroplacement(run, final_units, actual)
        durable = archive_evidence(run, final, final_sha)
        assert sha(final) == final_sha and sha(run / 'manifest.json') == row['manifest_sha256']
        result.update(macro_check='passed' if not changes else 'different_final_placement',
                      macro_count=len(expected),dbu_per_um=input_units,changed_macro_count=len(changes),
                      changes=changes,final_macro_geometry=geometric,
                      native_macroplacement=native,
                      input_def_sha256=sha(source),final_def=str(final),final_def_sha256=final_sha,
                      macros=actual,durable_archive_evidence=durable)
        results.append(result)
    checked = sum(r.get('macro_check') in ('passed','different_final_placement') for r in results)
    passed = sum(r.get('macro_check') == 'passed' for r in results)
    report = {'requested': 12, 'checked': checked, 'complete': checked == 12,
              'preserved_count': passed, 'all_macros_preserved': checked == 12 and passed == 12,
              'all_checked_final_macros_legal': all(r['final_macro_geometry']['legal_no_halo']
                                                  for r in results if 'final_macro_geometry' in r),
              'all_checked_final_match_native_macroplacement': all(r['native_macroplacement']['final_matches_native']
                                                   for r in results if 'native_macroplacement' in r),
              'macro_count_checked': sum(r.get('macro_count', 0) for r in results),
              'summary_sha256': args.summary_sha256, 'runner_sha256': sha(__file__),
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'rows': results,
              'scope': 'Exact final-versus-handoff macro identity, lower-left and orientation. '
                       'Final macro rectangle legality is independently recomputed with frozen LEF dimensions and zero DBU tolerance. '
                       'All final coordinates must equal generator coordinates and post-restoration integer-DBU OpenDB output. '
                       'Does not prove standard-cell/routing DRC, functional equivalence, or every intermediate placement. '
                       'Archive evidence binds inventory and committed manifest; compressed archives are not re-decoded here.'}
    assert sha(args.summary) == args.summary_sha256
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ('requested','checked','complete','macro_count_checked')}))
    return 0 if report['all_macros_preserved'] and report['all_checked_final_match_native_macroplacement'] and report['all_checked_final_macros_legal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
