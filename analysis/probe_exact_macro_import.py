#!/usr/bin/env python3
"""Isolated coordinate-preservation probe; no GP, CTS, routing or PPA tuning."""
import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/fact_data/bohanli/flowplace_fql_backend')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def brace(value):
    text = str(value)
    assert not any(c in text for c in '{}\\\r\n'), 'Unsupported Tcl token'
    return '{' + text + '}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--openroad', type=Path, required=True)
    p.add_argument('--mode', choices=['native-exact', 'odb-restore'], default='native-exact')
    args = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    source = ROOT / 'analysis/audit_table2_macro_preservation.py'
    assert sha(source) == '521fe0f12165be6590ca407b4bc52509cf6e014ce77a496e7011c664f3cc37e4'
    spec = importlib.util.spec_from_file_location('macro_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = args.run.resolve()
    manifest_raw = (run / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    assert sha(args.openroad) == manifest['openroad_sha256']
    original = Path(manifest['executed_def_path'])
    assert sha(original) == manifest['task']['def_sha256']
    units, macros = module.macros(original)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'source_manifest.json').write_bytes(manifest_raw)
    shutil.copyfile(__file__, args.output / Path(__file__).name)
    inventory = json.loads((run / 'provenance/flow_inputs_start.json').read_text())
    files = {e['relative_path']: e for e in inventory['files']}
    names = ['platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef',
             'platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef']
    names += sorted({'platforms/nangate45/lef/' + m['master'] + '.lef' for m in macros.values()})
    tcl, identities = [], {}
    for name in names:
        entry = files[name]
        path = Path(entry['path'])
        assert sha(path) == entry['sha256']
        identities[name] = entry['sha256']
        tcl.append('read_lef ' + brace(path))
    tcl += ['read_def ' + brace(original)]
    for name, m in macros.items():
        x, y = [str(Decimal(v) / units) for v in m['xy_dbu']]
        orientation = {'N': 'R0', 'FS': 'MX'}[m['orientation']]
        tcl += ['set probe_inst [[ord::get_db_block] findInst ' + brace(name) + ']',
                '$probe_inst setPlacementStatus PLACED']
        tcl.append('place_macro -macro_name ' + brace(name) + ' -location {' + x + ' ' + y +
                   '} -orientation ' + orientation + (' -exact' if args.mode == 'native-exact' else ''))
        if args.mode == 'odb-restore':
            tcl += ['set probe_status [$probe_inst getPlacementStatus]',
                    '$probe_inst setPlacementStatus PLACED',
                    '$probe_inst setLocation ' + ' '.join(str(v) for v in m['xy_dbu']),
                    '$probe_inst setPlacementStatus $probe_status',
                    'set probe_box [$probe_inst getBBox]',
                    'puts "ODB_STATUS\\t[$probe_inst getName]\\t$probe_status\\t[$probe_inst getPlacementStatus]"',
                    'puts "ODB_EXACT\\t[$probe_inst getName]\\t[$probe_inst getOrient]\\t[$probe_box xMin]\\t[$probe_box yMin]\\t[$probe_box xMax]\\t[$probe_box yMax]"']
    command_path = args.output / 'exact_import.tcl'
    command_path.write_text('\n'.join(tcl) + '\n')
    command = [str(args.openroad), '-exit', '-no_init', '-threads', '1', '-no_splash', str(command_path)]
    with (args.output / 'openroad.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'Native exact-import probe failed; inspect preserved OpenROAD log'
    log_path = args.output / 'openroad.log'
    pattern = (r'\[INFO MPL-0035\] Macro (.*?) placed\. Bounding box '
               r'\(([-\d.]+)um, ([-\d.]+)um\), \(([-\d.]+)um, ([-\d.]+)um\)\. Orientation (\S+)')
    observed = {}
    if args.mode == 'native-exact':
        records = [(name, [Decimal(v) * units for v in (x0, y0, x1, y1)], orient)
                   for name, x0, y0, x1, y1, orient in re.findall(pattern, log_path.read_text())]
    else:
        records = []
        for line in log_path.read_text().splitlines():
            if line.startswith('ODB_EXACT\t'):
                _, name, orient, *bbox = line.split('\t')
                assert len(bbox) == 4
                records.append((name, [Decimal(v) for v in bbox], orient))
    for name, values, orient in records:
        name = re.sub(r'\\(.)', r'\1', name)
        assert name in macros and name not in observed
        assert all(v == int(v) for v in values)
        observed[name] = dict(macros[name], xy_dbu=[int(v) for v in values[:2]],
                              orientation={'R0': 'N', 'MX': 'FS'}[orient])
    assert set(observed) == set(macros)
    differences = [name for name in macros if macros[name] != observed[name]]
    geometry = module.geometry_check(run, manifest, units, observed)
    statuses = [line.split('\t')[1:] for line in log_path.read_text().splitlines()
                if line.startswith('ODB_STATUS\t')]
    if args.mode == 'odb-restore':
        assert len(statuses) == len(macros) and all(before == after == 'LOCKED' for name, before, after in statuses)
    report = {'success': not differences and geometry['legal_no_halo'], 'run': str(run),
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'macro_count': len(macros), 'mode': args.mode,
              'differences': differences, 'geometry': geometry, 'input_def_sha256': sha(original),
              'openroad_sha256': sha(args.openroad), 'runner_sha256': sha(__file__),
              'lef_sha256': identities, 'command': command, 'command_tcl_sha256': sha(command_path),
              'native_log_sha256': sha(log_path), 'observed_macros': observed,
              'native_placement_status': statuses,
              'scope': 'Coordinate import only. No layout optimization, synthesis, CTS, routing, or PPA evaluation. Does not replace any primary row.'}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'success': report['success'], 'macro_count': len(macros), 'differences': differences}))
    assert report['success']


if __name__ == '__main__':
    main()
