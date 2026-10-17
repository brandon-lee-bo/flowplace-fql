"""Separate requested DEF coordinates from native snapping and actual GRT geometry."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess
from analysis.audit_table2_macro_preservation import macros
from .graph_o2o_archive import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
ORIENT = {'R0':'N', 'R90':'W', 'R180':'S', 'R270':'E', 'MY':'FN', 'MX':'FS',
          'MYR90':'FE', 'MXR90':'FW'}


def audit(run):
    run = Path(run)
    meta = json.loads((run/'manifest.json').read_text())
    if not meta.get('success') or not meta.get('time_end'):
        raise ValueError('Read-only GRT audit requires completed physical run')
    choices = list((run/'orfs_work/results/nangate45').glob('*/eval/5_1_grt.odb'))
    if len(choices)!=1:
        raise ValueError('Expected one saved final GRT database')
    odb = choices[0]
    bindings = dict(physical_manifest_sha256=sha(run/'manifest.json'), odb_sha256=sha(odb),
                    input_def_sha256=sha(meta['def_path']))
    path = run/'backend_macro_audit.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['bindings'] != bindings:
            raise ValueError('Existing macro geometry audit has changed inputs')
        return old
    output = run/'backend_macro_audit.log'
    with output.open('w') as log:
        subprocess.run(['bash', str(ROOT/'fql/graph_o2o_geometry.sh'), str(odb)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    core = None
    actual = {}
    for line in output.read_text().splitlines():
        fields = line.split('\t')
        if fields[0]=='GO2O_CORE':
            if core is not None or len(fields)!=6:
                raise ValueError('Invalid core geometry record')
            dbu = int(fields[1]); core = list(map(int, fields[2:]))
        elif fields[0]=='GO2O_MACRO':
            if len(fields)!=9 or fields[1] in actual:
                raise ValueError('Invalid/duplicate macro geometry record')
            actual[fields[1]] = dict(master=fields[2], orientation=ORIENT[fields[3]],
                                    status=fields[4], bbox_dbu=list(map(int, fields[5:])))
    input_dbu, expected = macros(Path(meta['def_path']))
    if core is None or dbu!=input_dbu or set(actual)!=set(expected):
        raise ValueError('Incomplete geometry export or macro identity mismatch')
    changes, overlaps, bounds = [], [], []
    for name, item in actual.items():
        before = expected[name]
        if item['master']!=before['master'] or item['orientation']!=before['orientation']:
            raise ValueError('Backend changed macro master/orientation unexpectedly')
        box = item['bbox_dbu']
        if item['status'] not in ['LOCKED', 'FIRM'] or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid macro placement status or dimensions')
        delta = [box[i]-before['xy_dbu'][i] for i in [0,1]]
        if delta != [0,0]:
            changes.append(dict(name=name, requested_xy_dbu=before['xy_dbu'],
                                actual_xy_dbu=box[:2], displacement_dbu=delta))
        if box[0]<core[0] or box[1]<core[1] or box[2]>core[2] or box[3]>core[3]:
            bounds.append(name)
    for i, (name, a) in enumerate(actual.items()):
        x = a['bbox_dbu']
        for other, b in list(actual.items())[:i]:
            y = b['bbox_dbu']
            dx,dy = min(x[2],y[2])-max(x[0],y[0]), min(x[3],y[3])-max(x[1],y[1])
            if dx>0 and dy>0:
                overlaps.append(dict(a=name,b=other,overlap_dbu=[dx,dy]))
    result = dict(version=1, stage='saved_final_global_route_database', bindings=bindings,
        macro_count=len(actual), dbu_per_um=dbu, core_dbu=core, actual_macros=actual,
        input_to_actual_changes=changes, zero_tolerance_dbu=True, overlap_pairs=overlaps,
        out_of_core=bounds, legal_no_halo=not overlaps and not bounds,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'fql/graph_o2o_geometry.py', ROOT/'fql/graph_o2o_geometry.tcl', ROOT/'fql/graph_o2o_geometry.sh',
             ROOT/'analysis/audit_table2_macro_preservation.py']}, output_sha256=sha(output),
        interpretation='Action requests bounded raw centers; environment includes legalizer, DEF rounding and native track snapping. Actual backend macro boxes retained separately; no exact coordinate preservation claim.',
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    if sha(run/'manifest.json')!=bindings['physical_manifest_sha256'] or sha(odb)!=bindings['odb_sha256']:
        raise ValueError('Read-only audit changed or raced physical evidence')
    atomic_json(path, result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    print(json.dumps(audit(args.run), indent=2))
