#!/bin/bash
#SBATCH --job-name=table2_native_macro_early
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/table2_native_macro_early_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
python3 - <<'PY'
import importlib.util, json, hashlib, pathlib, re, os
from decimal import Decimal
source = pathlib.Path('analysis/audit_table2_macro_preservation.py')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(source) == '521fe0f12165be6590ca407b4bc52509cf6e014ce77a496e7011c664f3cc37e4'
spec = importlib.util.spec_from_file_location('macro_audit', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
output = pathlib.Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports/native_macro_geometry_early_v1')
output.mkdir(exist_ok=False)
state = pathlib.Path('backend/plans/table2_background_v1/state.json').read_bytes()
(output / 'source_state.json').write_bytes(state)
rows = []
for key, item in json.loads(state)['selected'].items():
    if key not in ['FlowPlace/ariane133', 'FlowPlace/ariane136', 'DREAMPlace/ariane133',
                   'DREAMPlace/ariane136', 'DREAMPlace/bp', 'DREAMPlace/swerv_wrapper']:
        continue
    run = pathlib.Path('/fact_data/bohanli/flowplace_fql_backend/table2_results') / item['task']['run_id']
    raw = (run / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    (output / (run.name + '_manifest_snapshot.json')).write_bytes(raw)
    units, actual = module.macros(pathlib.Path(manifest['executed_def_path']))
    candidates = list((run / 'orfs_work/logs').glob('*/*/eval/2_3_floorplan_macro.log'))
    assert len(candidates) == 1
    log = candidates[0]
    text = log.read_bytes()
    pattern = (r'\[INFO MPL-0035\] Macro (.*?) placed\. Bounding box '
               r'\(([-\d.]+)um, ([-\d.]+)um\), \(([-\d.]+)um, ([-\d.]+)um\)\. Orientation (\S+)')
    placed = set()
    for name, x0, y0, x1, y1, orientation in re.findall(pattern, text.decode()):
        name = re.sub(r'\\(.)', r'\1', name)
        assert name in actual and name not in placed
        placed.add(name)
        values = [Decimal(v) * units for v in (x0, y0, x1, y1)]
        assert all(v == int(v) for v in values)
        actual[name]['xy_dbu'] = [int(v) for v in values[:2]]
        actual[name]['orientation'] = {'R0': 'N', 'MX': 'FS'}[orientation]
    assert placed == set(actual)
    geometry = module.geometry_check(run, manifest, units, actual)
    assert log.read_bytes() == text
    rows.append({'method_design': key, 'run_id': run.name, 'macro_count': len(actual),
                 'native_log': str(log), 'native_log_sha256': hashlib.sha256(text).hexdigest(),
                 'geometry': geometry})
assert len(rows) == 6
report = {'requested': 6, 'checked': len(rows), 'macro_count_checked': sum(r['macro_count'] for r in rows),
          'all_native_macros_legal': all(r['geometry']['legal_no_halo'] for r in rows),
          'source_state_sha256': hashlib.sha256(state).hexdigest(), 'runner_source_sha256': digest(source),
          'slurm_job_id': os.environ['SLURM_JOB_ID'], 'rows': rows,
          'scope': 'Early native macro-placement geometry only, not a final-DEF or PPA audit. All six primary physical jobs continue unchanged.'}
(output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['checked', 'macro_count_checked', 'all_native_macros_legal']}))
assert report['all_native_macros_legal']
PY
