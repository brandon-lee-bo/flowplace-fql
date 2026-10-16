#!/usr/bin/env python3
"""Run the pinned author's evaluator while retaining every intermediate artifact."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parents[1]
UPSTREAM = BASE.parent / 'upstream' / 'flowplace'
parser = argparse.ArgumentParser()
parser.add_argument('--design', default='bp_fe')
parser.add_argument('--def-path', type=Path)
parser.add_argument('--run-id', required=True)
parser.add_argument('--variant', default='eval')
args = parser.parse_args()
def_path = (args.def_path or BASE/'releases'/'flowplace_DEFS'/'openroad'/f'{args.design}.def').resolve()
work = BASE/'results'/args.run_id
work.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('author_evaluator', UPSTREAM/'benchmark_utils'/'openroad'/'openroad_ppa_evaluator.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
manifest = {
 'design': args.design, 'def_path': str(def_path),
 'def_sha256': hashlib.sha256(def_path.read_bytes()).hexdigest(),
 'run_id': args.run_id, 'variant': args.variant, 'time_start': time.time(),
 'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'cpus': os.getenv('NUM_CORES'),
 'openroad_exe': os.getenv('OPENROAD_EXE'),
 'command': subprocess.list2cmdline(__import__('sys').argv),
}
for executable in [os.getenv('OPENROAD_EXE','openroad')]:
 r=subprocess.run([executable,'-version'],capture_output=True,text=True)
 manifest['openroad_version']=r.stdout.strip() or r.stderr.strip()
 manifest['openroad_sha256']=hashlib.sha256(Path(executable).read_bytes()).hexdigest()
 manifest['backend_patch_sha256']=hashlib.sha256((BASE/'thirdparty'/'OpenROAD-flow-scripts'/'scripts'/'write_ref_sdc.tcl').read_bytes()).hexdigest()
 manifest['flow_archive_sha256']='4a0e7841a9ea6f580b3253bf7e57f696dde73f0e03545e2d4026c6229cba631f'
 manifest['tool_image_sha256']='85f0ce74d79a79c1190632f775f0ed818c8de6e0fafd87717ae3d4839207ea4c'
 manifest['environment']={k:os.getenv(k) for k in ['FLOW_HOME','DESIGN_HOME','PLATFORM_HOME','SCRIPTS_DIR','UTILS_DIR','NUM_CORES','OMP_NUM_THREADS','YOSYS_EXE','KLAYOUT_CMD']}
 meta=mod.resolve_design(str(BASE/'thirdparty'/'OpenROAD-flow-scripts'),'nangate45',args.design)
 config_path=Path(meta['config_path'])
 manifest['design_config']=str(config_path)
 manifest['design_config_sha256']=hashlib.sha256(config_path.read_bytes()).hexdigest()
 manifest['design_config_exports']=mod.parse_config_exports(str(config_path))
(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
try:
 result=mod.run_evaluation(str(def_path),args.design,'nangate45',args.variant,str(work),str(BASE),flow_work_home=str(work/'orfs_work'),cleanup_flow_work='never')
 manifest['time_end']=time.time()
 manifest['success']=bool(result) and all(result.get(k) is not None for k in ['DRT_WL','WNS','TNS','Power','StdCellArea'])
 manifest['route_drc_clean']=(result.get('DRC') == 0) if result and result.get('DRC') is not None else None
 manifest['success_definition']='Author flow completed and final PPA metrics present; route_drc_clean is a separate physical-validity check.'
 (work/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
finally:
 (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if not manifest['success']:
 raise SystemExit('Evaluation failed or final metrics incomplete; logs retained.')
