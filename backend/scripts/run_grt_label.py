#!/usr/bin/env python3
"""Evaluate a full placement through CTS/global route, explicitly not final PPA."""
import argparse, hashlib, importlib.util, json, os, re, socket, subprocess, time
from pathlib import Path
from result_storage import create_grt_run_dir
from grt_metrics import parse_grt_metrics, atomic_json, PARSER_VERSION
from grt_provenance import stamp_provenance
BASE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--design',default='bp_fe');p.add_argument('--def-path',type=Path);p.add_argument('--run-id',required=True);p.add_argument('--storage',choices=['home','bulk'],default='home')
a=p.parse_args();flow=BASE/'thirdparty'/'OpenROAD-flow-scripts'
def_path=(a.def_path or BASE/'releases'/'flowplace_DEFS'/'openroad'/f'{a.design}.def').resolve()
work,storage=create_grt_run_dir(BASE,a.run_id,a.storage);fw=work/'orfs_work'
spec=importlib.util.spec_from_file_location('author_evaluator',BASE.parent/'upstream'/'flowplace'/'benchmark_utils'/'openroad'/'openroad_ppa_evaluator.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
meta=m.resolve_design(str(flow),'nangate45',a.design)
if meta is None:raise SystemExit('Unknown design')
variant='eval';design=meta['design_nickname'];cfg=meta['config_path']
if not m.seed_synth_artifacts(str(flow),str(fw),'nangate45',design,variant):raise SystemExit('Missing synth seed')
manifest={'stage':'global_route_only','not_final_table2':True,'design':a.design,'def_path':str(def_path),'def_sha256':hashlib.sha256(def_path.read_bytes()).hexdigest(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'num_cores':os.getenv('NUM_CORES'),'time_start':time.time(),'design_config':cfg,'design_config_sha256':hashlib.sha256(Path(cfg).read_bytes()).hexdigest(),'openroad_exe':os.getenv('OPENROAD_EXE'),'openroad_sha256':hashlib.sha256(Path(os.environ['OPENROAD_EXE']).read_bytes()).hexdigest(),'flow_script_patch':'001-global-env-scope.patch','stage_wall_seconds':{},'storage':storage,'success':False}
manifest['hostname']=socket.gethostname()
manifest['tool_image_sha256']='85f0ce74d79a79c1190632f775f0ed818c8de6e0fafd87717ae3d4839207ea4c'
manifest['platform_config_sha256']=hashlib.sha256((flow/'platforms/nangate45/config.mk').read_bytes()).hexdigest()
manifest['grt_marker_helper_sha256']=hashlib.sha256((BASE/'scripts/write_grt_marker_status.tcl').read_bytes()).hexdigest()
manifest['grt_report_wrapper_sha256']=hashlib.sha256((BASE/'scripts/global_route_label.tcl').read_bytes()).hexdigest()
stamp_provenance(manifest,BASE)
(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
common=[f'DESIGN_CONFIG={cfg}',f'MACRO_DEF={def_path}',f'FLOW_VARIANT={variant}',f'WORK_HOME={fw}']
logdir=fw/'logs'/'nangate45'/design/variant
try:
 stages=[('macro_import',['run_mp']),('floorplan_place_cts',['do-floorplan','do-place','do-cts']),('global_route_label',['run',f'RUN_SCRIPT={BASE/"scripts"/"global_route_label.tcl"}',f'OR_ARGS=-metrics {logdir/"5_grt_label.json"}'])]
 for name,targets in stages:
  command=['make',*targets,*common];print(subprocess.list2cmdline(command),flush=True);start=time.time()
  with (work/f'{name}.log').open('w') as log:
   proc=subprocess.run(command,cwd=flow,stdout=log,stderr=subprocess.STDOUT)
  manifest['stage_wall_seconds'][name]=time.time()-start
  if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see retained log')
 metrics=parse_grt_metrics(work,time.time()-manifest['time_start'])
 atomic_json(work/'metrics.json',metrics);print(json.dumps(metrics,indent=2),flush=True)
 manifest['success']=True
 manifest['parser_version']=PARSER_VERSION
except Exception as exc:
 manifest['failure']=repr(exc)
 raise

finally:
 manifest['time_end']=time.time();(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
