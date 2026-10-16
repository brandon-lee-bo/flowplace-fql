#!/usr/bin/env python3
"""Resume a preempted fixed fullflow from its completed GRT checkpoint, unchanged cap."""
import argparse, hashlib, importlib.util, json, os, re, shutil, socket, subprocess, tarfile, time
from pathlib import Path
from grt_metrics import atomic_json
from result_storage import create_grt_run_dir

BASE = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument('--source-run', required=True); p.add_argument('--run-id', required=True)
a = p.parse_args()
source = BASE / 'results' / a.source_run
old = json.loads((source/'manifest.json').read_text())
evidence = json.loads((source/'preemption.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if evidence['state'] != 'PREEMPTED' or evidence['manifest_sha256'] != sha(source/'manifest.json'):
    raise SystemExit('Missing hash-bound source preemption evidence')
records = [line.split('|') for line in evidence['sacct_record'].splitlines() if line.strip()]
if str(evidence['slurm_job_id']) != str(old['slurm_job_id']) or not any(
        row[0] == str(old['slurm_job_id']) and row[1] == 'PREEMPTED' for row in records):
    raise SystemExit('Preemption evidence does not match original raw Slurm job ID')
if old.get('success') or sha(Path(old['def_path'])) != old['def_sha256']:
    raise SystemExit('Source must be an interrupted unchanged input')
flow = BASE/'thirdparty/OpenROAD-flow-scripts'
platform = flow/'platforms/nangate45/config.mk'
if sha(platform) != '3147d6564db3a82b0fe18ac3172ea7a8c355fde235c8204b0fe92564acf3d1fd':
    raise SystemExit('Released platform configuration changed')
if sha(Path(old['design_config'])) != old['design_config_sha256'] or sha(Path(os.environ['OPENROAD_EXE'])) != old['openroad_sha256']:
    raise SystemExit('Source/current design or binary identity mismatch')
actual_paths = {
    'tool_image_sha256': Path('/fact_data/bohanli/flowplace_fql_backend/containers/orfs_26Q4-151-g8b5e62fbb.sif'),
    'flow_archive_sha256': BASE/'assets/OpenROAD-flow-scripts.tar.gz',
    'backend_patch_sha256': flow/'scripts/write_ref_sdc.tcl',
}
actual_provenance = {}
for key, path in actual_paths.items():
    actual = sha(path)
    if actual != old[key]:
        raise SystemExit(f'Actual runtime {key} differs from original recorded value')
    actual_provenance[key] = {'path': str(path), 'actual_sha256': actual,
                              'original_recorded_sha256': old[key], 'matches': True}
spec = importlib.util.spec_from_file_location('author', BASE.parent/'upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py')
author = importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
meta = author.resolve_design(str(flow), 'nangate45', old['design']); design = meta['design_nickname']; variant = old['variant']
relative = Path('nangate45')/design/variant
grt_log = source/'orfs_work/logs'/relative/'5_1_grt.log'
if 'Elapsed time:' not in grt_log.read_text() or '[ERROR' in grt_log.read_text():
    raise SystemExit('Source GRT stage lacks successful completion evidence')
work, storage = create_grt_run_dir(BASE, a.run_id); fw = work/'orfs_work'
runner_copy = work/'runner_source.py'
shutil.copy2(Path(__file__), runner_copy)
# Capture this run's executed-flow inputs. There was no original startup tree
# snapshot; archive/patch agreement cannot retroactively establish one.
selected_roots = [flow/'scripts', flow/'util', flow/'platforms/nangate45',
                  Path(old['design_config']).parent, flow/'results'/relative]
selected = {flow/'Makefile'}
for directory in selected_roots:
    selected.update(path for path in directory.rglob('*') if path.is_file())
flow_files = {str(path.relative_to(flow)): sha(path) for path in sorted(selected)}
with tarfile.open(actual_paths['flow_archive_sha256']) as archive:
    for relative_path, actual in flow_files.items():
        if relative_path == 'scripts/write_ref_sdc.tcl':
            continue  # Its current patched bytes were verified separately above.
        member = archive.getmember('thirdparty/OpenROAD-flow-scripts/' + relative_path)
        original = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        if original != actual:
            raise SystemExit(f'Unexpected released-flow file modification: {relative_path}')
flow_snapshot = work/'flow_execution_files.json'
atomic_json(flow_snapshot, {'files': flow_files, 'flow_root': str(flow),
    'selection': ['Makefile', 'scripts/**', 'util/**', 'platforms/nangate45/**',
                  str(Path(old['design_config']).parent.relative_to(flow))+'/**',
                  str((flow/'results'/relative).relative_to(flow))+'/**'],
    'verified_against_release_archive_except_patch': True,
    'allowed_patch_file': 'scripts/write_ref_sdc.tcl',
    'source_startup_tree_hash_available': False,
    'limitation': 'This snapshot describes this resumed execution only. The interrupted attempt did not capture a startup flow-tree hash.'})
manifest = {key: old[key] for key in ['design','def_path','def_sha256','variant','openroad_sha256','tool_image_sha256','design_config','design_config_sha256','flow_archive_sha256','backend_patch_sha256']}
manifest.update(run_id=a.run_id, source_run=str(source), source_manifest_sha256=sha(source/'manifest.json'),
    source_preemption_sha256=sha(source/'preemption.json'), source_slurm_job_id=old['slurm_job_id'],
    slurm_job_id=os.getenv('SLURM_JOB_ID'), cpus=os.getenv('NUM_CORES'), hostname=socket.gethostname(),
    platform_config_sha256=sha(platform), storage=storage, time_start=time.time(), success=False,
    resumed_original_flow=True, source_drt_end_iter=4, resumed_drt_end_iter=4,
    optimization_overrides={}, stage_wall_seconds={}, stage_exit_codes={}, copied_checkpoint_files=[],
    runner_source_sha256=sha(runner_copy), runner_source_copy=str(runner_copy),
    runner_capture='Copied and hashed at startup before checkpoint copying or optimization',
    actual_runtime_provenance=actual_provenance, flow_execution_files_manifest=str(flow_snapshot),
    flow_execution_files_manifest_sha256=sha(flow_snapshot), source_flow_tree_startup_hash_available=False,
    source_flow_tree_comparability_limit='Original run has no startup tree snapshot; only its recorded archive/patch, binary, design, and completed GRT evidence can be checked.',
    runtime_scope='Resumed original-cap4 detailed route, fill and finish only; excludes prior pre-route execution and interrupted DRT work.')
atomic_json(work/'manifest.json', manifest)
try:
    author.seed_synth_artifacts(str(flow), str(fw), 'nangate45', design, variant)
    for group, names in [('results',['5_1_grt.odb','5_1_grt.sdc','route.guide']),('logs',['5_1_grt.log','5_1_grt.json'])]:
        for name in names:
            src=source/'orfs_work'/group/relative/name; dst=fw/group/relative/name
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
            if sha(src)!=sha(dst): raise RuntimeError('Checkpoint copy differs')
            manifest['copied_checkpoint_files'].append({'source':str(src),'destination':str(dst),'sha256':sha(src)})
    common=[f'DESIGN_CONFIG={meta["config_path"]}',f'FLOW_VARIANT={variant}',f'WORK_HOME={fw}']
    # Deliberately no DETAILED_ROUTE_ARGS override and no do-route target.
    stages=[('resumed_detailed_route',['do-5_2_route']),('fill_and_final_ppa',['do-5_3_fillcell','do-5_route','do-5_route.sdc','do-finish'])]
    for name, targets in stages:
        command=['make',*targets,*common]; manifest.update(active_stage=name,active_command=command)
        atomic_json(work/'manifest.json',manifest); start=time.time()
        with (work/f'{name}.log').open('w') as log:
            proc=subprocess.run(command,cwd=flow,stdout=log,stderr=subprocess.STDOUT)
        manifest['stage_wall_seconds'][name]=time.time()-start; manifest['stage_exit_codes'][name]=proc.returncode
        if proc.returncode: raise RuntimeError(f'{name} exited {proc.returncode}')
    stdout='\n'.join((work/f'{name}.log').read_text() for name,_ in stages)
    metrics=author.parse_metrics_from_files(stdout,str(fw),str(flow),'nangate45',design,variant)
    metrics.update(runtime=time.time()-manifest['time_start'],runtime_scope=manifest['runtime_scope'],resumed_original_flow=True)
    atomic_json(work/'metrics.json',metrics)
    for item in manifest['copied_checkpoint_files']:
        if sha(Path(item['source']))!=item['sha256'] or sha(Path(item['destination']))!=item['sha256']:
            raise RuntimeError('Original/copied GRT checkpoint changed')
    manifest['source_and_copied_grt_checkpoint_hashes_unchanged']=True
    if any(sha(flow/path) != expected for path,expected in flow_files.items()):
        raise RuntimeError('Captured executed flow file changed during recovery')
    manifest['captured_flow_files_unchanged_at_finish']=True
    manifest['route_drc_clean']=metrics.get('DRC')==0
    manifest['success']=all(metrics.get(k) is not None for k in ['DRT_WL','DRC','WNS','TNS','Power','StdCellArea'])
    if not manifest['success']: raise RuntimeError('Incomplete final PPA')
    print(json.dumps(metrics,indent=2))
except Exception as exc:
    manifest['failure']=repr(exc); raise
finally:
    manifest['time_end']=time.time(); atomic_json(work/'manifest.json',manifest)
