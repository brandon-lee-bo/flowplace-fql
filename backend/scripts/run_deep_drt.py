#!/usr/bin/env python3
"""Isolated detailed-route-cap diagnostic from an existing immutable GRT checkpoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from grt_metrics import atomic_json
from result_storage import create_grt_run_dir

BASE = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--source-run', required=True)
p.add_argument('--run-id', required=True)
p.add_argument('--end-iter', type=int, default=20)
a = p.parse_args()
source = BASE / 'results' / a.source_run
source_manifest = json.loads((source / 'manifest.json').read_text())
if not source_manifest.get('success'):
    raise SystemExit('Source fullflow must have completed successfully')
work, storage = create_grt_run_dir(BASE, a.run_id)
fw = work / 'orfs_work'
flow = BASE / 'thirdparty/OpenROAD-flow-scripts'
spec = importlib.util.spec_from_file_location('author_evaluator', BASE.parent / 'upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py')
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
meta = author.resolve_design(str(flow), 'nangate45', source_manifest['design'])
design = meta['design_nickname']
config = Path(meta['config_path'])
variant = 'eval'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if sha(config) != source_manifest['design_config_sha256']:
    raise SystemExit('Source/current design config differs')
if sha(Path(os.environ['OPENROAD_EXE'])) != source_manifest['openroad_sha256']:
    raise SystemExit('Source/current OpenROAD binary differs')
manifest = {
    'stage': 'diagnostic_deeper_detailed_route', 'not_default_table2_flow': True,
    'source_run': str(source), 'source_manifest_sha256': sha(source / 'manifest.json'),
    'source_slurm_job_id': source_manifest['slurm_job_id'],
    'source_def_sha256': source_manifest['def_sha256'],
    'source_drt_end_iter': 4, 'diagnostic_drt_end_iter': a.end_iter,
    'purpose': 'Test whether routing violations persist beyond the released iteration cap; not fit paper numbers',
    'design': source_manifest['design'], 'design_config_sha256': sha(config),
    'platform_config_sha256': sha(flow / 'platforms/nangate45/config.mk'),
    'openroad_sha256': source_manifest['openroad_sha256'],
    'tool_image_sha256': source_manifest['tool_image_sha256'],
    'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'hostname': socket.gethostname(),
    'num_cores': os.getenv('NUM_CORES'), 'storage': storage,
    'time_start': time.time(), 'success': False, 'stage_wall_seconds': {},
    'stage_exit_codes': {}, 'copied_checkpoint_files': []}
atomic_json(work / 'manifest.json', manifest)
try:
    author.seed_synth_artifacts(str(flow), str(fw), 'nangate45', design, variant)
    for group, names in [('results', ['5_1_grt.odb', '5_1_grt.sdc', 'route.guide']),
                         ('logs', ['5_1_grt.log', '5_1_grt.json'])]:
        for name in names:
            relative = Path(group) / 'nangate45' / design / variant / name
            old = source / 'orfs_work' / relative
            new = fw / relative
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
            if sha(old) != sha(new):
                raise RuntimeError(f'Checkpoint copy hash mismatch: {relative}')
            manifest['copied_checkpoint_files'].append({'source': str(old), 'destination': str(new), 'sha256': sha(old)})
    arguments = f'-droute_end_iter {a.end_iter} -verbose 1'
    common = [f'DESIGN_CONFIG={config}', f'FLOW_VARIANT={variant}', f'WORK_HOME={fw}', f'DETAILED_ROUTE_ARGS={arguments}']
    manifest['only_optimization_override'] = {'DETAILED_ROUTE_ARGS': arguments}
    atomic_json(work / 'manifest.json', manifest)
    # Explicit substeps avoid do-route, which would rerun global routing.
    stages = [('deep_detailed_route', ['do-5_2_route']),
              ('fill_and_final_ppa', ['do-5_3_fillcell', 'do-5_route', 'do-5_route.sdc', 'do-finish'])]
    for name, targets in stages:
        command = ['make', *targets, *common]
        manifest['active_stage'] = name
        manifest['active_command'] = command
        atomic_json(work / 'manifest.json', manifest)
        print(subprocess.list2cmdline(command), flush=True)
        start = time.time()
        with (work / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=flow, stdout=log, stderr=subprocess.STDOUT)
        manifest['stage_wall_seconds'][name] = time.time() - start
        manifest['stage_exit_codes'][name] = result.returncode
        atomic_json(work / 'manifest.json', manifest)
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}')
    stdout = '\n'.join((work / f'{name}.log').read_text() for name, _ in stages)
    metrics = author.parse_metrics_from_files(stdout, str(fw), str(flow), 'nangate45', design, variant)
    metrics['runtime'] = time.time() - manifest['time_start']
    metrics['stage'] = manifest['stage']
    metrics['diagnostic_drt_end_iter'] = a.end_iter
    metrics['not_default_table2_flow'] = True
    atomic_json(work / 'metrics.json', metrics)
    route_log = (fw / 'logs/nangate45' / design / variant / '5_2_route.log').read_text()
    manifest['drc_iteration_sequence'] = [int(x) for x in re.findall(r'Number of violations = (\d+)', route_log)]
    manifest['route_drc_clean'] = metrics.get('DRC') == 0
    for item in manifest['copied_checkpoint_files']:
        if sha(Path(item['source'])) != item['sha256'] or sha(Path(item['destination'])) != item['sha256']:
            raise RuntimeError('Original/copied GRT checkpoint changed during diagnostic')
    manifest['source_and_copied_grt_checkpoint_hashes_unchanged'] = True
    original = json.loads((source / 'metrics.json').read_text())
    comparison = {key: {'baseline_cap4': original[key], 'diagnostic_cap20': metrics.get(key)}
                  for key in ['DRT_WL', 'DRC', 'WNS', 'TNS', 'Power', 'StdCellArea']}
    atomic_json(work / 'baseline_comparison.json', comparison)
    manifest['success'] = all(metrics.get(key) is not None for key in ['DRT_WL', 'DRC', 'WNS', 'TNS', 'Power', 'StdCellArea'])
    print(json.dumps(metrics, indent=2), flush=True)
    if not manifest['success']:
        raise RuntimeError('Diagnostic finish metrics incomplete')
except Exception as exc:
    manifest['failure'] = repr(exc)
    raise
finally:
    manifest['time_end'] = time.time()
    atomic_json(work / 'manifest.json', manifest)
