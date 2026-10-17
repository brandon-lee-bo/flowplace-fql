"""Continue an immutable measured GRT checkpoint with a fixed detailed-route cap.

Run inside the pinned physical-tool container. This is a new common evaluation
protocol, not a restart of the paused historical Table 2 reproduction.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'backend'
sys.path.insert(0,str(BASE/'scripts'))
sys.path.insert(0,str(ROOT))
from result_storage import create_grt_run_dir
from grt_metrics import atomic_json
from grt_provenance import stamp_provenance
from canonical_table2_metrics import canonicalize
from analysis.audit_table2_macro_preservation import macros


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1024*1024),b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--end-iter',type=int,default=20)
    p.add_argument('--method',required=True)
    p.add_argument('--storage',choices=['bulk','scratch'],default='bulk')
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or args.end_iter<1:
        raise RuntimeError('Use Slurm and a positive common route cap')
    source=args.source_run.resolve()
    old=json.loads((source/'manifest.json').read_text())
    geometry_path=source/'backend_macro_audit.json'
    geometry=json.loads(geometry_path.read_text())
    if not old.get('success') or not old.get('time_end') or not geometry['legal_no_halo']:
        raise ValueError('Source must be a completed, independently macro-legal GRT checkpoint')
    if sha(source/'manifest.json')!=geometry['bindings']['physical_manifest_sha256']:
        raise ValueError('Source geometry/physical manifest mismatch')
    flow=BASE/'thirdparty/OpenROAD-flow-scripts'
    spec=importlib.util.spec_from_file_location('author_evaluator',ROOT/'upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py')
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    meta=author.resolve_design(str(flow),'nangate45',old['design'])
    config=Path(meta['config_path']); design=meta['design_nickname'];variant='eval'
    if sha(config)!=old['design_config_sha256'] or sha(os.environ['OPENROAD_EXE'])!=old['openroad_sha256']:
        raise ValueError('Source and detailed-route physical tool/config differ')
    work,storage=create_grt_run_dir(BASE,args.run_id,args.storage)
    fw=work/'orfs_work'
    manifest=dict(stage='finish_after_detailed_route', protocol='graph_o2o_grt_continuation_cap_v1',
        not_historical_table2=True, method=args.method, design=old['design'],
        source_grt_run=str(source), source_grt_manifest_sha256=sha(source/'manifest.json'),
        source_geometry_sha256=sha(geometry_path), source_grt_metrics_sha256=sha(source/'metrics.json'),
        def_path=old['def_path'],def_sha256=old['def_sha256'],
        design_config=str(config),design_config_sha256=sha(config),
        openroad_exe=os.environ['OPENROAD_EXE'],openroad_sha256=old['openroad_sha256'],
        num_cores=os.environ['NUM_CORES'],slurm_job_id=os.environ['SLURM_JOB_ID'],
        hostname=socket.gethostname(),time_start=time.time(),success=False,storage=storage,
        route_end_iteration=args.end_iter,stage_wall_seconds={},stage_exit_codes={},copied_inputs=[])
    stamp_provenance(manifest,BASE)
    # The shared helper identifies the GRT source flow, but its default runner
    # and parser fields describe run_grt_label.py rather than this continuation.
    manifest['shared_grt_provenance_defaults'] = {key: manifest[key] for key in
        ['runner_sha256', 'initial_parser_sha256', 'executed_reporting_revision', 'evaluator_revision']}
    manifest.update(runner_sha256=sha(__file__),
        initial_parser_sha256=sha(BASE/'scripts/canonical_table2_metrics.py'),
        shared_provenance_helper_sha256=sha(BASE/'scripts/grt_provenance.py'),
        source_grt_reporting_revision=old.get('executed_reporting_revision'),
        executed_reporting_revision='graph_o2o_drt_raw_final_json_v1',
        evaluator_revision='graph_o2o_grt_continuation_cap_v1')
    manifest['detailed_route_flow_components'] = {name: sha(flow/name) for name in
        ['Makefile', 'scripts/detail_route.tcl', 'scripts/fillcell.tcl',
         'scripts/final_report.tcl', 'scripts/report_metrics.tcl']}
    identity = dict(grt_components=manifest['physical_flow_components'],
                    finish_components=manifest['detailed_route_flow_components'],
                    route_end_iteration=args.end_iter, num_cores=int(os.environ['NUM_CORES']))
    manifest['detailed_route_physical_revision'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if manifest['physical_flow_components']!=old['physical_flow_components']:
        raise ValueError('Physical flow changed since GRT measurement')
    save=lambda:atomic_json(work/'manifest.json',manifest)
    save()
    try:
        provenance=work/'provenance';provenance.mkdir()
        for name in ['fql/graph_o2o_finish.py','fql/graph_o2o_finish.sh','backend/scripts/canonical_table2_metrics.py']:
            shutil.copyfile(ROOT/name,provenance/Path(name).name)
        if not author.seed_synth_artifacts(str(flow),str(fw),'nangate45',design,variant):
            raise ValueError('Missing common synthesis seeds')
        for name in ['1_synth.v','1_synth.sdc']:
            relative=Path('results/nangate45')/design/variant/name
            if sha(fw/relative)!=sha(source/'orfs_work'/relative):
                raise ValueError('Synthesis seed differs from the measured GRT environment')
        for name in ['5_1_grt.odb','5_1_grt.sdc','route.guide']:
            relative=Path('results/nangate45')/design/variant/name
            original=source/'orfs_work'/relative; copied=fw/relative
            copied.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,copied)
            if sha(original)!=sha(copied):
                raise ValueError('GRT checkpoint changed during copy')
            manifest['copied_inputs'].append(dict(source=str(original),copy=str(copied),sha256=sha(original)))
        for group in ['logs','reports']:
            original=source/'orfs_work'/group/'nangate45'/design/variant
            shutil.copytree(original,fw/group/'nangate45'/design/variant)
        arguments=f'-droute_end_iter {args.end_iter} -verbose 1'
        common=[f'DESIGN_CONFIG={config}',f'FLOW_VARIANT={variant}',f'WORK_HOME={fw}',f'DETAILED_ROUTE_ARGS={arguments}']
        manifest['only_optimization_override']={'DETAILED_ROUTE_ARGS':arguments}
        save()
        stages=[('detailed_route',['do-5_2_route']),
                ('finish',['do-5_3_fillcell','do-5_route','do-5_route.sdc','do-finish'])]
        for name,targets in stages:
            command=['make',*targets,*common];manifest['active_stage']=name;save()
            print(subprocess.list2cmdline(command),flush=True)
            start=time.time()
            with (work/(name+'.log')).open('w') as log:
                process=subprocess.run(command,cwd=flow,stdout=log,stderr=subprocess.STDOUT)
            manifest['stage_exit_codes'][name]=process.returncode
            manifest['stage_wall_seconds'][name]=time.time()-start;save()
            if process.returncode:
                raise RuntimeError(f'{name} exited {process.returncode}; preserve the failed physical outcome')
        text='\n'.join((work/(name+'.log')).read_text() for name,_ in stages)
        author_metrics=author.parse_metrics_from_files(text,str(fw),str(flow),'nangate45',design,variant)
        author_metrics.update(runtime=time.time()-manifest['time_start'],stage=manifest['stage'])
        atomic_json(work/'author_metrics.json',author_metrics)
        metrics,evidence=canonicalize(work,author_metrics)
        atomic_json(work/'metrics.json',metrics);atomic_json(work/'metric_canonicalization.json',evidence)
        logs=fw/'logs/nangate45'/design/variant; reports=fw/'reports/nangate45'/design/variant
        route_log=(logs/'5_2_route.log').read_text()
        counts=re.findall(r'Number of violations = (\d+)\.',route_log)
        if not counts or int(counts[-1])!=metrics['DRC'] or '[INFO DRT-0198] Complete detail routing.' not in route_log:
            raise ValueError('Final raw DRC metric lacks matching completed native route evidence')
        drc_report=reports/'5_route_drc.rpt'
        exported=Counter(re.findall(r'^violation type:\s*(.+)$',drc_report.read_text(),re.M)) if drc_report.exists() else Counter()
        if metrics['DRC']==0 and sum(exported.values()):
            raise ValueError('Zero raw DRC contradicts exported violations')
        if metrics['DRC']>0 and not exported:
            raise ValueError('Positive raw DRC without any exported location evidence')
        final=fw/'results/nangate45'/design/variant/'6_final.def'
        dbu,actual=macros(final)
        if dbu!=geometry['dbu_per_um'] or set(actual)!=set(geometry['actual_macros']):
            raise ValueError('Final macro inventory/unit mismatch')
        for name,item in actual.items():
            prior=geometry['actual_macros'][name]
            if (item['xy_dbu']!=prior['bbox_dbu'][:2] or item['master']!=prior['master'] or
                item['orientation']!=prior['orientation'] or item['status']!='FIXED'):
                raise ValueError('Detailed route changed frozen GRT macro geometry')
        for item in manifest['copied_inputs']:
            if sha(item['source'])!=item['sha256'] or sha(item['copy'])!=item['sha256']:
                raise ValueError('Source/copied GRT checkpoint changed during routing')
        manifest.update(success=True,route_drc_clean=metrics['DRC']==0,
            final_macros_match_audited_grt=True,final_def_sha256=sha(final),
            canonical_metrics_sha256=sha(work/'metrics.json'),
            metric_evidence_sha256=sha(work/'metric_canonicalization.json'),
            drc_native_final_count=int(counts[-1]),drc_iteration_counts=list(map(int,counts)),
            drc_exported_type_counts=dict(exported),drc_exported_total=sum(exported.values()),
            all_violation_locations_exported=sum(exported.values())==metrics['DRC'],
            drc_note='Full count is verified against native route and JSON. A shorter text export does not reduce DRC; no export-cap assumption is used.')
        print(json.dumps(metrics,indent=2),flush=True)
    except BaseException as error:
        manifest['failure']=repr(error)
        raise
    finally:
        manifest['time_end']=time.time();save()


if __name__=='__main__':
    main()
