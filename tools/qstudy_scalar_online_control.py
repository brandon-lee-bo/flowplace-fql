"""Fixed four-arm scalar-online control: metadata, stage dispatch and recovery.

Only ``manifest`` is login-safe (stdlib metadata). All source loading and
scientific validation happen in bounded Slurm workers through frozen APIs.
No method/seed/sample selection, new training math, or changed physical flow.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VERSION = 'qstudy_scalar_online_control_v1'
SEEDS = (0, 1, 2)
ARMS = ('bc', 'updated_q10', 'frozen_q10', 'updated_q30')
FQL_SOURCE = Path('/scratch/bohanli/flowplace_graph/qstudy_rule_reward_actors_v1/rule_grt_wl')
BC_SOURCE = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1/actors_v1/bc_frozen')
PHASES = ('prepare0', 'export0', 'native0')
FUTURE_PHASES = ('train1', 'gate1', 'prepare1', 'export1', 'native1', 'train2',
                 'prepare2', 'export2', 'native2', 'route_prepare', 'route')
LIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
        'REQUEUED', 'RESIZING', 'STAGE_OUT'}
RETRY = {'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL'}
EXCLUDE = 'natural-selection,enterprise,blue-space'
PROTOCOL = 'docs/qstudy_scalar_online_control_protocol.md'
HELPERS = ('tools/qstudy_scalar_online_control.py',
           'tools/qstudy_scalar_online_control.slurm',
           'tools/qstudy_scalar_online_control_worker.slurm')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def bound(path, digest=None):
    path = Path(path).resolve()
    actual = sha(path)
    if digest is not None and actual != digest:
        raise ValueError('Changed bound artifact: ' + str(path))
    return dict(path=str(path), sha256=actual)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp_' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable artifact changed: ' + str(path))
    else:
        atomic(path, value)


def settings(arm):
    return dict(total_updates=600, batch_size=16, alpha=30. if arm=='updated_q30' else 10.,
        q_weight=0. if arm=='bc' else 1., flow_steps=10,
        update_critic=arm in ('updated_q10','updated_q30'), huber_delta=1., grad_clip=1.)


def streams(round_number):
    return {str(s):dict(**({'collect':dict(seed=81000000+100*s+1000*round_number,count=8)}
        if round_number<2 else {}), evaluation=dict(seed=81400000,count=8)) for s in SEEDS}


def experiment(output, key):
    return Path(output)/'physical'/key


def physical_arm(key):
    return 'soc_' + key


def export_module(key):
    if key == 'fql_t0': return 'fql.qstudy_scalar_online_export'
    if key == 'bc_t0': return 'fql.qstudy_scalar_online_bc_export'
    arm, stage = key.rsplit('_t', 1)
    suffix = ('bc_' if arm=='bc' else '') + ('child_export' if stage=='1' else 'second_export')
    return 'fql.qstudy_scalar_online_' + suffix


def train_module(arm, cycle):
    return 'fql.qstudy_scalar_online_' + ('bc_' if arm=='bc' else '') + ('train' if cycle==1 else 'second_train')


def parent_key(arm, cycle):
    return ('bc_t0' if arm=='bc' else 'fql_t0') if cycle==1 else arm+'_t1'


def source_metadata(path):
    """Small JSON only; original prepare will hash/load actual checkpoints."""
    result = dict(config=bound(path/'config.json'), parents={})
    for seed in SEEDS:
        folder = path/f'seed_{seed}/round_0'; done = read(folder/'completed.json')
        if (done.get('status')!='completed' or done.get('steps')!=1000 or
                done['binding'].get('seed')!=seed or done['binding'].get('round')!=0 or
                done['binding'].get('parent') is not None or done['binding'].get('replay') or
                done['binding']['config_sha256']!=result['config']['sha256'] or
                not re.fullmatch('[0-9a-f]{64}',done['checkpoint_sha256'])):
            raise ValueError('Original completed1000 offline prefix required')
        result['parents'][str(seed)] = dict(checkpoint=dict(path=str(folder/'checkpoint.pt'),
            sha256=done['checkpoint_sha256']), completed=bound(folder/'completed.json'),
            binding=bound(folder/'binding.json'))
    return result


def check_certificate(certificate, sources):
    if (certificate.get('schema')!='qstudy_root_control_checkpoint_audit_v1' or
            certificate.get('status')!='pass' or len(certificate.get('rows',[]))!=6 or
            len(certificate.get('pairs',[]))!=3):
        raise ValueError('Actual independent matched checkpoint certificate must pass')
    rows={(r['arm'],r['seed']):r for r in certificate['rows']}
    for kind,arm in (('bc','bc'),('fql','rule_grt_wl')):
        for seed in SEEDS:
            original=sources[kind]['parents'][str(seed)];row=rows[(arm,seed)]
            for key,field in (('checkpoint','checkpoint'),('completion','completed'),('binding_artifact','binding')):
                if any(row[key][k]!=original[field][k] for k in ('path','sha256')):
                    raise ValueError('Certificate is for a different actual source')
            if (row['training_binding']['config_sha256']!=sources[kind]['config']['sha256'] or
                    row['actor_adam_steps']!=[1000.] or not row['teacher_identical_to_common'] or
                    not row['critic_identical_to_own_refit'] or not row['frozen_optimizer_states_empty']):
                raise ValueError('Certificate did not establish the required frozen source')
    for pair in certificate['pairs']:
        if not all(pair.get(k) is True for k in ('same_initial_actor_and_teacher_declared_by_pinned_loader',
                'teacher_final_tensor_equal_to_same_common','actor_fixed_buffers_equal',
                'actor_adam_configuration_and_steps_equal','all_saved_rng_states_equal','first_distillation_exactly_equal')):
            raise ValueError('Matched prefix certificate has an unproved control condition')


def noise_history():
    """Scoped closed-study/smoke declarations, not an unbounded filesystem scan."""
    home=Path('/fact_home/bohanli/flowplace_fql_recovery')
    report=bound(home/'development_noise_inventory_v1/snapshot_v1/report.json',
        '2934f9cf99781c8d8534845f26a93dd70ceca975aefcc4edd0c7cdc52611b0d2')
    old=read(report['path'])
    if old['status']!='completed' or old['integrity_errors']:
        raise ValueError('Historical actual-noise evidence is incomplete')
    seeds=set();hashes=set();bindings={report['path']:report}
    def walk(value,key=''):
        if isinstance(value,dict):
            if isinstance(value.get('seed'),int) and key in ('collect','evaluation','evaluation_stream','collection_stream'):
                seeds.add(value['seed'])
            for name,item in value.items():
                if name in ('noise_seed','evaluation_noise_seed','collection_noise_seed') and isinstance(item,int):seeds.add(item)
                if 'noise' in name and isinstance(item,str) and re.fullmatch('[0-9a-f]{64}',item):hashes.add(item)
                walk(item,name)
        elif isinstance(value,list):
            for item in value:walk(item,key)
    walk(old)
    # All later scalar integration declarations are small JSON metadata, not
    # tensors or saved noise arrays. The list itself is frozen in the manifest.
    paths=set(p for folder in home.glob('qstudy_scalar_online*')
              for p in folder.rglob('*.json'))
    # Include the subsequent reward/flow studies' explicit producer declarations;
    # their evaluation uses the already inspected 4600000 development pool.
    declared_roots=[Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')/name for name in
        ('actors_v1','bounded_actors_v1','source_actors_v1','flow_eval_v1','public_flow_eval_v1')]
    declared_roots += [Path('/scratch/bohanli/flowplace_graph')/name for name in
        ('qstudy_rule_reward_actors_v1','qstudy_final_feedback_actors_v1',
         'qstudy_quality_weighted_eval_v1')]
    for folder in declared_roots:
        if folder.exists():
            paths.update(folder.glob('config.json'))
            paths.update(folder.glob('*/config.json'))
            paths.update(folder.glob('declaration.json'))
            paths.update(folder.glob('*/declaration.json'))
    for path in sorted(paths):
        item=bound(path);bindings[item['path']]=item;walk(read(path))
    proposed={81400000,*(81000000+100*s+1000*r for s in SEEDS for r in (0,1))}
    if seeds&proposed:raise ValueError('Proposed collection/evaluation seed was already declared')
    return dict(scope='sealed old 43-manifest actual-noise inventory plus listed later scalar integration JSON and reward/flow producer declarations; not a universal RNG disjointness proof',
        bindings=list(bindings.values()),observed_noise_seeds=sorted(seeds),
        observed_noise_sha256=sorted(hashes),proposed_stream_seeds=sorted(proposed),
        declared_seed_collisions=[],actual_new_array_collision_check='required after export, before native0')


def source_names():
    names = set(HELPERS+(PROTOCOL,))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'fql').glob('qstudy_scalar_online*.py'))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'fql').glob('qstudy_scalar_online*.slurm'))
    names.update(('fql/qstudy_backend.py','fql/qstudy_controller.py','fql/qstudy_backend_batch.slurm',
        'fql/graph_o2o_controller.py','fql/graph_o2o_backend.py','fql/graph_o2o_models.py',
        'fql/graph_o2o_geometry.py','fql/graph_o2o_archive.py','fql/graph_o2o_full_route.py',
        'fql/graph_o2o_full_route.slurm','fql/graph_o2o_finish.py','fql/graph_o2o_finish.sh',
        'fql/graph_o2o_final_evidence.py','fql/qstudy_final_route.py','baseline/api.py',
        'backend/scripts/export_macro_def.py','analysis/placement_metrics.py'))
    names.update(str(p.relative_to(ROOT)) for p in
        (ROOT/'upstream/flowplace/flow_matching').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml'))
    names.add('upstream/flowplace/benchmark_utils/openroad_utils.py')
    for cfg in (read(FQL_SOURCE/'config.json'),read(BC_SOURCE/'config.json')):
        for name in ('rule_sources','source_sha256'):
            names.update(cfg.get(name,{}))
    return sorted(names)


def topology(output):
    """All physical identities are fixed before any measured continuation."""
    grt, drt = [], []
    keys = ['fql_t0','bc_t0'] + [a+'_t1' for a in ARMS] + [a+'_t2' for a in ARMS]
    for key in keys:
        round_number = int(key[-1]); exp = experiment(output,key); arm = physical_arm(key)
        for seed in SEEDS:
            for role in ('collect','evaluation') if round_number<2 else ('evaluation',):
                for index in range(8):
                    sample = exp/f'seed_{seed}/round_{round_number}/requests'/arm/role/str(index)
                    row = dict(key=f'{key}_s{seed}_{role}_{index}', physical_key=key,
                        experiment=str(exp), arm=arm, seed=seed, round=round_number,
                        role=role, index=index, sample_dir=str(sample),
                        noise_stream_seed=streams(round_number)[str(seed)][role]['seed'],
                        noise_draw_index=index,
                        training_feedback_eligible=role=='collect')
                    grt.append(row)
                    if round_number in (0,2) and role=='evaluation' and index<4:
                        routekey=f'{key}_s{seed}_n{index}'
                        drt.append(dict(key=routekey, group=key, method=key, arm=arm,
                            seed=seed, round=round_number, noise_index=index,
                            source_outcome=str(sample/'outcome.json'),
                            output=str(Path(output)/'route/full_route'/routekey),
                            run_id='socdrt1_'+routekey))
    if len(grt)!=384 or len(drt)!=72 or len({v['sample_dir'] for v in grt})!=384:
        raise ValueError('Fixed unique384/72 topology changed')
    return grt,drt


def make_manifest(args):
    output=Path(args.output).resolve()
    if not output.is_absolute() or output==ROOT or ROOT in output.parents:
        raise ValueError('Independent canonical output required')
    admission=bound(args.admission,args.admission_sha256); certificate=read(admission['path'])
    sources=dict(fql=source_metadata(FQL_SOURCE),bc=source_metadata(BC_SOURCE))
    check_certificate(certificate,sources)
    grt,drt=topology(output)
    value=dict(version=VERSION,output=str(output),source_metadata=sources,admission=admission,
        source_checkpoint_validation='declarations here; actual hashes/tensors are revalidated by original prepares',noise_history=noise_history(),
        arms={a:settings(a) for a in ARMS},seeds=list(SEEDS),cycles=[1,2],
        streams={str(r):streams(r) for r in (0,1,2)}, reserved_fresh_noise_seed=85000000,
        physical_grt_requests=grt,physical_drt_requests=drt,requested_grt=384,requested_drt=72,
        common_fql_T0_C0=True,BC_has_own_T0_C0=True,teacher_updates=0,
        root_actor_steps=1000,actor_steps_per_cycle=600,final_actor_steps=2200,
        implemented_phases=list(PHASES),future_phases=list(FUTURE_PHASES),
        diagnostic_interval=100,max_generations=3,
        stage_resources={phase:dict(options=resources(phase)[0],concurrency=resources(phase)[1]) for phase in PHASES},
        concurrency=dict(gpu=2,grt=4,drt=2,cpu=2),
        physical=dict(threads=8,route_end_iteration=20,storage='scratch'),
        source_sha256={name:sha(ROOT/name) for name in source_names()},
        eval_semantics='development only; R2 is the fixed primary endpoint; R0/R1 also reported; all eight fixed actual noises, never replayed; no Q/sample selection',
        gate1='alpha10/30 T1 critic+critic Adam+Q-generator exact equality per seed before child export',
        reporter_limit='fixed stage/native inventory; scientific multi-seed aggregate is a separate saved-only report')
    output.mkdir(parents=True,exist_ok=True);(output/'logs').mkdir(exist_ok=True)
    with (output/'manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,digest in value['source_sha256'].items():
            snapshot=output/'sources'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not snapshot.exists():shutil.copyfile(ROOT/name,snapshot)
            bound(snapshot,digest)
        immutable(output/'manifest.json',value)
    print(json.dumps(dict(manifest=bound(output/'manifest.json'),grt=384,drt=72,submitted_jobs=0)))


def checked_manifest(path,digest):
    value=read(bound(path,digest)['path'])
    output=Path(value['output'])
    if (Path(path).resolve()!=output/'manifest.json' or value['version']!=VERSION or
            value['arms']!={a:settings(a) for a in ARMS} or value['streams']!={str(r):streams(r) for r in (0,1,2)} or
            value['implemented_phases']!=list(PHASES) or value['future_phases']!=list(FUTURE_PHASES) or value['physical_grt_requests']!=topology(output)[0] or
            value['stage_resources']!={phase:dict(options=resources(phase)[0],concurrency=resources(phase)[1]) for phase in PHASES} or
            value['physical_drt_requests']!=topology(output)[1] or
            value['source_metadata']!=dict(fql=source_metadata(FQL_SOURCE),bc=source_metadata(BC_SOURCE))):
        raise ValueError('Fixed source/topology/budgets changed')
    bound(value['admission']['path'],value['admission']['sha256'])
    check_certificate(read(value['admission']['path']),value['source_metadata'])
    for item in value['noise_history']['bindings']:bound(item['path'],item['sha256'])
    for name,expected in value['source_sha256'].items():
        bound(ROOT/name,expected);bound(output/'sources'/name,expected)
    return value


def config_binding(output,key):
    exp=experiment(output,key)
    return dict(experiment=str(exp),config_sha256=sha(exp/'config.json'),
                declaration_sha256=sha(exp/'declaration.json'))


def export_request(plan,key):
    output=Path(plan['output']);stage=int(key[-1]);module=importlib.import_module(export_module(key))
    value=dict(version=module.VERSION,arm=physical_arm(key),run_prefix='soc1'+key.replace('_',''),
        streams=plan['streams'][str(stage)],evaluation_purpose='development_validation',
        backend_storage='scratch',output=str(experiment(output,key)))
    if stage==0:
        def cps(kind):return {s:v['checkpoint'] for s,v in plan['source_metadata'][kind]['parents'].items()}
        if key=='fql_t0':value.update(source_kind='rule',source_experiment=str(FQL_SOURCE),
            source_config_sha256=plan['source_metadata']['fql']['config']['sha256'],parent_checkpoints=cps('fql'))
        else:value.update(scalar_kind='rule',bc_experiment=str(BC_SOURCE),
            bc_config_sha256=plan['source_metadata']['bc']['config']['sha256'],bc_parent_checkpoints=cps('bc'),
            scalar_experiment=str(FQL_SOURCE),scalar_config_sha256=plan['source_metadata']['fql']['config']['sha256'],
            scalar_parent_checkpoints=cps('fql'))
    else:
        raise ValueError('Only reviewed T0 preparation is implemented in this thin stage driver')
    module.request_settings(value)
    path=output/'requests'/('export_'+key+'.json');immutable(path,value)
    return bound(path)


def run_module(name,args):
    subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m',name,*map(str,args)],cwd=ROOT,check=True)


def checked_export(plan,key,seed):
    from fql.graph_o2o_backend import load_requests
    module=importlib.import_module(export_module(key));cfgbind=config_binding(plan['output'],key)
    cfg,decl,physical=module.settings(cfgbind['experiment'],cfgbind['config_sha256'])
    rows=module.verify_export(Path(cfgbind['experiment']),seed,cfg,decl,physical)
    saved,tasks=load_requests(cfgbind['experiment'],seed,0)
    if saved!=rows:raise ValueError('Verified export and saved request inventory differ')
    return cfg,decl,physical,rows,tasks


def check_noise0(plan):
    """Metadata reduction of arrays already verified by each original exporter."""
    output=Path(plan['output']);hist=set(plan['noise_history']['observed_noise_sha256'])
    common=None;records=[];paired_collect={}
    for key in ('fql_t0','bc_t0'):
        for seed in SEEDS:
            receipt=checked_receipt(plan,'export0',phase_tasks(plan,'export0').index(dict(kind='export',physical_key=key,seed=seed)))
            tasks=Path(receipt['results'][0]['path'])
            rows=[json.loads(line) for line in tasks.read_text().splitlines()]
            evaluation=sorted((r for r in rows if r['role']=='evaluation'),key=lambda r:r['index'])
            collect=sorted((r for r in rows if r['role']=='collect'),key=lambda r:r['index'])
            if len(evaluation)!=8 or len(collect)!=8:raise ValueError('Actual T0 noise denominator changed')
            actual=[r['noise_sha256'] for r in evaluation]
            actual_collect=[r['noise_sha256'] for r in collect]
            if len(set(actual))!=8 or len(set(actual_collect))!=8:raise ValueError('Duplicate actual noise within a stream')
            if common is not None and actual!=common:raise ValueError('Shared actual evaluation arrays differ')
            common=actual
            if seed in paired_collect and actual_collect!=paired_collect[seed]:
                raise ValueError('Matched FQL/BC C0 actual input noises differ')
            paired_collect[seed]=actual_collect
            if (set(actual)|{r['noise_sha256'] for r in collect})&hist:
                raise ValueError('New actual CUDA noise repeats inspected historical noise')
            if set(actual)&{r['noise_sha256'] for r in collect}:raise ValueError('Collection/evaluation actual noise collides')
            records.append(dict(physical_key=key,seed=seed,tasks=bound(tasks),
                evaluation=[dict(index=r['index'],noise_seed=r['noise_seed'],noise_sha256=r['noise_sha256']) for r in evaluation],
                collect=[dict(index=r['index'],noise_seed=r['noise_seed'],noise_sha256=r['noise_sha256']) for r in collect]))
    if len(set().union(*(set(paired_collect[seed]) for seed in SEEDS)))!=8*len(SEEDS):
        raise ValueError('Different actor seeds reuse collection actual-noise arrays')
    artifact=output/'gates/T0_actual_noise.json'
    immutable(artifact,dict(status='pass',records=records,historical_actual_collisions=0,
        actual_array_regeneration_calls=0,shared_evaluation_noise_sha256=common,
        paired_FQL_BC_C0=True,cross_seed_collection_actual_collisions=0))
    return bound(artifact)


def inspect_native(plan,row):
    from argparse import Namespace
    from fql.qstudy_backend import context,checked_request,checked_existing
    from fql.graph_o2o_backend import load_requests
    cfg,verification,physical=context(row['experiment'])
    rows,tasks=load_requests(row['experiment'],row['seed'],row['round'])
    matches=[r for r in rows if r['role']==row['role'] and r['index']==row['index']]
    if len(matches)!=1:raise ValueError('Missing/duplicate native role/index')
    request=matches[0];sample=checked_request(request,Namespace(seed=row['seed'],round=row['round']),cfg,physical)
    if (str(sample.resolve())!=row['sample_dir'] or request['arm']!=row['arm'] or
            request['noise_seed']!=row['noise_stream_seed'] or request['index']!=row['noise_draw_index'] or
            request['training_feedback_eligible']!=(row['role']=='collect')):
        raise ValueError('Native task namespace changed')
    outcome=read(sample/'outcome.json')
    if outcome.get('physical_run') and read(Path(outcome['physical_run'])/'manifest.json').get('success'):
        if not (Path(outcome['physical_run'])/'backend_macro_audit.json').exists():
            raise ValueError('Saved native geometry audit missing; refusing implicit EDA')
    checked_existing(outcome,request,sample,tasks,physical,cfg,verification)
    if outcome['status'] not in ('success','physical_failure','geometry_failure'):
        raise ValueError('Unresolved/unknown native outcome cannot close request')
    return request,outcome,tasks,sample,physical


def phase_tasks(plan,phase):
    if phase=='prepare0':return [dict(kind='prepare',physical_key=k) for k in ('fql_t0','bc_t0')]
    if phase=='export0':return [dict(kind='export',physical_key=k,seed=s) for k in ('fql_t0','bc_t0') for s in SEEDS]
    if phase=='native0':return [dict(kind='native',row=r) for r in plan['physical_grt_requests'] if r['round']==0]
    raise ValueError('Later phases are prospectively declared but not implemented by this driver')


def task_run(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Worker requires bounded Slurm')
    plan=checked_manifest(args.manifest,args.manifest_sha256);phase=args.phase
    if phase not in PHASES:raise ValueError('Unknown phase')
    index=args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    tasks=phase_tasks(plan,phase)
    if not 0<=index<len(tasks):raise IndexError('Task outside fixed phase')
    task=tasks[index];output=Path(plan['output']);result=[]
    receipt=output/'receipts'/phase/(str(index)+'.json')
    if receipt.exists():
        checked_receipt(plan,phase,index);return
    kind=task['kind']
    if kind=='prepare':
        request=export_request(plan,task['physical_key'])
        run_module(export_module(task['physical_key']),['prepare','--request',request['path'],
            '--request-sha256',request['sha256'],'--output',experiment(output,task['physical_key'])])
        exp=experiment(output,task['physical_key']);result=[bound(exp/'config.json'),bound(exp/'declaration.json'),request]
    elif kind=='export':
        cfg=config_binding(output,task['physical_key'])
        run_module(export_module(task['physical_key']),['export','--experiment',cfg['experiment'],
            '--config-sha256',cfg['config_sha256'],'--seed',task['seed']])
        _,_,_,rows,path=checked_export(plan,task['physical_key'],task['seed'])
        result=[bound(path),bound(path.with_name('completed.json'))]
    elif kind=='native':
        row=task['row'];cfg,decl,physical,rows,path=checked_export(plan,row['physical_key'],row['seed'])
        positions=[i for i,r in enumerate(rows) if r['role']==row['role'] and r['index']==row['index']]
        if len(positions)!=1:raise ValueError('Native task role/index absent or duplicated')
        run_module('fql.qstudy_backend',['--experiment',row['experiment'],'--seed',row['seed'],
            '--round',row['round'],'--index',positions[0],'--storage','scratch'])
        _,_,_,sample,_=inspect_native(plan,row);result=[bound(sample/'outcome.json')]
    else:raise ValueError('Unknown immutable task kind')
    immutable(receipt,dict(version=VERSION,manifest=bound(args.manifest,args.manifest_sha256),
        phase=phase,index=index,task=task,results=result,slurm_job_id=os.environ['SLURM_JOB_ID']))


def checked_receipt(plan,phase,index):
    path=Path(plan['output'])/'receipts'/phase/(str(index)+'.json');value=read(path)
    if (value['version']!=VERSION or value['manifest']!=bound(Path(plan['output'])/'manifest.json') or
            value['phase']!=phase or value['index']!=index or value['task']!=phase_tasks(plan,phase)[index]):
        raise ValueError('Completed task receipt changed identity')
    for item in value['results']:bound(item['path'],item['sha256'])
    return value


def scheduler_states(job):
    queued=subprocess.run(['squeue','-j',str(job),'-h','-o','%i|%T'],text=True,capture_output=True,check=True)
    live=[line.split('|')[1].strip() for line in queued.stdout.splitlines() if '|' in line]
    if any(v in LIVE for v in live):return live,dict(squeue=queued.stdout,child_states={})
    accounting=subprocess.run(['sacct','-j',str(job),'-X','--duplicates','-n','-P',
        '-o','JobID%64,JobIDRaw,State,ExitCode,JobName%200'],text=True,capture_output=True,check=True)
    values=[];children={}
    for line in accounting.stdout.splitlines():
        fields=line.split('|')
        if len(fields)<5:continue
        state=fields[2].split()[0].rstrip('+');values.append(state)
        match=re.fullmatch(re.escape(str(job))+r'_(\d+)',fields[0])
        if match:children.setdefault(match[1],[]).append(state)
    return values or ['ACCOUNTING_PENDING'],dict(squeue=queued.stdout,sacct=accounting.stdout,child_states=children)


def resources(phase):
    if phase=='export0':
        return ['--partition=gpu-scavenger','--qos=gpu-scavenger','--gres=gpu:nvidia_l20:1',
            '--cpus-per-task=2','--mem=20G','--time=00:15:00'],2
    if phase=='native0':
        return ['--partition=cpu-scavenger','--qos=cpu-scavenger','--cpus-per-task=8',
            '--mem=24G','--time=02:00:00'],4
    if phase=='prepare0':
        return ['--partition=cpu-scavenger','--qos=cpu-scavenger','--cpus-per-task=2',
            '--mem=8G','--time=00:10:00'],2
    raise ValueError('Only the three reviewed stage0 resource contracts exist')


def adopt(name):
    """Exact-name lookup recovers an sbatch/receipt interruption, not a range scan."""
    queue=subprocess.run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j'],
        text=True,capture_output=True,check=True)
    found={line.split('|')[0].split('_')[0] for line in queue.stdout.splitlines()
        if '|' in line and line.split('|')[1]==name}
    accounting=subprocess.run(['sacct','-u',os.environ['USER'],'--name',name,'-S','2026-10-19',
        '-X','--duplicates','-n','-P','-o','JobID%64,JobIDRaw,JobName%200'],text=True,capture_output=True,check=True)
    for line in accounting.stdout.splitlines():
        fields=line.split('|')
        if len(fields)>=3 and fields[2]==name:found.add(fields[0].split('_')[0])
    if len(found)>1 or any(not v.isdigit() for v in found):raise ValueError('Ambiguous unique-name scheduler adoption')
    return next(iter(found)) if found else None,dict(squeue=queue.stdout,sacct=accounting.stdout)


def stage_state(plan,phase):
    output=Path(plan['output']);path=output/'submissions'/phase/'state.json'
    state=read(path) if path.exists() else dict(version=VERSION,
        manifest=bound(output/'manifest.json'),phase=phase,attempts=[])
    if state['manifest']!=bound(output/'manifest.json') or state['phase']!=phase:
        raise ValueError('Stage journal belongs to different manifest')
    return state,path


def missing_indices(plan,phase):
    result=[]
    for i in range(len(phase_tasks(plan,phase))):
        if (Path(plan['output'])/'receipts'/phase/(str(i)+'.json')).exists():checked_receipt(plan,phase,i)
        else:result.append(i)
    return result


def stage_submit(args):
    plan=checked_manifest(args.manifest,args.manifest_sha256);output=Path(plan['output'])
    phase=args.phase;path=output/'submissions'/phase/'state.json';path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state,path=stage_state(plan,phase)
        for prior in PHASES[:PHASES.index(phase)]:
            if missing_indices(plan,prior):raise ValueError('Previous phase has unfinished tasks: '+prior)
            priorstate,_=stage_state(plan,prior)
            if not priorstate['attempts'] or not priorstate['attempts'][-1].get('job'):
                raise ValueError('Previous phase has no bound allocated job')
            observed,_=scheduler_states(priorstate['attempts'][-1]['job'])
            if any(v in LIVE or v=='ACCOUNTING_PENDING' for v in observed):
                raise ValueError('Previous phase still live/accounting pending: '+prior)
        if phase=='native0':check_noise0(plan)
        missing=missing_indices(plan,phase)
        if not missing:
            state['status']='completed';atomic(path,state);print(json.dumps(dict(phase=phase,status='completed',submitted=0)));return
        if state['attempts']:
            last=state['attempts'][-1]
            if not last.get('job'):
                job,evidence=adopt(last['name']);last['adoption_evidence']=evidence
                if not job:
                    atomic(path,state);raise RuntimeError('Unresolved submission intent; no duplicate submission allowed')
                last['job']=job
            observed,evidence=scheduler_states(last['job']);last.update(states=observed,last_observation=evidence)
            atomic(path,state)
            if any(v in LIVE or v=='ACCOUNTING_PENDING' for v in observed):
                print(json.dumps(dict(phase=phase,status='live_or_accounting_pending',job=last['job'],missing=missing)));return
            missing=missing_indices(plan,phase)
            if not missing:
                state['status']='completed';atomic(path,state);return
            if any(v not in RETRY|{'COMPLETED'} for v in observed):
                raise RuntimeError('Terminal failure/missing receipt requires diagnosis: '+repr(observed))
            for index in missing:
                child=evidence['child_states'].get(str(index),[])
                if not child:
                    print(json.dumps(dict(phase=phase,status='child_accounting_pending',index=index)));return
                if any(v not in RETRY for v in child):
                    raise RuntimeError('Missing task is not proved infrastructure-preempted: '+str(index)+' '+repr(child))
        if len(state['attempts'])>=plan['max_generations']:raise RuntimeError('Bounded infrastructure recovery exhausted')
        generation=len(state['attempts']);name='soc_'+args.manifest_sha256[:8]+'_'+phase+'_'+str(generation)
        options,concurrency=resources(phase)
        command=['sbatch','--parsable','--job-name='+name,*options,'--exclude='+EXCLUDE,
            '--array='+','.join(map(str,missing))+'%'+str(concurrency),
            '--output='+str(output/'logs'/('%A_%a_'+phase+'.log')),
            str(ROOT/HELPERS[2]),'--manifest',str(args.manifest),'--manifest-sha256',args.manifest_sha256,'--phase',phase]
        attempt=dict(name=name,indices=missing,command=command,job=None)
        state['attempts'].append(attempt);atomic(path,state)
        env=dict(os.environ);env.pop('SLURM_RESTART_COUNT',None)
        submitted=subprocess.run(command,text=True,capture_output=True,env=env)
        attempt.update(returncode=submitted.returncode,stdout=submitted.stdout,stderr=submitted.stderr)
        if submitted.returncode:atomic(path,state);raise RuntimeError('sbatch rejected; no allocation inferred')
        job=submitted.stdout.strip().split(';')[0]
        if not job.isdigit():atomic(path,state);raise ValueError('Unknown sbatch response; intent retained')
        attempt['job']=job;atomic(path,state)
        print(json.dumps(dict(phase=phase,job=job,indices=missing,concurrency=concurrency)))


def status(args):
    plan=checked_manifest(args.manifest,args.manifest_sha256);rows=[]
    for phase in PHASES:
        state,path=stage_state(plan,phase);observed=None
        if state['attempts'] and state['attempts'][-1].get('job'):
            observed,_=scheduler_states(state['attempts'][-1]['job'])
        rows.append(dict(phase=phase,missing=missing_indices(plan,phase),states=observed,journal=str(path)))
    print(json.dumps(dict(status='metadata_only',stages=rows,future_phases_not_implemented=list(FUTURE_PHASES))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('manifest');p.add_argument('--output',required=True);p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    for name in ('submit','status','task'):
        p=sub.add_parser(name);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
        if name in ('submit','task'):p.add_argument('--phase',choices=PHASES,required=True)
        if name=='task':p.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command=='manifest':make_manifest(args)
    elif args.command=='submit':stage_submit(args)
    elif args.command=='status':status(args)
    else:task_run(args)


if __name__=='__main__':main()
