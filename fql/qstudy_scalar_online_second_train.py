"""Explicit second measured scalar continuation from a genuine first-cycle FQL.

C0 and C1 remain separately admitted under their real checkpoints/namespaces.
The original r0 supplies frozen TRAIN labels, teacher and normalizers, while
all trainable tensors, Adam and RNG continue from T1. This is not a generic
multi-cycle/BC runner and does not export, schedule physical work or select an
objective, budget, noise or checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from . import qstudy_scalar_online_child_export as child
from . import qstudy_scalar_online_export as original_export
from . import qstudy_scalar_online_train as first
from . import qstudy_scalar_online as core
from baseline.api import RUNTIME, load_problem
from .graph_o2o_archive import atomic_json
from .graph_o2o_models import GraphPolicy
from .qstudy_data import bind, sha
from .qstudy_screen import atomic_torch

ROOT=Path(__file__).resolve().parents[1]
VERSION='qstudy_scalar_online_second_train_v1'
SOURCE_NAMES=tuple(dict.fromkeys((*child.SOURCE_NAMES,
    'fql/qstudy_scalar_online_second_train.py','fql/qstudy_scalar_online_second_train.slurm')))
STOP=False


def read(path):return json.loads(Path(path).read_text())


def request_settings(value):
    required={'version','cycle','experiment','config_sha256','declaration_sha256','seed',
              'tasks_sha256','final_outcomes','settings','diagnostic_interval','output'}
    if set(value)!=required or value['version']!=VERSION or type(value['cycle']) is not int or value['cycle']!=2:
        raise ValueError('Only explicit second-cycle requests from the first FQL child provider are supported')
    translated={k:v for k,v in value.items() if k!='cycle'};translated['version']=first.VERSION
    first.request_settings(translated)
    for key in ('config_sha256','declaration_sha256','tasks_sha256'):
        if not isinstance(value[key],str) or not re.fullmatch('[0-9a-f]{64}',value[key]):
            raise ValueError('Externally pinned child export digests required')
    for identity in value['final_outcomes'].values():
        if (str(Path(identity['path']).resolve())!=identity['path'] or
            not re.fullmatch('[0-9a-f]{64}',identity['sha256'])):
            raise ValueError('Canonical externally pinned final evidence required')
    validate_settings(value['settings'])
    return value


def validate_settings(settings):
    required={'total_updates','batch_size','alpha','q_weight','flow_steps','update_critic','huber_delta','grad_clip'}
    if (set(settings)!=required or type(settings['update_critic']) is not bool or
        type(settings['total_updates']) is not int or settings['total_updates']<1 or
        type(settings['batch_size']) is not int or settings['batch_size']<2 or settings['batch_size']%2 or
        type(settings['flow_steps']) is not int or settings['flow_steps']<1 or
        any(type(settings[k]) not in (int,float) or not math.isfinite(settings[k]) for k in ('alpha','q_weight','huber_delta','grad_clip')) or
        min(settings['alpha'],settings['q_weight'])<0 or min(settings['huber_delta'],settings['grad_clip'])<=0):
        raise ValueError('Explicit finite second-segment update settings required')
    return settings


def adam_steps(optimizer,*,required=False):
    state=optimizer['state']
    if not state:
        if required:raise ValueError('Actual actor Adam prefix is absent')
        return 0
    steps=[]
    for entry in state.values():
        value=float(entry['step'])
        if not math.isfinite(value) or value<0 or value!=int(value):raise ValueError('Invalid actual Adam step')
        steps.append(int(value))
    if len(set(steps))!=1:raise ValueError('Adam parameter step counters disagree')
    return steps[0]


def source_from_verified(cfg,verified,seed):
    """Keep the real T1 payload/binding intact; attach a separate truthful source."""
    identity=verified['identity'];root=verified['source'];payload=verified['payload']
    if (identity!=cfg['verified_children'][str(seed)] or root['schema']!=cfg['scalar_schema'] or
        root['payload']['binding'].get('seed')!=seed):
        raise ValueError('Child provider is not the exact declared first FQL continuation')
    child.validate_payload(payload,root,identity['continuation_binding'])
    for name in ('checkpoint','completed','runner_binding','learning_curve','training_request'):
        item=identity[name];bind(item['path'],item['sha256'])
    root_actor=adam_steps(root['payload']['optimizers']['actor'],required=True)
    root_q=adam_steps(root['payload']['optimizers']['critic'])
    first_updates=identity['continuation_binding']['settings']['total_updates']
    first_q=first_updates if identity['continuation_binding']['settings']['update_critic'] and identity['continuation_binding']['settings']['q_weight']>0 else 0
    if (root_actor!=root['payload']['step'] or
        adam_steps(payload['optimizers']['actor'],required=True)!=root_actor+first_updates or
        adam_steps(payload['optimizers']['critic'])!=root_q+first_q):
        raise ValueError('Actual root/T1 Adam steps differ from the admitted prefix')
    binding=dict(version=VERSION,provider=child.VERSION,seed=seed,source_cycle=1,schema=copy.deepcopy(root['schema']),
        checkpoint=copy.deepcopy(identity['checkpoint']),first_continuation=copy.deepcopy(identity),
        root_source=copy.deepcopy(root['binding']),
        prefix_counts=dict(root_actor_updates=root_actor,root_critic_updates=root_q,
            first_actor_updates=first_updates,first_critic_updates=first_q,
            actor_adam_steps=root_actor+first_updates,critic_adam_steps=root_q+first_q),
        state_identity='unmodified genuine completed T1 payload; not an offline round0 checkpoint')
    return dict(payload=payload,binding=binding,schema=root['schema'],root=root,refit=root['refit'])


def combine_replay(components,source,evaluation_noise_sha256):
    """Combine exactly C0+C1, keeping original role/round/config/checkpoint proofs."""
    if len(components)!=2:raise ValueError('Exactly the original C0 and first-child C1 are required')
    seed=source['binding']['seed'];excluded=set(evaluation_noise_sha256)
    if not excluded:raise ValueError('Both historical evaluation pools must remain excluded')
    roots=(source['binding']['root_source']['checkpoint'],source['binding']['checkpoint'])
    seen_requests=set();seen_noise=set();records=[];arrays=[];labels=[];masks=[];success=[];failure=[];parts=[]
    for round_index,part in enumerate(components):
        replay=part['replay'];binding=replay['binding'];cfg=part['config'];experiment=Path(part['experiment']).resolve()
        if (part['source_round']!=round_index or binding['source_round']!=round_index or binding['seed']!=seed or
            binding['checkpoint_sha256']!=roots[round_index]['sha256'] or binding['schema']!=source['schema'] or
            binding['config']!=bind(experiment/'config.json') or not replay.get('ready') or
            replay['utility_spec_sha256']!=source['schema']['utility_spec_sha256'] or
            binding['records']!=replay['records'] or len(replay['records'])!=binding['count'] or
            replay['derived_B'].shape!=(binding['count'],1) or replay['mask'].shape!=replay['derived_B'].shape or
            not bool(replay['mask'].all()) or not bool(torch.isfinite(replay['derived_B']).all()) or
            bool((replay['derived_B']<0).any()) or
            sorted(r['index'] for r in replay['records'])!=list(range(binding['count']))):
            raise ValueError('Accumulated replay changed round/seed/parent/schema/denominator or remains unresolved')
        bind(binding['tasks']['path'],binding['tasks']['sha256'])
        for record in replay['records']:
            request=record['request'];bind(request['path'],request['sha256']);row=read(request['path'])
            if (row.get('role')!='collect' or row.get('training_feedback_eligible') is not True or
                row.get('seed')!=seed or row.get('arm')!=cfg['arm'] or row.get('round')!=round_index or
                row.get('index')!=record['index'] or row.get('config_sha256')!=binding['config']['sha256'] or
                row.get('checkpoint_sha256')!=roots[round_index]['sha256'] or row.get('source_checkpoint')!=roots[round_index]['path'] or
                row.get('noise_seed')!=binding['collection_noise_seed'] or row.get('noise_sha256')!=record['noise_sha256'] or
                row.get('raw_action_sha256')!=record['raw_action_sha256'] or row.get('action_sha256')!=record['action_sha256'] or
                row['noise_sha256'] in excluded):
                raise ValueError('Cross-arm/seed/parent/evaluation record cannot enter cumulative own replay')
            request_id=(request['path'],request['sha256'])
            if request_id in seen_requests or request['path'] in {p for p,_ in seen_requests} or row['noise_sha256'] in seen_noise:
                raise ValueError('Repeated request or collect noise cannot receive duplicate replay weight')
            seen_requests.add(request_id);seen_noise.add(row['noise_sha256'])
            records.append(dict(record,source_round=round_index,source_arm=cfg['arm'],source_experiment=str(experiment)))
        arrays.append(replay['x']);labels.append(replay['derived_B']);masks.append(replay['mask'])
        success.append(replay['success']);failure.append(replay['failure']);parts.append(copy.deepcopy(binding))
    binding=dict(version=VERSION,seed=seed,schema=copy.deepcopy(source['schema']),components=parts,count=len(records),
        records=records,excluded_evaluation_noise_sha256=sorted(excluded),
        sampling='uniform over all admitted own C0+C1 rows; half offline and half cumulative online minibatch')
    return dict(x=torch.cat(arrays),derived_B=torch.cat(labels),mask=torch.cat(masks),success=torch.cat(success),
        failure=torch.cat(failure),ready=True,utility_spec_sha256=source['schema']['utility_spec_sha256'],records=records,binding=binding)


def second_binding(source,replay,settings):
    validate_settings(settings)
    if (source['binding'].get('provider')!=child.VERSION or source['binding'].get('source_cycle')!=1 or
        source['binding']['checkpoint']!=source['binding']['first_continuation']['checkpoint'] or
        not replay.get('ready') or replay['binding']['schema']!=source['schema'] or
        replay['binding']['seed']!=source['binding']['seed'] or len(replay['binding']['components'])!=2):
        raise ValueError('A verified genuine T1 parent and complete C0+C1 replay are required')
    return dict(version=VERSION,cycle=2,parent=copy.deepcopy(source['binding']),replay=copy.deepcopy(replay['binding']),
        settings=copy.deepcopy(settings),source_sha256=sha(core.__file__),root_anchor=copy.deepcopy(source['root']['binding']),
        limitations='explicit second FQL cycle only; no BC, generic later cycles, physical collection or export')


def cumulative_counts(source,settings,local_step):
    prefix=source['binding']['prefix_counts']
    qsteps=local_step if settings['update_critic'] and settings['q_weight']>0 else 0
    return dict(root_actor_updates=prefix['root_actor_updates'],first_actor_updates=prefix['first_actor_updates'],
        second_actor_updates=local_step,actor_adam_steps=prefix['actor_adam_steps']+local_step,
        root_critic_updates=prefix['root_critic_updates'],first_critic_updates=prefix['first_critic_updates'],
        second_critic_updates=qsteps,critic_adam_steps=prefix['critic_adam_steps']+qsteps,teacher_updates=0)


def audit_state(payload,source,binding):
    """Audit counters against actual Adam; segment-local core fields stay local."""
    step=payload['step'];parent=source['payload'];root=source['root']['payload'];settings=binding['settings']
    if (payload.get('version')!=core.VERSION or payload.get('binding')!=binding or type(step) is not int or
        not 0<=step<=settings['total_updates'] or payload.get('parent_steps')!=parent['step']):
        raise ValueError('Second segment state/binding/local counter changed')
    counts=cumulative_counts(source,settings,step)
    for name in ('flow',):core.equal_tree(payload[name],root[name])
    for name in ('actor','critic'):
        if set(payload[name])!=set(parent[name]):raise ValueError('Second model tensor inventory changed')
        for key,value in payload[name].items():
            old=parent[name][key]
            if torch.is_tensor(value):
                if value.shape!=old.shape or value.dtype!=old.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('Second model tensor shape/finite state changed')
            else:core.equal_tree(value,root[name][key])
    for key in ('fixed','movable'):core.equal_tree(payload['actor'][key],root['actor'][key])
    for key in dict(source['refit']['model'].named_buffers()):core.equal_tree(payload['critic'][key],root['critic'][key])
    child._optimizer(payload['optimizers']['actor'],parent['optimizers']['actor'],counts['actor_adam_steps'],'actor')
    child._optimizer(payload['optimizers']['flow'],parent['optimizers']['flow'],0,'flow',unchanged=True)
    child._optimizer(payload['optimizers']['critic'],parent['optimizers']['critic'],counts['critic_adam_steps'],'critic',
        unchanged=not counts['second_critic_updates'])
    if not counts['second_critic_updates']:core.equal_tree(payload['critic'],parent['critic'])
    child._rng_state(payload['rng'],parent['rng'],'cpu');torch.Generator().set_state(payload['rng'].cpu())
    if len(payload['cuda_rng'])!=len(parent['cuda_rng']):raise ValueError('CUDA RNG device inventory changed')
    for i,(state,old) in enumerate(zip(payload['cuda_rng'],parent['cuda_rng'])):child._rng_state(state,old,'cuda'+str(i))
    if set(payload['generators'])!=set(core.GENERATOR_KEYS):raise ValueError('Named RNG inventory changed')
    for name in core.GENERATOR_KEYS:child._rng_state(payload['generators'][name],parent['generators'][name],name)
    core.equal_tree(payload['generators']['flow'],parent['generators']['flow'])
    if not counts['second_critic_updates']:core.equal_tree(payload['generators']['q'],parent['generators']['q'])
    return counts


def load_inputs(request):
    experiment=Path(request['experiment']);seed=request['seed']
    cfg,decl,physical=child.settings(experiment,request['config_sha256'])
    bind(experiment/'declaration.json',request['declaration_sha256'])
    rows=child.verify_export(experiment,seed,cfg,decl,physical)
    verified=child.checked_parent(cfg['parent_training'][str(seed)],seed)
    source=source_from_verified(cfg,verified,seed)
    first_request=first.request_settings(read(cfg['parent_training'][str(seed)]['request']['path']))
    older_cfg=verified['cfg'];older_experiment=Path(first_request['experiment'])
    old_rows=original_export.verify_export(older_experiment,seed,older_cfg,verified['declaration'],verified['physical'])
    cache_path=RUNTIME/'benchmark_cache'/(cfg['design']+'.pt');bind(cache_path,cfg['cache_sha256'])
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    excluded=sorted({r['noise_sha256'] for r in (*old_rows,*rows) if r['role']=='evaluation'})
    components=[]
    for round_index,physical_exp,pcfg,prows,previous in (
        (0,older_experiment,older_cfg,old_rows,first_request),(1,experiment,cfg,rows,request)):
        parent=source['root']['binding']['checkpoint'] if round_index==0 else source['binding']['checkpoint']
        collect=pcfg['streams'][str(seed)]['collect']
        if source['schema']['kind']=='rule' and previous['final_outcomes']:
            raise ValueError('Rule objective cannot claim final-route admission')
        replay=core.admit_collect(physical_exp,seed=seed,source_round=round_index,count=collect['count'],
            tasks_sha256=previous['tasks_sha256'],config_sha256=previous['config_sha256'],checkpoint_sha256=parent['sha256'],
            collection_noise_seed=collect['seed'],evaluation_noise_sha256=excluded,value=source['schema'],cache=cache,
            final_outcomes={int(k):v for k,v in previous['final_outcomes'].items()})
        components.append(dict(replay=replay,experiment=str(physical_exp),config=pcfg,source_round=round_index))
    replay=combine_replay(components,source,excluded)
    binding=second_binding(source,replay,request['settings'])
    # No new output may replace or nest any real ancestor/physical namespace.
    output=Path(request['output'])
    for old in (experiment,older_experiment,Path(first_request['output']),Path(older_cfg['source_experiment'])):
        if output==old or output in old.parents or old in output.parents:
            raise ValueError('Second training output overlaps a real parent/source namespace')
    return cfg,decl,source,replay,binding


def save_state(folder,continuation,source,runner_binding,history):
    payload=continuation.state();counts=audit_state(payload,source,continuation.binding)
    payload.update(runner_binding=copy.deepcopy(runner_binding),history=copy.deepcopy(history),cumulative_steps=counts,
        step_semantics='step and parent_steps are segment-local core fields; cumulative_steps contains actual Adam totals')
    atomic_torch(folder/'resume.pt',payload);atomic_json(folder/'learning_curve.json',history)
    return payload


def checked_resume(folder,runner_binding,binding,source):
    payload=first.checked_resume(folder,runner_binding,binding)
    if payload is not None:
        actual=audit_state(payload,source,binding)
        if payload.get('cumulative_steps')!=actual:raise ValueError('Saved cumulative step metadata differs from actual Adam')
        for row in payload['history']:
            if row.get('cycle')!=2 or row.get('cumulative_steps')!=cumulative_counts(source,binding['settings'],row['step']):
                raise ValueError('Second-segment history/cumulative counters changed')
    return payload


def completion_record(folder,binding,runner_binding,payload):
    return dict(status='completed',version=VERSION,cycle=2,steps=payload['step'],
        core_parent_segment_steps=payload['parent_steps'],cumulative_steps=payload['cumulative_steps'],
        checkpoint=bind(folder/'checkpoint.pt'),binding=runner_binding,continuation_binding=binding,
        parent_checkpoint=binding['parent']['checkpoint'],root_checkpoint=binding['root_anchor']['checkpoint'],
        replay_counts=[v['count'] for v in binding['replay']['components']],q_units=core.Q_UNITS,
        backend_calls=0,scope='second measured FQL cycle; no BC, generic later cycle or child export')


def checked_completion(folder,payload,runner_binding,binding):
    if (payload is None or payload['step']!=binding['settings']['total_updates'] or
        read(folder/'completed.json')!=completion_record(folder,binding,runner_binding,payload)):
        raise ValueError('Second continuation completion differs from its full checkpoint')


def run_loop(folder,continuation,source,offline,replay,runner_binding,history=(),*,diagnostic_interval):
    folder.mkdir(parents=True,exist_ok=True);history=list(history);settings=continuation.binding['settings']
    updates={k:v for k,v in settings.items() if k!='total_updates'}
    while continuation.step<settings['total_updates']:
        row=continuation.update(offline,replay,**updates)
        row.update(cycle=2,cumulative_steps=cumulative_counts(source,settings,row['step']))
        if row['step']==1 or row['step']%diagnostic_interval==0 or row['step']==settings['total_updates']:
            row['gradient_diagnostics']=first.gradient_probe(continuation)
        history.append(row)
        if row['step']%50==0 or row['step']==settings['total_updates'] or STOP:
            save_state(folder,continuation,source,runner_binding,history);print(json.dumps(row,allow_nan=False),flush=True)
        if STOP:raise SystemExit(85)
    payload=save_state(folder,continuation,source,runner_binding,history)
    (folder/'resume.pt').replace(folder/'checkpoint.pt')
    atomic_json(folder/'completed.json',completion_record(folder,continuation.binding,runner_binding,payload))


def run(request_path,request_sha256):
    core.slurm()
    if not torch.cuda.is_available():raise RuntimeError('Actual graph continuation requires allocated Slurm GPU')
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']));torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    request_binding=bind(request_path,request_sha256);request=request_settings(read(request_path))
    cfg,decl,source,replay,binding=load_inputs(request)
    output=Path(request['output']);original_cfg=read(source['root']['binding']['config']['path'])
    runner_binding=dict(version=VERSION,cycle=2,request=request_binding,config=bind(Path(request['experiment'])/'config.json'),
        declaration=bind(Path(request['experiment'])/'declaration.json'),continuation=binding,
        source_sha256={name:sha(ROOT/name) for name in SOURCE_NAMES},diagnostic_interval=request['diagnostic_interval'],backend_calls=0)
    offline=first.offline_labels(source['root']);problem=load_problem(cfg['design'],'cuda')
    for data in (offline,replay):
        data['x']=data['x'].reshape(-1,len(problem.x),2).cuda();data['derived_B']=data['derived_B'].float().cuda();data['mask']=data['mask'].cuda()
    models=dict(flow=GraphPolicy(problem),actor=GraphPolicy(problem),critic=source['refit']['model'].cuda())
    optimizers=dict(flow=torch.optim.Adam(models['flow'].parameters(),lr=original_cfg['flow_lr']),
        actor=torch.optim.Adam(models['actor'].parameters(),lr=original_cfg['actor_lr']),
        critic=torch.optim.AdamW(models['critic'].parameters(),lr=original_cfg['critic_lr'],weight_decay=1e-4))
    generators={name:torch.Generator(device='cuda') for name in core.GENERATOR_KEYS}
    output.mkdir(parents=True,exist_ok=True)
    with (output/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'binding.json'
        if path.exists() and read(path)!=runner_binding:raise ValueError('Second output belongs to different inputs/settings')
        atomic_json(path,runner_binding)
        for name,digest in runner_binding['source_sha256'].items():
            snapshot=output/'sources'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not snapshot.exists():shutil.copyfile(ROOT/name,snapshot)
            bind(snapshot,digest)
        payload=checked_resume(output,runner_binding,binding,source)
        if (output/'completed.json').exists():checked_completion(output,payload,runner_binding,binding);return
        continuation=core.Continuation(models,optimizers,generators,source['payload'] if payload is None else payload,
            binding,frozen_reference=source,resume=payload is not None)
        for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
        run_loop(output,continuation,source,offline,replay,runner_binding,() if payload is None else payload['history'],
            diagnostic_interval=request['diagnostic_interval'])


def stop(*_):
    global STOP
    STOP=True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-sha256',required=True);args=parser.parse_args()
    run(args.request.resolve(),args.request_sha256)


if __name__=='__main__':main()
