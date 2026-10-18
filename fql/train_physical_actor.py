"""Matched BC/FQL-style actor continuation with an existing GRT ridge critic.

One predeclared alpha, three seeds, two validation noises each: 12 backend
candidates. Teacher and critic frozen; no final heldout or new design claims.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL,legality
from .core import OneStepActor,write_json
from .fresh_flow import BehaviorFlow,student_loss
from .physical_ridge import PhysicalRidgeCritic
from .provenance import capture_source_provenance
from .run_fresh_flow import update
from baseline.api import load_problem,legalize,save_sample,RUNTIME
from backend.scripts.export_macro_def import convert


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def action_sha(action):return hashlib.sha256(np.ascontiguousarray(action,dtype='<f4').reshape(-1).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',required=True)
    p.add_argument('--critic-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1500)
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():raise RuntimeError('Use Slurm GPU')
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    snapshot=capture_source_provenance(out/'source_snapshot',extra_paths={n:root/n for n in
        ['fql/train_physical_actor.py','fql/physical_ridge.py','fql/fresh_flow.py',
         'fql/run_fresh_flow.py','fql/train_physical_actor.slurm','baseline/api.py',
         'analysis/placement_metrics.py','backend/scripts/export_macro_def.py']})
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pilot=Path(args.pilot);critic_dir=Path(args.critic_dir)
    pilot_report=json.loads((pilot/'report.json').read_text())
    cr=json.loads((critic_dir/'report.json').read_text())
    if pilot_report['status']!='completed' or cr['status']!='completed':raise ValueError('Inputs incomplete')
    selected=cr['models']['raw_physical']
    if not (selected['beats_constant_validation_mae'] and selected['positive_validation_rank']):
        raise ValueError('Predeclared raw physical critic did not pass exploratory sanity check')
    cache=RUNTIME/'benchmark_cache/bp_fe.pt'
    if sha(cache)!=cr['cache_sha256']:raise ValueError('Cache mismatch')
    critic_data=np.load(critic_dir/'raw_physical.npz',allow_pickle=False)
    critic=PhysicalRidgeCritic(torch.load(cache,map_location='cpu',weights_only=False),critic_data).cuda()
    macro_dataset=Path(pilot_report['dataset'])
    if sha(macro_dataset)!=pilot_report['dataset_sha256']:raise ValueError('Macro dataset changed')
    with np.load(macro_dataset,allow_pickle=False) as a:
        indices=np.flatnonzero(a['split']==1)[:2]
        uniform=a['noise'][indices].copy()
    if indices.tolist()!=[192,193]:raise ValueError('Frozen candidate families differ')
    z=torch.erfinv(torch.tensor(uniform,device='cuda'))*np.sqrt(2.)
    problem=load_problem('bp_fe','cuda');metric=MacroHPWL(problem.benchmark_info)
    request=out/'exports';request.mkdir()
    configurations=['bc_continued','fql_physical_alpha_1']
    plan={'phase':'exploratory','design':'bp_fe','indices':indices.tolist(),
        'methods':[f'seed_{s}/{n}' for s in [0,1,2] for n in configurations],
        'seeds':[0,1,2],'training_steps':args.steps,'alpha':1.,'new_backend_budget':12,'heldout':False,
        'metric_stage':'global_route_only','reward':'-GRT rWL / frozen TRAIN mean from existing labels',
        'dataset_sha256':pilot_report['dataset_sha256'],'critic_dataset_sha256':cr['dataset_sha256'],
        'critic_sha256':sha(critic_dir/'raw_physical.npz'),'source_macro_report_sha256':sha(pilot/'report.json'),
        'source_provenance':snapshot,'selection':'One alpha, fixed final step and fixed first two validation families. No Q-based candidate selection.',
        'physical_failure_note':'Keep all 12 candidates. Any failed case prevents an all-success method mean. No parameter relaxation or scientific-input retry.'}
    write_json(request/'selection_plan.json',plan)
    report={'status':'running','args':vars(args),'plan':plan,'seeds':[],'slurm_job_id':os.environ['SLURM_JOB_ID']}
    write_json(out/'report.json',report)
    tasks=[]
    for seed in [0,1,2]:
        checkpoint_path=pilot/f'seed_{seed}/checkpoint.pt'
        checkpoint=torch.load(checkpoint_path,map_location='cuda',weights_only=False)
        mask,fixed=checkpoint['mask'],checkpoint['fixed']
        width=checkpoint['args']['width'];widths=[width]*3
        flow=BehaviorFlow(uniform.shape[1],widths,mask,fixed).cuda();flow.load_state_dict(checkpoint['flow_state'])
        initial=OneStepActor(uniform.shape[1],widths,mask,fixed).cuda();initial.load_state_dict(checkpoint['actor_states']['distill_only'])
        actors={name:copy.deepcopy(initial) for name in configurations}
        optimizers={name:torch.optim.Adam(actor.parameters(),lr=3e-4) for name,actor in actors.items()}
        rng=torch.Generator(device='cuda').manual_seed(220000+seed)
        started=time.perf_counter();trace=[]
        for step in range(args.steps):
            noise=torch.randn(128,uniform.shape[1],generator=rng,device='cuda')
            target=flow.sample(noise,checkpoint['args']['flow_steps'])
            log={}
            for name,actor in actors.items():
                loss,dist,q=student_loss(actor,noise,target,critic,1.,float(name!='bc_continued'))
                update(loss,optimizers[name],actor.parameters())
                log[name]={'distill':float(dist.detach()),'predicted_q':float(q.detach())}
            if step==0 or (step+1)%500==0:trace.append(dict(step=step+1,**log))
        folder=out/f'seed_{seed}';folder.mkdir()
        torch.save({'actors':{k:v.state_dict() for k,v in actors.items()},'source_checkpoint_sha256':sha(checkpoint_path),'args':vars(args)},folder/'checkpoint.pt')
        record={'seed':seed,'seconds':time.perf_counter()-started,'trace':trace,'results':[]}
        for name,actor in actors.items():
            with torch.no_grad():raw=actor(z,clip=True).cpu().numpy();q=critic(torch.tensor(raw,device='cuda')).cpu().numpy()[:,0]
            for local,index in enumerate(indices):
                action=legalize(torch.tensor(raw[local].reshape(-1,2),device='cuda'),problem).cpu().numpy()
                if not legality(action,metric.sizes)['legal']:raise ValueError('Illegal candidate retained in failed export')
                path=request/f'seed_{seed}'/name/str(index)
                save_sample(path,torch.tensor(action),z[local].cpu().reshape(-1,2),problem,
                    {'phase':'exploratory','seed':seed,'family':int(index),'noise_prior':'Gaussian inverse CDF of cached uniform noise','predicted_q':float(q[local])})
                np.savez_compressed(path/'actions.npz',raw_action=raw[local:local+1],action=action.reshape(1,-1),noise=z[local:local+1].cpu().numpy())
                def_path=path/'bp_fe.def';audit=convert('bp_fe',path/'macro_physical.json',def_path)
                if not audit['legal_no_halo']:raise ValueError('Rounded DEF illegal')
                run_id=f'fqlphys_{os.environ["SLURM_JOB_ID"]}_s{seed}_{name}_i{index}'
                q_manifest=root/'datasets/bp_fe_grt_v1/manifest.json'
                tasks.append({'method':f'seed_{seed}/{name}','seed':seed,'training_seed':seed,'index':int(index),'split':1,'design':'bp_fe',
                    'run_id':run_id,'def_path':str(def_path),'def_sha256':sha(def_path),
                    'dataset_sha256':pilot_report['dataset_sha256'],'action_sha256':action_sha(action),
                    'action_path':str(path/'actions.npz'),'action_key':'action','action_row':0,'action_file_sha256':sha(path/'actions.npz'),
                    'raw_action_sha256':action_sha(raw[local]),'predicted_q':float(q[local]),
                    'q_reward_stage':'global_route_only','q_reward_scale_train_only':2341121.25,
                    'q_source_manifest':str(q_manifest),'q_source_manifest_sha256':sha(q_manifest),
                    'normalized_sizes':metric.sizes.tolist(),
                    'metrics_path':str(root/'backend/results'/run_id/'metrics.json'),
                    'manifest_path':str(root/'backend/results'/run_id/'manifest.json'),
                    'macro_hpwl':float(metric(action)),'phase':'exploratory'})
                record['results'].append({'method':name,'index':int(index),'macro_hpwl':float(metric(action)),'predicted_q':float(q[local])})
        report['seeds'].append(record);write_json(out/'report.json',report)
    if len(tasks)!=12:raise ValueError('Candidate count differs from frozen budget')
    path=request/'tasks.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in tasks))
    report.update(status='completed',tasks_jsonl=str(path),tasks_sha256=sha(path),backend_evaluated=False)
    write_json(out/'report.json',report)
    print(json.dumps({'status':'completed','tasks':str(path),'count':len(tasks)}),flush=True)


if __name__=='__main__':main()
