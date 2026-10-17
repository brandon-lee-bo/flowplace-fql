"""Matched extra-update control: continue FQL without any online feedback.

Start from the exact main pilot's offline FQL checkpoint. Run two matched
600-update stages on original TRAIN only. Fixed final checkpoint/noises, no
Q-ranked selection. This separates extra optimization from new measurements.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import legalize, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_train import initialize_problem, construct, update, save_point, parameter_changes, critic_stats
from .graph_o2o_models import actor_objective


def run(args):
    experiment=Path(args.experiment).resolve()
    cfg=json.loads((experiment/'config.json').read_text())
    source=Path(cfg['source_experiment'])
    if sha(source/'config.json')!=cfg['source_config_sha256']:
        raise ValueError('Parent configuration changed')
    parent=Path(cfg['parent_checkpoint'])
    if sha(parent)!=cfg['parent_checkpoint_sha256']:
        raise ValueError('The paired offline initialization changed')
    snapshot=experiment/('source_job_'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(exist_ok=False)
    for name in ['graph_o2o_offline_continue.py','graph_o2o_offline_continue.slurm',
                 'graph_o2o_train.py','graph_o2o_models.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    problem,data,manifest,cache,x,y,vx,vy,behavior=initialize_problem(args)
    if manifest['dataset_sha256']!=cfg['offline_dataset_sha256'] or len(x)!=64:
        raise ValueError('Original offline TRAIN data changed')
    for stage in range(1,cfg['extra_training_stages']+1):
        folder=experiment/f'training_stage_{stage}';folder.mkdir(exist_ok=True)
        if (folder/'completed.json').exists():
            done=json.loads((folder/'completed.json').read_text())
            if sha(folder/'checkpoint.pt')!=done['checkpoint_sha256']:
                raise ValueError('Completed control checkpoint changed')
            parent=folder/'checkpoint.pt';continue
        resume=folder/'resume.pt'
        saved=torch.load(resume if resume.exists() else parent,map_location='cuda',weights_only=False)
        expected=sha(experiment/'config.json') if resume.exists() or stage>1 else cfg['source_config_sha256']
        if saved['config_sha256']!=expected:
            raise ValueError('Resumed training belongs to a different configuration')
        flow,actor,critic,opts=construct(problem,cache,x,y,saved,cfg)
        begin=saved.get('iteration',0) if resume.exists() else 0
        if not resume.exists():torch.manual_seed(100000+args.seed*1000+stage)
        started=time.perf_counter()
        with (folder/'training.jsonl').open('a') as handle:
            for step in range(begin,cfg['online_update_steps']):
                n=cfg['batch_size'];online_n=round(n*cfg['online_batch_fraction'])
                index=torch.randint(len(x),(n,),device='cuda')
                # Discard the same-size extra index draws made by the online
                # arm, keeping fresh FM/actor noise aligned without feedback.
                torch.randint(stage*cfg['source_online_samples'],(online_n,),device='cuda')
                bindex=torch.randint(len(behavior),(n,),device='cuda')
                torch.randint(stage*cfg['source_online_samples'],(online_n,),device='cuda')
                z=2*torch.rand(n,len(problem.x),2,device='cuda')-1
                zbc=2*torch.rand_like(z)-1;t=torch.rand(n,device='cuda')
                aloss,distill,aq=actor_objective(actor,flow,critic,z,alpha=cfg['alpha'],
                    q_weight=1.,flow_steps=cfg['flow_steps'])
                floss=flow.flow_loss(behavior[bindex],zbc,t)
                qloss=(critic(x[index])-y[index,None]).square().mean()
                update(aloss,opts['actor'],actor);update(floss,opts['flow'],flow);update(qloss,opts['critic'],critic)
                if (step+1)%50==0 or step+1==cfg['online_update_steps']:
                    row=dict(stage=stage,step=step+1,actor_loss=float(aloss),flow_loss=float(floss),
                        critic_loss=float(qloss),distill=float(distill),q=float(aq),
                        measured_TRAIN_rows=len(x),online_training_rows=0,seconds=time.perf_counter()-started)
                    handle.write(json.dumps(row)+'\n');handle.flush();print(json.dumps(row),flush=True)
                if (step+1)%250==0 or step+1==cfg['online_update_steps']:
                    metadata=dict(iteration=step+1,stage=stage,seed=args.seed,config_sha256=sha(experiment/'config.json'),
                        parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),online_rows=0)
                    save_point(folder,'resume.pt',flow,actor,critic,opts,metadata)
                    save_point(folder,f'weights_step_{step+1}.pt',flow,actor,critic,opts,metadata,full=False)
            save_point(folder,'checkpoint.pt',flow,actor,critic,opts,
                dict(iteration=cfg['online_update_steps'],stage=stage,seed=args.seed,config_sha256=sha(experiment/'config.json'),
                     parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),online_rows=0))
        before=torch.load(parent,map_location='cpu',weights_only=False)
        changes={name:parameter_changes(module,before[name]) for name,module in [('flow',flow),('actor',actor),('critic',critic)]}
        del before
        atomic_json(folder/'completed.json',dict(status='completed',checkpoint_sha256=sha(folder/'checkpoint.pt'),
            stage=stage,online_training_rows=0,measured_train_rows=len(x),behavior_rows=len(behavior),
            parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),parameter_changes=changes,
            critic_train=critic_stats(critic,x,y.cpu().numpy()),critic_validation=critic_stats(critic,vx,vy),
            slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started))
        parent=folder/'checkpoint.pt'
        del flow,actor,critic,opts,saved;torch.cuda.empty_cache()
    dest=experiment/'seed_0/round_0/requests'
    if (dest/'completed.json').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    saved=torch.load(parent,map_location='cuda',weights_only=False)
    _,actor,critic,_=construct(problem,cache,x,y,saved,cfg)
    metric=MacroHPWL(problem.benchmark_info)
    generator=torch.Generator(device='cuda').manual_seed(cfg['evaluation_noise_seed'])
    noises=2*torch.rand(cfg['evaluation_samples'],len(problem.x),2,generator=generator,device='cuda')-1
    records=[]
    for i,noise in enumerate(noises):
        folder=dest/'fql_offline_continued'/'evaluation'/str(i)
        if (folder/'request.json').exists():
            records.append(json.loads((folder/'request.json').read_text()));continue
        folder.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():raw=actor(noise[None])[0];q=critic(raw[None]).cpu().tolist()[0];action=legalize(raw,problem)
        values=action.cpu().numpy();geometry=legality(values,metric.sizes)
        row=dict(arm='fql_offline_continued',seed=0,round=0,role='evaluation',index=i,
            noise_seed=cfg['evaluation_noise_seed'],graph='bp_fe',terminal_mask=0,
            raw_action=raw.cpu().tolist(),executed_action=values.tolist(),predicted_q=q,
            source_checkpoint=str(parent),checkpoint_sha256=sha(parent),geometry=geometry,
            action_sha256=hashlib.sha256(np.asarray(values,dtype='<f4').reshape(-1).tobytes()).hexdigest(),
            sample_dir=str(folder),run_id=f'go2o_offlinecontinue_v1_s0_eval_{i}',stage='global_route_only',
            training_feedback_eligible=False,macro_hpwl=float(metric(values)),
            control='Same parent and extra update budget as online FQL; only original offline TRAIN data')
        np.savez_compressed(folder/'actions.npz',noise=noise.cpu().numpy(),raw_action=raw.cpu().numpy(),action=values)
        if geometry['legal']:
            save_sample(folder,action,noise,problem,row)
            path=folder/('bp_fe_job'+os.environ['SLURM_JOB_ID']+'.def')
            audit=convert('bp_fe',folder/'macro_physical.json',path)
            row.update(def_path=str(path),def_sha256=sha(path),rounded_geometry=audit['legal_no_halo'])
            if not audit['legal_no_halo']:row['pre_backend_failure']='rounded_DEF_geometry'
        else:row['pre_backend_failure']='normalized_geometry'
        atomic_json(folder/'request.json',row);records.append(row)
    tasks=dest/'tasks.jsonl';tasks.write_text(''.join(json.dumps(row)+'\n'for row in records))
    atomic_json(dest/'completed.json',dict(status='completed',count=len(records),tasks_sha256=sha(tasks),
        config_sha256=sha(experiment/'config.json'),online_feedback_rows=0,
        extra_updates=cfg['extra_training_stages']*cfg['online_update_steps'],slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status='offline_continuation_exported',count=len(records),online_feedback_rows=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():raise RuntimeError('Use Slurm GPU')
    if args.seed!=0:raise ValueError('This control is explicitly paired to pilot seed0')
    run(args)
