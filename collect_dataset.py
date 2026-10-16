"""Collect a real-circuit terminal-action dataset, always inside Slurm.

Teacher/critic splits are by initial-noise family, including perturbations.
No backend PPA is inferred from the inexpensive macro-only reward.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from analysis.placement_metrics import MacroHPWL, legality, summarize
from baseline.api import load_problem, load_model, generate, legalize, save_sample, macro_net_hpwl, CHECKPOINT, RUNTIME
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--design', default='bp_fe')
    p.add_argument('--output', required=True)
    p.add_argument('--train', type=int, default=192)
    p.add_argument('--val', type=int, default=48)
    p.add_argument('--test', type=int, default=48)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--model-init-seed',type=int,default=300)
    p.add_argument('--perturbations', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--disable-compile', action='store_true')
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Submit dataset collection through Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    torch.use_deterministic_algorithms(True)
    if args.disable_compile:
        torch.compile = lambda fn, **kwargs: fn
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if min(args.train,args.val,args.test)<1 or args.perturbations<0:
        raise ValueError('Require nonempty train/validation/test splits and nonnegative perturbations')
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    provenance={'args':{k:v for k,v in vars(args).items() if k!='output'},
                'checkpoint_sha256':sha(CHECKPOINT),
                'cache_sha256':sha(RUNTIME/'benchmark_cache'/f'{args.design}.pt'),
                'collector_sha256':sha(__file__),'adapter_sha256':sha(ROOT/'baseline/api.py'),
                'torch_version':torch.__version__,'deterministic_algorithms':True,
                'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']}
    provenance_path=out/'collection_provenance.json'
    if provenance_path.exists():
        if json.loads(provenance_path.read_text())!=provenance:
            raise RuntimeError('Resume provenance differs; choose a fresh dataset directory')
    else:
        if any(out.glob('samples/*/dataset_row.npz')):
            raise RuntimeError('Refusing to resume samples without provenance')
        provenance_path.write_text(json.dumps(provenance,indent=2))
    problem = load_problem(args.design, args.device)
    torch.manual_seed(args.model_init_seed)
    model, config = load_model(problem, args.device)
    metric = MacroHPWL(problem.benchmark_info)
    if list(problem.macro_names)!=metric.names:
        raise ValueError('Adapter and independent metric disagree on macro ordering')
    np.testing.assert_allclose(metric.sizes,problem.cond.x.cpu().numpy(),rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(metric(problem.x.cpu().numpy()),macro_net_hpwl(problem.x,problem),rtol=1e-6)
    n = args.train + args.val + args.test
    split = np.repeat([0,1,2], [args.train,args.val,args.test]).astype(np.int64)
    rows, noise_list, teachers, sample_time = [], [], [], []
    start = time.time()
    for i in range(n):
        path = out / 'samples' / f'{i:04d}'
        if (path/'dataset_row.npz').exists():
            with np.load(path/'dataset_row.npz') as d:
                noise_list.append(d['noise']); teachers.append(d['teacher'])
                sample_time.append(float(d['seconds']))
        else:
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            t = time.time()
            x, z, _ = generate(model, problem, seed=args.seed+i)
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            elapsed = time.time()-t
            save_sample(path,x,z,problem,{'seed':args.seed+i,'seconds':elapsed,'split':int(split[i]),
                        'slurm_job_id':os.environ['SLURM_JOB_ID'],'compile_disabled':args.disable_compile})
            noise_list.append(z.cpu().numpy()); teachers.append(x.cpu().numpy());sample_time.append(elapsed)
            np.savez(path/'dataset_row.npz',noise=noise_list[-1],teacher=teachers[-1],seconds=elapsed)
        # RNG is per family, so resume does not alter collected data.
        local_rng=np.random.default_rng(args.seed+100000+i)
        teacher=teachers[-1]
        if not np.isfinite(teacher).all() or not np.isfinite(noise_list[-1]).all():
            raise FloatingPointError(f'Nonfinite teacher/noise at family {i}')
        variants=[teacher]
        for j in range(args.perturbations):
            scale=[.04,.15,.35][j%3]
            variants.append(np.clip(teacher+local_rng.normal(0,scale,teacher.shape),-1,1).astype('float32'))
        for j,raw in enumerate(variants):
            x=legalize(torch.as_tensor(raw,device=args.device),problem).cpu().numpy()
            report=legality(x,metric.sizes)
            if not report['finite']:raise FloatingPointError(f'Nonfinite legalizer output at family {i}, variant {j}')
            rows.append({'raw':raw,'action':x,'hpwl':metric(x),'legal':report['legal'],
                         'split':split[i],'family':i,'variant':j})
        if (i+1)%8==0 or i==0:
            print(json.dumps({'completed':i+1,'total':n,'wall_s':time.time()-start,
                              'teacher_legal':legality(teacher,metric.sizes)['legal']}),flush=True)
    train_hpwl=[r['hpwl'] for r in rows if r['split']==0 and r['legal']]
    scale=float(np.mean(train_hpwl))
    if not np.isfinite(scale) or scale<=0:raise RuntimeError('invalid training reward scale')
    raw=np.stack([r['raw'] for r in rows]).reshape(len(rows),-1)
    actions=np.stack([r['action'] for r in rows]).reshape(len(rows),-1)
    hpwl=np.array([r['hpwl'] for r in rows])
    valid=np.array([r['legal'] for r in rows])
    rewards=-hpwl/scale - (~valid)*10.
    np.savez_compressed(out/'dataset.npz',noise=np.stack(noise_list).reshape(n,-1),
        teacher_action=np.stack(teachers).reshape(n,-1),split=split,raw_action=raw,action=actions,
        reward=rewards.astype('float32'),critic_split=np.array([r['split'] for r in rows]),
        family_id=np.array([r['family'] for r in rows]),variant=np.array([r['variant'] for r in rows]),
        group_id=np.arange(n),critic_group_id=np.array([r['family'] for r in rows]),
        critic_parent_index=np.array([r['family'] for r in rows]),
        legal=valid,macro_hpwl=hpwl,teacher_seconds=np.array(sample_time))
    meta={'args':vars(args),'metric':metric.metadata(),'reward_kind':'macro_hpwl','reward_scale_train_only':scale,
          'reward':'-macro_only_hpwl/train_mean - 10*illegal','wall_s':time.time()-start,
          'slurm_job_id':os.environ['SLURM_JOB_ID'],'num_teacher':n,'num_critic':len(rows),
          'provenance':provenance,
          'legal_fraction':float(valid.mean()),'teacher_generation_seconds':summarize(sample_time),
          'notes':['single fixed circuit = terminal contextual bandit, no Bellman bootstrapping',
                   'critic labels measure identical legalization applied to raw action',
                   'validation and test families excluded from all training',
                   'full backend labels are not collected by this script']}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
