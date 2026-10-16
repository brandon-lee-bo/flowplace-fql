"""Replay released eval RNG ordering: one seed, one model, six benchmark graphs.

Intended as a method-reproduction check, not a guarantee of paper-identical PPA:
the paper does not fully specify release-result seed/backend versions.
"""
import argparse,json,os,platform,time,hashlib
from api import *
from plot_sample import plot_sample

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=300);p.add_argument('--output',type=Path,required=True);p.add_argument('--disable-compile',action='store_true');p.add_argument('--deterministic',action='store_true');args=p.parse_args()
if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Submit generation through Slurm')
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
if args.deterministic:torch.use_deterministic_algorithms(True)
if args.disable_compile:torch.compile=lambda fn,*a,**k:fn
args.output.mkdir(parents=True,exist_ok=True)
torch.manual_seed(args.seed)
problems=[load_problem(name,'cuda') for name in DESIGNS]
model,cfg=load_model(problems[0],'cuda',20)
cfg.seed=args.seed
OmegaConf.save(cfg,args.output/'config.yaml')
manifest={'seed':args.seed,'rng_mode':'official global torch RNG stream seeded before model initialization','design_order':DESIGNS,'host':platform.node(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),'compile_disabled':args.disable_compile,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
(args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
results=[]
for problem in problems:
    print('START '+problem.design_name,flush=True)
    torch.cuda.synchronize();start=time.perf_counter()
    sample,noise,trajectory=generate(model,problem,intermediate_every=1)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    out=args.output/problem.design_name/f'seed_{args.seed}'
    metrics=save_sample(out,sample,noise,problem,{'design':problem.design_name,'seed':args.seed,'steps':20,'rng_mode':'official-stream','generation_seconds_including_legalizer_init_compile':elapsed,'hard_constraint':True,'w_reg':0.1,'w_dist':1.0})
    torch.save(torch.stack([x.detach().cpu().squeeze(0) for x in trajectory]),out/'trajectory.pt');plot_sample(out)
    print(json.dumps(metrics),flush=True);results.append(metrics)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
