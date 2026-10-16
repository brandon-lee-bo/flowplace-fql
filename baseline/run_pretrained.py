"""Generate placement baselines with the released FlowPlace sampler/checkpoint."""
import argparse,json,os,platform,subprocess,time,hashlib
from pathlib import Path
from api import *
from plot_sample import plot_sample


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--designs',nargs='+',default=['bp_fe'])
    p.add_argument('--seeds',nargs='+',type=int,default=[300])
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--disable-compile',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    args=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Submit numerical generation through Slurm")
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if args.disable_compile:
        # Wrapper-only performance workaround: identical eager tensor operations.
        torch.compile=lambda fn,*a,**k:fn
    args.output.mkdir(parents=True,exist_ok=True)
    meta={'argv':sys.argv,'host':platform.node(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'source_commit':(UPSTREAM/'SOURCE_COMMIT').read_text().strip(),'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),'compile_disabled':args.disable_compile,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'deviations':['strict checkpoint loading','direct graph loading identical to official preprocessing','explicit seeded initial noise saved','no visualization or external placement conversion in timed generation']}
    (args.output/'run_manifest.json').write_text(json.dumps(meta,indent=2))
    results=[]
    for design in args.designs:
        problem=load_problem(design,args.device)
        torch.manual_seed(300)
        model,cfg=load_model(problem,args.device,args.steps)
        cfg_path=args.output/design/'config.yaml';cfg_path.parent.mkdir(parents=True,exist_ok=True)
        OmegaConf.save(cfg,cfg_path)
        for seed in args.seeds:
            print(f'START design={design} seed={seed}',flush=True)
            if args.device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            centers,noise,traj=generate(model,problem,seed=seed,intermediate_every=1)
            if args.device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            out=args.output/design/f'seed_{seed}'
            metrics=save_sample(out,centers,noise,problem,{'design':design,'seed':seed,'steps':args.steps,'generation_seconds_including_legalizer_init_compile':elapsed,'checkpoint':str(CHECKPOINT),'hard_constraint':True,'w_reg':0.1,'w_dist':1.0})
            torch.save(torch.stack([x.detach().cpu().squeeze(0) for x in traj]),out/'trajectory.pt')
            plot_sample(out)
            print(json.dumps(metrics),flush=True)
            results.append(metrics)
            (args.output/'summary.json').write_text(json.dumps(results,indent=2))
        del model
        if args.device=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':main()
