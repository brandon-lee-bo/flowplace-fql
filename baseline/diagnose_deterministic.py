"""Check reproducibility and compiled/eager equivalence with deterministic kernels."""
import json,os,time
from api import *
if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Slurm required')
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
out=ROOT/'baseline/artifacts/deterministic_diagnostic';out.mkdir(parents=True,exist_ok=True)
report={'job':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'seeds':{}}
original_compile=torch.compile
try:
    problem=load_problem('bp_fe','cuda');torch.manual_seed(300);model,_=load_model(problem,'cuda',steps=20)
    for seed in [300,301,302]:
        z=2*torch.rand((11,2),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))-1
        results={};meta={}
        for mode in ['compiled1','compiled2','eager']:
            torch.compile=original_compile if mode.startswith('compiled') else (lambda fn,*a,**k:fn)
            torch.cuda.synchronize();start=time.perf_counter()
            x,noise,traj=generate(model,problem,noise=z,intermediate_every=1)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            results[mode]=x.cpu()
            meta[mode]={'hpwl':macro_net_hpwl(x,problem),'seconds':elapsed,**legality(x,problem.cond.x)}
            torch.save({'centers':x.cpu(),'noise':noise.cpu(),'trajectory':torch.stack([t.cpu() for t in traj])},out/f'{seed}_{mode}.pt')
        meta['compiled_repeat_max_abs']=float((results['compiled1']-results['compiled2']).abs().max())
        meta['compiled_eager_max_abs']=float((results['compiled1']-results['eager']).abs().max())
        report['seeds'][str(seed)]=meta
        (out/'report.json').write_text(json.dumps(report,indent=2))
except Exception as e:
    report['error']=repr(e)
finally:
    torch.compile=original_compile
    (out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
