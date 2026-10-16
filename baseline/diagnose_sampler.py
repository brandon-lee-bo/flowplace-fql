"""Isolate compilation effects from model state, input noise and nondeterminism."""
import json,os,time
from api import *
if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Slurm required')
torch.set_num_threads(4)
out=ROOT/'baseline/artifacts/sampler_diagnostic';out.mkdir(parents=True,exist_ok=True)
reference=torch.load(ROOT/'baseline/artifacts/pretrained_original/bp_fe/seed_300/sample.pt',map_location='cpu',weights_only=False)
problem=load_problem('bp_fe','cuda')
torch.manual_seed(300)
model,_=load_model(problem,'cuda',steps=20)
noise=reference['noise'].cuda()
original_compile=torch.compile
samples={};trajectories={};report={'job':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),'runs':{},'comparisons':{}}
def difference(a,b):return {'max_abs':float((a-b).abs().max()),'mse':float(((a-b)**2).mean()),'equal':bool(torch.equal(a,b))}
def flush():(out/'report.json').write_text(json.dumps(report,indent=2))
for mode in ['compiled1','compiled2','eager1','eager2','compiled3']:
    torch.compile=original_compile if mode.startswith('compiled') else (lambda fn,*a,**k:fn)
    torch.cuda.synchronize();start=time.perf_counter()
    x,z,traj=generate(model,problem,noise=noise,intermediate_every=1)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    samples[mode]=x.cpu();trajectories[mode]=torch.stack([t.cpu().squeeze(0) for t in traj])
    torch.save({'centers':samples[mode],'trajectory':trajectories[mode]},out/f'{mode}.pt')
    report['runs'][mode]={'seconds':elapsed,'hpwl':macro_net_hpwl(x,problem),**legality(x,problem.cond.x)}
    flush()
for a,b in [('compiled1','compiled2'),('compiled1','compiled3'),('eager1','eager2'),('compiled1','eager1')]:
    report['comparisons'][a+'_vs_'+b]={**difference(samples[a],samples[b]),'per_step_max_abs':(trajectories[a]-trajectories[b]).abs().flatten(1).max(1).values.tolist()}
report['comparisons']['fresh_compiled_vs_saved']=difference(samples['compiled1'],reference['centers'])
# Same exact network prediction / legalizer input for both implementations.
torch.compile=original_compile
with torch.no_grad():
    inp=noise.unsqueeze(0); t=torch.zeros(1,device='cuda')
    v1=model(inp,problem.cond,t);v2=model(inp,problem.cond,t)
    report['same_forward_repeat']=difference(v1,v2)
    u=inp+v1
    eager_leg=Legalizer(problem.cond,grid_res=128,scores={'w_reg':.1,'w_dist':1.},device='cuda',dtype=u.dtype,padding=.001)
    compiled_leg=Legalizer(problem.cond,grid_res=128,scores={'w_reg':.1,'w_dist':1.},device='cuda',dtype=u.dtype,padding=.001)
    compiled_run=original_compile(compiled_leg.run,mode='reduce-overhead')
    c1=compiled_run(u);c2=compiled_run(u);e1=eager_leg.run(u);e2=eager_leg.run(u)
    report['fixed_legalizer_input']={'compiled_repeat':difference(c1,c2),'eager_repeat':difference(e1,e2),'compiled_vs_eager':difference(c1,e1)}
flush()
# Check whether deterministic backend kernels remove trajectory variability.
try:
    torch.use_deterministic_algorithms(True)
    deterministic={}
    for mode in ['compiled','eager']:
        torch.compile=original_compile if mode=='compiled' else (lambda fn,*a,**k:fn)
        x,_,_=generate(model,problem,noise=noise)
        deterministic[mode]=x.cpu()
    report['deterministic_compiled_vs_eager']=difference(deterministic['compiled'],deterministic['eager'])
except Exception as e:
    report['deterministic_error']=repr(e)
finally:
    torch.compile=original_compile
flush();print(json.dumps(report,indent=2),flush=True)
