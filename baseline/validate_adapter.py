"""Meaningful adapter checks: geometric failure detection and upstream equivalence."""
import json,argparse,time,os
if "SLURM_JOB_ID" not in os.environ:raise RuntimeError("Submit GPU validation through Slurm")
parser=argparse.ArgumentParser();parser.add_argument("--compiled-reference");parser.add_argument("--deterministic",action="store_true");args=parser.parse_args()
from api import *

torch.set_num_threads(4)
if args.deterministic:torch.use_deterministic_algorithms(True)
torch.compile=lambda fn,*a,**k:fn
# Small tests exercise independent legality including touching boundaries.
sizes=torch.tensor([[.5,.5],[.5,.5]])
assert legality(torch.tensor([[-.75,0],[.75,0]]),sizes)['is_legal']
assert legality(torch.tensor([[0.,0],[.1,.1]]),sizes)['overlap_pairs']==1
assert legality(torch.tensor([[1.,0],[0,0]]),sizes)['out_of_bounds_macros']==1
assert legality(torch.tensor([[-.25,0],[.25,0]]),sizes)['is_legal']
assert not legality(torch.tensor([[float('nan'),0],[0,0]]),sizes)['is_legal']
assert not legality(torch.tensor([[float('inf'),0],[0,0]]),sizes)['is_legal']
assert not legality(torch.tensor([[0,0],[0,0]]),torch.tensor([[0.,1],[1,1]]))['is_legal']
problem=load_problem('bp_fe','cuda')
model,cfg=load_model(problem,'cuda',steps=2)
# Same global seed must produce same initial noise and sampler result.
torch.manual_seed(77)
with torch.no_grad():
    direct,_=model.reverse_samples(1,problem.x.unsqueeze(0),problem.cond)
adapted,noise,_=generate(model,problem,seed=77)
assert torch.allclose(direct.squeeze(0),adapted,atol=1e-6,rtol=0), (direct-adapted).abs().max()
# Same supplied noise repeats exactly; no unintended RNG dependence.
repeated,_,_=generate(model,problem,noise=noise)
assert torch.allclose(adapted,repeated,atol=1e-6,rtol=0)
result={'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'geometry_checks':'passed','fixed_noise_equivalence_max_abs':float((direct.squeeze(0)-adapted).abs().max()),'fixed_noise_repeat_max_abs':float((adapted-repeated).abs().max()),'strict_checkpoint_loaded':True,'num_parameters':sum(p.numel() for p in model.parameters()),'generated_legality':legality(adapted,problem.cond.x)}
if args.compiled_reference:
    reference=Path(args.compiled_reference)
    saved=torch.load(reference/'sample.pt',map_location='cpu',weights_only=False)
    ref_metrics=json.loads((reference/'metrics.json').read_text())
    problem=load_problem(saved['design'],'cuda')
    model,_=load_model(problem,'cuda',steps=ref_metrics['steps'])
    torch.cuda.synchronize();start=time.perf_counter()
    eager,_,_=generate(model,problem,noise=saved['noise'])
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    difference=float((eager.cpu()-saved['centers']).abs().max())
    projected=legalize(eager,problem)
    result.update({'extra_final_projection_max_abs':float((projected-eager).abs().max()),'extra_final_projection_hpwl_before':macro_net_hpwl(eager,problem),'extra_final_projection_hpwl_after':macro_net_hpwl(projected,problem)})
    result.update({'compiled_reference':str(reference),'compiled_vs_eager_max_abs':difference,'compiled_vs_eager_allclose':bool(torch.allclose(eager.cpu(),saved['centers'],atol=1e-6,rtol=0)),'eager_generation_seconds':elapsed})
    (ROOT/'baseline/artifacts/adapter_validation.json').write_text(json.dumps(result,indent=2))
    assert result['compiled_vs_eager_allclose'],result
print(json.dumps(result,indent=2))
(ROOT/'baseline/artifacts/adapter_validation.json').write_text(json.dumps(result,indent=2))
