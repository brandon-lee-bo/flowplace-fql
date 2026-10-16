"""Measured, identical-legalizer evaluation of FQL and distilled proposals.

Reports original teacher AND re-legalized teacher because the released greedy
legalizer is not assumed idempotent. Never uses a Q prediction as a reward.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from analysis.placement_metrics import MacroHPWL, legality, summarize
from baseline.api import load_problem, legalize, save_sample
from fql.core import critic_diagnostics
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--training-dir',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--phase',choices=['validation','heldout','exploratory'],default='validation')
    p.add_argument('--selection',help='Required for heldout: validation-only selection JSON')
    p.add_argument('--export-per-split',type=int,default=3,
                   help='Fixed FIRST indices, not metric-selected best samples')
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    torch.use_deterministic_algorithms(True)
    ds=Path(args.dataset_dir);train=Path(args.training_dir)
    meta=json.loads((ds/'manifest.json').read_text())
    if meta.get('reward_stage')=='global_route_only':
        raise ValueError('This evaluator calibrates macro-only Q. GRT-trained proposals require measured backend rewards.')
    design=meta['args']['design'];scale=meta['reward_scale_train_only']
    problem=load_problem(design,args.device);metric=MacroHPWL(problem.benchmark_info)
    with np.load(ds/'dataset.npz') as f:
        noise=f['noise'];teacher=f['teacher_action'];split=f['split']
        support=f['raw_action'][f['critic_split']==0]
    wanted={'validation':[1],'heldout':[2],'exploratory':[1,2]}[args.phase]
    indices=np.flatnonzero(np.isin(split,wanted))
    selected=None
    if args.phase=='heldout':
        if not args.selection:raise ValueError('Lock methods using validation before heldout evaluation')
        selection=json.loads(Path(args.selection).read_text())
        selected=set(selection['methods'])
        if Path(selection['training_dir']).resolve()!=train.resolve():
            raise ValueError('Selection belongs to a different training directory')
    shape=(len(metric.names),2)
    export_ids=set()
    for value in np.unique(split[indices]):export_ids.update(np.flatnonzero(split==value)[:args.export_per_split])
    results={}

    def measure(method,raw,predicted_q=None,reproject=True):
        out=train/f'measured_{args.phase}'/method;out.mkdir(parents=True,exist_ok=True)
        records=[];positions=[]
        for i in indices:
            proposed=raw[i].reshape(shape)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            t=time.time()
            if reproject:
                x=legalize(torch.as_tensor(proposed,device=args.device),problem).cpu().numpy()
            else:x=proposed
            sec=time.time()-t
            r=legality(x,metric.sizes)
            if not r['finite']:raise FloatingPointError(f'Nonfinite layout {method} {i}')
            h=metric(x);reward=-h/scale-(0 if r['legal'] else 10)
            record={'index':int(i),'split':int(split[i]),'macro_hpwl':h,'reward':reward,
                    'legal':r['legal'],'overlap_pairs':r['overlap_pairs'],'out_of_bounds':r['out_of_bounds'],
                    'legalization_seconds':sec,'legalization_mse':float(np.square(x-proposed).mean()),
                    'nearest_training_raw_mse':float(np.square(support-proposed.reshape(1,-1)).mean(axis=1).min()),
                    'predicted_q':None if predicted_q is None else float(np.mean(predicted_q[i]))}
            if record['predicted_q'] is not None:record['q_error']=record['predicted_q']-reward
            records.append(record);positions.append(x)
            if i in export_ids:
                save_sample(out/f'sample_{i:04d}',torch.as_tensor(x),torch.as_tensor(noise[i].reshape(shape)),
                            problem,{'method':method,'index':int(i),'split':int(split[i]),
                                     'selection':'predeclared first evaluation indices, not best-of-K',
                                     'slurm_job_id':os.environ['SLURM_JOB_ID']})
        np.savez_compressed(out/'evaluated.npz',indices=indices,action=np.stack(positions),split=split[indices])
        (out/'records.json').write_text(json.dumps(records,indent=2))
        summary={}
        for value,label in [(1,'validation'),(2,'heldout')]:
            subset=[r for r in records if r['split']==value]
            if not subset:continue
            valid=[r for r in subset if r['legal']]
            summary[label]={'count':len(subset),'legal_fraction':len(valid)/len(subset),
                            'macro_hpwl_all':summarize([r['macro_hpwl'] for r in subset]),
                            'macro_hpwl_legal_only':summarize([r['macro_hpwl'] for r in valid]),
                            'reward':summarize([r['reward'] for r in subset]),
                            'nearest_training_raw_mse':summarize([r['nearest_training_raw_mse'] for r in subset]),
                            'legalization_mse':summarize([r['legalization_mse'] for r in subset])}
            if predicted_q is not None:
                summary[label]['q_signed_error']=summarize([r['q_error'] for r in subset])
                subset_ids=[r['index'] for r in subset]
                summary[label]['q_calibration']=critic_diagnostics(
                    predicted_q[subset_ids],np.array([r['reward'] for r in subset]))
            summary[label]['legalization_seconds']=summarize([r['legalization_seconds'] for r in subset])
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({'method':method,'summary':summary}),flush=True)
        results[method]=summary

    measure('teacher_original',teacher,reproject=False)
    measure('teacher_relegalized',teacher)
    for path in sorted(train.glob('seed_*/*/proposals.npz')):
        with np.load(path) as f:
            if not np.array_equal(f['split'],split) or not np.array_equal(f['noise'],noise):
                raise RuntimeError('Training proposals do not match dataset paired noise/splits')
            method=str(path.parent.relative_to(train))
            if selected is not None and method not in selected:continue
            measure(method,f['raw_action'],f['predicted_q'])
    (train/f'measured_{args.phase}_summary.json').write_text(json.dumps({'design':design,'metric':metric.metadata(),
        'phase':args.phase,'selection':args.selection,
        'reward_scale':scale,'methods':results,'slurm_job_id':os.environ['SLURM_JOB_ID'],
        'warning':'Macro proxy only. Full OpenROAD PPA is a separate experiment.'},indent=2))


if __name__=='__main__':main()
