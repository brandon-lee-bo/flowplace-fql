"""Predeclare a budgeted, family-stratified backend labeling set before outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL,legality
from baseline.api import load_problem,save_sample

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--train-families',type=int,default=32)
    p.add_argument('--val-families',type=int,default=8)
    p.add_argument('--test-families',type=int,default=8)
    p.add_argument('--variants',default='0,1')
    p.add_argument('--prefix',default='bp_fe_grt_v1')
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm for dataset preparation')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    ds=Path(args.dataset_dir).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    if Path(args.prefix).name!=args.prefix:raise ValueError('Prefix must be one path component')
    meta=json.loads((ds/'manifest.json').read_text());design=meta['args']['design']
    problem=load_problem(design,'cpu');metric=MacroHPWL(problem.benchmark_info)
    with np.load(ds/'dataset.npz') as f:data={k:f[k] for k in f.files}
    family_ids=[]
    for split,count in enumerate([args.train_families,args.val_families,args.test_families]):
        available=np.flatnonzero(data['split']==split)
        if count<1 or count>len(available):raise ValueError('Invalid family budget')
        family_ids.extend(available[:count].tolist())
    variants=[int(x) for x in args.variants.split(',')]
    rows=np.flatnonzero(np.isin(data['family_id'],family_ids)&np.isin(data['variant'],variants))
    if len(rows)!=len(family_ids)*len(variants):raise ValueError('Requested variant missing for some family')
    plan={'args':vars(args),'dataset':str(ds/'dataset.npz'),
          'dataset_sha256':hashlib.sha256((ds/'dataset.npz').read_bytes()).hexdigest(),
          'selected_rows':rows.tolist(),'selected_families':family_ids,
          'selection':'First predeclared families per split and fixed variants; no reward selection',
          'backend_stage':'global_route_only','not_final_table2':True,
          'objective':'Minimize global-route wirelength; other PPA metrics reported separately',
          'slurm_job_id':os.environ['SLURM_JOB_ID']}
    (out/'selection_plan.json').write_text(json.dumps(plan,indent=2))
    tasks=[]
    for row in rows:
        family=int(data['family_id'][row]);shape=(len(metric.names),2)
        x=data['action'][row].reshape(shape)
        if not legality(x,metric.sizes)['legal']:
            raise RuntimeError(f'Selected row {row} is illegal; preserve plan and handle explicitly')
        folder=out/'rows'/f'{row:04d}'
        save_sample(folder,torch.as_tensor(x),torch.as_tensor(data['noise'][family].reshape(shape)),problem,
                    {'row_index':int(row),'family':family,'variant':int(data['variant'][row]),
                     'split':int(data['critic_split'][row]),'selection':'predeclared family budget'})
        def_path=folder/f'{design}.def'
        command=[sys.executable,str(ROOT/'backend/scripts/export_macro_def.py'),
                 '--design',design,'--coordinates',str(folder/'macro_physical.json'),'--output',str(def_path)]
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        run_id=f'{args.prefix}_row{row:04d}'
        tasks.append({'row_index':int(row),'family':family,'split':int(data['critic_split'][row]),
                      'variant':int(data['variant'][row]),'design':design,'def_path':str(def_path),
                      'dataset_sha256':plan['dataset_sha256'],
                      'action_sha256':hashlib.sha256(np.ascontiguousarray(data['action'][row],dtype='<f4').tobytes()).hexdigest(),
                      'def_sha256':hashlib.sha256(def_path.read_bytes()).hexdigest(),
                      'run_id':run_id,'metrics_path':str(ROOT/'backend/results'/run_id/'metrics.json')})
    (out/'tasks.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in tasks))
    print(json.dumps({'tasks':len(tasks),'manifest':str(out/'tasks.jsonl'),'plan':str(out/'selection_plan.json')},indent=2))


if __name__=='__main__':main()
