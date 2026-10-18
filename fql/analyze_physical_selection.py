"""Post-outcome diagnostics, never used for model/candidate selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL
from .critic_gate import load_train_validation
from .core import critic_diagnostics, write_json
from .physical_critic_diagnostic import physical_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregate',required=True,type=Path)
    p.add_argument('--tasks',required=True,type=Path)
    p.add_argument('--critic',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm')
    torch.set_num_threads(1)
    b=json.loads(a.aggregate.read_text())
    if b['status'] not in ['completed','completed_with_physical_failures']:
        raise ValueError('Independently verified aggregate required')
    tasks=[json.loads(x) for x in a.tasks.read_text().splitlines() if x.strip()]
    lookup={(t['method'],t['index']):t for t in tasks}
    data,manifest,_=load_train_validation('datasets/bp_fe_grt_v1/dataset.npz')
    cache=torch.load('baseline/runtime/benchmark_cache/bp_fe.pt',map_location='cpu',weights_only=False)
    metric=MacroHPWL(cache)
    model=np.load(a.critic,allow_pickle=False)
    train_f,_=physical_features(data['train']['raw_action'],metric)
    train_z=(train_f-model['mean'])/model['std']
    rows=[]
    for record in b['records']:
        t=lookup[(record['method'],record['index'])]
        if sha(t['action_path'])!=t['action_file_sha256']:
            raise ValueError('Frozen action source changed')
        with np.load(t['action_path'],allow_pickle=False) as ar:
            raw,executed=ar['raw_action'],ar['action']
        feature,_=physical_features(raw,metric)
        z=(feature-model['mean'])/model['std']
        calculated=float((z@model['coeff']+model['intercept'])[0])
        if not np.isclose(calculated,float(t['predicted_q']),atol=2e-6,rtol=0):
            raise ValueError('Saved actor Q differs from independent feature implementation')
        row={k:record[k] for k in ['method','configuration','training_seed','index','status']}
        row.update(predicted_q=float(t['predicted_q']),
            predicted_rwl_um=-float(t['predicted_q'])*manifest['reward_scale_train_only'],
            projection_mse=float(np.square(raw-executed).mean()),
            min_train_standardized_feature_distance=float(np.linalg.norm(train_z-z,axis=1).min()),
            standardized_features_outside_train_range=int(((z[0]<train_z.min(0))|(z[0]>train_z.max(0))).sum()),
            raw_feature=feature[0].tolist())
        if record['status']=='success':
            row['measured_rwl_um']=record['values']['rWL_um']
            row['measured_reward']=-row['measured_rwl_um']/manifest['reward_scale_train_only']
            row['q_prediction_minus_measured_reward']=row['predicted_q']-row['measured_reward']
        rows.append(row)
    calibration={}
    for method in ['bc_continued','fql_physical_alpha_1']:
        selected=[r for r in rows if r['configuration']==method]
        calibration[method]=critic_diagnostics(
            np.asarray([[r['predicted_q']] for r in selected]),
            np.asarray([r['measured_reward'] for r in selected])) if all(r['status']=='success' for r in selected) else None
    report=dict(status='completed',slurm_job_id=os.environ['SLURM_JOB_ID'],rows=rows,
        calibration=calibration,source_hashes={str(p):sha(p) for p in [a.aggregate,a.tasks,a.critic,Path(__file__)]},
        scope='Post-outcome, single-circuit exploratory diagnostic; does not select any candidate or parameter.',
        limits=['Feature distances are descriptive, not calibrated epistemic uncertainty.',
                'Only 3 training seeds and 2 common noise families; no independent final test.',
                'Projection MSE describes raw-to-executed change, not a causal attribution.'])
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,5))
    for name,label in [('bc_continued','BC'),('fql_physical_alpha_1','FQL-style')]:
        selected=[r for r in rows if r['configuration']==name and r['status']=='success']
        ax.scatter([r['predicted_rwl_um']/1e6 for r in selected],
            [r['measured_rwl_um']/1e6 for r in selected],label=label)
    success=[r for r in rows if r['status']=='success']
    if success:
        bounds=[min(min(r['predicted_rwl_um'],r['measured_rwl_um']) for r in success)/1e6,
                max(max(r['predicted_rwl_um'],r['measured_rwl_um']) for r in success)/1e6]
        ax.plot(bounds,bounds,'k--',alpha=.5)
    ax.set_xlabel('Q-predicted global-route wirelength (M µm)')
    ax.set_ylabel('Measured global-route wirelength (M µm)')
    ax.set_title('Fixed candidates; no Q-based selection');ax.legend()
    fig.tight_layout();fig.savefig(a.output/'actor_q_calibration.png',dpi=160);plt.close(fig)
    print(json.dumps({'output':str(a.output),'calibration':calibration}),flush=True)


if __name__=='__main__':main()
