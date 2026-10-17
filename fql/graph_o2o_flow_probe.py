"""Separate BC-flow generation quality from its one-step distillation error.

Use the four already fixed evaluation noises and frozen offline flow weights.
No training, Q guidance, selection or new feedback to either policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import load_problem, legalize, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_models import GraphPolicy
from .graph_o2o_archive import atomic_json, sha


def main(args):
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    experiment=Path(args.experiment).resolve()
    cfg=json.loads((experiment/'config.json').read_text())
    source=Path(cfg['source_experiment'])/'seed_0/round_0'
    for arm in ['bc','fql']:
        if sha(source/arm/'checkpoint.pt')!=cfg['checkpoint_sha256'][arm]:
            raise ValueError('Frozen source checkpoint changed')
    saved=torch.load(source/'bc/checkpoint.pt',map_location='cpu',weights_only=False)
    other=torch.load(source/'fql/checkpoint.pt',map_location='cpu',weights_only=False)
    if any(not torch.equal(value,other['flow'][key]) for key,value in saved['flow'].items()):
        raise ValueError('Offline BC and FQL flows differ; a shared flow control is invalid')
    del other
    problem=load_problem('bp_fe','cuda')
    flow=GraphPolicy(problem);flow.load_state_dict(saved['flow']);flow.eval()
    del saved
    metric=MacroHPWL(problem.benchmark_info)
    folder=experiment/'seed_0/round_0/requests';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():
        raise FileExistsError('Probe already exported; inspect existing scheduler state instead of duplicating')
    snapshot=experiment/('source_job_'+os.environ['SLURM_JOB_ID']);snapshot.mkdir(exist_ok=False)
    shutil.copyfile(__file__,snapshot/Path(__file__).name)
    records=[]
    for i in cfg['evaluation_indices']:
        original=source/f'requests/bc/evaluation/{i}'
        old=json.loads((original/'request.json').read_text())
        with np.load(original/'actions.npz',allow_pickle=False) as arrays:
            noise=torch.as_tensor(arrays['noise'].copy(),device='cuda')
        with torch.no_grad():
            raw=flow.sample_flow(noise[None],cfg['flow_steps'])[0]
            action=legalize(raw,problem)
        geometry=legality(action.cpu().numpy(),metric.sizes)
        dest=folder/'bc_flow'/'evaluation'/str(i);dest.mkdir(parents=True,exist_ok=False)
        values=action.cpu().numpy()
        record=dict(arm='bc_flow',seed=0,round=0,role='evaluation',index=i,noise_seed=old['noise_seed'],
            graph='bp_fe',terminal_mask=0,raw_action=raw.cpu().tolist(),executed_action=values.tolist(),
            predicted_q=None,source_checkpoint=str(source/'bc/checkpoint.pt'),checkpoint_sha256=cfg['checkpoint_sha256']['bc'],
            geometry=geometry,action_sha256=hashlib.sha256(np.asarray(values,dtype='<f4').reshape(-1).tobytes()).hexdigest(),
            sample_dir=str(dest),run_id=f'go2o_flowprobe_v1_s0_bcflow_evaluation_{i}',stage='global_route_only',
            training_feedback_eligible=False,macro_hpwl=float(metric(values)),
            source_noise_artifact=str(original/'actions.npz'),source_noise_sha256=sha(original/'actions.npz'),
            mechanism='Frozen shared offline BC flow, ten Euler steps, then external legalization; no one-step actor or Q guidance')
        np.savez_compressed(dest/'actions.npz',noise=noise.cpu().numpy(),raw_action=raw.cpu().numpy(),action=values)
        if geometry['legal']:
            save_sample(dest,action,noise,problem,record)
            path=dest/'bp_fe.def';audit=convert('bp_fe',dest/'macro_physical.json',path)
            record.update(def_path=str(path),def_sha256=sha(path),rounded_geometry=audit['legal_no_halo'])
            if not audit['legal_no_halo']:record['pre_backend_failure']='rounded_DEF_geometry'
        else:
            record['pre_backend_failure']='normalized_geometry'
        atomic_json(dest/'request.json',record);records.append(record)
    path=folder/'tasks.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    atomic_json(folder/'completed.json',dict(status='completed',tasks_sha256=sha(path),count=len(records),
        config_sha256=sha(experiment/'config.json'),flow_weights_equal_between_offline_arms=True,
        all_outcomes_required=True,selection='All four original paired evaluation noises, no feedback to training',
        slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status='exported',count=len(records),all_input_legal=all(r['geometry']['legal'] for r in records))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation')
    main(args)
