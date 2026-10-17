"""Audit new-design behavior data before assigning any real backend reward.

This does not train RL or interpret the collector's macro-HPWL scores as PPA.
Only the first four predeclared TRAIN families are exported as import probes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL, legality
from baseline.api import CHECKPOINT, RUNTIME, load_problem, save_sample
from backend.scripts.export_macro_def import convert
from .graph_o2o_archive import atomic_json, sha


def run(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text())
    dataset=Path(plan['dataset']).resolve();output=Path(plan['audit_output']).resolve()
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((dataset/'manifest.json').read_text())
    provenance=json.loads((dataset/'collection_provenance.json').read_text())
    if (provenance['args']!=plan['collector_args'] or
            meta['provenance']!=provenance or
            meta['reward_kind']!='macro_hpwl'):
        raise ValueError('New-design behavior generation differs from the frozen plan')
    for key,path in [('checkpoint_sha256',CHECKPOINT),
                     ('cache_sha256',RUNTIME/'benchmark_cache'/f"{plan['design']}.pt")]:
        if sha(path)!=provenance[key] or provenance[key]!=plan[key]:
            raise ValueError('New-design cache/checkpoint identity changed')
    root=Path(__file__).resolve().parents[1]
    for name,digest in plan['source_sha256'].items():
        if sha(root/name)!=digest:
            raise ValueError('New-design source changed before data audit: '+name)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    problem=load_problem(plan['design'],'cpu')
    metric=MacroHPWL(problem.benchmark_info)
    if metric.names!=list(problem.macro_names):
        raise ValueError('Graph/cache macro identities differ')
    with np.load(dataset/'dataset.npz',allow_pickle=False)as source:
        arrays={key:source[key]for key in source.files}
    counts=[plan['collector_args'][key]for key in ['train','val','test']]
    n=sum(counts);variants=plan['collector_args']['perturbations']+1
    expected_split=np.repeat([0,1,2],counts)
    if (meta['num_teacher']!=n or meta['num_critic']!=n*variants or
            not np.array_equal(arrays['split'],expected_split) or
            not np.array_equal(arrays['critic_parent_index'],np.repeat(np.arange(n),variants)) or
            not np.array_equal(arrays['variant'],np.tile(np.arange(variants),n)) or
            not np.array_equal(arrays['critic_split'],expected_split[arrays['critic_parent_index']])):
        raise ValueError('Noise-family/variant/split accounting changed')
    dimensions=2*len(metric.names)
    for name,rows in [('noise',n),('teacher_action',n),('raw_action',n*variants),('action',n*variants)]:
        if arrays[name].shape!=(rows,dimensions)or not np.isfinite(arrays[name]).all():
            raise ValueError('Invalid finite coordinate matrix: '+name)
    family_sets=[set(arrays['critic_parent_index'][arrays['critic_split']==s].tolist())for s in range(3)]
    if any(family_sets[a]&family_sets[b]for a,b in [(0,1),(0,2),(1,2)]):
        raise ValueError('Train/validation/test families overlap')
    probes=[]
    for family in plan['import_probe_TRAIN_families']:
        if arrays['split'][family]!=0:
            raise ValueError('A fixed import probe is not TRAIN')
        row=family*variants
        action=arrays['action'][row].reshape(-1,2)
        raw=arrays['raw_action'][row].reshape(-1,2)
        geometry=legality(action,metric.sizes)
        if geometry['legal']!=bool(arrays['legal'][row]):
            raise ValueError('Independent normalized geometry disagrees with collection')
        folder=output/'import_probes'/f'family_{family:04d}'
        record=dict(source_row_index=row,source_family=family,variant=0,
            raw_action=raw.tolist(),executed_action=action.tolist(),geometry=geometry,
            source_dataset_sha256=sha(dataset/'dataset.npz'),real_backend_reward=None)
        path=folder/'probe.json'
        if not path.exists():
            folder.mkdir(parents=True,exist_ok=True)
            if geometry['legal']:
                save_sample(folder,torch.from_numpy(action),
                    torch.from_numpy(arrays['noise'][family].reshape(-1,2)),problem,record)
                def_path=folder/f"{plan['design']}.def"
                audit=convert(plan['design'],folder/'macro_physical.json',def_path)
                record.update(def_path=str(def_path),def_sha256=sha(def_path),
                              rounded_DEF_legal=audit['legal_no_halo'])
            atomic_json(path,record)
        else:
            old=json.loads(path.read_text())
            if any(old[key]!=value for key,value in record.items()):
                raise ValueError('A previously exported import probe changed')
            if old.get('def_path')and sha(old['def_path'])!=old['def_sha256']:
                raise ValueError('A previously exported probe DEF changed')
            record=old
        probes.append(dict(path=str(path),sha256=sha(path),record=record))
    snapshot=output/('source_job_'+os.environ['SLURM_JOB_ID'])
    snapshot.mkdir(exist_ok=False)
    shutil.copyfile(__file__,snapshot/Path(__file__).name)
    evidence=dict(status='behavior_data_audited_no_real_rewards_yet',
        plan_path=str(plan_path),plan_sha256=sha(plan_path),dataset_sha256=sha(dataset/'dataset.npz'),
        manifest_sha256=sha(dataset/'manifest.json'),provenance_sha256=sha(dataset/'collection_provenance.json'),
        design=plan['design'],teacher_families=n,TRAIN_families=counts[0],
        validation_families=counts[1],test_families=counts[2],critic_rows=n*variants,
        macro_count=len(metric.names),graph_edges=int(problem.cond.edge_index.shape[1]),
        normalized_legal_count=int(arrays['legal'].sum()),normalized_requested=n*variants,
        import_probes=probes,real_backend_rewards_collected=False,
        heldout_PPA_used=False,policy_trained=False,goal_completed=False,
        slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output/'evidence.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items()if k!='import_probes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    args=p.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Use Slurm for graph-data inspection/export')
    run(args)
