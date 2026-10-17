"""Diagnose a regularized critic before changing any placement policy."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from analysis.placement_metrics import MacroHPWL
from .critic_gate import load_train_validation
from .core import critic_diagnostics
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT, load_requests
from .graph_o2o_collect import audit_outcome
from .kernel_critic import fit, predict, select, KernelCritic
from .physical_critic_diagnostic import physical_features


def features(x, metric):
    physical, _ = physical_features(x, metric)
    raw = x.reshape(len(x), -1)
    return dict(physical=physical, coordinates=raw, physical_coordinates=np.column_stack([physical, raw]))


def main(args):
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    exp = args.experiment.resolve()
    cfg = json.loads((exp/'config.json').read_text())
    data, manifest, _ = load_train_validation(ROOT/'datasets/bp_fe_grt_v1/dataset.npz')
    cache_path = ROOT/'baseline/runtime/benchmark_cache/bp_fe.pt'
    if sha(cache_path) != manifest['source_proxy_manifest']['provenance']['cache_sha256']:
        raise ValueError('TRAIN data and graph cache differ')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    metric = MacroHPWL(cache)
    key = 'raw_action' if args.coordinates=='raw' else 'action'
    online_key = 'raw_action' if args.coordinates=='raw' else 'executed_action'
    folder = args.output/('job_'+os.environ['SLURM_JOB_ID'])
    folder.mkdir(parents=True, exist_ok=False)
    for name in ['kernel_critic.py','graph_o2o_kernel_diagnostic.py','physical_critic_diagnostic.py']:
        shutil.copyfile(Path(__file__).with_name(name), folder/name)
    cohorts = dict(validation=dict(x=data['validation'][key], y=data['validation']['reward'], paths=[]))
    for number in range(3):
        requests, _ = load_requests(exp, 0, number)
        for arm in ['bc','fql']:
            rows = [audit_outcome(r,cfg,manifest['reward_scale_train_only']) for r in requests
                    if r['arm']==arm and r['role']=='evaluation']
            if len(rows)!=4 or any(r['training_feedback_eligible'] for r in rows):
                raise ValueError('Changed evaluation denominator or replay eligibility')
            cohorts[f'{arm}/round_{number}'] = dict(x=np.asarray([r[online_key] for r in rows]),
                y=np.asarray([r['reward'] for r in rows]), paths=[r['outcome_path'] for r in rows])
    evidence = dict(config_sha256=sha(exp/'config.json'), dataset_sha256=manifest['dataset_sha256'],
        cache_sha256=sha(cache_path), selection='Five-fold TRAIN-family CV MAE only; fixed candidate grid',
        heldout_labels_used=False, policies_changed=False,
        feature_coordinates=args.coordinates,
        deployment='Executed-coordinate models are legalize-then-score only; coordinate derivatives do not differentiate the solver',
        uncertainty='Model posterior standard deviation, not a guaranteed physical error bound',
        slurm_job_id=os.environ['SLURM_JOB_ID'], fits={})
    table, per_action = [], []
    for number in [0,2]:
        train = data['train']
        x, y = train[key].copy(), train['reward'].astype(float)
        families = ['offline_'+str(g) for g in train['group_id']]
        used = set()
        if number:
            path = exp/'seed_0/replay_round_2.json'
            replay = json.loads(path.read_text())
            if replay['config_sha256'] != sha(exp/'config.json'):
                raise ValueError('Changed parent replay')
            rows = replay['arms']['fql']
            if len(rows)!=16:
                raise ValueError('Changed actual online budget')
            for row in rows:
                request = json.loads(Path(row['outcome_path']).with_name('request.json').read_text())
                audited = audit_outcome(request,cfg,manifest['reward_scale_train_only'])
                if audited['outcome_sha256']!=row['outcome_sha256'] or not audited['training_feedback_eligible']:
                    raise ValueError('Unbound replay row')
                used.add(row['outcome_path'])
            x = np.concatenate([x,np.asarray([r[online_key] for r in rows]).reshape(-1,*x.shape[1:])])
            y = np.concatenate([y,np.asarray([r['reward'] for r in rows])])
            families += ['online_'+row['run_id'] for row in rows]
        selected, model, candidates, folds = select(features(x,metric),y,np.asarray(families))
        name = 'fql/round_'+str(number)
        np.savez_compressed(folder/f'round_{number}_kernel.npz', **model, folds=folds)
        differentiable = KernelCritic(cache,model,selected['features'])
        expected, _ = predict(model,features(x,metric)[selected['features']])
        action = torch.as_tensor(x,dtype=torch.float64).reshape(len(x),-1,2).requires_grad_()
        prediction = differentiable(action)[:,0]
        if not np.allclose(prediction.detach().numpy(),expected,atol=2e-6,rtol=2e-6):
            raise ValueError('Differentiable critic disagrees with its fitted kernel')
        gradient = torch.autograd.grad(prediction.sum(),action)[0]
        if not torch.isfinite(gradient).all() or not bool(gradient.abs().max()>0):
            raise ValueError('Kernel critic has no valid action gradient')
        evidence['fits'][name] = dict(train_rows=len(y),independent_families=len(set(families)),
            selected=selected,all_train_cv_candidates=candidates,model_sha256=sha(folder/f'round_{number}_kernel.npz'),
            numpy_torch_predictions_match=True,finite_nonzero_action_gradients=True,
            maximum_absolute_action_gradient=float(gradient.abs().max()))
        for cohort_name, cohort in cohorts.items():
            if used & set(cohort['paths']):
                raise ValueError('Evaluation was replayed')
            f = features(cohort['x'],metric)[selected['features']]
            mean, std = predict(model,f)
            for kind, pred in [('kernel_mean',mean),('kernel_mean_minus_std',mean-std),
                               ('train_mean',np.full(len(mean),y.mean()))]:
                score = dict(critic_diagnostics(pred[:,None],cohort['y']),
                    optimism_bias=float((pred-cohort['y']).mean()),
                    average_model_std=float(std.mean()),
                    within_one_model_std_fraction=float((np.abs(mean-cohort['y'])<=std).mean()))
                table.append(dict(fitted_data=name,model=kind,cohort=cohort_name,**score))
                for i,(a,b,c) in enumerate(zip(pred,cohort['y'],std)):
                    per_action.append(dict(fitted_data=name,model=kind,cohort=cohort_name,index=i,
                                           prediction=float(a),target=float(b),model_std=float(c)))
        atomic_json(folder/'evidence.json',evidence)
    for name, rows in [('cohort_metrics.csv',table),('predictions.csv',per_action)]:
        with (folder/name).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 受约束 Q 的训练数据对照','',
        '核模型的特征、长度尺度和正则强度仅由训练数据的家族交叉验证确定；策略未改变。',
        '输入坐标：'+args.coordinates+'。executed 只表示合法化后评分，不宣称能通过求解器求导。',
        '以下两种拟合都评估同一批第二轮 FQL 布局。模型标准差不是实际误差保证。','',
        '| 拟合数据 | 模型 | MAE | 乐观偏差 | 排序相关 |',
        '| --- | --- | --- | --- | --- |']
    for row in table:
        if row['cohort']!='fql/round_2':continue
        rank='—'if row['spearman_r']is None else f"{row['spearman_r']:.4f}"
        lines.append(f"| {row['fitted_data']} | {row['model']} | {row['mae']:.6f} | {row['optimism_bias']:.6f} | {rank} |")
    lines += ['', '全部验证与固定布局组保存在 CSV。正负结果均保留；本诊断不证明接入策略后会改善。','']
    (folder/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='completed',folder=str(folder),policy_changed=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coordinates',choices=['raw','executed'],default='raw')
    args=parser.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Use Slurm')
    main(args)
