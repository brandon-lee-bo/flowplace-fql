"""Compare successive critics on the same measured action cohorts.

No placement, reward or policy is modified. Ridge penalties use only grouped
TRAIN cross-validation. Evaluation cohorts are diagnostics, never model
selection or new replay. Mark later in-sample collection rows explicitly.
"""
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
from .graph_o2o_models import FeatureCritics
from .physical_critic_diagnostic import physical_features, choose_penalty, fit_ridge, predict


def metrics(prediction, reward):
    prediction = np.asarray(prediction)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    error = prediction.mean(-1) - reward
    return dict(critic_diagnostics(prediction, reward),
                optimism_bias=float(error.mean()),
                overestimated_count=int((error > 0).sum()))


def rows_for_fit(exp, seed, number, arm, data, scale, cfg):
    x = data['train']['raw_action'].copy()
    y = data['train']['reward'].copy()
    groups = ['offline_family_'+str(g) for g in data['train']['group_id']]
    used = set()
    if number:
        replay_path = exp/f'seed_{seed}/replay_round_{number}.json'
        replay = json.loads(replay_path.read_text())
        if replay['config_sha256'] != sha(exp/'config.json') or replay['reward_scale'] != scale:
            raise ValueError('Replay provenance changed')
        online = replay['arms'][arm]
        for row in online:
            path = Path(row['outcome_path'])
            outcome = json.loads(path.read_text())
            request = json.loads((path.parent/'request.json').read_text())
            audited = audit_outcome(request, cfg, scale)
            if (sha(path) != row['outcome_sha256'] or outcome['role'] != 'collect'
                    or outcome['arm'] != arm or not outcome['training_feedback_eligible']
                    or audited['raw_action'] != row['raw_action'] or audited['reward'] != row['reward']):
                raise ValueError('Unbound or ineligible online fit row')
            used.add(str(path))
        if len(used) != number*cfg['online_samples']:
            raise ValueError('Incorrect actual replay budget')
        x = np.concatenate([x, np.asarray([r['raw_action'] for r in online]).reshape((-1, *x.shape[1:]))])
        y = np.concatenate([y, np.asarray([r['reward'] for r in online])])
        groups += ['online_'+r['run_id'] for r in online]
    return x, y, np.asarray(groups), used


def run(args):
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    exp = Path(args.experiment).resolve()
    cfg = json.loads((exp/'config.json').read_text())
    if cfg['design'] != 'bp_fe':
        raise ValueError('This diagnostic is bound to the current bp_fe pilot')
    last_round = cfg['online_rounds'] if args.through_round is None else args.through_round
    if not 0 <= last_round <= cfg['online_rounds']:
        raise ValueError('Requested diagnostic round is outside the frozen experiment')
    data, manifest, _ = load_train_validation(ROOT/'datasets/bp_fe_grt_v1/dataset.npz')
    cache_path = ROOT/'baseline/runtime/benchmark_cache/bp_fe.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    if sha(cache_path) != manifest['source_proxy_manifest']['provenance']['cache_sha256']:
        raise ValueError('Graph/cache differs from measured TRAIN data')
    macro_metric = MacroHPWL(cache)
    cohorts = {'validation': dict(x=data['validation']['raw_action'], y=data['validation']['reward'],
                                  paths=[], failures=0, role='validation')}
    outcome_hashes = {}
    for number in range(last_round+1):
        requests, _ = load_requests(exp, args.seed, number)
        for request in requests:
            row = audit_outcome(request, cfg, manifest['reward_scale_train_only'])
            key = f"{row['arm']}/round_{number}/{row['role']}"
            cohort = cohorts.setdefault(key, dict(x=[], y=[], paths=[], failures=0, role=row['role']))
            cohort['x'].append(row['raw_action']);cohort['y'].append(row['reward'])
            cohort['paths'].append(row['outcome_path'])
            cohort['failures'] += row['status'] != 'success'
            outcome_hashes[row['outcome_path']] = row['outcome_sha256']
    folder = exp/'q_transfer'/f'seed_{args.seed}_job_{os.environ["SLURM_JOB_ID"]}'
    folder.mkdir(parents=True, exist_ok=False)
    for name in ['graph_o2o_q_transfer.py', 'graph_o2o_models.py', 'physical_critic_diagnostic.py']:
        shutil.copyfile(Path(__file__).with_name(name), folder/name)
    for cohort in cohorts.values():
        cohort['x'] = np.asarray(cohort['x'], dtype=np.float32)
        cohort['y'] = np.asarray(cohort['y'], dtype=np.float64)
        cohort['physical'], _ = physical_features(cohort['x'], macro_metric)
    report = dict(config_sha256=sha(exp/'config.json'), dataset_sha256=manifest['dataset_sha256'],
                  cache_sha256=sha(cache_path), outcome_sha256=outcome_hashes,
                  seed=args.seed, slurm_job_id=os.environ['SLURM_JOB_ID'],
                  through_round=last_round,
                  source_sha256={name: sha(Path(__file__).with_name(name)) for name in
                      ['graph_o2o_q_transfer.py','graph_o2o_models.py','physical_critic_diagnostic.py',
                       'physical_ridge.py','critic_gate.py','core.py']},
                  scope='Fixed bp_fe diagnostic; not a learned-policy or independent-seed comparison',
                  selection='Ridge penalties by five-fold TRAIN-family CV only; no evaluation selection',
                  heldout_labels_used=False, evaluation_feedback_replayed=False, fits={})
    table, predictions = [], []
    for number in range(last_round+1):
        for arm in ['bc', 'fql']:
            key = f'{arm}/round_{number}'
            x, y, groups, used = rows_for_fit(exp, args.seed, number, arm, data,
                                            manifest['reward_scale_train_only'], cfg)
            physical, _ = physical_features(x, macro_metric)
            checkpoint = exp/f'seed_{args.seed}/round_{number}/{arm}/checkpoint.pt'
            completion_path = checkpoint.with_name('completed.json')
            completion = json.loads(completion_path.read_text())
            if sha(checkpoint) != completion['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after completion')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            critic = FeatureCritics(cache,
                torch.as_tensor(data['train']['raw_action']).reshape(-1, len(saved['critic']['movable']), 2),
                torch.as_tensor(data['train']['reward']), saved['critic']['movable'], cfg['critic_width'])
            critic.load_state_dict(saved['critic']);critic.eval()
            del saved
            with torch.no_grad():
                check = critic(torch.as_tensor(x, dtype=torch.float32).reshape(-1,len(critic.movable),2)).numpy()
            reproduced = metrics(check, y)
            if not np.isclose(reproduced['mae'], completion['critic_train']['mae'], atol=2e-6, rtol=2e-3):
                raise ValueError('CPU recomputation does not reproduce training-critic evidence')
            fit_report = dict(train_rows=len(y), independent_groups=len(np.unique(groups)),
                              checkpoint_sha256=sha(checkpoint), replayed_outcomes=sorted(used),
                              actual_neural_train_metrics=reproduced, alternatives={})
            report['fits'][key] = fit_report
            alternatives = {}
            for name, features in [('ridge_physical',physical),
                                   ('ridge_physical_coordinates',np.column_stack([physical,x.reshape(len(x),-1)]))]:
                penalty, cv, folds = choose_penalty(features,y,groups,[.001,.01,.1,1.,10.,100.])
                model = fit_ridge(features,y,penalty)
                fit_report['alternatives'][name] = dict(selected_penalty=penalty, train_group_cv=cv,
                    train=metrics(predict(model,features), y))
                alternatives[name] = model
                np.savez_compressed(folder/(key.replace('/','_')+'_'+name+'.npz'), **model, folds=folds)
            for cohort_name, cohort in cohorts.items():
                with torch.no_grad():
                    neural = critic(torch.as_tensor(cohort['x'],dtype=torch.float32).reshape(-1,len(critic.movable),2)).numpy()
                values = dict(neural=neural,train_mean=np.full((len(cohort['y']),1),y.mean()))
                for name, model in alternatives.items():
                    features = cohort['physical'] if name=='ridge_physical' else np.column_stack(
                        [cohort['physical'],cohort['x'].reshape(len(cohort['x']),-1)])
                    values[name] = predict(model,features)[:,None]
                in_sample = sum(path in used for path in cohort['paths'])
                if in_sample and in_sample != len(cohort['paths']):
                    raise ValueError('Mixed training/non-training cohort; split before assessing generalization')
                for model_name, value in values.items():
                    score = metrics(value, cohort['y'])
                    table.append(dict(fitted_critic=key,model=model_name,cohort=cohort_name,
                        count=len(cohort['y']),failure_count=cohort['failures'],used_for_fit=bool(in_sample),
                        **{k:score[k] for k in ['mae','rmse','spearman_r','optimism_bias','overestimated_count']}))
                    for i,(pred,target) in enumerate(zip(np.asarray(value).mean(-1),cohort['y'])):
                        predictions.append(dict(fitted_critic=key,model=model_name,cohort=cohort_name,index=i,
                            prediction=float(pred),target=float(target),used_for_fit=bool(in_sample)))
            del critic
            atomic_json(folder/'evidence.json', report)
    for name, rows in [('cohort_metrics.csv',table),('per_action_predictions.csv',predictions)]:
        with (folder/name).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Keep the same final held-out-of-replay policy actions fixed across critics.
    lines=['# 固定布局上的 Q 迁移诊断','',
        f'所有模型均评估同一组 round={last_round} 的 FQL 评估布局。这些布局及验证布局从未进入训练；不据此挑选策略。',
        'Ridge 的正则强度仅由 TRAIN 家族交叉验证决定。表中 reward 误差越小越好，正偏差表示过于乐观。','',
        '| 拟合数据 | Q 模型 | MAE | 排序相关 | 乐观偏差 |','| --- | --- | --- | --- | --- |']
    final_cohort=f'fql/round_{last_round}/evaluation'
    for row in table:
        if row['fitted_critic'].startswith('fql/') and row['cohort']==final_cohort:
            rank='—' if row['spearman_r'] is None else f'{row["spearman_r"]:.4f}'
            lines.append(f"| {row['fitted_critic']} | {row['model']} | {row['mae']:.6f} | {rank} | {row['optimism_bias']:.6f} |")
    lines += ['', '四个评估噪声仍只来自一个训练 seed。其他全部布局组、训练内标记和逐动作预测见 CSV。',
              '这是 Q 的诊断，不证明替换 critic 后策略会改善，也不使用 ridge 预测充当物理标签。','']
    (folder/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='completed',folder=str(folder),policy_changed=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--through-round',type=int)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm')
    run(args)
