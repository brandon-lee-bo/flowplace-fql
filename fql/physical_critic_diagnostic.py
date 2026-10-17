"""TRAIN-group-CV ridge diagnostics on existing real GRT labels.

No heldout predictions or new backend jobs. Raw-coordinate models could become
action critics; executed-coordinate models require legalization before scoring
and do not provide a gradient through that operation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from .critic_gate import load_train_validation
from .core import critic_diagnostics, write_json
from .provenance import capture_source_provenance
from analysis.placement_metrics import MacroHPWL


def physical_features(actions, metric):
    x = np.asarray(actions, float).reshape(len(actions), -1, 2)
    area = metric.sizes.prod(axis=-1)
    centroid = (x * area[None, :, None]).sum(axis=1) / area.sum()
    span = np.ptp(x, axis=1)
    centered = x - centroid[:, None]
    radius = np.square(centered).sum(axis=-1)
    area_radius = (radius * area).sum(axis=-1) / area.sum()
    lo, hi = x - metric.sizes / 2, x + metric.sizes / 2
    gaps = np.minimum(lo + 1, 1 - hi)
    out_excess = np.maximum(-gaps, 0).sum(axis=(1, 2))
    pair_ids = np.triu_indices(len(metric.sizes), 1)
    delta = np.abs(x[:, pair_ids[0]] - x[:, pair_ids[1]])
    pair_dist = delta.sum(axis=-1)
    sep = delta - (metric.sizes[pair_ids[0]] + metric.sizes[pair_ids[1]]) / 2
    overlap = np.maximum(-sep, 0).prod(axis=-1)
    hpwl = metric(x) / metric.canvas.sum()
    # Continuous, area-weighted soft density on a 2x2 grid, not a congestion label.
    centers = np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])
    density = (np.exp(-np.square(x[:, :, None] - centers).sum(axis=-1) / .5)
               * area[None, :, None]).sum(axis=1) / area.sum()
    features = np.column_stack([hpwl, centroid, span, area_radius, gaps.mean(axis=(1,2)),
        gaps.min(axis=(1,2)), out_excess, pair_dist.mean(axis=1), pair_dist.std(axis=1),
        pair_dist.min(axis=1), overlap.sum(axis=1), density])
    names = ['macro_hpwl', 'area_centroid_x', 'area_centroid_y', 'span_x', 'span_y',
        'area_radius2', 'mean_boundary_gap', 'min_boundary_gap', 'boundary_excess',
        'mean_pair_l1', 'std_pair_l1', 'min_pair_l1', 'overlap_area',
        'soft_density_00','soft_density_01','soft_density_10','soft_density_11']
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite physical features')
    return features, names


def fit_ridge(x, y, penalty):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-10, std, 1.)
    z = (x - mean) / std
    y_mean = y.mean()
    # Mean squared error + penalty * coefficient L2; unpenalized intercept.
    coeff = np.linalg.solve(z.T @ z / len(x) + penalty * np.eye(x.shape[1]),
                            z.T @ (y - y_mean) / len(x))
    return {'mean': mean, 'std': std, 'coeff': coeff, 'intercept': float(y_mean)}


def predict(model, x):
    return (x - model['mean']) / model['std'] @ model['coeff'] + model['intercept']


def choose_penalty(x, y, groups, penalties):
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError('Need at least five independent TRAIN families')
    rng = np.random.default_rng(20261017)
    shuffled = rng.permutation(unique)
    fold_map = {group: i % 5 for i, group in enumerate(shuffled)}
    folds = np.array([fold_map[g] for g in groups])
    results = []
    for penalty in penalties:
        oof = np.empty(len(y))
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            if set(groups[train]) & set(groups[valid]):
                raise ValueError('Family leakage in CV')
            model = fit_ridge(x[train], y[train], penalty)
            oof[valid] = predict(model, x[valid])
        results.append({'penalty':penalty, 'group_cv':critic_diagnostics(oof[:,None], y)})
    best = min(results, key=lambda r:(r['group_cv']['mae'],r['penalty']))
    return best['penalty'], results, folds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='datasets/bp_fe_grt_v1/dataset.npz')
    p.add_argument('--cache', default='baseline/runtime/benchmark_cache/bp_fe.pt')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Submit through Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',1)))
    out = Path(args.output).resolve();out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = capture_source_provenance(out / 'source_snapshot',extra_paths={
        name:root/name for name in ['fql/physical_critic_diagnostic.py',
            'fql/physical_critic_diagnostic.slurm','analysis/placement_metrics.py']})
    data, manifest, _ = load_train_validation(args.dataset)
    cache_sha = hashlib.sha256(Path(args.cache).read_bytes()).hexdigest()
    if cache_sha != manifest['source_proxy_manifest']['provenance']['cache_sha256']:
        raise ValueError('Dataset/cache identity mismatch')
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    metric = MacroHPWL(cache)
    train, val = data['train'], data['validation']
    y, vy = train['reward'].astype(float), val['reward'].astype(float)
    train_f, val_f, names = {}, {}, {}
    for location, key in [('raw','raw_action'),('executed','action')]:
        a, fn = physical_features(train[key], metric)
        b, _ = physical_features(val[key], metric)
        train_f[location+'_hpwl'], val_f[location+'_hpwl'], names[location+'_hpwl'] = a[:,:1], b[:,:1], fn[:1]
        train_f[location+'_physical'], val_f[location+'_physical'], names[location+'_physical'] = a,b,fn
        train_f[location+'_coordinates'] = train[key].reshape(len(y),-1).astype(float)
        val_f[location+'_coordinates'] = val[key].reshape(len(vy),-1).astype(float)
        names[location+'_coordinates'] = ['coordinate_'+str(i) for i in range(train_f[location+'_coordinates'].shape[1])]
    constant = critic_diagnostics(np.full((len(vy),1),y.mean()),vy)
    report = dict(status='running', args=vars(args), slurm_job_id=os.environ['SLURM_JOB_ID'],
        source_provenance=source, cache_sha256=cache_sha, dataset_sha256=manifest['dataset_sha256'],
        heldout_evaluated=False, training_families=len(np.unique(train['group_id'])),
        train_rows=len(y), validation_rows=len(vy), constant_validation=constant,
        selection='Ridge penalty selected by five-fold TRAIN family CV, no validation model selection',
        deployment_note='Executed features are legalize-then-score only; they do not differentiate through legalization',
        reward_stage=manifest['reward_stage'], models={})
    write_json(out/'report.json',report)
    for name, features in train_f.items():
        penalty, cv, folds = choose_penalty(features,y,train['group_id'],[.001,.01,.1,1.,10.,100.])
        model = fit_ridge(features,y,penalty)
        pred, vp = predict(model,features),predict(model,val_f[name])
        vm = critic_diagnostics(vp[:,None],vy)
        result = dict(features=names[name], selected_penalty=penalty, train_cv=cv,
            train=critic_diagnostics(pred[:,None],y), validation=vm,
            beats_constant_validation_mae=vm['mae'] < constant['mae'],
            positive_validation_rank=vm['spearman_r'] is not None and vm['spearman_r']>0)
        report['models'][name] = result
        np.savez_compressed(out/(name+'.npz'),**model,train_prediction=pred,validation_prediction=vp,
            train_source_row=train['source_row_index'],validation_source_row=val['source_row_index'],train_cv_folds=folds)
        write_json(out/'report.json',report)
        print(json.dumps({'model':name,'penalty':penalty,'validation':vm}),flush=True)
    report.update(status='completed',automatic_next_action=None,limitations=[
        'Only one circuit and 32 TRAIN families, previously inspected validation families',
        'This diagnoses representation, not evidence of policy improvement or new-circuit generalization',
        'An executed-coordinate scorer cannot directly replace an actor-gradient raw-action critic',
        'No new backend evaluation or final heldout scoring was performed'])
    write_json(out/'report.json',report)


if __name__=='__main__':
    main()
