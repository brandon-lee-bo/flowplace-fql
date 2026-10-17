"""Compare saved critics on identical, never-replayed evaluation layouts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .core import critic_diagnostics
from .critic_gate import load_train_validation
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_models import FeatureCritics


def score_checkpoint(checkpoint, cfg, data, cache, online, cohorts):
    checkpoint = Path(checkpoint)
    completion_path = checkpoint.with_name('completed.json')
    completion = json.loads(completion_path.read_text())
    digest = sha(checkpoint)
    if digest != completion['checkpoint_sha256']:
        raise ValueError('Critic checkpoint changed after completion')
    used = {row['outcome_path'] for row in online}
    if len(used) != len(online):
        raise ValueError('Repeated training feedback in the fixed-cohort audit')
    x = data['train']['raw_action']
    y = data['train']['reward']
    if online:
        x = np.concatenate([x, np.asarray([r['raw_action'] for r in online]).reshape(-1, *x.shape[1:])])
        y = np.concatenate([y, np.asarray([r['reward'] for r in online])])
    if len(online) != completion['online_training_rows']:
        raise ValueError('Audit training rows differ from the actual saved fit')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    movable = saved['critic']['movable']
    critic = FeatureCritics(cache,
        torch.as_tensor(data['train']['raw_action']).reshape(-1, len(movable), 2),
        torch.as_tensor(data['train']['reward']), movable, cfg['critic_width'])
    critic.load_state_dict(saved['critic'])
    critic.eval()
    del saved

    def predict(actions):
        values = torch.as_tensor(np.asarray(actions), dtype=torch.float32).reshape(-1, len(movable), 2)
        with torch.no_grad():
            return critic(values).numpy()

    training = critic_diagnostics(predict(x), y)
    if not np.isclose(training['mae'], completion['critic_train']['mae'], atol=2e-6, rtol=2e-3):
        raise ValueError('CPU critic recomputation does not reproduce its saved training MAE')
    scores, predictions = {}, []
    for name, rows in cohorts.items():
        if not rows or any(r['role'] != 'evaluation' or r['training_feedback_eligible'] or
                           r['outcome_path'] in used for r in rows):
            raise ValueError('Fixed-cohort evaluation is empty or contaminated by replay')
        pred = predict([r['raw_action'] for r in rows])
        target = np.asarray([r['reward'] for r in rows])
        errors = pred.mean(-1) - target
        scores[name] = dict(critic_diagnostics(pred, target), optimism_bias=float(errors.mean()),
            failed_physical_cases=sum(r['status'] != 'success' for r in rows), used_for_fit=False)
        for row, value in zip(rows, pred):
            predictions.append(dict(cohort=name, index=row['index'], run_id=row['run_id'],
                status=row['status'], prediction=float(value.mean()), target=row['reward'],
                outcome_path=row['outcome_path'], outcome_sha256=row['outcome_sha256']))
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
        completion_sha256=sha(completion_path), training_rows=len(y), online_rows=len(online),
        reproduced_training_metrics=training, scores=scores), predictions


def fixed_cohort_report(exp, cfg, feedback, cohorts, output):
    root = Path(__file__).resolve().parents[1]
    data, manifest, _ = load_train_validation(root/'datasets/bp_fe_grt_v1/dataset.npz')
    cache_path = root/'baseline/runtime/benchmark_cache/bp_fe.pt'
    if (manifest['dataset_sha256'] != cfg['offline_dataset_sha256'] or
            sha(cache_path) != manifest['source_proxy_manifest']['provenance']['cache_sha256']):
        raise ValueError('Fixed-cohort critic audit inputs differ from measured TRAIN')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    fits = {}
    for arm in ['bc', 'fql']:
        fits['before/'+arm] = (cfg['parent_checkpoints'][arm]['path'], feedback['arms'][arm][:16])
    for arm in ['bc', 'fql', 'fql_frozen_replay']:
        fits['after/'+arm] = (exp/f'seed_0/round_1/{arm}/checkpoint.pt', feedback['arms'][arm])
    records, per_action, table = {}, [], []
    for name, (checkpoint, replay) in fits.items():
        score, values = score_checkpoint(checkpoint, cfg, data, cache, replay, cohorts)
        records[name] = score
        per_action += [dict(fitted_critic=name, **r) for r in values]
        table += [dict(fitted_critic=name, cohort=c, **v) for c, v in score['scores'].items()]
    report = dict(config_sha256=sha(exp/'config.json'), dataset_sha256=manifest['dataset_sha256'],
        cache_sha256=sha(cache_path), source_sha256=sha(__file__), fits=records,
        purpose='Compare different critics on exactly the same measured actions; no policy selection',
        evaluation_feedback_replayed=False, heldout_labels_used=False)
    atomic_json(output/'fixed_cohort_q.json', report)
    for name, rows in [('fixed_cohort_q.csv', table), ('fixed_cohort_q_predictions.csv', per_action)]:
        with (output/name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report
