"""Bind every extra critic label to its frozen TRAIN action and real outcome.

The behavior flow keeps the original 256 targets. New labels affect critic
training and its TRAIN-only normalization; they are not online interaction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests, validate_request
from .graph_o2o_collect import audit_outcome
from .graph_o2o_offline_expand import sources


def audited_expansion(experiment, cfg):
    experiment = Path(experiment).resolve()
    declared = cfg['expanded_labels']
    source = Path(declared['experiment']).resolve()
    source, label_cfg, plan, arrays = sources(source)
    if (sha(source/'config.json') != declared['config_sha256'] or
            sha(source/'selection_plan.json') != declared['selection_plan_sha256'] or
            declared['requested'] != 320):
        raise ValueError('Expanded TRAIN source differs from the declared plan')
    replay_path = source/'report/offline_replay.json'
    evidence_path = source/'report/evidence.json'
    replay = json.loads(replay_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    scale = plan['reward_scale_frozen_original_TRAIN']
    if (evidence['status'] != 'all_requested_accounted' or evidence['requested'] != 320 or
            evidence['offline_replay_sha256'] != sha(replay_path) or
            evidence['selection_plan_sha256'] != sha(source/'selection_plan.json') or
            evidence['validation_heldout_used'] or evidence['online_feedback'] or
            replay['config_sha256'] != sha(source/'config.json') or
            replay['selection_plan_sha256'] != sha(source/'selection_plan.json') or
            replay['reward_scale'] != scale or replay['validation_heldout_used'] or
            replay['collection_kind'] != 'additional_offline_TRAIN_labels_not_on_policy'):
        raise ValueError('Additional labels lack a complete real-outcome audit')
    requests, task_path = load_requests(source, 0, 0)
    rows = replay['rows']
    if len(requests) != 320 or len(rows) != 320:
        raise ValueError('Expanded TRAIN denominator changed')
    bindings = []
    families = []
    for expected_source_row, request, row in zip(plan['selected_source_rows'], requests, rows):
        validate_request(request)
        outcome = audit_outcome(request, label_cfg, scale)
        if (request['source_row_index'] != expected_source_row or
                row['source_row_index'] != expected_source_row or
                row['source_noise_family'] != request['source_noise_family'] or
                row['source_variant'] != request['source_variant'] or
                not request['not_online_policy_feedback'] or
                request['data_collection_kind'] != 'offline_augmentation_of_frozen_TRAIN_behavior'):
            raise ValueError('A new label crossed its fixed TRAIN-family boundary')
        for field in ['raw_action', 'executed_action', 'reward', 'status', 'terminal_mask',
                      'outcome_path', 'outcome_sha256', 'action_sha256', 'metrics']:
            if row[field] != outcome[field]:
                raise ValueError('Additional replay differs from measured outcome: '+field)
        for name, key in [('raw_action', 'raw_action'), ('executed_action', 'action')]:
            value = np.asarray(row[name], dtype=np.float32).reshape(-1)
            if not np.array_equal(value, arrays[key][expected_source_row]):
                raise ValueError('An additional critic action differs from its frozen source')
        if row['terminal_mask'] != 0 or not np.isfinite(row['reward']):
            raise ValueError('Additional label is not a finite terminal reward')
        families.append(row['source_noise_family'])
        bindings.append(dict(source_row_index=expected_source_row,
            source_noise_family=row['source_noise_family'], source_variant=row['source_variant'],
            request_path=str(Path(request['sample_dir'])/'request.json'),
            request_sha256=outcome['request_sha256'], outcome_path=row['outcome_path'],
            outcome_sha256=row['outcome_sha256'], status=row['status']))
    if set(families) != set(range(32,192)):
        raise ValueError('Expanded TRAIN families are not exactly32..191')
    binding = dict(version=1, config_sha256=sha(experiment/'config.json'),
        measurement_config_sha256=sha(source/'config.json'),
        selection_plan_sha256=sha(source/'selection_plan.json'), tasks_sha256=sha(task_path),
        replay_path=str(replay_path), replay_sha256=sha(replay_path),
        evidence_path=str(evidence_path), evidence_sha256=sha(evidence_path),
        extra_requested_labels=320, extra_TRAIN_families=160, total_offline_critic_labels=384,
        unchanged_behavior_targets=256, reward_scale=scale, validation_heldout_used=False,
        added_labels_are_online_feedback=False, outcomes=bindings)
    path = experiment/'expanded_training_binding.json'
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('Expanded training data changed since the first training attempt')
    else:
        atomic_json(path,binding)
    return rows,binding
