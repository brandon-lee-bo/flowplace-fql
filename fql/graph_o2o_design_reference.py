"""Predeclare final routes and select measured-data controls without eval leakage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import load_requests
from .graph_o2o_design_collect import audit_outcome
from .graph_o2o_design_policy_data import context, load_dataset, measured_context


def initial_candidates(binding):
    labels = json.loads(Path(binding['backend_labels_path']).read_text())
    result = []
    for label in labels:
        if label['critic_split'] != 0:
            continue
        path = Path(label['outcome_path'])
        if sha(path) != label['outcome_sha256']:
            raise ValueError('Initial control candidate changed')
        out = json.loads(path.read_text())
        result.append(dict(out, outcome_path=str(path), outcome_sha256=sha(path),
                           origin='offline_TRAIN', source_row_index=label['source_row_index']))
    if len(result) != 64:
        raise ValueError('Best-observed control lost an initial TRAIN candidate')
    return result


def select(experiment, name, candidates, budget):
    experiment = Path(experiment)
    if len(candidates) != budget:
        raise ValueError('Best-observed control uses the wrong measurement budget')
    successful = [item for item in candidates if item['status'] == 'success']
    if not successful:
        raise ValueError('No successful measured layout is available for the control')
    # Stable original list order breaks exact ties; no final-route/eval selection.
    best = min(successful, key=lambda item: item['metrics']['rWL_um'])
    if not best.get('backend_macro_legal'):
        raise ValueError('Selected measured layout lacks legal backend geometry')
    geometry_path = best.get('backend_macro_geometry_path', best.get('backend_geometry_path'))
    geometry_sha = best.get('backend_macro_geometry_sha256', best.get('backend_geometry_sha256'))
    if not geometry_path or sha(geometry_path) != geometry_sha:
        raise ValueError('Selected control lost its exact actual macro evidence')
    record = dict(best, backend_macro_geometry_path=geometry_path,
        backend_macro_geometry_sha256=geometry_sha, selection_kind='best_observed_real_TRAIN_data',
        control_only_never_replayed=True, original_outcome_path=best['outcome_path'],
        original_outcome_sha256=best['outcome_sha256'], requested_label_budget=budget,
        successful_candidates=len(successful), failed_candidates=budget-len(successful),
        rule='Minimum successful GRT rWL; stable source order breaks ties. All failures retained in budget.',
        final_route_used_for_selection=False, evaluation_used_for_selection=False,
        candidate_outcomes=[dict(path=item['outcome_path'], sha256=item['outcome_sha256'],
                                  status=item['status']) for item in candidates])
    folder = experiment / 'references' / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'outcome.json'
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Previously selected immutable measured-data control changed')
    else:
        atomic_json(path, record)
    return dict(path=str(path), sha256=sha(path), requested=budget, successful=len(successful),
                observed_GRT_rWL_um=best['metrics']['rWL_um'])


def freeze(experiment):
    experiment, cfg, _, _, _, _ = context(experiment)
    data, manifest, binding = load_dataset(experiment, freeze=True)
    reference = select(experiment, 'best64', initial_candidates(binding), 64)
    plan_path = experiment / 'full_route_plan.json'
    tasks = []
    names = ['best64', 'best80_bc', 'best80_fql']
    for name in names:
        tasks.append(dict(key=name, method=name, group=name, noise_index=0,
            source_outcome=str(experiment / 'references' / name / 'outcome.json'),
            output=str(experiment / 'full_route' / name),
            run_id=cfg['run_id_prefix'] + '_drt_' + name))
    for arm, number in [('flowplace', 0), ('bc', 0), ('fql', 0),
                        ('bc', cfg['online_rounds']), ('fql', cfg['online_rounds'])]:
        for index in range(cfg['evaluation_samples']):
            key = f'{arm}_r{number}_n{index}'
            tasks.append(dict(key=key, method=f'{arm}_round_{number}', group=f'{arm}_r{number}',
                noise_index=index, source_outcome=str(experiment / 'seed_0' / f'round_{number}' /
                    'requests' / arm / 'evaluation' / str(index) / 'outcome.json'),
                output=str(experiment / 'full_route' / key),
                run_id=cfg['run_id_prefix'] + '_drt_' + key))
    plan = dict(version=1, design=cfg['design'], seed=0, requested=len(tasks), tasks=tasks,
        config_sha256=sha(experiment / 'config.json'),
        dataset_binding_sha256=sha(experiment / 'dataset_binding.json'),
        detailed_route_end_iteration=20, backend_storage='scratch', concurrency=2,
        original_Table2_resumed=False, evaluation_replayed=False,
        checkpoint_selection=cfg['checkpoint_selection'],
        controls='best64: one offline TRAIN layout; best80:64 offline +16 own collect labels actually used before final round2. No evaluation/unused final collect labels.',
        replicate_scope='One independent training seed per learned arm; four paired generation noises are within-seed samples.')
    if len(tasks) != 23 or cfg['online_rounds'] != 2 or cfg['evaluation_samples'] != 4:
        raise ValueError('Predeclared second-design final-route denominator changed')
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Previously frozen final-route scope changed')
    else:
        atomic_json(plan_path, plan)
    output = dict(status='all_80_labels_independently_audited_before_training', design=cfg['design'],
        offline_rows=len(data['train']['reward']), validation_rows=len(data['validation']['reward']),
        known_physical_failures=80-manifest['successful'], reward_scale=binding['reward_scale'],
        dataset_binding_sha256=sha(experiment / 'dataset_binding.json'), best64=reference,
        final_route_plan_sha256=sha(plan_path), final_route_requests=len(tasks),
        policy_training_started=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(experiment / 'initial_data_audit.json', output)
    print(json.dumps(output), flush=True)


def select_online(experiment):
    experiment = Path(experiment).resolve()
    cfg, binding, _ = measured_context(experiment)
    load_dataset(experiment)
    report = {}
    for arm in ['bc', 'fql']:
        candidates = initial_candidates(binding)
        for number in range(cfg['online_rounds']):
            rows, _ = load_requests(experiment, 0, number)
            selected = [row for row in rows if row['arm'] == arm and row['role'] == 'collect']
            if len(selected) != cfg['online_samples']:
                raise ValueError('Incomplete used-own-online collection budget')
            for row in selected:
                if not row['training_feedback_eligible']:
                    raise ValueError('Evaluation outcome entered the best-observed control')
                candidates.append(dict(audit_outcome(row, cfg, binding['reward_scale']), origin='own_online'))
        report[arm] = select(experiment, 'best80_' + arm, candidates, 80)
    atomic_json(experiment / 'references' / 'best80_selection.json', dict(
        rule='64 initial TRAIN plus16 own online actions used by final round2 policy', arms=report,
        final_collect_round_used=False, evaluation_used=False, slurm_job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--stage', required=True, choices=['freeze', 'select'])
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for complete data and result audits')
    (freeze if args.stage == 'freeze' else select_online)(args.experiment)
