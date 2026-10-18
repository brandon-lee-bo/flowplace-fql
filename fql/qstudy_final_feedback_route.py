"""Declare and seal the fixed72 final-feedback actor detailed-route requests.

This adapter only writes metadata and audits saved GRT sources. Physical work
uses the unchanged graph_o2o_route_controller and cap20 native backend. It
never selects layouts, launches EDA, or treats evaluation feedback as training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

from . import qstudy_final_feedback_actor as actor
from . import qstudy_final_route as native
from .qstudy_data import bind, sha

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'qstudy_final_feedback_actor_route_v1'
SEEDS = (0, 1, 2)
INDICES = (0, 1, 2, 3)
REQUESTED = 72
SOURCE_NAMES = ('fql/qstudy_final_feedback_route.py',
    'fql/qstudy_final_feedback_route.slurm', 'fql/qstudy_final_route.py',
    'fql/graph_o2o_route_controller.py', 'fql/graph_o2o_route_controller.slurm',
    'fql/graph_o2o_controller.py', 'fql/graph_o2o_full_route.py',
    'fql/graph_o2o_full_route.slurm', 'fql/graph_o2o_finish.py',
    'fql/graph_o2o_finish.sh', 'fql/graph_o2o_final_evidence.py',
    'fql/qstudy_actor_report.py')


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {**actor.sources(), **{name: sha(ROOT/name) for name in SOURCE_NAMES}}


def slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a bounded Slurm allocation')


def build_plan(output, actors_root, prefix='ffdrt1', concurrency=2):
    """Metadata-only declaration: do not inspect any evaluation outcome."""
    output, actors_root = Path(output).resolve(), Path(actors_root).resolve()
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', prefix) or
            type(concurrency) is not int or not 1 <= concurrency <= 2):
        raise ValueError('Invalid route namespace or declared at-most-two workers')
    if output == actors_root or output.is_relative_to(actors_root) or actors_root.is_relative_to(output):
        raise ValueError('Routing outputs must be disjoint from actor inputs')
    experiments, tasks = {}, []
    for objective in actor.OBJECTIVES:
        arm = actor.ARMS[objective][0]
        experiment = actors_root/arm
        configs = [actor.validate_config(experiment, seed, 0) for seed in SEEDS]
        cfg = configs[0]
        if any(value != cfg for value in configs[1:]):
            raise ValueError('One config must bind all three independent actor seeds')
        if (cfg['final_feedback_objective'] != objective or
                cfg['final_feedback_evaluation'] != actor.FINAL_EVALUATION or
                cfg['evaluation_purpose'] != 'development_validation' or
                cfg['backend_threads'] != 8 or cfg['backend_storage'] != 'scratch'):
            raise ValueError('Actor config differs from fixed final-feedback evaluation')
        experiments[arm] = dict(path=str(experiment),
            config_sha256=sha(experiment/'config.json'), config=cfg)
        for seed in SEEDS:
            for index in INDICES:
                key = native.task_key(arm, seed, 0, index)
                tasks.append(dict(key=key, group=arm, method=arm+'_round_0',
                    arm=arm, seed=seed, round=0, noise_index=index,
                    source_outcome=str(experiment/f'seed_{seed}/round_0/requests'/
                                       arm/'evaluation'/str(index)/'outcome.json'),
                    output=str(output/'full_route'/key), run_id=prefix+'_'+key))
    preparation = bind(actors_root/'preparation.json')
    return dict(version=1, schema=SCHEMA, study=SCHEMA, design='bp_fe',
        purpose='development_validation', fresh_confirmation=False,
        historical_table2=False, training_feedback_eligible=False,
        actors_root=str(actors_root), actor_preparation=preparation,
        actor_experiments=experiments, requested=REQUESTED,
        actor_seeds=list(SEEDS), rounds=[0], evaluation_indices=list(INDICES),
        original_evaluation_samples=8, original_grt_requests=144,
        selection='first four predeclared indices for every arm/seed; no metric/Q/layout selection',
        policy_estimand='within-seed fixed-noise mean, then three independent actor-seed means',
        final_metrics=list(native.METRICS), metric_units=native.UNITS,
        detailed_route_end_iteration=20, backend_threads=8,
        backend_storage='scratch', concurrency=concurrency, run_id_prefix=prefix,
        source_sha256=sources(), tasks=tasks,
        failure_protocol='Retain all72; unavailable/infrastructure outcomes remain NA; measured DRC>0 remains measured and receives B4 in all reward targets',
        power_interpretation='tool-default-activity estimate under common fixed native setup; not workload-specific or measured silicon power',
        inference='offline round0 evaluation only; none of these outcomes may enter training')


def validate_plan(experiment, expected_sha256=None):
    experiment = Path(experiment).resolve()
    path = experiment/'full_route_plan.json'
    if expected_sha256 is not None:
        bind(path, expected_sha256)
    plan = read(path)
    declaration = read(experiment/'route_request.json')
    if (declaration != dict(schema=SCHEMA, plan=bind(path), requested=REQUESTED,
            frozen_without_reading_evaluation_metrics=True,
            purpose='development_validation', helper_sha256=sha(__file__))):
        raise ValueError('Fixed72 route declaration changed')
    expected = build_plan(experiment, plan['actors_root'],
                          plan['run_id_prefix'], plan['concurrency'])
    if plan != expected:
        raise ValueError('Final-feedback route tasks, configs, source identities or budget changed')
    return plan


def inspect_sources(experiment, plan, *, seal=False):
    """Use unchanged native request/outcome/macro/noise/actor-chain auditor."""
    if plan.get('schema') != SCHEMA or len(plan.get('tasks', [])) != REQUESTED:
        raise ValueError('Expected the exact final-feedback route denominator')
    return native.inspect_sources(Path(experiment).resolve(), plan, seal=seal)


def plan_command(args):
    slurm()
    output = Path(args.output).resolve()
    plan = build_plan(output, args.actors_root, args.run_prefix, args.concurrency)
    output.mkdir(parents=True, exist_ok=True)
    native.freeze(output/'full_route_plan.json', plan)
    native.freeze(output/'route_request.json', dict(schema=SCHEMA,
        plan=bind(output/'full_route_plan.json'), requested=REQUESTED,
        frozen_without_reading_evaluation_metrics=True,
        purpose='development_validation', helper_sha256=sha(__file__)))
    print(json.dumps(dict(status='predeclared_metadata_only', requested=REQUESTED,
        output=str(output), backend_calls=0)), flush=True)


def resolve_command(args):
    slurm()
    experiment = Path(args.experiment).resolve()
    plan = validate_plan(experiment, args.plan_sha256)
    rows = inspect_sources(experiment, plan, seal=True)
    report = dict(schema=SCHEMA, status='complete' if all(
        row['state'] == 'resolved' for row in rows) else 'partial_or_invalid',
        requested=REQUESTED, plan=bind(experiment/'full_route_plan.json'),
        counts=dict(Counter(row['state'] for row in rows)), records=rows,
        backend_calls=0, training_feedback_eligible=False)
    folder = native.snapshot(experiment, 'source_resolution', report)
    native.csv_write(folder/'sources.csv', rows)
    print(json.dumps(dict(status=report['status'], requested=REQUESTED,
        report=str(folder/'report.json'), counts=report['counts'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--actors-root', required=True)
    plan.add_argument('--output', required=True)
    plan.add_argument('--run-prefix', default='ffdrt1')
    plan.add_argument('--concurrency', type=int, default=2)
    resolve = sub.add_parser('resolve')
    resolve.add_argument('--experiment', required=True)
    resolve.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    (plan_command if args.command == 'plan' else resolve_command)(args)


if __name__ == '__main__':
    main()
