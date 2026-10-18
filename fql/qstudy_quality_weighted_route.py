"""Fixed24 development DRT metadata and saved-result audits for weighted FM.

Physical work is exclusively the unchanged graph_o2o_route_controller. This
adapter never submits jobs, infers layouts, selects samples, or admits replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import statistics

import numpy as np

# Bind local analysis before the upstream baseline may modify import paths.
from analysis import qstudy_quality_weighted_results as weighted
from . import qstudy_final_route as native
from .graph_o2o_backend import load_requests
from .qstudy_data import bind, sha

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'qstudy_quality_weighted_route_v1'
ARMS, SEEDS, INDICES = ('uniform', 'quality_rank'), (0, 1, 2), (0, 1, 2, 3)
REQUESTED = 24
PASS = 'passes_declared_development_gate'
SOURCE_NAMES = tuple(dict.fromkeys((*native.SOURCES, *weighted.SOURCE_NAMES,
    'fql/qstudy_quality_weighted_route.py', 'fql/qstudy_quality_weighted_route.slurm',
    'fql/qstudy_quality_weighted_eval.slurm', 'fql/qstudy_quality_weighted_flow.slurm',
    'analysis/qstudy_quality_weighted_results.slurm',
    'fql/graph_o2o_route_controller.slurm', 'fql/graph_o2o_archive.py')))


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {name: sha(ROOT/name) for name in SOURCE_NAMES}


def checked_grt_report(path, expected_sha256):
    """Revalidate saved48, all six training chains and the declared gate.

    Observer-only imports can expand source-validation diagnostics; they may
    not change any producer, training, request, noise, native evidence or metric.
    """
    path = Path(path).resolve()
    bind(path, expected_sha256)
    report = read(path)
    completion = path.with_name('completed.json')
    done = read(completion)
    if (report.get('schema') != 'qstudy_quality_weighted_native_results_v1' or
            report.get('status') != 'complete' or report.get('stage') != 'global_route_only' or
            report.get('development_only') is not True or report.get('no_replay') is not True or
            report.get('selected_winner') is not None or report.get('new_backend_calls') != 0 or
            done.get('status') != 'completed' or done.get('report_status') != 'complete' or
            done.get('backend_calls') != 0 or done.get('model_inference_calls') != 0 or
            done.get('artifacts', {}).get('report.json') != expected_sha256 or
            report.get('protocol') != bind(ROOT/weighted.PROTOCOL, weighted.PROTOCOL_SHA)):
        raise ValueError('Require the completed, unchanged development48 GRT report')
    for name, digest in done['artifacts'].items():
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Noncanonical report artifact name')
        bind(path.parent/name, digest)
    for name, digest in report['source_sha256'].items():
        if name not in weighted.SOURCE_NAMES:
            raise ValueError('Unknown weighted-report source')
        bind(ROOT/name, digest)
    if set(report['source_sha256']) != set(weighted.SOURCE_NAMES):
        raise ValueError('Incomplete weighted-report source closure')
    saved = report['experiments'][:2]
    if [r.get('arm') for r in saved] != list(ARMS):
        raise ValueError('Require ordered uniform and quality_rank evidence')
    actual = [weighted.collect_arm(Path(r['experiment']).resolve(), arm)
              for arm, r in zip(ARMS, saved)]
    weighted.matched_context(actual)
    for old, current in zip(saved, actual):
        # This one field records imports of the *reader*, not producer inputs.
        stable = lambda value: {k:v for k,v in value.items() if k != 'producer_source_validation'}
        if stable(old) != stable(current):
            raise ValueError('Saved GRT report differs from revalidated native/source evidence')
        if (len(current['streams']) != 3 or any(
                s['success'] != 8 or not s['fully_accounted'] or s['integrity_errors']
                for s in current['streams'])):
            raise ValueError('All48 successful GRT requests required for numeric promotion gate')
    gate = weighted.protocol_gates(actual[1], actual[0])
    if gate != report['development_gates'] or gate['status'] != PASS:
        raise ValueError('The prespecified weighted-versus-uniform GRT gate did not pass')
    return report, actual, dict(report=bind(path), completion=bind(completion))


def build_plan(output, report_path, report_sha256, prefix='go2o_qswdrt1', concurrency=2,
               *, loaded=None):
    output = Path(output).resolve()
    if (not re.fullmatch(r'go2o_[A-Za-z0-9][A-Za-z0-9_.-]*', prefix) or
            type(concurrency) is not int or not 1 <= concurrency <= 2):
        raise ValueError('Require a new go2o_ namespace and at most two DRT workers')
    report, arms, evidence = loaded or checked_grt_report(report_path, report_sha256)
    protected = [ROOT, Path(report_path).resolve().parent]
    protected += [Path(r['experiment']).resolve() for r in arms]
    protected += [Path(r['training_manifest']['path']).resolve().parent for r in arms]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError('Route output overlaps source/report/training assets')
    experiments, tasks = {}, []
    for arm, result in zip(ARMS, arms):
        experiment = Path(result['experiment']).resolve()
        cfg = read(experiment/'config.json')
        experiments[arm] = dict(path=str(experiment), config=cfg,
            config_sha256=sha(experiment/'config.json'), declaration=result['declaration'],
            export_completion=result['completion'], training_manifest=result['training_manifest'])
        for seed in SEEDS:
            stream = next(s for s in result['streams'] if s['seed'] == seed)
            for index in INDICES:
                row = stream['rows'][index]
                key = native.task_key(arm, seed, 0, index)
                tasks.append(dict(key=key, group=arm, method=arm+'_quality_fm_round_0',
                    arm=arm, seed=seed, round=0, noise_index=index,
                    source_outcome=row['outcome']['path'],
                    output=str(output/'full_route'/key), run_id=prefix+'_'+key))
    return dict(version=1, schema=SCHEMA, study=SCHEMA, design='bp_fe',
        purpose='development_validation', fresh_confirmation=False, historical_table2=False,
        training_feedback_eligible=False, offline_train_eligible=False,
        grt_evidence=evidence, grt_gate=report['development_gates'],
        actor_experiments=experiments, requested=REQUESTED, original_grt_requests=48,
        actor_seeds=list(SEEDS), rounds=[0], evaluation_indices=list(INDICES),
        original_evaluation_samples=8, run_id_prefix=prefix, concurrency=concurrency,
        detailed_route_end_iteration=20, backend_threads=8, backend_storage='scratch',
        final_metrics=list(native.METRICS), metric_units=native.UNITS,
        source_sha256=sources(), tasks=tasks,
        selection='first four predeclared indices of both arms/all3seeds; no Q/metric/layout selection',
        primary_contrast='quality_rank minus uniform; unchanged common flow is not a new DRT arm',
        failure_protocol='Retain all24; missing/infrastructure PPA remains null; measured DRC>0 retained',
        power_interpretation='common tool-default activity estimate, not workload-specific silicon power',
        inference='development only; no fresh confirmation, automatic promotion or replay')


def validate_plan(experiment, expected_sha256=None, *, loaded=None):
    experiment = Path(experiment).resolve()
    path = experiment/'full_route_plan.json'
    if expected_sha256 is not None:
        bind(path, expected_sha256)
    plan = read(path)
    if read(experiment/'route_request.json') != dict(schema=SCHEMA, plan=bind(path),
            requested=REQUESTED, purpose='development_validation', helper_sha256=sha(__file__),
            conditional_on_prespecified_GRT_gate=True):
        raise ValueError('Fixed24 conditional route declaration changed')
    evidence = plan['grt_evidence']['report']
    expected = build_plan(experiment, evidence['path'], evidence['sha256'],
        plan['run_id_prefix'], plan['concurrency'], loaded=loaded)
    if plan != expected:
        raise ValueError('Route source/config/gate/budget identity changed')
    return plan


def inspect_sources(experiment, plan, *, seal=False, loaded=None):
    """Seal the original request/native geometry; never use actor-chain assumptions."""
    experiment = Path(experiment).resolve()
    evidence = plan['grt_evidence']['report']
    _, arms, _ = loaded or checked_grt_report(evidence['path'], evidence['sha256'])
    results = {r['arm']:r for r in arms}
    records, noise_hashes = [], defaultdict(set)
    request_cache = {}
    for task in plan['tasks']:
        path = experiment/'source_bindings'/(task['key']+'.json')
        item = dict(key=task['key'], state='missing', binding=None)
        try:
            result = results[task['arm']]
            seed = task['seed']
            stream = next(s for s in result['streams'] if s['seed'] == seed)
            metric = next(r for r in stream['rows'] if r['index'] == task['noise_index'])
            key = (task['arm'], seed)
            if key not in request_cache:
                request_cache[key] = load_requests(result['experiment'], seed, 0)
            rows, tasks_path = request_cache[key]
            row = next(r for r in rows if r['index'] == task['noise_index'])
            sample = Path(row['sample_dir']).resolve()
            out = read(task['source_outcome'])
            if (metric['state'] != 'success' or out['status'] != 'success' or
                    row != read(sample/'request.json') or stream['tasks'] != bind(tasks_path) or
                    metric['request'] != bind(sample/'request.json') or
                    metric['actions'] != bind(sample/'actions.npz') or
                    metric['outcome'] != bind(task['source_outcome']) or
                    any(metric[k] != row[k] for k in ('noise_sha256','raw_action_sha256','action_sha256')) or
                    Path(task['source_outcome']).resolve() != sample/'outcome.json' or
                    row['arm'] != task['arm'] or row['seed'] != seed or row['round'] != 0 or
                    row['role'] != 'evaluation' or row['training_feedback_eligible'] is not False or
                    row['offline_train_eligible'] is not False):
                raise ValueError('Route source/request differs from revalidated full48 evidence')
            with np.load(sample/'actions.npz', allow_pickle=False) as arrays:
                values = {name:arrays[name].tolist() for name in ('noise','raw_action','action')}
            binding = dict(version=1, key=task['key'], arm=task['arm'], seed=seed, round=0,
                noise_index=task['noise_index'], status=out['status'], source_kind=row['source_kind'],
                source_sampler=row['source_sampler'], source_outcome=str(sample/'outcome.json'),
                source_outcome_sha256=sha(sample/'outcome.json'), request_sha256=sha(sample/'request.json'),
                tasks_sha256=sha(tasks_path), export_completion_sha256=sha(tasks_path.with_name('completed.json')),
                actions_npz_sha256=sha(sample/'actions.npz'), raw_action_sha256=row['raw_action_sha256'],
                action_sha256=row['action_sha256'], noise_sha256=row['noise_sha256'], noise_seed=row['noise_seed'],
                exact_coordinates_and_noise=values, def_sha256=row.get('def_sha256'),
                trained_flow=result['all_paired_training'][task['arm']][str(seed)],
                source_training_artifacts=row['source_training_artifacts'],
                config_sha256=result['config']['sha256'], dataset_sha256=result['physical_binding']['dataset_sha256'],
                native=metric['native'], physical_flow_revision=result['physical_binding']['physical_flow_revision'],
                GRT_metrics=out['metrics'], final_metrics=None, failure_class=out.get('failure_class'),
                training_feedback_eligible=False, offline_train_eligible=False)
            if not seal and (not path.is_file() or read(path) != binding):
                raise ValueError('Source not sealed or binding changed; resolve before report')
            item.update(state='resolved', binding=binding)
            noise_hashes[task['noise_index']].add(row['noise_sha256'])
        except FileNotFoundError as error:
            item.update(state='invalid' if path.exists() else 'missing', reason=str(error))
        except Exception as error:
            item.update(state='invalid', reason=str(error))
        records.append(item)
    bad = {i for i, values in noise_hashes.items() if len(values) != 1}
    for task, item in zip(plan['tasks'], records):
        if task['noise_index'] in bad:
            item.update(state='invalid', reason='Matched actual noise mismatch')
        if item['state'] == 'resolved':
            path = experiment/'source_bindings'/(task['key']+'.json')
            try:
                if seal:
                    native.freeze(path, item['binding'])
                item['binding_sha256'] = sha(path)
            except Exception as error:
                item.update(state='invalid', reason=str(error))
    return records


def aggregate(records):
    """Four noise means per training seed, then three independent seed means."""
    seeds, groups, pairs = [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            rows = [r for r in records if r['arm'] == arm and r['seed'] == seed]
            good = [r for r in rows if r['state'] == 'completed']
            full = len(rows) == len(good) == 4
            seeds.append(dict(arm=arm, seed=seed, planned=4,
                counts=dict(Counter(r['state'] for r in rows)), complete=full,
                drc_clean=sum(r['metrics']['DRC'] == 0 for r in good),
                mean=native.metric_means(good) if full else None,
                successful_only_mean=native.metric_means(good)))
        subset = [s for s in seeds if s['arm'] == arm]
        rows = [r for r in records if r['arm'] == arm]
        complete = len(subset) == 3 and all(s['complete'] for s in subset)
        groups.append(dict(arm=arm, planned=12, planned_training_seeds=3,
            complete_training_seeds=sum(s['complete'] for s in subset),
            counts=dict(Counter(r['state'] for r in rows)),
            drc_clean=sum(r['state'] == 'completed' and r['metrics']['DRC'] == 0 for r in rows),
            statistics={k:native.summary([s['mean'][k] for s in subset]) for k in native.METRICS}
                if complete else None, conditional_PPA=not complete))
    for seed in SEEDS:
        by_arm = {arm:{r['noise_index']:r for r in records if r['arm']==arm and r['seed']==seed} for arm in ARMS}
        matched = []
        for index in INDICES:
            a, b = by_arm['quality_rank'].get(index), by_arm['uniform'].get(index)
            both = bool(a and b and a['state'] == b['state'] == 'completed' and
                        a['noise_sha256'] == b['noise_sha256'])
            matched.append(dict(index=index, both_completed=both,
                delta={k:a['metrics'][k]-b['metrics'][k] for k in native.METRICS} if both else None))
        full = all(r['both_completed'] for r in matched)
        pairs.append(dict(seed=seed, planned=4, matched_completed=sum(r['both_completed'] for r in matched),
            pairs=matched, delta={k:statistics.mean(r['delta'][k] for r in matched) for k in native.METRICS} if full else None))
    return dict(seed_means=seeds, policy_groups=groups, matched_deltas=dict(
        comparison='quality_rank minus uniform', planned_pairs=12,
        sign='negative helps DRT_WL/Power/StdCellArea/DRC; positive helps WNS/TNS',
        per_seed=pairs, independent_complete_seeds=sum(r['delta'] is not None for r in pairs),
        statistics={k:native.summary([r['delta'][k] for r in pairs]) for k in native.METRICS}
            if all(r['delta'] is not None for r in pairs) else None))


def report_value(experiment, plan, source_records):
    rows = []
    for task, source in zip(plan['tasks'], source_records):
        try:
            rows.append(native.inspect_final(experiment, plan, task, source))
        except Exception as error:
            rows.append(dict(**{k:task[k] for k in ('key','arm','seed','round','noise_index')},
                state='invalid', reason=str(error), metrics=None, noise_sha256=None))
    identities = {json.dumps(r['physical_identity'],sort_keys=True) for r in rows if r['state']=='completed'}
    runs = Counter(r['physical_run'] for r in rows if r['state']=='completed')
    for row in rows:
        if row['state']=='completed' and (len(identities)>1 or runs[row['physical_run']]>1):
            row.update(state='invalid',metrics=None,reason='Changed shared physical identity or duplicated native call')
    counts = dict(Counter(r['state'] for r in rows))
    accounted = sum(counts.get(s,0) for s in ('completed','upstream_physical_failure'))
    return dict(schema=SCHEMA, status='all_requested_accounted' if accounted==REQUESTED else 'partial_or_invalid',
        purpose='development_validation', fresh_confirmation=False, requested=REQUESTED,
        counts=counts, metric_units=native.UNITS, plan=bind(experiment/'full_route_plan.json'),
        source_sha256=sources(), grt_evidence=plan['grt_evidence'], records=rows,
        **aggregate(rows), new_backend_calls=0, no_replay=True, selected_winner=None,
        interpretation='Conditional development DRT follow-up of the passed GRT gate; no fresh confirmation or stable-method promotion',
        failure_interpretation='Missing/infra/invalid remain nonnumeric; complete means require all4 noises in all3seeds. DRC>0 retains measured PPA.',
        power_interpretation=plan['power_interpretation'],
        uncertainty='Sample SD across three training-seed means; four shared noises within each seed are not training replicates',
        congestion='No final routed congestion inferred from GRT overflow or DRT reports')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--grt-report', type=Path, required=True)
    p.add_argument('--grt-report-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-prefix', default='go2o_qswdrt1')
    p.add_argument('--concurrency', type=int, default=2)
    for name in ('resolve','report'):
        p = sub.add_parser(name)
        p.add_argument('--experiment', type=Path, required=True)
        p.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use bounded CPU Slurm for saved source/native audits')
    if args.stage == 'plan':
        output = args.output.resolve()
        if not (output/'full_route_plan.json').exists() and output.exists() and any(output.iterdir()):
            raise ValueError('Refuse a nonempty orphan routing namespace')
        value = build_plan(output,args.grt_report,args.grt_report_sha256,args.run_prefix,args.concurrency)
        native.freeze(output/'full_route_plan.json',value)
        native.freeze(output/'route_request.json',dict(schema=SCHEMA, plan=bind(output/'full_route_plan.json'),
            requested=REQUESTED,purpose='development_validation',helper_sha256=sha(__file__),
            conditional_on_prespecified_GRT_gate=True))
        print(json.dumps(dict(status='predeclared',requested=REQUESTED,plan=bind(output/'full_route_plan.json'),new_backend_calls=0)),flush=True)
        return
    experiment = args.experiment.resolve()
    raw_plan = read(experiment/'full_route_plan.json')
    evidence = raw_plan['grt_evidence']['report']
    loaded = checked_grt_report(evidence['path'],evidence['sha256'])
    plan = validate_plan(experiment,args.plan_sha256,loaded=loaded)
    records = inspect_sources(experiment,plan,seal=args.stage=='resolve',loaded=loaded)
    if args.stage == 'resolve':
        value = dict(schema=SCHEMA,status='complete' if all(r['state']=='resolved' for r in records) else 'partial_or_invalid',
            requested=REQUESTED,counts=dict(Counter(r['state'] for r in records)),records=records,
            plan=bind(experiment/'full_route_plan.json'),new_backend_calls=0,no_replay=True)
        folder = native.snapshot(experiment,'source_resolution',value)
        native.csv_write(folder/'sources.csv',records)
    else:
        value = report_value(experiment,plan,records)
        folder = native.snapshot(experiment,'development_report',value)
        for filename,key in (('layouts.csv','records'),('seed_means.csv','seed_means'),('policy_groups.csv','policy_groups')):
            native.csv_write(folder/filename,value[key])
        native.csv_write(folder/'paired_seed_deltas.csv',value['matched_deltas']['per_seed'])
    print(json.dumps(dict(status=value['status'],report=bind(folder/'report.json'),counts=value['counts'],new_backend_calls=0)),flush=True)


if __name__ == '__main__':
    main()
