"""Development-only common initial-Q selection on 72 already measured proposals.

Three common-flow seeds, BC and offline FQL, eight identical saved noises each.
Only frozen CPU float32 critics run. No policy, training, legalizer or EDA runs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import qstudy_flow_baseline_report as flow
from . import qstudy_selection_report as selection
from .graph_o2o_archive import atomic_json, sha
from .qstudy_actor_report import METRICS, train_scales
from .qstudy_backend import context
from .qstudy_data import action_sha, load_dataset
from .qstudy_screen import make_model

ROOT = Path(__file__).resolve().parents[1]
BULK = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')
INPUTS = {
    'flow': (BULK/'flow_baseline_native_report_v1/20261018T104500.097363Z_job1796858/report.json',
             'ee8d29d7996616f0949bfed10ed83f673e0400e6f57c85b7d5a107d93597b406'),
    'actor': (BULK/'matched_offline_r0_report_v1/20261018T104921.160264Z_job1797018/report.json',
              'dc19bc3b64d3ea7888cffb2b70e555f183e3a32f3cb7885b89b2e76e7340c7d6'),
    'selection': (BULK/'matched_offline_r0_selection_v1/20261018T105429_998452_job1797038/report.json',
                  'a1724a2f8df403afeda5c72de7082ea62e2c1ce213d8686c15df4c055744d3c9')}
ARMS = ('flow', 'bc', 'fql')
Q_RULE = 'max_frozen_initial_q'
RULES = tuple(Q_RULE if x == 'max_recorded_current_q' else x for x in selection.RULES)
SOURCES = ('fql/qstudy_fixed_q_selection.py', 'fql/qstudy_selection_report.py',
    'fql/qstudy_flow_baseline_report.py', 'fql/qstudy_actor_report.py',
    'fql/qstudy_screen.py', 'fql/qstudy_data.py', 'fql/qstudy_models.py',
    'fql/graph_o2o_models.py', 'fql/physical_ridge.py', 'fql/analytic.py',
    'fql/qstudy_backend.py', 'fql/graph_o2o_backend.py', 'baseline/api.py')


def read(path):
    return json.loads(Path(path).read_text())


def bound(path, digest=None):
    path = Path(path).resolve()
    value = sha(path)
    if digest is not None and value != digest:
        raise ValueError('Changed immutable input: '+str(path))
    return dict(path=str(path), sha256=value)


def immutable_report(path, digest):
    binding = bound(path, digest)
    marker = Path(path).with_name('completed.json')
    done, report = read(marker), read(path)
    if done.get('status') != 'completed' or digest != done.get('report_sha256', done.get('artifacts', {}).get('report.json')):
        raise ValueError('Unbound report completion')
    for name, value in report.get('source_sha256', {}).items():
        bound(ROOT/name, value)
    if report.get('report_source_sha256'):
        bound(ROOT/'fql/qstudy_selection_report.py', report['report_source_sha256'])
    return report, dict(**binding, completion=bound(marker))


def state_equal(left, right):
    """Full parameter AND buffer bit equality, including dtype and signed zero."""
    if set(left) != set(right):
        raise ValueError('Frozen critic state keys differ')
    for key in left:
        a, b = left[key], right[key]
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or a.shape != b.shape or a.dtype != b.dtype
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()
                or not torch.equal(a.detach().cpu().reshape(-1).view(torch.uint8),
                                   b.detach().cpu().reshape(-1).view(torch.uint8))):
            raise ValueError('Frozen critic tensor changed: '+key)


def checked_actor_critic(record, initial, common=None):
    folder = Path(record['experiment'])/f"seed_{record['seed']}/round_0"
    path, done = folder/'checkpoint.pt', read(folder/'completed.json')
    bound(path, record['training']['checkpoint_sha256'])
    if (done.get('offline_critic_frozen') is not True or done.get('steps') != 1000
            or sha(folder/'completed.json') != record['training']['completion_sha256']):
        raise ValueError('Round-0 offline freeze contract changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['binding'] != done['binding'] or saved['step'] != 1000:
        raise ValueError('Round-0 checkpoint binding/step differs')
    state_equal(saved['critic'], initial['model'])
    if common is not None:
        state_equal(saved['flow'], common['flow'])
    if saved['optimizers']['critic']['state']:
        raise ValueError('Round-0 critic optimizer was updated')
    return dict(checkpoint=bound(path), completion=bound(folder/'completed.json'),
                initial_state_bitwise_equal=True, critic_optimizer_state_empty=True,
                common_flow_state_bitwise_equal=True if common is not None else None)


def initial_model(cfg, seed):
    """Reconstruct from explicit full TRAIN, then load the frozen fitted state."""
    from baseline.api import RUNTIME, load_problem
    path = Path(cfg['critic_checkpoints'][str(seed)])
    bound(path, cfg['critic_checkpoint_sha256'][str(seed)])
    saved, done = torch.load(path, map_location='cpu', weights_only=False), read(path.with_name('completed.json'))
    if (done.get('status') != 'completed' or done['checkpoint_sha256'] != sha(path)
            or saved['binding'] != done['binding'] or saved['step'] != 2000
            or saved['train_indices'] != list(range(384)) or saved['model_config'] != {'legacy': True}
            or done['seed'] != seed or done['train_rows'] != 384):
        raise ValueError('Not the fixed full384 TRAIN refit critic')
    binding = saved['binding']
    for key in ('dataset_sha256', 'cache_sha256'):
        if binding[key] != cfg[key]:
            raise ValueError('Initial critic/data context changed: '+key)
    for name, digest in binding['source_sha256'].items():
        bound(ROOT/name, digest)
    dataset = Path(cfg['dataset_path'])
    bound(dataset, cfg['dataset_sha256'])
    bound(dataset.with_name('manifest.json'), binding['manifest_sha256'])
    data, _ = load_dataset(dataset)
    train = data['train']
    if len(train['raw_action']) != 384 or not train['metric_mask'][:, 0].all() or not (train['metrics'][:, 0] > 0).all():
        raise ValueError('Full384 scalar WL labels are required')
    cache_path = RUNTIME/'benchmark_cache'/(cfg['design']+'.pt')
    bound(cache_path, cfg['cache_sha256'])
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    movable = ~load_problem(cfg['design'], 'cpu').cond.is_ports.bool()
    x = torch.tensor(train['raw_action'], dtype=torch.float32).reshape(384, -1, 2)
    costs = torch.tensor(train['metrics'][:, :5], dtype=torch.float32)
    # make_model(legacy) uses only positive WL. Fork restores caller RNG exactly.
    with torch.random.fork_rng(devices=[]):
        model, _ = make_model(cache, x, costs, torch.tensor(train['metric_mask'][:, :5]),
                              movable, 'legacy', torch.tensor(train['family']), seed)
    for key in ('mean', 'std', 'reward_mean', 'reward_std', 'movable'):
        state_equal({key: model.state_dict()[key]}, {key: saved['model'][key]})
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    state_equal(model.state_dict(), saved['model'])
    return model, saved, dict(checkpoint=bound(path), completion=bound(path.with_name('completed.json')),
        binding=binding, train_rows=384, WL_reference_float64=float(train['metrics'][:, 0].mean()),
        WL_reference_float32=float(costs[:, 0].mean()), normalization_reconstructed_from_TRAIN=True)


def arrays(row):
    sample = Path(row['sample_dir'])
    bound(sample/'request.json', row['request_sha256'])
    bound(sample/'actions.npz', row['actions_npz_sha256'])
    request = read(sample/'request.json')
    with np.load(sample/'actions.npz', allow_pickle=False) as values:
        noise, raw, executed = [values[key].copy() for key in ('noise', 'raw_action', 'action')]
    for key, value in (('noise', noise), ('raw_action', raw), ('action', executed)):
        if (value.dtype != np.float32 or value.ndim != 2 or value.shape[-1] != 2
                or not np.isfinite(value).all() or action_sha(value) != request[key+'_sha256']):
            raise ValueError('Changed actual float32 '+key)
        if value.shape != raw.shape:
            raise ValueError('Coordinate/noise shape mismatch')
    if (request['index'] != row['index'] or request['role'] != 'evaluation'
            or request['noise_seed'] != 4600000 or request['round'] != 0
            or request['training_feedback_eligible'] is not False
            or request.get('offline_train_eligible', False) is not False
            or request['noise_sha256'] != row['noise_sha256']):
        raise ValueError('Unexpected evaluation request scope')
    if not np.array_equal(np.asarray(request['raw_action'], dtype=np.float32).reshape(raw.shape), raw):
        raise ValueError('Request raw action differs from saved actual array')
    return noise, raw, request


def score_pool(model, rows):
    result = []
    for row in sorted(rows, key=lambda x: x['index']):
        noise, raw, request = arrays(row)
        # Singleton shape matches original CUDA export; no RNG or policy calls.
        with torch.no_grad():
            value = model(torch.from_numpy(raw)[None]).mean(1)
        if value.dtype != torch.float32 or value.shape != (1,) or not torch.isfinite(value).all():
            raise ValueError('Expected one finite float32 twin-mean Q')
        recorded = row.get('predicted_q')
        result.append(dict(row, recorded_cuda_q=recorded, predicted_q=float(value[0]),
            cpu_minus_recorded_q=float(value[0])-recorded if recorded is not None else None,
            raw_action_sha256=action_sha(raw), action_sha256=request['action_sha256'],
            source_checkpoint_sha256=request.get('checkpoint_sha256'), actual_noise=noise))
    return result


def noise_identity(pools):
    expected = None
    for pool in pools:
        rows = pool['rows']
        if len(rows) != 8 or [r['index'] for r in rows] != list(range(8)):
            raise ValueError('Complete ordered eight-candidate pools required')
        if len({r['noise_sha256'] for r in rows}) != 8:
            raise ValueError('Duplicate actual noise')
        current = [(r['noise_sha256'], r['actual_noise']) for r in rows]
        if expected is not None and any(a[0] != b[0] or not np.array_equal(a[1], b[1]) for a,b in zip(expected,current)):
            raise ValueError('Actual noise arrays differ across generators/seeds')
        expected = current
    if {(p['arm'],p['seed']) for p in pools} != {(a,s) for a in ARMS for s in range(3)} or len(pools) != 9:
        raise ValueError('Expected exactly three generators times three common seeds')


def score_equivalence(rows):
    cpu = np.asarray([r['predicted_q'] for r in rows], dtype=float)
    chosen = lambda x: min(range(8), key=lambda i: (-x[i], i))
    result = dict(cpu_argmax_index=chosen(cpu), cpu_max_ties=int((cpu == cpu.max()).sum()),
                  cpu_top2_margin=float(np.sort(cpu)[-1]-np.sort(cpu)[-2]))
    if all(r['recorded_cuda_q'] is not None for r in rows):
        old = np.asarray([r['recorded_cuda_q'] for r in rows], dtype=float)
        result.update(recorded_argmax_index=chosen(old), argmax_identical=chosen(cpu)==chosen(old),
            exact_full_order_identical=sorted(range(8), key=lambda i:(-cpu[i],i)) == sorted(range(8), key=lambda i:(-old[i],i)),
            recorded_max_ties=int((old==old.max()).sum()), max_absolute_q_delta=float(np.abs(cpu-old).max()),
            all_scores_close=bool(np.allclose(cpu,old,atol=1e-6,rtol=1e-5)), atol=1e-6,rtol=1e-5)
    return result


def aggregate(pools):
    selected, means, pairs = [], [], []
    for pool in pools:
        chosen = selection.select_candidates(pool['rows'])
        for rule, row in chosen.items():
            selected.append(dict(row, arm=pool['arm'], seed=pool['seed'],
                rule=Q_RULE if rule=='max_recorded_current_q' else rule, pool_candidates=8,
                deployment_backend_budget=8 if rule.startswith('oracle_') else 1))
        means.append(dict(arm=pool['arm'], seed=pool['seed'], **selection.pool_mean(pool['rows'])))
    groups = [dict(arm=a,rule=r,summary=selection.metric_summary([x for x in selected if x['arm']==a and x['rule']==r]))
              for a in ARMS for r in RULES]
    mean_groups = []
    for arm in ARMS:
        rows = [x for x in means if x['arm']==arm]
        mean_groups.append(dict(arm=arm, seed_count=3,
            common_utility=selection.stats([x['common_utility_mean'] for x in rows]),
            native_full={m:selection.stats([x['native_full_mean'][m] for x in rows]) for m in METRICS}
                if all(x['native_full_mean'] is not None for x in rows) else None))
    picks = {(r['arm'],r['seed'],r['rule']):r for r in selected}
    for lhs,rhs in (('bc','flow'),('fql','flow'),('fql','bc')):
        for rule in RULES:
            values=[]
            for seed in range(3):
                a,b = picks[lhs,seed,rule],picks[rhs,seed,rule]
                values.append(dict(seed=seed, selected_index=a['index'],reference_index=b['index'],
                    states=[a['state'],b['state']],common_utility_delta=a['common_utility']-b['common_utility'],
                    native_delta={m:a['metrics'][m]-b['metrics'][m] for m in METRICS}
                        if a['state']==b['state']=='success' else None))
            pairs.append(dict(arm=lhs,reference=rhs,rule=rule,paired_seeds=3,seed_values=values,
                both_success_seed_count=sum(x['native_delta'] is not None for x in values),
                common_utility_delta=selection.stats([x['common_utility_delta'] for x in values]),
                native_full_delta={m:selection.stats([x['native_delta'][m] for x in values]) for m in METRICS}
                    if all(x['native_delta'] is not None for x in values) else None))
    return selected, groups, means, mean_groups, pairs


def score_rows(pools):
    """Flatten actor/current-flow rows after checking inherited identities."""
    result=[]
    for pool in pools:
        for row in pool['rows']:
            if any(key in row and row[key]!=pool[key] for key in ('arm','seed')):
                raise ValueError('Score row identity differs from its generator/seed pool')
            result.append({**row,'arm':pool['arm'],'seed':pool['seed']})
    return result


def audited_inputs():
    reports, bindings = {}, {}
    for name,(path,digest) in INPUTS.items():
        reports[name],bindings[name] = immutable_report(path,digest)
    actors,_ = selection.audit_snapshot(*INPUTS['actor'])
    previous = reports['selection']
    if (previous['parent']['sha256'] != INPUTS['actor'][1] or previous['stage']!='global_route_only'
            or len(previous['pools'])!=6 or any(p['state']!='complete' for p in previous['pools'])):
        raise ValueError('Existing selection does not bind these complete round-0 actor pools')
    frozen = reports['flow']
    if frozen['status']!='complete' or frozen['same_actual_eight_noises'] is not True:
        raise ValueError('Current-flow native report incomplete')
    current = next(x for x in frozen['baselines'] if x['baseline']=='current')
    audit = flow.checked_audit(flow.checked_file(frozen['export_audit']))
    if flow.collect(Path(current['experiment']), 'current', audit) != current:
        raise ValueError('Current-flow audited immutable evidence changed')
    cfg = read(current['config']['path'])
    exps = dict(bc=cfg['bc_experiment'], fql=cfg['fql_experiment'])
    if set(actors['experiment_bindings']) != set(exps.values()) or len(actors['rounds'])!=6:
        raise ValueError('Unexpected actor experiment/round scope')
    reference_cfg=actors['experiment_bindings'][exps['bc']]['config']
    scales,normalization=train_scales(reference_cfg)
    if normalization != previous['reference']['normalization'] or reference_cfg['weights'] != [1,0,0,0,0]:
        raise ValueError('Common initial TRAIN utility changed')
    pools,records=[],{}
    for arm,exp in exps.items():
        binding=actors['experiment_bindings'][exp]
        c=binding['config']
        for key in ('design','dataset_path','dataset_sha256','cache_sha256','common_checkpoints','common_checkpoint_sha256','evaluation_noise_seed','evaluation_samples'):
            if c[key]!=cfg[key]:
                raise ValueError('Flow/actor matched context differs: '+key)
        for key in ('critic_checkpoints','critic_checkpoint_sha256'):
            if c[key]!=reference_cfg[key]:
                raise ValueError('Different initial critic across actors')
        if (c['critic_variant']!='legacy' or c['weights']!=[1,0,0,0,0] or c['uncertainty_beta']!=0
                or c['offline_actor_steps']!=1000 or c['offline_critic_update'] is not False
                or c['q_weight'] != (0. if arm=='bc' else 1.) or c['alpha']!=10.):
            raise ValueError('Expected frozen WL-only offline actor controls')
        phys=dict(binding['backend_binding']);flow_phys=dict(current['physical_binding'])
        phys.pop('config_sha256');flow_phys.pop('config_sha256')
        if phys!=flow_phys or binding['normalization']!=normalization:
            raise ValueError('Different native flow/normalization')
        for record in actors['rounds']:
            if record['experiment']!=exp: continue
            if record['round']!=0 or record['seed'] not in range(3):
                raise ValueError('Only offline round0 supported')
            rows=selection.audit_pool(record,binding,context(exp),reference_cfg['weights'],scales)
            records[arm,record['seed']]=record
            pools.append(dict(arm=arm,seed=record['seed'],rows=rows))
    for stream in current['streams']:
        rows=[]
        for old in stream['rows']:
            if old['state'] not in selection.ADMITTED: raise ValueError('Unaccounted flow outcome')
            request=read(old['request']['path'])
            rows.append(dict(old,index=old['index'],predicted_q=None,macro_hpwl=request['macro_hpwl'],
                common_utility=selection.common_utility(old['metrics'],old['state'],reference_cfg['weights'],scales),
                request_sha256=old['request']['sha256'],outcome_sha256=old['outcome']['sha256'],
                actions_npz_sha256=old['actions']['sha256']))
        pools.append(dict(arm='flow',seed=stream['seed'],rows=rows))
    return pools,records,reference_cfg,dict(inputs=bindings,normalization=normalization,
        reference_weights=reference_cfg['weights'],reference_scales=scales.tolist(),failure_utility=-4.)


def run(output):
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('CPU scoring requires bounded Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    torch.use_deterministic_algorithms(True)
    sources={name:sha(ROOT/name) for name in SOURCES}
    pools,records,cfg,binding=audited_inputs()
    output=Path(output).resolve()
    protected=[p.parent for p,_ in INPUTS.values()]+[Path(p['experiment']) for p in records.values()]+[Path(read(INPUTS['flow'][0])['baselines'][0]['experiment'])]
    if any(output==p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Output overlaps immutable input tree')
    critics={}
    for seed in range(3):
        model,initial,initial_binding=initial_model(cfg,seed)
        common_path=Path(cfg['common_checkpoints'][str(seed)])
        common_binding=bound(common_path,cfg['common_checkpoint_sha256'][str(seed)])
        common=torch.load(common_path,map_location='cpu',weights_only=False)
        frozen={arm:checked_actor_critic(records[arm,seed],initial,common) for arm in ('bc','fql')}
        critics[str(seed)]=dict(initial=initial_binding,common=common_binding,actors=frozen)
        for pool in pools:
            if pool['seed']==seed: pool['rows']=score_pool(model,pool['rows'])
    noise_identity(pools)
    for pool in pools:
        pool['noise_pool']=[r['noise_sha256'] for r in pool['rows']]
        pool['recorded_cuda_equivalence']=score_equivalence(pool['rows'])
        for row in pool['rows']: row.pop('actual_noise')
    selected,groups,means,mean_groups,pairs=aggregate(pools)
    calls={str(Path(r['native']['physical_run']).resolve()) for p in pools for r in p['rows'] if r['native'].get('physical_run')}
    # Recheck sources and report files after numerical work; never emit mixed-version evidence.
    for name,digest in sources.items(): bound(ROOT/name,digest)
    for path,digest in INPUTS.values(): bound(path,digest)
    for pool in pools:
        for row in pool['rows']:
            sample=Path(row['sample_dir'])
            for name,key in (('request.json','request_sha256'),('actions.npz','actions_npz_sha256'),('outcome.json','outcome_sha256')):
                bound(sample/name,row[key])
    result=dict(schema='qstudy_fixed_initial_q_selection_v1',stage='global_route_only',development_only=True,
        fresh_confirmation=False,selected_winner=None,new_training_updates=0,new_policy_inference=0,
        new_critic_sample_evaluations=72,new_backend_calls=0,new_replay_rows=0,
        existing_unique_native_calls=len(calls),binding=binding,source_sha256=sources,critics=critics,
        pools=pools,selections=selected,selected_aggregates=groups,policy_pool_means=means,
        policy_pool_mean_aggregates=mean_groups,selected_vs_selected=pairs,
        conventions=dict(Q='CPU float32 singleton raw-proposal twin mean; -GRTWL/full384 TRAIN meanWL; beta0, no alpha/q_weight rescale',
            equivalence='Record exact argmax/ties/fullorder and CUDA deltas; common CPU scorer used for ALL arms regardless of closeness',
            uncertainty='Sample SD of three independent common/actor seed values or seed means; not 24 independent candidates',
            selection='Eight declared actual noises; tie lowest index; no filtering native failures for fixed/HPWL/Q',
            budgets='8 proposals and 8 Q scores per pool. Free selectors need1 selected backend evaluation; oracle references require8. All72 native results already measured; 0 new physical calls',
            limitations='Posthoc development; generator+fixed scorer comparison, not fresh or equal-FLOP training/generation comparison. Flow Euler10 vs one-step actors. Raw Q does not differentiate external legalization. All7 native GRT metrics, not final DRT/DRC'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'_job'+os.environ['SLURM_JOB_ID']
    snapshot=output/stamp;snapshot.mkdir(parents=True,exist_ok=False)
    atomic_json(snapshot/'report.json',result)
    selection.csv_write(snapshot/'scores.csv',score_rows(pools))
    selection.csv_write(snapshot/'selections.csv',selected)
    selection.csv_write(snapshot/'selected_vs_selected.csv',pairs)
    (snapshot/'source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(snapshot/'completed.json',dict(status='completed',report_sha256=sha(snapshot/'report.json'),
        artifacts={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()},score_count=72,new_backend_calls=0))
    print(json.dumps(dict(snapshot=str(snapshot),score_count=72,new_backend_calls=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)


if __name__=='__main__': main()
