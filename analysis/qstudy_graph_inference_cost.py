"""Warm batch-one latency of the completed graph actors and their Euler10 teacher.

This is a cost measurement, with eight saved development inputs, not a new
placement evaluation. No Q call, noise draw, training, DEF or backend is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')
VERSION = 'qstudy_graph_inference_cost_v1'
SEEDS = (0, 1, 2)
LABELS = ('bc_r2_actor1', 'fql_frozen_r2_actor1', 'common_behavior_euler10')
SCOPES = ('model_only', 'legalizer_only', 'full_generation')
CONFIGS = (
    (BASE/'actors_v1/bc_frozen/config.json', 'a82c8145a9293e416f1e180eb799f6aa587d6df5b59450f0e5029b25ae59f966'),
    (BASE/'actors_v1/legacy_frozen/config.json', 'e7535dc73cdeccf58caf2879cd08c43f61a3a7442c1ab15f5533f3c26897edb3'),
    (BASE/'flow_eval_v1/config.json', 'bff6403f32aeddaced445a664b42a56319516e1111e101bbce874420861bbe7c'))
FLOW_DECLARATION_SHA = '58beb2c7568657891bef18e8ae604ca53b6a72f0ee36ffd422cf78e8a8425635'
WARMUP, REPEATS, COUNT, THREADS = 10, 32, 8, 2
SOURCE_NAMES = ('analysis/qstudy_graph_inference_cost.py',
                'analysis/qstudy_graph_inference_cost.slurm',
                'docs/qstudy_graph_inference_cost.md')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def bind(path, expected=None):
    path = Path(path).absolute()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Changed graph-latency evidence: '+str(path))
    return dict(path=str(path), sha256=digest)


def declared_checkpoint(path, expected):
    """Prepare binds the producer's digest without reading bulk weights."""
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Invalid declared checkpoint SHA')
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return dict(path=str(Path(path).absolute()), sha256=expected)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp_'+str(os.getpid()))
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes()).hexdigest()


def runtime():
    # Resolve the local analysis namespace before baseline adds upstream paths.
    from analysis.placement_metrics import MacroHPWL, legality
    from baseline.api import CHECKPOINT, RUNTIME, legalize, load_problem
    from fql.graph_o2o_models import GraphPolicy
    return SimpleNamespace(**locals())


def software():
    import torch_geometric
    return dict(python=platform.python_version(), python_abi=f'{sys.version_info.major}.{sys.version_info.minor}',
        torch=str(torch.__version__),
        numpy=np.__version__, torch_geometric=torch_geometric.__version__, cuda=torch.version.cuda,
        executable=str(Path(sys.executable).absolute()))


def software_contract(value):
    # All compared methods execute in one allocation. Record the complete patch
    # version, but compatible CPython patch versions do not change the ABI gate.
    return {key: value[key] for key in
        ('python_abi', 'torch', 'numpy', 'torch_geometric', 'cuda', 'executable')}


def source_paths():
    runtime()
    result = {name: ROOT/name for name in SOURCE_NAMES}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename or not str(filename).endswith('.py'):
            continue
        path = Path(filename).absolute()
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            continue
        if not name.startswith('.venv/'):
            result[name] = path
    for name in ('config_eval.yaml', 'model/att-gnn.yaml', 'model/conv-layer/gat.yaml',
                 'model/size/large.yaml', 'guidance/none.yaml', 'legalizer/none.yaml', 'sc_placer/sgd.yaml'):
        name = 'upstream/flowplace/flow_matching/configs/'+name
        result[name] = ROOT/name
    return result


def actor_metadata(config_path, cfg, seed):
    """Recheck the completed r0→r2 metadata chain; do not torch.load or hash .pt."""
    result, previous = [], None
    config_sha = sha(config_path)
    for round_number in range(3):
        folder = config_path.parent/f'seed_{seed}/round_{round_number}'
        done, binding = read(folder/'completed.json'), read(folder/'binding.json')
        expected = dict(config_sha256=config_sha, dataset_sha256=cfg['dataset_sha256'],
            common_sha256=cfg['common_checkpoint_sha256'][str(seed)],
            initial_critic_sha256=cfg['critic_checkpoint_sha256'][str(seed)], seed=seed, round=round_number)
        if (done['status'] != 'completed' or done['binding'] != binding or
                any(binding[k] != v for k, v in expected.items()) or
                done['steps'] != (1000 if round_number == 0 else 600)):
            raise ValueError('The fixed completed main actor prefix changed')
        parent = None if previous is None else dict(checkpoint_sha256=previous['checkpoint']['sha256'],
            completed_sha256=previous['completion']['sha256'], binding_sha256=previous['binding']['sha256'])
        if binding['parent'] != parent:
            raise ValueError('Actor r0→r2 parent chain changed')
        expected_replay = {str((config_path.parent/f'seed_{seed}/round_{r}/requests'/cfg['arm']/
            'collect'/str(i)/'outcome.json').resolve()) for r in range(round_number) for i in range(COUNT)}
        actual = []
        for entry in binding['replay']:
            out = read(bind(entry['path'], entry['sha256'])['path'])
            if (out['status'] != entry['status'] or out['arm'] != cfg['arm'] or out['seed'] != seed or
                    out['role'] != 'collect' or not out['training_feedback_eligible'] or
                    out['round'] >= round_number):
                raise ValueError('Actor prefix has foreign/evaluation replay')
            actual.append(str(Path(entry['path']).resolve()))
        if len(actual) != len(expected_replay) or set(actual) != expected_replay:
            raise ValueError('Actor prefix replay denominator changed')
        previous = dict(round=round_number, checkpoint=declared_checkpoint(folder/'checkpoint.pt', done['checkpoint_sha256']),
            completion=bind(folder/'completed.json'), binding=bind(folder/'binding.json'))
        result.append(previous)
    return result


def export_metadata(config_path, cfg, seed, round_number, checkpoint):
    folder = config_path.parent/f'seed_{seed}/round_{round_number}/requests'
    tasks = folder/'tasks.jsonl'; marker = read(folder/'completed.json')
    if (marker['status'] != 'completed' or marker['count'] != COUNT or
            marker['tasks_sha256'] != sha(tasks) or marker['config_sha256'] != sha(config_path)):
        raise ValueError('Completed eight-noise export changed')
    rows = [json.loads(line) for line in tasks.read_text().splitlines()]
    if len(rows) != COUNT or sorted(r['index'] for r in rows) != list(range(COUNT)):
        raise ValueError('Expected every predeclared noise index exactly once')
    result = []
    for row in sorted(rows, key=lambda r: r['index']):
        sample = Path(row['sample_dir']); request = bind(sample/'request.json')
        if (read(request['path']) != row or row['role'] != 'evaluation' or row['training_feedback_eligible'] or
                row['seed'] != seed or row['round'] != round_number or row['arm'] != cfg['arm'] or
                row['noise_seed'] != 4600000 or row['source_checkpoint'] != checkpoint['path'] or
                row['checkpoint_sha256'] != checkpoint['sha256'] or row['config_sha256'] != sha(config_path)):
            raise ValueError('Export identity/noise/eligibility changed')
        result.append(dict(index=row['index'], request=request, actions=bind(sample/'actions.npz'),
            noise_sha256=row['noise_sha256'], raw_action_sha256=row['raw_action_sha256'],
            action_sha256=row['action_sha256']))
    return dict(tasks=bind(tasks), completion=bind(folder/'completed.json'), rows=result)


def prepare(output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Prepare the measurement in a bounded Slurm CPU allocation')
    rt = runtime()
    configs = [read(bind(path, digest)['path']) for path, digest in CONFIGS]
    for cfg in configs:
        if (cfg['design'] != 'bp_fe' or cfg['seeds'] != list(SEEDS) or cfg['flow_steps'] != 10 or
                cfg['online_flow_update'] is not False or cfg['evaluation_noise_seed'] != 4600000 or
                cfg['evaluation_samples'] != COUNT):
            raise ValueError('Unexpected fixed main/behavior-flow configuration')
    for cfg in configs[1:]:
        for key in ('dataset_sha256', 'cache_sha256', 'common_checkpoints', 'common_checkpoint_sha256'):
            if cfg[key] != configs[0][key]:
                raise ValueError('Different graph/common initialization across methods')
    declaration = bind(CONFIGS[2][0].with_name('declaration.json'), FLOW_DECLARATION_SHA)
    declared = read(declaration['path'])
    if declared['expected_requests'] != 24 or declared['source_noise_seed'] != 4600000:
        raise ValueError('Different original behavior evaluation declaration')
    seeds = []
    for seed in SEEDS:
        actors = []
        for (path, _), cfg in zip(CONFIGS[:2], configs[:2]):
            chain = actor_metadata(path, cfg, seed)
            actors.append(dict(chain=chain, export=export_metadata(path, cfg, seed, 2, chain[-1]['checkpoint'])))
        common = declared_checkpoint(configs[0]['common_checkpoints'][str(seed)],
            configs[0]['common_checkpoint_sha256'][str(seed)])
        flow_export = export_metadata(CONFIGS[2][0], configs[2], seed, 0, common)
        seeds.append(dict(seed=seed, actors=actors, common=common, flow_export=flow_export))
    files = source_paths()
    plan = dict(version=VERSION, seeds=seeds, configs=[bind(p, h) for p, h in CONFIGS],
        flow_declaration=declaration, design='bp_fe', labels=list(LABELS), scopes=list(SCOPES),
        graph_cache=declared_checkpoint(rt.RUNTIME/'benchmark_cache/bp_fe.pt', configs[0]['cache_sha256']),
        construction_checkpoint=declared_checkpoint(rt.CHECKPOINT,
            '0fe61ef666c7ff34c32b1ea1b8fe23a1c8b45151955f0f16366a121fde5fd359'),
        batch_size=1, distinct_noises=COUNT, original_noise_seed=4600000, warmup=WARMUP,
        repeats=REPEATS, repeats_per_noise=4, threads=THREADS, precision='torch.float32',
        expected_nfe=dict(zip(LABELS, (1, 1, 10))), software=software(), gpu_name_contains='L20',
        selection='all three fixed completed R2 seeds and all eight old development arrays; no selection',
        sources={name: sha(path) for name, path in sorted(files.items())},
        timing_order='scope, repetition, method rotated by repetition modulo3; seed0,1,2',
        counts=dict(training_updates=0, backend_calls=0, new_noise_draws=0),
        excluded=['checkpoint loading', 'noise generation', 'host/device transfer', 'Q scoring',
                  'coordinate/geometry audit', 'hashing', 'DEF and file I/O', 'placement/routing backend'],
        official_flowplace20_hard_benchmark=False)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    for name, source in files.items():
        target = output/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(output/'plan.json', plan)
    print(json.dumps(dict(plan=bind(output/'plan.json'), training_updates=0, backend_calls=0)), flush=True)


def checked_plan(path, expected):
    path = Path(path).resolve(); plan = read(bind(path, expected)['path'])
    required = dict(version=VERSION, labels=list(LABELS), scopes=list(SCOPES), batch_size=1,
        distinct_noises=COUNT, original_noise_seed=4600000, warmup=WARMUP, repeats=REPEATS,
        repeats_per_noise=4, threads=THREADS, expected_nfe=dict(zip(LABELS, (1, 1, 10))))
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError('Measurement protocol changed')
    for name, digest in plan['sources'].items():
        bind(ROOT/name, digest); bind(path.parent/'sources'/name, digest)
    if software_contract(plan['software']) != software_contract(software()):
        raise ValueError('Prepared software differs from measurement runtime')
    return plan


def load_arrays(export):
    bind(export['tasks']['path'], export['tasks']['sha256'])
    bind(export['completion']['path'], export['completion']['sha256'])
    arrays = []
    for row in export['rows']:
        bind(row['request']['path'], row['request']['sha256'])
        bind(row['actions']['path'], row['actions']['sha256'])
        with np.load(row['actions']['path'], allow_pickle=False) as saved:
            local = {key: saved[key].copy() for key in ('noise', 'raw_action', 'action')}
        shape = local['noise'].shape
        if (local['noise'].dtype != np.float32 or len(shape) != 2 or shape[-1] != 2 or
                any(a.shape != shape or not np.isfinite(a).all() for a in local.values()) or
                np.any(np.abs(local['noise']) > 1) or
                any(array_sha(local[key]) != row[hash_key] for key, hash_key in
                    (('noise', 'noise_sha256'), ('raw_action', 'raw_action_sha256'), ('action', 'action_sha256')))):
            raise ValueError('Saved coordinate/noise arrays differ from their frozen row')
        arrays.append(local)
    if len(arrays) != COUNT or len({array_sha(a['noise']) for a in arrays}) != COUNT:
        raise ValueError('Need eight distinct actual saved noises')
    return arrays


def equal_state(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k].cpu(), right[k].cpu()) for k in left)


def state_sha(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(x.dtype).encode())
        result.update(str(tuple(x.shape)).encode()); result.update(x.numpy().tobytes())
    return result.hexdigest()


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def measured_call(function, argument, device, clock=time.perf_counter):
    synchronize(device); start = clock()
    result = function(argument)
    synchronize(device); elapsed = clock()-start
    if elapsed <= 0 or not np.isfinite(elapsed):
        raise ValueError('Invalid synchronized wall-clock duration')
    return result, elapsed


def count_nfe(backbone, function, argument, expected):
    count = [0]
    def hook(*_): count[0] += 1
    handle = backbone.register_forward_hook(hook)
    try:
        output = function(argument)
    finally:
        handle.remove()
    if count[0] != expected:
        raise ValueError('Actual graph-backbone forward count differs from declared NFE')
    return output, count[0]


def make_case(policy, noises, label, legalizer, problem):
    if label not in LABELS:
        raise ValueError('Unknown measured method')
    generator = (lambda noise: policy.sample_flow(noise, steps=10)) if label == LABELS[2] else (
        lambda noise: policy(noise, clip=True))
    legal = lambda raw: legalizer(raw[0], problem).unsqueeze(0)
    return dict(label=label, policy=policy, expected_nfe=10 if label == LABELS[2] else 1,
        noises=noises, generate=generator, legalize=legal,
        full=lambda noise: legal(generator(noise)))


def audit_case(case, saved, legality, sizes):
    policy = case['policy']; result, raws = [], []
    backbone = policy.model._reverse_model
    for index, noise in enumerate(case['noises']):
        raw, nfe = count_nfe(backbone, case['generate'], noise, case['expected_nfe'])
        executed, legal_nfe = count_nfe(backbone, case['legalize'], raw, 0)
        full, full_nfe = count_nfe(backbone, case['full'], noise, case['expected_nfe'])
        if not torch.equal(executed, full):
            raise ValueError('Full generation and the two explicit stages disagree')
        fixed = ~policy.movable
        if (not torch.equal(raw[0, fixed], policy.fixed[fixed]) or
                not torch.equal(executed[0, fixed], policy.fixed[fixed])):
            raise ValueError('Inference or legalizer moved a fixed node')
        r, a = raw[0].cpu().numpy(), executed[0].cpu().numpy()
        if not np.isfinite(r).all() or not np.isfinite(a).all():
            raise ValueError('Nonfinite generated coordinates')
        result.append(dict(index=index, noise_sha256=array_sha(noise.cpu().numpy()),
            raw_action_sha256=array_sha(r), executed_action_sha256=array_sha(a),
            fixednode_count=int(fixed.sum()), fixednodes_preserved=True,
            raw_geometry=legality(r, sizes), executed_geometry=legality(a, sizes),
            nfe=dict(model_only=nfe, legalizer_only=legal_nfe, full_generation=full_nfe),
            saved_raw_bitwise_equal=np.array_equal(r, saved[index]['raw_action']),
            saved_executed_bitwise_equal=np.array_equal(a, saved[index]['action']),
            saved_raw_max_abs_delta=float(np.max(np.abs(r-saved[index]['raw_action']))),
            saved_executed_max_abs_delta=float(np.max(np.abs(a-saved[index]['action'])))))
        raws.append(raw)
    case['raws'] = raws
    return result


def benchmark_cases(cases, warmup=WARMUP, repeats=REPEATS, device='cuda', clock=time.perf_counter):
    if warmup < 0 or repeats < 1 or any(len(c['noises']) != COUNT or len(c['raws']) != COUNT for c in cases):
        raise ValueError('Invalid measurement count/input population')
    rows = []
    for scope in SCOPES:
        for repeat in range(warmup+repeats):
            index = repeat % COUNT if repeat < warmup else (repeat-warmup) % COUNT
            for position in range(len(cases)):
                case = cases[(position+repeat) % len(cases)]
                function = {'model_only': case['generate'], 'legalizer_only': case['legalize'],
                            'full_generation': case['full']}[scope]
                value = case['raws'][index] if scope == 'legalizer_only' else case['noises'][index]
                if repeat < warmup:
                    function(value)
                    synchronize(device)
                    continue
                _, seconds = measured_call(function, value, device, clock)
                rows.append(dict(label=case['label'], scope=scope, repeat=repeat-warmup,
                    noise_index=index, seconds=seconds,
                    expected_backbone_nfe=0 if scope == 'legalizer_only' else case['expected_nfe']))
    return rows


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), sample_sd=float(x.std(ddof=1)) if len(x)>1 else None,
        median=float(np.median(x)), p95=float(np.quantile(x, .95)), minimum=float(x.min()), maximum=float(x.max()))


def summarize(rows):
    result = []
    for label in LABELS:
        for scope in SCOPES:
            seed_rows = []
            for seed in SEEDS:
                local = [r for r in rows if r['seed'] == seed and r['label'] == label and r['scope'] == scope]
                if len(local) != REPEATS or any(sum(r['noise_index']==i for r in local) != 4 for i in range(COUNT)):
                    raise ValueError('Timing denominator is not32 calls/eight noises/four repeats')
                seed_rows.append(dict(seed=seed, milliseconds=describe([r['seconds']*1000 for r in local])))
            result.append(dict(label=label, scope=scope, seeds=seed_rows,
                three_seed_mean_ms=describe([r['milliseconds']['mean'] for r in seed_rows]),
                replicate_note='32 calls reuse eight layouts, not32 independent layouts or training seeds'))
    return result


def run(plan_path, expected, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('The real graph measurement requires a Slurm GPU')
    plan = checked_plan(plan_path, expected); rt = runtime()
    torch.set_num_threads(THREADS); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if plan['gpu_name_contains'] not in torch.cuda.get_device_name():
        raise ValueError('Measurement requires the prospectively specified L20 GPU')
    for item in plan['configs']+[plan['flow_declaration'], plan['graph_cache'], plan['construction_checkpoint']]:
        bind(item['path'], item['sha256'])
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    inputs, all_rows, audits, weights = [], [], [], []
    shared_noises = None
    for seed_entry in plan['seeds']:
        seed = seed_entry['seed']; arrays = [load_arrays(a['export']) for a in seed_entry['actors']]
        arrays.append(load_arrays(seed_entry['flow_export']))
        hashes = [[array_sha(a['noise']) for a in group] for group in arrays]
        if any(h != hashes[0] for h in hashes[1:]) or (shared_noises is not None and shared_noises != hashes[0]):
            raise ValueError('Methods/seeds do not share the same eight actual noise arrays')
        shared_noises = hashes[0]
        common_item = seed_entry['common']; bind(common_item['path'], common_item['sha256'])
        common = torch.load(common_item['path'], map_location='cpu', weights_only=False)
        problem = rt.load_problem('bp_fe', 'cuda'); metric = rt.MacroHPWL(problem.benchmark_info)
        noises = [torch.from_numpy(a['noise']).cuda().unsqueeze(0) for a in arrays[0]]
        cases = []
        for label, entry, stored in zip(LABELS, seed_entry['actors']+[None], arrays):
            if entry is None:
                state = common['flow']; source = common_item
            else:
                for prefix in entry['chain']:
                    bind(prefix['completion']['path'], prefix['completion']['sha256'])
                    bind(prefix['binding']['path'], prefix['binding']['sha256'])
                last = entry['chain'][-1]; source = last['checkpoint']; bind(source['path'], source['sha256'])
                saved = torch.load(source['path'], map_location='cpu', weights_only=False)
                if saved['binding'] != read(last['binding']['path']) or not equal_state(saved['flow'], common['flow']):
                    raise ValueError('R2 checkpoint binding/frozen common teacher changed')
                state = saved['actor']
            policy = rt.GraphPolicy(problem).eval()
            if not torch.equal(state['fixed'], policy.fixed.cpu()) or not torch.equal(state['movable'], policy.movable.cpu()):
                raise ValueError('Checkpoint belongs to another graph/fixed-node partition')
            policy.load_state_dict(state, strict=True)
            for parameter in policy.parameters(): parameter.requires_grad_(False)
            case = make_case(policy, noises, label, rt.legalize, problem)
            case['source_state_sha256'] = state_sha(policy.state_dict())
            with torch.no_grad():
                evidence = audit_case(case, stored, rt.legality, metric.sizes)
            audits.append(dict(seed=seed, label=label, checkpoint=source, rows=evidence))
            cases.append(case)
        synchronize('cuda')
        with torch.no_grad():
            rows = benchmark_cases(cases)
        all_rows.extend(dict(r, seed=seed) for r in rows)
        for case in cases:
            after = state_sha(case['policy'].state_dict())
            if after != case['source_state_sha256']:
                raise ValueError('Measurement changed model weights/buffers')
            weights.append(dict(seed=seed, label=case['label'], state_sha256=after, unchanged=True))
        inputs.append(dict(seed=seed, noises=hashes[0]))
        del cases, case, policy, saved, common, problem, noises
        torch.cuda.empty_cache()
    driver = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
        text=True, capture_output=True, check=True)
    report = dict(version=VERSION, status='completed', plan=bind(plan_path, expected),
        sources=plan['sources'], software=software(), prepared_software=plan['software'],
        software_compatibility='Python major.minor ABI and exact torch/numpy/pyg/CUDA/executable; patch recorded separately',
        slurm_job_id=os.environ['SLURM_JOB_ID'],
        hardware=dict(gpu=torch.cuda.get_device_name(), properties=str(torch.cuda.get_device_properties(0)),
                      nvidia_smi=driver.stdout, cpu_threads=torch.get_num_threads()),
        precision=dict(dtype='float32', autocast=False, deterministic_algorithms=True,
            cudnn_deterministic=True, cudnn_benchmark=False, matmul_tf32=False, cudnn_tf32=False),
        clock='perf_counter host wall, CUDA synchronized immediately before/after every timed call',
        batch_size=1, noise_seed=4600000, eight_actual_noises=inputs, warmup_per_scope_method=WARMUP,
        repeats_per_scope_method=REPEATS, repeats_per_noise=4, independent_training_seeds=3,
        summaries=summarize(all_rows), output_audits=audits, immutable_weights=weights,
        call_accounting=dict(timed_calls=len(all_rows), warmup_calls=3*3*3*WARMUP,
            untimed_audit_calls=3*3*COUNT*3,
            timed_backbone_nfe=sum(r['expected_backbone_nfe'] for r in all_rows),
            warmup_backbone_nfe=3*(1+1+10)*2*WARMUP,
            actual_hook_counted_audit_nfe=sum(sum(r['nfe'].values()) for a in audits for r in a['rows']),
            hook_scope='Actual forwards counted in untimed probes; timing runs have no hook overhead',
            repetition_is_not_new_data='All calls reuse the same eight actual noises; no EDA labels are collected'),
        excluded=plan['excluded'], backend_calls=0, training_updates=0, new_noise_draws=0,
        warnings=['Euler10 behavior sampler is not the released FlowPlace20-hard sampler or paper speed',
            'Full generation is measured directly, not a sum of separate stage medians',
            'Legalizer-only includes constructing a fresh original Legalizer for every call',
            'Warm resident-input inference excludes loading, transfers, DEF/I/O, Q and physical backend',
            'Geometry/hash checks are outside timing; no layouts are filtered for legality or Q'])
    atomic_json(output/'report.json', report)
    with (output/'timings.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    atomic_json(output/'completed.json', dict(status='completed', report=bind(output/'report.json'),
        timings=bind(output/'timings.csv'), plan=bind(plan_path, expected), training_updates=0, backend_calls=0))
    print(json.dumps(dict(report=bind(output/'report.json'), timings=bind(output/'timings.csv'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    else: run(args.plan, args.plan_sha256, args.output)


if __name__ == '__main__':
    main()
