"""Prospective scalar-B replay and exact continuation core; no production runner.

This is an explicitly controlled FQL adaptation with a fixed behavior teacher.
It does not export collect layouts, schedule GRT/DRT, choose rewards/budgets, or
turn the existing offline-only adapters into online pipelines. Callers must
predeclare their own collect population and pass externally pinned artifacts.
Native reward/PPA remain unchanged; only ``derived_B`` is a new training label.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .qstudy_data import action_sha, bind, sha
from .qstudy_models import frozen_critic

VERSION = 'qstudy_scalar_online_core_v1'
Q_UNITS = '-mean predicted B; no mean(B) renormalization'
MODEL_KEYS = ('flow', 'actor', 'critic')
GENERATOR_KEYS = ('q', 'actor', 'flow')


def slurm():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Scalar online numerical work requires explicit Slurm')


def read(path):
    return json.loads(Path(path).read_text())


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def screen_for(kind):
    if kind == 'rule':
        from . import qstudy_rule_reward_screen as screen
    elif kind == 'final':
        from . import qstudy_final_feedback_screen as screen
    else:
        raise ValueError('Scalar source must be rule or final')
    return screen


def schema(kind, spec, *, final_conditions=None):
    """Bind an already fitted full-TRAIN specification; never estimate scales."""
    screen_for(kind).validate_spec(spec)
    if kind == 'final' and not isinstance(final_conditions, dict):
        raise ValueError('Final targets require the original audited TRAIN power conditions')
    if kind == 'rule' and final_conditions is not None:
        raise ValueError('Rule schema cannot claim final-route admission')
    return dict(version=VERSION, kind=kind, utility_spec=copy.deepcopy(spec),
        utility_spec_sha256=json_sha(spec), final_conditions=copy.deepcopy(final_conditions),
        q_units=Q_UNITS, teacher_fixed=True, native_reward_unchanged=True)


def validate_schema(value):
    expected = schema(value['kind'], value['utility_spec'], final_conditions=value['final_conditions'])
    if value != expected:
        raise ValueError('Fixed scalar TRAIN schema/scale identity changed')
    return value


def targets(value, cost_values, statuses, cost_mask=None):
    """Success and confirmed B4 failures share one positive scalar target/mask.

    Missing active data and infrastructure statuses produce NaN/invalid, never
    a failure target or fabricated PPA. Frozen label helpers define the exact
    same utility used by the corresponding offline critic.
    """
    value = validate_schema(value)
    costs = np.asarray(cost_values, dtype=np.float64)
    statuses = np.asarray(statuses)
    screen = screen_for(value['kind'])
    if costs.ndim != 2 or costs.shape != (len(statuses), len(screen.COST_NAMES)):
        raise ValueError('Scalar target rows/cost columns differ')
    mask = np.isfinite(costs) if cost_mask is None else np.asarray(cost_mask, bool)
    if mask.shape != costs.shape:
        raise ValueError('Scalar cost observation mask differs')
    data = dict(cost_values=costs, cost_mask=mask, statuses=statuses)
    result = screen.labels(data, np.arange(len(statuses)), value['utility_spec'])
    return {**result, 'derived_B': result['cost'][:, None], 'mask': result['valid'][:, None],
            'utility_spec_sha256': value['utility_spec_sha256']}


def scalar_q(critic, actions):
    predicted = critic(actions)
    if predicted.ndim != 3 or predicted.shape[0] != len(actions) or predicted.shape[2] != 1:
        raise ValueError('Scalar critic must return [batch, ensemble, 1] positive-cost predictions')
    # Do not bound/clip predicted costs: that would hide extrapolation errors.
    return -predicted.mean((1, 2))


def validate_collect_identity(row, outcome, expected):
    """Pure identity barrier, in addition to the actual native validators below."""
    for key in ('arm', 'seed', 'round', 'config_sha256', 'checkpoint_sha256'):
        if row.get(key) != expected[key] or outcome.get(key) != expected[key]:
            raise ValueError('Replay belongs to a foreign policy/parent: '+key)
    for item in (row, outcome):
        if (item.get('role') != 'collect' or item.get('training_feedback_eligible') is not True
                or item.get('terminal_mask') != 0 or item.get('index') != row.get('index')):
            raise ValueError('Evaluation or nonterminal records cannot enter scalar replay')
    if row.get('noise_seed') != expected['collection_noise_seed']:
        raise ValueError('Collect stream differs from its prospective declaration')
    if row['noise_sha256'] in expected['evaluation_noise_sha256']:
        raise ValueError('Evaluation noise cannot be replayed under a collect alias')
    for key in ('raw_action_sha256', 'action_sha256', 'noise_sha256'):
        if row.get(key) != outcome.get(key):
            raise ValueError('Replay outcome changed the actual proposal/execution/noise')
    if (action_sha(row['raw_action']) != row['raw_action_sha256'] or
            action_sha(row['executed_action']) != row['action_sha256']):
        raise ValueError('Replay inline coordinates differ from their fingerprints')


def _costs(value, cache, executed, native, final):
    from analysis.placement_metrics import MacroHPWL
    from analysis.qstudy_rule_audit import rules, structure
    context, _ = structure(cache)
    executed = np.asarray(executed, dtype=np.float32).reshape(len(context['names']), 2)
    score = rules(executed, context, MacroHPWL(cache))
    experts = [score[k] for k in ('edge_clearance', 'cluster_bbox_fraction',
                                  'dataflow_distance', 'narrow_channel_risk')]
    if value['kind'] == 'rule':
        result = [native['rWL_um'], score['macro_hpwl'], *experts]
    else:
        result = [native['rWL_um'], final['DRT_WL'], max(-final['TNS'], 0.),
                  final['Power'], final['StdCellArea'], *experts]
    return np.asarray([np.nan if x is None else x for x in result], dtype=np.float64)


def checked_final(path, digest, sample, grt, value):
    """Real DRT + same-source geometry + native power QA; never launches EDA."""
    from .graph_o2o_final_evidence import audit_final_outcome
    from analysis import qstudy_final_power_audit as power
    identity = bind(path, digest)
    out = read(path)
    if (out.get('status') != 'completed' or
            Path(out['spec']['source_outcome']).resolve() != (sample/'outcome.json').resolve() or
            out['spec']['source_outcome_sha256'] != sha(sample/'outcome.json') or
            Path(out['spec']['source_run']).resolve() != Path(grt['physical_run']).resolve() or
            out['spec']['source_manifest_sha256'] != grt['physical_manifest_sha256']):
        raise ValueError('Final feedback lacks the exact completed collect GRT source')
    audit = audit_final_outcome(out, design='bp_fe', route_cap=20)
    captured = power.Capture()
    qa = power.checked_native_power(out, captured)
    if qa['common_conditions'] != value['final_conditions']:
        raise ValueError('Online final feedback changed fixed TRAIN tool/clock/power conditions')
    captured.verify()
    if sha(path) != identity['sha256']:
        raise ValueError('Final feedback changed during native verification')
    return out['metrics'], dict(outcome=identity, physical_identity=audit['physical_identity'],
                               power_evidence_sha256=captured.files)


def admit_collect(experiment, *, seed, source_round, count, tasks_sha256,
                  config_sha256, checkpoint_sha256, collection_noise_seed,
                  evaluation_noise_sha256, value, cache, final_outcomes=None):
    """Read the complete predeclared own-collect population, retaining invalids.

    ``final_outcomes[index]`` is an externally pinned {path,sha256}, not a
    best-layout lookup. Missing final evidence remains invalid and prevents a
    ready replay. GRT confirmed geometry/physical failure already proves B4.
    Every successful final outcome is independently audited even for grt_wl.
    """
    slurm()
    from . import qstudy_backend as backend
    from .graph_o2o_backend import load_requests
    value = validate_schema(value)
    experiment = Path(experiment).resolve()
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or source_round < 0:
        raise ValueError('Explicit positive collect count/nonnegative source round required')
    if sha(experiment/'config.json') != config_sha256:
        raise ValueError('Prospective physical config binding changed')
    cfg, verification, physical_binding = backend.context(experiment)
    rows, tasks = load_requests(experiment, seed, source_round)
    bind(tasks, tasks_sha256)
    collect = [r for r in rows if r['role'] == 'collect']
    if len(collect) != count or sorted(r['index'] for r in collect) != list(range(count)):
        raise ValueError('Complete collect denominator/index population changed')
    evaluation = set(evaluation_noise_sha256)
    if not evaluation or len(evaluation) != len(evaluation_noise_sha256):
        raise ValueError('Explicit distinct held-out evaluation-noise fingerprints are required')
    if any(r['noise_sha256'] not in evaluation for r in rows if r['role'] == 'evaluation'):
        raise ValueError('Exported evaluation pool differs from the excluded pool')
    final_outcomes = {} if final_outcomes is None else final_outcomes
    if any(k not in range(count) for k in final_outcomes):
        raise ValueError('Final evidence contains undeclared collect indices')
    expected = dict(arm=cfg['arm'], seed=seed, round=source_round, config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256, collection_noise_seed=collection_noise_seed,
        evaluation_noise_sha256=evaluation)
    records, arrays, values, statuses = [], [], [], []
    for row in sorted(collect, key=lambda r:r['index']):
        sample = backend.checked_request(row, SimpleNamespace(seed=seed, round=source_round), cfg, physical_binding)
        if not (sample/'outcome.json').exists():
            out, status = None, 'missing'
        else:
            out = read(sample/'outcome.json')
            validate_collect_identity(row, out, expected)
            # backend.verify_run would otherwise create this geometry audit.
            if out.get('physical_run') and not (Path(out['physical_run'])/'backend_macro_audit.json').exists():
                raise ValueError('Read-only replay refuses an absent native geometry audit')
            backend.checked_existing(out, row, sample, tasks, physical_binding, cfg, verification)
            status = out['status']
        cost = np.full(len(screen_for(value['kind']).COST_NAMES), np.nan)
        final, final_proof = None, None
        if status == 'success':
            if value['kind'] == 'final':
                final_binding = final_outcomes.get(row['index'])
                if final_binding is None:
                    status = 'missing_final'
                else:
                    final, final_proof = checked_final(final_binding['path'], final_binding['sha256'], sample, out, value)
                    status = 'physical_failure' if final['DRC'] > 0 else 'success'
            if status in ('success', 'physical_failure'):
                cost = _costs(value, cache, row['executed_action'], out['metrics'], final)
        record = dict(index=row['index'], status=status, raw_native_reward=None if out is None else out['reward'],
            native_grt_metrics=None if out is None else copy.deepcopy(out['metrics']),
            native_final_metrics=copy.deepcopy(final), final_proof=final_proof,
            request=bind(sample/'request.json'), outcome=None if out is None else bind(sample/'outcome.json'),
            action_sha256=row['action_sha256'], raw_action_sha256=row['raw_action_sha256'],
            noise_sha256=row['noise_sha256'])
        records.append(record); arrays.append(row['raw_action']); values.append(cost); statuses.append(status)
    derived = targets(value, values, statuses)
    for i, record in enumerate(records):
        record.update(derived_B=float(derived['cost'][i]) if derived['valid'][i] else None,
                      success=bool(derived['success'][i]), scalar_target_valid=bool(derived['valid'][i]))
    return dict(x=torch.tensor(np.asarray(arrays), dtype=torch.float32), **derived,
        records=records, ready=bool(derived['valid'].all()),
        binding=dict(version=VERSION, config=bind(experiment/'config.json', config_sha256),
            tasks=bind(tasks, tasks_sha256), count=count, seed=seed, source_round=source_round,
            checkpoint_sha256=checkpoint_sha256, collection_noise_seed=collection_noise_seed,
            excluded_evaluation_noise_sha256=sorted(evaluation), schema=copy.deepcopy(value),
            records=records))


def load_offline_source(kind, experiment, seed, expected_checkpoint_sha256):
    """Load/verify a completed scalar round0; no graph forward or GPU is needed."""
    slurm()
    if kind == 'rule':
        from . import qstudy_rule_reward_actor as adapter
        prefix = 'rule'
    elif kind == 'final':
        from . import qstudy_final_feedback_actor as adapter
        prefix = 'final_feedback'
    else:
        raise ValueError('Unknown scalar source kind')
    experiment = Path(experiment).resolve()
    cfg = adapter.validate_config(experiment, seed, 0)
    plan = cfg[prefix+'_plan']
    screen = screen_for(kind)
    refit = (screen.checked_refit(plan['path'], cfg[prefix+'_objective'], seed) if kind == 'rule' else
             screen.checked_refit(plan['path'], cfg[prefix+'_objective'], seed, plan['sha256']))
    if refit['spec'] != cfg[prefix+'_utility_spec']:
        raise ValueError('Offline critic spec differs from its prepared full-TRAIN spec')
    for key in ('checkpoint', 'completion', 'normalization'):
        if refit[key] != cfg[prefix+'_refits'][str(seed)][key]:
            raise ValueError('Offline scalar refit source changed: '+key)
    folder = experiment/f'seed_{seed}'/'round_0'
    checkpoint = bind(folder/'checkpoint.pt', expected_checkpoint_sha256)
    done, binding = read(folder/'completed.json'), read(folder/'binding.json')
    saved = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
    if (done.get('status') != 'completed' or done.get('steps') != cfg['offline_actor_steps'] or
        done.get('checkpoint_sha256') != checkpoint['sha256'] or done.get('binding') != binding or
        saved.get('binding') != binding or saved.get('step') != cfg['offline_actor_steps'] or
        binding.get('round') != 0 or binding.get('seed') != seed or binding.get('parent') is not None or
        binding.get('replay') != [] or binding.get('config_sha256') != sha(experiment/'config.json') or
        done.get('offline_critic_frozen') is not True or done.get('online_flow_update') is not False):
        raise ValueError('Scalar parent is not the exact completed offline round0 prefix')
    common = bind(cfg['common_checkpoints'][str(seed)], cfg['common_checkpoint_sha256'][str(seed)])
    if (binding.get('common_sha256') != common['sha256'] or
            binding.get('initial_critic_sha256') != cfg['critic_checkpoint_sha256'][str(seed)] or
            binding.get('dataset_sha256') != sha(cfg['dataset_path'])):
        raise ValueError('Offline parent initialization/dataset identities differ')
    original = torch.load(common['path'], map_location='cpu', weights_only=False)
    equal_tree(saved['flow'], original['flow'])
    equal_tree(saved['critic'], refit['model'].state_dict())
    conditions = None
    if kind == 'final':
        # checked_refit/checked_plan have revalidated this original TRAIN QA.
        conditions = read(refit['plan']['inputs']['power_audit']['path'])['common_captured_conditions']
    utility = schema(kind, refit['spec'], final_conditions=conditions)
    return dict(payload=saved, schema=utility, refit=refit,
        binding=dict(version=VERSION, schema=utility, config=bind(experiment/'config.json'),
            checkpoint=checkpoint, completed=bind(folder/'completed.json'), training_binding=bind(folder/'binding.json'),
            common=common, critic_refit=refit['checkpoint'], source_sha256=sha(__file__)))


def equal_tree(first, second):
    """Exact tensor/state identity including optimizer and named RNG payloads."""
    if torch.is_tensor(first):
        if not torch.is_tensor(second) or first.dtype != second.dtype or not torch.equal(first.cpu(), second.cpu()):
            raise ValueError('Tensor state differs')
    elif isinstance(first, dict):
        if not isinstance(second, dict) or set(first) != set(second):
            raise ValueError('State dictionary keys differ')
        for key in first:
            equal_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        if type(first) != type(second) or len(first) != len(second):
            raise ValueError('State sequence differs')
        for a, b in zip(first, second):
            equal_tree(a, b)
    elif first != second:
        raise ValueError('State scalar differs')


def normalization(critic):
    return {name:t.detach().cpu().clone() for name,t in critic.named_buffers()}


def continuation_binding(source, replay, settings):
    """Freeze caller-declared update semantics; no defaults choose a new study."""
    required = {'total_updates', 'batch_size', 'alpha', 'q_weight', 'flow_steps',
                'update_critic', 'huber_delta', 'grad_clip'}
    if set(settings) != required or not isinstance(settings['update_critic'], bool):
        raise ValueError('Explicit complete scalar update settings required')
    if (not replay.get('ready') or replay['binding']['schema'] != source['schema'] or
            replay.get('utility_spec_sha256') != source['schema']['utility_spec_sha256']):
        raise ValueError('Replay is unresolved or uses another fixed TRAIN utility')
    if (replay['binding']['seed'] != source['payload']['binding']['seed'] or
            replay['binding']['checkpoint_sha256'] != source['binding']['checkpoint']['sha256'] or
            replay['binding']['source_round'] != 0):
        raise ValueError('Initial online replay was not collected by the completed offline parent')
    if (not isinstance(settings['total_updates'], int) or settings['total_updates'] < 1 or
            not isinstance(settings['batch_size'], int) or settings['batch_size'] < 2 or settings['batch_size'] % 2 or
            not isinstance(settings['flow_steps'], int) or settings['flow_steps'] < 1 or
            any(not math.isfinite(float(settings[k])) for k in ('alpha','q_weight','huber_delta','grad_clip')) or
            min(settings['alpha'],settings['q_weight']) < 0 or min(settings['huber_delta'],settings['grad_clip']) <= 0):
        raise ValueError('Invalid prospective update budget/objective settings')
    return dict(version=VERSION, parent=copy.deepcopy(source['binding']), replay=copy.deepcopy(replay['binding']),
                settings=copy.deepcopy(settings), source_sha256=sha(__file__),
                limitations='core only: collect exporter, final-route scheduling, and complete online chain are not implemented')


class Continuation:
    """One fixed-teacher update core, not an online experiment orchestrator.

    Parent and resume restore all model/optimizer/global/named RNG state without
    reseeding at a round boundary. This intentionally differs from the old
    online worker's per-round reseeding; future matched arms must share this
    new continuation convention. ``update_critic=False`` supports the future
    same-parent frozen-Q continuation counterfactual; no production arm is
    selected here. BC alone cannot identify the effect of online Q updates.
    """
    def __init__(self, models, optimizers, generators, payload, binding, *, frozen_reference, resume=False):
        slurm()
        if set(models) != set(MODEL_KEYS) or set(optimizers) != set(MODEL_KEYS) or set(generators) != set(GENERATOR_KEYS):
            raise ValueError('Full teacher/actor/critic optimizer and named RNG states required')
        if resume and (payload.get('binding') != binding or payload.get('version') != VERSION):
            raise ValueError('Scalar continuation resume binding changed')
        if not all(k in payload for k in (*MODEL_KEYS, 'optimizers', 'rng', 'cuda_rng', 'generators', 'step')):
            raise ValueError('Incomplete parent/resume state cannot preserve an exact prefix')
        # The caller retains the independently verified round0 source even when
        # resuming, rather than trusting a resumed file to define its own scale.
        if binding.get('parent') != frozen_reference['binding']:
            raise ValueError('Continuation does not bind its verified offline source')
        if not resume:
            equal_tree(payload, frozen_reference['payload'])
        equal_tree(payload['flow'], frozen_reference['payload']['flow'])
        normalizer_names = dict(models['critic'].named_buffers())
        equal_tree({k:payload['critic'][k] for k in normalizer_names},
                   {k:frozen_reference['payload']['critic'][k] for k in normalizer_names})
        self.models, self.opts, self.generators = models, optimizers, generators
        self.binding = copy.deepcopy(binding)
        for name in MODEL_KEYS:
            models[name].load_state_dict(payload[name], strict=True)
            # CPU load_state_dict can reuse optimizer tensors without copying.
            # Never let a tiny update mutate the caller's immutable parent.
            optimizers[name].load_state_dict(copy.deepcopy(payload['optimizers'][name]))
        torch.set_rng_state(payload['rng'].cpu())
        if payload['cuda_rng']:
            if not torch.cuda.is_available():
                raise ValueError('CUDA RNG prefix requires an allocated matching GPU')
            torch.cuda.set_rng_state_all([r.cpu() for r in payload['cuda_rng']])
        if set(payload['generators']) != set(GENERATOR_KEYS):
            raise ValueError('Parent named RNG stream inventory differs')
        for name, generator in generators.items():
            generator.set_state(payload['generators'][name].cpu())
        self.step = payload['step'] if resume else 0
        self.parent_steps = payload['parent_steps'] if resume else payload['step']
        self.teacher_state = copy.deepcopy(models['flow'].state_dict())
        self.fixed_buffers = normalization(models['critic'])
        models['flow'].eval()

    def update(self, offline, replay, *, batch_size, alpha, q_weight, flow_steps,
               update_critic=True, huber_delta=1., grad_clip=1.):
        """One Q-then-actor step, half offline/half own measured replay.

        Every valid B (including confirmed B4) goes through the same frozen
        standardized scalar Huber objective. No refitting of normalizers,
        teacher update, Bellman bootstrap, failure-MSE special branch or
        differentiation through legalizer/physical tools occurs.
        """
        if batch_size < 2 or batch_size % 2 or min(alpha, q_weight) < 0 or grad_clip <= 0 or flow_steps < 1:
            raise ValueError('Even positive paired batch and finite nonnegative objective weights required')
        if not all(math.isfinite(float(v)) for v in (alpha, q_weight, huber_delta, grad_clip)):
            raise ValueError('Nonfinite scalar update setting')
        actual = dict(batch_size=batch_size, alpha=alpha, q_weight=q_weight, flow_steps=flow_steps,
                      update_critic=update_critic, huber_delta=huber_delta, grad_clip=grad_clip)
        if (actual != {k:v for k,v in self.binding['settings'].items() if k != 'total_updates'} or
                self.step >= self.binding['settings']['total_updates'] or replay['binding'] != self.binding['replay']):
            raise ValueError('Update/replay changed its bound prospective semantics or exhausted budget')
        if not replay.get('ready'):
            raise ValueError('Unresolved replay population cannot train or be silently filtered')
        for data in (offline, replay):
            if (data['derived_B'].shape != (len(data['x']), 1) or data['mask'].shape != data['derived_B'].shape or
                not bool(data['mask'].all()) or not bool(torch.isfinite(data['derived_B']).all()) or
                bool((data['derived_B'] < 0).any()) or not len(data['x']) or
                data.get('utility_spec_sha256') != self.binding['parent']['schema']['utility_spec_sha256']):
                raise ValueError('Only fully admitted scalar B targets can enter Huber training')
        actor, flow, critic = (self.models[k] for k in ('actor', 'flow', 'critic'))
        device = next(actor.parameters()).device
        qloss = torch.zeros((), device=device)
        if update_critic and q_weight > 0:
            half = batch_size//2
            ix = [torch.randint(len(d['x']), (half,), device=device, generator=self.generators['q'])
                  for d in (offline, replay)]
            bx = torch.cat([d['x'].to(device)[i] for d,i in zip((offline,replay),ix)])
            by = torch.cat([d['derived_B'].to(device)[i] for d,i in zip((offline,replay),ix)])
            bm = torch.ones_like(by, dtype=torch.bool)
            qloss, _ = critic.loss(bx, by, bm, regression='huber', huber_delta=huber_delta)
            if not bool(torch.isfinite(qloss)):
                raise ValueError('Nonfinite scalar Huber loss')
            self.opts['critic'].zero_grad(set_to_none=True)
            qloss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip, error_if_nonfinite=True)
            self.opts['critic'].step()
        # Clear stale Q-update grads so actor-only differentiation is observable.
        self.opts['critic'].zero_grad(set_to_none=True)
        self.opts['flow'].zero_grad(set_to_none=True)
        noise = 2*torch.rand(batch_size, len(actor.fixed), 2, device=device,
                             generator=self.generators['actor'])-1
        with torch.no_grad():
            teacher = flow.sample_flow(noise, flow_steps)
        prediction = actor(noise, clip=False)
        distill = (prediction[:,actor.movable]-teacher[:,actor.movable]).square().mean()
        with frozen_critic(critic):
            q = scalar_q(critic, actor.restore(prediction, clip=True)).mean()
        loss = alpha*distill-q_weight*q
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite scalar actor loss (also checked for zero-Q controls)')
        self.opts['actor'].zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip, error_if_nonfinite=True)
        self.opts['actor'].step()
        equal_tree(self.teacher_state, flow.state_dict())
        equal_tree(self.fixed_buffers, normalization(critic))
        self.step += 1
        return dict(step=self.step, parent_steps=self.parent_steps, loss=float(loss.detach()),
            distill=float(distill.detach()), q=float(q.detach()), qloss=float(qloss.detach()),
            actor_grad=float(norm), teacher_updates=0, critic_updates=int(update_critic and q_weight>0),
            q_units=Q_UNITS, critic_loss='fixed-normalizer scalar Huber; success and B4 use one mask')

    def state(self):
        """Caller atomically saves this payload; no hidden restart reseeding."""
        equal_tree(self.teacher_state, self.models['flow'].state_dict())
        equal_tree(self.fixed_buffers, normalization(self.models['critic']))
        return copy.deepcopy(dict(version=VERSION, binding=self.binding, step=self.step,
            parent_steps=self.parent_steps, **{k:v.state_dict() for k,v in self.models.items()},
            optimizers={k:v.state_dict() for k,v in self.opts.items()}, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            generators={k:g.get_state() for k,g in self.generators.items()}))
