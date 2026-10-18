"""Explicit old-BC actor prefix plus independently verified scalar critic.

This CPU-only source provider neither trains nor exports a policy. The original
BC checkpoint remains the real collection parent. Its legacy critic is not
relabeled as scalar-trained: a new dual-source binding describes the assembled
initial state, including the deliberately fresh, empty critic optimizer.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

# Resolve the project package before upstream FlowPlace adds its own analysis.py.
from analysis.placement_metrics import MacroHPWL as _MacroHPWL
from . import qstudy_scalar_online as core
from .qstudy_data import bind, load_dataset, sha
from .qstudy_bounded_screen import json_sha as behavior_json_sha

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qstudy_scalar_online_bc_source_v1'
SHARED = ('design', 'dataset_sha256', 'cache_sha256', 'seeds', 'alpha',
          'offline_actor_steps', 'batch_size', 'flow_steps', 'actor_lr',
          'flow_lr', 'critic_lr', 'online_flow_update', 'offline_critic_update',
          'terminal_mask', 'physical_failure_reward')
FIXED = dict(design='bp_fe', seeds=[0, 1, 2], alpha=10., q_weight=0.,
    offline_actor_steps=1000, batch_size=16, flow_steps=10, actor_lr=1e-5,
    flow_lr=1e-5, critic_lr=3e-4, online_flow_update=False,
    offline_critic_update=False, terminal_mask=0, physical_failure_reward=-4,
    critic_variant='legacy', weights=[1, 0, 0, 0, 0])
OLD_SOURCE_NAMES = ('fql/qstudy_actor.py', 'fql/qstudy_models.py',
    'fql/qstudy_screen.py', 'fql/graph_o2o_models.py', 'baseline/api.py')


def state_sha(value):
    """Deterministic exact tensor/state identity, not a model evaluation."""
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda k:(type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False,
                                     sort_keys=True).encode())
    visit(value)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _behavior_identity(data):
    actions, families = np.asarray(data['behavior_action']), np.asarray(data['behavior_family'])
    if actions.shape != (256, 22) or families.shape != (256,) or not np.isfinite(actions).all():
        raise ValueError('The original fixed256 behavior target identity differs')
    return dict(shape=[256, 22], targets=256,
                actions_sha256=behavior_json_sha(actions.tolist()),
                families_sha256=behavior_json_sha(families.tolist()))


def _bc_payload(experiment, seed, checkpoint_sha256, config_sha256, target, target_cfg):
    cfg_binding = bind(experiment/'config.json', config_sha256)
    cfg = _read(experiment/'config.json')
    if (cfg.get('arm') != 'bc' or cfg.get('run_prefix') != 'bcfr' or
            any(cfg.get(k) != v for k, v in FIXED.items()) or seed not in FIXED['seeds']):
        raise ValueError('Source is not the original fixed1000-step zero-Q BC arm')
    if any(cfg.get(k) != target_cfg.get(k) for k in SHARED):
        raise ValueError('BC and scalar source policy/data/teacher budgets differ')
    if target_cfg.get('q_weight') != 1.:
        raise ValueError('Scalar source must be the declared offline FQL source')
    dataset = bind(cfg['dataset_path'], cfg['dataset_sha256'])
    if dataset != bind(target_cfg['dataset_path'], target_cfg['dataset_sha256']):
        raise ValueError('BC and scalar source datasets differ')
    data, _ = load_dataset(Path(dataset['path']))
    behavior = _behavior_identity(data)
    prefix = 'rule' if target['schema']['kind'] == 'rule' else 'final_feedback'
    if behavior != target_cfg[prefix+'_behavior']:
        raise ValueError('Scalar source uses different256 behavior targets')
    common = bind(cfg['common_checkpoints'][str(seed)], cfg['common_checkpoint_sha256'][str(seed)])
    if common != target['binding']['common']:
        raise ValueError('BC and scalar source common checkpoint differ')
    folder = experiment/f'seed_{seed}'/'round_0'
    checkpoint = bind(folder/'checkpoint.pt', checkpoint_sha256)
    done, binding = _read(folder/'completed.json'), _read(folder/'binding.json')
    sources = {name:sha(ROOT/name) for name in OLD_SOURCE_NAMES}
    expected = dict(config_sha256=cfg_binding['sha256'], dataset_sha256=dataset['sha256'],
        common_sha256=common['sha256'],
        initial_critic_sha256=cfg['critic_checkpoint_sha256'][str(seed)],
        round=0, seed=seed, parent=None, replay=[], source_sha256=sources)
    if (binding != expected or done.get('binding') != expected or
            done.get('status') != 'completed' or done.get('steps') != 1000 or
            done.get('checkpoint_sha256') != checkpoint['sha256'] or
            done.get('offline_critic_frozen') is not True or done.get('online_flow_update') is not False):
        raise ValueError('Original BC completion/binding/source identity changed')
    payload = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
    if (payload.get('binding') != expected or payload.get('step') != 1000 or
            set(payload.get('optimizers', {})) != set(core.MODEL_KEYS) or
            set(payload.get('generators', {})) != set(core.GENERATOR_KEYS) or
            not all(k in payload for k in (*core.MODEL_KEYS, 'rng', 'cuda_rng'))):
        raise ValueError('BC is not a complete real optimizer/RNG prefix')
    original = torch.load(common['path'], map_location='cpu', weights_only=False)
    core.equal_tree(payload['flow'], original['flow'])
    core.equal_tree(payload['flow'], target['payload']['flow'])
    old_critic = bind(cfg['critic_checkpoints'][str(seed)], expected['initial_critic_sha256'])
    core.equal_tree(payload['critic'], torch.load(old_critic['path'], map_location='cpu',
                                                weights_only=False)['model'])
    if payload['optimizers']['flow']['state'] or payload['optimizers']['critic']['state']:
        raise ValueError('Original BC unexpectedly updated its teacher or critic')
    actor_opt = payload['optimizers']['actor']
    states = actor_opt.get('state', {})
    if (not states or any(float(v['step']) != 1000. for v in states.values()) or
            any(g['lr'] != cfg['actor_lr'] for g in actor_opt['param_groups'])):
        raise ValueError('BC actor Adam does not contain its real1000-step prefix')
    # Neither round0 arm consumes the q/flow streams; actor draws are paired.
    core.equal_tree(payload['generators'], target['payload']['generators'])
    return payload, dict(config=cfg_binding, checkpoint=checkpoint,
        completed=bind(folder/'completed.json'), training_binding=bind(folder/'binding.json'),
        common=common, old_legacy_critic=old_critic, behavior=behavior)


def assemble_source(bc_payload, bc_binding, scalar_source, *, critic_lr=3e-4):
    """Assemble audited inputs; deliberately do not alter the old BC binding."""
    core.slurm()
    core.validate_schema(scalar_source['schema'])
    if (bc_payload['binding']['seed'] != scalar_source['payload']['binding']['seed'] or
            bc_payload['binding']['round'] != 0 or bc_payload['step'] != 1000):
        raise ValueError('BC and scalar source seed/prefix differ')
    core.equal_tree(bc_payload['flow'], scalar_source['payload']['flow'])
    core.equal_tree(bc_payload['generators'], scalar_source['payload']['generators'])
    model = scalar_source['refit']['model']
    refit_state = copy.deepcopy(model.state_dict())
    core.equal_tree(refit_state, scalar_source['payload']['critic'])
    if critic_lr != 3e-4:
        raise ValueError('Fixed fresh critic optimizer learning rate changed')
    fresh = torch.optim.AdamW(model.parameters(), lr=critic_lr, weight_decay=1e-4).state_dict()
    if fresh['state']:
        raise ValueError('Fresh scalar critic optimizer unexpectedly has moments')
    core.equal_tree(fresh, scalar_source['payload']['optimizers']['critic'])
    payload = copy.deepcopy(bc_payload)
    payload['critic'] = refit_state
    payload['optimizers']['critic'] = copy.deepcopy(fresh)
    preserved = {k:state_sha(bc_payload[k]) for k in ('flow', 'actor', 'rng', 'cuda_rng', 'generators')}
    preserved['actor_optimizer'] = state_sha(bc_payload['optimizers']['actor'])
    preserved['flow_optimizer'] = state_sha(bc_payload['optimizers']['flow'])
    binding = dict(version=VERSION, schema=copy.deepcopy(scalar_source['schema']),
        config=copy.deepcopy(bc_binding['config']), checkpoint=copy.deepcopy(bc_binding['checkpoint']),
        completed=copy.deepcopy(bc_binding['completed']),
        training_binding=copy.deepcopy(bc_binding['training_binding']),
        common=copy.deepcopy(bc_binding['common']), actor_prefix=copy.deepcopy(bc_binding),
        target_scalar_source=copy.deepcopy(scalar_source['binding']),
        critic_refit=copy.deepcopy(scalar_source['refit']['checkpoint']),
        critic_optimizer=dict(type='AdamW', lr=critic_lr, weight_decay=1e-4,
            moments='fresh_empty', state_sha256=state_sha(fresh),
            not_old_legacy_optimizer=True, not_screen_refit_optimizer=True),
        preserved_state_sha256=preserved, assembled_payload_sha256=state_sha(payload),
        source_sha256=sha(__file__),
        semantics='Real BC actor/Adam/RNG continuation plus separately initialized scalar critic; not an original scalar-trained BC checkpoint',
        limitations='CPU source provider only; no physical collect/export/controller or new zero-Q effective-update certificate')
    return dict(payload=payload, schema=copy.deepcopy(scalar_source['schema']),
                refit=scalar_source['refit'], binding=binding)


def load_bc_source(kind, bc_experiment, scalar_experiment, seed, *,
                   expected_bc_config_sha256, expected_bc_checkpoint_sha256,
                   expected_scalar_config_sha256, expected_scalar_checkpoint_sha256):
    """Verify both external identities before loading any large checkpoint."""
    core.slurm()
    bc_experiment, scalar_experiment = Path(bc_experiment).resolve(), Path(scalar_experiment).resolve()
    if bc_experiment == scalar_experiment:
        raise ValueError('BC prefix and scalar source must remain separate experiments')
    bind(bc_experiment/'config.json', expected_bc_config_sha256)
    bind(scalar_experiment/'config.json', expected_scalar_config_sha256)
    target = core.load_offline_source(kind, scalar_experiment, seed, expected_scalar_checkpoint_sha256)
    target_cfg = _read(scalar_experiment/'config.json')
    if target['binding']['config'] != bind(scalar_experiment/'config.json', expected_scalar_config_sha256):
        raise ValueError('Scalar source configuration differs from its external identity')
    payload, binding = _bc_payload(bc_experiment, seed, expected_bc_checkpoint_sha256,
                                  expected_bc_config_sha256, target, target_cfg)
    result = assemble_source(payload, binding, target, critic_lr=target_cfg['critic_lr'])
    # Revalidate all small root artifacts after assembly; originals stay untouched.
    for artifact in (binding['config'], binding['checkpoint'], binding['completed'],
                     binding['training_binding'], binding['common'], binding['old_legacy_critic'],
                     target['binding']['config'], target['binding']['checkpoint']):
        bind(artifact['path'], artifact['sha256'])
    return result
