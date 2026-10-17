"""Runtime shape/gradient and data-integrity checks, never experimental labels."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_backend import ROOT
from .graph_o2o_design_labels import inputs
from .graph_o2o_design_policy_data import validate_arrays


def semantic_checks(values):
    _, plan, design, _, source, selected = values
    indices = np.asarray([item[2] for item in selected])
    split = np.asarray([item[3] for item in selected])
    groups = np.asarray([item[0] for item in selected])
    # Explicitly synthetic code-test scalars. Never written as real PPA/replay.
    rwl = 4_000_000. + np.arange(80) * 1000
    success = np.ones(80, dtype=np.bool_)
    success[0] = False
    rwl[0] = np.nan
    scale = rwl[(split == 0) & success].mean()
    reward = np.where(success, -rwl / scale, -4.)
    arrays = dict(raw_action=source['raw_action'][indices].copy(), action=source['action'][indices].copy(),
        reward=reward.astype(np.float32), backend_rwl_um=rwl, backend_success=success,
        source_row_index=indices, critic_split=split, critic_group_id=groups,
        critic_parent_index=groups.copy(), legal=source['legal'][indices].copy())
    labels = []
    for i, item in enumerate(selected):
        labels.append(dict(terminal_mask=0, critic_split=int(split[i]), source_family=int(groups[i]),
            source_row_index=int(indices[i]), status='success' if success[i] else 'physical_failure',
            raw_action=arrays['raw_action'][i].reshape(-1, 2).tolist(),
            executed_action=arrays['action'][i].reshape(-1, 2).tolist(), reward=float(reward[i]),
            metrics=dict(rWL_um=float(rwl[i])) if success[i] else None))
    manifest = dict(reward_stage='global_route_only', requested=80, TRAIN_requested=64,
        validation_requested=16, all_requested_accounted=True, test_labels_collected=False,
        proxy_rewards_used=False, outcomes=[dict(fixture=True) for _ in range(80)],
        reward_scale_train_only=float(scale), physical_failure_reward=-4, successful=79)
    loaded = validate_arrays(arrays, manifest, labels)
    if len(loaded['train']['reward']) != 64 or loaded['train']['backend_success'][0]:
        raise AssertionError('Known physical failure was dropped')
    tests = []

    def negative(name, mutate):
        a, m, l = copy.deepcopy((arrays, manifest, labels))
        mutate(a, m, l)
        try:
            validate_arrays(a, m, l)
        except (ValueError, AssertionError):
            tests.append(dict(name=name, rejected=True))
        else:
            raise AssertionError('Negative fixture admitted: ' + name)

    negative('fake_numeric_PPA_for_failure', lambda a, m, l: a['backend_rwl_um'].__setitem__(0, 1.))
    negative('validation_family_overlaps_TRAIN', lambda a, m, l: a['critic_group_id'].__setitem__(64, 0))
    negative('changed_frozen_scale', lambda a, m, l: m.__setitem__('reward_scale_train_only', scale + 1000))
    negative('missing_requested_outcome', lambda a, m, l: m['outcomes'].pop())
    negative('changed_raw_action', lambda a, m, l: a['raw_action'].__setitem__((1, 0), a['raw_action'][1, 0] + .1))
    negative('nonterminal_transition', lambda a, m, l: l[1].__setitem__('terminal_mask', 1))
    negative('test_labels_consumed', lambda a, m, l: m.__setitem__('test_labels_collected', True))
    negative('altered_measured_reward', lambda a, m, l: l[1].__setitem__('reward', -.1))
    negative('unknown_failure_as_training_label', lambda a, m, l: l[0].__setitem__('status', 'infrastructure_failure'))
    negative('repeated_source_row', lambda a, m, l: a['source_row_index'].__setitem__(1, a['source_row_index'][0]))
    return dict(positive_fixture_retains_all_80_including_one_failure=True, negative_checks=tests,
                fixture_rewards_are_synthetic_unit_tests_only=True)


def method_checks():
    original_path = ROOT / 'fql/graph_o2o_train.py'
    new_path = ROOT / 'fql/graph_o2o_design_train.py'
    old = ast.parse(original_path.read_text())
    new = ast.parse(new_path.read_text())
    functions = ['construct', 'state_of', 'update', 'critic_stats', 'parameter_changes', 'save_point']
    matches = []
    for name in functions:
        a = next(item for item in old.body if isinstance(item, ast.FunctionDef) and item.name == name)
        b = next(item for item in new.body if isinstance(item, ast.FunctionDef) and item.name == name)
        if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
            raise AssertionError('Method helper changed: ' + name)
        matches.append(name)

    def loss_nodes(tree):
        train = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'train')
        return [ast.dump(item, include_attributes=False) for item in ast.walk(train)
            if (isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and
                target.id in ['aloss', 'floss', 'qloss'] for target in item.targets)) or
            (isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and
                item.func.id in ['actor_objective', 'update'])]
    if loss_nodes(old) != loss_nodes(new):
        raise AssertionError('Actor/flow/critic losses or optimizer application changed')
    return dict(original_train_sha256=sha(original_path), design_train_sha256=sha(new_path),
                identical_method_helpers=matches, identical_core_losses_and_update_calls=True)


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261018)
    values = inputs(args.label_plan)
    plan_path, plan, design, _, source, selected = values
    semantic = semantic_checks(values)
    method = method_checks()
    from analysis.placement_metrics import MacroHPWL, legality
    from baseline.api import RUNTIME, load_problem, legalize, save_sample
    from backend.scripts.export_macro_def import convert
    from .graph_o2o_models import GraphPolicy, FeatureCritics, actor_objective

    problem = load_problem(design['design'], 'cuda')
    cache = torch.load(RUNTIME / 'benchmark_cache' / (design['design'] + '.pt'),
                       map_location='cpu', weights_only=False)
    train_rows = [item[2] for item in selected if item[3] == 0]
    x = torch.as_tensor(source['raw_action'][train_rows], device='cuda').reshape(64, len(problem.x), 2)
    # Dummy targets test dimensions/gradients only; no optimizer/checkpoint/replay.
    y = torch.linspace(-1.1, -.9, 64, device='cuda')
    flow = GraphPolicy(problem)
    actor = copy.deepcopy(flow)
    critic = FeatureCritics(cache, x, y, actor.movable).cuda()
    with torch.no_grad():
        for member in critic.members:
            member[-1].weight.normal_(std=.01)
    z = 2 * torch.rand(2, len(problem.x), 2, device='cuda') - 1
    raw = actor(z)
    if raw.shape != (2, len(problem.x), 2) or critic(raw).shape != (2, 2):
        raise AssertionError('Second-design actor/critic shape mismatch')
    derivative = torch.autograd.grad(critic(raw).mean(), raw, retain_graph=True)[0]
    if not torch.isfinite(derivative).all() or float(derivative.norm()) <= 0:
        raise AssertionError('Q has no finite derivative with respect to raw action')
    loss, distill, q = actor_objective(actor, flow, critic, z, flow_steps=10)
    loss.backward()
    actor_grad = sum(float(p.grad.square().sum()) for p in actor.parameters() if p.grad is not None) ** .5
    if not np.isfinite(actor_grad) or actor_grad <= 0:
        raise AssertionError('No finite actor training gradient')
    if any(p.grad is not None for p in flow.parameters()) or any(p.grad is not None for p in critic.parameters()):
        raise AssertionError('Actor loss updated detached flow teacher or frozen critic parameters')
    flow_loss = flow.flow_loss(x[:2], z, torch.tensor([.2, .8], device='cuda'))
    flow_loss.backward()
    flow_grad = sum(float(p.grad.square().sum()) for p in flow.parameters() if p.grad is not None) ** .5
    if not np.isfinite(flow_grad) or flow_grad <= 0:
        raise AssertionError('No finite flow training gradient')
    with torch.no_grad():
        executed = legalize(raw[0], problem)
    metric = MacroHPWL(problem.benchmark_info)
    geometry = legality(executed.cpu().numpy(), metric.sizes)
    if not geometry['legal']:
        raise AssertionError('Second-design legalizer failed on diagnostic actor output')
    sample = output / 'diagnostic_sample_not_training_data'
    sample.mkdir()
    save_sample(sample, executed, z[0], problem, dict(test_only=True, no_backend_measurement=True))
    export_audit = convert(design['design'], sample / 'macro_physical.json', sample / 'diagnostic.def')
    if not export_audit['legal_no_halo']:
        raise AssertionError('Rounded DEF diagnostic is illegal')
    result = dict(status='runtime_shape_gradient_and_semantic_checks_passed', design=design['design'],
        label_plan_sha256=sha(plan_path), slurm_job_id=os.environ['SLURM_JOB_ID'],
        GPU=torch.cuda.get_device_name(), macro_count=len(problem.x), edge_count=int(problem.cond.edge_index.shape[-1]),
        actor_output_shape=list(raw.shape), critic_input_feature_count=len(critic.mean),
        critic_output_shape=list(critic(raw).shape), raw_action_q_gradient_norm=float(derivative.norm()),
        actor_gradient_norm=actor_grad, flow_gradient_norm=flow_grad,
        flow_teacher_and_critic_detached_during_actor_backward=True,
        normalized_macro_legal=geometry['legal'], rounded_DEF_legal=export_audit['legal_no_halo'],
        method_parity=method, data_semantics=semantic,
        no_experimental_policy_checkpoint_written=True, no_optimizer_steps=True,
        physical_labels_generated=0, no_claim_of_policy_quality=True,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / 'fql/graph_o2o_design_policy_data.py', ROOT / 'fql/graph_o2o_models.py']},
        finished_unix=time.time())
    atomic_json(output / 'evidence.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Run checks in a Slurm GPU allocation')
    run(args)
