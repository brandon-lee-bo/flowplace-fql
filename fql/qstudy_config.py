"""Create immutable, hash-bound actor configurations from completed Q screening."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .graph_o2o_archive import atomic_json, sha
from .flow_policy_config import RELEASED, LEGACY, STEPS, PUBLIC_SHA, validate_teacher_config

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')


def common_checkpoint(design, seed):
    if design == 'bp_fe':
        if seed == 0:
            return Path('/fact_data/bohanli/flowplace_fql/graph_o2o_v1/seed_0/round_0/common.pt')
        return Path(f'/scratch/bohanli/flowplace_graph/graph_o2o_replication_v1/seed_{seed}_experiment/seed_{seed}/round_0/common.pt')
    if design == 'bp_be' and seed == 0:
        return Path('/scratch/bohanli/flowplace_graph/graph_o2o_second_design_policy_v1/seed_0/round_0/common.pt')
    raise ValueError('No audited independent-seed common warmup; generate one before requesting this configuration')


def prepare(args):
    if args.flow_policy == RELEASED and args.online_flow_update:
        raise ValueError('The public FlowPlace teacher is frozen; remove --online-flow-update')
    dataset = BASE/'data_audit'/args.dataset/'dataset.npz'
    manifest = json.loads(dataset.with_name('manifest.json').read_text())
    design = manifest['design']
    experiment = args.output.resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    cfg = dict(design=design, dataset_path=str(dataset), dataset_sha256=sha(dataset),
        cache_sha256=sha(ROOT/'baseline/runtime/benchmark_cache'/(design+'.pt')),
        arm=args.arm, run_prefix=args.run_prefix, critic_variant=args.variant, seeds=args.seeds,
        weights=args.weights, alpha=args.alpha, q_weight=args.q_weight, uncertainty_beta=args.beta,
        online_flow_update=args.online_flow_update, offline_actor_steps=args.steps,
        online_updates=args.online_updates, online_samples=args.online_samples, online_rounds=2,
        evaluation_samples=args.evaluation_samples, evaluation_noise_seed=args.evaluation_noise_seed,
        evaluation_purpose=args.evaluation_purpose,
        collection_noise_seed=4500000, batch_size=16,
        flow_policy=args.flow_policy, flow_steps=STEPS if args.flow_policy == RELEASED else 10,
        actor_lr=1e-5, flow_lr=1e-5, critic_lr=3e-4,
        physical_failure_reward=-4, backend_threads=8, backend_storage='scratch',
        offline_critic_update=False, common_checkpoints={}, common_checkpoint_sha256={},
        critic_checkpoints={}, critic_checkpoint_sha256={}, terminal_mask=0,
        checkpoint_selection='fixed step budget; all predetermined evaluation noises',
        note='Offline Q fits independently before actor training; online Q uses measured own collect data.')
    if args.flow_policy == RELEASED:
        cfg.update(flow_checkpoint_sha256=PUBLIC_SHA, flow_warmup_steps=0,
            actor_initialization='Audited common student weights only; common flow weights are ignored')
    validate_teacher_config(cfg)
    for seed in args.seeds:
        common = common_checkpoint(design, seed)
        qfolder = args.screen/args.dataset/args.variant/f'seed_{seed}'/'refit'
        qpath = qfolder/'resume.pt'
        done = json.loads((qfolder/'completed.json').read_text())
        if done['checkpoint_sha256'] != sha(qpath) or done['steps'] != 2000:
            raise ValueError('Only completed fixed-budget screening checkpoints are admitted')
        cfg['common_checkpoints'][str(seed)] = str(common)
        cfg['common_checkpoint_sha256'][str(seed)] = sha(common)
        cfg['critic_checkpoints'][str(seed)] = str(qpath)
        cfg['critic_checkpoint_sha256'][str(seed)] = sha(qpath)
    path = experiment/'config.json'
    if path.exists():
        if json.loads(path.read_text()) != cfg:
            raise ValueError('Refusing to change an existing actor experiment')
    else:
        atomic_json(path, cfg)
    print(json.dumps(dict(experiment=str(experiment), config_sha256=sha(path), seeds=args.seeds)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', choices=['bp_fe64', 'bp_fe384', 'bp_be64'], default='bp_fe384')
    p.add_argument('--screen', type=Path, default=BASE/'screen_v1')
    p.add_argument('--variant', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--run-prefix', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--weights', nargs=5, type=float, default=[1, 0, 0, 0, 0])
    p.add_argument('--alpha', type=float, default=10.)
    p.add_argument('--q-weight', type=float, default=1.)
    p.add_argument('--beta', type=float, default=0.)
    p.add_argument('--online-flow-update', action='store_true')
    p.add_argument('--flow-policy', choices=[RELEASED, LEGACY], default=RELEASED,
                   help='Frozen released FlowPlace by default; legacy only reproduces old experiments')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--online-updates', type=int, default=600)
    p.add_argument('--online-samples', type=int, default=8)
    p.add_argument('--evaluation-samples', type=int, default=8)
    p.add_argument('--evaluation-noise-seed', type=int, default=4600000)
    p.add_argument('--evaluation-purpose', choices=['development_validation', 'locked_confirmation'], default='development_validation')
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
