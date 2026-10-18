"""Resumable, fixed-budget TRAIN-grouped critic screening; no EDA or actor fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_models import FeatureCritics
from .qstudy_data import METRIC_NAMES, load_dataset
from .qstudy_models import MultiMetricCritic, measured_costs

ROOT = Path(__file__).resolve().parents[1]
METRICS = tuple(METRIC_NAMES[:5])
OBJECTIVES = dict(wl=[1, 0, 0, 0, 0], timing=[1, .15, .35, 0, 0],
                  timing_congestion=[1, .15, .35, .10, .10])
VARIANTS = {
    "legacy": dict(legacy=True),
    "multi_mse": dict(regression="mse"),
    "multi_huber": dict(regression="huber"),
    "huber_rank": dict(regression="huber", ranking_weight=.1),
    "huber_local": dict(regression="huber", difference_weight=.1),
    "huber_residual": dict(regression="huber", hpwl_residual=True),
    "huber_large": dict(regression="huber", width=256, depth=3),
}
STOP = False


def _stop(*_):
    global STOP
    STOP = True


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def correlations(prediction, truth):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    p, y = prediction[valid], truth[valid]
    result = dict(n=len(y), mae=None, spearman=None, kendall=None, top_regret=None)
    if not len(y):
        return result
    result["mae"] = float(np.abs(p-y).mean())
    result["top_regret"] = float(y[np.argmin(p)]-y.min())
    if len(y) > 2 and np.ptp(p) > 1e-10 and np.ptp(y) > 1e-10:
        result.update(spearman=float(spearmanr(p, y).statistic), kendall=float(kendalltau(p, y).statistic))
    return result


def diagnostics(prediction, truth, families):
    result = correlations(prediction, truth)
    i, j = np.triu_indices(len(truth), 1)
    local = ((families[i] == families[j]) & np.isfinite(truth[i]) & np.isfinite(truth[j]) &
             np.isfinite(prediction[i]) & np.isfinite(prediction[j]) & (abs(truth[i]-truth[j]) > 1e-9))
    result["family_pairs"] = int(local.sum())
    result["family_pair_accuracy"] = (float(np.mean(
        (prediction[i[local]]-prediction[j[local]])*(truth[i[local]]-truth[j[local]]) > 0))
        if local.any() else None)
    return result


def make_model(cache, x, costs, mask, movable, variant, families, seed):
    cfg = VARIANTS[variant]
    if cfg.get("legacy"):
        reward = -costs[:, 0]/costs[:, 0].mean()
        return FeatureCritics(cache, x, reward, movable), None
    model = MultiMetricCritic(cache, x, costs, movable, METRICS, train_mask=mask,
        train_ids=[f"family{int(f)}_row{i}" for i, f in enumerate(families)], train_splits=["train"]*len(x),
        width=cfg.get("width", 128), depth=cfg.get("depth", 2),
        hpwl_residual=cfg.get("hpwl_residual", False), seed=seed)
    return model, model.normalization_metadata()


def predictions(model, x, train_costs, legacy):
    with torch.no_grad():
        if legacy:
            return (-model(x).mean(1)*train_costs[:, 0].mean())[:, None].cpu().numpy()
        return model(x).mean(1).cpu().numpy()


def fit(args, context, name, train_idx, target_idx, target_data):
    folder = args.output/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"completed.json").exists():
        done = json.loads((folder/"completed.json").read_text())
        if done["binding"] != context["binding"]:
            raise ValueError("Completed run binding changed")
        return done
    x, costs, mask, families = [context[k][train_idx] for k in ["x", "costs", "mask", "families"]]
    tx, tc, tm, tf = [target_data[k][target_idx] for k in ["x", "costs", "mask", "families"]]
    torch.manual_seed(args.seed)
    model, normalization = make_model(context["cache"], x, costs, mask, context["movable"],
                                     args.variant, families, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Architecture initialization must not change the paired minibatch stream.
    batch_generator = torch.Generator(device="cpu").manual_seed(810000+args.seed)
    checkpoint = folder/"resume.pt"
    begin = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["binding"] != context["binding"] or saved["train_indices"] != train_idx.tolist():
            raise ValueError("Resume identity or TRAIN split changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        batch_generator.set_state(saved["batch_rng"])
        begin = saved["step"]
    cfg = VARIANTS[args.variant]
    legacy = cfg.get("legacy", False)
    start = time.monotonic()
    for step in range(begin, args.steps):
        batch = torch.randint(len(x), (min(args.batch_size, len(x)),), generator=batch_generator)
        if legacy:
            pred = model(x[batch])
            truth = -costs[batch, 0]/costs[:, 0].mean()
            loss = (pred-truth[:, None]).square().mean()
            parts = dict(regression=loss)
        else:
            pairs = None
            if cfg.get("difference_weight"):
                i, j = torch.triu_indices(len(batch), len(batch), offset=1)
                keep = (families[batch[i]] == families[batch[j]]) & (batch[i] != batch[j])
                pairs = torch.stack((i[keep], j[keep]), 1)
            loss, parts = model.loss(x[batch], costs[batch], mask[batch],
                regression=cfg.get("regression", "mse"), ranking_weight=cfg.get("ranking_weight", 0.),
                difference_weight=cfg.get("difference_weight", 0.), pairs=pairs)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite critic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 100 == 0 or STOP or step+1 == args.steps:
            atomic_torch(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(), batch_rng=batch_generator.get_state(), step=step+1,
                train_indices=train_idx.tolist(), binding=context["binding"],
                normalization=normalization, model_config=cfg))
            row = dict(step=step+1, loss=float(loss), grad=float(grad),
                       parts={k: float(v) for k, v in parts.items()}, job=os.environ["SLURM_JOB_ID"])
            with (folder/"learning_curve.jsonl").open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print(json.dumps(dict(fit=name, **row)), flush=True)
        if STOP:
            raise SystemExit(85)
    model.eval()
    pred = predictions(model, tx, costs, legacy)
    truth = tc.cpu().numpy()
    pred_train = predictions(model, x, costs, legacy)
    metrics = {}
    for k, metric in enumerate(METRICS[:pred.shape[1]]):
        metrics[metric] = diagnostics(pred[:, k], truth[:, k], tf.cpu().numpy())
        metrics[metric]["train_mae"] = float(np.nanmean(abs(pred_train[:, k]-costs.cpu().numpy()[:, k])))
        scale = float(np.nanmean(costs.cpu().numpy()[:, k])) if legacy else float(model.objective_scale[k])
        metrics[metric]["normalized_mae"] = metrics[metric]["mae"]/max(scale, 1.) if scale == 0 else metrics[metric]["mae"]/scale
    objectives = {}
    if not legacy:
        for objective, weights in OBJECTIVES.items():
            with torch.no_grad():
                observed = -model.observed_reward(tc, weights, tm).cpu().numpy()
                for beta in [0., .5, 1.]:
                    predicted = -model.reward(tx, weights, uncertainty_beta=beta).cpu().numpy()
                    objectives[objective+"_beta"+str(beta)] = diagnostics(predicted, observed, tf.cpu().numpy())
    np.savez_compressed(folder/"predictions.npz", prediction=pred, truth=truth,
        family=tf.cpu().numpy(), target_indices=target_idx, train_indices=train_idx,
        prediction_train=pred_train)
    done = dict(status="completed", binding=context["binding"], variant=args.variant, seed=args.seed,
        steps=args.steps, train_rows=len(x), target_rows=len(tx), metrics=metrics, objectives=objectives,
        normalization=normalization, checkpoint_sha256=sha(checkpoint),
        elapsed_this_attempt_s=time.monotonic()-start, slurm_job_id=os.environ["SLURM_JOB_ID"])
    atomic_json(folder/"completed.json", done)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run model fitting under Slurm")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    torch.use_deterministic_algorithms(True)
    for sig in [signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(sig, _stop)
    args.output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_dataset(args.dataset)
    design = manifest["design"]
    cache_path = ROOT/"baseline/runtime/benchmark_cache"/(design+".pt")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    arrays = {}
    for split in ["train", "validation"]:
        source = data[split]
        action = torch.tensor(source["raw_action"], dtype=torch.float32).reshape(len(source["raw_action"]), -1, 2)
        costs, mask = measured_costs(torch.tensor(source["metrics"][:, :5], dtype=torch.float32),
                                    METRICS, torch.tensor(source["metric_mask"][:, :5]))
        # Physical failures stay accounted in audit; no fabricated numerical PPA enters regression.
        successful = mask[:, 0] & (costs[:, 0] > 0)
        arrays[split] = dict(x=action[successful], costs=costs[successful], mask=mask[successful],
            families=torch.tensor(source["family"], dtype=torch.long)[successful])
    # Cache order is the public model order. Ports are restored and never learned as coordinates.
    from baseline.api import load_problem
    movable = ~load_problem(design, "cpu").cond.is_ports.bool()
    source_files = [Path(__file__), ROOT/"fql/qstudy_models.py", ROOT/"fql/qstudy_data.py",
                    ROOT/"fql/graph_o2o_models.py", ROOT/"fql/physical_ridge.py", ROOT/"fql/analytic.py"]
    binding = dict(dataset_sha256=sha(args.dataset), manifest_sha256=sha(args.dataset.with_name("manifest.json")),
        cache_sha256=sha(cache_path), config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files},
        heldout_used=False, backend_calls=0, fold_seed=41307)
    old_path = args.output/"binding.json"
    if old_path.exists() and json.loads(old_path.read_text()) != binding:
        raise ValueError("Refusing to mix changed data/config/source in existing run")
    atomic_json(old_path, binding)
    context = {**arrays["train"], "cache": cache, "movable": movable, "binding": binding}
    groups = np.unique(context["families"].numpy())
    np.random.default_rng(41307).shuffle(groups)
    groups_by_fold = np.array_split(groups, args.folds)
    outcomes = []
    for fold, held in enumerate(groups_by_fold):
        target_idx = np.flatnonzero(np.isin(context["families"], held))
        train_idx = np.flatnonzero(~np.isin(context["families"], held))
        outcomes.append(fit(args, context, "cv_"+str(fold), train_idx, target_idx, arrays["train"]))
    outcomes.append(fit(args, context, "refit", np.arange(len(context["x"])),
                        np.arange(len(arrays["validation"]["x"])), arrays["validation"]))
    atomic_json(args.output/"completed.json", dict(status="completed", binding=binding, outcomes=outcomes))


if __name__ == "__main__":
    main()
