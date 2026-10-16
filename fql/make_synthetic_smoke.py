"""Generate a tiny mathematical fixture. NOT chip data or research evidence."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-output", help="Optional toy two-macro cache for analytic-control software checks")
    args = p.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1, 1, (96, 4)).astype(np.float32)
    teacher = (0.5 * noise + np.array([0.2, -0.2, 0.1, -0.1], dtype=np.float32)).astype(np.float32)
    # Toy quantization stands in for a non-differentiable external operation.
    action = (np.round(teacher * 10) / 10).astype(np.float32)
    reward = -np.square(action - 0.25).mean(axis=1)
    split = np.repeat([0, 1, 2], [64, 16, 16])
    np.savez(path, noise=noise, teacher_action=teacher, raw_action=teacher, action=action,
             reward=reward, split=split, critic_split=split, critic_parent_index=np.arange(96))
    path.with_suffix(".txt").write_text("SYNTHETIC SOFTWARE TEST ONLY. Not a circuit or placement result.\n")
    if args.cache_output:
        import torch
        cache = {"node_id_to_name": ["a", "b"], "ratio_x": 10/224, "ratio_y": 20/224,
                 "node_info": {n: {"x": 1, "y": 1} for n in "ab"},
                 "net_info": {"toy_net": {"nodes": {"a": {"x_offset": 0.2}, "b": {"y_offset": -0.3}}}}}
        torch.save(cache, args.cache_output)


if __name__ == "__main__":
    main()
