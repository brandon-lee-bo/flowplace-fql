"""Generate one-step raw proposals from a trained checkpoint and arbitrary noise."""
import argparse
import numpy as np
import torch

try:
    from .core import OneStepActor
except ImportError:
    from core import OneStepActor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--noise", required=True, help="NPY array, or NPZ with noise key")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = OneStepActor(checkpoint["dim"], checkpoint["widths"], checkpoint["movable_mask"], checkpoint["fixed_action"])
    model.load_state_dict(checkpoint["actor_state"])
    model.to(args.device).eval()
    value = np.load(args.noise, allow_pickle=False)
    noise = value["noise"] if isinstance(value, np.lib.npyio.NpzFile) else value
    noise = np.asarray(noise, dtype=np.float32).reshape(len(noise), -1)
    if noise.shape[1] != checkpoint["dim"]:
        raise ValueError("Noise shape does not match checkpoint")
    with torch.no_grad():
        proposal = torch.cat([model(torch.tensor(batch, device=args.device), clip=True)
                              for batch in np.array_split(noise, max(1, (len(noise) + 255) // 256))])
    np.savez_compressed(args.output, noise=noise, raw_action=proposal.cpu().numpy(),
                        coordinate_shape=np.asarray(checkpoint["coordinate_shape"]))
    print("Saved raw proposals. Run the external legalizer and evaluate true metrics before comparing quality.")


if __name__ == "__main__":
    main()
