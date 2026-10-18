#!/usr/bin/env python3
"""Replay the retained Abacus boundary failure through the installed operator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from dreamplace.ops.abacus_legalize.abacus_legalize import AbacusLegalize
from dreamplace.ops.legality_check.legality_check import LegalityCheck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundary-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.snapshot / 'metadata.json').read_text())
    raw = np.load(args.snapshot / 'inputs.npz')
    names = ['node_size_x', 'node_size_y', 'node_weights', 'flat_region_boxes',
             'flat_region_boxes_start', 'node2fence_region_map']
    scalar_names = ['xl', 'yl', 'xh', 'yh', 'site_width', 'row_height',
                    'num_bins_x', 'num_bins_y', 'num_movable_nodes', 'num_terminal_NIs', 'num_filler_nodes']
    scalar = {name: metadata[name] for name in scalar_names}
    n = len(raw['pos']) // 2
    ids = np.array([31897, 167689])
    # The two actual touching rectangles, copied without changing precision,
    # location or dimensions. This isolates wrong row membership.
    boundary = {name: torch.from_numpy(raw[name][ids].copy()) for name in names[:3]}
    boundary.update(flat_region_boxes=torch.empty(0, dtype=torch.float32),
                    flat_region_boxes_start=torch.zeros(1, dtype=torch.int32),
                    node2fence_region_map=torch.full((2,), 2147483647, dtype=torch.int32))
    small_pos = torch.from_numpy(np.concatenate([raw['pos'][ids], raw['pos'][n + ids]]))
    small_scalar = dict(scalar, num_movable_nodes=2, num_terminal_NIs=0, num_filler_nodes=0)
    small_result = AbacusLegalize(**boundary, **small_scalar)(small_pos, small_pos)
    assert torch.equal(small_result, small_pos), 'Touching boundary fixture was moved.'
    report = {'boundary_fixture_passed': True, 'boundary_node_ids': ids.tolist(),
              'boundary_positions_unchanged': True, 'input_dtype': str(raw['pos'].dtype),
              'snapshot_sha256': hashlib.sha256((args.snapshot / 'inputs.npz').read_bytes()).hexdigest(),
              'snapshot_metadata_sha256': hashlib.sha256((args.snapshot / 'metadata.json').read_bytes()).hexdigest(),
              'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if not args.boundary_only:
        tensors = {name: torch.from_numpy(raw[name].copy()) for name in names}
        init_pos = torch.from_numpy(raw['init_pos'].copy())
        pos = torch.from_numpy(raw['pos'].copy())
        check_args = {name: tensors[name] for name in names if name != 'node_weights'}
        check_args.update({name: scalar[name] for name in ['xl', 'yl', 'xh', 'yh', 'site_width', 'row_height', 'num_movable_nodes']})
        # Nangate45 source LEF site width is exactly 380 DBU. DMP scales to 1.
        check_args.update(scale_factor=1.0 / 380, num_terminals=0)
        checker = LegalityCheck(**check_args)
        before_legal = bool(checker(pos))
        assert before_legal, 'Retained stock greedy output is not legal.'
        result = AbacusLegalize(**tensors, **scalar)(init_pos, pos)
        assert bool(torch.isfinite(result).all())
        after_legal = bool(checker(result))
        assert after_legal, 'Patched Abacus output is not legal.'
        macro_ids = np.flatnonzero(raw['node_size_y'][:scalar['num_movable_nodes']] > 5 * scalar['row_height'])
        values = result.numpy()
        maximum_macro_delta = float(max(np.max(np.abs(values[macro_ids] - raw['pos'][macro_ids])),
                                        np.max(np.abs(values[n + macro_ids] - raw['pos'][n + macro_ids]))))
        np.savez_compressed(args.output / 'output.npz', pos=values)
        report.update(snapshot_replay_passed=True, before_legal=before_legal, after_legal=after_legal,
                      macro_count=len(macro_ids), maximum_macro_coordinate_delta=maximum_macro_delta,
                      output_sha256=hashlib.sha256((args.output / 'output.npz').read_bytes()).hexdigest())
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
