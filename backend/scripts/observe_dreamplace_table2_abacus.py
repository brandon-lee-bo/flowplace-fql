#!/usr/bin/env python3
"""Observe stock Abacus inputs without changing the algorithm or return value."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-id', required=True)
    args, _ = parser.parse_known_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit the observation through Slurm.')
    work = Path('/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2') / args.run_id
    if work.exists():
        raise FileExistsError('Refusing to observe into an existing run.')
    import numpy as np
    from dreamplace.ops.abacus_legalize.abacus_legalize import AbacusLegalize
    import run_dreamplace_table2 as runner

    original = AbacusLegalize.__call__
    observations = []

    def observed(operator, init_pos, pos):
        # Copies and synchronization only; no mutation, random draw, new
        # objective, changed precision, or alternative operator is introduced.
        snapshot = work / ('abacus_input_' + str(len(observations)))
        snapshot.mkdir(exist_ok=False)
        arrays = {'init_pos': init_pos.detach().cpu().numpy().copy(),
                  'pos': pos.detach().cpu().numpy().copy()}
        for name in ['node_size_x', 'node_size_y', 'node_weights',
                     'flat_region_boxes', 'flat_region_boxes_start', 'node2fence_region_map']:
            arrays[name] = getattr(operator, name).detach().cpu().numpy().copy()
        np.savez_compressed(snapshot / 'inputs.npz', **arrays)
        metadata = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'run_id': args.run_id,
                    'observer_sha256': sha(__file__), 'runner_sha256': sha(runner.__file__),
                    'operator_python_sha256': sha(sys.modules[AbacusLegalize.__module__].__file__),
                    'input_snapshot_sha256': sha(snapshot / 'inputs.npz'),
                    'input_dtype': str(arrays['pos'].dtype), 'input_device': str(pos.device),
                    'scope': 'Original Abacus __call__ inputs after stock final macro and greedy legalization. Observer does not alter either input or original output.',
                    'started_at': time.time(), 'original_call_returned': False}
        for name in ['xl', 'yl', 'xh', 'yh', 'site_width', 'row_height',
                     'num_bins_x', 'num_bins_y', 'num_movable_nodes', 'num_terminal_NIs', 'num_filler_nodes']:
            value = getattr(operator, name)
            metadata[name] = value.item() if hasattr(value, 'item') else value
        runner.write_json(snapshot / 'metadata.json', metadata)
        shutil.copyfile(__file__, snapshot / Path(__file__).name)
        observations.append(snapshot)
        print('Saved unmodified stock Abacus inputs: ' + str(snapshot), flush=True)
        result = original(operator, init_pos, pos)
        # A C++ assertion may terminate before this point. A missing output is
        # retained as explicit evidence, never converted into a successful run.
        np.savez_compressed(snapshot / 'output.npz', pos=result.detach().cpu().numpy().copy())
        metadata.update(original_call_returned=True, finished_at=time.time(),
                        output_snapshot_sha256=sha(snapshot / 'output.npz'))
        runner.write_json(snapshot / 'metadata.json', metadata)
        return result

    AbacusLegalize.__call__ = observed
    return runner.main()


if __name__ == '__main__':
    sys.exit(main())
