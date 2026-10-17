"""Measure lossless long-window recompression without changing an old result."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .graph_o2o_archive import atomic_json, sha, verify_archive


def run(args):
    outcome = json.loads(args.source_outcome.read_text())
    source = Path(outcome['physical_run'])
    record = json.loads((source/'grt_artifact_archive.json').read_text())
    archive = source/'grt_intermediates.tar.zst'
    inventory = source/'grt_intermediates_inventory.json'
    if sha(archive) != record['archive_sha256'] or sha(source/'manifest.json') != outcome['physical_manifest_sha256']:
        raise ValueError('Changed completed source measurement')
    expected = json.loads(inventory.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    scratch = Path('/scratch/bohanli/flowplace_graph/archive_compression_probe')/os.environ['SLURM_JOB_ID']
    scratch.mkdir(parents=True, exist_ok=False)
    staged = scratch/'long28.tar.zst'
    started = time.perf_counter()
    decoder = subprocess.Popen(['zstd','-dc','--long=28',str(archive)], stdout=subprocess.PIPE)
    try:
        encoder = subprocess.run(['zstd','-T2','-3','--long=28','-o',str(staged)],
                                 stdin=decoder.stdout, check=True)
        if decoder.wait(): raise ValueError('Original archive decoding failed')
    finally:
        if decoder.poll() is None: decoder.kill()
        decoder.wait()
    elapsed = time.perf_counter()-started
    verify_archive(staged, expected)
    reproduced = None
    if args.reproduce_original:
        restored = scratch/'original_archive_reproduced.tar.zst'
        decoder = subprocess.Popen(['zstd','-dc','--long=28',str(staged)], stdout=subprocess.PIPE)
        try:
            subprocess.run(['zstd','-T2','-3','-o',str(restored)], stdin=decoder.stdout, check=True)
            if decoder.wait(): raise ValueError('Compact archive decoding failed')
        finally:
            if decoder.poll() is None: decoder.kill()
            decoder.wait()
        reproduced = dict(original_bytes_reproduced=sha(restored)==sha(archive),
            regenerated_sha256=sha(restored), original_sha256=sha(archive),
            restored_archive=str(restored), encoder_command=['zstd','-T2','-3'],
            encoder_version=subprocess.check_output(['zstd','--version'],text=True).strip(),
            encoder_binary_sha256=sha(shutil.which('zstd')))
        if not reproduced['original_bytes_reproduced']:
            raise ValueError('Original archive cannot be regenerated bit-for-bit; leave it unchanged')
    atomic_json(output/'report.json', dict(source_outcome=str(args.source_outcome),
        source_outcome_sha256=sha(args.source_outcome), source_manifest_sha256=sha(source/'manifest.json'),
        source_archive_sha256=sha(archive), source_inventory_sha256=sha(inventory),
        original_archive_bytes=archive.stat().st_size, long28_archive_bytes=staged.stat().st_size,
        long28_archive_sha256=sha(staged), staged_archive=str(staged),
        compression_seconds=elapsed, all_decoded_file_hashes_match=True,
        original_archive_reproduction=reproduced,
        original_files_changed=False, slurm_job_id=os.environ['SLURM_JOB_ID'], source_sha256=sha(__file__)))
    shutil.copyfile(__file__, output/'source.py')
    print((output/'report.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-outcome',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reproduce-original',action='store_true')
    args=parser.parse_args()
    if 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('Use Slurm')
    run(args)
