#!/usr/bin/env python3
"""Package verified superblue18 backend inputs; never run or contact Innovus.

Run through prepare_remote_bundle.slurm. The archive contains flat filenames,
both unchanged DREAMPlace outputs, and a manifest of byte-level provenance.
The old-layout rc.spef is intentionally excluded.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('/fact_data/bohanli/flowplace_iccad')
DEFAULT_SCRATCH = Path('/scratch/bohanli/flowplace_iccad')
RUNS = (
    ('released', 'released_superblue18_cpu_job1787685', '1787685',
     'superblue18.released.gp.def',
     '24e755230ebd537ac065ce9cbd4fa16ce65a5d5e5333866c8d881af275b7b378'),
    ('generated_seed300', 'generated_superblue18_seed300_cpu_v1', '1787687',
     'superblue18.generated_seed300.gp.def',
     '1995c2ef093b26c481a17669325c8000452e38f6bf9d08bc1f4a75f8f23f18bb'),
)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load_evidence(path):
    raw = path.read_bytes()
    return json.loads(raw), {
        'source_path': str(path), 'source_resolved_path': str(path.resolve()),
        'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw),
    }


def copy_input(source, destination, role, source_jobs, expected=None):
    before = source.stat()
    result = hashlib.sha256()
    with source.open('rb') as reader, destination.open('xb') as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b''):
            writer.write(chunk)
            result.update(chunk)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source changed during copy: ' + str(source))
    sha256 = result.hexdigest()
    if expected is not None and sha256 != expected:
        raise RuntimeError('Source differs from verified bytes: ' + str(source))
    return {
        'archive_name': destination.name, 'role': role,
        'source_path': str(source), 'source_resolved_path': str(source.resolve()),
        'sha256': sha256, 'size_bytes': destination.stat().st_size,
        'source_jobs': source_jobs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-name', help='New output directory basename')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--scratch-root', type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        parser.error('Package through Slurm; SLURM_JOB_ID is required.')
    name = args.bundle_name or 'scut_bundle_' + job_id
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        parser.error('--bundle-name must be a simple directory basename.')
    for path, allowed in ((args.output_root, '/fact_data/bohanli/'),
                          (args.scratch_root, '/scratch/bohanli/')):
        if not str(path.resolve()).startswith(allowed):
            parser.error('Storage must remain under ' + allowed)
    output = args.output_root / name
    if output.exists():
        raise FileExistsError('Refusing to replace existing bundle: ' + str(output))
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    assets = ROOT / 'iccad/assets'
    benchmark = assets / 'benchmarks/iccad2015.ot/superblue18'
    inputs = [(assets / 'tech.lef', 'tech.lef', 'routing_technology_lef', [], None)]
    evidence = {}
    common_hashes = None
    for label, run_name, placement_job, archive_name, expected_def in RUNS:
        run = assets / 'runs' / run_name
        meta, meta_source = load_evidence(run / 'manifest.json')
        legality, legality_source = load_evidence(run / 'all_cell_legality_float64.json')
        if meta['returncode'] != 0 or str(meta['slurm_job']) != placement_job:
            raise RuntimeError('Placement did not complete successfully: ' + str(run))
        if legality.get('legal') is not True or legality.get('sha256') != expected_def:
            raise RuntimeError('Missing matching all-cell legality evidence: ' + str(run))
        output_def = run / 'placement/superblue18/superblue18.gp.def'
        if Path(legality['output_def']).resolve() != output_def.resolve():
            raise RuntimeError('Legality evidence refers to another DEF: ' + str(run))
        run_evidence = {'placement_manifest': dict(meta_source, contents=meta),
                        'all_cell_legality': dict(legality_source, contents=legality)}
        source_jobs = [placement_job, str(legality['slurm_job'])]
        if label == 'released':
            if meta.get('complete') is not True or meta.get('output_sha256') != expected_def:
                raise RuntimeError('Released DEF export was not verified.')
        else:
            verified, verification_source = load_evidence(run / 'output_verification.json')
            if verified.get('verified') is not True or verified.get('output_sha256') != expected_def:
                raise RuntimeError('Generated DEF export was not verified.')
            if verified['original_manifest_sha256'] != meta_source['sha256']:
                raise RuntimeError('Generated verification and manifest disagree.')
            run_evidence['export_verification'] = dict(verification_source, contents=verified)
            source_jobs.extend(['1787679', str(verified['slurm_job'])])
        evidence[label] = run_evidence
        inputs.append((output_def, archive_name, label + '_final_dreamplace_def', source_jobs, expected_def))
        shared = {key: value for key, value in meta['input_sha256'].items() if key != 'def_input'}
        if common_hashes is not None and shared != common_hashes:
            raise RuntimeError('The two DREAMPlace runs used different benchmark inputs.')
        common_hashes = shared
    for filename, key, role in (
        ('superblue18.lef', 'lef_input', 'cell_geometry_lef'),
        ('superblue18.v', 'verilog_input', 'gate_level_verilog'),
        ('superblue18.sdc', 'sdc_input', 'timing_constraints'),
        ('superblue18_Early.lib', 'early_lib_input', 'early_liberty'),
        ('superblue18_Late.lib', 'late_lib_input', 'late_liberty'),
    ):
        inputs.append((benchmark / filename, filename, role, ['1787656'], common_hashes[key]))
    tcl = ROOT / 'iccad/scripts/innovus_egr.tcl'
    if tcl.is_file():
        inputs.append((tcl, tcl.name, 'unexecuted_innovus_flow_snapshot', [], None))
    manifest = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'packaging_slurm_job': job_id, 'design': 'superblue18',
        'packaging_script': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__))},
        'purpose': 'DREAMPlace 4.1.0 outputs prepared for Cadence Innovus EarlyGlobalRoute evaluation',
        'not_a_result': 'Packaging verifies inputs only; no Innovus import, routing or timing is executed.',
        'excluded': {'rc.spef': 'Parasitics correspond to the old layout and must not be reused.'},
        'evidence': evidence, 'files': [],
    }
    with tempfile.TemporaryDirectory(prefix=name + '_', dir=str(args.scratch_root)) as temporary:
        staging = Path(temporary)
        payload = staging / 'payload'
        payload.mkdir()
        for source, archive_name, role, source_jobs, expected in inputs:
            entry = copy_input(source, payload / archive_name, role, source_jobs, expected)
            manifest['files'].append(entry)
            print('PACKAGED', archive_name, entry['size_bytes'], entry['sha256'], flush=True)
        (payload / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        checksums = ['{}  {}\n'.format(entry['sha256'], entry['archive_name']) for entry in manifest['files']]
        checksums.append('{}  manifest.json\n'.format(digest(payload / 'manifest.json')))
        (payload / 'SHA256SUMS').write_text(''.join(checksums))
        archive = staging / (name + '.tar.gz')
        with archive.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=3, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|') as tar:
                    for item in sorted(payload.iterdir()):
                        info = tar.gettarinfo(str(item), arcname=item.name)
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ''
                        info.mode = 0o644
                        with item.open('rb') as stream:
                            tar.addfile(info, stream)
        archive_sha256 = digest(archive)
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(archive, output / archive.name)
        if digest(output / archive.name) != archive_sha256:
            raise RuntimeError('Archive checksum changed during durable copy.')
        shutil.copyfile(payload / 'manifest.json', output / 'manifest.json')
        (output / (archive.name + '.sha256')).write_text(archive_sha256 + '  ' + archive.name + '\n')
        (output / 'complete.json').write_text(json.dumps({
            'complete': True, 'slurm_job': job_id,
            'archive': str(output / archive.name), 'archive_sha256': archive_sha256,
            'archive_size_bytes': archive.stat().st_size,
        }, indent=2) + '\n')
    print('BUNDLE_COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()
