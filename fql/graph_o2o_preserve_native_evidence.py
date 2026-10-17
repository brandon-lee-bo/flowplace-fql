"""Preserve exact native result evidence without moving or rewriting experiments.

This compact archive retains final DEFs, native logs/reports, imported actions
and their bindings. Large OpenDB/SPEF/intermediate archives stay in their
original locations; this is evidence preservation, not an EDA resume archive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from backend.scripts.result_storage import MIN_HEADROOM, storage_status
from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_checkpoint_archive import verify
from .graph_o2o_final_evidence import audit_final_outcome, LEGACY_REPEAT_PLAN

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = [ROOT, Path('/fact_data/bohanli'), Path('/scratch/bohanli')]


def preserve(plan_path):
    plan = json.loads(plan_path.read_text())
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Registered evidence-preservation source changed: ' + name)
    files, aliases, audits = {}, {}, []

    def add(path):
        original = Path(path).absolute()
        source = original.resolve(strict=True)
        if not source.is_file() or not any(source.is_relative_to(p) for p in ALLOWED):
            raise ValueError('Evidence source is not an allowed existing file: ' + str(original))
        member = 'files/' + str(source).lstrip('/')
        aliases[str(original)] = str(source)
        files[member] = source

    def physical_files(run, design, final=False):
        run = Path(run)
        for source in run.iterdir():
            if source.is_file() and source.suffix in {'.json', '.log'}:
                add(source)
        manifest = json.loads((run / 'manifest.json').read_text())
        add(manifest['def_path'])
        if sha(manifest['def_path']) != manifest['def_sha256']:
            raise ValueError('Original imported macro DEF changed')
        for name in ['actions.npz', 'sample.json', 'request.json', 'outcome.json']:
            candidate = Path(manifest['def_path']).parent / name
            if candidate.is_file():
                add(candidate)
        for kind in ['logs', 'reports']:
            folder = run / 'orfs_work' / kind / 'nangate45' / design / 'eval'
            for source in folder.iterdir():
                if source.is_file() and source.suffix in {'.json', '.log', '.rpt'}:
                    add(source)
        if final:
            add(run / 'orfs_work/results/nangate45' / design / 'eval/6_final.def')

    def inspect(outcome, design):
        checked = audit_final_outcome(outcome, design=design, route_cap=20,
                                      legacy_plan=LEGACY_REPEAT_PLAN)
        audits.append(dict(physical_run=outcome['physical_run'], audit=checked))
        physical_files(outcome['physical_run'], design, final=True)
        physical_files(checked['source_run'], design)
        legacy = checked['legacy_exact_repeat_evidence']
        if legacy is not None:
            add(legacy['plan'])
            add(legacy['repeat_outcome'])
            legacy_rows = json.loads(Path(legacy['plan']).read_text())['entries']
            entry = next(r for r in legacy_rows if
                         r['repeat_outcome'] == legacy['repeat_outcome'])
            add(entry['original_outcome'])
            inspect(json.loads(Path(legacy['repeat_outcome']).read_text()), design)

    requested = 0
    for source in plan['summaries']:
        path = Path(source['path'])
        if sha(path) != source['sha256']:
            raise ValueError('Completed final-route summary changed')
        report = json.loads(path.read_text())
        if (report['status'] != 'all_requested_outcomes_accounted' or
                report['requested'] != source['requested'] or
                len(report['records']) != source['requested']):
            raise ValueError('Incomplete native result denominator')
        add(path)
        experiment = path.parent.parent
        route_plan = experiment / 'full_route_plan.json'
        if sha(route_plan) != report['plan_sha256']:
            raise ValueError('Final route plan changed')
        add(route_plan)
        add(experiment / 'config.json')
        for row in report['records']:
            requested += 1
            if row['result']['status'] == 'completed':
                inspect(row['result'], source['design'])
            elif row['result']['status'] != 'upstream_physical_failure':
                raise ValueError('Unresolved native result cannot be archived as final')
    if requested != plan['requested_logical_results']:
        raise ValueError('Evidence preservation changed the requested scope')
    for item in plan.get('extra_files', []):
        if sha(item['path']) != item['sha256']:
            raise ValueError('Extra completed evidence changed')
        add(item['path'])
    for name in plan['source_sha256']:
        add(ROOT / name)
    add(plan_path)

    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    inventory = {member: dict(bytes=path.stat().st_size, sha256=sha(path))
                 for member, path in sorted(files.items())}
    scratch = Path(os.environ['TMPDIR'])
    scratch.mkdir(parents=True, exist_ok=True)
    tree = scratch / 'tree'
    tree.mkdir(exist_ok=True)
    for member, source in files.items():
        target = tree / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source)
        elif target.resolve() != source:
            raise ValueError('Interrupted staging link points to different evidence')
    names = scratch / 'files.null'
    names.write_bytes(b''.join(member.encode() + b'\0' for member in sorted(inventory)))
    staged = scratch / 'native_evidence.tar.zst'
    subprocess.run(['tar', '--dereference', '-I', 'zstd -T2 -3 --long=28',
                    '-cf', str(staged), '-C', str(tree), '--null', '-T', str(names)], check=True)
    verify(staged, inventory)
    for member, path in files.items():
        if sha(path) != inventory[member]['sha256']:
            raise ValueError('Source changed while evidence was archived')
    archive = output / 'native_evidence.tar.zst'
    storage = storage_status(Path('/fact_data/bohanli'), output)
    if not archive.exists():
        for key in ['filesystem_free_bytes', 'quota_remaining_bytes']:
            if storage.get(key) is not None and storage[key] - staged.stat().st_size < MIN_HEADROOM:
                raise RuntimeError('Evidence preservation would breach8GiB reserve')
        temporary = output / ('pending_job_' + os.environ['SLURM_JOB_ID'] + '_' +
                              os.environ.get('SLURM_RESTART_COUNT', '0') + '.tar.zst')
        with temporary.open('xb') as target, staged.open('rb') as stream:
            shutil.copyfileobj(stream, target, length=4 * 1024**2)
            target.flush()
            os.fsync(target.fileno())
        if sha(temporary) != sha(staged):
            raise ValueError('Durable evidence copy checksum differs')
        temporary.rename(archive)
    verify(archive, inventory)
    atomic_json(output / 'inventory.json', inventory)
    atomic_json(output / 'path_aliases.json', aliases)
    atomic_json(output / 'native_audits.json', audits)
    record = dict(status='native_evidence_durably_preserved_and_fully_decoded',
        plan_sha256=sha(plan_path), archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size, file_count=len(inventory),
        source_bytes=sum(row['bytes'] for row in inventory.values()),
        requested_logical_results=requested, native_audit_count=len(audits),
        unique_final_physical_runs=len({r['physical_run'] for r in audits}),
        inventory_sha256=sha(output / 'inventory.json'),
        native_audits_sha256=sha(output / 'native_audits.json'),
        aliases_sha256=sha(output / 'path_aliases.json'),
        original_files_unchanged=True, new_EDA_runs=0, EDA_resume_archive=False,
        limitations='OpenDB,SPEF and full intermediate archives remain at original locations; optimizer checkpoints are preserved separately.',
        restoration='Decode archive into an empty directory; files/ mirrors resolved absolute source paths. Verify inventory hashes before any deliberate restore; never overwrite current experiments.',
        storage_before=storage, slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output / 'manifest.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for native evidence auditing and compression')
    preserve(args.plan.resolve())
