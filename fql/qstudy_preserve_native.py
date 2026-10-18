"""Stage compact Q-study proofs, then separately admit an exact durable copy.

Only SHA-bound explicit report/inventory entries are traversed. No EDA,
model loading, source rewriting, deletion, or source/cache linking occurs.
The compact archive is not an OpenDB/SPEF/intermediate resume backup.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

import numpy as np

from backend.scripts.result_storage import storage_status
from .graph_o2o_archive import sha
from .graph_o2o_final_evidence import audit_final_outcome
from .graph_o2o_backend import parse_grt_metrics

ROOT = Path(__file__).resolve().parents[1]
QROOT = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')
SCRATCH = Path('/scratch/bohanli/flowplace_graph')
BULK = Path('/fact_data/bohanli/flowplace_fql')
NATIVE = Path('/fact_data/bohanli/flowplace_fql_backend')
QUOTA = Path('/fact_data/bohanli')
VERSION = 'qstudy_compact_native_preservation_v1'
METRICS = ('rWL_um', 'WNS_ns', 'TNS_ns', 'horizontal_overflow_sum',
           'vertical_overflow_sum', 'Power_W', 'StdCellArea_um2')
SOURCE_NAMES = ('fql/qstudy_preserve_native.py', 'fql/qstudy_preserve_native.slurm',
    'fql/graph_o2o_archive.py', 'fql/graph_o2o_checkpoint_archive.py',
    'fql/graph_o2o_final_evidence.py', 'fql/graph_o2o_backend.py',
    'backend/scripts/grt_metrics.py', 'backend/scripts/grt_provenance.py', 'backend/scripts/result_storage.py',
    'analysis/audit_table2_macro_preservation.py')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=sha(path))


def within(path, roots):
    return any(path == root.resolve() or path.is_relative_to(root.resolve()) for root in roots)


def allowed_file(path):
    original = Path(path).absolute()
    source = original.resolve(strict=True)
    if not source.is_file() or not within(source, (ROOT, BULK, NATIVE, SCRATCH)):
        raise ValueError('Outside this project regular-file scope: ' + str(original))
    return source


def checked(item):
    if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
        raise ValueError('Expected an exact path/SHA256 binding')
    source = allowed_file(item['path'])
    if sha(source) != item['sha256']:
        raise ValueError('Changed bound source: ' + item['path'])
    return source


def write_once(path, value):
    """Never overwrite committed manifests or an interrupted copy."""
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def contains_binding(value, item):
    if isinstance(value, dict):
        # Fresh export inventories use the exact absolute-path -> SHA mapping.
        if value.get(item['path']) == item['sha256']:
            return True
        if value.get('path') == item['path'] and value.get('sha256') == item['sha256']:
            return True
        # Actual actor/fresh report rows store a sample path plus its outcome SHA.
        if (value.get('sample_dir') and value.get('outcome_sha256') == item['sha256'] and
                str(Path(value['sample_dir']) / 'outcome.json') == item['path']):
            return True
        return any(contains_binding(v, item) for v in value.values())
    return isinstance(value, list) and any(contains_binding(v, item) for v in value)


def safe_member(name):
    path = PurePosixPath(name)
    if (not name.startswith('files/') or path.is_absolute() or '..' in path.parts or
            str(path) != name or '\x00' in name or len(path.parts) < 3):
        raise ValueError('Unsafe archive member: ' + name)


def verify_archive(path, inventory):
    """Adapt the proven streaming verifier, also closing its subprocess pipe."""
    for name in inventory:
        safe_member(name)
    process = subprocess.Popen(['zstd', '-dc', '--long=28', str(path)], stdout=subprocess.PIPE)
    seen = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
            for item in stream:
                if (not item.isfile() or item.name not in inventory or item.name in seen or
                        item.size != inventory[item.name]['bytes']):
                    raise ValueError('Unexpected/nonregular/duplicate archive member')
                digest = hashlib.sha256()
                with stream.extractfile(item) as raw:
                    for part in iter(lambda: raw.read(1024**2), b''):
                        digest.update(part)
                seen[item.name] = dict(bytes=item.size, sha256=digest.hexdigest())
        if process.wait() or seen != inventory:
            raise ValueError('Decoded archive differs from source inventory')
    finally:
        if process.poll() is None:
            process.kill()
        process.stdout.close()
        process.wait()


class Evidence:
    def __init__(self):
        self.files, self.aliases, self.runs, self.audits = {}, {}, {}, []

    def add(self, path, expected=None):
        original = Path(path).absolute()
        source = allowed_file(original)
        member = 'files/' + str(source).lstrip('/')
        safe_member(member)
        item = dict(bytes=source.stat().st_size, sha256=sha(source))
        if expected is not None and item['sha256'] != expected:
            raise ValueError('Evidence differs from declared hash: ' + str(original))
        if member in self.files and self.files[member][1] != item:
            raise ValueError('Evidence changed during discovery: ' + str(original))
        self.files[member] = (source, item)
        self.aliases[str(original)] = str(source)
        return source

    def recheck(self):
        for member, (source, item) in self.files.items():
            if source.stat().st_size != item['bytes'] or sha(source) != item['sha256']:
                raise ValueError('Source changed during preservation: ' + member)
        for original, resolved in self.aliases.items():
            if str(Path(original).resolve(strict=True)) != resolved:
                raise ValueError('Source alias changed during preservation: ' + original)

    def physical(self, run, design, final=False):
        run = Path(run).resolve(strict=True)
        if not within(run, (SCRATCH, NATIVE)) or run.name in ('', '.', '..'):
            raise ValueError('Physical run escaped this project roots')
        manifest_path = run / 'manifest.json'
        manifest = read(manifest_path)
        stage = 'finish_after_detailed_route' if final else 'global_route_only'
        if manifest.get('design') != design or manifest.get('stage') != stage:
            raise ValueError('Physical design/stage differs from declared evidence')
        previous = self.runs.get(str(run))
        if previous is not None:
            if previous['manifest_sha256'] != sha(manifest_path):
                raise ValueError('Previously discovered native manifest changed')
            return
        self.runs[str(run)] = dict(stage=stage, design=design,
            manifest_sha256=sha(manifest_path), terminal=bool(manifest.get('time_end')))
        # Bounded enumeration: only the declared physical root and two native folders.
        for source in sorted(run.iterdir()):
            if source.is_file() and source.suffix in ('.json', '.log'):
                self.add(source)
        if manifest.get('def_path'):
            source = self.add(manifest['def_path'], manifest['def_sha256'])
            for candidate in (source.with_suffix('.audit.json'),):
                if candidate.is_file():
                    self.add(candidate)
        for kind in ('logs', 'reports'):
            folder = run / 'orfs_work' / kind / 'nangate45' / design / 'eval'
            retained = 0
            if folder.exists():
                for source in sorted(folder.iterdir()):
                    if source.is_file() and source.suffix in ('.json', '.log', '.rpt'):
                        self.add(source)
                        retained += 1
            if manifest.get('success') and not retained:
                raise ValueError('Successful physical proof lacks native ' + kind)
        if final:
            final_def = run / 'orfs_work/results/nangate45' / design / 'eval/6_final.def'
            if final_def.is_file():
                self.add(final_def)
            elif manifest.get('success'):
                raise ValueError('Successful final run lacks final DEF')


def native_grt(evidence, outcome, request):
    status = outcome['status']
    if outcome.get('request_sha256') != sha(request):
        raise ValueError('GRT outcome is not bound to the declared request')
    row = read(request)
    if status not in ('success', 'physical_failure', 'geometry_failure'):
        raise ValueError('Unknown status is not a physical failure')
    if status != 'success' and outcome.get('metrics') is not None:
        raise ValueError('Failure has invented numerical PPA')
    if not outcome.get('physical_run'):
        if status != 'geometry_failure':
            raise ValueError('Only pre-backend geometry failure may have no native run')
        return dict(status=status, stage='pre_backend', metrics=None)
    run = Path(outcome['physical_run'])
    manifest = read(run / 'manifest.json')
    if sha(run / 'manifest.json') != outcome['physical_manifest_sha256']:
        raise ValueError('Changed native GRT manifest')
    if (manifest.get('def_sha256') != row.get('def_sha256') or not manifest.get('time_end') or
            int(manifest.get('num_cores', 0)) != 8 or
            manifest.get('physical_flow_revision') != outcome.get('physical_flow_revision')):
        raise ValueError('GRT is not terminal or measured another imported DEF')
    evidence.physical(run, manifest['design'])
    if manifest.get('success'):
        saved = read(run / 'metrics.json')
        if parse_grt_metrics(run, saved['runtime_seconds']) != saved:
            raise ValueError('Saved seven metrics differ from native source reparse')
        if any(type(saved.get(k)) not in (int, float) or not math.isfinite(saved[k]) for k in METRICS):
            raise ValueError('Incomplete seven-metric GRT proof')
        geometry = read(run / 'backend_macro_audit.json')  # Never create/recover an audit.
        geometry_path = run / 'backend_macro_audit.json'
        if (geometry['bindings']['physical_manifest_sha256'] != sha(run / 'manifest.json') or
                geometry['bindings']['input_def_sha256'] != sha(manifest['def_path']) or
                geometry['output_sha256'] != sha(run / 'backend_macro_audit.log')):
            raise ValueError('Saved native geometry/export proof changed')
        if outcome.get('backend_macro_geometry_sha256') != sha(geometry_path):
            raise ValueError('Outcome does not bind saved actual geometry audit')
        if status == 'success' and (not geometry['legal_no_halo'] or
                outcome['metrics'] != saved or sha(run / 'metrics.json') != outcome['metrics_sha256']):
            raise ValueError('GRT success PPA/geometry identity changed')
        if status == 'physical_failure' and (geometry['legal_no_halo'] or
                outcome.get('observed_metrics') != saved or
                outcome.get('failure_class') != 'backend_macro_overlap_or_bounds'):
            raise ValueError('Native-success failure lacks its explicit geometry cause')
    elif (status != 'physical_failure' or
            outcome.get('failure_class') != 'cts_hold_repair_max_buffer_count' or
            'floorplan_place_cts exited 2' not in manifest.get('failure', '') or
            '[ERROR RSZ-0060] Max buffer count reached.' not in (run / 'floorplan_place_cts.log').read_text() or
            'do-4_1_cts] Error 1' not in (run / 'floorplan_place_cts.log').read_text() or
            'do-cts] Error 2' not in (run / 'floorplan_place_cts.log').read_text() or
            (run / 'metrics.json').exists()):
        raise ValueError('Unrecognized physical failure; preserve only as an attempt')
    return dict(status=status, physical_run=str(run.resolve()), native_metrics_reparsed=bool(manifest.get('success')),
        saved_geometry_only=True, ODB_rehashed=False, EDA_runs=0)


def collect(plan_path, expected):
    plan_path = checked(dict(path=str(Path(plan_path).absolute()), sha256=expected))
    plan = read(plan_path)
    if plan.get('version') != VERSION or not plan.get('scope_reason'):
        raise ValueError('Unknown or unexplained preservation plan')
    for name in SOURCE_NAMES:
        if plan['source_sha256'].get(name) != sha(ROOT / name):
            raise ValueError('Preservation source closure changed: ' + name)
    evidence = Evidence()
    anchors = []
    for item in plan['anchors']:
        anchors.append(read(checked(item)))
        evidence.add(item['path'], item['sha256'])
    ids = set()
    records = plan['records']
    for record in records:
        if record['id'] in ids or record['stage'] not in ('GRT', 'DRT'):
            raise ValueError('Duplicate logical identity or unknown native stage')
        ids.add(record['id'])
        for key in ('request', 'outcome'):
            item = record.get(key)
            if item is not None:
                if not any(contains_binding(a, item) for a in anchors):
                    raise ValueError('Result is not declared by bound report/inventory: ' + item['path'])
                path = checked(item)
                if not within(path, (QROOT,)):
                    raise ValueError('Current Q-study result escaped its declared project')
                evidence.add(item['path'], item['sha256'])
        if record.get('request'):
            request = Path(record['request']['path'])
            row = read(request)
            for name in ('actions.npz', 'sample.json', 'measurement.json'):
                path = request.parent / name
                if path.is_file():
                    evidence.add(path)
            arrays_path = request.parent / 'actions.npz'
            with np.load(arrays_path, allow_pickle=False) as arrays:
                for name, inline, digest_key in (('raw_action', 'raw_action', 'raw_action_sha256'),
                        ('action', 'executed_action', 'action_sha256'), ('noise', None, 'noise_sha256')):
                    actual = np.asarray(arrays[name], dtype='<f4')
                    if not np.isfinite(actual).all() or (inline and not np.array_equal(
                            actual, np.asarray(row[inline], dtype='<f4'))):
                        raise ValueError('Raw/executed/noise arrays differ from declared request')
                    if hashlib.sha256(actual.reshape(-1).tobytes()).hexdigest() != row[digest_key]:
                        raise ValueError('Raw/executed/noise SHA differs from declared request')
            if row.get('trajectory_sha256'):
                evidence.add(request.parent / 'trajectory.pt', row['trajectory_sha256'])
            if row.get('def_path'):
                path = Path(row['def_path'])
                evidence.add(path, row['def_sha256'])
                evidence.add(path.with_suffix('.audit.json'))
        if not record.get('outcome'):
            if not record.get('request') or record['state'] != 'missing':
                raise ValueError('Missing outcome must retain a declared request and missing state')
            continue
        outcome = read(record['outcome']['path'])
        if record['state'] != outcome['status']:
            raise ValueError('Declared outcome status differs from saved source')
        if record['stage'] == 'GRT':
            if not record.get('request'):
                raise ValueError('GRT proof requires its exact request')
            audit = native_grt(evidence, outcome, Path(record['request']['path']))
        elif outcome['status'] == 'completed':
            audit = audit_final_outcome(outcome, design=record['design'], route_cap=20)
            evidence.physical(outcome['physical_run'], record['design'], final=True)
            evidence.physical(audit['source_run'], record['design'])
        elif outcome['status'] == 'upstream_physical_failure' and outcome.get('metrics') is None:
            source = checked(dict(path=outcome['source_outcome'], sha256=outcome['source_outcome_sha256']))
            if read(source).get('status') not in ('geometry_failure', 'physical_failure'):
                raise ValueError('Upstream failure is not a confirmed GRT physical failure')
            evidence.add(source)
            audit = dict(status='upstream_physical_failure', metrics=None)
        else:
            raise ValueError('Unknown detailed-route state is not numerical PPA')
        evidence.audits.append(dict(id=record['id'], stage=record['stage'], audit=audit))
    for attempt in plan.get('attempts', []):
        item = attempt['binding']
        if not any(contains_binding(a, item) for a in anchors):
            raise ValueError('Attempt is not declared by bound inventory')
        source = checked(item)
        data = read(source)
        if source.name not in ('attempt.json', 'binding.json'):
            raise ValueError('Unexpected attempt proof filename')
        evidence.add(source)
        # Exact known attempt directory only; no search for jobs or other runs.
        for path in sorted(source.parent.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.log', '.py', '.sh', '.tcl'):
                evidence.add(path)
        if data.get('physical_run') and (Path(data['physical_run']) / 'manifest.json').is_file():
            run = Path(data['physical_run'])
            manifest = read(run / 'manifest.json')
            evidence.physical(run, manifest['design'], manifest.get('stage') == 'finish_after_detailed_route')
    for item in plan.get('extra_files', []):
        evidence.add(checked(item), item['sha256'])
    historical = {}
    for reference in plan.get('historical_references', []):
        path = checked(reference)
        old = read(path)
        inventory = path.with_name('inventory.json')
        archives = {'native_evidence_durably_preserved_and_fully_decoded': 'native_evidence.tar.zst',
            'offline_inputs_and_native_evidence_preserved_and_fully_decoded': 'offline_inputs.tar.zst',
            'used_online_feedback_durably_preserved_and_fully_decoded': 'online_feedback.tar.zst'}
        archive = path.with_name(archives.get(old.get('status'), 'UNKNOWN'))
        if (old.get('status') not in archives or
                sha(inventory) != old['inventory_sha256'] or not archive.is_file() or
                archive.stat().st_size != old['archive_bytes'] or sha(archive) != old['archive_sha256']):
            raise ValueError('Historical durable preservation reference is not intact')
        historical.update(read(inventory))
        evidence.add(path)
        evidence.add(inventory)
        aliases = path.with_name('aliases.json' if old.get('version') == VERSION else 'path_aliases.json')
        evidence.add(aliases, old['aliases_sha256'])
    for item in plan.get('historical_runs', []):
        source = Path(item['path']).resolve()
        if source.name != 'manifest.json' or not within(source, (NATIVE, SCRATCH)):
            raise ValueError('Historical reference is not an own-project physical manifest')
        member = 'files/' + str(source).lstrip('/')
        if historical.get(member, {}).get('sha256') != item['sha256']:
            raise ValueError('Historical physical manifest lacks a durable inventory reference: ' + item['path'])
    for name in SOURCE_NAMES:
        evidence.add(ROOT / name, plan['source_sha256'][name])
    evidence.add(plan_path, expected)
    evidence.recheck()
    return plan, evidence


def stage(plan_path, expected, output):
    checked(dict(path=str(Path(plan_path).absolute()), sha256=expected))
    output = Path(output).absolute()
    stage_root = SCRATCH / 'qstudy_native_preservation'
    if not within(output.resolve(), (stage_root,)) or output.resolve() == stage_root.resolve():
        raise ValueError('Stage output must stay under project scratch')
    if output.exists():
        manifest = read(output / 'stage.json')
        if manifest['plan_sha256'] != expected:
            raise ValueError('Existing stage belongs to another immutable plan')
        verify_stage(output, sha(output / 'stage.json'))
        return manifest
    plan, evidence = collect(plan_path, expected)
    level = plan.get('compression_level', 3)
    if type(level) is not int or level not in (3, 10, 19):
        raise ValueError('Only separately declared zstd levels3,10,19 are supported')
    inventory = {key: item for key, (_, item) in sorted(evidence.files.items())}
    tar_bytes = 0
    for name, item in inventory.items():
        header = tarfile.TarInfo(name)
        header.size, header.mode, header.mtime = item['bytes'], 0o600, 0
        tar_bytes += len(header.tobuf(format=tarfile.PAX_FORMAT)) + ((item['bytes'] + 511) // 512) * 512
    tar_bytes += 1024
    tar_bytes += (-tar_bytes) % tarfile.RECORDSIZE
    if shutil.disk_usage(output.parent if output.parent.exists() else SCRATCH).free < 2 * tar_bytes + 1024**2:
        raise RuntimeError('Scratch free space is below conservative staging working budget')
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'inventory.json', inventory)
    write_once(output / 'aliases.json', evidence.aliases)
    write_once(output / 'audits.json', evidence.audits)
    archive = output / 'native_evidence.tar.zst'
    with archive.open('xb') as destination:
        process = subprocess.Popen(['zstd', '-T2', '-' + str(level), '--long=28', '-c'],
                                   stdin=subprocess.PIPE, stdout=destination)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as stream:
                for name, (source, item) in sorted(evidence.files.items()):
                    header = tarfile.TarInfo(name)
                    header.size, header.mode, header.mtime = item['bytes'], 0o600, 0
                    with source.open('rb') as raw:
                        stream.addfile(header, raw)
            process.stdin.close()
            if process.wait():
                raise RuntimeError('Evidence zstd process failed')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        destination.flush()
        os.fsync(destination.fileno())
    verify_archive(archive, inventory)
    evidence.recheck()
    manifest = dict(version=VERSION, status='scratch_staged_fully_decoded_not_durable',
        plan=dict(path=str(Path(plan_path).absolute()), sha256=expected), plan_sha256=expected,
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        source_bytes=sum(v['bytes'] for v in inventory.values()), file_count=len(inventory),
        uncompressed_tar_bytes=tar_bytes, scratch_working_budget_bytes=2 * tar_bytes + 1024**2,
        compression=dict(level=level, threads=2, long_window_log=28,
                         encoder_sha256=sha(Path(shutil.which('zstd')).resolve())),
        unique_physical_runs=evidence.runs, logical_records=len(plan['records']),
        state_counts=dict(Counter(r['state'] for r in plan['records'])),
        historical_unique_manifest_references=len(plan.get('historical_runs', [])),
        sidecars={name: sha(output / name) for name in ('inventory.json', 'aliases.json', 'audits.json')},
        source_sha256=plan['source_sha256'], durable_output=plan['durable_output'],
        original_files_unchanged=True, EDA_runs=0, EDA_resume_archive=False,
        limitations='Compact selected proof only; OpenDB/SPEF/intermediate archives and model/optimizer checkpoints are excluded. Saved geometry proof is retained, not a new OpenDB geometry re-audit. Keep original scratch for unfinished routing/auditors.',
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), utc=datetime.now(timezone.utc).isoformat())
    write_once(output / 'stage.json', manifest)
    return manifest


def verify_stage(output, expected):
    output = Path(output).resolve(strict=True)
    if not within(output, (SCRATCH / 'qstudy_native_preservation',)) or sha(output / 'stage.json') != expected:
        raise ValueError('Stage identity or project scratch scope changed')
    manifest = read(output / 'stage.json')
    if manifest.get('version') != VERSION or manifest.get('status') != 'scratch_staged_fully_decoded_not_durable':
        raise ValueError('Not a committed scratch stage')
    for name in SOURCE_NAMES:
        if manifest['source_sha256'].get(name) != sha(ROOT / name):
            raise ValueError('Preservation source changed after staging: ' + name)
    for name, digest in manifest['sidecars'].items():
        if name not in ('inventory.json', 'aliases.json', 'audits.json') or sha(output / name) != digest:
            raise ValueError('Stage sidecar changed')
    if set(manifest['sidecars']) != {'inventory.json', 'aliases.json', 'audits.json'}:
        raise ValueError('Incomplete stage sidecars')
    archive = output / 'native_evidence.tar.zst'
    if sha(archive) != manifest['archive_sha256'] or archive.stat().st_size != manifest['archive_bytes']:
        raise ValueError('Staged archive changed')
    inventory = read(output / 'inventory.json')
    verify_archive(archive, inventory)
    return manifest, inventory


def capacity(storage, archive_bytes, sidecar_bytes, budget):
    names = ('preservation_reserve_bytes', 'checkpoint_reserve_bytes', 'atomic_writer_reserve_bytes')
    for key in (*names, 'max_archive_bytes', 'max_sidecar_bytes'):
        if type(budget.get(key)) is not int or budget[key] < 0:
            raise ValueError('Budget must explicitly declare nonnegative integer ' + key)
    if budget['preservation_reserve_bytes'] == 0:
        raise ValueError('A nonzero preservation-only operational reserve is required')
    if archive_bytes > budget['max_archive_bytes'] or sidecar_bytes > budget['max_sidecar_bytes']:
        raise RuntimeError('Measured archive/sidecars exceed reviewed budget')
    required = archive_bytes + sidecar_bytes + sum(budget[k] for k in names)
    # Missing Ceph quota is not permission to spend unknown capacity.
    if any(type(storage.get(key)) is not int or storage[key] < required
           for key in ('filesystem_free_bytes', 'quota_remaining_bytes')):
        raise RuntimeError('Insufficient/unknown capacity after all declared reserves')
    return required


def regular_destination(path):
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('Durable payload must be a regular non-symlink file: ' + str(path))


def durable_destination(output):
    """A dedicated-root alias must never turn a scratch copy into durable proof."""
    destination_root = QROOT / 'native_evidence_preservation_v1'
    try:
        relative = output.relative_to(destination_root)
    except ValueError:
        raise ValueError('Durable destination escaped its dedicated directory')
    if not relative.parts or '..' in relative.parts:
        raise ValueError('Durable destination escaped its dedicated directory')
    if (not within(QROOT.resolve(), (BULK,)) or
            not within(output.resolve(), (QROOT,))):
        raise ValueError('Durable destination escaped actual project bulk storage')
    current = QROOT
    for component in ('native_evidence_preservation_v1', *relative.parts):
        current = current / component
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ValueError('Durable destination directory is an alias/non-directory')
    for name in ('manifest.json', 'admission.json', 'native_evidence.tar.zst',
                 'inventory.json', 'aliases.json', 'audits.json', 'stage.json', 'budget.json'):
        regular_destination(output / name)


def admit(stage_path, stage_sha, budget_path, budget_sha):
    manifest, inventory = verify_stage(stage_path, stage_sha)
    budget = read(checked(dict(path=str(Path(budget_path).absolute()), sha256=budget_sha)))
    if (budget.get('version') != VERSION or budget.get('stage_sha256') != stage_sha or
            not budget.get('review_reason')):
        raise ValueError('Durable admission requires an explicit reviewed exact-stage budget')
    output = Path(budget['durable_output']).absolute()
    if str(output) != manifest['durable_output']:
        raise ValueError('Durable destination differs from declared Q-study output')
    durable_destination(output)
    if (output / 'manifest.json').exists():
        saved = read(output / 'manifest.json')
        if (saved.get('version') != VERSION or
                saved.get('status') != 'native_evidence_durably_preserved_and_fully_decoded' or
                saved['stage_sha256'] != stage_sha or saved['budget_sha256'] != budget_sha or
                saved['archive_sha256'] != manifest['archive_sha256'] or
                saved['inventory_sha256'] != manifest['sidecars']['inventory.json'] or
                saved['aliases_sha256'] != manifest['sidecars']['aliases.json'] or
                saved['audits_sha256'] != manifest['sidecars']['audits.json']):
            raise ValueError('Durable output already belongs to another admission')
        if sha(output / 'native_evidence.tar.zst') != saved['archive_sha256']:
            raise ValueError('Committed durable archive changed')
        for name, digest in manifest['sidecars'].items():
            if sha(output / name) != digest:
                raise ValueError('Committed durable sidecar changed')
        if sha(output / 'stage.json') != stage_sha or sha(output / 'budget.json') != budget_sha:
            raise ValueError('Committed durable stage/budget changed')
        verify_archive(output / 'native_evidence.tar.zst', inventory)
        return saved
    # The lock serializes only this adapter, never pretends to lock other writers.
    lock = QROOT / '.native_evidence_preservation.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        durable_destination(output)
        identity = dict(version=VERSION, stage_sha256=stage_sha, budget_sha256=budget_sha,
                        archive_sha256=manifest['archive_sha256'])
        if output.exists() and read(output / 'admission.json') != identity:
            raise ValueError('Interrupted durable directory belongs to another admission')
        files = dict(manifest['sidecars'], **{'stage.json': stage_sha})
        for name, digest in files.items():
            if sha(Path(stage_path) / name) != digest:
                raise ValueError('Stage sidecar changed before durable copy')
        sidecar_bytes = sum((Path(stage_path) / name).stat().st_size for name in files)
        sidecar_bytes += Path(budget_path).stat().st_size + 32768
        archive = output / 'native_evidence.tar.zst'
        if archive.exists() and sha(archive) != manifest['archive_sha256']:
            raise ValueError('Interrupted committed archive changed')
        before = storage_status(QUOTA, output)
        required = capacity(before, 0 if archive.exists() else manifest['archive_bytes'], sidecar_bytes, budget)
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            write_once(output / 'admission.json', identity)
        if not archive.exists():
            # A killed partial copy is retained as evidence and never overwritten.
            pending = output / ('native_evidence.pending_' + str(os.environ.get('SLURM_JOB_ID')) +
                                '_' + str(os.environ.get('SLURM_RESTART_COUNT', '0')))
            with pending.open('xb') as target, (Path(stage_path) / 'native_evidence.tar.zst').open('rb') as raw:
                shutil.copyfileobj(raw, target, length=4 * 1024**2)
                target.flush()
                os.fsync(target.fileno())
            if sha(pending) != manifest['archive_sha256']:
                raise ValueError('Durable copy differs from verified stage')
            verify_archive(pending, inventory)
        else:
            verify_archive(archive, inventory)
        # Copy is already charged. Reserve must still be available before commit.
        after = storage_status(QUOTA, output)
        capacity(after, 0, sidecar_bytes, dict(budget, max_archive_bytes=0))
        if not archive.exists():
            pending.rename(archive)
        for name, digest in files.items():
            if sha(Path(stage_path) / name) != digest:
                raise ValueError('Stage sidecar changed before durable copy')
            regular_destination(output / name)
            if not (output / name).exists():
                with (output / name).open('xb') as target, (Path(stage_path) / name).open('rb') as raw:
                    shutil.copyfileobj(raw, target)
                    target.flush()
                    os.fsync(target.fileno())
            if sha(output / name) != digest or sha(Path(stage_path) / name) != digest:
                raise ValueError('Durable sidecar changed')
        if not (output / 'budget.json').exists():
            with (output / 'budget.json').open('xb') as target:
                target.write(Path(budget_path).read_bytes())
                target.flush()
                os.fsync(target.fileno())
        if sha(output / 'budget.json') != budget_sha:
            raise ValueError('Admission budget changed while copying')
        # Do not publish a manifest from a stage replaced after its first audit.
        verify_stage(stage_path, stage_sha)
        durable_destination(output)
        committed = dict(files, **{'budget.json': budget_sha,
                                  'native_evidence.tar.zst': manifest['archive_sha256']})
        if any(sha(output / name) != digest for name, digest in committed.items()):
            raise ValueError('Durable payload changed before manifest publication')
        saved = dict(version=VERSION, status='native_evidence_durably_preserved_and_fully_decoded',
            stage_sha256=stage_sha, budget_sha256=budget_sha, archive_sha256=manifest['archive_sha256'],
            archive_bytes=manifest['archive_bytes'], source_bytes=manifest['source_bytes'],
            file_count=manifest['file_count'], logical_records=manifest['logical_records'],
            inventory_sha256=sha(output / 'inventory.json'), aliases_sha256=sha(output / 'aliases.json'),
            audits_sha256=sha(output / 'audits.json'), storage_before=before, storage_after=after,
            required_admission_bytes=required, preservation_only_reserve_override=True,
            global_EDA_storage_guards_unchanged=True, source_files_unchanged_by_adapter=True,
            EDA_runs=0, EDA_resume_archive=False, limitations=manifest['limitations'],
            restoration='Decode only into a NEW EMPTY directory. files/ mirrors resolved absolute paths; validate every inventory SHA. Never restore over current experiments.',
            slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        write_once(output / 'manifest.json', saved)
        descriptor = os.open(str(output), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    staged = sub.add_parser('stage')
    staged.add_argument('--plan', type=Path, required=True)
    staged.add_argument('--plan-sha256', required=True)
    staged.add_argument('--output', type=Path, required=True)
    admitted = sub.add_parser('admit')
    admitted.add_argument('--stage', type=Path, required=True)
    admitted.add_argument('--stage-sha256', required=True)
    admitted.add_argument('--budget', type=Path, required=True)
    admitted.add_argument('--budget-sha256', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) > 2:
        raise RuntimeError('Preservation requires bounded CPU Slurm (at most2CPUs)')
    result = (stage(args.plan, args.plan_sha256, args.output) if args.command == 'stage' else
              admit(args.stage, args.stage_sha256, args.budget, args.budget_sha256))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
