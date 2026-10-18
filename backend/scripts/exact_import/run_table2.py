#!/usr/bin/env python3
"""Run one frozen Table 2 task through the author's unchanged physical flow.

The JSONL row must contain design, def_path, def_sha256 and run_id. Large
artifacts are durable under /fact_data; a backend/results symlink is provided.
No previous run is resumed or replaced, and intermediate flow files are kept.
The exact author metric dictionary is retained separately; primary PPA values
are extracted uniformly from final JSON after checking its text-report values.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_storage import storage_status, MIN_HEADROOM


BASE = Path(__file__).resolve().parents[2]
FLOW = BASE / 'thirdparty/OpenROAD-flow-scripts'
EVALUATOR = BASE.parent / 'upstream/flowplace/benchmark_utils/openroad/openroad_ppa_evaluator.py'
BULK = Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')
REQUIRED = ('DRT_WL', 'WNS', 'TNS', 'Power', 'StdCellArea')
ALLOWED_MAKE_OVERRIDES = frozenset(('GPL_TIMING_DRIVEN', 'GPL_ROUTABILITY_DRIVEN'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_identity(path):
    path = Path(path)
    return {'path': str(path), 'resolved_path': str(path.resolve()),
            'size_bytes': path.stat().st_size, 'sha256': sha(path)}


def validate_make_overrides(value):
    """Allow only the two predeclared paper-description diagnostic switches."""
    if not isinstance(value, dict):
        raise ValueError('task.make_overrides must be an object.')
    unknown = set(value) - ALLOWED_MAKE_OVERRIDES
    if unknown:
        raise ValueError('Unsupported make overrides: ' + ', '.join(sorted(unknown)))
    normalized = {}
    for key, item in sorted(value.items()):
        if type(item) not in (str, int) or str(item) not in ('0', '1'):
            raise ValueError(key + ' override must be integer/string 0 or 1.')
        normalized[key] = str(item)
    return normalized


def flow_inputs(meta, variant):
    """Enumerate only relevant, bounded input trees, excluding changing results."""
    files = set(path for path in FLOW.iterdir() if path.is_file())
    # All nangate45 configs are included because one design can source another's
    # scripts/SDC. Libraries, LEFs, RC rules, scripts and utilities are hashed.
    for relative in ('scripts', 'util', 'platforms/nangate45', 'designs/nangate45'):
        directory = FLOW / relative
        for root, directories, names in os.walk(directory, followlinks=False):
            for name in directories:
                if (Path(root) / name).is_symlink():
                    raise RuntimeError('Directory symlink requires an explicit provenance rule: ' + str(Path(root) / name))
            files.update(Path(root) / name for name in names if (Path(root) / name).is_file())
    # Synthesis is bypassed by the author's flow, but these exact V/SDC seeds
    # are consumed by physical design and therefore belong in its identity.
    return sorted(files)


def fingerprint(files):
    entries = []
    for path in files:
        identity = file_identity(path)
        identity['relative_path'] = str(path.relative_to(FLOW))
        entries.append(identity)
    canonical = [{'path': item['relative_path'], 'sha256': item['sha256'],
                  'size_bytes': item['size_bytes']} for item in entries]
    tree_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'tree_sha256': tree_hash, 'files': entries}


def tool_identity(executable, args):
    executable = shutil.which(executable) or executable
    result = file_identity(Path(executable))
    probe = subprocess.run([executable, *args], capture_output=True, text=True, timeout=60)
    result.update({'version_command': [executable, *args], 'version_returncode': probe.returncode,
                   'version_stdout': probe.stdout.strip(), 'version_stderr': probe.stderr.strip()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True, type=Path)
    parser.add_argument('--tasks-sha256', required=True)
    parser.add_argument('--index', type=int, default=None)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        parser.error('Submit table2_fresh.slurm; SLURM_JOB_ID is required.')
    raw_tasks = args.tasks.read_bytes()
    tasks_sha = hashlib.sha256(raw_tasks).hexdigest()
    if tasks_sha != args.tasks_sha256:
        raise RuntimeError('Task manifest differs from its frozen SHA256.')
    rows = [json.loads(line) for line in raw_tasks.decode().splitlines() if line.strip()]
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(rows):
        raise IndexError('Array index outside the frozen task manifest.')
    task = rows[index]
    method = task.get('method', 'FlowPlace')
    if method not in ('FlowPlace', 'DREAMPlace'):
        raise ValueError('Table 2 task method must be FlowPlace or DREAMPlace.')
    for key in ('design', 'def_path', 'def_sha256', 'run_id'):
        if not isinstance(task.get(key), str) or not task[key]:
            raise ValueError('Missing task field: ' + key)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', task['run_id']):
        raise ValueError('run_id must be one safe path component.')
    if task.get('variant', 'eval') != 'eval' or task.get('platform', 'nangate45') != 'nangate45':
        raise ValueError('Table 2 uses the published eval/nangate45 flow.')
    if task.get('macro_import_policy') != 'preserve_generator_dbu_coordinates_v1' or not task.get('diagnostic'):
        raise ValueError('Exact-import tasks must declare the fixed compatibility policy and diagnostic classification.')
    make_overrides = validate_make_overrides(task.get('make_overrides', {}))
    alias = BASE / 'results' / task['run_id']
    work = BULK / task['run_id']
    if os.path.lexists(alias) or os.path.lexists(work):
        raise FileExistsError('Refusing to replace an existing run: ' + task['run_id'])
    storage = storage_status(Path('/fact_data/bohanli'), BULK)
    for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
        if storage.get(key) is not None and storage[key] < MIN_HEADROOM:
            raise RuntimeError('Less than 8 GiB storage headroom: ' + key)
    work.mkdir(parents=True, exist_ok=False)
    manifest = {
        'run_id': task['run_id'], 'design': task['design'], 'variant': 'eval',
        'method': method,
        'method_evidence': ('task' if 'method' in task else 'legacy_flowplace_task_default'),
        'platform': 'nangate45', 'stage': 'full_author_flow', 'success': False,
        'evaluation_complete': False, 'verification_complete': False,
        'time_start': time.time(), 'status': 'initializing', 'hostname': socket.gethostname(),
        'task_manifest': str(args.tasks.resolve()), 'task_manifest_sha256': tasks_sha,
        'task_index': index, 'task': task, 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
        'slurm_array_job_id': os.getenv('SLURM_ARRAY_JOB_ID'),
        'slurm_array_task_id': os.getenv('SLURM_ARRAY_TASK_ID'),
        'num_cores': os.getenv('NUM_CORES'), 'cleanup_flow_work': 'never',
        'storage': dict(storage, physical_result_dir=str(work), result_alias=str(alias)),
        'stages': [], 'command': sys.argv,
        'make_overrides': make_overrides,
        'diagnostic': bool(make_overrides or task.get('diagnostic')),
        'not_release_recipe': True,
        'recipe_label': task.get('recipe_label') or ('paper_description_gpl_switch_diagnostic' if make_overrides
                                                   else 'released_author_flow'),
        'override_note': ('Predeclared diagnostic only: make command-line GPL switches override the '
                          'released config. Shared release files are unchanged; do not pool with '
                          'the primary released-recipe reproduction.' if make_overrides
                          else 'No make parameter overrides; released flow configuration retained.'),
        'success_definition': 'Author full flow and required finite final metrics present. Independent audit and DRC cleanliness are reported separately.',
    }
    save = lambda: atomic_json(work / 'manifest.json', manifest)
    save()
    active = []

    def interrupted(signum, _frame):
        manifest['failure'] = 'Interrupted by signal ' + str(signum)
        manifest['status'] = 'interrupted'
        save()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(work, target_is_directory=True)
        provenance = work / 'provenance'
        provenance.mkdir()
        (provenance / 'tasks.jsonl').write_bytes(raw_tasks)
        atomic_json(provenance / 'task.json', task)
        original_def = Path(task['def_path']).resolve()
        manifest.update(def_path=str(original_def), def_sha256=sha(original_def))
        if manifest['def_sha256'] != task['def_sha256']:
            raise RuntimeError('DEF does not match the frozen task hash.')
        executed_def = provenance / (task['design'] + '.def')
        shutil.copyfile(original_def, executed_def)
        if sha(executed_def) != task['def_sha256']:
            raise RuntimeError('DEF changed during snapshot copy.')
        manifest['executed_def_path'] = str(executed_def)
        for source in (Path(__file__), Path(__file__).with_name('macro_import.py'),
                       Path(__file__).with_name('table2_exact.slurm'),
                       BASE.parent / 'analysis/audit_table2_macro_preservation.py',
                       EVALUATOR, BASE / 'scripts/table2_fresh.slurm',
                       BASE / 'scripts/audit_fullflow.py', BASE / 'scripts/compare_paper.py',
                       BASE / 'scripts/compare_table2_method.py',
                       BASE / 'scripts/canonical_table2_metrics.py',
                       BASE.parent / 'analysis/summarize_table2_generated.py',
                       BASE / 'scripts/grt_metrics.py', BASE / 'scripts/result_storage.py',
                       BASE / 'references' / ('paper_table2_' + method.lower() + '.json')):
            shutil.copyfile(source, provenance / source.name)
        reference = provenance / ('paper_table2_' + method.lower() + '.json')
        manifest['paper_reference'] = file_identity(reference)
        manifest['runner_sha256'] = sha(provenance / Path(__file__).name)
        manifest['evaluator'] = file_identity(provenance / EVALUATOR.name)
        spec = importlib.util.spec_from_file_location('table2_author_evaluator', provenance / EVALUATOR.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        meta = module.resolve_design(str(FLOW), 'nangate45', task['design'])
        if meta is None:
            raise RuntimeError('Unknown author-flow design: ' + task['design'])
        seed_dir = module.find_synth_seed_dir(str(FLOW), 'nangate45', meta['design_nickname'], 'eval')
        if seed_dir is None:
            raise RuntimeError('Missing published synthesis seeds.')
        sources = flow_inputs(meta, 'eval')
        sources.extend(Path(seed_dir) / name for name in ('1_synth.v', '1_synth.sdc'))
        sources = sorted(set(sources))
        initial_flow = fingerprint(sources)
        atomic_json(provenance / 'flow_inputs_start.json', initial_flow)
        manifest['flow_inputs_start_sha256'] = initial_flow['tree_sha256']
        # Preserve small executable/config files in addition to hashing all
        # libraries, geometry, RC data and synthesis seeds.
        snapshot = provenance / 'flow_snapshot'
        for source in sources:
            relative = source.relative_to(FLOW)
            if source.suffix in ('.tcl', '.mk', '.py', '.sh', '.yaml', '.yml', '.sdc') or source.name == 'Makefile':
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        config = Path(meta['config_path'])
        manifest.update(design_config=str(config), design_config_sha256=sha(config),
                        design_config_exports=module.parse_config_exports(str(config)), resolved_design=meta)
        image = Path(os.environ['FLOWPLACE_TOOL_IMAGE'])
        manifest['tool_image'] = file_identity(image)
        manifest['tool_image_sha256'] = manifest['tool_image']['sha256']
        if task.get('tool_image_sha256') and task['tool_image_sha256'] != manifest['tool_image_sha256']:
            raise RuntimeError('Tool image differs from the frozen task identity.')
        archive = BASE / 'assets/OpenROAD-flow-scripts.tar.gz'
        if archive.is_file():
            manifest['flow_archive'] = file_identity(archive)
            manifest['flow_archive_sha256'] = manifest['flow_archive']['sha256']
        manifest['tools'] = {
            'openroad': tool_identity(os.environ['OPENROAD_EXE'], ['-version']),
            'yosys': tool_identity(os.environ['YOSYS_EXE'], ['-V']),
            'klayout': tool_identity(os.environ['KLAYOUT_CMD'], ['-v']),
        }
        manifest['openroad_exe'] = manifest['tools']['openroad']['path']
        manifest['openroad_sha256'] = manifest['tools']['openroad']['sha256']
        if task.get('openroad_sha256') and task['openroad_sha256'] != manifest['openroad_sha256']:
            raise RuntimeError('OpenROAD binary differs from the frozen task identity.')
        manifest['openroad_version'] = manifest['tools']['openroad']['version_stdout']
        manifest['backend_patch_sha256'] = sha(FLOW / 'scripts/write_ref_sdc.tcl')
        manifest['environment'] = {key: os.getenv(key) for key in (
            'FLOW_HOME', 'DESIGN_HOME', 'PLATFORM_HOME', 'SCRIPTS_DIR', 'UTILS_DIR',
            'WORK_HOME', 'NUM_CORES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'OPENROAD_EXE', 'YOSYS_EXE', 'KLAYOUT_CMD', 'TMPDIR')}
        manifest['status'] = 'running'
        save()
        print(json.dumps({'run_id': task['run_id'], 'recipe_label': manifest['recipe_label'],
                          'diagnostic': manifest['diagnostic'],
                          'make_overrides': make_overrides}, indent=2), flush=True)

        exact_source = provenance / 'macro_import.py'
        exact_spec = importlib.util.spec_from_file_location('table2_exact_import', exact_source)
        exact_module = importlib.util.module_from_spec(exact_spec)
        exact_spec.loader.exec_module(exact_module)
        manifest['macro_import_implementation'] = file_identity(exact_source)

        class RecordedProcess(subprocess.Popen):
            def __init__(self, command, *positional, **kwargs):
                original_command = list(command)
                command = list(command)
                if len(command) > 1 and command[0:2] == ['make', 'run_wo_synth']:
                    manifest['macro_import'] = exact_module.prepare(work, manifest)
                    save()
                if make_overrides:
                    if len(command) < 2 or command[0] != 'make' or command[1] not in ('run_mp', 'run_wo_synth'):
                        raise RuntimeError('Diagnostic overrides only support the two author make stages.')
                    command.extend(key + '=' + value for key, value in make_overrides.items())
                self.stage_record = {'command': command, 'author_command': original_command,
                                     'make_overrides': make_overrides,
                                     'cwd': str(kwargs.get('cwd', '')),
                                     'time_start': time.time(), 'status': 'starting'}
                manifest['stages'].append(self.stage_record)
                save()
                print('EXECUTED_MAKE_COMMAND ' + json.dumps(command), flush=True)
                try:
                    super().__init__(command, *positional, **kwargs)
                except BaseException as exc:
                    self.stage_record.update(status='launch_failed', failure=repr(exc), time_end=time.time())
                    save()
                    raise
                active.append(self)
                self.stage_record.update(status='running', pid=self.pid)
                save()

            def wait(self, *positional, **kwargs):
                returncode = super().wait(*positional, **kwargs)
                self.stage_record.update(returncode=returncode, time_end=time.time(),
                                         status='completed' if returncode == 0 else 'failed')
                self.stage_record['wall_seconds'] = self.stage_record['time_end'] - self.stage_record['time_start']
                save()
                return returncode

        # Replace this evaluator module's subprocess reference only. With no
        # diagnostic task overrides, all commands and flow parameters remain
        # unchanged. Explicit overrides append only the two validated switches.
        module.subprocess = SimpleNamespace(Popen=RecordedProcess, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
        result = module.run_evaluation(str(executed_def), task['design'], 'nangate45', 'eval',
                                       str(work), str(BASE), flow_work_home=str(work / 'orfs_work'),
                                       cleanup_flow_work='never', reuse_existing_logs=False)
        atomic_json(work / 'author_metrics.json', result)
        manifest['author_metrics'] = file_identity(work / 'author_metrics.json')
        # Keep the original parser result even when the physical flow failed.
        atomic_json(work / 'metrics.json', result)
        valid = bool(result) and all(isinstance(result.get(key), (int, float))
                                    and math.isfinite(result[key]) for key in REQUIRED)
        manifest['evaluation_complete'] = True
        manifest['success'] = valid
        manifest['route_drc_clean'] = result.get('DRC') == 0 if result and result.get('DRC') is not None else None
        final_flow = fingerprint(sources)
        atomic_json(provenance / 'flow_inputs_end.json', final_flow)
        manifest['flow_inputs_end_sha256'] = final_flow['tree_sha256']
        manifest['source_inputs_unchanged'] = final_flow['tree_sha256'] == initial_flow['tree_sha256']
        save()
        if not valid:
            raise RuntimeError('Author flow failed or required finite final PPA metrics are missing; retained stage logs show details.')
        if not manifest['source_inputs_unchanged']:
            raise RuntimeError('Flow inputs changed during execution; numerical results are not a frozen-flow reproduction.')
        if len(manifest['stages']) != 2 or any(stage.get('returncode') != 0 for stage in manifest['stages']):
            raise RuntimeError('Both recorded author physical stages must complete successfully.')
        if sha(exact_source) != manifest['macro_import_implementation']['sha256']:
            raise RuntimeError('Exact-import helper changed during execution.')
        if sha(provenance / 'macro_import_exact.tcl') != manifest['macro_import']['executed_tcl_sha256']:
            raise RuntimeError('Exact-import executed Tcl changed during execution.')
        canonical_source = provenance / 'canonical_table2_metrics.py'
        spec = importlib.util.spec_from_file_location('table2_canonical_metrics', canonical_source)
        canonical_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(canonical_module)
        result, metric_evidence = canonical_module.canonicalize(work, result)
        atomic_json(work / 'metrics.json', result)
        atomic_json(work / 'metric_canonicalization.json', metric_evidence)
        manifest['metric_policy'] = canonical_module.METRIC_POLICY
        manifest['canonicalizer'] = file_identity(canonical_source)
        manifest['metric_canonicalization'] = file_identity(work / 'metric_canonicalization.json')
        manifest['canonical_metrics'] = file_identity(work / 'metrics.json')
        save()
        for script, label in (('audit_fullflow.py', 'independent_audit'), ('compare_table2_method.py', 'paper_comparison')):
            if sha(BASE / 'scripts' / script) != sha(provenance / script):
                raise RuntimeError('Verification script changed during this run: ' + script)
            command = [sys.executable, str(BASE / 'scripts' / script), str(work)]
            if script == 'compare_table2_method.py':
                helper = BASE.parent / 'analysis/summarize_table2_generated.py'
                if sha(helper) != sha(provenance / helper.name):
                    raise RuntimeError('Metric verification helper changed during this run.')
                command.extend(['--reference', str(reference), '--reference-sha256', manifest['paper_reference']['sha256']])
            start = time.time()
            with (work / (label + '.log')).open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            manifest[label] = {'command': command, 'returncode': completed.returncode,
                               'wall_seconds': time.time() - start}
            save()
            if completed.returncode != 0:
                raise RuntimeError(label + ' failed; see retained log.')
        manifest['verification_complete'] = True
        manifest['status'] = 'completed'
        print(json.dumps({'run_id': task['run_id'], 'metrics': result,
                          'route_drc_clean': manifest['route_drc_clean'], 'work': str(work)}, indent=2), flush=True)
    except BaseException as exc:
        manifest.setdefault('failure', repr(exc))
        manifest['traceback'] = traceback.format_exc()
        if manifest['status'] != 'interrupted':
            manifest['status'] = 'failed'
        raise
    finally:
        manifest['time_end'] = time.time()
        manifest['wall_seconds'] = manifest['time_end'] - manifest['time_start']
        save()


if __name__ == '__main__':
    main()
