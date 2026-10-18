#!/usr/bin/env python3
"""Candidate standard DREAMPlace 4.1 macro baseline; no backend PPA is run.

The missing author replace.def is reconstructed from released topology with
all movable positions reset. Only original DREAMPlace operators are executed.
This is an explicitly derived smoke protocol, not an exact published recipe.
"""

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'backend'
FLOW = BASE / 'thirdparty/OpenROAD-flow-scripts'
INSTALLED = ROOT / 'iccad/assets/dmp_install_cpu'
SOURCE = ROOT / 'iccad/assets/DREAMPlace-4.1.0'
BULK = Path('/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2')
COMPONENT = re.compile(r'^\s*-\s+(\S+)\s+(\S+)(.*?);', re.M | re.S)
POSITION = re.compile(r'\+\s+(FIXED|PLACED|COVER)\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)\s+(\S+)')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    def scalar(item):
        if hasattr(item, 'tolist'):
            return item.tolist()
        raise TypeError('Unsupported JSON value: ' + type(item).__name__)
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n')
    tmp.replace(path)


def canonical(name):
    return re.sub(r'\\(.)', r'\1', name)


def split_def(text):
    header = re.search(r'^COMPONENTS\s+(\d+)\s*;\s*\n', text, re.M)
    if not header:
        raise ValueError('DEF has no COMPONENTS section.')
    end = text.index('END COMPONENTS', header.end())
    body = text[header.end():end]
    entries = list(COMPONENT.finditer(body))
    if len(entries) != int(header[1]):
        raise ValueError('DEF component parse count differs from its declaration.')
    return text[:header.end()], body, text[end:], entries


def inventories(text):
    _, _, _, entries = split_def(text)
    net = re.search(r'^NETS\s+(\d+)\s*;', text, re.M)
    pins = re.search(r'^PINS\s+(\d+)\s*;', text, re.M)
    design = re.search(r'^DESIGN\s+(\S+)\s*;', text, re.M)
    if not net or not pins or not design:
        raise ValueError('Expected explicit NETS, PINS and DESIGN headers.')
    components = {canonical(e[1]): e[2] for e in entries}
    if len(components) != len(entries):
        raise ValueError('Duplicate normalized DEF component names.')
    return {'components': components, 'net_count': int(net[1]),
            'pin_count': int(pins[1]), 'top_module': design[1]}


def fixed_io_positions(text):
    header = re.search(r'^PINS\s+\d+\s*;\s*\n', text, re.M)
    if not header:
        raise ValueError('Missing DEF PINS section.')
    section = text[header.end():text.index('END PINS', header.end())]
    result = {}
    for match in re.finditer(r'^\s*-\s+(\S+)(.*?);', section, re.M | re.S):
        position = POSITION.search(match[0])
        if not position:
            raise ValueError('IO pin lacks an explicit position: ' + match[1])
        result[canonical(match[1])] = {'status': position[1], 'orientation': position[4],
                                     'xy_dbu': [float(position[2]), float(position[3])]}
    return result


def def_connections(text):
    header = re.search(r'^NETS\s+(\d+)\s*;\s*\n', text, re.M)
    if not header:
        raise ValueError('Missing explicit NETS section.')
    section = text[header.end():text.index('END NETS', header.end())]
    nets = {}
    for match in re.finditer(r'^\s*-\s+(\S+)(.*?);', section, re.M | re.S):
        name = canonical(match[1])
        if name in nets:
            raise ValueError('Duplicate DEF net name: ' + name)
        # Ignore + USE/ROUTED/PROPERTY clauses, whose parentheses are not pins.
        connections = re.split(r'\s+\+', match[2], maxsplit=1)[0]
        nets[name] = Counter((canonical(node), canonical(pin)) for node, pin in
                             re.findall(r'\(\s*(\S+)\s+(\S+)\s*\)', connections))
    if len(nets) != int(header[1]):
        raise ValueError('DEF NETS parsing count mismatch.')
    return nets


def connection_digest(nets):
    digest = hashlib.sha256()
    for name, connections in sorted(nets.items()):
        digest.update(json.dumps([name, sorted((node, pin, count) for (node, pin), count in connections.items())],
                                 separators=(',', ':')).encode() + b'\n')
    return digest.hexdigest()


def synth_inventory(path, top):
    """Inventory the published Yosys hierarchy; unsupported syntax fails count checks."""
    modules, current = {}, None
    for line in path.open():
        match = re.match(r'^module\s+(\S+?)(?:\s|\()', line)
        if match:
            current = match[1].lstrip('\\')
            modules[current] = []
            continue
        if line.startswith('endmodule'):
            current = None
            continue
        match = re.match(r'^\s{2}(\S+)\s+(\S+)\s*\(', line)
        if match and current:
            modules[current].append((match[2].lstrip('\\'), match[1].lstrip('\\')))
    leaves = {}

    def visit(module, prefix='', ancestors=()):
        if module in ancestors:
            raise ValueError('Recursive Verilog hierarchy.')
        for name, master in modules[module]:
            instance = prefix + name
            if master in modules and modules[master]:
                visit(master, instance + '/', ancestors + (module,))
            else:
                if instance in leaves:
                    raise ValueError('Duplicate elaborated leaf instance: ' + instance)
                leaves[instance] = master
    visit(top)
    return leaves


def neutral_def(text):
    prefix, body, suffix, entries = split_def(text)
    points = re.search(r'^DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', text, re.M)
    if not points:
        raise ValueError('Only rectangular DIEAREA is supported.')
    x0, y0, x1, y1 = map(int, points.groups())
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    macros, movable, fixed = {}, {}, {}

    def replace(match):
        name, master = canonical(match[1]), match[2]
        position = POSITION.search(match[0])
        if not position:
            raise ValueError('Template component lacks an explicit orientation/position: ' + name)
        is_macro = master.startswith('fakeram')
        target = macros if is_macro else None
        if target is not None:
            target[name] = {'master': master, 'orientation': position[4]}
        if is_macro or position[1] == 'PLACED':
            movable[name] = master
            return POSITION.sub('+ PLACED ( {} {} ) {}'.format(cx, cy, position[4]), match[0], count=1)
        fixed[name] = {'master': master, 'status': position[1], 'xy_dbu': [float(position[2]), float(position[3])],
                       'orientation': position[4]}
        return match[0]
    new_body = COMPONENT.sub(replace, body)
    result = prefix + new_body + suffix
    check_prefix, _, check_suffix, _ = split_def(result)
    if check_prefix != prefix or check_suffix != suffix:
        raise AssertionError('Non-component sections changed during neutralization.')
    return result, {'macros': macros, 'movable': movable, 'fixed': fixed,
                    'neutral_center_dbu': [cx, cy], 'non_component_sections_preserved': True,
                    'initialization': 'All movable coordinates replaced before original random-center initialization.'}


def positioned_def(template, positions, fix_macros=False):
    prefix, body, suffix, _ = split_def(template)

    def replace(match):
        name = canonical(match[1])
        position = POSITION.search(match[0])
        if not position:
            raise ValueError('Missing component position: ' + name)
        if name not in positions:
            return match[0]
        x, y = positions[name]
        if fix_macros:
            # Preserve the stock final DEF's exact coordinate tokens. This
            # handoff changes status only, never position or rounding.
            if max(abs(float(position[2]) - x), abs(float(position[3]) - y)) > 1e-6:
                raise ValueError('Handoff coordinates differ from the stock output: ' + name)
            fixed = re.sub(r'^(\+\s+)(FIXED|PLACED|COVER)', r'\1FIXED', position[0], count=1)
            return match[0][:position.start()] + fixed + match[0][position.end():]
        # Stage DEF is a diagnostic export only. Exact solver coordinates stay
        # in before/after.npz; integer DBU keeps this view syntactically valid.
        return POSITION.sub('+ {} ( {} {} ) {}'.format(position[1], round(x), round(y), position[4]), match[0], count=1)
    return prefix + COMPONENT.sub(replace, body) + suffix


def orientation_audit(source_text, neutral, geometry, row_height_dbu):
    """Audit stock rawdb.apply's row-aligned vertical flip, without changing it."""
    rows = sorted((int(y), orient) for y, orient in re.findall(
        r'^ROW\s+\S+\s+\S+\s+-?\d+\s+(-?\d+)\s+(\S+)\s+DO\s+\d+\s+BY\s+1\s+STEP', source_text, re.M))
    if len(rows) < 2 or len({row[0] for row in rows}) != len(rows):
        raise RuntimeError('Orientation audit requires one uniform row per y coordinate.')
    pitch = rows[1][0] - rows[0][0]
    if any(b[0] - a[0] != pitch for a, b in zip(rows, rows[1:])):
        raise RuntimeError('Orientation audit requires uniform row pitch.')
    if abs(pitch - row_height_dbu) > 1e-6:
        raise RuntimeError('DEF row pitch differs from stock LEF row height used by getRowIndex.')
    audited = {}
    for escaped_name, macro in geometry['macros'].items():
        name = canonical(escaped_name)
        before = neutral['macros'][name]['orientation']
        after = macro['orientation']
        # Published templates and rows use N/FS only. These have identical
        # rectangular footprints and differ by y -> height-y for cell pins.
        if before not in ('N', 'FS') or after not in ('N', 'FS'):
            raise RuntimeError('Unaudited macro orientation: ' + name)
        row_id = max(0, min(len(rows) - 1, int((macro['xy_dbu'][1] - rows[0][0]) // pitch)))
        row_y, row_orientation = rows[row_id]
        if row_orientation not in ('N', 'FS'):
            raise RuntimeError('Unaudited row orientation: ' + row_orientation)
        # place_io.cpp apply: row == vflip(node) causes a vertical flip;
        # with N/FS-only inputs this is exactly final orientation == row.
        expected = row_orientation
        audited[name] = {'before': before, 'after': after, 'changed': before != after,
                         'expected_from_stock_apply': expected, 'row_index': row_id,
                         'row_y_dbu': row_y, 'row_orientation': row_orientation,
                         'footprint_preserved': True, 'matches_stock_apply': after == expected}
    return {'source': 'dreamplace/ops/place_io/src/place_io.cpp apply and PlaceDB.h getRowIndex',
            'policy': 'Preserve stock final orientation; never force original orientation back.',
            'changed_macro_count': sum(item['changed'] for item in audited.values()),
            'all_match_stock_apply': all(item['matches_stock_apply'] for item in audited.values()),
            'macros': audited}


def macro_pin_offsets(db, macro_names):
    """Read Python offsets normalized to N, independent of parser ordering."""
    decode = lambda value: canonical(value.decode() if isinstance(value, bytes) else str(value))
    names = [decode(name) for name in db.node_names]
    nets = [decode(name) for name in db.net_names]
    result = {}
    for index, node_id in enumerate(db.pin2node_map):
        node_id = int(node_id)
        name = names[node_id]
        if name not in macro_names:
            continue
        key = (name, decode(db.pin_names[index]), nets[int(db.pin2net_map[index])])
        result.setdefault(key, []).append((float(db.pin_offset_x[index]), float(db.pin_offset_y[index])))
    return {key: sorted(values) for key, values in result.items()}


def raw_macro_pin_offsets(db, macro_names):
    """Read physical-orientation pin offsets from bound raw C++ scalar APIs."""
    raw, result, geometry = db.rawdb, {}, {}
    for node in raw.nodes():
        name = canonical(raw.nodeName(node))
        if name not in macro_names:
            continue
        orient = int(node.orient())
        if orient not in (0, 5):
            raise RuntimeError('Raw macro orientation is not N/FS: ' + name)
        geometry[name] = {'xy_dbu': [node.xl(), node.yl()], 'width_dbu': node.width(),
                          'height_dbu': node.height(), 'orientation': {0: 'N', 5: 'FS'}[orient]}
        for pin_id in node.pins():
            pin = raw.pin(pin_id)
            key = (name, canonical(raw.macroPin(pin).name()), canonical(raw.netName(raw.net(pin.netId()))))
            # Pin.name and Point.x/y are not exposed by this stock binding.
            # Node.pinX/Y are bound scalar APIs and read actual raw pin.offset.
            result.setdefault(key, []).append((node.pinX(pin) - node.xl(), node.pinY(pin) - node.yl()))
    return {key: sorted(values) for key, values in result.items()}, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', default='bp_fe', choices=['bp_fe', 'bp', 'ariane133', 'ariane136', 'bp_be', 'swerv_wrapper'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--halo', type=float, help='Explicit same-axis halo in DMP scaled-coordinate units; default author config.')
    parser.add_argument('--gpu', action='store_true', help='Use one allocated GPU and an audited CUDA-enabled installation.')
    parser.add_argument('--install', type=Path, default=INSTALLED, help='DREAMPlace installation root; default existing CPU installation.')
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        parser.error('Run through the Slurm wrapper.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Use a safe run-id basename.')
    if args.halo is not None and (args.halo < 0 or not __import__('math').isfinite(args.halo)):
        parser.error('--halo must be finite and nonnegative.')
    installed = args.install.resolve()
    if args.gpu and not (installed.parent / 'BUILD_AND_OPERATOR_TESTS_PASSED').is_file():
        parser.error('GPU installation requires the recorded build/operator-test success marker.')
    work = BULK / args.run_id
    alias = BASE / 'results' / args.run_id
    if os.path.lexists(work) or os.path.lexists(alias):
        raise FileExistsError('Refusing to overwrite existing results.')
    from result_storage import storage_status, MIN_HEADROOM
    storage = storage_status(Path('/fact_data/bohanli'), BULK)
    for key in ('filesystem_free_bytes', 'quota_remaining_bytes'):
        if storage.get(key) is not None and storage[key] < MIN_HEADROOM:
            raise RuntimeError('Less than 8 GiB storage headroom: ' + key)
    work.mkdir(parents=True, exist_ok=False)
    manifest = {'design': args.design, 'run_id': args.run_id, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                'time_start': time.time(), 'success': False, 'ppa_evaluated': False,
                'protocol': 'derived_DREAMPlace_4.1_' + ('GPU' if args.gpu else 'CPU') + '_original_two_stage_macro_smoke',
                'requested_device': 'cuda' if args.gpu else 'cpu', 'installation_root': str(installed),
                'not_exact_author_baseline': True, 'result_dir': str(work), 'macro_legalizer_calls': []}
    manifest['storage'] = storage
    manifest['slurm_stdout_log'] = str(BASE / 'logs' / ('table2_dmp_' + os.environ['SLURM_JOB_ID'] + '.log'))
    save = lambda: write_json(work / 'manifest.json', manifest)
    save()
    try:
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(work, target_is_directory=True)
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(work / 'dreamplace.log')])
        author_config = ROOT / 'upstream/flowplace/benchmark_utils/openroad/dmp_config/or_cases' / (args.design + '.json')
        original_def = BASE / 'releases/flowplace_DEFS/openroad' / (args.design + '.def')
        seed = FLOW / 'results/nangate45' / args.design / 'eval'
        source_text = original_def.read_text()
        inv = inventories(source_text)
        expected_connections = def_connections(source_text)
        verilog_instances = synth_inventory(seed / '1_synth.v', inv['top_module'])
        check = {'def_components': len(inv['components']), 'synth_leaf_instances': len(verilog_instances),
                 'def_net_count': inv['net_count'], 'def_pin_count': inv['pin_count'],
                 'only_def': sorted(set(inv['components']) - set(verilog_instances)),
                 'only_synth': sorted(set(verilog_instances) - set(inv['components'])),
                 'changed_masters': [name for name in set(inv['components']) & set(verilog_instances)
                                     if inv['components'][name] != verilog_instances[name]],
                 'scope': 'Hierarchically expanded instance/master inventory; not a pin-connectivity equivalence proof.'}
        write_json(work / 'input_inventory_audit.json', check)
        if check['only_def'] or check['only_synth'] or check['changed_masters']:
            raise RuntimeError('DEF/synthesis instance inventories differ; retained audit requires inspection.')
        source_snapshot = work / 'original.def'
        source_snapshot.write_text(source_text)
        derived_text, neutral = neutral_def(source_text)
        expected_count = {'ariane133': 132, 'ariane136': 136, 'bp': 24, 'bp_be': 10, 'bp_fe': 11, 'swerv_wrapper': 28}[args.design]
        if len(neutral['macros']) != expected_count:
            raise RuntimeError('Template fakeram count differs from the audited benchmark count.')
        derived = work / (args.design + '.neutral.def')
        derived.write_text(derived_text)
        write_json(work / 'neutralization.json', neutral)
        original_cfg = json.loads(author_config.read_text())
        schema = json.loads((installed / 'dreamplace/params.json').read_text())
        ignored = {key: value for key, value in original_cfg.items() if key not in schema}
        cfg = {key: value for key, value in original_cfg.items() if key in schema}
        for key in ('lef_input', 'lib_input'):
            cfg[key] = [str(FLOW / 'platforms/nangate45' / value.split('nangate45/', 1)[1]) for value in original_cfg[key]]
            if not all(Path(path).is_file() for path in cfg[key]):
                raise FileNotFoundError('Missing remapped ' + key)
        def_units = int(re.search(r'^UNITS DISTANCE MICRONS\s+(\d+)\s*;', source_text, re.M)[1])
        lef_units = set()
        for lef in cfg['lef_input']:
            match = re.search(r'DATABASE\s+MICRONS\s+(\d+)\s*;', Path(lef).read_text())
            if match:
                lef_units.add(int(match[1]))
        if lef_units != {def_units}:
            raise RuntimeError('LEF and DEF raw units differ; candidate requires an explicit conversion protocol.')
        manifest['coordinate_units'] = {'def_dbu_per_um': def_units, 'lef_dbu_per_um': sorted(lef_units),
                                        'unscaled_dmp_coordinates_are_def_dbu': True}
        cfg.update(def_input=str(derived), gpu=int(args.gpu), num_threads=int(os.environ['SLURM_CPUS_PER_TASK']),
                   macro_place_flag=1, use_bb=1, random_center_init_flag=1,
                   result_dir=str(work / 'placement'), detailed_place_engine='', detailed_place_command='',
                   detailed_place_flag=0, timing_opt_flag=0, routability_opt_flag=0,
                   two_stage_density_scaler=1000)
        if len(cfg['global_place_stages']) != 1:
            raise RuntimeError('Expected one configured GP stage; stock macro mode duplicates it internally.')
        if args.halo is not None:
            cfg.update(macro_halo_x=args.halo, macro_halo_y=args.halo)
        write_json(work / 'author_config.json', original_cfg)
        write_json(work / 'config.json', cfg)
        write_json(work / 'unapplied_custom_keys.json', {'keys': ignored,
                   'reason': 'Absent from standard installed 4.1 schema; not claimed to have an effect.'})
        manifest['config_changes'] = {key: {'before': original_cfg.get(key), 'after': value}
                                      for key, value in cfg.items() if value != original_cfg.get(key)}
        manifest['unapplied_custom_keys'] = ignored
        input_files = [original_def, derived, source_snapshot, author_config, seed / '1_synth.v', seed / '1_synth.sdc']
        input_files.extend(Path(path) for path in cfg['lef_input'] + cfg['lib_input'])
        manifest['input_sha256'] = {str(path): sha(path) for path in input_files}
        manifest['source_commit'] = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        (work / 'source.patch').write_text(subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--submodule=diff'], text=True))
        install_hashes = {str(path.relative_to(installed)): sha(path)
                          for path in sorted((installed / 'dreamplace').rglob('*'))
                          if path.is_file() and path.suffix in ('.py', '.so', '.json')}
        write_json(work / 'installation_file_hashes.json', install_hashes)
        manifest['installation_file_hashes_sha256'] = sha(work / 'installation_file_hashes.json')
        manifest['config_sha256'] = sha(work / 'config.json')
        shutil.copyfile(__file__, work / Path(__file__).name)
        manifest['runner_sha256'] = sha(Path(__file__))
        if args.gpu:
            build = installed.parent
            required_build_files = ['BUILD_AND_OPERATOR_TESTS_PASSED', 'environment.json', 'source_commit.txt',
                                    'source_file_hashes.json', 'installation_file_hashes.json']
            if any(not (build / name).is_file() for name in required_build_files):
                raise RuntimeError('GPU build provenance is incomplete.')
            manifest['gpu_build_metadata'] = {name: {'path': str(build / name), 'sha256': sha(build / name)}
                for name in required_build_files + ['source.patch', 'submodule_worktree_patches.txt',
                    'submitted_script.slurm', 'installed_configure.json', 'loaded_operator_modules.json',
                    'loaded_cuda_libraries.txt', 'nvcc.txt', 'gcc.txt', 'cmake.txt', 'cuda_arch.txt',
                    'python_packages.json', 'exit_code.txt', 'test_hpwl.log',
                    'test_weighted_average_wirelength.log', 'test_electric_potential.log']
                if (build / name).is_file()}
            completion_path = build / 'build_completion.json'
            successful_build = build
            if completion_path.is_file():
                completion = json.loads(completion_path.read_text())
                successful_build = Path(completion['retry_dir']).resolve()
                if not successful_build.is_relative_to(build / 'retries'):
                    raise RuntimeError('GPU successful retry directory lies outside recorded build retries.')
                retry_files = ['compatibility.json', 'cccl_namespace.patch', 'submitted_script.slurm',
                               'configure.log', 'installed_configure.json', 'loaded_operator_modules.json',
                               'loaded_cuda_libraries.txt', 'CMakeCache.txt', 'exit_code.txt',
                               'test_hpwl.log', 'test_weighted_average_wirelength.log',
                               'test_electric_potential.log', 'test_pin_pos.log']
                if 'test_fixture_patch_sha256' in completion:
                    retry_files += ['test_fixture.patch', 'validation_continuation.json',
                                    'weighted_average_wirelength_unittest.py']
                if any(not (successful_build / name).is_file() for name in retry_files):
                    raise RuntimeError('GPU retry compatibility or operator-test provenance is incomplete.')
                if sha(successful_build / 'cccl_namespace.patch') != completion['compatibility_patch_sha256']:
                    raise RuntimeError('GPU retry compatibility patch hash does not match build completion.')
                if 'test_fixture_patch_sha256' in completion:
                    if sha(successful_build / 'test_fixture.patch') != completion['test_fixture_patch_sha256']:
                        raise RuntimeError('GPU validation fixture patch hash does not match build completion.')
                    continuation = json.loads((successful_build / 'validation_continuation.json').read_text())
                    if sha(successful_build / 'weighted_average_wirelength_unittest.py') != continuation['fixture_adapted_sha256']:
                        raise RuntimeError('GPU adapted test fixture differs from its recorded checksum.')
                    for name, record in continuation['reused_successful_artifacts'].items():
                        if sha(successful_build / name) != record['sha256']:
                            raise RuntimeError('GPU reused validation artifact differs from its recorded checksum: ' + name)
                manifest['gpu_build_completion'] = {'path': str(completion_path), 'sha256': sha(completion_path),
                                                    'contents': completion}
                manifest['gpu_retry_metadata'] = {name: {'path': str(successful_build / name),
                                                         'sha256': sha(successful_build / name)}
                                                  for name in retry_files}
            if (successful_build / 'exit_code.txt').read_text().strip() != '0':
                raise RuntimeError('GPU successful build or retry did not finish with exit code zero.')
            if (build / 'source_commit.txt').read_text().strip() != manifest['source_commit']:
                raise RuntimeError('GPU build source revision differs from the recorded source checkout.')
            expected_installation = json.loads((build / 'installation_file_hashes.json').read_text())
            expected_installation = {name: digest for name, digest in expected_installation.items()
                                     if name.startswith('dreamplace/')}
            if set(expected_installation) != set(install_hashes):
                raise RuntimeError('GPU installed file inventory differs from the tested build.')
            changed_installation = [name for name, digest in install_hashes.items()
                                    if expected_installation.get(name) != digest]
            if changed_installation:
                raise RuntimeError('GPU installation differs from the tested build: ' + repr(changed_installation[:10]))
            operator_patch_path = build / 'operator_patch_validation.json'
            if operator_patch_path.exists():
                patch = json.loads(operator_patch_path.read_text())
                if not patch.get('success') or not patch['validation'].get('snapshot_replay_passed') or not patch['validation'].get('after_legal'):
                    raise RuntimeError('Local operator patch has no successful independent snapshot validation.')
                patch_checks = {'parent_installation_file_hashes.json': 'parent_installation_manifest_sha256',
                                'abacus_floor_div.patch': 'patch_sha256',
                                'abacus_legalize_cpu.h': 'source_header_sha256',
                                'parent_build_completion.json': 'inherited_parent_completion_sha256',
                                'compile_commands.json': 'compile_commands_sha256',
                                'validate_dreamplace_abacus_patch.py': 'validator_sha256',
                                'patched_validation/validation.json': 'validation_report_sha256'}
                for name, digest_key in patch_checks.items():
                    if sha(build / name) != patch[digest_key]:
                        raise RuntimeError('Local operator patch evidence changed: ' + name)
                parent_hashes = json.loads((build / 'parent_installation_file_hashes.json').read_text())
                parent_hashes = {name: value for name, value in parent_hashes.items() if name.startswith('dreamplace/')}
                actual_changes = sorted(name for name in parent_hashes if install_hashes.get(name) != parent_hashes[name])
                expected_change = 'dreamplace/ops/abacus_legalize/abacus_legalize_cpp.cpython-39-x86_64-linux-gnu.so'
                if set(parent_hashes) != set(install_hashes) or actual_changes != [expected_change] or patch['changed_installed_files'] != actual_changes:
                    raise RuntimeError('Local operator patch changed more than the one recorded Abacus binary.')
                if patch['original_binary_sha256'] != parent_hashes[expected_change] or patch['patched_binary_sha256'] != install_hashes[expected_change]:
                    raise RuntimeError('Local operator patch binary hash is inconsistent.')
                manifest['operator_patch'] = {'path': str(operator_patch_path), 'sha256': sha(operator_patch_path),
                    'contents': patch, 'evidence': {name: {'path': str(build / name), 'sha256': sha(build / name)}
                                                   for name in patch_checks}}
                manifest['protocol'] += '_abacus_row_boundary_floorDiv_fix'
                manifest['local_algorithm_implementation_correction'] = 'Single Abacus row-bin lower-bound truncation replaced by the existing library floorDiv to handle floating-point row-boundary roundoff; objectives, optimizer, seed and halo unchanged.'
        manifest['halo_note'] = 'Values add to scaled DMP coordinates. No unit conversion or custom macro legalizer is applied.'
        save()

        # Imports occur only inside this scheduled execution, after provenance
        # setup. Use the existing successful CPU installation without changes.
        sys.path[:0] = [str(installed / 'dreamplace'), str(installed), str(ROOT / 'iccad/scripts')]
        import numpy as np
        import torch
        import Params
        import PlaceDB
        import NonLinearPlace
        from dreamplace import configure
        from audit_inputs import inspect_def, lef_blocks
        if not Path(configure.__file__).resolve().is_relative_to(installed):
            raise RuntimeError('DREAMPlace configure was imported outside the requested installation.')
        environment = {'python_executable': sys.executable, 'python_version': sys.version,
                       'hostname': os.uname().nodename, 'torch_version': torch.__version__,
                       'torch_compiled_cuda': torch.version.cuda, 'torch_file': torch.__file__,
                       'numpy_version': np.__version__, 'configure_file': configure.__file__,
                       'compile_configurations': configure.compile_configurations,
                       'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
                       'slurm_gpus': os.getenv('SLURM_JOB_GPUS'), 'num_threads': cfg['num_threads']}
        environment['python_module_search_path'] = sys.path
        environment['drawing_import_dependencies'] = {
            name: {'file': module.__file__, 'version': getattr(module, '__version__', None),
                   'file_sha256': sha(module.__file__)}
            for name in ('cffi', '_cffi_backend', 'cairocffi')
            for module in [sys.modules.get(name)] if module is not None and getattr(module, '__file__', None)}
        if args.gpu:
            if configure.compile_configurations['CUDA_FOUND'] != 'TRUE' or not torch.cuda.is_available():
                raise RuntimeError('Requested GPU path unavailable; CPU fallback is forbidden.')
            if torch.cuda.device_count() != 1:
                raise RuntimeError('GPU candidate requires exactly one scheduler-visible GPU.')
            environment.update(gpu_name=torch.cuda.get_device_name(0),
                               gpu_capability=list(torch.cuda.get_device_capability(0)))
            build_environment = json.loads((installed.parent / 'environment.json').read_text())
            device_checks = {'gpu_name': build_environment['gpu'],
                             'gpu_capability': build_environment['capability'],
                             'torch_version': build_environment['torch'],
                             'torch_compiled_cuda': build_environment['torch_compiled_cuda']}
            mismatched_device = {key: {'build': expected, 'runtime': environment[key]}
                                 for key, expected in device_checks.items() if environment[key] != expected}
            if mismatched_device:
                raise RuntimeError('GPU runtime differs from the tested hardware/software cohort: ' + repr(mismatched_device))
            environment['matches_tested_build_device_and_torch'] = True
        write_json(work / 'execution_environment.json', environment)
        manifest['execution_environment_sha256'] = sha(work / 'execution_environment.json')
        manifest['execution_environment'] = environment
        save()
        params = Params.Params()
        params.load(str(work / 'config.json'))
        np.random.seed(params.random_seed)
        torch.set_num_threads(params.num_threads)
        db = PlaceDB.PlaceDB()
        db(params)
        names = [canonical(name.decode() if isinstance(name, bytes) else str(name)) for name in db.node_names]
        expected_macros = set(neutral['macros'])
        detected_macros = {names[index] for index in np.flatnonzero(db.movable_macro_mask)}
        legalizer_macros = {names[index] for index in range(db.num_movable_nodes)
                            if db.node_size_y[index] > 5 * db.row_height}
        db_check = {'num_physical_nodes': int(db.num_physical_nodes), 'num_movable_nodes': int(db.num_movable_nodes),
                    'num_terminal_NIs': int(db.num_terminal_NIs), 'num_nets': int(db.num_nets),
                    'expected_component_count': len(inv['components']), 'expected_net_count': inv['net_count'],
                    'expected_macro_names': sorted(expected_macros), 'detected_macro_names': sorted(detected_macros),
                    'legalizer_height_gt_5rows_macro_names': sorted(legalizer_macros),
                    'scale_factor': float(params.scale_factor), 'shift_factor': list(map(float, params.shift_factor)),
                    'macro_halo_internal': [params.macro_halo_x, params.macro_halo_y]}
        write_json(work / 'dmp_database_audit.json', db_check)
        if db.num_physical_nodes - db.num_terminal_NIs != len(inv['components']) or db.num_nets != inv['net_count']:
            raise RuntimeError('DMP component/net counts differ from the input DEF.')
        if detected_macros != expected_macros:
            raise RuntimeError('Original DMP macro classifier differs from the fakeram set.')
        if legalizer_macros != expected_macros:
            raise RuntimeError('Original C++ macro legalizer height classifier differs from the fakeram set.')
        if set(names[:db.num_movable_nodes]) != set(neutral['movable']):
            raise RuntimeError('DMP movable instance set differs from the derived DEF.')
        net_names = [canonical(name.decode() if isinstance(name, bytes) else str(name)) for name in db.net_names]
        db_connections = {name: Counter() for name in net_names}
        for pin_index, net_id in enumerate(db.pin2net_map):
            node_id = int(db.pin2node_map[pin_index])
            pin = db.pin_names[pin_index]
            pin = canonical(pin.decode() if isinstance(pin, bytes) else str(pin))
            pair = ('PIN', names[node_id]) if node_id >= db.num_physical_nodes - db.num_terminal_NIs else (names[node_id], pin)
            db_connections[net_names[int(net_id)]][pair] += 1
        mismatches = [name for name in set(expected_connections) | set(db_connections)
                      if expected_connections.get(name) != db_connections.get(name)]
        connectivity_check = {'def_connection_sha256': connection_digest(expected_connections),
                              'dmp_connection_sha256': connection_digest(db_connections),
                              'net_count': len(expected_connections),
                              'def_pin_connections': sum(sum(pins.values()) for pins in expected_connections.values()),
                              'dmp_pin_connections': len(db.pin2net_map),
                              'mismatched_net_count': len(mismatches),
                              'mismatched_net_examples': sorted(mismatches)[:20],
                              'scope': 'Per-net instance/pin multisets from DEF versus parsed DMP database, including repeated pins and IO.'}
        write_json(work / 'connectivity_audit.json', connectivity_check)
        if mismatches:
            raise RuntimeError('DMP parsed connectivity differs from the source DEF.')
        initial_macro_pins = macro_pin_offsets(db, expected_macros)
        initial_raw_pins, initial_raw_geometry = raw_macro_pin_offsets(db, expected_macros)
        placer = NonLinearPlace.NonLinearPlace(params, db, timer=None)
        manifest['actual_placement_tensor_device'] = str(placer.pos[0].device)
        if bool(placer.pos[0].is_cuda) != args.gpu:
            raise RuntimeError('Actual placement tensor device differs from the requested CPU/GPU mode.')
        original_sizes_x = placer.data_collections.node_size_x.detach().cpu().numpy().copy()
        original_sizes_y = placer.data_collections.node_size_y.detach().cpu().numpy().copy()
        initial = placer.pos[0].detach().cpu().numpy().copy()
        if not np.isfinite(initial).all():
            raise RuntimeError('DMP initialization contains nonfinite coordinates.')
        np.savez_compressed(work / 'initial_coordinates.npz', pos=initial, names=np.asarray(names),
                            num_nodes=db.num_nodes, num_movable_nodes=db.num_movable_nodes)
        manifest['initial_coordinates_sha256'] = sha(work / 'initial_coordinates.npz')
        manifest['initialized_movable_count'] = int(db.num_movable_nodes)
        macro_original = placer.op_collections.macro_legalize_op
        manifest['macro_legalizer_observation_scope'] = 'First-stage macro_legalize_op only. Stock final legalize_op also processes macros; final DEF is audited separately.'

        def coordinate_map(pos, selected=None):
            values = pos.detach().cpu().numpy()
            return {name: [float(values[index]) / params.scale_factor + params.shift_factor[0],
                           float(values[db.num_nodes + index]) / params.scale_factor + params.shift_factor[1]]
                    for index, name in enumerate(names[:db.num_movable_nodes])
                    if selected is None or name in selected}

        def observe_macro_legalizer(pos):
            number = len(manifest['macro_legalizer_calls'])
            stage = work / ('macro_legalizer_' + str(number))
            stage.mkdir()
            np.savez_compressed(stage / 'before.npz', pos=pos.detach().cpu().numpy())
            start = time.time()
            result = macro_original(pos)
            np.savez_compressed(stage / 'after.npz', pos=result.detach().cpu().numpy())
            (stage / 'after.def').write_text(positioned_def(derived_text, coordinate_map(result)))
            write_json(stage / 'macro_coordinates_dbu.json', coordinate_map(result, expected_macros))
            manifest['macro_legalizer_calls'].append({'index': number, 'wall_seconds': time.time() - start,
                                                     'after_def': str(stage / 'after.def'),
                                                     'after_def_sha256': sha(stage / 'after.def'),
                                                     'def_snapshot_note': 'Diagnostic only: coordinates rounded to nearest integer DBU; exact optimizer vectors in NPZ. Not used for PPA.',
                                                     'implementation': 'Original DREAMPlace macro_legalize_op; observation wrapper only.'})
            save()
            return result

        placer.op_collections.macro_legalize_op = observe_macro_legalizer
        if args.gpu:
            torch.cuda.synchronize()
        start = time.time()
        metrics = placer(params, db)
        if args.gpu:
            torch.cuda.synchronize()
        manifest['placement_wall_seconds'] = time.time() - start
        if not manifest['macro_legalizer_calls']:
            raise RuntimeError('No original macro legalization stage executed.')
        output = work / (args.design + '.dreamplace.def')
        finite_arrays = {'final_coordinates': placer.pos[0],
                         'node_size_x': placer.data_collections.node_size_x,
                         'node_size_y': placer.data_collections.node_size_y,
                         'pin_offset_x': placer.data_collections.pin_offset_x,
                         'pin_offset_y': placer.data_collections.pin_offset_y}
        manifest['finite_checks'] = {name: bool(torch.isfinite(values).all().item()) for name, values in finite_arrays.items()}
        if not all(manifest['finite_checks'].values()):
            raise RuntimeError('Nonfinite final coordinates, dimensions or pin offsets.')
        db.write(params, str(output))
        final_sizes_x = placer.data_collections.node_size_x.detach().cpu().numpy()
        final_sizes_y = placer.data_collections.node_size_y.detach().cpu().numpy()
        size_check = {'max_width_difference_internal': float(np.max(np.abs(final_sizes_x - original_sizes_x))),
                      'max_height_difference_internal': float(np.max(np.abs(final_sizes_y - original_sizes_y))),
                      'absolute_tolerance_internal': 1e-5, 'relative_tolerance': 1e-6,
                      'restored': bool(np.allclose(final_sizes_x, original_sizes_x, atol=1e-5, rtol=1e-6)
                                       and np.allclose(final_sizes_y, original_sizes_y, atol=1e-5, rtol=1e-6))}
        write_json(work / 'halo_size_restoration_audit.json', size_check)
        if not size_check['restored']:
            raise RuntimeError('Original macro halos were not removed from the final node dimensions.')
        write_json(work / 'effective_params_after_run.json', params.toJson())
        (work / 'metrics_repr.txt').write_text(repr(metrics) + '\n')
        blocks = {}
        for lef in cfg['lef_input']:
            # Technology-only LEFs have no cell definitions.
            if 'MACRO ' in Path(lef).read_text():
                blocks.update(lef_blocks(lef))
        # inspect_def uses actual names, which can contain DEF escaping.
        _, _, _, entries = split_def(output.read_text())
        actual_macro_names = {entry[1] for entry in entries if canonical(entry[1]) in expected_macros}
        geometry = inspect_def(output, blocks, actual_macro_names)
        # Independent core limits come from unchanged DEF rows, not the outer
        # die or a new optimization constraint.
        rows = [tuple(map(int, match)) for match in re.findall(
            r'^ROW\s+\S+\s+\S+\s+(-?\d+)\s+(-?\d+)\s+\S+\s+DO\s+(\d+)\s+BY\s+1\s+STEP\s+(\d+)\s+0\s*;', source_text, re.M)]
        if not rows:
            raise RuntimeError('No independently auditable single-row DEF rows.')
        ys = sorted({row[1] for row in rows})
        height = min(b - a for a, b in zip(ys, ys[1:]))
        units = geometry['dbu_per_um']
        core = [min(row[0] for row in rows) / units, min(ys) / units,
                max(row[0] + row[2] * row[3] for row in rows) / units,
                (max(ys) + height) / units]
        outside_core = [name for name, macro in geometry['macros'].items()
                        if macro['bbox_um'][0] < core[0] - 1e-8 or macro['bbox_um'][1] < core[1] - 1e-8
                        or macro['bbox_um'][2] > core[2] + 1e-8 or macro['bbox_um'][3] > core[3] + 1e-8]
        geometry.update(core_row_bbox_um=core, out_of_core=outside_core)
        write_json(work / 'macro_geometry.json', geometry)
        if geometry['component_count'] != len(inv['components']) or geometry['macro_count'] != len(expected_macros):
            raise RuntimeError('Final DEF inventory changed.')
        final_inv = inventories(output.read_text())
        if final_inv['components'] != inv['components'] or final_inv['net_count'] != inv['net_count']:
            raise RuntimeError('Final DEF component or net inventory changed.')
        final_connections = def_connections(output.read_text())
        connectivity_check['final_def_connection_sha256'] = connection_digest(final_connections)
        connectivity_check['final_def_connections_preserved'] = final_connections == expected_connections
        write_json(work / 'connectivity_audit.json', connectivity_check)
        if not connectivity_check['final_def_connections_preserved']:
            raise RuntimeError('Final DEF connectivity changed.')
        original_io, output_io = fixed_io_positions(source_text), fixed_io_positions(output.read_text())
        io_check = {'original_pin_count': len(original_io), 'final_pin_count': len(output_io),
                    'pin_names_equal': set(original_io) == set(output_io), 'max_coordinate_difference_dbu': 0.0}
        if not io_check['pin_names_equal']:
            raise RuntimeError('Fixed IO pin inventory changed.')
        for name in original_io:
            before, after = original_io[name], output_io[name]
            if before['status'] != after['status'] or before['orientation'] != after['orientation']:
                raise RuntimeError('Fixed IO status/orientation changed: ' + name)
            io_check['max_coordinate_difference_dbu'] = max(io_check['max_coordinate_difference_dbu'],
                *(abs(a - b) for a, b in zip(before['xy_dbu'], after['xy_dbu'])))
        write_json(work / 'fixed_io_audit.json', io_check)
        if io_check['max_coordinate_difference_dbu'] > 1e-6:
            raise RuntimeError('Fixed IO position changed.')
        for entry in entries:
            name = canonical(entry[1])
            if name in neutral['fixed']:
                before = neutral['fixed'][name]
                position = POSITION.search(entry[0])
                if not position or position[1] != before['status'] or position[4] != before['orientation']:
                    raise RuntimeError('Fixed nonmacro instance status/orientation changed: ' + name)
                if max(abs(float(position[2]) - before['xy_dbu'][0]), abs(float(position[3]) - before['xy_dbu'][1])) > 1e-6:
                    raise RuntimeError('Fixed nonmacro instance position changed: ' + name)
        orientation_check = orientation_audit(source_text, neutral, geometry, db.row_height / params.scale_factor)
        write_json(work / 'macro_orientation_audit.json', orientation_check)
        if not orientation_check['all_match_stock_apply']:
            raise RuntimeError('Final macro orientation differs from stock row-alignment rule.')
        if geometry['overlap_count'] or geometry['out_of_die'] or geometry['out_of_core']:
            raise RuntimeError('Original DMP output has illegal macro geometry; no repair is applied.')
        # Stock Python offsets are always normalized to N by convertOrient(),
        # even when node_orient says FS. The fresh raw C++ database retains actual
        # oriented offsets from LEF+DEF. Audit these separately; run no optimizer.
        reload_cfg = dict(cfg, def_input=str(output), macro_place_flag=0, enable_fillers=0,
                          global_place_flag=0, legalize_flag=0, random_center_init_flag=0)
        write_json(work / 'pin_reload_config.json', reload_cfg)
        reload_params = Params.Params()
        reload_params.load(str(work / 'pin_reload_config.json'))
        reloaded_db = PlaceDB.PlaceDB()
        reloaded_db(reload_params)
        if reload_params.scale_factor != params.scale_factor:
            raise RuntimeError('Pin reload changed coordinate scale.')
        reloaded_pins = macro_pin_offsets(reloaded_db, expected_macros)
        reloaded_raw_pins, reloaded_raw_geometry = raw_macro_pin_offsets(reloaded_db, expected_macros)
        if not (set(initial_macro_pins) == set(reloaded_pins) == set(initial_raw_pins) == set(reloaded_raw_pins)):
            raise RuntimeError('Macro pin identities changed on final DEF reload.')
        pin_check = {'final_def_sha256': sha(output), 'read_only_reload': True,
                     'python_offsets_normalized_to_N': True,
                     'scope': 'Python N-normalized offsets must stay unchanged. Fresh raw C++ physical offsets must follow N/FS (x,height-y); raw coordinates/orientation match final DEF.',
                     'python_absolute_tolerance_internal': 1e-4, 'python_relative_tolerance': 1e-6,
                     'raw_absolute_tolerance_dbu': 0,
                     'pin_connection_count': sum(len(values) for values in reloaded_pins.values()),
                     'max_python_offset_difference_internal': 0.0, 'max_raw_offset_difference_dbu': 0,
                     'all_match': True, 'macros': {}}
        final_geometry_by_name = {canonical(name): macro for name, macro in geometry['macros'].items()}
        for name, raw_geometry in reloaded_raw_geometry.items():
            final_macro = final_geometry_by_name[name]
            if raw_geometry['xy_dbu'] != final_macro['xy_dbu'] or raw_geometry['orientation'] != final_macro['orientation']:
                raise RuntimeError('Raw final macro coordinate/orientation disagrees with final DEF: ' + name)
            if any(raw_geometry[key] != initial_raw_geometry[name][key] for key in ('width_dbu', 'height_dbu')):
                raise RuntimeError('Raw macro dimensions changed on final DEF reload: ' + name)
        for key, before in initial_macro_pins.items():
            name = key[0]
            changed = orientation_check['macros'][name]['changed']
            actual = reloaded_pins[key]
            expected_raw = sorted((x, initial_raw_geometry[name]['height_dbu'] - y if changed else y)
                                  for x, y in initial_raw_pins[key])
            actual_raw = reloaded_raw_pins[key]
            if not (len(before) == len(actual) == len(expected_raw) == len(actual_raw)):
                raise RuntimeError('Macro pin multiplicity changed: ' + repr(key))
            difference = float(np.max(np.abs(np.asarray(before) - np.asarray(actual))))
            raw_difference = int(np.max(np.abs(np.asarray(expected_raw) - np.asarray(actual_raw))))
            passed = bool(np.isfinite(actual).all() and np.allclose(before, actual, atol=1e-4, rtol=1e-6)
                          and raw_difference == 0)
            per_macro = pin_check['macros'].setdefault(name, {'pin_connections': 0, 'max_python_offset_difference_internal': 0.0,
                                                            'max_raw_offset_difference_dbu': 0, 'all_match': True})
            per_macro['pin_connections'] += len(actual)
            per_macro['max_python_offset_difference_internal'] = max(per_macro['max_python_offset_difference_internal'], difference)
            per_macro['max_raw_offset_difference_dbu'] = max(per_macro['max_raw_offset_difference_dbu'], raw_difference)
            per_macro['all_match'] &= passed
            pin_check['all_match'] &= passed
            pin_check['max_python_offset_difference_internal'] = max(pin_check['max_python_offset_difference_internal'], difference)
            pin_check['max_raw_offset_difference_dbu'] = max(pin_check['max_raw_offset_difference_dbu'], raw_difference)
        write_json(work / 'final_orientation_pin_audit.json', pin_check)
        if not pin_check['all_match']:
            raise RuntimeError('Final DEF reload pin offsets disagree with the stock orientation transform.')
        coordinates = {canonical(name): macro['xy_dbu'] for name, macro in geometry['macros'].items()}
        handoff = work / (args.design + '.handoff.def')
        handoff.write_text(positioned_def(output.read_text(), coordinates, fix_macros=True))
        handoff_geometry = inspect_def(handoff, blocks, actual_macro_names)
        write_json(work / 'handoff_macro_geometry.json', handoff_geometry)
        write_json(work / 'macro_coordinates_dbu.json', coordinates)
        if handoff_geometry['nonfixed_macros'] or handoff_geometry['overlap_count'] or handoff_geometry['out_of_die']:
            raise RuntimeError('Macro handoff validation failed.')
        if handoff_geometry['macros'] != {name: dict(macro, status='FIXED') for name, macro in geometry['macros'].items()}:
            raise RuntimeError('Handoff changed more than macro status.')
        write_json(work / 'handoff_audit.json', {'design': args.design, 'output_sha256': sha(handoff),
                   'legal_no_halo': True, 'macro_count': len(coordinates),
                   'scope': 'Macro rectangles only, no overlap/die/core violations; not routed DRC.',
                   'source_geometry_audit_path': str(work / 'macro_geometry.json'),
                   'source_geometry_audit_sha256': sha(work / 'macro_geometry.json'),
                   'orientation_audit_sha256': sha(work / 'macro_orientation_audit.json'),
                   'pin_reload_audit_sha256': sha(work / 'final_orientation_pin_audit.json')})
        manifest.update(success=True, output_def=str(output), output_sha256=sha(output),
                        handoff_def=str(handoff), handoff_sha256=sha(handoff),
                        macro_count=len(coordinates), final_macro_legal=True,
                        handoff_audit=str(work / 'handoff_audit.json'), handoff_audit_sha256=sha(work / 'handoff_audit.json'),
                        final_orientation_audited=True, final_pin_offsets_reloaded=True)
        manifest['limitations'] = ['Original replace.def and standalone author macro-baseline settings unavailable.',
                                  ('GPU hardware/software may differ from the paper; no hardware-matched speedup claim.' if args.gpu
                                   else 'CPU implementation; numerical output and runtime cannot be assumed equivalent to GPU.'),
                                  'Original public 4.1 Tetris/min-cost-flow legalizers; exact author baseline settings are unpublished.',
                                  'Macro rectangle checks are not detailed-route DRC or PPA.',
                                  'Published custom config keys absent from standard 4.1 are not applied.']
        if 'operator_patch' in manifest:
            manifest['limitations'].append('A disclosed one-line Abacus float row-boundary correction is applied uniformly; this is not the unmodified public binary.')
    except BaseException as exc:
        manifest['failure'] = repr(exc)
        manifest['traceback'] = traceback.format_exc()
        raise
    finally:
        manifest['time_end'] = time.time()
        save()
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
