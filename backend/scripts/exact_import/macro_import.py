"""Versioned macro-import correction; restores generator coordinates, no optimization."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

POLICY = 'preserve_generator_dbu_coordinates_v1'
PARSER_SHA = '521fe0f12165be6590ca407b4bc52509cf6e014ce77a496e7011c664f3cc37e4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def brace(text):
    assert not any(c in text for c in '{}\\\r\n'), 'Unsupported instance name in exact-import correction'
    return '{' + text + '}'


def prepare(work, manifest):
    assert manifest['task']['macro_import_policy'] == POLICY
    provenance = work / 'provenance'
    parser = provenance / 'audit_table2_macro_preservation.py'
    assert sha(parser) == PARSER_SHA
    spec = importlib.util.spec_from_file_location('exact_import_macro_parser', parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = Path(manifest['executed_def_path'])
    assert sha(source) == manifest['task']['def_sha256']
    units, macros = module.macros(source)
    assert module.geometry_check(work, manifest, units, macros)['legal_no_halo']
    paths = list((work / 'orfs_work/results').glob('*/*/eval/macro_out'))
    assert len(paths) == 1, 'Expected exactly one author-generated macro placement Tcl'
    path = paths[0]
    original = path.read_bytes()
    commands = [line for line in original.decode().splitlines() if line.strip()]
    assert len(commands) == len(macros)
    assert all(line.startswith('place_macro -macro_name ') and ' -location {' in line
               and (' -orientation R0' in line or ' -orientation MX' in line)
               and '-exact' not in line for line in commands)
    before = provenance / 'macro_import_original.tcl'
    after = provenance / 'macro_import_exact.tcl'
    assert not before.exists() and not after.exists()
    before.write_bytes(original)
    lines = ['', '# Exact DBU-coordinate restoration after unchanged native author macro commands.']
    for name, macro in macros.items():
        orient = {'N': 'R0', 'FS': 'MX'}[macro['orientation']]
        lines += [
            'set table2_exact_inst [[ord::get_db_block] findInst ' + brace(name) + ']',
            'if {$table2_exact_inst eq "NULL"} {error "Missing exact-import macro"}',
            'if {[$table2_exact_inst getPlacementStatus] ne "LOCKED"} {error "Native macro status changed"}',
            'if {[$table2_exact_inst getOrient] ne "' + orient + '"} {error "Native macro orientation changed"}',
            '$table2_exact_inst setPlacementStatus PLACED',
            '$table2_exact_inst setLocation ' + ' '.join(str(v) for v in macro['xy_dbu']),
            '$table2_exact_inst setPlacementStatus LOCKED',
            'set table2_exact_bbox [$table2_exact_inst getBBox]',
            'puts "TABLE2_ODB_EXACT\\t[$table2_exact_inst getName]\\t[[$table2_exact_inst getMaster] getName]\\t[$table2_exact_inst getOrient]\\t[$table2_exact_inst getPlacementStatus]\\t[$table2_exact_bbox xMin]\\t[$table2_exact_bbox yMin]\\t[$table2_exact_bbox xMax]\\t[$table2_exact_bbox yMax]"',
        ]
    exact = original + ('\n'.join(lines) + '\n').encode()
    after.write_bytes(exact)
    temporary = path.with_name('macro_out.exact_tmp')
    temporary.write_bytes(exact)
    temporary.replace(path)
    assert path.read_bytes() == after.read_bytes() == exact
    return {'policy': POLICY, 'macro_count': len(macros), 'dbu_per_um': units,
            'input_def_sha256': sha(source), 'original_tcl': str(before), 'original_tcl_sha256': sha(before),
            'executed_tcl': str(after), 'executed_tcl_sha256': sha(after),
            'generated_tcl_path': str(path), 'helper_sha256': sha(__file__),
            'parser_sha256': sha(parser), 'scope': 'Keep all author place_macro commands, then restore each original input lower-left in integer DBU and restore native LOCKED status. No coordinate search, repair objective, seed or PPA selection. Both methods use the same policy.'}
