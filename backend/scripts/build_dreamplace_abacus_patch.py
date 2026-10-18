#!/usr/bin/env python3
"""Build one isolated Abacus floorDiv correction; preserve the audited parent."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path('/fact_data/bohanli/flowplace_fql_backend/dreamplace_gpu_build/job_1787994')
OLD_SCRATCH = Path('/scratch/bohanli/flowplace_table2/dmp_gpu_build_1787994')
SNAPSHOT = Path('/fact_data/bohanli/flowplace_fql_backend/dreamplace_table2/dmp_public41_h20_ariane133_abacus_observe_v1/abacus_input_0')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def hashes(install):
    return {str(path.relative_to(install)): sha(path) for path in sorted(install.rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.so', '.json')}


def main():
    job = os.environ['SLURM_JOB_ID']
    run = PARENT.parent / ('abacus_floor_div_job_' + job)
    scratch = Path('/scratch/bohanli/flowplace_table2') / ('abacus_floor_div_' + job)
    run.mkdir(exist_ok=False)
    scratch.mkdir(exist_ok=False)
    status = {'parent_build': str(PARENT), 'slurm_job_id': job, 'success': False,
              'scope': 'Single Abacus multi-row lower-bin boundary changes raw truncation to existing floorDiv; all other operators reused byte-for-byte.'}
    write(run / 'operator_patch_validation.json', status)
    try:
        expected = json.loads((PARENT / 'installation_file_hashes.json').read_text())
        assert hashes(PARENT / 'install') == expected, 'Parent installation no longer matches validated inventory.'
        shutil.copytree(PARENT / 'install', run / 'install')
        shutil.copytree(PARENT / 'retries', run / 'retries')
        (run / 'env').symlink_to(PARENT / 'env', target_is_directory=True)
        # Preserve existing validation as inherited evidence, not newly rerun tests.
        for path in PARENT.iterdir():
            if path.is_file() and path.name not in ('BUILD_AND_OPERATOR_TESTS_PASSED', 'installation_file_hashes.json'):
                shutil.copyfile(path, run / path.name)
        shutil.copyfile(PARENT / 'installation_file_hashes.json', run / 'parent_installation_file_hashes.json')
        shutil.copyfile(PARENT / 'build_completion.json', run / 'parent_build_completion.json')
        completion = json.loads((run / 'build_completion.json').read_text())
        completion['retry_dir'] = str(run / Path(completion['retry_dir']).relative_to(PARENT))
        completion['inherited_parent_build'] = str(PARENT)
        write(run / 'build_completion.json', completion)
        shutil.copyfile(__file__, run / Path(__file__).name)
        shutil.copyfile(ROOT / 'backend/scripts/build_dreamplace_abacus_patch.slurm', run / 'operator_patch_submitted.slurm')
        validator = ROOT / 'backend/scripts/validate_dreamplace_abacus_patch.py'
        shutil.copyfile(validator, run / validator.name)
        operation = OLD_SCRATCH / 'source/dreamplace/ops/abacus_legalize'
        overlay = scratch / 'source/dreamplace/ops/abacus_legalize'
        shutil.copytree(operation, overlay)
        header = overlay / 'src/abacus_legalize_cpu.h'
        original = header.read_text()
        old = 'int bin_id_yl = std::max((y[node_id] - yl) / bin_size_y, (T)0);'
        new = 'int bin_id_yl = std::max(floorDiv(y[node_id] - yl, bin_size_y), (T)0);'
        assert original.count(old) == 1
        changed = original.replace(old, new)
        header.write_text(changed)
        patch = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                         fromfile='a/dreamplace/ops/abacus_legalize/src/abacus_legalize_cpu.h',
                         tofile='b/dreamplace/ops/abacus_legalize/src/abacus_legalize_cpu.h'))
        (run / 'abacus_floor_div.patch').write_text(patch)
        shutil.copyfile(header, run / header.name)
        cmake = OLD_SCRATCH / 'build/dreamplace/ops/abacus_legalize/CMakeFiles/abacus_legalize_cpp.dir'
        flags_path = cmake / 'flags.make'
        flags = dict(line.split(' = ', 1) for line in flags_path.read_text().splitlines() if ' = ' in line)
        obj = scratch / 'abacus_legalize.cpp.o'
        compile_command = ['/usr/bin/c++', '-I' + str(overlay.parent)]
        for key in ['CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS']:
            compile_command.extend(shlex.split(flags[key]))
        compile_command += ['-c', str(overlay / 'src/abacus_legalize.cpp'), '-o', str(obj)]
        link_path = cmake / 'link.txt'
        link_command = shlex.split(link_path.read_text())
        binary_name = 'abacus_legalize_cpp.cpython-39-x86_64-linux-gnu.so'
        binary = run / 'install/dreamplace/ops/abacus_legalize' / binary_name
        replacements = {'CMakeFiles/abacus_legalize_cpp.dir/src/abacus_legalize.cpp.o': str(obj),
                        binary_name: str(binary), '../utility/libutility_cxx.a': str(OLD_SCRATCH / 'build/dreamplace/ops/utility/libutility_cxx.a')}
        link_command = [replacements.get(token, token) for token in link_command]
        write(run / 'compile_commands.json', {'compile': compile_command, 'link': link_command,
              'parent_flags_sha256': sha(flags_path), 'parent_link_sha256': sha(link_path),
              'unchanged_utility_archive_sha256': sha(replacements['../utility/libutility_cxx.a'])})
        with (run / 'operator_compile.log').open('w') as log:
            subprocess.run(compile_command, check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(link_command, check=True, stdout=log, stderr=subprocess.STDOUT)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(run / 'install') + ':' + env['PYTHONPATH']
        python = PARENT / 'env/bin/python'
        # Small original fixture must reproduce failure, but does not modify
        # the parent installation. Full original failure was already retained.
        control = [str(python), str(validator), '--snapshot', str(SNAPSHOT), '--output', str(run / 'original_boundary_control'), '--boundary-only']
        with (run / 'original_boundary_control.log').open('w') as log:
            control_code = subprocess.run(control, env=os.environ, stdout=log, stderr=subprocess.STDOUT).returncode
        assert control_code == -6, 'Original isolated boundary did not reproduce SIGABRT: ' + str(control_code)
        command = [str(python), str(validator), '--snapshot', str(SNAPSHOT), '--output', str(run / 'patched_validation')]
        with (run / 'patched_validation.log').open('w') as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        validation_path = run / 'patched_validation/validation.json'
        validation = json.loads(validation_path.read_text())
        assert validation['snapshot_replay_passed'] and validation['after_legal']
        actual = hashes(run / 'install')
        changed_files = [name for name in expected if actual.get(name) != expected[name]]
        binary_relative = str(binary.relative_to(run / 'install'))
        assert set(actual) == set(expected) and changed_files == [binary_relative], changed_files
        assert hashes(PARENT / 'install') == expected, 'Parent installation unexpectedly changed.'
        write(run / 'installation_file_hashes.json', actual)
        status.update(success=True, original_boundary_control_exit_code=control_code,
                      parent_installation_manifest_sha256=sha(run / 'parent_installation_file_hashes.json'),
                      patch_sha256=sha(run / 'abacus_floor_div.patch'), source_header_sha256=sha(header),
                      original_header_sha256=hashlib.sha256(original.encode()).hexdigest(),
                      changed_installed_files=changed_files, original_binary_sha256=expected[binary_relative],
                      patched_binary_sha256=actual[binary_relative], validator_sha256=sha(validator),
                      validation_report_sha256=sha(validation_path), validation=validation,
                      snapshot_path=str(SNAPSHOT), snapshot_sha256=sha(SNAPSHOT / 'inputs.npz'),
                      inherited_parent_completion_sha256=sha(run / 'parent_build_completion.json'),
                      compile_commands_sha256=sha(run / 'compile_commands.json'))
        write(run / 'operator_patch_validation.json', status)
        (run / 'BUILD_AND_OPERATOR_TESTS_PASSED').write_text('Parent four operator tests inherited byte-for-byte; isolated Abacus patch boundary/full snapshot validation passed.\n')
        print('PATCH_BUILD=' + str(run), flush=True)
    except BaseException as exc:
        status['failure'] = repr(exc)
        write(run / 'operator_patch_validation.json', status)
        raise


if __name__ == '__main__':
    main()
