#!/usr/bin/env python3
"""Read-only native OpenROAD comparison of released DEF and cached synthesis.

This checks structural pin connectivity, not functional/formal equivalence.
No placement, repair, synthesis, or metric extraction is performed.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/fact_data/bohanli/flowplace_fql_backend')
TOPS = {'bp_fe': 'bp_fe_top', 'bp': 'black_parrot', 'bp_be': 'bp_be_top',
        'ariane133': 'ariane', 'ariane136': 'ariane', 'swerv_wrapper': 'swerv_wrapper'}

EXPORT = r'''
proc safe_name {name} {
  if {[regexp {[\t\r\n]} $name]} {error "Unsupported tab/newline in database name"}
  return $name
}
set block [ord::get_db_block]
set out [open $::env(CONNECTIVITY_EXPORT) w]
foreach inst [$block getInsts] {
  puts $out "INSTANCE\t[safe_name [$inst getName]]\t[safe_name [[$inst getMaster] getName]]"
}
foreach net [$block getNets] {
  puts $out "NET\t[safe_name [$net getName]]\t[$net getSigType]"
  foreach iterm [$net getITerms] {
    puts $out "IT\t[safe_name [[$iterm getInst] getName]]\t[safe_name [[$iterm getMTerm] getName]]"
  }
  foreach bterm [$net getBTerms] {
    puts $out "BT\t[safe_name [$bterm getName]]"
  }
}
close $out
'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def tcl_path(path):
    value = str(path)
    if any(c in value for c in '{}\n\r'):
        raise ValueError('Unsupported Tcl path')
    return '{' + value + '}'


def read_graph(path):
    instances, nets, examples = {}, Counter(), {}
    ignored = Counter()
    net = None
    endpoints = []
    total_endpoints = 0

    def flush():
        nonlocal total_endpoints
        if net is None:
            return
        if net[2] in ('POWER', 'GROUND'):
            ignored[net[2]] += 1
            return
        if not endpoints:
            ignored['empty_signal_net'] += 1
            return
        key = hashlib.sha256(json.dumps(sorted(endpoints), separators=(',', ':')).encode()).hexdigest()
        nets[key] += 1
        total_endpoints += len(endpoints)
        if key not in examples:
            examples[key] = {'name': net[1], 'endpoint_count': len(endpoints),
                             'first_endpoints': sorted(endpoints)[:12]}

    with path.open() as stream:
        for line in stream:
            row = line.rstrip('\n').split('\t')
            if row[0] == 'INSTANCE':
                assert len(row) == 3 and row[1] not in instances
                instances[row[1]] = row[2]
            elif row[0] == 'NET':
                flush()
                assert len(row) == 3
                net, endpoints = row, []
            elif row[0] == 'IT':
                assert len(row) == 3 and net is not None
                endpoints.append(tuple(row))
            elif row[0] == 'BT':
                assert len(row) == 2 and net is not None
                endpoints.append(tuple(row))
            else:
                raise ValueError('Unrecognized graph record')
    flush()
    return instances, nets, examples, {'instances': len(instances), 'nonempty_signal_nets': sum(nets.values()),
                                      'signal_endpoints': total_endpoints, 'excluded_net_counts': dict(ignored)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True, choices=TOPS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--openroad', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    output = args.output.resolve()
    assert output.is_relative_to(DATA / 'table2_reports')
    output.mkdir(parents=True, exist_ok=False)
    scratch = Path(os.environ['TMPDIR']) / args.design
    scratch.mkdir(parents=True, exist_ok=False)
    source = DATA / 'dreamplace_table2' / ('dmp_public41_h20_abacusfix_' + args.design + '_v1') / 'manifest.json'
    frozen = json.loads(source.read_text())
    hashes = frozen['input_sha256']
    paths = [Path(p) for p in hashes]
    libraries = [p for p in paths if p.suffix == '.lib']
    lefs = [p for p in paths if p.suffix == '.lef']
    lefs.sort(key=lambda p: (not p.name.endswith('.tech.lef'), p.name))
    verilog, = [p for p in paths if p.name == '1_synth.v']
    original, = [p for p in paths if p == ROOT / 'backend/releases/flowplace_DEFS/openroad' / (args.design + '.def')]
    checked = {str(p): digest(p) for p in [*libraries, *lefs, verilog, original]}
    assert all(hashes[p] == value for p, value in checked.items())
    expected_tool = '93fe61454a52b927602fd6c798c4402214c19e80e7d614636345f0f427ea2f46'
    assert digest(args.openroad) == expected_tool
    report = {'design': args.design, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'started_unix': time.time(), 'input_sha256': checked, 'runner_sha256': digest(__file__),
              'generation_manifest': str(source), 'generation_manifest_sha256': digest(source),
              'openroad_sha256': expected_tool, 'success': False,
              'scope': 'Exact leaf-instance/master and signal-net endpoint partition comparison; net names ignored.',
              'limitations': ['Not a functional/formal equivalence proof.',
                              'Power/ground and signal nets with no connected endpoints excluded and counted.',
                              'No geometry, placement, optimization, or PPA is performed.']}
    graphs = {}
    for mode in ('def', 'verilog'):
        commands = ['set_thread_count 1'] + [f'read_lef {tcl_path(p)}' for p in lefs]
        if mode == 'def':
            commands += [f'read_def {tcl_path(original)}']
        else:
            commands += [f'read_liberty {tcl_path(p)}' for p in libraries]
            commands += [f'read_verilog {tcl_path(verilog)}', f'link_design {TOPS[args.design]}']
        script = output / (mode + '_export.tcl')
        script.write_text('\n'.join(commands) + '\n' + EXPORT)
        exported = scratch / (mode + '.tsv')
        environment = dict(os.environ, CONNECTIVITY_EXPORT=str(exported))
        with (output / (mode + '.log')).open('w') as log:
            result = subprocess.run([str(args.openroad), '-exit', str(script)], env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=600)
        assert result.returncode == 0, mode + ' export failed; inspect log'
        graphs[mode] = read_graph(exported)
        report[mode] = dict(graphs[mode][3], export_sha256=digest(exported),
                            log_sha256=digest(output / (mode + '.log')), tcl_sha256=digest(script))
    di, dn, de, _ = graphs['def']
    vi, vn, ve, _ = graphs['verilog']
    left, right = dn - vn, vn - dn
    report.update(instance_master_equal=di == vi, net_endpoint_partitions_equal=dn == vn,
                  def_only_instance_count=len(di.keys() - vi.keys()), verilog_only_instance_count=len(vi.keys() - di.keys()),
                  changed_master_count=sum(di[k] != vi[k] for k in di.keys() & vi.keys()),
                  def_only_net_partition_count=sum(left.values()), verilog_only_net_partition_count=sum(right.values()),
                  def_only_examples=[dict(de[k], multiplicity=n) for k, n in list(left.items())[:10]],
                  verilog_only_examples=[dict(ve[k], multiplicity=n) for k, n in list(right.items())[:10]])
    assert all(digest(p) == value for p, value in checked.items()), 'Input changed during audit'
    report.update(success=di == vi and dn == vn, ended_unix=time.time())
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('design', 'success', 'instance_master_equal',
                                            'net_endpoint_partitions_equal', 'def_only_net_partition_count',
                                            'verilog_only_net_partition_count')}), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
