"""Recheck existing outputs after fixing serialization-sensitive comparison.

Retain the original failed process manifest. No placement operation is executed.
"""
import json
import os
from pathlib import Path
import sys
from audit_inputs import compare_fixed, inspect_def, lef_blocks, sha

assert os.environ.get('SLURM_JOB_ID')
run=Path(sys.argv[1])
meta=json.loads((run/'manifest.json').read_text())
assert meta['returncode']==0
cfg=json.loads((run/'config.json').read_text())
assert sha(cfg['def_input'])==meta['input_sha256']['def_input']
assert sha(cfg['lef_input'])==meta['input_sha256']['lef_input']
output=Path(cfg['result_dir'])/meta['design']/(meta['design']+'.gp.def')
cells=lef_blocks(cfg['lef_input'])
before=inspect_def(cfg['def_input'],cells)
after=inspect_def(output,cells,set(before['macros']))
fixed=compare_fixed(before,after)
assert before['component_identity_multiset_digest']==after['component_identity_multiset_digest']
assert not after['overlap_count'] and not after['out_of_die']
out=dict(slurm_job=os.environ['SLURM_JOB_ID'],script_sha256=sha(__file__),
    audit_script_sha256=sha(Path(__file__).with_name('audit_inputs.py')),
    original_manifest_sha256=sha(run/'manifest.json'),
    original_manifest_complete=meta['complete'],verified=True,
    input_sha256=before['sha256'],output_def=str(output.resolve()),output_sha256=after['sha256'],
    fixed_macro_check=fixed,component_inventory_preserved=True,
    scope='Existing DEF export and fixed-instance fidelity only; all-cell legality evaluated separately')
(run/'output_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
