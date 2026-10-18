#!/usr/bin/env python3
"""Read-only geometry diagnostic at the existing native macro-import boundary."""
import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shutil
import time
from audit_table2_macro_preservation import geometry_check, macros, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    assert args.output.is_relative_to(Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports'))
    args.output.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    state=json.loads((root/'backend/plans/table2_background_v1/state.json').read_text())
    out=[]
    for design in ['bp','ariane133','ariane136']:
        item=state['selected']['DREAMPlace/'+design]
        run=Path('/fact_data/bohanli/flowplace_fql_backend/table2_results')/item['task']['run_id']
        mp=run/'manifest.json';mh=sha(mp);manifest=json.loads(mp.read_text())
        assert manifest['task']==item['task']
        source=Path(manifest['executed_def_path']);assert sha(source)==item['task']['def_sha256']
        units,expected=macros(source)
        log=next((run/'orfs_work/logs').glob('*/*/eval/2_3_floorplan_macro.log'))
        lh=sha(log)
        pattern=r'\[INFO MPL-0035\] Macro (.*?) placed\. Bounding box \(([-\d.]+)um, ([-\d.]+)um\), \(([-\d.]+)um, ([-\d.]+)um\)\. Orientation (\S+)'
        actual={}
        for name,x0,y0,x1,y1,orient in re.findall(pattern,log.read_text()):
            name=re.sub(r'\\(.)',r'\1',name);assert name not in actual and name in expected
            coords=[Decimal(v)*units for v in [x0,y0,x1,y1]];assert all(v==int(v) for v in coords)
            actual[name]=dict(expected[name],xy_dbu=[int(v) for v in coords[:2]],orientation={'R0':'N','MX':'FS'}[orient])
        assert set(actual)==set(expected)
        geometry=geometry_check(run,manifest,units,actual)
        assert sha(mp)==mh and sha(log)==lh
        out.append({'design':design,'run_id':run.name,'manifest_sha256':mh,'native_log':str(log),
            'native_log_sha256':lh,'macro_count':len(actual),'geometry':geometry})
    result={'slurm_job_id':os.environ['SLURM_JOB_ID'],'runner_sha256':sha(__file__),
        'imported_auditor_sha256':sha(root/'analysis/audit_table2_macro_preservation.py'),
        'scope':'Native import boundary only, derived from actual printed bounding boxes and frozen LEF sizes; not final routed geometry or a PPA evaluation.',
        'time':time.time(),'rows':out}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copyfile(__file__,args.output/Path(__file__).name)
    print(json.dumps([{'design':v['design'],'macros':v['macro_count'],
        'overlap_pairs':len(v['geometry']['overlaps']),'out_of_core':len(v['geometry']['out_of_core'])} for v in out]))

if __name__=='__main__':main()
