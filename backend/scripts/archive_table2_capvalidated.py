#!/usr/bin/env python3
"""Archive physical-complete runs whose original audit failed only on a proven DRC export cap.
Original FAILED manifest/audit/scheduler states remain unchanged; archive status only describes preservation.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import archive_table2_staged as base
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
import table2_drc_cap as cap

BASE_SHA='7fe87e6e021d8da8f4ad680515640d0af716f731d7fce18650dbf4a63f50fcbf'
CAP_SHA='a0b377454028e6b369e7372220fde02dd0adfbd03ba487fbc1a34ce9ce8707e6'

def eligible(run):
    run=base.direct_child(run,base.RUN_ROOT)
    manifest=json.loads((run/'manifest.json').read_text());audit=json.loads((run/'fullflow_audit.json').read_text())
    base.require(manifest.get('run_id')==run.name and cap.is_cap_audit_failure(manifest,audit),'Unsupported failure, not an eligible posthoc-cap-validated physical run')
    _,_,verified=cap.verify_metrics(run,manifest)
    array,task=manifest.get('slurm_array_job_id'),manifest.get('slurm_array_task_id')
    job=str(array)+'_'+str(task) if array is not None and task is not None else str(manifest.get('slurm_job_id',''))
    base.require(re.fullmatch(r'[0-9]+(?:_[0-9]+)?',job),'Missing exact source job')
    raw=subprocess.check_output(['sacct','-n','-P','-j',job,'-o','JobID%64,State%30,ExitCode'],text=True)
    rows=[r.split('|') for r in raw.splitlines() if r.split('|')[0]==job]
    base.require(len(rows)==1 and rows[0][1:3]==['FAILED','1:0'],'Original source failure evidence differs')
    return manifest,verified['posthoc_drc_cap_evidence']

def main():
    base.require(os.environ.get('SLURM_JOB_ID'),'Use Slurm')
    base.require(base.sha(Path(base.__file__))==BASE_SHA,'Frozen base archive helper changed')
    base.require(base.sha(Path(cap.__file__))==CAP_SHA,'Frozen cap verifier changed')
    # The proven physical-completion classification replaces eligibility only.
    # Compression, complete decode/hash verification, durable flush and commit remain frozen.
    base.completed_run=lambda run:eligible(run)[0]
    action=sys.argv[1]
    destination=Path(sys.argv[sys.argv.index('--archive-dir')+1])
    if action=='create':
        run=Path(sys.argv[sys.argv.index('--run')+1]);_,proof=eligible(run)
    else:
        record=json.loads((destination/'archive_manifest.json').read_text());run=Path(record['run']);_,proof=eligible(run)
    base.main()
    if action=='create':
        evidence={'classification':'physical_complete_posthoc_cap_validated_original_failed',
                  'original_manifest_status':'failed','original_scheduler_state':'FAILED','original_audit_all_checks_pass':False,
                  'source_run':str(run),'preservation_only_not_a_success_reclassification':True,'proof':proof,
                  'helper_sha256':base.sha(__file__),'base_helper_sha256':BASE_SHA,'cap_verifier_sha256':CAP_SHA}
        base.save(destination/'posthoc_cap_archive_eligibility.json',evidence)
        for path in (Path(__file__),Path(__file__).with_suffix('.slurm'),Path(cap.__file__)):
            import shutil
            shutil.copyfile(path,destination/path.name)
    print(json.dumps({'classification':'physical_complete_posthoc_cap_validated_original_failed','action':action,'original_failure_records_preserved':True}),flush=True)
if __name__=='__main__':main()
