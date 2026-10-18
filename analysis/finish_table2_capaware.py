#!/usr/bin/env python3
"""Complete both frozen cohorts with transparent posthoc DRC-cap evidence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = Path('/fact_data/bohanli/flowplace_fql_backend/table2_reports')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

def wait_job(job,allowed=(('COMPLETED','0:0'),)):
    while True:
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',str(job),'-o','JobID%64,State%30,ExitCode'],text=True)
        rows=[r.split('|') for r in raw.splitlines() if r.split('|')[0]==str(job)]
        if rows:
            state=rows[0][1].split()[0].rstrip('+');code=rows[0][2]
            if state not in ('RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'):
                assert (state,code) in allowed, (job,state,code)
                return state,code
        time.sleep(30)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True);args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID');assert sha(args.config)==args.config_sha256
    cfg=json.loads(args.config.read_text());base=Path(cfg['base']);os.chdir(ROOT)
    journal=base/'capaware_finish_v1.json'
    with (base/'.capaware_finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not journal.exists(),'Inspect prior journal before recovery'
        state={'status':'waiting_for_terminal_cohort','slurm_job_id':os.environ['SLURM_JOB_ID'],'config_sha256':args.config_sha256,'children':{},'created_unix':time.time()};save(journal,state)
        try:
            for name,digest in cfg['source_sha256'].items():assert sha(ROOT/name)==digest, name
            batch=json.loads((base/'state.json').read_text());assert batch['status']=='audit_submitted'
            invocation={k:batch[k] for k in ['plans','infrastructure_retries','source_sha256','additional_accounting_jobs']}
            invocation['source_sha256']=dict(invocation['source_sha256'],**cfg['source_sha256'])
            invocation['output']=str(REPORTS/cfg['prefix'])
            inv=base/'final_capaware_invocation_v1.json';assert not inv.exists();save(inv,invocation)
            def submit(name,argv):
                job=subprocess.check_output(['sbatch','--parsable','--job-name=t2capfinish_'+name]+argv,text=True).strip().split(';')[0]
                state['children'][name]={'job':job,'command':argv};save(journal,state);return job
            audit_job=submit('audit',['analysis/run_table2_capaware_audit.slurm','--invocation',str(inv),'--invocation-sha256',sha(inv)])
            state['status']='auditing_metrics';save(journal,state);wait_job(audit_job)
            summary=Path(invocation['output'])/'report/summary.json';report=json.loads(summary.read_text())
            assert report['validated_count']==12 and report['comparison_ready'] is True and report['same_shared_backend'] is True
            assert report['primary_comparison_ready'] is (cfg['cohort']=='original')
            state['summary']={'path':str(summary),'sha256':sha(summary)};save(journal,state)
            exact=cfg['cohort']=='exact'
            macro_stem='audit_table2_exact_import' if exact else 'audit_table2_macro_preservation'
            plot_stem='plot_table2_exact_import' if exact else 'plot_table2_reproduction'
            writer_stem='write_table2_exact_import_report_capaware' if exact else 'write_table2_original_issue_report_capaware'
            macro=REPORTS/(cfg['prefix']+'_macro');plots=REPORTS/(cfg['prefix']+'_plots');output=REPORTS/(cfg['prefix']+'_report')
            macro_job=submit('macro',['analysis/'+macro_stem+'.slurm',str(summary),sha(summary),str(macro),cfg['source_sha256']['analysis/'+macro_stem+'.py']])
            plot_job=submit('plots',['analysis/'+plot_stem+'.slurm',str(summary),sha(summary),str(plots),cfg['source_sha256']['analysis/'+plot_stem+'.py']])
            state['status']='auditing_macro_geometry';save(journal,state)
            wait_job(macro_job,allowed=(('COMPLETED','0:0'),) if exact else (('FAILED','1:0'),))
            wait_job(plot_job)
            geometry=json.loads((macro/'summary.json').read_text())
            assert geometry['complete'] is True and geometry['checked']==12
            assert geometry['all_checked_final_match_native_macroplacement'] is True
            assert geometry['all_checked_final_macros_legal'] is exact
            argv=['analysis/'+writer_stem+'.slurm',cfg['source_sha256']['analysis/'+writer_stem+'.py']]
            for name,path in [('summary',summary),('macro',macro/'summary.json'),('plots',plots/'manifest.json'),('connectivity',REPORTS/'connectivity_v1/summary.json')]:
                argv+=['--'+name,str(path),'--'+name+'-sha256',sha(path)]
            argv+=['--output',str(output)]
            report_job=submit('report',argv);state['status']='writing_report';save(journal,state);wait_job(report_job)
            state.update(status='completed',output=str(output),completed_unix=time.time());save(journal,state)
        except Exception as exc:
            state.update(status='failed',error=repr(exc),failed_unix=time.time());save(journal,state);raise
if __name__=='__main__':main()
