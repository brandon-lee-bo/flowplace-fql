"""Scheduled normal-license startup diagnostics; no installation changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import socket
import subprocess

job=os.environ['SLURM_JOB_ID']
run=Path('/scratch/bohanli/flowplace_iccad')/('innovus_retry_'+job)
run.mkdir(exist_ok=True)
shutil.copy2(__file__,run/'retry_innovus.py')
tool='/fact_data/softwares/cadence/INNOVUS251/bin/innovus'
lmutil='/fact_data/softwares/cadence/INNOVUS251/tools.lnx86/bin/lmutil'
license_path=Path('/fact_data/softwares/cadence/license/license.dat')
keys=['CDS_LIC_FILE','LM_LICENSE_FILE','CDS_LIC_ONLY','SNPSLMD_LICENSE_FILE']
results={'job':job,'host':socket.gethostname(),'run':str(run),'tool':tool,'cases':[]}

def save():
    (run/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

def command(name,argv,env=None,limit=60,cwd=None):
    folder=run/name;folder.mkdir(exist_ok=True)
    try:
        p=subprocess.run(argv,env=env,cwd=cwd or folder,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,text=True,timeout=limit)
        code,output=p.returncode,p.stdout
    except subprocess.TimeoutExpired as e:
        code=124;output=e.stdout or ''
        if isinstance(output,bytes):output=output.decode(errors='replace')
    (folder/'console.log').write_text(output)
    print(json.dumps({'step':name,'exit_code':code}),flush=True)
    return code,output

command('version',[tool,'-version'])
dump='import os,json;print("ENV_JSON:"+json.dumps({k:os.environ.get(k) for k in '+repr(keys)+'}))'
shell_command='/usr/bin/python3 -c '+"'"+dump.replace("'","'\"'\"'")+"'"
environments={}
for label,shell,args in [('bash_login','/bin/bash',['-lc']),('zsh_login_interactive','/bin/zsh',['-lic'])]:
    code,out=command(label,[shell,*args,shell_command+'; command -v innovus; module -t avail 2>&1'],limit=60)
    lines=[x.partition('ENV_JSON:')[2] for x in out.splitlines() if 'ENV_JSON:' in x]
    environments[label]=json.loads(lines[-1]) if lines else {}
results['login_license_environments']=environments

# Record routing headers / feature metadata only, never signatures or keys.
servers=[];features=[]
for line in license_path.read_text(errors='replace').splitlines():
    words=line.split()
    if words and words[0]=='SERVER' and len(words)>=3:
        servers.append({'host':words[1],'port':words[3] if len(words)>3 else None})
    if len(words)>=5 and words[0] in ('FEATURE','INCREMENT') and any(x in words[1].lower() for x in ('innovus','encounter')):
        features.append({'name':words[1],'vendor':words[2],'version':words[3],'expires':words[4]})
results['shared_license_metadata']={'sha256':hashlib.sha256(license_path.read_bytes()).hexdigest(),
                                   'servers':servers,'innovus_feature_metadata':features}
save()
code,out=command('shared_lmstat',[lmutil,'lmstat','-s','-c',str(license_path)],limit=30)
results['shared_license_status']={'returncode':code,'output':out}
# Query only the endpoint already present in the admin's shared EDA setup;
# it is labelled Synopsys there, not assumed to provide Cadence licenses.
cmd='source /fact_data/softwares/setup_eda.env; '+shlex.quote(lmutil)+' lmstat -s -c "$LM_LICENSE_FILE"'
code,out=command('admin_eda_license_status',['/bin/bash','-lc',cmd],limit=45)
results['admin_eda_license_status']={'returncode':code,'output':out}

tcl='''puts "FLOWPLACE_INNOVUS_TCL_STARTED"
setMultiCpuUsage -localCpu 2
foreach cmd {earlyGlobalRoute timeDesign reportCongestion init_design defIn} {
    puts "COMMAND $cmd AVAILABLE [expr {[llength [info commands $cmd]] > 0}]"
}
puts "FLOWPLACE_INNOVUS_TCL_COMPLETED"
exit
'''
base=os.environ.copy()
for label,changes in [
    ('normal_bash_environment',environments.get('bash_login',{})),
    ('normal_zsh_environment',environments.get('zsh_login_interactive',{})),
    ('shared_file_normal_search',{'CDS_LIC_FILE':str(license_path),'CDS_LIC_ONLY':None}),
    ('shared_file_exclusive_search',{'CDS_LIC_FILE':str(license_path),'CDS_LIC_ONLY':'1'}),
    ('shared_file_plus_admin_setup',{'CDS_LIC_FILE':str(license_path),'CDS_LIC_ONLY':None}),
]:
    env=base.copy()
    for key,val in changes.items():
        if val is None:env.pop(key,None)
        else:env[key]=val
    folder=run/label;folder.mkdir(exist_ok=True)
    (folder/'probe.tcl').write_text(tcl)
    argv=[tool,'-no_gui','-batch','-files','probe.tcl','-log','probe']
    if label=='normal_bash_environment':argv=['/bin/bash','-lc','exec '+shlex.join(argv)]
    if label=='normal_zsh_environment':argv=['/bin/zsh','-lic','exec '+shlex.join(argv)]
    if label=='shared_file_plus_admin_setup':argv=['/bin/bash','-lc','source /fact_data/softwares/setup_eda.env; exec '+shlex.join(argv)]
    rc,out=command(label,argv,env=env,limit=120,cwd=folder)
    log=(folder/'probe.log').read_text(errors='replace') if (folder/'probe.log').exists() else ''
    success='FLOWPLACE_INNOVUS_TCL_COMPLETED' in out+log
    errors=[line for line in (out+log).splitlines() if any(x in line for x in ('IMPLIC-','LMC-','FLEX','License','license'))]
    case={'name':label,'exit_code':rc,'tcl_completed':success,'launcher_input_license_environment':{k:env.get(k) for k in keys},'diagnostics':errors[:35]}
    results['cases'].append(case);save();print(json.dumps(case),flush=True)
    if success:break
results['usable']=any(x['tcl_completed'] for x in results['cases'])
save();print(json.dumps({'job':job,'usable':results['usable'],'summary':str(run/'summary.json')}),flush=True)
