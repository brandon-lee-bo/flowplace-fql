"""Recover only the 81 explicitly audited EDQUOT requests, without changing labels.

Six old runners ended with an infrastructure EDQUOT exception; the other 75
never wrote attempt.json and therefore never invoked the physical runner.
The exact frozen measure bytecode is used in a private namespace.  Under its
original sample flock, only the six immutable, unsuccessful quota attempts are
omitted from its attempt iterator.  Their FAILED accounting and files remain
unchanged.  Successful/completed/new attempts retain the original evaluator's
verification and duplicate-work guards.  No infrastructure failure is a label.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

# Local analysis/backend imports must precede upstream baseline path changes.
from . import qstudy_backend as backend
from .graph_o2o_archive import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/fact_data/bohanli/flowplace_fql/qstudy_v1')
DIAGNOSIS = Path('/scratch/bohanli/flowplace_graph/qstudy_quota_recovery_v1/diagnosis_job1797461.json')
DIAGNOSIS_SHA = 'd08dfe4c6ac5725b5177ac6de8b8df921dbfac2dd533859d7f87c99e475177fd'
BACKEND_SHA = 'c88bbbfa21a0e816e1f645c91ed4f66deabe6d1f112f9f820786afeb93a43265'
SPECS = {
    'source': dict(array='1797162', indices=list(range(15, 48)),
        matrix=str(BASE/'source_actors_v1/controller/backend_round_0.jsonl'),
        matrix_sha256='64ea726c4d66512501b39f906ad61ba678da2a44df0a464221c5490ca81f7e1c'),
    'flow': dict(array='1797375', indices=list(range(48)),
        matrix=str(BASE/'actors_v1/flowupdate_controller/backend_round_1.jsonl'),
        matrix_sha256='940ed9870c0123ec1f12cd45db00e7a28cbd02978cfff9a86c7ce82a13cc5866'),
}
SOURCE_FILES = (
    'fql/qstudy_quota_recovery.py', 'fql/qstudy_quota_recovery.slurm',
    'fql/qstudy_backend.py', 'fql/qstudy_backend.slurm', 'fql/qstudy_data.py',
    'fql/graph_o2o_backend.py', 'fql/graph_o2o_design_verify.py',
    'fql/graph_o2o_archive.py', 'fql/graph_o2o_label.sh',
    'fql/graph_o2o_geometry.py', 'fql/graph_o2o_geometry.sh',
    'fql/graph_o2o_geometry.tcl', 'backend/scripts/run_grt_label.py',
    'backend/scripts/result_storage.py', 'backend/scripts/grt_metrics.py',
    'backend/scripts/grt_provenance.py', 'analysis/audit_table2_macro_preservation.py',
)
ATTEMPT_FILES = tuple(p for p in SOURCE_FILES if p not in (
    'fql/qstudy_quota_recovery.py', 'fql/qstudy_quota_recovery.slurm')) + (
        'attempt.json', 'attempt.json.tmp', 'process_exit.json',
        'process_exit.json.tmp', 'runner.log')
NATIVE_FILES = ('manifest.json', 'metrics.json', 'metrics.json.tmp',
    'backend_macro_audit.json', 'macro_import.log', 'floorplan_place_cts.log',
    'global_route_label.log')
QUOTA_FAILURE = "OSError(122, 'Disk quota exceeded')"


def read(path):
    return json.loads(Path(path).read_text())


def bound(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError('Immutable quota evidence fingerprint changed: '+str(path))
    return path


def fingerprint(path):
    path = Path(path)
    return dict(path=str(path), exists=path.exists(),
        sha256=sha(path) if path.exists() else None)


def check_fingerprint(value):
    path = Path(value['path'])
    if path.exists() != value['exists']:
        raise ValueError('Old quota evidence presence changed: '+str(path))
    if value['exists']:
        bound(path, value['sha256'])


def stream_path(task):
    return (Path(task['experiment'])/f"seed_{task['seed']}"/
            f"round_{task['round']}"/'requests/tasks.jsonl')


def snapshot_attempt(old):
    """Only the known finite source/sidecar paths in this one old attempt."""
    directory = Path(old['directory'])
    if sorted(p.name for p in directory.iterdir()) != old['root_files']:
        raise ValueError('Old attempt changed since independent quota diagnosis')
    for evidence in old['files']:
        bound(evidence['path'],evidence['sha256'])
    if old['attempt_json_exists']:
        if read(directory/'attempt.json') != old['metadata']:
            raise ValueError('Old attempt metadata changed since quota diagnosis')
        bound(old['native']['path'],old['native']['sha256'])
    value = dict(directory=str(directory), root_names=sorted(p.name for p in directory.iterdir()),
        files=[fingerprint(directory/p) for p in ATTEMPT_FILES],
        metadata=read(directory/'attempt.json') if (directory/'attempt.json').exists() else None)
    if value['metadata'] is not None:
        run = Path(value['metadata']['physical_run'])
        value['native_files'] = [fingerprint(run/p) for p in NATIVE_FILES]
        value['native_manifest'] = read(run/'manifest.json')
    return value


def original_states(row):
    # Query the exact array element, not its JobIDRaw: the final element can
    # have raw ID equal to the array parent and must not observe all siblings.
    task_id = row['array']+'_'+str(row['flat_index'])
    return backend.states(task_id)


def validate_original(row, *, check_scheduler=True):
    """Recheck all original evidence; production invokes this inside flock."""
    task = row['task']
    bound(row['matrix']['path'], row['matrix']['sha256'])
    log = bound(row['stdout']['path'], row['stdout']['sha256'])
    if '[Errno 122] Disk quota exceeded' not in log.read_text():
        raise ValueError('Failure is not the audited EDQUOT infrastructure event')
    bound(stream_path(task), task['tasks_sha256'])
    sample = Path(task['sample_dir'])
    bound(sample/'request.json',row['request_sha256'])
    if read(sample/'request.json') != row['request']:
        raise ValueError('Request sidecar differs from diagnosed request')
    bound(Path(task['experiment'])/'config.json',row['request']['config_sha256'])
    for evidence in row['input_files']:
        check_fingerprint(evidence)
    if len(row['attempts']) != 1:
        raise ValueError('Only the one original audited attempt may be excluded')
    old = row['attempts'][0]
    directory = Path(old['directory'])
    if sorted(p.name for p in directory.iterdir()) != old['root_names']:
        raise ValueError('Original quota attempt directory changed')
    for evidence in old['files']:
        check_fingerprint(evidence)
    meta = old['metadata']
    if meta is not None:
        if (read(directory/'attempt.json') != meta or meta['array_job_id'] != row['array']
                or int(meta['array_task_id']) != row['flat_index']
                or str(meta['job_id']) != row['accounting'][1]
                or meta['request_sha256'] != row['request_sha256']
                or meta['tasks_sha256'] != task['tasks_sha256']):
            raise ValueError('Original quota attempt metadata changed its identity')
        for evidence in old['native_files']:
            check_fingerprint(evidence)
        manifest = read(Path(meta['physical_run'])/'manifest.json')
        if (manifest != old['native_manifest'] or manifest.get('success') is not False
                or not manifest.get('time_end') or manifest.get('failure') != QUOTA_FAILURE
                or int(manifest['num_cores']) != 8
                or manifest['def_sha256'] != row['request']['def_sha256']
                or manifest['physical_flow_revision'] != backend.PHYSICAL_REVISIONS[row['request']['graph']]):
            raise ValueError('Only a terminal, unsuccessful, exact EDQUOT native attempt may be omitted')
    if check_scheduler and original_states(row) != ['FAILED']:
        raise RuntimeError('Original quota job must remain truly FAILED and never live')


def diagnosis():
    value = read(bound(DIAGNOSIS, DIAGNOSIS_SHA))
    bound(ROOT/'fql/qstudy_backend.py', BACKEND_SHA)
    if value.get('schema') != 'qstudy_EDQUOT_readonly_diagnosis_v1' or value['job_id'] != '1797461':
        raise ValueError('Wrong independently recorded EDQUOT diagnosis')
    expected = [(kind, i) for kind, spec in SPECS.items() for i in spec['indices']]
    if [(r['kind'], r['flat_index']) for r in value['rows']] != expected:
        raise ValueError('Not the exact 33 source plus 48 flow quota requests')
    for row in value['rows']:
        spec = SPECS[row['kind']]
        if (row['array'] != spec['array'] or row['matrix'] != dict(path=spec['matrix'],sha256=spec['matrix_sha256'])
                or row['accounting'][0] != row['array']+'_'+str(row['flat_index'])
                or row['accounting'][2:] != ['FAILED','1:0','0'] or row['outcome_exists']):
            raise ValueError('Diagnosis does not record the exact FAILED/exit1/restart0 event')
    return value


def checked_plan(path, digest=None):
    if digest is not None:
        bound(path, digest)
    value = read(path)
    if (value.get('schema') != 'qstudy_EDQUOT_recovery_plan_v1' or value.get('specs') != SPECS
            or value.get('diagnosis') != dict(path=str(DIAGNOSIS),sha256=DIAGNOSIS_SHA)
            or set(value.get('source_sha256',{})) != set(SOURCE_FILES)):
        raise ValueError('Wrong narrow quota recovery declaration')
    for name, digest in value['source_sha256'].items():
        bound(ROOT/name, digest)
    evidence = diagnosis()
    expected = [(r['kind'],r['flat_index']) for r in evidence['rows']]
    if [(r['kind'],r['flat_index']) for r in value['rows']] != expected:
        raise ValueError('Recovery row scope changed')
    for row, original in zip(value['rows'],evidence['rows']):
        for key in ('kind','array','flat_index','accounting','task','matrix','request_sha256','stdout'):
            if row[key] != original[key]:
                raise ValueError('Recovery differs from original diagnosis: '+key)
    return value


def plan(args):
    evidence = diagnosis()
    output = args.plan.resolve()
    if output.exists():
        value = checked_plan(output)
        print(json.dumps(dict(status='existing_immutable_plan',path=str(output),sha256=sha(output),rows=len(value['rows']))))
        return
    rows = []
    for original in evidence['rows']:
        row = {k: original[k] for k in ('kind','array','flat_index','accounting','task','matrix','request_sha256','stdout')}
        task = row['task']; sample = Path(task['sample_dir']); request = read(sample/'request.json')
        matrix = bound(row['matrix']['path'],row['matrix']['sha256'])
        tasks = [json.loads(s) for s in matrix.read_text().splitlines()]
        if tasks[row['flat_index']] != task or (sample/'outcome.json').exists():
            raise ValueError('Dispatch/measurement changed before recovery declaration')
        row['request'] = request
        row['input_files'] = [fingerprint(p) for p in [
            Path(task['experiment'])/'config.json', stream_path(task),
            stream_path(task).with_name('completed.json'), sample/'request.json',
            sample/'actions.npz', Path(request['def_path']), Path(request['def_path']).with_suffix('.audit.json')]]
        row['attempts'] = [snapshot_attempt(old) for old in original['attempts']]
        row['excluded_terminal_infra_attempts'] = [str(Path(a['directory'])/'attempt.json')
            for a in row['attempts'] if a['metadata'] is not None]
        validate_original(row)
        rows.append(row)
    skipped = [(r['kind'],r['flat_index']) for r in rows if r['excluded_terminal_infra_attempts']]
    if skipped != [('source',15),('source',16),('source',17),('source',18),('flow',0),('flow',1)]:
        raise ValueError('Only the six diagnosed terminal quota attempts can be omitted')
    protected = []
    source_tasks = [json.loads(s) for s in Path(SPECS['source']['matrix']).read_text().splitlines()]
    for task in source_tasks[:15]:
        protected.append(fingerprint(Path(task['sample_dir'])/'outcome.json'))
    if not all(p['exists'] for p in protected):
        raise ValueError('An earlier completed source outcome disappeared')
    value = dict(schema='qstudy_EDQUOT_recovery_plan_v1',specs=SPECS,
        diagnosis=dict(path=str(DIAGNOSIS),sha256=DIAGNOSIS_SHA),rows=rows,
        protected_completed_outcomes=protected,source_sha256={p:sha(ROOT/p) for p in SOURCE_FILES},
        created_job_id=os.environ['SLURM_JOB_ID'],created_unix=time.time(),
        logical_requests=81,original_real_native_attempts=6,uninvoked_original_requests=75,
        maximum_new_native_attempts=81,cpus_per_native_task=8,memory='24G',
        physical_settings_unchanged=True,real_failed_states_preserved=True,
        note='No terminal unsuccessful native transaction is promoted to success. Retries retain original physical attempts in cost accounting; no new logical label budget.')
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,value)
    print(json.dumps(dict(status='planned',path=str(output),sha256=sha(output),rows=81,
        excluded_terminal_infra=6,partial_uninvoked=75)))


def target_guards(row, real_load=backend.load_requests, real_check=backend.checked_request):
    task = row['task']; expected = Path(task['sample_dir']).resolve()
    omitted = {Path(p).resolve() for p in row['excluded_terminal_infra_attempts']}

    class QuotaSamplePath(type(Path())):
        def glob(self, pattern):
            # Explicit, private omission of immutable infrastructure failures.
            # Every later/new attempt remains visible to the frozen evaluator.
            values = Path(self).glob(pattern)
            if Path(self).resolve() == expected/'backend_attempts' and pattern == '*/attempt.json':
                return (p for p in values if p.resolve() not in omitted)
            return values

    def load(experiment, seed, round_number):
        if (Path(experiment).resolve() != Path(task['experiment']).resolve()
                or seed != task['seed'] or round_number != task['round']):
            raise ValueError('Recovery selected an unaudited experiment/seed/round')
        rows,path = real_load(experiment,seed,round_number)
        bound(path,task['tasks_sha256'])
        if (not 0 <= task['index'] < len(rows) or rows[task['index']] != row['request']
                or Path(rows[task['index']]['sample_dir']).resolve() != expected):
            raise ValueError('Recovery stream now resolves to another request')
        bound(expected/'request.json',row['request_sha256'])
        return rows,path

    def check(actual,args,cfg,binding):
        sample = real_check(actual,args,cfg,binding)
        if (Path(sample).resolve() != expected or actual != row['request']
                or args.seed != task['seed'] or args.round != task['round']):
            raise ValueError('Frozen validator selected an unaudited recovery request')
        bound(stream_path(task),task['tasks_sha256'])
        bound(expected/'request.json',row['request_sha256'])
        return QuotaSamplePath(sample)

    return load,check


def lock_guard(row, plan_path, plan_sha256, real_flock=fcntl.flock):
    """Checks run after acquisition, including outcome/empty-attempt paths."""
    def flock(handle, flags):
        real_flock(handle,flags)
        expected = Path(row['task']['sample_dir'])/'.qstudy_backend.lock'
        if Path(handle.name).resolve() != expected.resolve():
            raise ValueError('Recovery acquired a different sample lock')
        value = checked_plan(plan_path,plan_sha256)
        found = next(r for r in value['rows'] if (r['kind'],r['flat_index']) == (row['kind'],row['flat_index']))
        if found != row:
            raise ValueError('Quota recovery row changed inside original flock')
        validate_original(row)
        folder = expected.parent/'quota_infrastructure_recovery'
        folder.mkdir(exist_ok=True)
        dest = folder/(os.environ['SLURM_JOB_ID']+'.json')
        record = dict(schema='qstudy_EDQUOT_under_lock_admission_v1',plan_path=str(plan_path),
            plan_sha256=plan_sha256,kind=row['kind'],original_flat_index=row['flat_index'],
            actual_original_scheduler_states=['FAILED'],original_attempt_files_preserved=True,
            excluded_terminal_infra_attempts=row['excluded_terminal_infra_attempts'],
            adapter_source_sha256=sha(__file__),job_id=os.environ['SLURM_JOB_ID'],
            original_sample_flock_held=True,created_unix=time.time())
        if dest.exists():
            old = read(dest)
            if any(old.get(k)!=v for k,v in record.items() if k!='created_unix'):
                raise ValueError('Quota admission provenance changed')
        else:
            atomic_json(dest,record)
    return SimpleNamespace(flock=flock,LOCK_EX=fcntl.LOCK_EX,LOCK_NB=fcntl.LOCK_NB)


def private_measure(original,loader,checker,lock):
    namespace = dict(original.__globals__)
    namespace.update(load_requests=loader,checked_request=checker,fcntl=lock)
    copied = FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    copied.__kwdefaults__ = original.__kwdefaults__
    return copied


def worker(args):
    if not args.plan_sha256:
        raise ValueError('Worker requires exact immutable plan SHA')
    value = checked_plan(args.plan,args.plan_sha256)
    index = args.flat_index if args.flat_index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if args.kind not in SPECS or index not in SPECS[args.kind]['indices']:
        raise IndexError('Outside the exact 81 audited quota requests')
    row = next(r for r in value['rows'] if (r['kind'],r['flat_index']) == (args.kind,index))
    task = row['task']; load,check = target_guards(row)
    measure = private_measure(backend.measure,load,check,lock_guard(row,args.plan,args.plan_sha256))
    if measure.__code__ is not backend.measure.__code__:
        raise AssertionError('Frozen evaluator bytecode changed')
    arguments = argparse.Namespace(experiment=task['experiment'],seed=task['seed'],round=task['round'],
        index=task['index'],storage='scratch')
    measure(arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['plan','worker'],required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--kind',choices=list(SPECS))
    parser.add_argument('--flat-index',type=int)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for quota audit/recovery')
    if args.stage == 'plan':
        plan(args)
    else:
        if int(os.environ.get('SLURM_CPUS_PER_TASK','0')) != 8:
            raise RuntimeError('Physical worker needs exactly eight allocated CPUs')
        worker(args)


if __name__ == '__main__':
    main()
