"""Translate a descriptive legacy plan field without changing any experiment.

The fresh pilot accidentally used physical_failure_policy for prose. That key
is reserved by the older preregistered 48-case experiment's cryptographic
binding. Preserve/bind the original and rename only the descriptive field.
All twelve physical outcomes must succeed under the unchanged strict auditor.
"""
import argparse
import json
import os
from pathlib import Path
from .aggregate_backend_candidates import aggregate
from .relabel_backend import sha256_file, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',required=True,type=Path)
    p.add_argument('--retry-map',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Use Slurm')
    original=a.tasks.with_name('selection_plan.json').resolve()
    plan=json.loads(original.read_text())
    if plan['phase']!='exploratory' or not isinstance(plan.get('physical_failure_policy'),str):
        raise ValueError('This adapter only translates the fresh pilot descriptive field')
    if plan['indices']!=[192,193] or plan['new_backend_budget']!=12:
        raise ValueError('Pilot scope changed')
    translated=dict(plan)
    translated['physical_failure_note']=translated.pop('physical_failure_policy')
    translated['source_bindings']=dict(plan.get('source_bindings',{}),
        original_selection_plan={'path':str(original),'sha256':sha256_file(original)})
    # All scientific fields retain exactly the same values, with only a prose-key rename.
    restored=dict(translated)
    restored['physical_failure_policy']=restored.pop('physical_failure_note')
    restored.pop('source_bindings')
    if plan.get('source_bindings'):restored['source_bindings']=plan['source_bindings']
    if restored!=plan:raise ValueError('Unexpected plan alteration')
    a.output.mkdir(parents=True,exist_ok=False)
    fixed=a.output/'selection_plan_compatibility.json';write_json(fixed,translated)
    write_json(a.output/'compatibility_proof.json',dict(
        original_plan=str(original),original_plan_sha256=sha256_file(original),
        translated_plan_sha256=sha256_file(fixed),source_tasks_sha256=sha256_file(a.tasks),
        sole_change='Rename descriptive physical_failure_policy prose to physical_failure_note, bind original plan.',
        scientific_fields_identical=True,strict_all_success_required=True,
        source_sha256=sha256_file(__file__)))
    result=aggregate(a.tasks,a.output/'aggregate',plan_path=fixed,infrastructure_retry_map=a.retry_map)
    if result['status']!='completed' or result['successful_count']!=12:
        raise RuntimeError('Strict complete 12-case aggregate required')
    print(json.dumps(dict(status=result['status'],successful_count=result['successful_count'])),flush=True)


if __name__=='__main__':main()
