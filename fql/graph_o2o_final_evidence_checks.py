"""Exercise final-report evidence against completed physical results, read-only."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_final_evidence import audit_final_outcome, physical_identity, LEGACY_REPEAT_PLAN


def main(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    if plan['source_sha256'] != {name: sha(Path(__file__).with_name(name))
            for name in plan['source_sha256']}:
        raise ValueError('Evidence audit/check source changed since declaration')
    out = Path(plan['output']) / ('job_' + os.environ['SLURM_JOB_ID'] +
        '_restart_' + os.environ.get('SLURM_RESTART_COUNT', '0'))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    checked, negatives = [], []
    for item in plan['outcomes']:
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('Declared actual physical result changed')
        result = json.loads(path.read_text())
        actual = audit_final_outcome(result, design=item['design'], route_cap=20, legacy_plan=LEGACY_REPEAT_PLAN)
        checked.append(dict(outcome=str(path), outcome_sha256=sha(path),
            identity=actual['physical_identity'],
            actual_final_macro_count=actual['final_macro_count'],
            native_route_log_sha256=actual['native_route_log_sha256'],
            legacy_exact_repeat_evidence=actual['legacy_exact_repeat_evidence']))
        if len(checked) != 1:
            continue
        m, g = actual['manifest'], actual['source_manifest']
        assert isinstance(m['num_cores'], str), 'Regression must exercise the actual string-thread schema'
        assert 'tool_image_sha256' not in m, 'Regression must exercise inherited container provenance'
        integer = copy.deepcopy(m); integer['num_cores'] = 8
        assert physical_identity(integer, g, design=item['design']) == actual['physical_identity']
        mutations = [
            ('changed_threads', lambda x, y: x.update(num_cores='4')),
            ('changed_source_threads', lambda x, y: y.update(num_cores='4')),
            ('changed_route_cap', lambda x, y: x.update(route_end_iteration=4)),
            ('mismatched_container', lambda x, y: x.update(tool_image_sha256='0' * 64)),
            ('missing_bound_image', lambda x, y: y.pop('tool_image_sha256')),
            ('changed_tool', lambda x, y: x.update(openroad_sha256='0' * 64)),
            ('changed_design', lambda x, y: x.update(design='other_netlist')),
            ('changed_input_DEF', lambda x, y: x.update(def_sha256='0' * 64)),
            ('changed_physical_revision', lambda x, y: x.update(detailed_route_physical_revision='0' * 64)),
            ('unfinished_source', lambda x, y: y.update(success=False)),
        ]
        for name, mutate in mutations:
            a, b = copy.deepcopy(m), copy.deepcopy(g)
            mutate(a, b)
            try:
                physical_identity(a, b, design=item['design'])
            except (ValueError, KeyError):
                negatives.append(name)
            else:
                raise ValueError('Invalid physical evidence accepted: ' + name)
        altered = copy.deepcopy(result); altered['metrics']['DRT_WL'] += 1
        try:
            audit_final_outcome(altered, design=item['design'])
        except ValueError:
            negatives.append('modified_reported_PPA')
        else:
            raise ValueError('Modified final PPA was accepted')
    atomic_json(out / 'evidence.json', dict(status='actual_final_physical_schema_and_native_evidence_verified',
        plan_sha256=sha(plan_path), slurm_job_id=os.environ['SLURM_JOB_ID'], actual_results=checked,
        rejected_negative_fixtures=negatives, new_EDA_runs=0, new_training_steps=0,
        fixtures_never_enter_replay=True, goal_completed=False))
    print(json.dumps(dict(output=str(out), checked=len(checked), rejected=len(negatives))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run actual DEF audits through Slurm')
    main(args)
