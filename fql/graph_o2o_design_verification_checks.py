"""Read-only real-probe checks and rejection tests for the design verifier."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from .graph_o2o_archive import atomic_json, sha
from .graph_o2o_design_verify import verify


def run(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    original = json.loads(Path(plan['original_probe_plan']).read_text())
    design = json.loads(Path(original['design_plan']).read_text())
    evidence = json.loads((Path(design['audit_output']) / 'evidence.json').read_text())
    output = Path(original['output']) / 'verification_checks' / os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True, exist_ok=False)
    passed = []
    for probe in evidence['import_probes']:
        row = probe['record']
        outcome_path = Path(original['output']) / f"family_{row['source_family']:04d}" / 'outcome.json'
        outcome = json.loads(outcome_path.read_text())
        if outcome['verification_plan_sha256'] != sha(path):
            raise ValueError('Probe was admitted by another verification plan')
        result = verify(outcome['physical_run'], row, plan)
        for key in ['physical_manifest_sha256', 'physical_flow_revision', 'metrics_sha256', 'metrics']:
            if result[key] != outcome[key]:
                raise ValueError('Independent real-result recheck differs: ' + key)
        passed.append(dict(outcome=str(outcome_path), sha256=sha(outcome_path),
                           physical_manifest_sha256=result['physical_manifest_sha256']))
    immutable = []
    for physical, expected in plan['completed_physical_runs_before_recovery'].items():
        if sha(Path(physical) / 'manifest.json') != expected:
            raise ValueError('Recovering an old successful run changed its terminal physical manifest')
        immutable.append(physical)
    source = json.loads(Path(passed[0]['outcome']).read_text())
    row = evidence['import_probes'][0]['record']
    base = json.loads((Path(source['physical_run']) / 'manifest.json').read_text())
    mutations = {
        'wrong_design': {'design': 'bp_fe'},
        'wrong_design_config': {'design_config_sha256': '0' * 64},
        'old_bp_fe_flow_digest': {'physical_flow_revision': '9250cb616501862e16a17fde5c8f408b7e0d51e99d1ee5f4ff8d4d27a9e4fd3e'},
        'wrong_image': {'tool_image_sha256': '0' * 64},
        'wrong_threads': {'num_cores': '4'},
        'wrong_action_DEF': {'def_sha256': '0' * 64},
        'unfinished_run': {'time_end': None},
        'changed_common_flow': {'physical_flow_components': dict(base['physical_flow_components'], openroad_binary_sha256='0' * 64)},
    }
    rejected = []
    for name, patch in mutations.items():
        mutated = output / ('rejected_' + name)
        mutated.mkdir()
        manifest = copy.deepcopy(base)
        manifest.update(patch)
        atomic_json(mutated / 'manifest.json', manifest)
        try:
            verify(mutated, row, plan)
        except ValueError as error:
            rejected.append(dict(mutation=name, error=str(error)))
        else:
            raise AssertionError('A changed evaluator/action was accepted: ' + name)
    result = dict(status='real_probe_verification_and_rejection_checks_passed',
        verification_plan_sha256=sha(path), positive_real_probes=passed,
        recovered_terminal_manifests_unchanged=immutable, rejected_mutations=rejected,
        rejected_manifests_are_test_fixtures_not_measurements=True, new_EDA_runs=0,
        final_detailed_route_PPA=False, source_sha256=sha(__file__), slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(output / 'evidence.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verification-plan', required=True)
    args = p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Use Slurm for independent physical-evidence checks')
    run(args.verification_plan)
