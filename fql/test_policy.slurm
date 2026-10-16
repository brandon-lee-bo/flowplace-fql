#!/bin/bash
#SBATCH --job-name=fql-grt-policy-test
#SBATCH --partition=cpu-scavenger
#SBATCH --qos=cpu-scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/logs/grt_policy_test_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python - <<'PY'
import json, os, pathlib, unittest
from fql.provenance import capture_source_provenance
output = pathlib.Path('runs') / ('software_grt_policy_' + os.environ['SLURM_JOB_ID'])
output.mkdir(parents=True, exist_ok=False)
extras = {name: pathlib.Path('fql') / name for name in ['test_physical_failure_policy.py', 'test_policy.slurm', 'test_backend_workflow.py', 'test_grt_export.py']}
snapshot = capture_source_provenance(output / 'source_snapshot', extras, 'startup_before_software_boundary_tests')
(output / 'source_provenance.json').write_text(json.dumps(snapshot, indent=2) + '\n')
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
names = [
    'fql.test_physical_failure_policy',
    'fql.test_backend_workflow.BackendWorkflowTests.test_exploratory_physical_failure_preserves_denominator',
    'fql.test_backend_workflow.BackendWorkflowTests.test_unknown_or_corrupt_physical_failure_remains_hard_error',
    'fql.test_backend_workflow.BackendWorkflowTests.test_fixed_control_index_does_not_create_noise_pairs',
    'fql.test_backend_workflow.BackendWorkflowTests.test_seal_success_and_reject_changed_heldout_plan',
    'fql.test_grt_export.GRTExportTests.test_validation_reuse_binding_and_sealed_heldout',
]
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
summary = dict(status='passed' if result.wasSuccessful() else 'failed', software_fixture_only=True,
               job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__, tests_run=result.testsRun,
               failures=[(str(test), detail) for test, detail in result.failures],
               errors=[(str(test), detail) for test, detail in result.errors], skipped=[(str(test), reason) for test, reason in result.skipped])
(output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
