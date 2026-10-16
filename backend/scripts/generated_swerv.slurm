#!/bin/bash
#SBATCH --job-name=generated_swerv_ppa
#SBATCH --partition=eda
#SBATCH --qos=eda
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend/logs/generated_swerv_%j.log
set -euo pipefail
cd /fact_home/bohanli/RL_Placement/experiments/flowplace_fql/backend
run_id=generated_swerv_deterministic_seed300_job${SLURM_JOB_ID}
input_dir=results/${run_id}_input
mkdir "$input_dir"
python3 scripts/export_macro_def.py --design swerv_wrapper \
  --coordinates ../baseline/artifacts/official_stream_deterministic_seed300/swerv_wrapper/seed_300/macro_physical.json \
  --output "$input_dir/swerv_wrapper.def"
python3 - "$input_dir" "$run_id" <<'PY'
import hashlib, json, pathlib, sys
source = pathlib.Path('../baseline/artifacts/official_stream_deterministic_seed300/swerv_wrapper/seed_300').resolve()
out = pathlib.Path(sys.argv[1]).resolve()
paths = [source/'macro_physical.json', source/'metrics.json', source/'cache_geometry.json',
         out/'swerv_wrapper.def', out/'swerv_wrapper.audit.json']
record = {'scope': 'Newly generated deterministic seed300 swerv baseline; does not repair or replace the released-DEF Table2 row',
          'run_id': sys.argv[2], 'source_generation': str(source),
          'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(out/'generation_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
PY
status=0
bash scripts/evaluate.slurm --design swerv_wrapper --def-path "$input_dir/swerv_wrapper.def" --run-id "$run_id" || status=$?
if [[ -d results/$run_id ]]; then
  cp "$input_dir/generation_provenance.json" "results/$run_id/generation_provenance.json"
fi
if [[ $status -eq 0 ]]; then
  python3 scripts/audit_fullflow.py "results/$run_id"
fi
exit "$status"
