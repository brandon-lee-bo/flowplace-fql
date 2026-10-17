"""Verify a pinned design-specific evaluator without weakening common checks.

The physical-flow digest includes the design config. A different netlist must
therefore have its own expected digest, even when tools and flow are identical.
The bp_fe pilot's verifier and all existing pilot labels remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .graph_o2o_archive import sha
from .graph_o2o_backend import ROOT, REFERENCE, parse_grt_metrics


def context(plan):
    if sha(REFERENCE) != plan['reference_manifest_sha256']:
        raise ValueError('The common physical reference changed')
    reference = json.loads(REFERENCE.read_text())
    expected = dict(reference['physical_flow_components'])
    config = ROOT / plan['design_config']
    if sha(config) != plan['design_config_sha256']:
        raise ValueError('The design-specific physical config changed')
    expected['design_config_sha256'] = plan['design_config_sha256']
    revision = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if (expected != plan['expected_physical_flow_components'] or
            revision != plan['expected_physical_flow_revision']):
        raise ValueError('Design-specific evaluator was not derived from the pinned common flow')
    for name, digest in plan['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Design-verification implementation changed: ' + name)
    return expected, revision


def verify(run, row, plan):
    run = Path(run)
    expected, revision = context(plan)
    manifest = json.loads((run / 'manifest.json').read_text())
    if not manifest.get('time_end'):
        raise ValueError('Physical run has no terminal manifest')
    if (manifest['design'] != plan['design'] or
            Path(manifest['design_config']).resolve() != (ROOT / plan['design_config']).resolve() or
            manifest['design_config_sha256'] != plan['design_config_sha256'] or
            manifest['physical_flow_components'] != expected or
            manifest['physical_flow_revision'] != revision or
            manifest['tool_image_sha256'] != plan['tool_image_sha256'] or
            int(manifest['num_cores']) != 8):
        raise ValueError('Physical evaluator differs from the pinned design/tools/thread count')
    if manifest['def_sha256'] != row['def_sha256'] or sha(manifest['def_path']) != row['def_sha256']:
        raise ValueError('Measured DEF differs from the requested fixed action')
    result = dict(physical_run=str(run), physical_manifest_sha256=sha(run / 'manifest.json'),
                  physical_flow_revision=revision, physical_slurm_job_id=manifest['slurm_job_id'],
                  stage='global_route_only', not_final_table2=True, detailed_route_DRC=None,
                  result_storage_policy=manifest['storage']['storage_policy'])
    if manifest['success']:
        saved = json.loads((run / 'metrics.json').read_text())
        runtime = saved['runtime_seconds']
        if not math.isfinite(runtime) or not 0 < runtime <= manifest['time_end'] - manifest['time_start'] + 1:
            raise ValueError('Invalid physical runtime')
        metrics = parse_grt_metrics(run, runtime)
        if metrics != saved:
            raise ValueError('Reparsed PPA differs from the saved physical measurement')
        result.update(status='success', metrics=metrics, metrics_sha256=sha(run / 'metrics.json'))
    else:
        log = run / 'floorplan_place_cts.log'
        text = log.read_text() if log.exists() else ''
        if ('floorplan_place_cts exited 2' in manifest.get('failure', '') and
                '[ERROR RSZ-0060] Max buffer count reached.' in text and
                'do-4_1_cts] Error 1' in text and 'do-cts] Error 2' in text and
                not (run / 'metrics.json').exists()):
            result.update(status='physical_failure', failure_class='cts_hold_repair_max_buffer_count',
                          metrics=None, failure_log_sha256=sha(log))
        else:
            raise RuntimeError('Unrecognized physical/infrastructure failure; no reward/PPA assigned')
    return result
