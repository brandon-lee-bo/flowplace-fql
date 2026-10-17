"""Read-only checks of actual GRT-to-DRT evidence, including native macro DEFs.

The DRT runner inherits its image from the cryptographically bound GRT source.
Its thread count is serialized from the shell and can therefore be a string.
Neither representation changes the physical experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from analysis.audit_table2_macro_preservation import macros
from .graph_o2o_archive import sha

LEGACY_REPEAT_PLAN = Path('/fact_data/bohanli/flowplace_fql/graph_o2o_report_schema_repair_v1/legacy_repeat_plan.json')


def physical_identity(manifest, source, *, design=None, route_cap=20):
    if (not manifest.get('success') or not manifest.get('time_end') or
            not source.get('success') or not source.get('time_end') or
            manifest['stage'] != 'finish_after_detailed_route' or
            source['stage'] != 'global_route_only' or
            manifest['design'] != source['design'] or
            (design is not None and manifest['design'] != design) or
            int(manifest['num_cores']) != 8 or int(source['num_cores']) != 8 or
            manifest['route_end_iteration'] != route_cap):
        raise ValueError('Incomplete or mismatched design, stage, threads or route budget')
    image = source['tool_image_sha256']
    if not re.fullmatch(r'[0-9a-f]{64}', image):
        raise ValueError('Source GRT lacks a recorded container image digest')
    if manifest.get('tool_image_sha256', image) != image:
        raise ValueError('DRT and its bound GRT source use different container images')
    for key in ['physical_flow_components', 'physical_flow_revision', 'openroad_sha256',
                'design_config_sha256', 'def_sha256']:
        if manifest[key] != source[key]:
            raise ValueError('DRT changed its measured GRT input or physical settings: ' + key)
    components = manifest['physical_flow_components']
    if (components['openroad_binary_sha256'] != manifest['openroad_sha256'] or
            components['design_config_sha256'] != manifest['design_config_sha256']):
        raise ValueError('Physical component and tool/configuration digests disagree')
    identity = dict(grt_components=components,
                    finish_components=manifest['detailed_route_flow_components'],
                    route_end_iteration=route_cap, num_cores=8)
    revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if revision != manifest['detailed_route_physical_revision']:
        raise ValueError('Detailed-route fingerprint does not match actual physical settings')
    return dict(design=manifest['design'], physical_flow_components=components,
                physical_flow_revision=manifest['physical_flow_revision'],
                detailed_route_flow_components=manifest['detailed_route_flow_components'],
                detailed_route_physical_revision=revision, tool_image_sha256=image,
                num_cores=8, route_end_iteration=route_cap)


def audit_final_outcome(outcome, *, design=None, route_cap=20, legacy_plan=None):
    """Re-read actual files; a known upstream failure is handled by its caller."""
    if outcome['status'] != 'completed':
        raise ValueError('Only completed physical outcomes have final numerical PPA')
    run = Path(outcome['physical_run'])
    manifest_path, metrics_path = run / 'manifest.json', run / 'metrics.json'
    if (sha(manifest_path) != outcome['physical_manifest_sha256'] or
            sha(metrics_path) != outcome['metrics_sha256']):
        raise ValueError('Actual final manifest/metrics changed')
    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if (metrics != outcome['metrics'] or
            manifest['canonical_metrics_sha256'] != sha(metrics_path)):
        raise ValueError('Numerical final PPA does not match its actual source')
    source = Path(manifest['source_grt_run'])
    source_manifest_path, source_metrics_path = source / 'manifest.json', source / 'metrics.json'
    if (sha(source_manifest_path) != manifest['source_grt_manifest_sha256'] or
            sha(source_metrics_path) != manifest['source_grt_metrics_sha256'] or
            Path(outcome['spec']['source_run']).resolve() != source.resolve() or
            outcome['spec']['source_manifest_sha256'] != sha(source_manifest_path) or
            outcome['spec']['route_end_iteration'] != route_cap):
        raise ValueError('Final result is not bound to the actual measured GRT source')
    source_manifest = json.loads(source_manifest_path.read_text())
    source_metrics = json.loads(source_metrics_path.read_text())
    legacy_proof = None
    if 'detailed_route_flow_components' in manifest and 'detailed_route_physical_revision' in manifest:
        identity = physical_identity(manifest, source_manifest, design=design, route_cap=route_cap)
    else:
        if legacy_plan is None:
            raise ValueError('Legacy DRT needs an explicit verified same-source repeat, not an inferred fingerprint')
        plan = json.loads(Path(legacy_plan).read_text())
        matches = [item for item in plan['entries']
            if item['original_physical_manifest_sha256'] == sha(manifest_path)]
        if len(matches) != 1 or plan['threads'] != 8 or plan['route_cap'] != route_cap:
            raise ValueError('Missing or ambiguous legacy repeat declaration')
        item = matches[0]
        original_path = Path(item['original_outcome'])
        if (sha(original_path) != item['original_outcome_sha256'] or
                json.loads(original_path.read_text()) != outcome or
                item['source_grt_manifest_sha256'] != sha(source_manifest_path)):
            raise ValueError('Legacy repeat does not bind this exact original result and source')
        repeat_path = Path(item['repeat_outcome'])
        repeat = json.loads(repeat_path.read_text())
        measured_repeat = audit_final_outcome(repeat, design=design, route_cap=route_cap)
        if (Path(measured_repeat['source_run']).resolve() != source.resolve() or
                any(repeat['metrics'][key] != metrics[key] for key in
                    ['DRT_WL', 'WNS', 'TNS', 'Power', 'StdCellArea', 'DRC'])):
            raise ValueError('Same-source repeat did not reproduce all six original physical metrics exactly')
        identity = physical_identity(measured_repeat['manifest'], source_manifest,
                                     design=design, route_cap=route_cap)
        legacy_proof = dict(plan=str(Path(legacy_plan).resolve()), plan_sha256=sha(legacy_plan),
            repeat_outcome=str(repeat_path), repeat_outcome_sha256=sha(repeat_path),
            repeat_physical_manifest_sha256=repeat['physical_manifest_sha256'],
            all_six_metrics_identical=True, original_metadata_not_rewritten=True,
            original_DRT_fingerprint_not_captured=True)
    for key in ['DRT_WL', 'WNS', 'TNS', 'Power', 'StdCellArea', 'DRC']:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Nonfinite or nonnumerical physical metric: ' + key)
    if (metrics['DRT_WL'] <= 0 or metrics['StdCellArea'] <= 0 or metrics['Power'] < 0 or
            metrics['DRC'] < 0 or int(metrics['DRC']) != metrics['DRC'] or
            manifest['drc_native_final_count'] != metrics['DRC'] or
            manifest['route_drc_clean'] != (metrics['DRC'] == 0) or
            outcome['route_drc_clean'] != (metrics['DRC'] == 0) or
            not manifest['final_macros_match_audited_grt'] or
            not outcome['final_macros_match_audited_grt']):
        raise ValueError('Native final DRC or preserved macro evidence disagrees')
    geometry_path = source / 'backend_macro_audit.json'
    geometry = json.loads(geometry_path.read_text())
    if (sha(geometry_path) != manifest['source_geometry_sha256'] or
            geometry['bindings']['physical_manifest_sha256'] != sha(source_manifest_path) or
            not geometry['legal_no_halo']):
        raise ValueError('Measured GRT macro geometry is invalid or changed')
    final = run / 'orfs_work/results/nangate45' / manifest['design'] / 'eval/6_final.def'
    if sha(final) != manifest['final_def_sha256']:
        raise ValueError('Actual final DEF changed')
    units, actual = macros(final)
    if units != geometry['dbu_per_um'] or set(actual) != set(geometry['actual_macros']):
        raise ValueError('Final macro inventory or DEF units differ from audited GRT')
    for name, value in actual.items():
        prior = geometry['actual_macros'][name]
        if (value['xy_dbu'] != prior['bbox_dbu'][:2] or value['master'] != prior['master'] or
                value['orientation'] != prior['orientation'] or value['status'] != 'FIXED'):
            raise ValueError('Actual final DEF moved a frozen macro: ' + name)
    route_log = run / 'orfs_work/logs/nangate45' / manifest['design'] / 'eval/5_2_route.log'
    text = route_log.read_text()
    counts = re.findall(r'Number of violations = (\d+)\.', text)
    if ('[INFO DRT-0198] Complete detail routing.' not in text or not counts or
            int(counts[-1]) != metrics['DRC']):
        raise ValueError('Native detailed route did not complete with the reported DRC')
    return dict(physical_identity=identity, manifest=manifest, source_manifest=source_manifest,
                source_metrics=source_metrics, source_run=str(source.resolve()),
                final_def_sha256=sha(final), final_macro_count=len(actual),
                native_route_log_sha256=sha(route_log), actual_macro_DEF_reparsed=True,
                legacy_exact_repeat_evidence=legacy_proof)
