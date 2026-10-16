"""Separate immutable physical-flow identity from reporting/parser revisions."""
import hashlib
import json
import time
from pathlib import Path

FLOW_ARCHIVE_SHA = '4a0e7841a9ea6f580b3253bf7e57f696dde73f0e03545e2d4026c6229cba631f'
REPORT_V1_SHA = '89b3de3662643e1483c524c01d4ceba1abbaa8f1cc6020258c4a69daaa161de8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp_provenance(manifest, base, retrospective=False):
    flow = base / 'thirdparty/OpenROAD-flow-scripts'
    components = {
        'author_flow_archive_sha256': FLOW_ARCHIVE_SHA,
        'openroad_binary_sha256': manifest['openroad_sha256'],
        'design_config_sha256': manifest['design_config_sha256'],
        'platform_config_sha256': sha(flow / 'platforms/nangate45/config.mk'),
        'patched_write_ref_sdc_sha256': sha(flow / 'scripts/write_ref_sdc.tcl'),
        'global_route_optimization_sha256': sha(flow / 'scripts/global_route.tcl'),
    }
    manifest['physical_flow_components'] = components
    manifest['physical_flow_revision'] = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    wrapper = manifest.get('grt_report_wrapper_sha256')
    if wrapper is None and retrospective:
        # The first two jobs predated runtime wrapper hashing. Preserve that fact.
        manifest['reconstructed_reporting_wrapper_sha256'] = REPORT_V1_SHA
        manifest['reporting_hash_provenance'] = 'Reconstructed from archived v1 wrapper and execution chronology; not captured at original job start'
        wrapper = REPORT_V1_SHA
    manifest['executed_reporting_revision'] = ('grt_report_v1_timing_only' if wrapper == REPORT_V1_SHA
                                                else 'grt_report_v2_final_marker_evidence')
    manifest['evaluator_revision'] = 'author_release_1.0.0_envscope1_' + manifest['executed_reporting_revision']
    manifest['physical_flow_note'] = 'Marker reporting only reads OpenDB category/count and writes text; no placement/routing/optimization command changes.'
    if retrospective:
        manifest.setdefault('metadata_audit_events', []).append({
            'time': time.time(),
            'note': 'Added explicit physical-flow/reporting revision after execution; runtime-captured hashes preserved, absent original hashes remain identified as reconstructed.'})
    else:
        manifest['runner_sha256'] = sha(base / 'scripts/run_grt_label.py')
        manifest['initial_parser_sha256'] = sha(base / 'scripts/grt_metrics.py')
    return manifest
