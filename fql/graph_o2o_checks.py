"""Exercise durable archive safety and revalidate one existing real GRT label."""
import json
import os
from pathlib import Path
import time
from .graph_o2o_archive import ROOT as BULK, archive_run, atomic_json, inventory, sha, verify_archive
from .graph_o2o_backend import ROOT, verify_physical


def main():
    job = os.environ['SLURM_JOB_ID']
    fixture = BULK/('go2o_archive_smoke_job'+job)
    fixture.mkdir(exist_ok=False)
    for name, content in [('results/a/final.bin', bytes(range(256))*1024),
                          ('objects/b/state.dat', b'archiving-contract\n'*2048)]:
        target = fixture/'orfs_work'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    atomic_json(fixture/'manifest.json', dict(success=True, time_end=time.time(),
                storage_fixture_only=True, not_a_physical_measurement=True))
    manifest_hash = sha(fixture/'manifest.json')
    first = archive_run(fixture)
    saved = json.loads((fixture/'grt_intermediates_inventory.json').read_text())
    assert (fixture/'orfs_work/results').is_symlink()
    assert (fixture/'orfs_work/objects').is_symlink()
    assert sha(fixture/'manifest.json') == manifest_hash
    second = archive_run(fixture)
    assert second['archive_sha256'] == first['archive_sha256']
    verify_archive(fixture/'grt_intermediates.tar.zst', saved)
    bad = dict(saved)
    key = next(iter(bad))
    bad[key] = dict(bytes=bad[key]['bytes'], sha256='0'*64)
    try:
        verify_archive(fixture/'grt_intermediates.tar.zst', bad)
    except ValueError:
        rejected_bad_inventory = True
    else:
        raise AssertionError('Archive verification accepted corrupt inventory')
    real = ROOT/'backend/results/bp_fe_grt_v1_row0000'
    meta = json.loads((real/'manifest.json').read_text())
    measurement = verify_physical(real, dict(def_sha256=meta['def_sha256']), True)
    assert measurement['status']=='success'
    result = dict(status='passed', slurm_job_id=job, storage_fixture=str(fixture),
                  fixture_not_a_physical_result=True, repeated_archive_idempotent=True,
                  corrupt_inventory_rejected=rejected_bad_inventory,
                  original_manifest_unchanged=True, real_label_reparsed=measurement)
    output = Path('/fact_data/bohanli/flowplace_fql')/('graph_o2o_checks_'+job)
    output.mkdir(exist_ok=False)
    atomic_json(output/'report.json', result)
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
