"""Small filesystem/archive fixtures only; CPU Slurm, never real EDA outputs."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import qstudy_preserve_native as p


class PreservationTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run fixtures under bounded CPU Slurm')
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.base = Path(self.temp.name)
        self.repo, self.bulk, self.native, self.scratch = [self.base / name for name in ('repo', 'bulk', 'native', 'scratch')]
        self.qroot = self.bulk / 'qstudy_v1'
        for path in (self.repo, self.qroot, self.native, self.scratch):
            path.mkdir(parents=True)
        self.patches = [patch.object(p, key, value) for key, value in dict(ROOT=self.repo,
            BULK=self.bulk, NATIVE=self.native, SCRATCH=self.scratch, QROOT=self.qroot, QUOTA=self.bulk).items()]
        for item in self.patches:
            item.start()
        for name in p.SOURCE_NAMES:
            self.file(self.repo / name, ('sealed fixture ' + name).encode())
        self.sample = self.qroot / 'experiment/seed_0/round_0/requests/evaluation/0'
        self.sample.mkdir(parents=True)
        raw = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
        noise = -raw
        np.savez(self.sample / 'actions.npz', raw_action=raw, action=raw, noise=noise)
        digest = lambda a: hashlib.sha256(a.astype('<f4').reshape(-1).tobytes()).hexdigest()
        self.row = dict(raw_action=raw.tolist(), executed_action=raw.tolist(),
            raw_action_sha256=digest(raw), action_sha256=digest(raw), noise_sha256=digest(noise))
        self.js(self.sample / 'request.json', self.row)
        self.outcome = dict(status='geometry_failure', metrics=None,
            request_sha256=p.sha(self.sample / 'request.json'))
        self.js(self.sample / 'outcome.json', self.outcome)
        self.record = dict(id='fixture:0', stage='GRT', state='geometry_failure',
            request=p.binding(self.sample / 'request.json'), outcome=p.binding(self.sample / 'outcome.json'))
        self.plan_path = self.qroot / 'preservation_plan.json'
        self.anchor = self.qroot / 'sealed_inventory.json'
        self.output = self.scratch / 'qstudy_native_preservation/staged'
        self.plan = dict(version=p.VERSION, scope_reason='explicit fixture only',
            durable_output=str(self.qroot / 'native_evidence_preservation_v1/batch'), records=[self.record], attempts=[],
            extra_files=[], historical_references=[], historical_runs=[],
            source_sha256={name:p.sha(self.repo / name) for name in p.SOURCE_NAMES})
        self.save()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def file(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def js(self, path, value):
        return self.file(path, (json.dumps(value, sort_keys=True) + '\n').encode())

    def save(self):
        self.js(self.anchor, dict(records=self.plan['records'], attempts=self.plan['attempts']))
        self.plan['anchors'] = [p.binding(self.anchor)]
        self.js(self.plan_path, self.plan)
        return p.sha(self.plan_path)

    def staged(self):
        return p.stage(self.plan_path, self.save(), self.output)

    def budget(self, staged):
        path = self.qroot / 'reviewed_budget.json'
        budget = dict(version=p.VERSION, stage_sha256=p.sha(self.output / 'stage.json'),
            durable_output=self.plan['durable_output'], review_reason='fixture capacity reviewed',
            max_archive_bytes=staged['archive_bytes'], max_sidecar_bytes=1024**2,
            preservation_reserve_bytes=65536, checkpoint_reserve_bytes=100000,
            atomic_writer_reserve_bytes=50000)
        self.js(path, budget)
        return path, budget

    def test_scratch_stage_roundtrip_no_durable_copy_no_source_links_or_changes(self):
        before = {path:p.sha(path) for path in self.sample.iterdir()}
        staged = self.staged()
        self.assertEqual(staged['status'], 'scratch_staged_fully_decoded_not_durable')
        self.assertFalse(Path(self.plan['durable_output']).exists())
        self.assertFalse(staged['EDA_resume_archive'])
        self.assertEqual(staged['state_counts'], {'geometry_failure':1})
        self.assertEqual(staged['uncompressed_tar_bytes'], len(subprocess.check_output(
            ['zstd','-dc','--long=28',str(self.output/'native_evidence.tar.zst')])))
        self.assertEqual(staged['compression']['level'],3)
        self.assertEqual(before, {path:p.sha(path) for path in self.sample.iterdir()})
        self.assertTrue(all(not path.is_symlink() for path in self.sample.iterdir()))
        original = (self.output / 'stage.json').read_bytes()
        p.stage(self.plan_path, p.sha(self.plan_path), self.output)
        self.assertEqual(original, (self.output / 'stage.json').read_bytes())

    def test_alias_deduplicates_bytes_but_repointing_is_refused(self):
        evidence = p.Evidence()
        source = self.sample / 'request.json'
        alias = self.scratch / 'alias.json';alias.symlink_to(source)
        evidence.add(source);evidence.add(alias)
        self.assertEqual(len(evidence.files), 1)
        self.assertEqual(len(evidence.aliases), 2)
        alias.unlink();alias.symlink_to(self.sample / 'outcome.json')
        with self.assertRaisesRegex(ValueError, 'alias changed'):
            evidence.recheck()

    def test_scope_and_report_binding_and_source_closure_refuse_tamper(self):
        unrelated = self.file(self.base / 'unrelated.json', b'{}')
        alias = self.scratch / 'outside.json';alias.symlink_to(unrelated)
        with self.assertRaisesRegex(ValueError, 'scope'):
            p.allowed_file(alias)
        digest = self.save()
        self.js(self.anchor, dict(records=[]))
        with self.assertRaisesRegex(ValueError, 'Changed bound'):
            p.collect(self.plan_path, digest)
        self.save()
        (self.repo / p.SOURCE_NAMES[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'closure changed'):
            p.collect(self.plan_path, p.sha(self.plan_path))

    def test_exact_fresh_export_path_digest_mapping_is_supported_not_approximate(self):
        bindings = {item['path']:item['sha256'] for item in (self.record['request'], self.record['outcome'])}
        self.js(self.anchor, dict(export_inventory=bindings))
        self.plan['anchors']=[p.binding(self.anchor)];self.js(self.plan_path,self.plan)
        _, evidence=p.collect(self.plan_path,p.sha(self.plan_path))
        self.assertTrue(any(name.endswith('/outcome.json') for name in evidence.files))
        bindings[self.record['request']['path']]='0'*64
        self.js(self.anchor,dict(export_inventory=bindings))
        self.plan['anchors']=[p.binding(self.anchor)];self.js(self.plan_path,self.plan)
        with self.assertRaisesRegex(ValueError,'not declared'):
            p.collect(self.plan_path,p.sha(self.plan_path))

    def test_undeclared_outcome_and_array_mutation_refused(self):
        digest = self.save()
        self.js(self.anchor, dict(records=[]))
        self.plan['anchors'] = [p.binding(self.anchor)];self.js(self.plan_path, self.plan)
        with self.assertRaisesRegex(ValueError, 'not declared'):
            p.collect(self.plan_path, p.sha(self.plan_path))
        self.save()
        with np.load(self.sample / 'actions.npz') as arrays:
            saved = {name:arrays[name].copy() for name in arrays.files}
        saved['noise'][0,0] += 1
        np.savez(self.sample / 'actions.npz', **saved)
        with self.assertRaisesRegex(ValueError, 'noise SHA'):
            p.collect(self.plan_path, p.sha(self.plan_path))

    def test_changed_source_during_full_decode_prevents_stage_commit(self):
        original = p.verify_archive
        def mutate(*args):
            original(*args)
            (self.sample / 'actions.npz').write_bytes(b'changed externally')
        with patch.object(p, 'verify_archive', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                self.staged()
        self.assertFalse((self.output / 'stage.json').exists())
        self.assertFalse(Path(self.plan['durable_output']).exists())

    def test_tar_members_traversal_links_duplicates_and_wrong_hash_rejected(self):
        for name in ('../escape', 'files/a/../../escape', '/files/a/b', 'files//a/b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.safe_member(name)
        for kind in ('link', 'duplicate', 'wrong_hash'):
            path = self.scratch / (kind + '.tar.zst')
            buffer = io.BytesIO()
            member = 'files/a/proof'
            with tarfile.open(fileobj=buffer, mode='w') as archive:
                item = tarfile.TarInfo(member)
                if kind == 'link':
                    item.type = tarfile.SYMTYPE;item.linkname = '/outside';archive.addfile(item)
                else:
                    item.size = 1;archive.addfile(item, io.BytesIO(b'x'))
                    if kind == 'duplicate':archive.addfile(item, io.BytesIO(b'x'))
            with path.open('wb') as target:
                subprocess.run(['zstd', '-c'], input=buffer.getvalue(), stdout=target, check=True)
            expected = {member:dict(bytes=1, sha256=hashlib.sha256(b'y' if kind == 'wrong_hash' else b'x').hexdigest())}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                p.verify_archive(path, expected)

    def test_capacity_known_exact_bytes_reserves_and_limits(self):
        budget = dict(max_archive_bytes=100, max_sidecar_bytes=20,
            preservation_reserve_bytes=10, checkpoint_reserve_bytes=30, atomic_writer_reserve_bytes=40)
        self.assertEqual(p.capacity(dict(filesystem_free_bytes=200, quota_remaining_bytes=200),100,20,budget), 200)
        for storage in (dict(filesystem_free_bytes=199,quota_remaining_bytes=200),
                dict(filesystem_free_bytes=200,quota_remaining_bytes=199),
                dict(filesystem_free_bytes=200,quota_remaining_bytes=None)):
            with self.assertRaises(RuntimeError):p.capacity(storage,100,20,budget)
        with self.assertRaises(RuntimeError):p.capacity(dict(filesystem_free_bytes=1000,quota_remaining_bytes=1000),101,20,budget)
        with self.assertRaises(ValueError):p.capacity({},0,0,dict(budget,preservation_reserve_bytes=0))

    def test_admit_requires_budget_checks_twice_atomic_verified_and_idempotent(self):
        staged = self.staged();path,budget = self.budget(staged)
        with patch.object(p, 'storage_status', return_value=dict(filesystem_free_bytes=10**8,quota_remaining_bytes=10**8)) as storage:
            saved = p.admit(self.output, p.sha(self.output / 'stage.json'), path, p.sha(path))
            self.assertEqual(storage.call_count,2)
            output = Path(self.plan['durable_output'])
            self.assertEqual(saved['archive_sha256'],p.sha(output / 'native_evidence.tar.zst'))
            self.assertFalse(saved['EDA_resume_archive'])
            self.assertTrue(saved['global_EDA_storage_guards_unchanged'])
            p.admit(self.output,p.sha(self.output / 'stage.json'),path,p.sha(path))
            self.assertEqual(storage.call_count,2)
        self.js(output / 'audits.json', dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'sidecar changed'):
            p.admit(self.output,p.sha(self.output / 'stage.json'),path,p.sha(path))

    def test_capacity_refusal_creates_no_durable_output_or_false_completion(self):
        staged = self.staged();path,budget = self.budget(staged)
        with patch.object(p,'storage_status',return_value=dict(filesystem_free_bytes=1,quota_remaining_bytes=1)):
            with self.assertRaises(RuntimeError):p.admit(self.output,p.sha(self.output/'stage.json'),path,p.sha(path))
        self.assertFalse(Path(self.plan['durable_output']).exists())
        budget['stage_sha256'] = '0'*64;self.js(path,budget)
        with self.assertRaisesRegex(ValueError,'exact-stage budget'):
            p.admit(self.output,p.sha(self.output/'stage.json'),path,p.sha(path))

    def test_stage_sidecar_change_after_initial_verify_never_publishes_durable_manifest(self):
        staged=self.staged();path,_=self.budget(staged)
        stage_sha=p.sha(self.output/'stage.json')
        for name in ('inventory.json','stage.json'):
            source=self.output/name;original=source.read_bytes();calls=[]
            def capacity_with_mutation(*args):
                if not calls:source.write_bytes(b'{"externally_changed":true}\n')
                calls.append(True)
                return dict(filesystem_free_bytes=10**8,quota_remaining_bytes=10**8)
            with self.subTest(sidecar=name),patch.object(p,'storage_status',side_effect=capacity_with_mutation):
                with self.assertRaisesRegex(ValueError,'sidecar changed'):
                    p.admit(self.output,stage_sha,path,p.sha(path))
            self.assertFalse((Path(self.plan['durable_output'])/'manifest.json').exists())
            source.write_bytes(original)

    def test_durable_root_alias_to_scratch_is_not_durable_admission(self):
        staged=self.staged();path,_=self.budget(staged)
        destination_root=Path(self.plan['durable_output']).parent
        with self.assertRaisesRegex(ValueError,'dedicated directory'):
            p.durable_destination(destination_root/'..'/'escaped')
        destination_root.symlink_to(self.scratch,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'actual project bulk|alias'):
            p.admit(self.output,p.sha(self.output/'stage.json'),path,p.sha(path))
        self.assertFalse((self.scratch/'batch').exists())

    def test_committed_durable_archive_and_sidecars_cannot_be_scratch_symlinks(self):
        staged=self.staged();path,_=self.budget(staged)
        with patch.object(p,'storage_status',return_value=dict(filesystem_free_bytes=10**8,quota_remaining_bytes=10**8)):
            p.admit(self.output,p.sha(self.output/'stage.json'),path,p.sha(path))
        output=Path(self.plan['durable_output'])
        for name in ('native_evidence.tar.zst','inventory.json','manifest.json','admission.json'):
            source=output/name;temporary=self.scratch/('aliased_'+name)
            source.rename(temporary);source.symlink_to(temporary)
            with self.subTest(payload=name),self.assertRaisesRegex(ValueError,'non-symlink'):
                p.admit(self.output,p.sha(self.output/'stage.json'),path,p.sha(path))
            source.unlink();temporary.rename(source)

    def test_interrupted_copy_resume_retains_partial_and_never_rewrites_sources(self):
        staged = self.staged();path,budget = self.budget(staged)
        def interrupted(raw,target,**kwargs):
            target.write(raw.read(7))
            raise RuntimeError('fixture preemption')
        capacity = dict(filesystem_free_bytes=10**8,quota_remaining_bytes=10**8)
        with patch.object(p,'storage_status',return_value=capacity), patch.object(p.shutil,'copyfileobj',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'fixture preemption'):
                p.admit(self.output,p.sha(self.output/'stage.json'),path,p.sha(path))
        output=Path(self.plan['durable_output'])
        partial=next(output.glob('native_evidence.pending_*'))
        self.assertEqual(partial.stat().st_size,7)
        self.assertFalse((output/'manifest.json').exists())
        before=partial.read_bytes()
        with patch.object(p,'storage_status',return_value=capacity), patch.dict(os.environ,SLURM_JOB_ID='fixture_recovery'):
            p.admit(self.output,p.sha(self.output/'stage.json'),path,p.sha(path))
        self.assertEqual(partial.read_bytes(),before)
        self.assertEqual(p.sha(output/'native_evidence.tar.zst'),staged['archive_sha256'])

    def test_historical_durable_reference_coverage_without_duplicate_payload(self):
        old_manifest = self.native/'go2o_historical/manifest.json'
        self.js(old_manifest,dict(stage='global_route_only',design='bp_fe'))
        self.plan['extra_files']=[p.binding(old_manifest)]
        staged=self.staged()
        old=self.bulk/'old_preservation';old.mkdir()
        for source,target in (('native_evidence.tar.zst','native_evidence.tar.zst'),
                ('inventory.json','inventory.json'),('aliases.json','path_aliases.json')):
            (old/target).write_bytes((self.output/source).read_bytes())
        self.js(old/'manifest.json',dict(status='native_evidence_durably_preserved_and_fully_decoded',
            archive_sha256=staged['archive_sha256'],archive_bytes=staged['archive_bytes'],
            inventory_sha256=p.sha(old/'inventory.json'),aliases_sha256=p.sha(old/'path_aliases.json')))
        self.plan['extra_files']=[]
        self.plan['historical_references']=[p.binding(old/'manifest.json')]
        self.plan['historical_runs']=[p.binding(old_manifest)]
        second=self.scratch/'qstudy_native_preservation/second'
        current=p.stage(self.plan_path,self.save(),second)
        inventory=p.read(second/'inventory.json')
        self.assertEqual(current['historical_unique_manifest_references'],1)
        self.assertFalse(any(name.endswith('/go2o_historical/manifest.json') for name in inventory))
        self.assertFalse(any(name.endswith('/old_preservation/native_evidence.tar.zst') for name in inventory))
        self.plan['historical_runs'][0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'lacks a durable inventory'):
            p.collect(self.plan_path,self.save())

    def test_final_physical_proof_keeps_DEF_and_DRC_logs_without_OpenDB(self):
        run=self.scratch/'physical_results/go2o_final'
        self.js(run/'manifest.json',dict(stage='finish_after_detailed_route',design='bp_fe',success=True,time_end=1))
        self.file(run/'orfs_work/logs/nangate45/bp_fe/eval/5_2_route.log',b'Number of violations = 0.')
        self.file(run/'orfs_work/reports/nangate45/bp_fe/eval/6_finish.rpt',b'final power area and timing source')
        final=self.file(run/'orfs_work/results/nangate45/bp_fe/eval/6_final.def',b'final fixture DEF')
        self.file(final.with_suffix('.odb'),b'excluded large database')
        evidence=p.Evidence();evidence.physical(run,'bp_fe',final=True)
        self.assertTrue(any(name.endswith('/6_final.def') for name in evidence.files))
        self.assertTrue(any(name.endswith('/5_2_route.log') for name in evidence.files))
        self.assertFalse(any(name.endswith('.odb') for name in evidence.files))

    def test_confirmed_failure_retained_missing_not_fabricated_and_attempt_proof(self):
        missing = self.sample.parent / '1';missing.mkdir()
        self.js(missing/'request.json',self.row)
        (missing/'actions.npz').write_bytes((self.sample/'actions.npz').read_bytes())
        self.plan['records'].append(dict(id='fixture:missing',stage='GRT',state='missing',request=p.binding(missing/'request.json')))
        attempt = self.sample/'backend_attempts/old/attempt.json'
        self.js(attempt,dict(physical_run=str(self.scratch/'never_started'),request_sha256=self.record['request']['sha256']))
        self.file(attempt.with_name('runner.log'),b'infrastructure interruption, no PPA')
        self.plan['attempts'] = [dict(binding=p.binding(attempt))]
        staged = self.staged()
        self.assertEqual(staged['state_counts'],dict(geometry_failure=1,missing=1))
        self.assertEqual(staged['logical_records'],2)
        inventory = p.read(self.output/'inventory.json')
        self.assertTrue(any(name.endswith('/runner.log') for name in inventory))
        self.assertFalse(any(name.endswith('/1/outcome.json') for name in inventory))

    def test_real_seven_metric_reparse_and_native_compact_inventory(self):
        run = self.scratch/'physical_results/go2o_fixture';run.mkdir(parents=True)
        imported = self.file(self.sample/'input.def',b'fixture imported DEF')
        self.js(imported.with_suffix('.audit.json'),dict(output_sha256=p.sha(imported)))
        self.row.update(def_path=str(imported),def_sha256=p.sha(imported));self.js(self.sample/'request.json',self.row)
        manifest = dict(success=True,stage='global_route_only',design='bp_fe',time_end=10,
            num_cores=8,physical_flow_revision='fixture',def_path=str(imported),def_sha256=p.sha(imported))
        self.js(run/'manifest.json',manifest)
        logs = run/'orfs_work/logs/nangate45/bp_fe/eval';reports=run/'orfs_work/reports/nangate45/bp_fe/eval'
        raw = {'globalroute__global_route__wirelength':2000,'globalroute__timing__setup__tns':-10,
            'globalroute__timing__setup__ws':-.2,'globalroute__power__total':.1,
            'globalroute__design__instance__area__stdcell':3000}
        self.js(logs/'5_grt_label.json',raw)
        self.file(reports/'5_fql_global_route_label.rpt',b'actual fixture timing source')
        self.js(reports/'fql_grt_marker_status.json',dict(category_present=False,marker_count=0,evidence_source='fixture ODB marker export'))
        self.file(run/'orfs_work/results/nangate45/bp_fe/eval/5_1_grt.odb',b'not an actual OpenDB; never read by a tool')
        values = p.parse_grt_metrics(run,1);self.js(run/'metrics.json',values)
        self.file(run/'backend_macro_audit.log',b'fixture native macro export')
        self.js(run/'backend_macro_audit.json',dict(legal_no_halo=True,bindings=dict(
            physical_manifest_sha256=p.sha(run/'manifest.json'),input_def_sha256=p.sha(imported)),
            output_sha256=p.sha(run/'backend_macro_audit.log')))
        self.outcome = dict(status='success',metrics=values,request_sha256=p.sha(self.sample/'request.json'),
            physical_run=str(run),physical_manifest_sha256=p.sha(run/'manifest.json'),
            physical_flow_revision='fixture',metrics_sha256=p.sha(run/'metrics.json'),
            backend_macro_geometry_sha256=p.sha(run/'backend_macro_audit.json'))
        self.js(self.sample/'outcome.json',self.outcome)
        self.record.update(state='success',request=p.binding(self.sample/'request.json'),outcome=p.binding(self.sample/'outcome.json'))
        staged = self.staged()
        self.assertEqual(len(staged['unique_physical_runs']),1)
        inventory=p.read(self.output/'inventory.json')
        self.assertTrue(any(name.endswith('/fql_grt_marker_status.json') for name in inventory))
        self.assertFalse(any(name.endswith('.odb') for name in inventory))
        self.assertTrue(p.read(self.output/'audits.json')[0]['audit']['native_metrics_reparsed'])
        values['TNS_ns']=-999;self.js(run/'metrics.json',values);self.outcome['metrics']=values
        self.outcome['metrics_sha256']=p.sha(run/'metrics.json');self.js(self.sample/'outcome.json',self.outcome)
        self.record['outcome']=p.binding(self.sample/'outcome.json');self.save()
        with self.assertRaisesRegex(ValueError,'native source reparse'):
            p.collect(self.plan_path,p.sha(self.plan_path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
