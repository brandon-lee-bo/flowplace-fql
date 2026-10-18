"""Small TRAIN-DRT software contracts; no model or physical tool invocation."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_train_final_route as q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows_fixture():
    return [dict(split='train', family=f, variant=v,
        metrics={'rWL_um':1000.+f*3+v}) for f in range(192) for v in range(2)]


class TrainRouteTests(unittest.TestCase):
    def test_selection_balanced_deterministic_and_family_cv_disjoint(self):
        rows = rows_fixture()
        selected = q.select_families(rows)
        self.assertEqual(selected, q.select_families(list(reversed(rows))))
        self.assertEqual(len(selected), 32)
        self.assertEqual(len({r['family'] for r in selected}), 32)
        for quartile in range(4):
            chosen = [r for r in selected if r['quartile'] == quartile]
            self.assertEqual(len(chosen), 8)
            self.assertEqual({fold:sum(r['fold'] == fold for r in chosen) for fold in range(4)}, {0:2,1:2,2:2,3:2})
            self.assertTrue(all(quartile*48 <= r['family'] < (quartile+1)*48 for r in chosen))
        for fold in range(4):
            held = {r['family'] for r in selected if r['fold'] == fold}
            train = {r['family'] for r in selected if r['fold'] != fold}
            self.assertEqual(len(held), 8)
            self.assertFalse(held & train)

    def test_exact_ties_and_excluding_validation_are_explicit(self):
        rows = rows_fixture()
        for r in rows:
            r['metrics']['rWL_um'] = 4.
        chosen = q.select_families(rows)
        self.assertTrue(all(r['family_mean_GRT_WL_um'] == 4 for r in chosen))
        self.assertTrue(all(r['quartile']*48 <= r['family'] < (r['quartile']+1)*48 for r in chosen))
        rows[-1]['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'TRAIN rows only'):
            q.select_families(rows)

    def test_pair_or_nonfinite_label_cannot_be_replaced(self):
        rows = rows_fixture()
        with self.assertRaisesRegex(ValueError, 'both original variants'):
            q.select_families(rows[:-1])
        rows[-1]['metrics']['rWL_um'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid TRAIN-only'):
            q.select_families(rows)

    def native_fixture(self, root):
        run = root/'native';run.mkdir()
        defp=root/'input.def';defp.write_text('frozen DEF')
        metrics = {key:(1000. if key=='rWL_um' else -.2) for key in q.METRIC_NAMES}
        write(run/'metrics.json',metrics)
        manifest=dict(success=True,time_end=1,design='bp_fe',stage='global_route_only',num_cores=8,
            physical_flow_revision=q.PHYSICAL_REVISIONS['bp_fe'],def_sha256=q.sha(defp),def_path=str(defp),
            tool_image_sha256='a'*64, design_config=str(root/'designs/nangate45/bp_fe_top/config_eval.mk'))
        write(run/'manifest.json',manifest)
        geometry=dict(bindings={'physical_manifest_sha256':q.sha(run/'manifest.json')},legal_no_halo=True)
        write(run/'backend_macro_audit.json',geometry)
        row=dict(metrics=metrics,bindings={
            'physical_manifest':q.bind(run/'manifest.json'),'physical_metrics':q.bind(run/'metrics.json'),
            'input_def':q.bind(defp),'outcome':None,'request':None},
            actual_backend_geometry=q.bind(run/'backend_macro_audit.json'))
        for name in q.CHECKPOINT_NAMES:
            p=run/(q.RESULT_ROOT+name);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('source'+name).encode())
        for kind in ('logs','reports'):
            p=run/f'orfs_work/{kind}/nangate45/bp_fe/eval/native.log'
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text('native '+kind)
        return row, q.native_source(row)

    def test_native_source_rejects_changed_label_def_threads_and_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            row,native=self.native_fixture(Path(temp))
            self.assertEqual(q.native_source(row),native)
            changed=copy.deepcopy(row);changed['metrics']['rWL_um']+=1
            with self.assertRaisesRegex(ValueError,'does not bind'):
                q.native_source(changed)
            Path(row['bindings']['input_def']['path']).write_text('changed DEF')
            with self.assertRaisesRegex(ValueError,'Changed TRAIN source'):
                q.native_source(row)

    def test_config_parent_is_not_the_recorded_result_nickname(self):
        with tempfile.TemporaryDirectory() as temp:
            row,native=self.native_fixture(Path(temp))
            self.assertEqual(Path(q.read(Path(native['run'])/'manifest.json')['design_config']).parent.name,'bp_fe_top')
            checkpoint=q.checkpoint_binding(native)
            self.assertTrue(all(q.RESULT_ROOT+n in checkpoint['files'] for n in q.CHECKPOINT_NAMES))
            self.assertFalse(any('/bp_fe_top/' in name for name in checkpoint['files']))

    def test_full64_declaration_is_idempotent_and_rejects_plan_source_family_tampering(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,SLURM_JOB_ID='123'):
            root=Path(temp);dataset=root/'dataset.npz';dataset.write_bytes(b'no arrays loaded in fixture')
            write(root/'manifest.json',{'fixture':True})
            (root/'rows.jsonl').write_text('fixture\n')
            rows=rows_fixture();manifest={'rows_sha256':q.sha(root/'rows.jsonl')}
            output=root/'new_study';snapshot=root/'source.py';snapshot.write_text('frozen source')
            image=root/'image.sif';image.write_bytes(b'bound physical tool fixture')
            source_map={'source.py':q.sha(snapshot)}
            with patch.object(q,'DATASET_SHA',q.sha(dataset)),patch.object(q,'ROOT',root), \
                 patch.object(q,'CONTAINER',image), \
                 patch.object(q,'SCRATCH_ROOT',root),patch.object(q,'sources',return_value=source_map), \
                 patch.object(q,'load_train',return_value=(rows,manifest)), \
                 patch.object(q,'native_source',return_value={'run':'original','tool_image_sha256':q.sha(image)}), \
                 patch.object(q,'checkpoint_binding',return_value={'restore_required':False}):
                plan=q.prepare(dataset,output)
                self.assertEqual(len(plan['tasks']),64)
                self.assertEqual(len({t['output'] for t in plan['tasks']}),64)
                plan_path=output/'plan.json';plan_bytes=plan_path.read_bytes();plan_sha=q.sha(plan_path)
                self.assertEqual(q.checked_plan(plan_path,plan_sha),plan)
                with patch.dict(os.environ,SLURM_JOB_ID='124'):
                    self.assertEqual(q.prepare(dataset,output),plan)
                image.write_bytes(b'changed container')
                with self.assertRaisesRegex(ValueError,'Changed TRAIN source'):
                    q.checked_plan(plan_path,plan_sha)
                image.write_bytes(b'bound physical tool fixture')
                changed=copy.deepcopy(plan);changed['tasks'][0]['checkpoint']['archive']='different ODB source'
                write(plan_path,changed)
                with self.assertRaisesRegex(ValueError,'unpinned TRAIN-DRT plan SHA'):
                    q.checked_plan(plan_path,plan_sha)
                changed=copy.deepcopy(plan);changed['selection']['folds']=8
                write(output/'plan.json',changed)
                with self.assertRaisesRegex(ValueError,'selection or folds changed'):
                    q.checked_plan(plan_path,q.sha(plan_path))
                changed=copy.deepcopy(plan);changed['tasks'][0]['provenance']['split']='validation'
                write(output/'plan.json',changed)
                with self.assertRaisesRegex(ValueError,'task identity'):
                    q.checked_plan(plan_path,q.sha(plan_path))
                plan_path.write_bytes(plan_bytes)
                (output/'sources/source.py').write_text('tampered snapshot')
                with self.assertRaisesRegex(ValueError,'Changed TRAIN source'):
                    q.checked_plan(plan_path,plan_sha)

    def make_archive(self, run, files, *, legacy=False):
        archive=run/'grt_intermediates.tar.zst'
        raw=run/'fixture.tar'
        with tarfile.open(raw,'w') as tar:
            for relative,payload in files.items():
                info=tarfile.TarInfo(relative);info.size=len(payload)
                tar.addfile(info,io.BytesIO(payload))
        subprocess.run(['zstd','-q','-f',str(raw),'-o',str(archive)],check=True)
        inventory={name:dict(bytes=len(data),sha256=q.hashlib.sha256(data).hexdigest()) for name,data in files.items()}
        record=dict(archive_path=str(archive),archive_sha256=q.sha(archive),
            physical_manifest_sha256=q.sha(run/'manifest.json'),decoded_full_content_verified=True,file_count=len(inventory))
        write(run/'grt_intermediates_inventory.json',inventory)
        write(run/'grt_artifact_archive.json',record)
        if legacy:
            compact=run/'grt_intermediates.long28.tar.zst';archive.rename(compact)
            write(run/'legacy_archive_repack.json',dict(status='durable_compact_archive_and_exact_original_cache',
                original_archive_sha256=record['archive_sha256'],physical_manifest_sha256=record['physical_manifest_sha256'],
                inventory_sha256=q.sha(run/'grt_intermediates_inventory.json'),
                archive_record_sha256=q.sha(run/'grt_artifact_archive.json'),compact_path=str(compact),compact_sha256=q.sha(compact)))
            archive.symlink_to(run/'evicted-original-cache.tar.zst')
        return inventory

    def test_missing_cache_restores_verified_compact_without_old_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,SLURM_JOB_ID='123'):
            root=Path(temp);row,native=self.native_fixture(root)
            run=Path(native['run']);relative=q.RESULT_ROOT+'5_1_grt.odb'
            payload=(run/relative).read_bytes()
            self.make_archive(run,{relative:payload},legacy=True)
            (run/relative).unlink()
            cp=q.checkpoint_binding(native)
            self.assertEqual(cp['missing_at_declaration'],[relative])
            self.assertIsNotNone(cp['archive']['legacy_repack'])
            output=root/'experiment';output.mkdir();write(output/'plan.json',{'immutable':'plan'})
            task=dict(key='f001_v0',provenance=row,native=native,checkpoint=cp)
            before={p.name:q.sha(p) for p in (run/'manifest.json',run/'metrics.json',run/'backend_macro_audit.json')}
            restored=q.ready_source({'output':str(output)},task)
            self.assertNotEqual(restored,run)
            self.assertEqual((restored/relative).read_bytes(),payload)
            self.assertFalse((run/relative).exists())
            self.assertEqual(q.ready_source({'output':str(output)},task),restored)
            self.assertEqual(before,{p.name:q.sha(p) for p in (run/'manifest.json',run/'metrics.json',run/'backend_macro_audit.json')})
            (restored/relative).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Private restored checkpoint'):
                q.ready_source({'output':str(output)},task)

    def test_full_decoded_inventory_rejects_hidden_corruption_and_path_traversal(self):
        for unsafe in (False,True):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);row,native=self.native_fixture(root);run=Path(native['run'])
                files={'wanted':b'right',('../escape' if unsafe else 'other'):b'also checked'}
                inventory=self.make_archive(run,files)
                if not unsafe:
                    inventory['other']['sha256']='0'*64
                target=root/'restore';target.mkdir()
                with self.assertRaisesRegex(ValueError,'Unsafe|Full decoded'):
                    q.decode_required(q.bind(run/'grt_intermediates.tar.zst'),inventory,['wanted'],target)
                self.assertFalse((root/'escape').exists())

    def test_live_changed_checkpoint_cannot_use_archive_to_hide_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            row,native=self.native_fixture(Path(temp));run=Path(native['run'])
            relative=q.RESULT_ROOT+'5_1_grt.odb'
            self.make_archive(run,{relative:b'wrong-old-checkpoint'})
            with self.assertRaisesRegex(ValueError,'Live checkpoint differs'):
                q.checkpoint_binding(native)

    def worker_fixture(self, root):
        folder=root/'full_route/f001_v0'
        task=dict(index=0,key='f001_v0',output=str(folder),run_id='go2o_train_drt64_f001_v0',method='measured_original_TRAIN_DRT64')
        plan={'tasks':[task]*64,'output':str(root)}
        p=root/'plan.json';write(p,{'immutable':'plan'})
        return plan,p,folder

    def test_completed_worker_reuses_validated_result_without_restore_or_EDA(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,SLURM_JOB_ID='123',SLURM_CPUS_PER_TASK='8'):
            plan,p,folder=self.worker_fixture(Path(temp));write(folder/'outcome.json',{})
            with patch.object(q,'checked_plan',return_value=plan),patch.object(q,'inspect_result',return_value={'state':'completed'}) as inspect, \
                 patch.object(q,'ready_source') as restore,patch.object(q.subprocess,'run') as backend:
                self.assertEqual(q.worker(p,0,q.sha(p)),{'state':'completed'})
                inspect.assert_called_once();restore.assert_not_called();backend.assert_not_called()

    def test_exact_original_worker_command_and_no_fake_exit_success(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,SLURM_JOB_ID='123',SLURM_CPUS_PER_TASK='8',SLURM_RESTART_COUNT='7'):
            root=Path(temp);plan,p,folder=self.worker_fixture(root)
            with patch.object(q,'checked_plan',return_value=plan),patch.object(q,'prior_attempts_safe'), \
                 patch.object(q,'ready_source',return_value=root/'source'),patch.object(q,'sources',return_value={}), \
                 patch.object(q.subprocess,'run') as backend,patch.object(q,'inspect_result',return_value={'state':'missing'}):
                with self.assertRaisesRegex(RuntimeError,'without a validated native'):
                    q.worker(p,0,q.sha(p))
            args,kwargs=backend.call_args
            cmd=args[0]
            self.assertEqual(cmd[cmd.index('-m')+1],'fql.graph_o2o_full_route')
            self.assertEqual(cmd[cmd.index('--end-iter')+1],'20')
            self.assertEqual(cmd[cmd.index('--storage')+1],'scratch')
            self.assertEqual(cmd[cmd.index('--source-run')+1],str(root/'source'))
            self.assertNotIn('SLURM_RESTART_COUNT',kwargs['env'])

    def test_true_failed_live_and_unknown_attempts_refuse_but_preemption_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write(root/'attempt_1/binding.json',dict(physical_run=str(root/'native'),slurm_job_id='11'))
            for state in ('FAILED','RUNNING','ACCOUNTING_PENDING','TIMEOUT'):
                with self.subTest(state=state),patch.object(q,'states',return_value=[state]):
                    with self.assertRaisesRegex(RuntimeError,'requires diagnosis'):
                        q.prior_attempts_safe(root)
            with patch.object(q,'states',return_value=['PREEMPTED']):
                q.prior_attempts_safe(root)
            write(root/'native/manifest.json',dict(success=True,time_end=1))
            with patch.object(q,'states',side_effect=AssertionError('completed native should recover reporting')):
                q.prior_attempts_safe(root)

    def test_failed_attempt_is_retained_without_inventing_final_label(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,SLURM_JOB_ID='123'):
            root=Path(temp);plan,p,folder=self.worker_fixture(root)
            plan['selection']={'selected_families':[]}
            plan['tasks']=[dict(index=i,key=str(i),output=str(root/'full_route'/str(i))) for i in range(64)]
            failed=Path(plan['tasks'][0]['output'])
            write(failed/'attempt_11/binding.json',dict(slurm_job_id='11',physical_run=str(root/'native')))
            write(failed/'attempt_11/exit.json',dict(returncode=1))
            write(root/'native/manifest.json',dict(success=False,time_end=2,error='tool error requiring diagnosis'))
            with patch.object(q,'checked_plan',return_value=plan),patch.object(q,'sources',return_value={}):
                result=q.report(p,root/'reports/result.json',q.sha(p))
            self.assertEqual(result['requested'],64)
            self.assertEqual(result['counts'],{'unresolved_attempt_failure':1,'missing':63})
            self.assertIsNone(result['records'][0]['metrics'])
            self.assertEqual(result['records'][0]['attempts'][0]['returncode'],1)
            self.assertFalse(result['records'][0]['attempts'][0]['native_success'])

    def test_wrong_cpu_count_and_array_index_fail_before_backend(self):
        with patch.dict(os.environ,SLURM_CPUS_PER_TASK='2'):
            with self.assertRaisesRegex(RuntimeError,'exactly8'):
                q.worker(Path('not_read'),0,'unread')
        with patch.dict(os.environ,SLURM_CPUS_PER_TASK='8'),patch.object(q,'checked_plan',return_value={'tasks':[]}):
            with self.assertRaises(IndexError):
                q.worker(Path('not_read'),64,'unread')

    def test_array_dispatch_requires_external_plan_digest(self):
        argv=['train_drt','--stage','worker','--plan','/declared/plan.json','--plan-sha256','a'*64]
        with patch.dict(os.environ,SLURM_JOB_ID='123',SLURM_ARRAY_TASK_ID='63'), \
             patch.object(q.sys,'argv',argv),patch.object(q,'worker',return_value={'state':'completed'}) as worker, \
             patch('builtins.print'):
            q.main()
            worker.assert_called_once_with(Path('/declared/plan.json'),63,'a'*64)
        with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(q.sys,'argv',argv[:-2]), \
             patch.object(q,'worker') as worker,patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):
                q.main()
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
