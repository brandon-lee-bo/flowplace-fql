"""CPU contracts for old-parent/new-namespace export; no graph or EDA calls."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from . import qstudy_scalar_online_export as exporter
from . import qstudy_scalar_online as core
from . import qstudy_backend as backend
from .qstudy_data import action_sha, bind, sha


class ParentExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.slurm()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='scalar-export-',dir=os.environ['TMPDIR'])
        self.root=Path(self.tmp.name).resolve()
        self.source=self.root/'offline';self.source.mkdir()
        self.output=self.root/'new_physical'
        self.cfg=dict(design='bp_fe',dataset_path=str(self.root/'audited.npz'),dataset_sha256='d'*64,
            cache_sha256='e'*64,physical_failure_reward=-4.,rule_sources={})
        (self.source/'config.json').write_text(json.dumps(self.cfg))
        rule=core.screen_for('rule')
        spec=dict(version=rule.SPEC_VERSION,objective='grt_wl',cost_names=list(rule.COST_NAMES),
            weights=list(rule.OBJECTIVES['grt_wl']),scales=[10.]*6,
            reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64),
            rule_definition_sha256=rule.json_sha(rule.rule_definition()),success_cap=2.,failure_cost=4.)
        self.schema=core.schema('rule',spec)
        self.request=dict(version=exporter.VERSION,source_kind='rule',source_experiment=str(self.source),
            source_config_sha256=sha(self.source/'config.json'),parent_checkpoints={},arm='bridge',run_prefix='br1',
            streams={str(s):dict(collect=dict(seed=101+s,count=2),evaluation=dict(seed=201,count=2)) for s in (0,1,2)},
            evaluation_purpose='development_validation',backend_storage='scratch',output=str(self.output))
        for seed in (0,1,2):
            folder=self.source/f'seed_{seed}'/'round_0';folder.mkdir(parents=True)
            # Native request verification does not reinterpret old torch metadata.
            # These bytes deliberately belong to the OLD source namespace.
            (folder/'checkpoint.pt').write_bytes(('original-offline-checkpoint-seed'+str(seed)).encode())
            (folder/'completed.json').write_text(json.dumps(dict(status='completed',step=1000,seed=seed)))
            (folder/'binding.json').write_text(json.dumps(dict(old_config_sha256=self.request['source_config_sha256'],seed=seed)))
            self.request['parent_checkpoints'][str(seed)]=bind(folder/'checkpoint.pt')
        self.request_path=self.root/'request.json';self.write_request()
        self.initial={str(p):sha(p) for p in self.source.rglob('*') if p.is_file()}
        self.context_patch=mock.patch.object(exporter,'context',side_effect=self.context)
        self.context_patch.start()
        self.loader=mock.patch.object(core,'load_offline_source',side_effect=self.load_source).start()
        self.addCleanup(mock.patch.stopall)

    def tearDown(self):self.tmp.cleanup()

    def write_request(self):self.request_path.write_text(json.dumps(self.request))

    def context(self,experiment):
        config=json.loads((Path(experiment)/'config.json').read_text())
        physical=dict(config_sha256=sha(Path(experiment)/'config.json'),dataset_sha256=config['dataset_sha256'],
            reward_scale=config.get('reward_scale',100.),design=config['design'],physical_flow_revision='frozen-fixture')
        return config,{},physical

    def load_source(self,kind,experiment,seed,expected_sha):
        self.assertEqual(kind,'rule');self.assertEqual(Path(experiment),self.source)
        self.assertEqual(expected_sha,self.request['parent_checkpoints'][str(seed)]['sha256'])
        return dict(schema=self.schema,binding=dict(seed=seed,schema=self.schema,
            checkpoint=bind(self.source/f'seed_{seed}'/'round_0'/'checkpoint.pt'),
            old_config_sha256=self.request['source_config_sha256']))

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            exporter.prepare(self.request_path,sha(self.request_path),self.output)
        return exporter.settings(self.output,sha(self.output/'config.json'))

    def population(self):
        cfg,decl,physical=self.prepare()
        folder=self.output/'seed_0'/'round_0'/'requests';folder.mkdir(parents=True)
        rows=[]
        for role in exporter.ROLES:
            for index in range(2):
                sample=folder/cfg['arm']/role/str(index);sample.mkdir(parents=True)
                raw=np.full((3,2),.1,dtype=np.float32)
                noise=np.full((3,2),(.2 if role=='collect' else .5)+index*.01,dtype=np.float32)
                np.savez_compressed(sample/'actions.npz',noise=noise,raw_action=raw,action=raw)
                parent=cfg['parent_identities']['0']['checkpoint']
                row=dict(arm=cfg['arm'],graph=cfg['design'],seed=0,round=0,role=role,index=index,
                    terminal_mask=0,training_feedback_eligible=role=='collect',noise_seed=cfg['streams']['0'][role]['seed'],
                    source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],
                    config_sha256=physical['config_sha256'],dataset_sha256=cfg['dataset_sha256'],
                    source_config_sha256=cfg['source_config']['sha256'],
                    parent_identity_sha256=core.json_sha(cfg['verified_parents']['0']),
                    export_declaration_sha256=sha(self.output/'declaration.json'),
                    scalar_utility_spec_sha256=self.schema['utility_spec_sha256'],q_units=core.Q_UNITS,
                    predicted_q=[-1.23],evaluation_purpose=cfg['evaluation_purpose'],
                    sample_dir=str(sample),run_id=f"go2o_qs1_{cfg['run_prefix']}_s0_r0_{role}_{index}",
                    objective_weights=self.schema['utility_spec']['weights'],raw_action=raw.tolist(),executed_action=raw.tolist(),
                    raw_action_sha256=action_sha(raw),action_sha256=action_sha(raw),noise_sha256=action_sha(noise),
                    pre_backend_failure='normalized_geometry',geometry=dict(legal=False),stage='global_route_only')
                (sample/'request.json').write_text(json.dumps(row));rows.append(row)
        tasks=folder/'tasks.jsonl';self.seal(cfg,decl,physical,rows,tasks)
        return cfg,decl,physical,rows,tasks

    def seal(self,cfg,decl,physical,rows,tasks):
        tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        tasks.with_name('completed.json').write_text(json.dumps(exporter.completion(cfg,decl,physical,self.output,0,rows,tasks)))

    def test_prepare_is_metadata_only_immutable_and_preserves_original_files(self):
        cfg,decl,physical=self.prepare()
        self.assertEqual(self.loader.call_count,3)
        self.assertEqual(cfg['source_config']['sha256'],self.request['source_config_sha256'])
        self.assertNotEqual(physical['config_sha256'],self.request['source_config_sha256'])
        self.assertEqual(decl['schema'],self.schema)
        self.assertEqual(decl['backend_calls'],0)
        self.assertFalse(list(self.output.rglob('*.pt')))
        self.assertFalse(list(self.output.rglob('actions.npz')))
        before={str(p):sha(p) for p in self.output.rglob('*') if p.is_file()}
        self.prepare()
        self.assertEqual(before,{str(p):sha(p) for p in self.output.rglob('*') if p.is_file()})
        self.assertEqual(self.initial,{str(p):sha(p) for p in self.source.rglob('*') if p.is_file()})

    def test_explicit_noise_roles_source_kinds_and_paths_are_not_chosen(self):
        for change in ('overlap','duplicate_collect','missing_seed','bc','relative','same_namespace','zero_count'):
            bad=copy.deepcopy(self.request)
            if change=='overlap':bad['streams']['0']['collect']['seed']=201
            elif change=='duplicate_collect':bad['streams']['1']['collect']['seed']=101
            elif change=='missing_seed':del bad['streams']['2']
            elif change=='bc':bad['source_kind']='bc'
            elif change=='relative':bad['output']='relative'
            elif change=='same_namespace':bad['output']=str(self.source/'child')
            else:bad['streams']['0']['evaluation']['count']=0
            with self.subTest(change=change),self.assertRaises(ValueError):exporter.request_settings(bad)
        # Shared evaluation seed is explicit and permitted; streams are not auto-selected.
        self.assertEqual(exporter.request_settings(self.request)['streams'],self.request['streams'])

    def test_external_request_and_real_parent_binding_are_required(self):
        with self.assertRaises(ValueError):exporter.prepare(self.request_path,'0'*64,self.output)
        bad=copy.deepcopy(self.request);bad['parent_checkpoints']['0']['path']=str(self.root/'fake.pt')
        with self.assertRaises(ValueError):exporter.source_identities(bad)
        self.request['parent_checkpoints']['0']['sha256']='0'*64;self.write_request()
        with self.assertRaises(ValueError):exporter.prepare(self.request_path,sha(self.request_path),self.output)

    def test_changed_original_or_declaration_closure_refused(self):
        self.prepare()
        declaration=self.output/'declaration.json';old=declaration.read_text()
        changed=json.loads(old);changed['legalizer']['grid_res']=64;declaration.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):exporter.settings(self.output,sha(self.output/'config.json'))
        declaration.write_text(old)
        snapshot=self.output/'sources'/'fql/qstudy_scalar_online_export.py';snapshot.write_text('tampered')
        with self.assertRaises(ValueError):exporter.settings(self.output,sha(self.output/'config.json'))
        snapshot.write_bytes((exporter.ROOT/'fql/qstudy_scalar_online_export.py').read_bytes())
        (self.source/'seed_0/round_0/completed.json').write_text('{}')
        with self.assertRaises(ValueError):exporter.settings(self.output,sha(self.output/'config.json'))

    def test_real_backend_accepts_old_checkpoint_with_new_config_and_finite_scalar_q(self):
        cfg,decl,physical,rows,tasks=self.population()
        for row in exporter.verify_export(self.output,0,cfg,decl,physical):
            sample=backend.checked_request(row,type('Args',(),dict(seed=0,round=0))(),cfg,physical)
            self.assertEqual(sha(row['source_checkpoint']),self.request['parent_checkpoints']['0']['sha256'])
            self.assertEqual(row['predicted_q'],[-1.23])
            self.assertTrue((sample/'request.json').is_file())
        # Real generic validator accepts this identity bridge without touching the old bytes.
        self.assertEqual(self.initial,{str(p):sha(p) for p in self.source.rglob('*') if p.is_file()})

    def test_scalar_collect_admission_keeps_native_reward_and_excludes_evaluation(self):
        cfg,decl,physical,rows,tasks=self.population()
        for row in rows[:2]:
            sample=Path(row['sample_dir'])
            out=dict(row,status='geometry_failure',metrics=None,reward=-4.,
                request_sha256=sha(sample/'request.json'),tasks_sha256=sha(tasks))
            (sample/'outcome.json').write_text(json.dumps(out))
        with mock.patch.object(backend,'context',side_effect=self.context):
            replay=core.admit_collect(self.output,seed=0,source_round=0,count=2,tasks_sha256=sha(tasks),
                config_sha256=physical['config_sha256'],checkpoint_sha256=rows[0]['checkpoint_sha256'],
                collection_noise_seed=101,evaluation_noise_sha256=[r['noise_sha256'] for r in rows[2:]],
                value=self.schema,cache={})
        self.assertTrue(replay['ready']);self.assertEqual(len(replay['records']),2)
        self.assertEqual(replay['derived_B'].flatten().tolist(),[4.,4.])
        self.assertEqual([r['raw_native_reward'] for r in replay['records']],[-4.,-4.])
        self.assertTrue(all(not r['success'] and r['scalar_target_valid'] for r in replay['records']))

    def test_actual_noise_config_checkpoint_and_q_tampering_refused(self):
        cfg,decl,physical,rows,tasks=self.population()
        for key,value in [('source_config_sha256','0'*64),('training_feedback_eligible',False),
                          ('checkpoint_sha256','0'*64),('noise_seed',999),('predicted_q',[float('nan')]),
                          ('predicted_q',None),('q_units','negative native GRT reward'),('run_id','foreign_namespace')]:
            bad=dict(rows[0]);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                exporter.check_row(bad,cfg,decl,physical,self.output,0,'collect',0)
        sample=Path(rows[0]['sample_dir'])
        with np.load(sample/'actions.npz',allow_pickle=False) as arrays:
            values={k:arrays[k] for k in arrays.files}
        values['noise']=values['noise']+.01;np.savez_compressed(sample/'actions.npz',**values)
        with self.assertRaises(ValueError):exporter.verify_export(self.output,0,cfg,decl,physical)

    def test_missing_reordered_duplicate_noises_and_changed_marker_refused(self):
        cfg,decl,physical,rows,tasks=self.population()
        self.seal(cfg,decl,physical,rows[:-1],tasks)
        with self.assertRaises(ValueError):exporter.verify_export(self.output,0,cfg,decl,physical)
        self.seal(cfg,decl,physical,rows[::-1],tasks)
        with self.assertRaises(ValueError):exporter.verify_export(self.output,0,cfg,decl,physical)
        rows[1]['noise_sha256']=rows[0]['noise_sha256']
        # Rebind actual saved noise as well: semantic duplication must still fail.
        first=Path(rows[0]['sample_dir'])/'actions.npz';second=Path(rows[1]['sample_dir'])/'actions.npz'
        second.write_bytes(first.read_bytes())
        (Path(rows[1]['sample_dir'])/'request.json').write_text(json.dumps(rows[1]))
        self.seal(cfg,decl,physical,rows,tasks)
        with self.assertRaisesRegex(ValueError,'duplicated or overlap'):exporter.verify_export(self.output,0,cfg,decl,physical)

    def test_completed_export_reentry_uses_saved_artifacts_without_gpu_forward(self):
        cfg,decl,physical,rows,tasks=self.population()
        with mock.patch.object(exporter.torch.cuda,'is_available',return_value=False),\
             mock.patch.object(exporter,'GraphPolicy',side_effect=AssertionError('no model load')):
            self.assertEqual(exporter.export(self.output,physical['config_sha256'],0),rows)
        marker=tasks.with_name('completed.json');value=json.loads(marker.read_text());value['actor_updates']=1
        marker.write_text(json.dumps(value))
        with self.assertRaises(ValueError):exporter.verify_export(self.output,0,cfg,decl,physical)


if __name__=='__main__':unittest.main()
