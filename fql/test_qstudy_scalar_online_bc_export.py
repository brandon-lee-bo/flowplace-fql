"""Bounded synthetic dual-source physical-contract tests; no inference or EDA."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import qstudy_scalar_online_bc_export as exporter
from . import qstudy_scalar_online as core
from . import qstudy_backend as backend
from .qstudy_data import action_sha, bind, sha


class BCExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):core.slurm()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='scalar-bc-export-',dir=os.environ['TMPDIR'])
        self.root=Path(self.temp.name).resolve()
        self.bc=self.root/'bc';self.scalar=self.root/'scalar';self.output=self.root/'physical'
        self.bc.mkdir();self.scalar.mkdir()
        base=dict(design='bp_fe',dataset_path=str(self.root/'audited.npz'),dataset_sha256='d'*64,
            cache_sha256='e'*64,physical_failure_reward=-4.)
        self.write(self.bc/'config.json',dict(base,arm='bc',q_weight=0.))
        self.write(self.scalar/'config.json',dict(base,arm='rule_grt_wl',q_weight=1.,rule_sources={}))
        rule=core.screen_for('rule')
        spec=dict(version=rule.SPEC_VERSION,objective='grt_wl',cost_names=list(rule.COST_NAMES),
            weights=list(rule.OBJECTIVES['grt_wl']),scales=[10.]*6,
            reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64),
            rule_definition_sha256=rule.json_sha(rule.rule_definition()),success_cap=2.,failure_cost=4.)
        self.schema=core.schema('rule',spec)
        self.request=dict(version=exporter.VERSION,scalar_kind='rule',bc_experiment=str(self.bc),
            bc_config_sha256=sha(self.bc/'config.json'),bc_parent_checkpoints={},
            scalar_experiment=str(self.scalar),scalar_config_sha256=sha(self.scalar/'config.json'),
            scalar_parent_checkpoints={},arm='bcs',run_prefix='bcs1',
            streams={str(s):dict(collect=dict(seed=101+s,count=2),evaluation=dict(seed=201,count=2)) for s in (0,1,2)},
            evaluation_purpose='development_validation',backend_storage='scratch',output=str(self.output))
        self.sources={}
        for seed in (0,1,2):
            items={}
            for kind,experiment in (('bc',self.bc),('scalar',self.scalar)):
                folder=experiment/f'seed_{seed}'/'round_0';folder.mkdir(parents=True)
                (folder/'checkpoint.pt').write_bytes((kind+' real original prefix '+str(seed)).encode())
                self.write(folder/'completed.json',dict(status='completed',steps=1000))
                self.write(folder/'binding.json',dict(seed=seed,source_sha256={}))
                item=dict(config=bind(experiment/'config.json'),checkpoint=bind(folder/'checkpoint.pt'),
                    completed=bind(folder/'completed.json'),training_binding=bind(folder/'binding.json'))
                self.request[kind+'_parent_checkpoints'][str(seed)]=item['checkpoint'];items[kind]=item
            refit=self.root/('scalar_refit_'+str(seed)+'.pt');refit.write_bytes(b'refit scalar model')
            self.sources[seed]=dict(schema=self.schema,binding=dict(version=exporter.provider.VERSION,
                schema=self.schema,**items['bc'],target_scalar_source=items['scalar'],critic_refit=bind(refit),
                assembled_payload_sha256=str(seed+1)*64,source_sha256=sha(exporter.provider.__file__)))
        self.request_path=self.root/'request.json';self.write(self.request_path,self.request)
        self.initial={str(p):sha(p) for experiment in (self.bc,self.scalar) for p in experiment.rglob('*') if p.is_file()}
        self.loader_patch=mock.patch.object(exporter,'load_source',side_effect=lambda request,seed:copy.deepcopy(self.sources[seed]))
        self.loader=self.loader_patch.start()
        mock.patch.object(exporter,'context',side_effect=self.context).start()
        self.addCleanup(mock.patch.stopall)

    def tearDown(self):self.temp.cleanup()

    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value))

    def context(self,experiment):
        cfg=json.loads((Path(experiment)/'config.json').read_text())
        return cfg,{},dict(config_sha256=sha(Path(experiment)/'config.json'),design=cfg['design'],
            dataset_sha256=cfg['dataset_sha256'],reward_scale=cfg.get('reward_scale',100.),
            physical_flow_revision='fixed-physical-fixture')

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
                noise=np.full((3,2),(.2 if role=='collect' else .5)+.01*index,dtype=np.float32)
                np.savez_compressed(sample/'actions.npz',noise=noise,raw_action=raw,action=raw)
                bound=cfg['verified_parents']['0'];parent=bound['checkpoint']
                row=dict(arm=cfg['arm'],graph=cfg['design'],seed=0,round=0,role=role,index=index,
                    terminal_mask=0,training_feedback_eligible=role=='collect',noise_seed=cfg['streams']['0'][role]['seed'],
                    source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],
                    config_sha256=physical['config_sha256'],dataset_sha256=cfg['dataset_sha256'],
                    source_config_sha256=cfg['source_config']['sha256'],parent_identity_sha256=core.json_sha(bound),
                    export_declaration_sha256=sha(self.output/'declaration.json'),
                    scalar_utility_spec_sha256=self.schema['utility_spec_sha256'],q_units=core.Q_UNITS,
                    actor_source_kind='genuine_bc_prefix',q_weight=0.,
                    scalar_source_config_sha256=cfg['scalar_config']['sha256'],
                    scalar_source_checkpoint_sha256=cfg['scalar_parent_identities']['0']['checkpoint']['sha256'],
                    scalar_critic_refit=bound['critic_refit'],assembled_payload_sha256=bound['assembled_payload_sha256'],
                    predicted_q=[-1.23],evaluation_purpose=cfg['evaluation_purpose'],sample_dir=str(sample),
                    run_id=f"go2o_qs1_{cfg['run_prefix']}_s0_r0_{role}_{index}",
                    objective_weights=self.schema['utility_spec']['weights'],raw_action=raw.tolist(),executed_action=raw.tolist(),
                    raw_action_sha256=action_sha(raw),action_sha256=action_sha(raw),noise_sha256=action_sha(noise),
                    pre_backend_failure='normalized_geometry',geometry=dict(legal=False),stage='global_route_only')
                self.write(sample/'request.json',row);rows.append(row)
        tasks=folder/'tasks.jsonl';self.seal(cfg,decl,physical,rows,tasks)
        return cfg,decl,physical,rows,tasks

    def seal(self,cfg,decl,physical,rows,tasks):
        tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.write(tasks.with_name('completed.json'),exporter.completion(cfg,decl,physical,self.output,0,rows,tasks))

    def test_metadata_prepare_binds_both_sources_without_rewriting_or_inference(self):
        cfg,decl,_=self.prepare()
        self.assertEqual(self.loader.call_count,3)
        self.assertEqual(cfg['q_weight'],0.)
        self.assertEqual(cfg['source_config'],bind(self.bc/'config.json'))
        self.assertEqual(cfg['scalar_config'],bind(self.scalar/'config.json'))
        self.assertEqual(decl['actor_updates'],0);self.assertEqual(decl['q_selection'],False)
        self.assertFalse(list(self.output.rglob('*.pt')));self.assertFalse(list(self.output.rglob('actions.npz')))
        before={str(p):sha(p) for p in self.output.rglob('*') if p.is_file()}
        self.prepare()
        self.assertEqual(before,{str(p):sha(p) for p in self.output.rglob('*') if p.is_file()})
        self.assertEqual(self.initial,{str(p):sha(p) for e in (self.bc,self.scalar) for p in e.rglob('*') if p.is_file()})

    def test_two_external_configs_checkpoints_and_namespaces_are_required(self):
        for kind in ('bc','scalar'):
            bad=copy.deepcopy(self.request);bad[kind+'_config_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):exporter.source_identities(bad)
            bad=copy.deepcopy(self.request);bad[kind+'_parent_checkpoints']['0']['path']=str(self.root/'foreign.pt')
            with self.subTest(kind=kind),self.assertRaises(ValueError):exporter.source_identities(bad)
        with self.assertRaises(ValueError):exporter.prepare(self.request_path,'0'*64,self.output)

    def test_streams_and_output_are_explicit_not_automatically_selected(self):
        for change in ('overlap','duplicate','missing_seed','same_source','same_output','kind','zero'):
            bad=copy.deepcopy(self.request)
            if change=='overlap':bad['streams']['0']['collect']['seed']=201
            elif change=='duplicate':bad['streams']['1']['collect']['seed']=101
            elif change=='missing_seed':del bad['streams']['2']
            elif change=='same_source':bad['scalar_experiment']=str(self.bc)
            elif change=='same_output':bad['output']=str(self.bc/'child')
            elif change=='kind':bad['scalar_kind']='bc'
            else:bad['streams']['0']['evaluation']['count']=0
            with self.subTest(change=change),self.assertRaises(ValueError):exporter.request_settings(bad)

    def test_provider_receives_independent_external_identities(self):
        self.loader_patch.stop()
        with mock.patch.object(exporter.provider,'load_bc_source',return_value='verified') as loader:
            self.assertEqual(exporter.load_source(self.request,1),'verified')
            loader.assert_called_once_with('rule',str(self.bc),str(self.scalar),1,
                expected_bc_config_sha256=self.request['bc_config_sha256'],
                expected_bc_checkpoint_sha256=self.request['bc_parent_checkpoints']['1']['sha256'],
                expected_scalar_config_sha256=self.request['scalar_config_sha256'],
                expected_scalar_checkpoint_sha256=self.request['scalar_parent_checkpoints']['1']['sha256'])

    def test_changed_original_and_snapshot_and_declaration_refused(self):
        self.prepare();decl=self.output/'declaration.json';original=decl.read_bytes()
        value=json.loads(original);value['q_weight']=1.;self.write(decl,value)
        with self.assertRaises(ValueError):exporter.settings(self.output,sha(self.output/'config.json'))
        decl.write_bytes(original)
        snapshot=self.output/'sources'/'fql/qstudy_scalar_online_bc.py';snapshot.write_bytes(b'changed')
        with self.assertRaises(ValueError):exporter.settings(self.output,sha(self.output/'config.json'))
        snapshot.write_bytes(Path(exporter.provider.__file__).read_bytes())
        (self.scalar/'seed_0/round_0/completed.json').write_text('{}')
        with self.assertRaises(ValueError):exporter.settings(self.output,sha(self.output/'config.json'))

    def test_real_backend_contract_uses_genuine_bc_parent_not_scalar_checkpoint(self):
        cfg,decl,physical,rows,_=self.population()
        self.assertNotEqual(rows[0]['checkpoint_sha256'],rows[0]['scalar_source_checkpoint_sha256'])
        for row in exporter.verify_export(self.output,0,cfg,decl,physical):
            sample=backend.checked_request(row,type('Args',(),dict(seed=0,round=0))(),cfg,physical)
            self.assertEqual(row['checkpoint_sha256'],self.request['bc_parent_checkpoints']['0']['sha256'])
            self.assertEqual(sample,Path(row['sample_dir']))
        self.assertEqual(self.initial,{str(p):sha(p) for e in (self.bc,self.scalar) for p in e.rglob('*') if p.is_file()})

    def test_attached_refit_bytes_cannot_change_after_declaration(self):
        self.prepare()
        Path(self.sources[0]['binding']['critic_refit']['path']).write_bytes(b'changed scalar refit')
        with self.assertRaises(ValueError):exporter.settings(self.output,sha(self.output/'config.json'))

    def test_actual_core_collect_admission_preserves_native_reward_excludes_eval(self):
        cfg,decl,physical,rows,tasks=self.population()
        for row in rows[:2]:
            sample=Path(row['sample_dir']);out=dict(row,status='geometry_failure',metrics=None,reward=-4.,
                request_sha256=sha(sample/'request.json'),tasks_sha256=sha(tasks))
            self.write(sample/'outcome.json',out)
        with mock.patch.object(backend,'context',side_effect=self.context):
            replay=core.admit_collect(self.output,seed=0,source_round=0,count=2,tasks_sha256=sha(tasks),
                config_sha256=physical['config_sha256'],checkpoint_sha256=rows[0]['checkpoint_sha256'],
                collection_noise_seed=101,evaluation_noise_sha256=[r['noise_sha256'] for r in rows[2:]],
                value=self.schema,cache={})
        self.assertTrue(replay['ready']);self.assertEqual(len(replay['records']),2)
        self.assertEqual(replay['derived_B'].flatten().tolist(),[4.,4.])
        self.assertEqual([r['raw_native_reward'] for r in replay['records']],[-4.,-4.])
        self.assertEqual([r['request'] for r in replay['records']],
            [bind(Path(row['sample_dir'])/'request.json') for row in rows[:2]])

    def test_actor_scalar_roles_units_and_actual_noise_tampering_refused(self):
        cfg,decl,physical,rows,_=self.population()
        for key,value in [('source_checkpoint',rows[0]['scalar_source_checkpoint_sha256']),('q_weight',1.),
            ('actor_source_kind','scalar_trained_bc'),('scalar_source_checkpoint_sha256','0'*64),
            ('scalar_critic_refit',{}),('assembled_payload_sha256','0'*64),('training_feedback_eligible',False),
            ('predicted_q',[float('nan')]),('q_units','legacy reward'),('run_id','foreign')]:
            bad=dict(rows[0]);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                exporter.check_row(bad,cfg,decl,physical,self.output,0,'collect',0)
        with self.assertRaises(ValueError):exporter.check_row(rows[0],cfg,decl,physical,self.output,0,'collect',0,np.zeros((3,2)))

    def test_incomplete_reordered_duplicate_and_marker_exports_refused(self):
        cfg,decl,physical,rows,tasks=self.population()
        for changed in (rows[:-1],rows[::-1]):
            self.seal(cfg,decl,physical,changed,tasks)
            with self.assertRaises(ValueError):exporter.verify_export(self.output,0,cfg,decl,physical)
        self.seal(cfg,decl,physical,rows,tasks)
        marker=tasks.with_name('completed.json');value=json.loads(marker.read_text());value['evaluation_never_replayed']=False
        self.write(marker,value)
        with self.assertRaises(ValueError):exporter.verify_export(self.output,0,cfg,decl,physical)
        rows[1]['noise_sha256']=rows[0]['noise_sha256']
        (Path(rows[1]['sample_dir'])/'actions.npz').write_bytes((Path(rows[0]['sample_dir'])/'actions.npz').read_bytes())
        self.write(Path(rows[1]['sample_dir'])/'request.json',rows[1]);self.seal(cfg,decl,physical,rows,tasks)
        with self.assertRaisesRegex(ValueError,'duplicate/overlap'):exporter.verify_export(self.output,0,cfg,decl,physical)

    def test_completed_reentry_reads_only_saved_proofs_without_provider_or_gpu(self):
        cfg,decl,physical,rows,_=self.population()
        self.loader.side_effect=AssertionError('no reassembly/model loading')
        with mock.patch.object(exporter.torch.cuda,'is_available',side_effect=AssertionError('no CUDA')):
            self.assertEqual(exporter.export(self.output,physical['config_sha256'],0),rows)

    def test_reuse_does_not_mutate_frozen_parent_globals_or_source(self):
        before=exporter.parent.VERSION,exporter.parent.export.__globals__,sha(exporter.parent.__file__)
        self.population()
        self.assertEqual(exporter.parent.VERSION,before[0])
        self.assertIs(exporter.parent.export.__globals__,before[1])
        self.assertEqual(sha(exporter.parent.__file__),before[2])


if __name__=='__main__':unittest.main()
