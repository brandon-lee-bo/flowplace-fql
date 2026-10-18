"""Bounded actual-state/native contracts; no graph forward or physical call."""
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
import torch

from . import qstudy_scalar_online_bc_child_export as child
from . import qstudy_scalar_online as core
from . import qstudy_scalar_online_bc_train as training
from . import qstudy_scalar_online_train as common
from . import test_qstudy_scalar_online_bc as bc_fixture
from . import qstudy_backend as backend
from . import test_qstudy_scalar_online as tiny
from .qstudy_data import action_sha, bind, sha


class ChildExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.slurm();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.fixture=tiny.ScalarOnlineTests(methodName='test_prefix_optimizer_rng_and_named_states_restored_exactly')
        self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        self.root=self.fixture.root.resolve();self.output=self.root/'child'
        self.source=self.root/'old_physical';self.source.mkdir()
        self.original=self.root/'original_offline';self.original.mkdir()
        self.cfg=dict(design='bp_fe',dataset_path=str(self.root/'dataset.npz'),dataset_sha256='d'*64,
            cache_sha256='e'*64,physical_failure_reward=-4.,reward_scale=100.,
            scalar_schema=self.fixture.schema,source_experiment=str(self.original),
            scalar_experiment=str(self.root/'scalar_original'),alpha=10.,
            streams={str(s):dict(collect=dict(seed=100+s,count=1),evaluation=dict(seed=200,count=1)) for s in child.SEEDS})
        (self.source/'config.json').write_text(json.dumps(self.cfg));(self.source/'declaration.json').write_text('{}')
        self.request=dict(version=child.VERSION,logical_round=1,parent_training={},arm='continued',run_prefix='child1',
            streams={str(s):dict(collect=dict(seed=300+s,count=2),evaluation=dict(seed=400,count=2)) for s in child.SEEDS},
            evaluation_purpose='development_validation',backend_storage='scratch',output=str(self.output))
        self.results={}
        for seed in child.SEEDS:
            folder=self.root/f'train_{seed}';folder.mkdir()
            for name in ('checkpoint.pt','completed.json','binding.json','learning_curve.json'):
                (folder/name).write_text(json.dumps(dict(seed=seed,name=name)))
            train_request=dict(version=training.VERSION,experiment=str(self.source),config_sha256=sha(self.source/'config.json'),
                declaration_sha256=sha(self.source/'declaration.json'),seed=seed,tasks_sha256='f'*64,final_outcomes={},
                settings=dict(self.fixture.settings,q_weight=0.,update_critic=False),diagnostic_interval=2,output=str(folder))
            request_path=self.root/f'train_request_{seed}.json';request_path.write_text(json.dumps(train_request))
            parent=dict(request=bind(request_path),checkpoint=bind(folder/'checkpoint.pt'))
            self.request['parent_training'][str(seed)]=parent
            original_cp=self.original/f'bc_{seed}.pt';original_cp.write_text('real BC prefix')
            scalar_refit=self.original/f'scalar_{seed}.pt';scalar_refit.write_text('separate scalar refit')
            source=dict(version=child.provider.VERSION,actor_prefix=dict(checkpoint=bind(original_cp)),
                target_scalar_source=dict(seed=seed,kind='rule'),critic_refit=bind(scalar_refit))
            identity=dict(training_request=parent['request'],checkpoint=parent['checkpoint'],
                completed=bind(folder/'completed.json'),runner_binding=bind(folder/'binding.json'),
                learning_curve=bind(folder/'learning_curve.json'),parent_export_config=bind(self.source/'config.json'),
                parent_export_declaration=bind(self.source/'declaration.json'),offline_source=source,bc_actor_prefix=source['actor_prefix'],
                scalar_source=source['target_scalar_source'],scalar_critic_refit=source['critic_refit'],
                continuation_binding=dict(seed=seed,settings=train_request['settings']),state_audit=dict(verified=True),
                previous_noise_sha256=dict(collect=['1'*64],evaluation=['2'*64]))
            self.results[seed]=dict(cfg=self.cfg,source=dict(schema=self.fixture.schema),identity=identity,
                declaration=dict(source_sha256={}))
        self.request_path=self.root/'request.json';self.write_request()
        self.initial={str(p):sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(child,'context',side_effect=self.context).start()
        self.loader=mock.patch.object(child,'checked_parent',side_effect=lambda p,s:self.results[s]).start()

    def write_request(self):self.request_path.write_text(json.dumps(self.request))

    def context(self,experiment):
        cfg=json.loads((Path(experiment)/'config.json').read_text())
        return cfg,{},dict(config_sha256=sha(Path(experiment)/'config.json'),dataset_sha256=cfg['dataset_sha256'],
            reward_scale=cfg['reward_scale'],design=cfg['design'],physical_flow_revision='frozen-fixture')

    def prepare(self):
        with redirect_stdout(io.StringIO()):child.prepare(self.request_path,sha(self.request_path),self.output)
        return child.settings(self.output,sha(self.output/'config.json'))

    def population(self):
        cfg,decl,physical=self.prepare();folder=self.output/'seed_0/round_1/requests';folder.mkdir(parents=True)
        rows=[]
        for role in child.ROLES:
            for index in range(2):
                sample=folder/cfg['arm']/role/str(index);sample.mkdir(parents=True)
                raw=np.full((3,2),.1,dtype=np.float32)
                noise=np.full((3,2),(.25 if role=='collect' else .65)+.01*index,dtype=np.float32)
                np.savez_compressed(sample/'actions.npz',noise=noise,raw_action=raw,action=raw)
                parent=cfg['parent_training']['0']['checkpoint'];identity=cfg['verified_children']['0']
                row=dict(arm=cfg['arm'],graph=cfg['design'],seed=0,round=1,role=role,index=index,
                    terminal_mask=0,training_feedback_eligible=role=='collect',noise_seed=cfg['streams']['0'][role]['seed'],
                    source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],config_sha256=physical['config_sha256'],
                    dataset_sha256=cfg['dataset_sha256'],source_config_sha256=identity['parent_export_config']['sha256'],
                    continuation_identity_sha256=core.json_sha(identity),training_request_sha256=identity['training_request']['sha256'],
                    export_declaration_sha256=sha(self.output/'declaration.json'),
                    scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],q_units=core.Q_UNITS,
                    predicted_q=[-1.25],evaluation_purpose=cfg['evaluation_purpose'],sample_dir=str(sample),
                    actor_source_kind='genuine_bc_continuation',q_weight=0.,
                    critic_source_kind='separately_bound_frozen_scalar_refit',
                    original_bc_checkpoint=identity['bc_actor_prefix']['checkpoint'],scalar_critic_refit=identity['scalar_critic_refit'],
                    run_id=f"go2o_qsbc_{cfg['run_prefix']}_s0_r1_{role}_{index}",objective_weights=cfg['scalar_schema']['utility_spec']['weights'],
                    raw_action=raw.tolist(),executed_action=raw.tolist(),raw_action_sha256=action_sha(raw),
                    action_sha256=action_sha(raw),noise_sha256=action_sha(noise),
                    pre_backend_failure='normalized_geometry',geometry=dict(legal=False),stage='global_route_only')
                (sample/'request.json').write_text(json.dumps(row));rows.append(row)
        tasks=folder/'tasks.jsonl';self.seal(cfg,rows,tasks)
        return cfg,decl,physical,rows,tasks

    def seal(self,cfg,rows,tasks):
        tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        tasks.with_name('completed.json').write_text(json.dumps(child.completion(cfg,self.output,0,rows,tasks)))

    def test_prepare_three_real_parent_artifact_bindings_is_immutable_and_metadata_only(self):
        cfg,decl,physical=self.prepare()
        self.assertEqual(self.loader.call_count,3)
        self.assertEqual(cfg['logical_round'],1);self.assertEqual(decl['backend_calls'],0)
        self.assertEqual(cfg['parent_training'],self.request['parent_training'])
        self.assertFalse(list(self.output.rglob('*.pt')));self.assertFalse(list(self.output.rglob('actions.npz')))
        before={str(p):sha(p) for p in self.output.rglob('*') if p.is_file()};self.prepare()
        self.assertEqual(before,{str(p):sha(p) for p in self.output.rglob('*') if p.is_file()})
        self.assertEqual(self.initial,{p:sha(p) for p in self.initial})
        self.assertNotEqual(physical['config_sha256'],sha(self.source/'config.json'))

    def test_request_rejects_implicit_round_role_alias_and_namespace(self):
        for change in ('round','missing','duplicate_collect','overlap','relative','zero_count','bc_shape','fql_version'):
            bad=copy.deepcopy(self.request)
            if change=='round':bad['logical_round']=0
            elif change=='missing':del bad['parent_training']['2']
            elif change=='duplicate_collect':bad['streams']['1']['collect']['seed']=300
            elif change=='overlap':bad['streams']['0']['collect']['seed']=400
            elif change=='relative':bad['output']='relative'
            elif change=='zero_count':bad['streams']['0']['collect']['count']=0
            elif change=='fql_version':bad['version']=child.state_validation.VERSION
            else:bad['parent_training']['0']['actor_prefix']={}
            with self.subTest(change=change),self.assertRaises(ValueError):child.request_settings(bad)
        self.request['streams']['0']['collect']['seed']=100;self.write_request()
        with self.assertRaisesRegex(ValueError,'previous'):self.prepare()

    def test_external_request_changed_continuation_and_sources_refused(self):
        with self.assertRaises(ValueError):child.prepare(self.request_path,'0'*64,self.output)
        self.prepare();configsha=sha(self.output/'config.json')
        path=self.output/'sources/fql/qstudy_scalar_online_child_export.py';original=path.read_bytes();path.write_text('changed')
        with self.assertRaises(ValueError):child.settings(self.output,configsha)
        path.write_bytes(original)
        parent=Path(self.request['parent_training']['0']['checkpoint']['path']);parent.write_text('different')
        with self.assertRaises(ValueError):child.settings(self.output,configsha)

    def test_real_native_checked_request_accepts_actual_child_checkpoint_round_one(self):
        cfg,decl,physical,rows,tasks=self.population()
        actual=child.verify_export(self.output,0,cfg,decl,physical)
        for row in actual:
            backend.checked_request(row,type('Args',(),dict(seed=0,round=1))(),cfg,physical)
            self.assertEqual(row['source_checkpoint'],self.request['parent_training']['0']['checkpoint']['path'])
            self.assertNotEqual(row['source_checkpoint'],str(self.original/'seed_0/round_0/checkpoint.pt'))
        self.assertEqual(actual,rows)

    def test_real_scalar_admit_round_one_keeps_failure_b4_and_excludes_evaluation(self):
        cfg,decl,physical,rows,tasks=self.population()
        for row in rows[:2]:
            sample=Path(row['sample_dir'])
            (sample/'outcome.json').write_text(json.dumps(dict(row,status='geometry_failure',metrics=None,reward=-4.,
                request_sha256=sha(sample/'request.json'),tasks_sha256=sha(tasks))))
        kwargs=dict(seed=0,source_round=1,count=2,tasks_sha256=sha(tasks),config_sha256=physical['config_sha256'],
            checkpoint_sha256=rows[0]['checkpoint_sha256'],collection_noise_seed=300,
            evaluation_noise_sha256=[r['noise_sha256'] for r in rows[2:]],value=self.fixture.schema,cache=self.fixture.cache)
        with mock.patch.object(backend,'context',side_effect=self.context):replay=core.admit_collect(self.output,**kwargs)
        self.assertTrue(replay['ready']);self.assertEqual(replay['derived_B'].flatten().tolist(),[4.,4.])
        self.assertEqual([r['raw_native_reward'] for r in replay['records']],[-4.,-4.])
        self.assertEqual(replay['binding']['source_round'],1)
        with mock.patch.object(backend,'context',side_effect=self.context),self.assertRaisesRegex(ValueError,'denominator'):
            core.admit_collect(self.output,**dict(kwargs,count=4))

    def test_row_rejects_forged_round_parent_noise_q_or_training_role(self):
        cfg,decl,physical,rows,tasks=self.population()
        for key,value in [('round',0),('checkpoint_sha256','0'*64),('training_request_sha256','0'*64),
                          ('training_feedback_eligible',False),('noise_seed',999),('predicted_q',None),
                          ('q_units','negative GRT reward'),('continuation_identity_sha256','0'*64),
                          ('actor_source_kind','FQL'),('q_weight',1.),('original_bc_checkpoint',{}),
                          ('scalar_critic_refit',{})]:
            bad=dict(rows[0]);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):child.check_row(bad,cfg,decl,physical,self.output,0,'collect',0)
        path=Path(rows[0]['sample_dir'])/'actions.npz'
        with np.load(path,allow_pickle=False) as arrays:values={k:arrays[k] for k in arrays.files}
        values['noise']+=.1;np.savez_compressed(path,**values)
        with self.assertRaises(ValueError):child.verify_export(self.output,0,cfg,decl,physical)

    def test_completed_reentry_no_model_or_gpu_and_changed_marker_refused(self):
        cfg,decl,physical,rows,tasks=self.population()
        with mock.patch.object(child.torch.cuda,'is_available',return_value=False),\
             mock.patch.object(child,'GraphPolicy',side_effect=AssertionError('no model forward')):
            self.assertEqual(child.export(self.output,physical['config_sha256'],0),rows)
        path=tasks.with_name('completed.json');value=json.loads(path.read_text());value['logical_round']=0
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):child.verify_export(self.output,0,cfg,decl,physical)

    def test_missing_reordered_population_or_previous_observed_noise_refused(self):
        cfg,decl,physical,rows,tasks=self.population()
        for subset in (rows[:-1],rows[::-1]):
            self.seal(cfg,subset,tasks)
            with self.assertRaises(ValueError):child.verify_export(self.output,0,cfg,decl,physical)
        cfg['verified_children']['0']['previous_noise_sha256']['evaluation'].append(rows[0]['noise_sha256'])
        with self.assertRaises(ValueError):child.check_row(rows[0],cfg,decl,physical,self.output,0,'collect',0)


class GenuineBCStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bc_fixture.BCSourceTests.setUpClass()

    def setUp(self):
        self.fixture=bc_fixture.BCSourceTests(methodName='test_real_bc_identity_and_all_preserved_states_with_two_source_binding')
        self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        self.source=self.fixture.load()
        self.settings=dict(total_updates=3,batch_size=16,alpha=10.,q_weight=0.,flow_steps=10,
            update_critic=False,huber_delta=1.,grad_clip=1.)
        self.continuation,self.replay=self.fixture.make_core(self.source,self.settings)
        for _ in range(self.settings['total_updates']):
            self.continuation.update(self.fixture.offline,self.replay,
                **{k:v for k,v in self.settings.items() if k!='total_updates'})
        self.payload=self.continuation.state();self.binding=self.continuation.binding

    def test_genuine_bc_actor_adam_continues_but_separate_scalar_and_unused_rng_do_not(self):
        before=copy.deepcopy(self.source['payload'])
        result=child.validate_payload(self.payload,self.source,self.binding)
        self.assertEqual(result['optimizer_audit']['actor']['actual_steps'],[1003.])
        self.assertEqual(result['optimizer_audit']['critic']['parameter_states'],0)
        self.assertEqual(result['continued_critic_updates'],0)
        self.assertEqual(result['actor_source_kind'],'genuine_bc_continuation')
        self.assertFalse(torch.equal(self.payload['actor']['linear.bias'],before['actor']['linear.bias']))
        for name in ('flow','critic'):
            core.equal_tree(self.payload[name],before[name])
            core.equal_tree(self.payload['optimizers'][name],before['optimizers'][name])
        for name in ('q','flow'):core.equal_tree(self.payload['generators'][name],before['generators'][name])
        core.equal_tree(self.source['payload'],before)
        self.assertEqual(self.payload['binding'],self.binding)
        self.assertEqual(self.source['payload']['binding']['initial_critic_sha256'],sha(self.fixture.old_critic))

    def test_actor_optimizer_rng_or_frozen_scalar_corruption_cannot_masquerade_as_bc(self):
        for change in ('teacher','scalar','scalar_adam','q_rng','actor_step','cpu_rng','partial','actor_fixed','source'):
            value=copy.deepcopy(self.payload);source=copy.deepcopy(self.source)
            if change=='teacher':value['flow']['linear.bias']+=.1
            elif change=='scalar':next(v for k,v in value['critic'].items() if k.startswith('members.') and torch.is_tensor(v)).add_(.1)
            elif change=='scalar_adam':value['optimizers']['critic']['state']={0:dict(step=torch.tensor(1.))}
            elif change=='q_rng':value['generators']['q']=torch.Generator().manual_seed(123).get_state()
            elif change=='actor_step':next(iter(value['optimizers']['actor']['state'].values()))['step']+=1
            elif change=='cpu_rng':value['rng']=torch.zeros(3,dtype=torch.uint8)
            elif change=='partial':value['step']-=1
            elif change=='actor_fixed':value['actor']['fixed']+=1
            else:source['binding']['version']='fake FQL prefix'
            with self.subTest(change=change),self.assertRaises((ValueError,RuntimeError)):
                child.validate_payload(value,source,self.binding)

    def test_explicit_fql_or_q_optimized_request_is_not_a_bc_parent(self):
        path=self.fixture.root/'request.json'
        request=dict(version=common.VERSION,experiment=str(self.fixture.old),config_sha256='a'*64,
            declaration_sha256='b'*64,seed=0,tasks_sha256='c'*64,final_outcomes={},
            settings=self.settings,diagnostic_interval=2,output=str(self.fixture.root/'continued'))
        for change in ('version','q_weight','update_critic'):
            value=copy.deepcopy(request)
            if change!='version':value['version']=training.VERSION
            if change=='q_weight':value['settings']['q_weight']=1.
            if change=='update_critic':value['settings']['update_critic']=True
            path.write_text(json.dumps(value))
            with self.subTest(change=change),self.assertRaises(ValueError):training.request_settings(value)

    def test_checked_parent_uses_real_bc_train_completion_history_and_two_sources(self):
        root=self.fixture.root.resolve();experiment=root/'physical';experiment.mkdir()
        output=root/'continued';output.mkdir();runtime=root/'runtime'
        cachepath=runtime/'benchmark_cache/bp_fe.pt';cachepath.parent.mkdir(parents=True)
        torch.save(dict(tiny='fixture'),cachepath)
        source_request=root/'source_request.json';source_request.write_text(json.dumps(
            dict(bc_experiment=str(self.fixture.old),scalar_experiment=str(self.fixture.other))))
        cfg=dict(design='bp_fe',cache_sha256=sha(cachepath),alpha=10.,scalar_schema=self.source['schema'],
            prospective_request=bind(source_request),verified_parents={'0':self.source['binding']},
            parent_identities={'0':dict(checkpoint=self.source['binding']['checkpoint'])},
            streams={'0':dict(collect=dict(seed=100,count=2))})
        (experiment/'config.json').write_text(json.dumps(cfg));(experiment/'declaration.json').write_text('{}')
        request=dict(version=training.VERSION,experiment=str(experiment),config_sha256=sha(experiment/'config.json'),
            declaration_sha256=sha(experiment/'declaration.json'),seed=0,tasks_sha256='f'*64,final_outcomes={},
            settings=self.settings,diagnostic_interval=2,output=str(output))
        requestpath=root/'train_request.json';requestpath.write_text(json.dumps(request))
        runner=dict(version=training.VERSION,request=bind(requestpath),config=bind(experiment/'config.json'),
            declaration=bind(experiment/'declaration.json'),continuation=self.binding,
            source_sha256={n:sha(child.ROOT/n) for n in training.SOURCE_NAMES},diagnostic_interval=2,backend_calls=0)
        (output/'binding.json').write_text(json.dumps(runner))
        state,replay=self.fixture.make_core(self.source,self.settings);training.STOP=False
        with redirect_stdout(io.StringIO()):
            training.run_loop(output,state,self.fixture.offline,replay,runner,diagnostic_interval=2)
        parent=dict(request=bind(requestpath),checkpoint=bind(output/'checkpoint.pt'))
        patches=[mock.patch.object(child,'RUNTIME',runtime),
            mock.patch.object(child.parent_export,'settings',return_value=(cfg,dict(source_sha256={}),{})),
            mock.patch.object(child.parent_export,'verify_export',return_value=[dict(role='evaluation',noise_sha256='2'*64)]),
            mock.patch.object(child.parent_export,'load_source',return_value=self.source),
            mock.patch.object(core,'admit_collect',return_value=replay)]
        for patch in patches:patch.start();self.addCleanup(patch.stop)
        result=child.checked_parent(parent,0)
        self.assertEqual(set(result),{'payload','source','cfg','physical','declaration','identity'})
        identity=result['identity'];self.assertEqual(identity['checkpoint'],parent['checkpoint'])
        self.assertEqual(identity['state_audit']['optimizer_audit']['actor']['actual_steps'],[1003.])
        self.assertEqual(identity['bc_actor_prefix'],self.source['binding']['actor_prefix'])
        self.assertEqual(identity['scalar_source'],self.source['binding']['target_scalar_source'])
        self.assertNotEqual(identity['checkpoint'],identity['bc_actor_prefix']['checkpoint'])
        self.assertEqual(identity['continuation_binding']['parent'],self.source['binding'])
        core.equal_tree(result['payload']['actor'],state.state()['actor'])
        original=(output/'completed.json').read_text();value=json.loads(original);value['critic_updates']=1
        (output/'completed.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'marker/checkpoint'):child.checked_parent(parent,0)
        (output/'completed.json').write_text(original)
        (output/'learning_curve.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'history'):child.checked_parent(parent,0)


if __name__=='__main__':unittest.main()
