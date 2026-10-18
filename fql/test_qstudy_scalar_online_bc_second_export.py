"""CPU fixtures for genuine T2 provenance and unchanged native row contracts."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import shutil
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch

from . import qstudy_scalar_online_bc_second_export as export
from . import qstudy_scalar_online_bc_second_train as training
from . import qstudy_scalar_online as core
from . import qstudy_backend as backend
from . import test_qstudy_scalar_online as tiny
from . import test_qstudy_scalar_online_bc_second_train as cycle
from .qstudy_data import action_sha, bind, sha


class SecondExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.slurm();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.fixture=tiny.ScalarOnlineTests(methodName='test_prefix_optimizer_rng_and_named_states_restored_exactly')
        self.fixture.setUp();self.addCleanup(self.fixture.tearDown);self.addCleanup(mock.patch.stopall)
        self.root=self.fixture.root.resolve();self.output=self.root/'second_export'
        self.child=self.root/'child_physical';self.child.mkdir()
        self.offline=self.root/'offline';self.offline.mkdir()
        self.first_export=self.root/'first_physical';self.first_export.mkdir()
        self.base=dict(design='bp_fe',dataset_path=str(self.root/'dataset.npz'),dataset_sha256='d'*64,
            cache_sha256='e'*64,physical_failure_reward=-4.,reward_scale=100.,scalar_schema=self.fixture.schema,
            streams={str(s):dict(collect=dict(seed=300+s,count=1),evaluation=dict(seed=400,count=2)) for s in export.SEEDS})
        for folder in (self.child,self.offline,self.first_export):
            self.write(folder/'config.json',self.base);self.write(folder/'declaration.json',{})
        self.request=dict(version=export.VERSION,logical_round=2,parent_training={},arm='actualT2',run_prefix='second2',
            streams={str(s):dict(evaluation=dict(seed=400,count=2)) for s in export.SEEDS},
            evaluation_purpose='development_validation',backend_storage='scratch',output=str(self.output))
        self.results={}
        for seed in export.SEEDS:
            folder=self.root/f'second_train_{seed}';folder.mkdir()
            for name in ('checkpoint.pt','completed.json','binding.json','learning_curve.json'):self.write(folder/name,dict(seed=seed,name=name))
            request=dict(version=training.VERSION,cycle=2,experiment=str(self.child),config_sha256=sha(self.child/'config.json'),
                declaration_sha256=sha(self.child/'declaration.json'),seed=seed,tasks_sha256='f'*64,final_outcomes={},
                settings=dict(self.fixture.settings,q_weight=0.,update_critic=False),diagnostic_interval=2,output=str(folder))
            request_path=self.root/f'train_request_{seed}.json';self.write(request_path,request)
            parent=dict(request=bind(request_path),checkpoint=bind(folder/'checkpoint.pt'))
            self.request['parent_training'][str(seed)]=parent
            first_folder=self.root/f'first_train_{seed}';first_folder.mkdir();self.write(first_folder/'checkpoint.pt',dict(t1=seed))
            bc_checkpoint=self.offline/f'bc_{seed}.pt';self.write(bc_checkpoint,dict(original_bc=seed))
            scalar_refit=self.offline/f'scalar_{seed}.pt';self.write(scalar_refit,dict(separate_scalar=seed))
            identity=dict(training_request=parent['request'],checkpoint=parent['checkpoint'],completed=bind(folder/'completed.json'),
                runner_binding=bind(folder/'binding.json'),learning_curve=bind(folder/'learning_curve.json'),
                parent_export_config=bind(self.child/'config.json'),parent_export_declaration=bind(self.child/'declaration.json'),
                root_source=dict(config=bind(self.offline/'config.json')),
                first_continuation=dict(checkpoint=bind(first_folder/'checkpoint.pt'),
                    first_continuation=dict(parent_export_config=bind(self.first_export/'config.json'),
                        continuation_binding=dict(settings=dict(self.fixture.settings,q_weight=0.,update_critic=False)))),
                continuation_binding=dict(settings=dict(self.fixture.settings,q_weight=0.,update_critic=False)),cumulative_steps=dict(actor_adam_steps=10),
                previous_collect_noise_sha256=['1'*64],previous_evaluation_noise_sha256=['2'*64],
                previous_collect_seeds=[100,101,102,300,301,302],state_cycle=2,
                bc_actor_prefix=dict(checkpoint=bind(bc_checkpoint)),scalar_source=dict(config=bind(self.offline/'config.json')),
                scalar_critic_refit=bind(scalar_refit),q_weight=0.)
            payload=copy.deepcopy(self.fixture.payload)
            # The BC T2 actor differs, while the separate diagnostic scalar critic stays frozen.
            payload['actor']['linear.bias']+=.13
            self.results[seed]=dict(cfg=self.base,identity=identity,declaration=dict(source_sha256={}),
                source=dict(schema=self.fixture.schema,refit=dict(model=copy.deepcopy(self.fixture.critic))),payload=payload)
        self.request_path=self.root/'export_request.json';self.write(self.request_path,self.request)
        self.initial={str(p):sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.loader=mock.patch.object(export,'checked_parent',side_effect=lambda p,s:self.results[s]).start()
        mock.patch.object(export,'context',side_effect=self.context).start()

    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value))

    def context(self,experiment):
        cfg=json.loads((Path(experiment)/'config.json').read_text())
        return cfg,{},dict(config_sha256=sha(Path(experiment)/'config.json'),dataset_sha256=cfg['dataset_sha256'],
            reward_scale=cfg['reward_scale'],design=cfg['design'],physical_flow_revision='fixture')

    def prepare(self):
        with redirect_stdout(io.StringIO()):export.prepare(self.request_path,sha(self.request_path),self.output)
        return export.settings(self.output,sha(self.output/'config.json'))

    def population(self):
        cfg,decl,physical=self.prepare();folder=self.output/'seed_0/round_2/requests';folder.mkdir(parents=True)
        rows=[];identity=cfg['verified_seconds']['0'];parent=self.request['parent_training']['0']['checkpoint']
        for index in range(2):
            sample=folder/cfg['arm']/'evaluation'/str(index);sample.mkdir(parents=True)
            raw=np.full((3,2),.1,dtype=np.float32);noise=np.full((3,2),.6+index*.01,dtype=np.float32)
            np.savez_compressed(sample/'actions.npz',noise=noise,raw_action=raw,action=raw)
            row=dict(arm=cfg['arm'],seed=0,round=2,role='evaluation',index=index,graph='bp_fe',terminal_mask=0,
                training_feedback_eligible=False,offline_train_eligible=False,noise_seed=400,raw_action=raw.tolist(),
                executed_action=raw.tolist(),predicted_q=[-1.1],q_units=core.Q_UNITS,source_checkpoint=parent['path'],
                checkpoint_sha256=parent['sha256'],source_state_key='actor',critic_state_key='critic',
                q_weight=0.,actor_source_kind='genuine_bc_second_continuation',
                critic_source_kind='separately_bound_frozen_scalar_refit',
                original_bc_checkpoint=identity['bc_actor_prefix']['checkpoint'],scalar_critic_refit=identity['scalar_critic_refit'],
                source_config_sha256=identity['parent_export_config']['sha256'],continuation_identity_sha256=core.json_sha(identity),
                training_request_sha256=identity['training_request']['sha256'],export_declaration_sha256=sha(self.output/'declaration.json'),
                scalar_utility_spec_sha256=cfg['scalar_schema']['utility_spec_sha256'],config_sha256=physical['config_sha256'],
                dataset_sha256=cfg['dataset_sha256'],sample_dir=str(sample),run_id=f"go2o_qsbc2_{cfg['run_prefix']}_s0_r2_evaluation_{index}",
                action_sha256=action_sha(raw),raw_action_sha256=action_sha(raw),noise_sha256=action_sha(noise),
                geometry=dict(legal=False),pre_backend_failure='normalized_geometry',stage='global_route_only',
                macro_hpwl=1.,evaluation_purpose=cfg['evaluation_purpose'],objective_weights=cfg['scalar_schema']['utility_spec']['weights'])
            self.write(sample/'request.json',row);rows.append(row)
        tasks=folder/'tasks.jsonl';self.seal(cfg,rows,tasks)
        return cfg,decl,physical,rows,tasks

    def seal(self,cfg,rows,tasks):
        tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.write(tasks.with_name('completed.json'),export.completion(cfg,self.output,0,rows,tasks))

    def test_metadata_prepare_all_three_real_parent_bindings_is_immutable_and_eval_only(self):
        cfg,decl,physical=self.prepare();self.assertEqual(self.loader.call_count,3)
        self.assertEqual(cfg['logical_round'],2);self.assertEqual(decl['roles'],['evaluation']);self.assertFalse(decl['q_selection'])
        self.assertFalse(list(self.output.rglob('*.pt')));self.assertFalse(list(self.output.rglob('actions.npz')))
        before={str(p):sha(p) for p in self.output.rglob('*') if p.is_file()};self.prepare()
        self.assertEqual(before,{str(p):sha(p) for p in self.output.rglob('*') if p.is_file()})
        self.assertEqual(self.initial,{p:sha(p) for p in self.initial})

    def test_request_forbids_t1_fql_implicit_seed_collect_or_missing_parent(self):
        for change in ('round','bc','collect','missing','bool','zero','relative','fql_version'):
            bad=copy.deepcopy(self.request)
            if change=='round':bad['logical_round']=1
            elif change=='bc':bad['parent_training']['0']['actor_prefix']={}
            elif change=='collect':bad['streams']['0']['collect']=dict(seed=999,count=1)
            elif change=='missing':del bad['parent_training']['2']
            elif change=='bool':bad['streams']['0']['evaluation']['seed']=True
            elif change=='zero':bad['streams']['0']['evaluation']['count']=0
            elif change=='fql_version':bad['version']='qstudy_scalar_online_second_export_v1'
            else:bad['output']='relative'
            with self.subTest(change=change),self.assertRaises(ValueError):export.request_settings(bad)
        self.request['streams']['0']['evaluation']['seed']=100;self.write(self.request_path,self.request)
        with self.assertRaisesRegex(ValueError,'collect stream'):self.prepare()

    def test_source_snapshot_actual_parent_and_external_request_mutation_refused(self):
        with self.assertRaises(ValueError):export.prepare(self.request_path,'0'*64,self.output)
        self.prepare();digest=sha(self.output/'config.json')
        snapshot=self.output/'sources/fql/qstudy_scalar_online_bc_second_export.py';original=snapshot.read_bytes();snapshot.write_text('changed')
        with self.assertRaises(ValueError):export.settings(self.output,digest)
        snapshot.write_bytes(original)
        Path(self.request['parent_training']['0']['checkpoint']['path']).write_text('different')
        with self.assertRaises(ValueError):export.settings(self.output,digest)

    def test_actual_native_round_two_contract_and_eval_cannot_enter_scalar_replay(self):
        cfg,decl,physical,rows,tasks=self.population()
        self.assertEqual(export.verify_export(self.output,0,cfg,decl,physical),rows)
        for row in rows:backend.checked_request(row,type('Args',(),dict(seed=0,round=2))(),cfg,physical)
        with mock.patch.object(backend,'context',side_effect=self.context),self.assertRaisesRegex(ValueError,'denominator'):
            core.admit_collect(self.output,seed=0,source_round=2,count=1,tasks_sha256=sha(tasks),
                config_sha256=physical['config_sha256'],checkpoint_sha256=rows[0]['checkpoint_sha256'],
                collection_noise_seed=999,evaluation_noise_sha256=[r['noise_sha256'] for r in rows],
                value=self.fixture.schema,cache=self.fixture.cache)

    def test_row_source_round_role_raw_noise_and_replay_collision_are_bound(self):
        cfg,decl,physical,rows,tasks=self.population()
        for key,value in [('round',1),('role','collect'),('source_state_key','flow'),('critic_state_key','actor'),
            ('training_feedback_eligible',True),('offline_train_eligible',True),('checkpoint_sha256','0'*64),
            ('training_request_sha256','0'*64),('q_units','old WL normalized'),('predicted_q',[float('nan')]),
            ('q_weight',1.),('actor_source_kind','FQL'),('original_bc_checkpoint',{}),('scalar_critic_refit',{})]:
            bad=dict(rows[0]);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):export.check_row(bad,cfg,decl,physical,self.output,0,0)
        with self.assertRaisesRegex(ValueError,'actual noise'):
            export.check_row(rows[0],cfg,decl,physical,self.output,0,0,np.zeros((3,2),dtype=np.float32))
        bad=copy.deepcopy(cfg);bad['prior_collect_noise_sha256']=[rows[0]['noise_sha256']]
        with self.assertRaisesRegex(ValueError,'collect noise'):export.check_row(rows[0],bad,decl,physical,self.output,0,0)
        path=Path(rows[0]['sample_dir'])/'actions.npz'
        with np.load(path,allow_pickle=False) as arrays:values={k:arrays[k] for k in arrays.files}
        values['raw_action']+=.01;np.savez_compressed(path,**values)
        with self.assertRaises(ValueError):export.verify_export(self.output,0,cfg,decl,physical)

    def test_completed_reentry_no_graph_or_gpu_marker_and_population_tamper_refused(self):
        cfg,decl,physical,rows,tasks=self.population()
        with mock.patch.object(torch.cuda,'is_available',return_value=False),mock.patch.object(export,'GraphPolicy',side_effect=AssertionError('no graph')):
            self.assertEqual(export.export(self.output,physical['config_sha256'],0),rows)
        for change in ('status','role','count'):
            self.seal(cfg,rows,tasks);marker=export.read(tasks.with_name('completed.json'))
            if change=='status':marker['status']='partial'
            elif change=='role':marker['evaluation_never_replayed']=False
            else:marker['count']=1
            self.write(tasks.with_name('completed.json'),marker)
            with self.subTest(change=change),self.assertRaises(ValueError):export.verify_export(self.output,0,cfg,decl,physical)
        self.seal(cfg,rows[::-1],tasks)
        with self.assertRaisesRegex(ValueError,'population'):export.verify_export(self.output,0,cfg,decl,physical)

    def inference_patches(self):
        runtime=self.root/'runtime';cache=runtime/'benchmark_cache/bp_fe.pt';cache.parent.mkdir(parents=True);cache.write_text('fixture')
        for result in self.results.values():result['cfg']['cache_sha256']=sha(cache)
        self.write(self.child/'config.json',self.base)
        # Re-pin captured child config after the fixture-only cache is created.
        for result in self.results.values():
            result['identity']['parent_export_config']=bind(self.child/'config.json')
        problem=type('Problem',(),dict(x=torch.zeros(3,2),benchmark_info={}))()
        rand=torch.rand;generator=torch.Generator
        def cpu_rand(*args,**kwargs):kwargs['device']='cpu';return rand(*args,**kwargs)
        def cpu_generator(*args,**kwargs):return generator(device='cpu')
        metric=type('Metric',(),dict(sizes=np.ones((3,2)),__call__=lambda self,a:1.))()
        return [mock.patch.object(export,'RUNTIME',runtime),mock.patch.object(torch.cuda,'is_available',return_value=True),
            mock.patch.object(export,'load_problem',return_value=problem),mock.patch.object(export,'GraphPolicy',side_effect=lambda p:tiny.TinyPolicy()),
            mock.patch.object(export,'MacroHPWL',return_value=metric),mock.patch.object(export,'legalize',side_effect=lambda x,p:x),
            mock.patch.object(export,'legality',return_value=dict(legal=False)),mock.patch.object(export.torch,'rand',side_effect=cpu_rand),
            mock.patch.object(export.torch,'Generator',side_effect=cpu_generator),
            mock.patch.object(torch.nn.Module,'cuda',lambda value:value)]

    def test_cpu_mocked_export_loads_actual_t2_bc_actor_and_frozen_scalar_q(self):
        patches=self.inference_patches()
        for patch in patches:patch.start()
        cfg,decl,physical=self.prepare()
        rows=export.export(self.output,physical['config_sha256'],0)
        actor=tiny.TinyPolicy();actor.load_state_dict(self.results[0]['payload']['actor'],strict=True)
        critic=copy.deepcopy(self.fixture.critic);critic.load_state_dict(self.results[0]['payload']['critic'],strict=True)
        for row in rows:
            noise=torch.tensor(np.load(Path(row['sample_dir'])/'actions.npz')['noise'])
            with torch.no_grad():expected=actor(noise[None]);q=float(core.scalar_q(critic,expected)[0])
            np.testing.assert_array_equal(np.asarray(row['raw_action'],np.float32),expected[0].numpy())
            self.assertEqual(row['predicted_q'],[q]);self.assertFalse(row['training_feedback_eligible'])
        self.assertEqual(len(rows),2)

    def test_interrupted_export_reuses_original_row_and_exact_saved_noise(self):
        for patch in self.inference_patches():patch.start()
        cfg,decl,physical=self.prepare();atomic=export.atomic_json
        def interrupted(path,value):
            if Path(path).name=='request.json' and Path(path).parent.name=='1':raise RuntimeError('synthetic interruption')
            return atomic(path,value)
        with mock.patch.object(export,'atomic_json',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'interruption'):
            export.export(self.output,physical['config_sha256'],0)
        rowpath=self.output/f"seed_0/round_2/requests/{cfg['arm']}/evaluation/0/request.json";before=rowpath.read_bytes()
        rows=export.export(self.output,physical['config_sha256'],0)
        self.assertEqual(rowpath.read_bytes(),before);self.assertEqual(len(rows),2)
        with np.load(Path(rows[0]['sample_dir'])/'actions.npz') as arrays:
            export.check_row(rows[0],cfg,decl,physical,self.output,0,0,arrays['noise'])


class ActualBCT2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cycle.BCSecondCycleTests.setUpClass()

    def setUp(self):
        self.fixture=cycle.BCSecondCycleTests(methodName='test_second_initial_state_restores_true_t1_actor_adam_and_all_rng')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.addCleanup(mock.patch.stopall)
        self.root=self.fixture.root
        self.cfg=dict(design='bp_fe',streams={str(s):dict(collect=dict(seed=300+s,count=1)) for s in export.SEEDS})
        folder=self.root/'source_metadata';folder.mkdir()
        self.write(folder/'config.json',self.cfg);self.write(folder/'declaration.json',{})
        self.output=self.root/'actual_BC_T2'
        self.request=dict(version=training.VERSION,cycle=2,experiment=str(folder),config_sha256=sha(folder/'config.json'),
            declaration_sha256=sha(folder/'declaration.json'),seed=0,tasks_sha256='f'*64,final_outcomes={},
            settings=self.fixture.settings,diagnostic_interval=2,output=str(self.output))
        self.path=self.root/'actual_second_request.json';self.write(self.path,self.request)
        # Original real C0+C1 admission is supplied by the BCsecond fixtures;
        # state updates, Adam, completion and export checks remain actual code.
        mock.patch.object(training,'SOURCE_NAMES',('fql/qstudy_scalar_online_bc_second_train.py',)).start()
        mock.patch.object(training,'load_inputs',side_effect=self.inputs).start()

    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value))

    def inputs(self,request):
        binding=training.second_binding(self.fixture.source,self.fixture.replay,request['settings'])
        return self.cfg,dict(source_sha256={}),self.fixture.source,self.fixture.replay,binding

    def completed(self):
        self.write(self.path,self.request);_,_,source,replay,binding=self.inputs(self.request)
        runner=dict(version=training.VERSION,cycle=2,request=bind(self.path),config=bind(Path(self.request['experiment'])/'config.json'),
            declaration=bind(Path(self.request['experiment'])/'declaration.json'),continuation=binding,
            source_sha256={n:sha(export.ROOT/n) for n in training.SOURCE_NAMES},diagnostic_interval=2,backend_calls=0)
        self.output.mkdir();self.write(self.output/'binding.json',runner)
        for name in training.SOURCE_NAMES:
            dest=self.output/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(export.ROOT/name,dest)
        state=self.fixture.continuation();training.STOP=False
        with redirect_stdout(io.StringIO()):
            training.run_loop(self.output,state,source,self.fixture.fixture.offline,replay,runner,diagnostic_interval=2)
        return dict(request=bind(self.path),checkpoint=bind(self.output/'checkpoint.pt')),state

    def test_actual_bc_t2_actor_adam_history_and_two_source_provenance(self):
        parent,state=self.completed();before={str(p):sha(p) for p in self.root.rglob('*') if p.is_file()}
        result=export.checked_parent(parent,0);identity=result['identity']
        self.assertEqual(identity['cumulative_steps']['actor_adam_steps'],1007)
        self.assertEqual(identity['cumulative_steps']['critic_adam_steps'],0)
        self.assertEqual([v['count'] for v in identity['continuation_binding']['replay']['components']],[1,1])
        self.assertEqual(identity['bc_actor_prefix'],self.fixture.root_source['binding']['actor_prefix'])
        self.assertEqual(identity['scalar_source'],self.fixture.root_source['binding']['target_scalar_source'])
        self.assertEqual(identity['scalar_critic_refit'],self.fixture.root_source['binding']['critic_refit'])
        self.assertNotEqual(identity['checkpoint'],identity['bc_actor_prefix']['checkpoint'])
        core.equal_tree(result['payload']['actor'],state.state()['actor'])
        core.equal_tree(result['source']['refit']['model'].state_dict(),state.state()['critic'])
        self.assertFalse(torch.equal(result['payload']['actor']['linear.bias'],self.fixture.first_payload['actor']['linear.bias']))
        self.assertEqual(before,{p:sha(p) for p in before})

    def test_bc_t2_scalar_teacher_adam_and_unused_rng_remain_bitwise_frozen(self):
        parent,state=self.completed();result=export.checked_parent(parent,0)
        for name in ('flow','critic'):
            core.equal_tree(result['payload'][name],self.fixture.root_source['payload'][name])
            core.equal_tree(result['payload']['optimizers'][name],self.fixture.root_source['payload']['optimizers'][name])
        for name in ('q','flow'):
            core.equal_tree(result['payload']['generators'][name],self.fixture.root_source['payload']['generators'][name])
        self.assertEqual(result['payload']['parent_steps'],4)
        self.assertEqual(result['payload']['step'],3)

    def test_fql_t1_partial_history_scalar_adam_and_teacher_corruption_refused(self):
        parent,state=self.completed();checkpoint=self.output/'checkpoint.pt';original=checkpoint.read_bytes()
        wrong=dict(parent,checkpoint=bind(self.fixture.first_folder/'checkpoint.pt'))
        with self.assertRaises(ValueError):export.checked_parent(wrong,0)
        for change in ('t1','history','cumulative','teacher','critic','critic_opt','unused_rng','fql_request'):
            checkpoint.write_bytes(original);payload=torch.load(checkpoint,weights_only=False)
            if change=='t1':payload=copy.deepcopy(self.fixture.first_payload)
            elif change=='history':payload['history']=payload['history'][:-1]
            elif change=='cumulative':payload['cumulative_steps']['actor_adam_steps']-=1
            elif change=='teacher':payload['flow']['linear.weight']+=.1
            elif change=='critic':next(v for k,v in payload['critic'].items() if k.startswith('members.') and torch.is_tensor(v)).add_(.1)
            elif change=='critic_opt':payload['optimizers']['critic']['param_groups'][0]['lr']=.1
            elif change=='unused_rng':payload['generators']['q'][0]^=1
            else:
                value=dict(self.request,version=training.second.VERSION);self.write(self.path,value)
            torch.save(payload,checkpoint);wrong=dict(request=bind(self.path),checkpoint=bind(checkpoint))
            with self.subTest(change=change),self.assertRaises((ValueError,RuntimeError)):export.checked_parent(wrong,0)
        checkpoint.write_bytes(original);self.write(self.path,self.request)

    def test_actual_parent_external_seed_curve_snapshot_and_marker_tamper_refused(self):
        parent,state=self.completed()
        with self.assertRaisesRegex(ValueError,'seed differs'):export.checked_parent(parent,1)
        for path in (self.output/'learning_curve.json',self.output/'completed.json',
                     self.output/'sources/fql/qstudy_scalar_online_bc_second_train.py'):
            original=path.read_bytes();path.write_text('[]')
            with self.subTest(path=path.name),self.assertRaises((ValueError,TypeError)):export.checked_parent(parent,0)
            path.write_bytes(original)


if __name__=='__main__':unittest.main()
