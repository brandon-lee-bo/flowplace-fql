"""Actual tiny two-cycle state and unchanged native-admission CPU contracts."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch

from . import qstudy_scalar_online_second_train as second
from . import qstudy_scalar_online as core
from . import qstudy_scalar_online_train as first
from . import qstudy_scalar_online_child_export as child
from . import qstudy_backend as backend
from . import test_qstudy_scalar_online as tiny
from .qstudy_data import action_sha, bind, sha


class SecondCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.slurm();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.fixture=tiny.ScalarOnlineTests(methodName='test_prefix_optimizer_rng_and_named_states_restored_exactly')
        self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
        self.root=self.fixture.root.resolve();self.root_source=self.fixture.source
        self.root_source['refit']=dict(model=copy.deepcopy(self.fixture.critic))
        rootcp=self.root/'original_r0.pt';torch.save(self.root_source['payload'],rootcp)
        self.root_source['binding']['checkpoint']=bind(rootcp)
        self.fixture.replay['binding']['checkpoint_sha256']=sha(rootcp)
        self.first_state=self.fixture.core()
        self.first_folder=self.root/'first_train';self.first_runner=dict(version=first.VERSION,request=dict(path='fixture',sha256='a'*64))
        first.STOP=False;second.STOP=False
        with redirect_stdout(io.StringIO()):
            first.run_loop(self.first_folder,self.first_state,self.fixture.offline,self.fixture.replay,self.first_runner,diagnostic_interval=2)
        self.first_payload=first.checked_resume(self.first_folder,self.first_runner,self.first_state.binding)
        training_request=self.root/'first_request.json';training_request.write_text('{}')
        identity=dict(checkpoint=bind(self.first_folder/'checkpoint.pt'),completed=bind(self.first_folder/'completed.json'),
            runner_binding=bind(self.first_folder/'binding.json') if (self.first_folder/'binding.json').exists() else self.write(self.first_folder/'binding.json',self.first_runner),
            learning_curve=bind(self.first_folder/'learning_curve.json'),training_request=bind(training_request),
            continuation_binding=self.first_state.binding)
        self.verified=dict(source=self.root_source,payload=self.first_payload,identity=identity)
        self.cfg=dict(verified_children={'0':identity},scalar_schema=self.fixture.schema)
        self.source=second.source_from_verified(self.cfg,self.verified,0)
        self.parts=[];self.eval=[]
        for round_index in (0,1):
            part,pool=self.native_part(round_index)
            self.parts.append(part);self.eval+=pool
        self.admit()
        self.settings=dict(self.fixture.settings,total_updates=3)
        self.binding=second.second_binding(self.source,self.replay,self.settings)
        self.runner=dict(version=second.VERSION,request=dict(path='fixture',sha256='b'*64),continuation=self.binding)

    @staticmethod
    def write(path,value):
        Path(path).write_text(json.dumps(value));return bind(path)

    def native_part(self,round_index):
        experiment=self.root/f'physical_{round_index}';experiment.mkdir()
        cfg=dict(design='bp_fe',arm='original' if round_index==0 else 'actual_child',
            dataset_sha256='d'*64,physical_failure_reward=-4.,reward_scale=100.)
        self.write(experiment/'config.json',cfg)
        parent=self.root_source['binding']['checkpoint'] if round_index==0 else self.source['binding']['checkpoint']
        folder=experiment/f'seed_0/round_{round_index}/requests';folder.mkdir(parents=True)
        rows=[]
        for role in ('collect','evaluation'):
            sample=folder/cfg['arm']/role/'0';sample.mkdir(parents=True)
            raw=np.full((3,2),.1,dtype=np.float32)
            noise=np.full((3,2),.1+round_index*.2+(.05 if role=='evaluation' else 0),dtype=np.float32)
            np.savez_compressed(sample/'actions.npz',noise=noise,raw_action=raw,action=raw)
            row=dict(arm=cfg['arm'],graph='bp_fe',seed=0,round=round_index,role=role,index=0,
                terminal_mask=0,training_feedback_eligible=role=='collect',noise_seed=100+round_index if role=='collect' else 200+round_index,
                source_checkpoint=parent['path'],checkpoint_sha256=parent['sha256'],
                config_sha256=sha(experiment/'config.json'),dataset_sha256='d'*64,
                raw_action=raw.tolist(),executed_action=raw.tolist(),raw_action_sha256=action_sha(raw),
                action_sha256=action_sha(raw),noise_sha256=action_sha(noise),sample_dir=str(sample),
                run_id=f'go2o_fixture_s0_r{round_index}_{role}_0',pre_backend_failure='normalized_geometry',
                geometry=dict(legal=False),stage='global_route_only')
            self.write(sample/'request.json',row);rows.append(row)
        tasks=folder/'tasks.jsonl';tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.write(folder/'completed.json',dict(count=2,tasks_sha256=sha(tasks),config_sha256=sha(experiment/'config.json')))
        row=rows[0];sample=Path(row['sample_dir'])
        self.write(sample/'outcome.json',dict(row,status='geometry_failure',metrics=None,reward=-4.,
            request_sha256=sha(sample/'request.json'),tasks_sha256=sha(tasks)))
        physical=dict(config_sha256=sha(experiment/'config.json'),dataset_sha256='d'*64,reward_scale=100.)
        return dict(experiment=str(experiment),config=cfg,source_round=round_index,rows=rows,tasks=tasks,physical=physical),[rows[1]['noise_sha256']]

    def admit(self):
        # Only native context is a synthetic fixture; checked_request/existing,
        # load_requests, raw-array hashes and scalar label admission are real.
        for part in self.parts:
            index=part['source_round'];row=part['rows'][0]
            with mock.patch.object(backend,'context',return_value=(part['config'],None,part['physical'])):
                part['replay']=core.admit_collect(Path(part['experiment']),seed=0,source_round=index,count=1,
                    tasks_sha256=sha(part['tasks']),config_sha256=part['physical']['config_sha256'],
                    checkpoint_sha256=row['checkpoint_sha256'],collection_noise_seed=100+index,
                    evaluation_noise_sha256=self.eval,value=self.fixture.schema,cache=self.fixture.cache)
        self.replay=second.combine_replay(self.parts,self.source,self.eval)

    def continuation(self,*,payload=None,resume=False,settings=None):
        settings=settings or self.settings;binding=second.second_binding(self.source,self.replay,settings)
        models=copy.deepcopy(self.fixture.models)
        opts={k:(torch.optim.AdamW(v.parameters(),lr=99.) if k=='critic' else torch.optim.Adam(v.parameters(),lr=99.)) for k,v in models.items()}
        generators={k:torch.Generator().manual_seed(999) for k in core.GENERATOR_KEYS}
        return core.Continuation(models,opts,generators,self.source['payload'] if payload is None else payload,
            binding,frozen_reference=self.source,resume=resume)

    def test_real_c0_and_c1_native_admission_preserves_two_true_parents_and_b4(self):
        self.assertEqual(self.replay['derived_B'].flatten().tolist(),[4.,4.])
        self.assertEqual([v['raw_native_reward'] for v in self.replay['records']],[-4.,-4.])
        components=self.replay['binding']['components']
        self.assertEqual([v['source_round'] for v in components],[0,1])
        self.assertNotEqual(components[0]['checkpoint_sha256'],components[1]['checkpoint_sha256'])
        self.assertEqual([v['count'] for v in components],[1,1])
        self.assertEqual([v['source_arm'] for v in self.replay['records']],['original','actual_child'])
        self.assertEqual(self.source['payload']['binding'],self.first_payload['binding'])
        self.assertNotIn('seed',self.source['payload']['binding'])  # No forged r0 payload binding.

    def test_second_prefix_restores_true_t1_models_adam_and_rng_exactly(self):
        state=self.continuation().state()
        for name in core.MODEL_KEYS:
            core.equal_tree(state[name],self.first_payload[name])
            core.equal_tree(state['optimizers'][name],self.first_payload['optimizers'][name])
        core.equal_tree(state['rng'],self.first_payload['rng']);core.equal_tree(state['generators'],self.first_payload['generators'])
        self.assertEqual(state['step'],0);self.assertEqual(state['parent_steps'],4)
        self.assertEqual(second.audit_state(state,self.source,self.binding)['actor_adam_steps'],6)

    def test_completed_second_segment_adam_counts_teacher_and_root_normalizers(self):
        state=self.continuation()
        with redirect_stdout(io.StringIO()):
            second.run_loop(self.root/'complete',state,self.source,self.fixture.offline,self.replay,self.runner,diagnostic_interval=2)
        payload=second.checked_resume(self.root/'complete',self.runner,self.binding,self.source)
        second.checked_completion(self.root/'complete',payload,self.runner,self.binding)
        self.assertEqual(payload['step'],3)
        self.assertEqual(payload['cumulative_steps'],dict(root_actor_updates=2,first_actor_updates=4,second_actor_updates=3,
            actor_adam_steps=9,root_critic_updates=0,first_critic_updates=4,second_critic_updates=3,critic_adam_steps=7,teacher_updates=0))
        core.equal_tree(payload['flow'],self.root_source['payload']['flow'])
        for key in dict(self.fixture.critic.named_buffers()):
            core.equal_tree(payload['critic'][key],self.root_source['payload']['critic'][key])

    def test_second_resume_exact_all_tensors_adam_rng_and_history(self):
        full=self.continuation()
        with redirect_stdout(io.StringIO()):
            second.run_loop(self.root/'full',full,self.source,self.fixture.offline,self.replay,self.runner,diagnostic_interval=2)
        partial=self.continuation();settings={k:v for k,v in self.settings.items() if k!='total_updates'}
        row=partial.update(self.fixture.offline,self.replay,**settings)
        row.update(cycle=2,cumulative_steps=second.cumulative_counts(self.source,self.settings,row['step']),
            gradient_diagnostics=first.gradient_probe(partial))
        folder=self.root/'partial';folder.mkdir();second.save_state(folder,partial,self.source,self.runner,[row])
        payload=second.checked_resume(folder,self.runner,self.binding,self.source)
        resumed=self.continuation(payload=payload,resume=True)
        with redirect_stdout(io.StringIO()):
            second.run_loop(folder,resumed,self.source,self.fixture.offline,self.replay,self.runner,payload['history'],diagnostic_interval=2)
        core.equal_tree(full.state(),resumed.state())
        self.assertEqual(json.loads((folder/'learning_curve.json').read_text()),json.loads((self.root/'full/learning_curve.json').read_text()))

    def test_readiness_missing_final_or_infra_cannot_be_filtered_from_accumulation(self):
        for change in ('unready','invalid','nan','negative','missing','wrong_schema'):
            parts=copy.deepcopy(self.parts)
            if change=='unready':parts[1]['replay']['ready']=False
            elif change=='invalid':parts[1]['replay']['mask'][0]=False
            elif change=='nan':parts[1]['replay']['derived_B'][0]=float('nan')
            elif change=='negative':parts[1]['replay']['derived_B'][0]=-1.
            elif change=='missing':parts=parts[:1]
            else:parts[1]['replay']['utility_spec_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):second.combine_replay(parts,self.source,self.eval)

    def test_foreign_round_seed_parent_arm_and_eval_rows_refused(self):
        part=self.parts[1];record=part['replay']['records'][0];path=Path(record['request']['path']);original=path.read_bytes()
        for key,value in [('role','evaluation'),('training_feedback_eligible',False),('seed',1),('arm','foreign'),('round',0),
                          ('source_checkpoint',self.root_source['binding']['checkpoint']['path']),('noise_sha256',self.eval[0])]:
            parts=copy.deepcopy(self.parts);row=json.loads(original);row[key]=value;path.write_text(json.dumps(row))
            replacement=bind(path);parts[1]['replay']['records'][0]['request']=replacement
            parts[1]['replay']['binding']['records']=parts[1]['replay']['records']
            with self.subTest(key=key),self.assertRaises(ValueError):second.combine_replay(parts,self.source,self.eval)
        path.write_bytes(original)

    def test_duplicate_collect_and_changed_original_proofs_refused(self):
        parts=copy.deepcopy(self.parts);parts[1]['replay']['records'][0]['noise_sha256']=parts[0]['replay']['records'][0]['noise_sha256']
        parts[1]['replay']['binding']['records']=parts[1]['replay']['records']
        with self.assertRaises(ValueError):second.combine_replay(parts,self.source,self.eval)
        path=self.parts[0]['tasks'];path.write_text('changed')
        with self.assertRaises(ValueError):second.combine_replay(self.parts,self.source,self.eval)

    def test_t1_and_root_identity_or_optimizer_count_cannot_be_forged(self):
        for change in ('identity','root_seed','actor_adam','root_teacher'):
            cfg=copy.deepcopy(self.cfg);verified=copy.deepcopy(self.verified)
            if change=='identity':cfg['verified_children']['0']['checkpoint']['sha256']='0'*64
            elif change=='root_seed':verified['source']['payload']['binding']['seed']=1
            elif change=='actor_adam':next(iter(verified['payload']['optimizers']['actor']['state'].values()))['step']+=1
            else:verified['source']['payload']['flow']['linear.weight']+=.1
            with self.subTest(change=change),self.assertRaises(ValueError):second.source_from_verified(cfg,verified,0)

    def test_second_frozen_q_control_keeps_t1_q_adam_rng_and_finite_zero_pressure(self):
        settings=dict(self.settings,update_critic=False)
        state=self.continuation(settings=settings)
        for _ in range(3):state.update(self.fixture.offline,self.replay,**{k:v for k,v in settings.items() if k!='total_updates'})
        result=second.audit_state(state.state(),self.source,state.binding)
        self.assertEqual(result['critic_adam_steps'],4);self.assertEqual(result['second_critic_updates'],0)
        core.equal_tree(state.state()['critic'],self.first_payload['critic'])
        core.equal_tree(state.state()['generators']['q'],self.first_payload['generators']['q'])
        altered=copy.deepcopy(state.state());next(v for k,v in altered['critic'].items() if k.startswith('members.')).add_(.1)
        with self.assertRaises(ValueError):second.audit_state(altered,self.source,state.binding)

    def test_completed_reentry_and_cumulative_counter_tamper_rejected(self):
        folder=self.root/'done';state=self.continuation()
        with redirect_stdout(io.StringIO()):second.run_loop(folder,state,self.source,self.fixture.offline,self.replay,self.runner,diagnostic_interval=2)
        payload=second.checked_resume(folder,self.runner,self.binding,self.source)
        marker=folder/'completed.json';value=json.loads(marker.read_text());value['cycle']=1;marker.write_text(json.dumps(value))
        with self.assertRaises(ValueError):second.checked_completion(folder,payload,self.runner,self.binding)
        payload['cumulative_steps']['actor_adam_steps']=999;torch.save(payload,folder/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError,'cumulative'):second.checked_resume(folder,self.runner,self.binding,self.source)

    def test_second_stop_atomic_resume_and_gradient_probe_do_not_reseed(self):
        state=self.continuation();before=state.state();first.gradient_probe(state);core.equal_tree(before,state.state())
        folder=self.root/'stop';second.STOP=True
        with self.assertRaises(SystemExit) as result:
            with redirect_stdout(io.StringIO()):second.run_loop(folder,state,self.source,self.fixture.offline,self.replay,self.runner,diagnostic_interval=2)
        self.assertEqual(result.exception.code,85);self.assertFalse((folder/'completed.json').exists())
        payload=second.checked_resume(folder,self.runner,self.binding,self.source)
        self.assertEqual(payload['step'],1);self.assertEqual(payload['cumulative_steps']['actor_adam_steps'],7)

    def test_request_is_explicit_cycle_two_no_implicit_reward_or_budget(self):
        value=dict(version=second.VERSION,cycle=2,experiment=str(self.root/'child'),config_sha256='a'*64,
            declaration_sha256='b'*64,seed=0,tasks_sha256='c'*64,final_outcomes={},settings=self.settings,
            diagnostic_interval=2,output=str(self.root/'second'))
        self.assertEqual(second.request_settings(value),value)
        for change in ('cycle','missing','odd','nan','nested','bc'):
            bad=copy.deepcopy(value)
            if change=='cycle':bad['cycle']=3
            elif change=='missing':del bad['settings']['alpha']
            elif change=='odd':bad['settings']['batch_size']=3
            elif change=='nan':bad['settings']['alpha']=float('nan')
            elif change=='nested':bad['output']=str(self.root/'child'/'nested')
            else:bad['bc_source']={}
            with self.subTest(change=change),self.assertRaises(ValueError):second.request_settings(bad)


if __name__=='__main__':unittest.main()
