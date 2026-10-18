"""Tiny genuine BC dual-source/two-native-cycle contracts; no graph or EDA."""
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

from . import qstudy_scalar_online_bc_second_train as runner
from . import qstudy_scalar_online_bc_child_export as child
from . import qstudy_scalar_online_bc_train as first_bc
from . import qstudy_scalar_online_train as common
from . import qstudy_scalar_online as core
from . import test_qstudy_scalar_online_bc as bc_fixture
from . import qstudy_backend as backend
from .qstudy_data import action_sha, bind, sha


class BCSecondCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bc_fixture.BCSourceTests.setUpClass()

    def setUp(self):
        self.fixture = bc_fixture.BCSourceTests(
            methodName='test_real_bc_identity_and_all_preserved_states_with_two_source_binding')
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root.resolve()
        self.root_source = self.fixture.load()
        self.first_settings = dict(total_updates=4, batch_size=4, alpha=10., q_weight=0.,
            flow_steps=10, update_critic=False, huber_delta=1., grad_clip=1.)
        self.runtime = self.root/'runtime'
        (self.runtime/'benchmark_cache').mkdir(parents=True)
        self.cache_path = self.runtime/'benchmark_cache/bp_fe.pt'
        torch.save({}, self.cache_path)
        self.parts = [self.native_part(0, self.root_source['binding']['checkpoint'])]
        self.evaluation = [self.parts[0]['rows'][1]['noise_sha256']]
        self.admit_part(self.parts[0], self.evaluation)
        self.first_folder = self.root/'first_train'
        self.first_request = dict(version=first_bc.VERSION, experiment=self.parts[0]['experiment'],
            config_sha256=sha(Path(self.parts[0]['experiment'])/'config.json'),
            declaration_sha256=sha(Path(self.parts[0]['experiment'])/'declaration.json'), seed=0,
            tasks_sha256=sha(self.parts[0]['tasks']), final_outcomes={}, settings=self.first_settings,
            diagnostic_interval=2, output=str(self.first_folder))
        self.first_request_path = self.root/'first_request.json'
        self.write(self.first_request_path, self.first_request)
        base, _ = self.fixture.make_core(self.root_source, self.first_settings)
        first_binding = core.continuation_binding(self.root_source, self.parts[0]['replay'], self.first_settings)
        self.first_state = core.Continuation(base.models, base.opts, base.generators,
            self.root_source['payload'], first_binding, frozen_reference=self.root_source)
        first_runner = dict(version=first_bc.VERSION, request=bind(self.first_request_path))
        first_bc.STOP = False
        runner.STOP = False
        with redirect_stdout(io.StringIO()):
            first_bc.run_loop(self.first_folder, self.first_state, self.fixture.offline,
                self.parts[0]['replay'], first_runner, diagnostic_interval=2)
        self.first_payload = common.checked_resume(self.first_folder, first_runner, first_binding)
        self.write(self.first_folder/'binding.json', first_runner)
        identity = dict(checkpoint=bind(self.first_folder/'checkpoint.pt'),
            completed=bind(self.first_folder/'completed.json'), runner_binding=bind(self.first_folder/'binding.json'),
            learning_curve=bind(self.first_folder/'learning_curve.json'), training_request=bind(self.first_request_path),
            parent_export_config=bind(Path(self.parts[0]['experiment'])/'config.json'),
            parent_export_declaration=bind(Path(self.parts[0]['experiment'])/'declaration.json'),
            offline_source=self.root_source['binding'], bc_actor_prefix=self.root_source['binding']['actor_prefix'],
            scalar_source=self.root_source['binding']['target_scalar_source'],
            scalar_critic_refit=self.root_source['binding']['critic_refit'], continuation_binding=first_binding)
        self.verified = dict(source=self.root_source, payload=self.first_payload, identity=identity,
            cfg=self.parts[0]['config'], declaration={}, physical=self.parts[0]['physical'])
        self.cfg = dict(verified_children={'0':identity}, scalar_schema=self.fixture.schema)
        self.source = runner.source_from_verified(self.cfg, self.verified, 0)
        self.parts.append(self.native_part(1, self.source['binding']['checkpoint']))
        self.evaluation.append(self.parts[1]['rows'][1]['noise_sha256'])
        for part in self.parts:
            self.admit_part(part, self.evaluation)
        self.replay = runner.combine_replay(self.parts, self.source, self.evaluation)
        self.settings = dict(self.first_settings, total_updates=3)
        self.binding = runner.second_binding(self.source, self.replay, self.settings)
        self.runner_binding = dict(version=runner.VERSION,
            request=dict(path='synthetic', sha256='b'*64), continuation=self.binding)

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value))
        return bind(path)

    def native_part(self, round_index, parent):
        experiment = self.root/f'physical_{round_index}'
        experiment.mkdir()
        cfg = dict(design='bp_fe', arm='bc_root' if round_index==0 else 'bc_true_t1',
            dataset_sha256='d'*64, cache_sha256=sha(self.cache_path), physical_failure_reward=-4., reward_scale=100.,
            source_experiment=str(self.fixture.old), scalar_experiment=str(self.fixture.other),
            streams={'0':dict(collect=dict(seed=100+round_index, count=1),
                evaluation=dict(seed=200+round_index, count=1))})
        self.write(experiment/'config.json', cfg)
        self.write(experiment/'declaration.json', {})
        folder = experiment/f'seed_0/round_{round_index}/requests'
        folder.mkdir(parents=True)
        rows = []
        for role in ('collect', 'evaluation'):
            sample = folder/cfg['arm']/role/'0'
            sample.mkdir(parents=True)
            raw = np.full((3,2), .1, dtype=np.float32)
            noise = np.full((3,2), .1+.2*round_index+(.05 if role=='evaluation' else 0), dtype=np.float32)
            np.savez_compressed(sample/'actions.npz', noise=noise, raw_action=raw, action=raw)
            row = dict(arm=cfg['arm'], graph='bp_fe', seed=0, round=round_index, role=role, index=0,
                terminal_mask=0, training_feedback_eligible=role=='collect',
                noise_seed=cfg['streams']['0'][role]['seed'], source_checkpoint=parent['path'],
                checkpoint_sha256=parent['sha256'], config_sha256=sha(experiment/'config.json'),
                dataset_sha256='d'*64, raw_action=raw.tolist(), executed_action=raw.tolist(),
                raw_action_sha256=action_sha(raw), action_sha256=action_sha(raw), noise_sha256=action_sha(noise),
                sample_dir=str(sample), run_id=f'go2o_bcfixture_s0_r{round_index}_{role}_0',
                pre_backend_failure='normalized_geometry', geometry=dict(legal=False), stage='global_route_only')
            self.write(sample/'request.json', row)
            rows.append(row)
        tasks = folder/'tasks.jsonl'
        tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.write(folder/'completed.json', dict(count=2, tasks_sha256=sha(tasks), config_sha256=sha(experiment/'config.json')))
        row = rows[0]
        sample = Path(row['sample_dir'])
        self.write(sample/'outcome.json', dict(row, status='geometry_failure', metrics=None, reward=-4.,
            request_sha256=sha(sample/'request.json'), tasks_sha256=sha(tasks)))
        physical = dict(config_sha256=sha(experiment/'config.json'), dataset_sha256='d'*64, reward_scale=100.)
        return dict(experiment=str(experiment), config=cfg, source_round=round_index, rows=rows,
            tasks=tasks, physical=physical)

    def admit_part(self, part, excluded):
        row = part['rows'][0]
        with mock.patch.object(backend, 'context', return_value=(part['config'], None, part['physical'])):
            part['replay'] = core.admit_collect(Path(part['experiment']), seed=0,
                source_round=part['source_round'], count=1, tasks_sha256=sha(part['tasks']),
                config_sha256=part['physical']['config_sha256'], checkpoint_sha256=row['checkpoint_sha256'],
                collection_noise_seed=row['noise_seed'], evaluation_noise_sha256=excluded,
                value=self.fixture.schema, cache={})

    def continuation(self, payload=None, *, resume=False):
        models = copy.deepcopy(self.fixture.make_core(self.root_source, self.first_settings)[0].models)
        opts = {k:(torch.optim.AdamW(m.parameters(), lr=99.) if k=='critic'
                   else torch.optim.Adam(m.parameters(), lr=99.)) for k,m in models.items()}
        generators = {k:torch.Generator().manual_seed(999) for k in core.GENERATOR_KEYS}
        return core.Continuation(models, opts, generators, self.source['payload'] if payload is None else payload,
            self.binding, frozen_reference=self.source, resume=resume)

    def finish(self, folder, state, history=()):
        with redirect_stdout(io.StringIO()):
            runner.run_loop(folder, state, self.source, self.fixture.offline, self.replay,
                self.runner_binding, history, diagnostic_interval=2)

    def request(self):
        return dict(version=runner.VERSION, cycle=2, experiment=self.parts[1]['experiment'],
            config_sha256=sha(Path(self.parts[1]['experiment'])/'config.json'),
            declaration_sha256=sha(Path(self.parts[1]['experiment'])/'declaration.json'), seed=0,
            tasks_sha256=sha(self.parts[1]['tasks']), final_outcomes={}, settings=self.settings,
            diagnostic_interval=2, output=str(self.root/'second_train'))

    def test_true_bc_t1_and_separate_scalar_refit_remain_explicit_and_unmodified(self):
        core.equal_tree(self.source['payload'], self.first_payload)
        self.assertEqual(self.source['binding']['provider'], child.VERSION)
        self.assertEqual(self.source['binding']['bc_actor_prefix'], self.root_source['binding']['actor_prefix'])
        self.assertEqual(self.source['binding']['scalar_critic_refit'], self.root_source['binding']['critic_refit'])
        self.assertNotEqual(self.source['binding']['checkpoint'], self.root_source['binding']['checkpoint'])
        self.assertEqual(self.source['payload']['binding'], self.first_state.binding)
        self.assertNotIn('seed', self.source['payload']['binding'])

    def test_real_native_admission_retains_c0_c1_b4_and_union_of_eval_exclusions(self):
        self.assertEqual(self.replay['derived_B'].flatten().tolist(), [4.,4.])
        self.assertEqual([r['raw_native_reward'] for r in self.replay['records']], [-4.,-4.])
        components = self.replay['binding']['components']
        self.assertEqual([p['source_round'] for p in components], [0,1])
        self.assertNotEqual(components[0]['checkpoint_sha256'], components[1]['checkpoint_sha256'])
        for part in components:
            self.assertEqual(part['excluded_evaluation_noise_sha256'], sorted(self.evaluation))
        self.assertEqual(self.replay['binding']['count'], 2)

    def test_second_initial_state_restores_true_t1_actor_adam_and_all_rng(self):
        state = self.continuation().state()
        for name in core.MODEL_KEYS:
            core.equal_tree(state[name], self.first_payload[name])
            core.equal_tree(state['optimizers'][name], self.first_payload['optimizers'][name])
        for name in ('rng', 'cuda_rng', 'generators'):
            core.equal_tree(state[name], self.first_payload[name])
        self.assertEqual(state['step'], 0)
        self.assertEqual(state['parent_steps'], 4)
        self.assertEqual(runner.audit_state(state, self.source, self.binding)['actor_adam_steps'], 1004)

    def test_second_actor_matches_independent_bc_updates_with_frozen_q_teacher_and_adam(self):
        state = self.continuation()
        reference, teacher = copy.deepcopy(state.models['actor']), copy.deepcopy(state.models['flow'])
        opt = torch.optim.Adam(reference.parameters(), lr=1.)
        opt.load_state_dict(copy.deepcopy(self.first_payload['optimizers']['actor']))
        gen = torch.Generator()
        gen.set_state(self.first_payload['generators']['actor'])
        for _ in range(3):
            noise = 2*torch.rand(4,3,2,generator=gen)-1
            with torch.no_grad():
                target = teacher.sample_flow(noise,10)
            loss = 10*(reference(noise,clip=False)[:,reference.movable]-target[:,reference.movable]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
            opt.step()
        self.finish(self.root/'done', state)
        payload = runner.checked_resume(self.root/'done', self.runner_binding, self.binding, self.source)
        runner.checked_completion(self.root/'done', payload, self.runner_binding, self.binding)
        core.equal_tree(payload['actor'], reference.state_dict())
        core.equal_tree(payload['optimizers']['actor'], opt.state_dict())
        core.equal_tree(payload['generators']['actor'], gen.get_state())
        first_bc.checked_frozen_source(payload, self.root_source)
        self.assertEqual(payload['cumulative_steps'], dict(root_actor_updates=1000, first_actor_updates=4,
            second_actor_updates=3, actor_adam_steps=1007, root_critic_updates=0, first_critic_updates=0,
            second_critic_updates=0, critic_adam_steps=0, teacher_updates=0))
        for row in payload['history']:
            if 'gradient_diagnostics' in row:
                self.assertEqual(row['gradient_diagnostics']['q_ascent_grad_norm'],0.)

    def test_signal_resume_exact_model_adam_rng_history_and_probe_does_not_consume(self):
        full = self.continuation()
        self.finish(self.root/'full', full)
        interrupted = self.continuation()
        before = interrupted.state()
        common.gradient_probe(interrupted)
        core.equal_tree(before, interrupted.state())
        runner.STOP = True
        with self.assertRaises(SystemExit) as stopped:
            self.finish(self.root/'partial', interrupted)
        self.assertEqual(stopped.exception.code,85)
        self.assertFalse((self.root/'partial/completed.json').exists())
        payload = runner.checked_resume(self.root/'partial', self.runner_binding, self.binding, self.source)
        self.assertEqual(payload['cumulative_steps']['actor_adam_steps'],1005)
        resumed = self.continuation(payload, resume=True)
        runner.STOP = False
        self.finish(self.root/'partial', resumed, payload['history'])
        core.equal_tree(full.state(), resumed.state())
        self.assertEqual(common.read(self.root/'full/learning_curve.json'), common.read(self.root/'partial/learning_curve.json'))

    def test_resume_rejects_any_changed_diagnostic_q_teacher_optimizer_or_unused_rng(self):
        state = self.continuation()
        self.finish(self.root/'done',state)
        original = runner.checked_resume(self.root/'done',self.runner_binding,self.binding,self.source)
        for change in ('critic','q_opt','teacher_opt','q_rng','flow_rng','count','history'):
            payload = copy.deepcopy(original)
            if change=='critic':
                key = next(n for n,_ in state.models['critic'].named_parameters())
                payload['critic'][key].flatten()[0] += .1
            elif change=='q_opt': payload['optimizers']['critic']['param_groups'][0]['lr'] = .1
            elif change=='teacher_opt': payload['optimizers']['flow']['param_groups'][0]['lr'] = .1
            elif change.endswith('_rng'): payload['generators'][change[:-4]][0] ^= 1
            elif change=='count': payload['cumulative_steps']['critic_adam_steps'] = 1
            else: payload['history'][0]['critic_updates'] = 1
            torch.save(payload,self.root/'done/checkpoint.pt')
            with self.subTest(change=change),self.assertRaises(ValueError):
                runner.checked_resume(self.root/'done',self.runner_binding,self.binding,self.source)

    def test_foreign_source_identity_and_fake_adam_prefix_refused(self):
        for change in ('provider','bc_prefix','scalar_refit','root_seed','actor_adam','q_opt'):
            cfg, verified = copy.deepcopy(self.cfg), copy.deepcopy(self.verified)
            if change=='provider': verified['source']['binding']['version'] = core.VERSION
            elif change=='bc_prefix': verified['identity']['bc_actor_prefix'] = {}
            elif change=='scalar_refit': verified['identity']['scalar_critic_refit']['sha256'] = '0'*64
            elif change=='root_seed': verified['source']['payload']['binding']['seed'] = 1
            elif change=='actor_adam': next(iter(verified['payload']['optimizers']['actor']['state'].values()))['step'] += 1
            else: verified['payload']['optimizers']['critic']['state'] = {0:{'step':torch.tensor(1.)}}
            with self.subTest(change=change),self.assertRaises(ValueError):
                runner.source_from_verified(cfg,verified,0)

    def test_request_rejects_fql_training_implicit_cycle_and_nonzero_q(self):
        value = self.request()
        self.assertIs(runner.request_settings(value),value)
        for change in ('version','cycle','q','q_bool','critic','missing','odd','nan','nested'):
            bad = copy.deepcopy(value)
            if change=='version': bad['version'] = runner.second.VERSION
            elif change=='cycle': bad['cycle'] = 1
            elif change=='q': bad['settings']['q_weight'] = 1.
            elif change=='q_bool': bad['settings']['q_weight'] = False
            elif change=='critic': bad['settings']['update_critic'] = True
            elif change=='missing': del bad['settings']['alpha']
            elif change=='odd': bad['settings']['batch_size'] = 3
            elif change=='nan': bad['settings']['alpha'] = float('nan')
            else: bad['output'] = str(Path(value['experiment'])/'nested')
            with self.subTest(change=change),self.assertRaises(ValueError): runner.request_settings(bad)

    def test_matched_actor_settings_and_bc_provider_required_for_binding(self):
        for key, value in (('batch_size',8),('alpha',30.),('flow_steps',20),('grad_clip',2.)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.second_binding(self.source,self.replay,dict(self.settings,**{key:value}))
        source = copy.deepcopy(self.source)
        source['binding']['provider'] = runner.second.child.VERSION
        with self.assertRaises(ValueError): runner.second_binding(source,self.replay,self.settings)

    def test_unresolved_missing_or_nan_replay_cannot_silently_be_filtered(self):
        for change in ('unready','mask','nan','negative','missing','schema'):
            parts = copy.deepcopy(self.parts)
            if change=='unready': parts[1]['replay']['ready'] = False
            elif change=='mask': parts[1]['replay']['mask'][0] = False
            elif change=='nan': parts[1]['replay']['derived_B'][0] = float('nan')
            elif change=='negative': parts[1]['replay']['derived_B'][0] = -1.
            elif change=='missing': parts = parts[:1]
            else: parts[1]['replay']['utility_spec_sha256'] = '0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):
                runner.combine_replay(parts,self.source,self.evaluation)

    def test_foreign_role_round_arm_seed_checkpoint_and_eval_noise_rejected(self):
        path = Path(self.parts[1]['replay']['records'][0]['request']['path'])
        saved = path.read_bytes()
        for key,value in (('role','evaluation'),('training_feedback_eligible',False),('round',0),
                ('arm','foreign'),('seed',1),('source_checkpoint',self.root_source['binding']['checkpoint']['path']),
                ('noise_sha256',self.evaluation[0])):
            parts = copy.deepcopy(self.parts)
            row = json.loads(saved)
            row[key] = value
            path.write_text(json.dumps(row))
            parts[1]['replay']['records'][0]['request'] = bind(path)
            parts[1]['replay']['binding']['records'] = parts[1]['replay']['records']
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.combine_replay(parts,self.source,self.evaluation)
        path.write_bytes(saved)

    def test_duplicate_collect_or_changed_tasks_cannot_reweight_replay(self):
        parts = copy.deepcopy(self.parts)
        parts[1]['replay']['records'][0]['noise_sha256'] = parts[0]['replay']['records'][0]['noise_sha256']
        parts[1]['replay']['binding']['records'] = parts[1]['replay']['records']
        with self.assertRaises(ValueError): runner.combine_replay(parts,self.source,self.evaluation)
        self.parts[0]['tasks'].write_text('changed')
        with self.assertRaises(ValueError): runner.combine_replay(self.parts,self.source,self.evaluation)

    def test_completion_cannot_claim_fql_q_updates_wrong_parent_or_missing_population(self):
        state = self.continuation()
        self.finish(self.root/'done',state)
        payload = runner.checked_resume(self.root/'done',self.runner_binding,self.binding,self.source)
        marker = self.root/'done/completed.json'
        original = common.read(marker)
        for key,value in (('version',runner.second.VERSION),('critic_updates',1),('cycle',1),('replay_counts',[1])):
            marker.write_text(json.dumps(dict(original,**{key:value})))
            with self.subTest(key=key),self.assertRaises(ValueError):
                runner.checked_completion(self.root/'done',payload,self.runner_binding,self.binding)

    def test_load_inputs_rechecks_both_real_native_populations_with_full_eval_union(self):
        cfg = dict(self.parts[1]['config'], **self.cfg,
            parent_training={'0':dict(request=bind(self.first_request_path),checkpoint=self.source['binding']['checkpoint'])})
        request = self.request()
        by_path = {p['experiment']:p for p in self.parts}
        def physical_context(path):
            part = by_path[str(Path(path))]
            return part['config'],None,part['physical']
        with mock.patch.object(child,'settings',return_value=(cfg,{},self.parts[1]['physical'])), \
                mock.patch.object(child,'verify_export',return_value=self.parts[1]['rows']), \
                mock.patch.object(child,'checked_parent',return_value=self.verified), \
                mock.patch.object(runner.original_export,'verify_export',return_value=self.parts[0]['rows']), \
                mock.patch.object(runner,'RUNTIME',self.runtime), \
                mock.patch.object(backend,'context',side_effect=physical_context), \
                mock.patch.object(core,'admit_collect',wraps=core.admit_collect) as admissions:
            actual_cfg,_,source,replay,binding = runner.load_inputs(request)
        self.assertEqual(admissions.call_count,2)
        self.assertEqual([c.kwargs['source_round'] for c in admissions.call_args_list],[0,1])
        for call in admissions.call_args_list:
            self.assertEqual(call.kwargs['evaluation_noise_sha256'],sorted(self.evaluation))
        self.assertEqual(binding,self.binding)
        core.equal_tree(replay,self.replay)
        self.assertEqual(actual_cfg,cfg)
        core.equal_tree(source['payload'],self.first_payload)


if __name__ == '__main__':
    unittest.main()
