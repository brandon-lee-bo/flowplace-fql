"""CPU Slurm contracts for a duration-only actor continuation.

Small synthetic prefixes exercise immutable import and export. One test uses
the actual unchanged 1000/2000-step actor loop with tiny CPU networks and checks
every optimizer/RNG/generator tensor against uninterrupted execution. No graph
policy GPU inference, physical tools, production config or scheduler call runs.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import qstudy_duration as duration
from . import qstudy_actor as actor
from . import qstudy_backend as backend
from . import test_qstudy_actor as fixtures
from .graph_o2o_archive import atomic_json, sha


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(6,2)

    def forward(self,x):return self.linear(x.flatten(1))


class TinyMetric:
    def __init__(self,cache):self.sizes=np.zeros((3,2),dtype=np.float32)
    def __call__(self,x):return 0.


class InterruptedImport(RuntimeError):pass


class DurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:raise RuntimeError('Run duration contracts under CPU Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.fx=fixtures.ActorRunnerTests('test_common_initialization_frozen_offline_q_and_zero_q_bc')
        self.fx.setUp();self.root=self.fx.root
        self.fx.problem.cond.is_ports[-1]=True
        torch.manual_seed(17)
        self.fx.critic_template=TinyCritic()
        torch.save(dict(model=self.fx.critic_template.state_dict()),self.fx.qpath)
        flow,one_step=fixtures.TinyPolicy(self.fx.problem),fixtures.TinyPolicy(self.fx.problem)
        torch.save(dict(flow=flow.state_dict(),actor=one_step.state_dict()),self.fx.common)
        runtime=self.root/'runtime';cache=runtime/'benchmark_cache/bp_fe.pt'
        cache.parent.mkdir(parents=True);cache.write_bytes(b'Explicit synthetic graph identity')
        self.fx.cfg.update(seeds=[0,1,2],critic_variant='legacy',weights=[1,0,0,0,0],
            alpha=10.,q_weight=1.,offline_actor_steps=1000,offline_critic_update=False,
            online_flow_update=False,flow_steps=10,evaluation_samples=8,evaluation_noise_seed=4600000,
            evaluation_purpose='development_validation',cache_sha256=sha(cache),backend_threads=8)
        self.fx.cfg['common_checkpoint_sha256']['0']=sha(self.fx.common)
        self.fx.cfg['critic_checkpoint_sha256']['0']=sha(self.fx.qpath)
        for field in ('common_checkpoints','critic_checkpoints','common_checkpoint_sha256','critic_checkpoint_sha256'):
            self.fx.cfg[field]={str(s):self.fx.cfg[field]['0'] for s in (0,1,2)}
        self.source=self.fx.experiment('source');self.target=self.root/'duration'
        self.stack=ExitStack()
        self.stack.enter_context(mock.patch.object(actor,'RUNTIME',runtime))
        self.stack.enter_context(mock.patch.object(backend,'context',side_effect=self.context))
        self.stack.enter_context(mock.patch('subprocess.run',side_effect=AssertionError('No tool or scheduler call permitted')))
        for seed in (0,1,2):self.synthetic_prefix(seed)
        self.before=duration.prefix.inventory(self.source)

    def tearDown(self):
        self.stack.close();self.fx.tearDown()

    def context(self,exp):
        cfg=duration.read(Path(exp)/'config.json')
        return cfg,None,dict(config_sha256=sha(Path(exp)/'config.json'),dataset_sha256=cfg['dataset_sha256'])

    def synthetic_prefix(self,seed):
        """Opaque saved state fixture, not a claim of 1000 actual updates."""
        folder=self.source/f'seed_{seed}/round_0';folder.mkdir(parents=True,exist_ok=True)
        common=torch.load(self.fx.common,weights_only=False)
        flow,one_step=fixtures.TinyPolicy(self.fx.problem),fixtures.TinyPolicy(self.fx.problem)
        critic=copy.deepcopy(self.fx.critic_template)
        flow.load_state_dict(common['flow']);one_step.load_state_dict(common['actor'])
        opts=dict(flow=torch.optim.Adam(flow.parameters(),lr=1e-3),
            actor=torch.optim.Adam(one_step.parameters(),lr=1e-3),
            critic=torch.optim.AdamW(critic.parameters(),lr=1e-3))
        # Nonempty actor Adam moments ensure weight-only imports cannot pass.
        opts['actor'].zero_grad();one_step(torch.zeros(1,3,2)).square().sum().backward();opts['actor'].step()
        binding=duration.expected_binding(self.source,self.fx.cfg,seed)
        saved=dict(flow=flow.state_dict(),actor=one_step.state_dict(),critic=critic.state_dict(),
            optimizers={k:v.state_dict() for k,v in opts.items()},rng=torch.get_rng_state(),cuda_rng=[],
            generators={k:torch.Generator().manual_seed(100+seed+i).get_state() for i,k in enumerate(('q','actor','flow'))},
            binding=binding,step=1000)
        torch.save(saved,folder/'checkpoint.pt');atomic_json(folder/'binding.json',binding)
        atomic_json(folder/'completed.json',dict(status='completed',steps=1000,binding=binding,
            checkpoint_sha256=sha(folder/'checkpoint.pt'),offline_critic_frozen=True,online_flow_update=False))
        (folder/'learning_curve.jsonl').write_text(json.dumps(dict(step=1000,synthetic_fixture=True))+'\n')
        self.emit(self.source,seed)

    def emit(self,exp,seed,*,noise_seed=4600000):
        cfg=duration.read(exp/'config.json');folder=exp/f'seed_{seed}/round_0'
        checkpoint=folder/'checkpoint.pt';dest=folder/'requests';dest.mkdir(exist_ok=True)
        noises=2*torch.rand(8,3,2,generator=torch.Generator().manual_seed(noise_seed))-1
        records=[]
        for index,noise in enumerate(noises):
            sample=dest/cfg['arm']/'evaluation'/str(index);sample.mkdir(parents=True,exist_ok=True)
            raw=np.zeros((3,2),dtype=np.float32);n=noise.numpy()
            np.savez_compressed(sample/'actions.npz',noise=n,raw_action=raw,action=raw)
            row=dict(arm=cfg['arm'],seed=seed,round=0,role='evaluation',index=index,graph='bp_fe',
                terminal_mask=0,training_feedback_eligible=False,noise_seed=noise_seed,
                evaluation_purpose=cfg['evaluation_purpose'],source_checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                config_sha256=sha(exp/'config.json'),raw_action=raw.tolist(),executed_action=raw.tolist(),
                action_sha256=actor.action_hash(raw),raw_action_sha256=actor.action_hash(raw),noise_sha256=actor.action_hash(n),
                sample_dir=str(sample),run_id=f'synthetic_{cfg["arm"]}_{seed}_{index}')
            atomic_json(sample/'request.json',row);records.append(row)
        tasks=dest/'tasks.jsonl';tasks.write_text(''.join(json.dumps(r)+'\n' for r in records))
        atomic_json(dest/'completed.json',dict(status='completed',count=8,tasks_sha256=sha(tasks),
            config_sha256=sha(exp/'config.json'),evaluation_never_replayed=True))

    def args(self,target=None):
        return SimpleNamespace(source=self.source,source_sha256=sha(self.source/'config.json'),
            output=target or self.target,arm='duration',run_prefix='dur2k')

    def finish_synthetic_target(self,seed=0):
        folder=self.target/f'seed_{seed}/round_0';saved=torch.load(folder/'resume.pt',weights_only=False)
        saved['step']=2000;torch.save(saved,folder/'checkpoint.pt');(folder/'resume.pt').unlink()
        atomic_json(folder/'completed.json',dict(status='completed',steps=2000,binding=saved['binding'],
            checkpoint_sha256=sha(folder/'checkpoint.pt'),offline_critic_frozen=True,online_flow_update=False))

    def test_full_payload_import_source_immutability_and_idempotence(self):
        gate=duration.prepare(self.args())
        self.assertEqual(gate['coordinate_contract'],duration.COORDINATE_CONTRACT)
        self.assertEqual(gate['new_backend_calls'],0)
        self.assertEqual(gate['export_role_overlay'],dict(online_rounds=0))
        for seed in (0,1,2):
            original=torch.load(self.source/f'seed_{seed}/round_0/checkpoint.pt',weights_only=False)
            imported=torch.load(self.target/f'seed_{seed}/round_0/resume.pt',weights_only=False)
            self.assertEqual(duration.prefix.state_fingerprints(original),duration.prefix.state_fingerprints(imported))
            self.assertNotEqual(original['binding']['config_sha256'],imported['binding']['config_sha256'])
            self.assertFalse((self.target/f'seed_{seed}/round_0/completed.json').exists())
            duration.checked_import(self.target,seed)
        before=duration.prefix.inventory(self.target)
        self.assertEqual(gate,duration.prepare(self.args()))
        self.assertEqual(before,duration.prefix.inventory(self.target))
        self.assertEqual(self.before,duration.prefix.inventory(self.source))
        self.finish_synthetic_target();duration.checked_import(self.target,0)
        completed=duration.prefix.inventory(self.target)
        duration.prepare(self.args())
        self.assertEqual(completed,duration.prefix.inventory(self.target))

    def test_strict_configuration_allowlist_and_fixed_budgets(self):
        source=duration.read(self.source/'config.json')
        target=dict(source,arm='duration',run_prefix='dur2k',offline_actor_steps=2000)
        duration.configurations(source,target)
        for key,value in [('alpha',2.),('q_weight',0.),('online_rounds',0),('dataset_path','other'),
                          ('evaluation_noise_seed',85000000),('online_flow_update',True),('critic_variant','multi_huber')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                duration.configurations(source,dict(target,**{key:value}))
        source['optional_none']=None
        with self.assertRaises(ValueError):duration.configurations(source,target)
        with self.assertRaises(ValueError):duration.configurations(source,dict(target,optional_none=None,arm='../bad'))

    def test_original_checkpoint_completion_config_and_cache_tampering_refused(self):
        paths=(self.source/'config.json',self.source/'seed_0/round_0/checkpoint.pt',
            self.source/'seed_0/round_0/completed.json',Path(actor.RUNTIME)/'benchmark_cache/bp_fe.pt')
        args=self.args()
        for path in paths:
            old=path.read_bytes();path.write_bytes(old+b'changed')
            with self.subTest(path=path),self.assertRaises((ValueError,json.JSONDecodeError)):
                duration.prepare(args)
            path.write_bytes(old)
        self.assertFalse(self.target.exists())

    def test_saved_source_actual_noise_array_tampering_refused(self):
        sample=self.source/'seed_0/round_0/requests/test/evaluation/0'
        path=sample/'actions.npz'
        with np.load(path,allow_pickle=False) as arrays:values={k:arrays[k].copy() for k in arrays.files}
        values['noise'][0,0]+=.01;np.savez_compressed(path,**values)
        with self.assertRaisesRegex(ValueError,'Noise identity'):duration.prepare(self.args())

    def test_cross_seed_actual_noise_pairing_is_required(self):
        self.emit(self.source,1,noise_seed=4600001)
        # Even with the declared seed restored, self-consistent different actual
        # noise arrays cannot masquerade as paired evaluation.
        folder=self.source/'seed_1/round_0/requests';tasks=folder/'tasks.jsonl'
        rows=[json.loads(line) for line in tasks.read_text().splitlines()]
        for row in rows:
            row['noise_seed']=4600000;atomic_json(Path(row['sample_dir'])/'request.json',row)
        tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        done=duration.read(folder/'completed.json');done['tasks_sha256']=sha(tasks);atomic_json(folder/'completed.json',done)
        with self.assertRaisesRegex(ValueError,'same actual eight'):duration.prepare(self.args())

    def test_partial_import_recovers_without_overwriting_published_seed(self):
        original=duration.atomic_torch;calls=[]
        def interrupted(path,value):
            calls.append(1)
            if len(calls)==2:raise InterruptedImport('Synthetic interruption')
            return original(path,value)
        with mock.patch.object(duration,'atomic_torch',side_effect=interrupted),self.assertRaises(InterruptedImport):
            duration.prepare(self.args())
        published=sha(self.target/'seed_0/round_0/resume.pt')
        self.assertFalse((self.target/'duration_import.json').exists())
        with self.assertRaises(FileNotFoundError):duration.checked_import(self.target,0)
        duration.prepare(self.args())
        self.assertEqual(published,sha(self.target/'seed_0/round_0/resume.pt'))
        self.assertFalse((self.target/'.duration_seed_1.staging').exists())
        self.assertEqual(self.before,duration.prefix.inventory(self.source))

    def test_import_optimizer_rng_and_frozen_teacher_tampering_refused(self):
        duration.prepare(self.args());path=self.target/'seed_0/round_0/resume.pt'
        initial=torch.load(path,weights_only=False)
        for key in ('actor','flow','critic','optimizers','rng','generators'):
            saved=copy.deepcopy(initial)
            if key in ('actor','flow','critic'):
                next(v for v in saved[key].values() if v.is_floating_point()).add_(.01)
            elif key=='optimizers':saved[key]['actor']['param_groups'][0]['lr']*=2
            elif key=='rng':saved[key][0]^=1
            else:saved[key]['actor'][0]^=1
            torch.save(saved,path)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'state differs|teacher or Q'):
                duration.checked_import(self.target,0)
        torch.save(initial,path);duration.checked_import(self.target,0)

    def test_private_export_overlay_eight_actual_noises_and_no_module_mutation(self):
        duration.prepare(self.args());self.finish_synthetic_target()
        cfg_before=(self.target/'config.json').read_bytes();stop_before=actor.STOP
        with self.fx.cpu_fixture(),mock.patch.object(actor,'MacroHPWL',TinyMetric),\
             mock.patch.object(actor,'legalize',side_effect=lambda raw,problem:raw),\
             mock.patch.object(actor,'legality',return_value=dict(legal=False)):
            namespace=duration.isolated_runtime(evaluation_only=True)
            self.assertIs(namespace['export'].__code__,actor.export.__code__)
            self.assertEqual(namespace['setup'](self.target,0)[0]['online_rounds'],0)
            self.assertNotIn('online_rounds',self.fx.setup(self.target,0)[0])
            namespace['stop']();self.assertTrue(namespace['STOP']);self.assertEqual(actor.STOP,stop_before)
            duration.run(SimpleNamespace(experiment=self.target,seed=0,stage='export'))
            # Completed-export reuse still validates exact saved request/noise.
            duration.run(SimpleNamespace(experiment=self.target,seed=0,stage='export'))
        cfg,gate=duration.checked_import(self.target,0)
        evidence=duration.verify_export(self.target,0,cfg,gate)
        self.assertEqual(evidence['requests'],8)
        self.assertEqual(cfg_before,(self.target/'config.json').read_bytes())
        from .graph_o2o_backend import load_requests
        rows,_=load_requests(self.target,0,0)
        self.assertEqual([r['noise_sha256'] for r in rows],gate['evaluation_noise_sha256'])
        self.assertTrue(all(r['role']=='evaluation' and not r['training_feedback_eligible'] for r in rows))
        gate=dict(gate,evaluation_noise_sha256=['wrong']*8)
        with self.assertRaisesRegex(ValueError,'fresh noise stream'):duration.verify_export(self.target,0,cfg,gate)

    def test_actual_1000_prefix_2000_continuation_and_preemption_match_uninterrupted(self):
        # Only this contract actually runs full step1000/2000 loops. Tiny CPU
        # policies preserve the frozen loop, optimizer, noise and resume logic.
        shutil.rmtree(self.source/'seed_0')
        full=self.fx.experiment('uninterrupted',offline_actor_steps=2000)
        with self.fx.cpu_fixture():
            duration.isolated_runtime()['train'](SimpleNamespace(experiment=self.source,seed=0,round=0))
            duration.isolated_runtime()['train'](SimpleNamespace(experiment=full,seed=0,round=0))
        self.emit(self.source,0)
        duration.prepare(self.args())
        interrupted=self.root/'interrupted_duration'
        second=self.args(interrupted);second.arm='durationInterrupted';second.run_prefix='durInt'
        duration.prepare(second)
        with self.fx.cpu_fixture():
            duration.run(SimpleNamespace(experiment=self.target,seed=0,stage='train'))
            namespace=duration.isolated_runtime();original_step=torch.optim.Adam.step;calls=[]
            def interrupt(optimizer,*args,**kwargs):
                result=original_step(optimizer,*args,**kwargs);calls.append(1)
                if len(calls)==3:namespace['STOP']=True
                return result
            with mock.patch.object(torch.optim.Adam,'step',new=interrupt),self.assertRaises(SystemExit) as error:
                namespace['train'](SimpleNamespace(experiment=interrupted,seed=0,round=0))
            self.assertEqual(error.exception.code,85)
            resumed=torch.load(interrupted/'seed_0/round_0/resume.pt',weights_only=False)
            self.assertEqual(resumed['step'],1003)
            duration.checked_import(interrupted,0)
            duration.run(SimpleNamespace(experiment=interrupted,seed=0,stage='train'))
            # Resume after successful completion uses checkpoint.pt, no resume.pt.
            completed_sha=sha(self.target/'seed_0/round_0/checkpoint.pt')
            duration.run(SimpleNamespace(experiment=self.target,seed=0,stage='train'))
            self.assertEqual(completed_sha,sha(self.target/'seed_0/round_0/checkpoint.pt'))
        checkpoints=[torch.load(exp/'seed_0/round_0/checkpoint.pt',weights_only=False)
            for exp in (full,self.target,interrupted)]
        reference=duration.prefix.state_fingerprints(checkpoints[0])
        self.assertEqual(reference,duration.prefix.state_fingerprints(checkpoints[1]))
        self.assertEqual(reference,duration.prefix.state_fingerprints(checkpoints[2]))
        initial=torch.load(self.source/'seed_0/round_0/checkpoint.pt',weights_only=False)
        for key in ('flow','critic'):
            self.assertEqual(duration.prefix.tree_sha(initial[key]),duration.prefix.tree_sha(checkpoints[1][key]))
        self.assertFalse(torch.equal(initial['actor']['linear.weight'],checkpoints[1]['actor']['linear.weight']))
        rows=[json.loads(line) for line in (self.target/'seed_0/round_0/learning_curve.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['step'],1001);self.assertEqual(rows[-1]['step'],2000)
        self.assertTrue(all(r['qloss']==0 and r['flowloss']==0 for r in rows))


if __name__=='__main__':unittest.main()
