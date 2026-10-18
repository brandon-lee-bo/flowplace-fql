"""Bounded CPU synthetic fits/configs and frozen-loop equivalence; no EDA/GPU."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_bounded_actor as adapter
from . import qstudy_bounded_screen as screen
from . import qstudy_actor as frozen
from .qstudy_branch import atomic_json, tree_sha
from .qstudy_data import bind, sha
from .qstudy_models import frozen_critic
from .test_qstudy_actor import TinyPolicy


class BoundedActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run bounded actor tests through Slurm CPU')
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='bounded-actor-test-',dir=os.environ['TMPDIR'])
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.dict(screen.CONFIG, steps=2))
        self.stack.enter_context(mock.patch.dict(adapter.FIXED, offline_actor_steps=4))
        self.stack.enter_context(mock.patch('subprocess.run',side_effect=AssertionError('No jobs/EDA')))
        self.cache = dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.) for n in 'abc'},
            net_info={'n':{'nodes':{n:{} for n in 'abc'}}})
        self.problem = SimpleNamespace(x=torch.zeros(3,2),
            cond=SimpleNamespace(is_ports=torch.zeros(3,dtype=torch.bool)),benchmark_info=self.cache)
        self.stack.enter_context(mock.patch('baseline.api.load_problem',return_value=self.problem))
        self.cache_path = self.root/'cache/bp_fe.pt'
        self.cache_path.parent.mkdir()
        torch.save(self.cache,self.cache_path)
        self.dataset = self.root/'data/dataset.npz'
        self.dataset.parent.mkdir()
        self.dataset.write_bytes(b'synthetic dataset identity; arrays injected separately')
        atomic_json(self.dataset.with_name('manifest.json'),dict(design='bp_fe'))
        self.dataset.with_name('rows.jsonl').write_text('{}\n')
        self.data = {}
        generator = torch.Generator().manual_seed(17)
        for split,n in [('train',384),('validation',16)]:
            x = torch.rand(n,3,2,generator=generator)*1.2-.6
            native = torch.stack((100+3*x[:,0,0],-(.4+.1*x[:,1,0]),-(25+x[:,2,1]),
                (3*x[:,0,1]).clamp_min(0),(4*x[:,1,1]).clamp_min(0),
                torch.ones(n)*.2,torch.ones(n)*1000),-1).double().numpy()
            self.data[split] = dict(raw_action=x.numpy().reshape(n,-1),executed_action=(x*.9).numpy().reshape(n,-1),
                metrics=native,metric_mask=np.isfinite(native),family=np.arange(n)//2+(0 if split=='train' else 192),
                source_row=np.arange(n),variant=np.arange(n)%2,status=np.full(n,'success',dtype='<U32'))
        self.data['behavior_action'] = self.data['train']['raw_action'][:256].copy()
        self.data['behavior_family'] = np.arange(256)
        self.inputs = dict(dataset_path=str(self.dataset),dataset_sha256=sha(self.dataset),
            manifest_sha256=sha(self.dataset.with_name('manifest.json')),rows_sha256=sha(self.dataset.with_name('rows.jsonl')),
            cache_path=str(self.cache_path),cache_sha256=sha(self.cache_path))
        self.arrays = {s:screen.arrays(self.data[s],s) for s in ['train','validation']}
        def inputs(_):
            current = dict(self.inputs,dataset_sha256=sha(self.dataset),cache_sha256=sha(self.cache_path))
            return self.arrays,dict(design='bp_fe'),current
        self.stack.enter_context(mock.patch.object(screen,'load_inputs',side_effect=inputs))
        self.stack.enter_context(mock.patch.object(adapter,'load_dataset',return_value=(self.data,{})))
        output = self.root/'screen'
        output.mkdir()
        sources = {name:sha(adapter.ROOT/name) for name in screen.SOURCE_NAMES}
        for name in sources:
            dest = output/'source_snapshot'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(adapter.ROOT/name,dest)
        folds = screen.family_folds(self.arrays['train']['families'],3)
        folds.append(dict(name='refit',train_indices=list(range(384)),target_indices=list(range(16))))
        tasks = [dict(index=i,objective=o,seed=s) for i,(o,s) in
                 enumerate((o,s) for o in screen.OBJECTIVES for s in range(3))]
        self.plan = dict(schema='qstudy_bounded_scalar_screen_v1',output=str(output),design='bp_fe',
            inputs=self.inputs,source_sha256=sources,config=screen.CONFIG.copy(),
            objectives={k:list(v) for k,v in screen.OBJECTIVES.items()},tasks=tasks,folds=folds,
            fold_plan_sha256=screen.json_sha(folds),train_source_ids=self.arrays['train']['source_ids'],
            train_source_ids_sha256=screen.json_sha(self.arrays['train']['source_ids']),
            validation_source_ids=self.arrays['validation']['source_ids'],
            native_reward_unchanged=True,actor_advanced=False,heldout_used=False,backend_calls=0)
        self.plan_path = output/'plan.json'
        atomic_json(self.plan_path,self.plan)
        for task in tasks:
            folder = output/task['objective']/f"seed_{task['seed']}"
            run = dict(schema=self.plan['schema'],**self.inputs,plan_sha256=sha(self.plan_path),task=task,
                source_sha256=sources,fold_plan_sha256=self.plan['fold_plan_sha256'],threads=1,
                torch_version=torch.__version__,numpy_version=np.__version__)
            context = dict(**self.arrays['train'],cache=self.cache,movable=torch.ones(3,dtype=torch.bool),binding=run)
            args = SimpleNamespace(**screen.CONFIG,objective=task['objective'],seed=task['seed'],output=folder)
            outcomes = [screen.fit(args,context,f['name'],f['train_indices'],f['target_indices'],
                self.arrays['validation'] if f['name']=='refit' else context) for f in folds]
            atomic_json(folder/'summary.json',dict(status='completed',binding=run,objective=task['objective'],
                seed=task['seed'],outcomes=outcomes,**screen.summarize(outcomes)))
            atomic_json(folder/'completed.json',dict(status='completed',binding=run,
                summary_sha256=sha(folder/'summary.json'),backend_calls=0,actor_advanced=False))
        self.base = dict(adapter.FIXED,design='bp_fe',arm='reference',run_prefix='reference',
            critic_variant='legacy',weights=[1,0,0,0,0],online_rounds=2,online_updates=600,
            online_samples=8,collection_noise_seed=4500000,evaluation_purpose='development_validation',
            dataset_path=str(self.dataset),dataset_sha256=sha(self.dataset),cache_sha256=sha(self.cache_path),
            common_checkpoints={},common_checkpoint_sha256={},critic_checkpoints={},critic_checkpoint_sha256={})
        for seed in range(3):
            experiment = self.root/f'common_{seed}'
            folder = experiment/f'seed_{seed}/round_0'
            folder.mkdir(parents=True)
            atomic_json(experiment/'config.json',dict(synthetic=True,seed=seed))
            torch.manual_seed(200+seed)
            torch.save(dict(flow=TinyPolicy(self.problem).state_dict(),actor=TinyPolicy(self.problem).state_dict()),folder/'common.pt')
            atomic_json(folder/'training_status.json',dict(status='completed',seed=seed,round=0,
                graph_cache_sha256=sha(self.cache_path),no_heldout_labels_used=True,
                config_sha256=sha(experiment/'config.json')))
            self.base['common_checkpoints'][str(seed)] = str(folder/'common.pt')
            self.base['common_checkpoint_sha256'][str(seed)] = sha(folder/'common.pt')
        self.base_path = self.root/'base/config.json'
        self.base_path.parent.mkdir()
        atomic_json(self.base_path,self.base)
        self.args = SimpleNamespace(plan=self.plan_path,base_config=self.base_path,output=self.root/'actors')
        frozen.STOP = False

    def tearDown(self):
        screen.STOP = False
        frozen.STOP = False
        self.stack.close()
        self.temp.cleanup()

    def prepared(self):
        return adapter.prepare(self.args)

    def test_three_objective_configs_are_immutable_and_keep_native_reward(self):
        before = sha(self.plan_path),sha(self.dataset),sha(self.base_path)
        manifest = self.prepared()
        self.assertEqual(manifest['critic_fits'],36)
        self.assertEqual(manifest['actor_jobs_submitted'],0)
        self.assertEqual(self.prepared(),manifest)
        scales = []
        for objective,(arm,_) in adapter.ARMS.items():
            cfg = adapter.validate_config(self.args.output/arm,0)
            self.assertEqual(cfg['online_rounds'],0)
            self.assertEqual(cfg['weights'],list(screen.OBJECTIVES[objective]))
            self.assertEqual(cfg['physical_failure_reward'],-4)
            self.assertEqual(cfg['common_checkpoint_sha256'],self.base['common_checkpoint_sha256'])
            self.assertEqual(cfg['bounded_behavior']['targets'],256)
            scales.append(cfg['bounded_scale_identity'])
        self.assertEqual(len(set(scales)),1)
        self.assertEqual(before,(sha(self.plan_path),sha(self.dataset),sha(self.base_path)))
        path = self.args.output/'bounded_wl/config.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'publication gate'):
            adapter.validate_config(path.parent,0)
        altered = adapter.read(path)
        altered['alpha']=11
        atomic_json(path,altered)
        with self.assertRaisesRegex(ValueError,'Immutable'):
            self.prepared()

    def test_missing_fit_source_and_normalization_tampering_are_refused(self):
        fit = Path(self.plan['output'])/'timing/seed_2/cv_1'
        marker = fit/'completed.json'
        raw = marker.read_bytes()
        marker.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepared()
        self.assertFalse(self.args.output.exists())
        marker.write_bytes(raw)
        # Re-signing the file checksum is insufficient to hide changed buffers.
        path = fit/'resume.pt'
        saved = torch.load(path,weights_only=False)
        saved['model']['feature_mean'][0] += .01
        torch.save(saved,path)
        done = adapter.read(marker)
        done['checkpoint_sha256']=sha(path)
        atomic_json(marker,done)
        folder=fit.parent
        summary=adapter.read(folder/'summary.json')
        summary['outcomes'][1]=done
        atomic_json(folder/'summary.json',summary)
        rootdone=adapter.read(folder/'completed.json')
        rootdone['summary_sha256']=sha(folder/'summary.json')
        atomic_json(folder/'completed.json',rootdone)
        with self.assertRaisesRegex(ValueError,'normalization buffer differs'):
            self.prepared()

    def test_bc_targets_and_common_metadata_must_match_original(self):
        original = copy.deepcopy(self.data)
        original['behavior_action'][0,0] += .1
        with mock.patch.object(adapter,'load_dataset',side_effect=[(original,{}),(self.data,{})]), \
             self.assertRaisesRegex(ValueError,'BC targets/families differ'):
            self.prepared()
        path = Path(self.base['common_checkpoints']['1']).with_name('training_status.json')
        metadata=adapter.read(path)
        metadata['seed']=2
        atomic_json(path,metadata)
        with self.assertRaisesRegex(ValueError,'Common checkpoint metadata'):
            self.prepared()

    def test_partial_publication_is_resumable_and_online_is_explicitly_refused(self):
        self.prepared()
        (self.args.output/'preparation.json').unlink()
        exp=self.args.output/'bounded_wl'
        with self.assertRaises(FileNotFoundError):
            adapter.validate_config(exp,0)
        self.prepared()
        with self.assertRaisesRegex(ValueError,'online rounds are not implemented'):
            adapter.validate_config(exp,0,1)
        with self.assertRaisesRegex(ValueError,'seed differs'):
            adapter.validate_config(exp,3)
        cfg=adapter.read(exp/'config.json')
        cfg['online_rounds']=1
        atomic_json(exp/'config.json',cfg)
        with self.assertRaisesRegex(ValueError,'round/seed differs'):
            adapter.validate_config(exp,0)

    def test_runtime_isolation_and_scalar_q_gradients_have_true_units(self):
        a,b=adapter.isolated_runtime(),adapter.isolated_runtime()
        for name in ['train','export','stop']:
            self.assertIs(a[name].__code__,getattr(frozen,name).__code__)
            self.assertIs(a[name].__globals__,a)
        a['stop']()
        self.assertTrue(a['STOP'])
        self.assertFalse(b['STOP'])
        self.assertFalse(frozen.STOP)
        _,_,context,selected,specs,_=adapter.checked_study(self.plan_path)
        info=selected[('timing',0)]
        model=adapter.checked_model(context,specs['timing'],0,list(range(384)),
            info['checkpoint']['path'],adapter.read(info['completion']['path']))
        x=context['x'][:4].clone().requires_grad_(True)
        with frozen_critic(model):
            q=adapter.reward(model,x,None,None)
            torch.testing.assert_close(q,-model(x).mean((1,2)))
            self.assertFalse(torch.allclose(q,model.reward(x,[1.])))
            q.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @contextmanager
    def cpu_runtime(self, namespace):
        def setup(exp,seed):
            cfg=adapter.read(exp/'config.json')
            context=dict(**self.arrays['train'],cache=self.cache,movable=torch.ones(3,dtype=torch.bool),
                binding=dict(dataset_sha256=self.inputs['dataset_sha256']))
            path=Path(cfg['critic_checkpoints'][str(seed)])
            model=adapter.checked_model(context,cfg['bounded_utility_spec'],seed,list(range(384)),path,
                adapter.read(path.with_name('completed.json')))
            labels=screen.labels(context,list(range(384)),cfg['bounded_utility_spec'])
            return cfg,self.problem,context['x'],labels['cost'].float()[:,None],labels['valid'][:,None], \
                torch.as_tensor(self.data['behavior_action']).reshape(-1,3,2),model,torch.as_tensor(context['families'])
        namespace.update(setup=setup,GraphPolicy=TinyPolicy,read_replay=lambda *_:None,legalize=lambda x,_:x,
                         legality=lambda *_:dict(legal=False))
        with ExitStack() as stack:
            for name in ['tensor','rand','randint']:
                original=getattr(torch,name)
                def redirected(*args,_original=original,**kwargs):
                    if str(kwargs.get('device','')).startswith('cuda'):
                        kwargs['device']='cpu'
                    return _original(*args,**kwargs)
                stack.enter_context(mock.patch.object(torch,name,side_effect=redirected))
            original_generator,original_load=torch.Generator,torch.load
            stack.enter_context(mock.patch.object(torch,'Generator',side_effect=lambda device='cpu':original_generator()))
            def load(*args,**kwargs):
                if str(kwargs.get('map_location','')).startswith('cuda'):
                    kwargs['map_location']='cpu'
                return original_load(*args,**kwargs)
            stack.enter_context(mock.patch.object(torch,'load',side_effect=load))
            stack.enter_context(mock.patch.object(torch.cuda,'get_rng_state_all',return_value=[]))
            stack.enter_context(mock.patch.object(torch.cuda,'set_rng_state_all'))
            stack.enter_context(mock.patch.object(frozen.signal,'signal'))
            yield

    def test_exact_cpu_resume_frozen_q_flow_and_same_actual_evaluation_noises(self):
        self.prepared()
        exp=self.args.output/'bounded_timing'
        args=SimpleNamespace(experiment=exp,seed=0,round=0)
        namespace=adapter.isolated_runtime()
        with self.cpu_runtime(namespace):
            namespace['train'](args)
        checkpoint=exp/'seed_0/round_0/checkpoint.pt'
        reference=torch.load(checkpoint,weights_only=False)
        common=torch.load(self.base['common_checkpoints']['0'],weights_only=False)
        self.assertEqual(tree_sha(reference['flow']),tree_sha(common['flow']))
        self.assertEqual(tree_sha(reference['critic']),tree_sha(torch.load(
            adapter.read(exp/'config.json')['critic_checkpoints']['0'],weights_only=False)['model']))
        self.assertNotEqual(tree_sha(reference['actor']),tree_sha(common['actor']))
        shutil.rmtree(exp/'seed_0')
        namespace=adapter.isolated_runtime()
        original_step=torch.optim.Adam.step
        calls=[]
        def interrupt(optimizer,*a,**kw):
            result=original_step(optimizer,*a,**kw)
            calls.append(1)
            if len(calls)==2:
                namespace['stop']()
            return result
        with self.cpu_runtime(namespace),mock.patch.object(torch.optim.Adam,'step',new=interrupt), \
             self.assertRaises(SystemExit) as error:
            namespace['train'](args)
        self.assertEqual(error.exception.code,85)
        namespace=adapter.isolated_runtime()
        with self.cpu_runtime(namespace):
            namespace['train'](args)
            namespace['export'](args)
        actual=torch.load(checkpoint,weights_only=False)
        for key in ['flow','actor','critic','optimizers','rng','cuda_rng','generators','step','binding']:
            self.assertEqual(tree_sha(reference[key]),tree_sha(actual[key]),key)
        records=[json.loads(line) for line in (checkpoint.parent/'requests/tasks.jsonl').read_text().splitlines()]
        self.assertEqual(len(records),8)
        noise=2*torch.rand(8,3,2,generator=torch.Generator().manual_seed(4600000))-1
        for i,row in enumerate(records):
            self.assertEqual(row['role'],'evaluation')
            self.assertFalse(row['training_feedback_eligible'])
            self.assertEqual(row['noise_sha256'],frozen.action_hash(noise[i].numpy()))
            cfg=adapter.read(exp/'config.json')
            binding=dict(config_sha256=sha(exp/'config.json'),dataset_sha256=cfg['dataset_sha256'])
            adapter.backend.checked_request(row,args,cfg,binding)


if __name__=='__main__':
    unittest.main()
