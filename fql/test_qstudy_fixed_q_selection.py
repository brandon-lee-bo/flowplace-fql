"""Bounded synthetic CPU inference/integrity tests; no real layouts or EDA."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import qstudy_fixed_q_selection as q


class FixedQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run numerical tests through bounded CPU Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='fixed-q-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def write(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def row(self,index=0,arm='flow',seed=0):
        sample=self.root/arm/f's{seed}'/str(index)
        sample.mkdir(parents=True,exist_ok=True)
        noise=np.full((3,2),.01*index,dtype=np.float32)
        raw=np.full((3,2),.01*(index+1),dtype=np.float32)
        executed=raw+np.float32(.4)
        np.savez_compressed(sample/'actions.npz',noise=noise,raw_action=raw,action=executed)
        request=dict(index=index,role='evaluation',round=0,noise_seed=4600000,
            noise_sha256=q.action_sha(noise),raw_action_sha256=q.action_sha(raw),
            action_sha256=q.action_sha(executed),training_feedback_eligible=False,
            raw_action=raw.tolist())
        self.write(sample/'request.json',request)
        return dict(index=index,sample_dir=str(sample),request_sha256=q.sha(sample/'request.json'),
            actions_npz_sha256=q.sha(sample/'actions.npz'),noise_sha256=request['noise_sha256'],
            predicted_q=None,state='success',macro_hpwl=float(8-index),common_utility=-(100+index)/100,
            metrics={m:float(100+index) for m in q.METRICS},native={'physical_run':str(sample/'native')})

    @staticmethod
    def simple_pools():
        pools=[]
        for arm_i,arm in enumerate(q.ARMS):
            for seed in range(3):
                rows=[]
                for i in range(8):
                    value=100+10*arm_i+seed+i
                    rows.append(dict(index=i,state='success',predicted_q=float(i if arm!='bc' else 7-i),
                        recorded_cuda_q=None,macro_hpwl=float(8-i),common_utility=-value/100,
                        noise_sha256=str(i),actual_noise=np.full((3,2),i,dtype=np.float32),
                        metrics={m:float(value) for m in q.METRICS}))
                pools.append(dict(arm=arm,seed=seed,rows=rows))
        return pools

    def test_full_state_includes_buffers_dtype_and_bitwise_signed_zero(self):
        state={'weight':torch.tensor([1.,0.]),'buffer':torch.tensor(2)}
        q.state_equal(state,copy.deepcopy(state))
        changes=[{'weight':torch.tensor([1.,-0.]),'buffer':torch.tensor(2)},
                 {'weight':state['weight'].double(),'buffer':state['buffer']},
                 {'weight':state['weight'],'buffer':torch.tensor(3)},
                 {'weight':torch.tensor([float('nan'),0.]),'buffer':state['buffer']},
                 {'weight':state['weight']}]
        for changed in changes:
            with self.subTest(changed=changed),self.assertRaises(ValueError):q.state_equal(state,changed)

    def test_initial_checkpoint_reconstruction_full_train_no_rng_training_or_policy(self):
        from baseline import api
        cache=dict(node_id_to_name=['a','b','c'],ratio_x=10/224,ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.) for n in 'abc'},
            net_info={'n':{'nodes':{n:{} for n in 'abc'}}})
        runtime=self.root/'runtime';(runtime/'benchmark_cache').mkdir(parents=True)
        cache_path=runtime/'benchmark_cache/bp_fe.pt';torch.save(cache,cache_path)
        train=dict(raw_action=np.random.default_rng(7).uniform(-.8,.8,(384,6)).astype(np.float32),
            metrics=np.tile(np.asarray([100.,-.4,-10.,2.,1.,.2,1000.]),(384,1)),
            metric_mask=np.ones((384,7),dtype=bool),family=np.arange(192).repeat(2))
        dataset=self.root/'data/dataset.npz';dataset.parent.mkdir();dataset.write_bytes(b'fixture bound data')
        manifest=self.write(dataset.with_name('manifest.json'),{'test':True})
        movable=torch.tensor([True,True,False])
        x=torch.tensor(train['raw_action']).reshape(384,3,2)
        model,_=q.make_model(cache,x,torch.tensor(train['metrics'][:,:5],dtype=torch.float32),
            torch.tensor(train['metric_mask'][:,:5]),movable,'legacy',torch.tensor(train['family']),0)
        # Nonzero learned head proves state is loaded, rather than only initialized.
        with torch.no_grad():model.members[0][-1].bias.fill_(.2)
        path=self.root/'refit/resume.pt';path.parent.mkdir()
        binding=dict(dataset_sha256=q.sha(dataset),manifest_sha256=q.sha(manifest),cache_sha256=q.sha(cache_path),source_sha256={})
        saved=dict(binding=binding,step=2000,train_indices=list(range(384)),model_config={'legacy':True},model=model.state_dict())
        torch.save(saved,path)
        self.write(path.with_name('completed.json'),dict(status='completed',checkpoint_sha256=q.sha(path),
            binding=binding,seed=0,train_rows=384))
        cfg=dict(design='bp_fe',dataset_path=str(dataset),dataset_sha256=q.sha(dataset),cache_sha256=q.sha(cache_path),
            critic_checkpoints={'0':str(path)},critic_checkpoint_sha256={'0':q.sha(path)})
        problem=SimpleNamespace(cond=SimpleNamespace(is_ports=~movable))
        before=torch.get_rng_state().clone()
        with patch.object(api,'RUNTIME',runtime),patch.object(api,'load_problem',return_value=problem),\
             patch.object(q,'load_dataset',return_value=({'train':train},{})),\
             patch.object(api,'load_model',side_effect=AssertionError('No policy inference')),\
             patch.object(api,'legalize',side_effect=AssertionError('No legalization')),\
             patch('torch.optim.AdamW',side_effect=AssertionError('No optimizer')):
            actual,_,meta=q.initial_model(cfg,0)
            q.state_equal(actual.state_dict(),saved['model'])
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(meta['train_rows'],384)
            self.assertFalse(actual.training)
            self.assertTrue(all(not p.requires_grad for p in actual.parameters()))
            cfg['dataset_sha256']='a'*64
            with self.assertRaisesRegex(ValueError,'context changed'):q.initial_model(cfg,0)

    def actor_fixture(self):
        exp=self.root/'actor';folder=exp/'seed_0/round_0';folder.mkdir(parents=True)
        state={'weight':torch.tensor([1.,2.]),'normalizer':torch.tensor(3.)}
        saved=dict(critic=state,step=1000,binding={'test':'binding'},optimizers={'critic':{'state':{}}})
        path=folder/'checkpoint.pt';torch.save(saved,path)
        done=dict(binding=saved['binding'],offline_critic_frozen=True,steps=1000)
        self.write(folder/'completed.json',done)
        record=dict(experiment=str(exp),seed=0,training=dict(checkpoint_sha256=q.sha(path),completion_sha256=q.sha(folder/'completed.json')))
        return record,dict(model=copy.deepcopy(state)),saved,path

    def test_actor_critic_true_full_state_freeze_not_metadata_only(self):
        record,initial,saved,path=self.actor_fixture()
        self.assertTrue(q.checked_actor_critic(record,initial)['initial_state_bitwise_equal'])
        saved['critic']['normalizer'].add_(1);torch.save(saved,path)
        record['training']['checkpoint_sha256']=q.sha(path)
        with self.assertRaisesRegex(ValueError,'tensor changed'):q.checked_actor_critic(record,initial)

    def test_actor_optimizer_update_refused_even_when_weights_equal(self):
        record,initial,saved,path=self.actor_fixture()
        saved['optimizers']['critic']['state']={0:{'step':torch.tensor(1.)}};torch.save(saved,path)
        record['training']['checkpoint_sha256']=q.sha(path)
        with self.assertRaisesRegex(ValueError,'optimizer was updated'):q.checked_actor_critic(record,initial)

    def test_common_flow_state_same_warmup_not_just_same_seed_integer(self):
        record,initial,saved,path=self.actor_fixture()
        saved['flow']={'parameter':torch.tensor([2.]),'fixed':torch.tensor([.4])}
        common={'flow':copy.deepcopy(saved['flow'])}
        torch.save(saved,path);record['training']['checkpoint_sha256']=q.sha(path)
        self.assertTrue(q.checked_actor_critic(record,initial,common)['common_flow_state_bitwise_equal'])
        common['flow']['fixed'].add_(.1)
        with self.assertRaisesRegex(ValueError,'tensor changed'):q.checked_actor_critic(record,initial,common)

    def test_cpu_scores_raw_not_executed_exact_twin_mean_and_no_policy(self):
        class Critic(torch.nn.Module):
            def forward(self,x):
                self.asserted=(x.dtype,x.device.type,torch.is_grad_enabled(),tuple(x.shape))
                mean=x.mean((1,2))
                return torch.stack((mean,mean+2),1)
        model=Critic()
        rows=[self.row(i) for i in range(8)]
        with patch('baseline.api.generate',side_effect=AssertionError('No policy')),\
             patch('baseline.api.legalize',side_effect=AssertionError('No legalizer')),\
             patch('torch.Tensor.backward',side_effect=AssertionError('No training')):
            result=q.score_pool(model,rows)
        self.assertEqual(model.asserted,(torch.float32,'cpu',False,(1,3,2)))
        self.assertAlmostEqual(result[0]['predicted_q'],1.01,places=6)
        self.assertIsNone(result[0]['recorded_cuda_q'])
        self.assertEqual(result[0]['raw_action_sha256'],q.action_sha(np.full((3,2),.01,np.float32)))

    def test_request_and_npz_tamper_refused_no_synthetic_noise(self):
        row=self.row();sample=Path(row['sample_dir'])
        q.arrays(row)
        request=q.read(sample/'request.json');request['noise_seed']=61
        self.write(sample/'request.json',request)
        with self.assertRaisesRegex(ValueError,'Changed immutable'):q.arrays(row)
        row['request_sha256']=q.sha(sample/'request.json')
        with self.assertRaisesRegex(ValueError,'scope'):q.arrays(row)

    def test_noise_all_arrays_across_three_generators_and_seeds_not_integer_seed(self):
        pools=self.simple_pools();q.noise_identity(pools)
        pools[0]['rows'][0]['actual_noise'][0,0]=.2
        with self.assertRaisesRegex(ValueError,'Actual noise arrays'):q.noise_identity(pools)
        pools=self.simple_pools();pools[0]['rows'].pop()
        with self.assertRaisesRegex(ValueError,'eight-candidate'):q.noise_identity(pools)
        pools=self.simple_pools();pools[0]['rows'][7]['noise_sha256']='0'
        with self.assertRaisesRegex(ValueError,'Duplicate'):q.noise_identity(pools)

    def test_cuda_equivalence_does_not_silently_override_common_cpu_near_tie(self):
        rows=[dict(predicted_q=1.,recorded_cuda_q=1.) for _ in range(8)]
        rows[1]['recorded_cuda_q']=1.+1e-7
        value=q.score_equivalence(rows)
        self.assertTrue(value['all_scores_close'])
        self.assertFalse(value['argmax_identical'])
        self.assertFalse(value['exact_full_order_identical'])
        self.assertEqual(value['cpu_argmax_index'],0)
        self.assertEqual(value['cpu_max_ties'],8)

    def test_selected_vs_selected_different_indices_three_seed_sd_not_24_rows(self):
        pools=self.simple_pools();q.noise_identity(pools)
        picks,groups,means,mg,pairs=q.aggregate(pools)
        chosen=next(x for x in groups if x['arm']=='flow' and x['rule']==q.Q_RULE)
        self.assertEqual(chosen['summary']['native_full']['rWL_um'],dict(n=3,mean=108.,sd=1.))
        pair=next(x for x in pairs if x['arm']=='bc' and x['reference']=='flow' and x['rule']==q.Q_RULE)
        self.assertEqual([(x['selected_index'],x['reference_index']) for x in pair['seed_values']],[(0,7)]*3)
        self.assertEqual(pair['native_full_delta']['rWL_um']['mean'],3.)
        self.assertEqual(next(x for x in mg if x['arm']=='flow')['native_full']['rWL_um']['mean'],104.5)
        self.assertEqual(len(picks),45);self.assertEqual(len(means),9)

    def test_failures_not_filtered_missing_withheld_and_no_fabricated_native(self):
        pools=self.simple_pools()
        pool=next(p for p in pools if p['arm']=='fql' and p['seed']==0)
        pool['rows'][7].update(state='physical_failure',metrics=None,common_utility=-4.)
        picks,groups,_,_,pairs=q.aggregate(pools)
        value=next(x for x in picks if x['arm']=='fql' and x['seed']==0 and x['rule']==q.Q_RULE)
        self.assertEqual(value['index'],7);self.assertIsNone(value['metrics'])
        group=next(x for x in groups if x['arm']=='fql' and x['rule']==q.Q_RULE)
        self.assertIsNone(group['summary']['native_full'])
        self.assertEqual(group['summary']['successful_only_native']['rWL_um']['n'],2)
        pair=next(x for x in pairs if x['arm']=='fql' and x['reference']=='flow' and x['rule']==q.Q_RULE)
        self.assertIsNone(pair['native_full_delta']);self.assertEqual(pair['both_success_seed_count'],2)
        pool['rows'][7]['state']='missing'
        with self.assertRaisesRegex(ValueError,'accounted'):q.aggregate(pools)

    def test_exact_report_sha_completion_and_source_binding_required(self):
        source=self.root/'source.py';source.write_text('fixture')
        report=self.write(self.root/'report/report.json',dict(source_sha256={str(source):q.sha(source)}))
        digest=q.sha(report)
        self.write(report.with_name('completed.json'),dict(status='completed',artifacts={'report.json':digest}))
        q.immutable_report(report,digest)
        source.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Changed immutable'):q.immutable_report(report,digest)
        source.write_text('fixture');report.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Changed immutable'):q.immutable_report(report,digest)


if __name__=='__main__':unittest.main()
