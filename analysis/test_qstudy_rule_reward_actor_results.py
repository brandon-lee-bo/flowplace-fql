"""Pure saved-state/native-accounting fixtures; no training/inference/EDA."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from analysis import qstudy_rule_reward_actor_results as m


class ActorRuleResultsTests(unittest.TestCase):
    def setUp(self):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run fixtures through CPU Slurm')
        self.temporary=tempfile.TemporaryDirectory(prefix='qstudy-rule-actor-report-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.root=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.cache=dict(node_id_to_name=['a','b'],ratio_x=10/224,ratio_y=20/224,
            node_info={name:dict(x=1.,y=1.) for name in 'ab'},
            net_info={'net':dict(nodes={'a':{},'b':{}})},macro_clusters=[['a','b']],
            id2index={'a':0,'b':1},dataflow_mat=np.array([[0.,1.],[1.,0.]]))
        self.context=dict(cache=self.cache,movable=torch.tensor([True,True]))
        self.specs={name:dict(version=m.screen.SPEC_VERSION,objective=name,
            cost_names=list(m.screen.COST_NAMES),weights=list(weights),scales=[1000.,10.,.1,.4,.2,.1],
            reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64),
            rule_definition_sha256=m.screen.json_sha(m.screen.rule_definition()),success_cap=2.,failure_cost=4.)
            for name,weights in m.screen.OBJECTIVES.items()}

    def rows(self,seed=0,delta=0.):
        rows=[]
        for index in range(8):
            metrics=dict(zip(m.native.METRICS,[1000.+100*seed+index+delta,-.4,-20.,10.,12.,.2,70000.]))
            rows.append(dict(seed=seed,index=index,noise_sha256='noise'+str(index),
                state='success',metrics=metrics,costs=dict(zip(m.screen.COST_NAMES,[metrics['rWL_um'],10.,.1,.4,.2,.1])),
                rewards={name:-metrics['rWL_um']/1000 for name in m.screen.OBJECTIVES},
                predicted_q=-metrics['rWL_um']/1000,legalizer_jump_rms=.01))
        return rows

    def streams(self,objective='grt_wl',deltas=(0.,0.,0.)):
        return [m.stream_summary(objective,seed,self.rows(seed,deltas[seed]),
            dict(completed=True,pairing={'seed':seed,'fixed':'same'}),[]) for seed in m.SEEDS]

    def replace(self,streams,seed,index,**changes):
        rows=copy.deepcopy(streams[seed]['rows']);rows[index].update(changes)
        streams[seed]=m.stream_summary(streams[seed]['objective'],seed,rows,streams[seed]['training'],[])

    def test_three_training_seed_SD_not_24_noise_replicates(self):
        summary=next(r for r in m.aggregate(self.streams()) if r['objective']=='grt_wl')
        self.assertEqual(summary['request_accounting']['planned'],24)
        self.assertEqual(summary['metrics']['rWL_um']['n'],3)
        self.assertEqual(summary['metrics']['rWL_um']['mean'],1103.5)
        self.assertEqual(summary['metrics']['rWL_um']['sd'],100.)

    def test_same_noise_pairs_then_three_seed_deltas(self):
        streams=self.streams()+self.streams('hpwl_edge',(-30.,-20.,-10.))
        result=m.paired(streams,'hpwl_edge','grt_wl')
        self.assertEqual((result['planned_pairs'],result['paired_exported'],result['both_success']),(24,24,24))
        self.assertEqual(result['three_seed_metric_deltas']['rWL_um'],dict(n=3,mean=-20.,sd=10.,eligible=True))
        self.assertEqual(result['three_seed_metric_deltas']['WNS_ns']['mean'],0.)

    def test_missing_request_preserves_plan_and_withholds_seed_aggregate(self):
        target=self.streams('hpwl_edge')
        self.replace(target,1,7,state='missing',metrics=None,costs=None,rewards={o:None for o in m.screen.OBJECTIVES})
        summary=next(r for r in m.aggregate(target) if r['objective']=='hpwl_edge')
        self.assertFalse(summary['aggregate_eligible'])
        self.assertEqual(summary['request_accounting']['missing'],1)
        self.assertEqual(summary['request_accounting']['planned'],24)
        pair=m.paired(self.streams()+target,'hpwl_edge','grt_wl')
        self.assertFalse(pair['fully_accounted'])
        self.assertIsNone(pair['three_seed_metric_deltas'])

    def test_physical_failure_no_fakePPA_but_common_utility_penalty(self):
        target=self.streams('hpwl_edge')
        self.replace(target,0,0,state='physical_failure',metrics=None,costs=None,
            rewards={o:-4. for o in m.screen.OBJECTIVES})
        summary=next(r for r in m.aggregate(target) if r['objective']=='hpwl_edge')
        self.assertTrue(summary['aggregate_eligible'])
        self.assertTrue(summary['conditional_PPA'])
        self.assertEqual(summary['request_accounting']['physical_failure'],1)
        pair=m.paired(self.streams()+target,'hpwl_edge','grt_wl')
        self.assertIsNone(pair['pairs'][0]['deltas'])
        self.assertIsNotNone(pair['pairs'][0]['common_grt_B_utility_delta'])

    def test_zero_success_seed_not_silently_removed(self):
        target=self.streams()
        for index in range(8):
            self.replace(target,2,index,state='geometry_failure',metrics=None,costs=None,
                rewards={o:-4. for o in m.screen.OBJECTIVES})
        summary=next(r for r in m.aggregate(target) if r['objective']=='grt_wl')
        self.assertTrue(summary['aggregate_eligible'])
        self.assertEqual(summary['metrics']['rWL_um']['n'],2)
        self.assertIsNone(summary['metrics']['rWL_um']['mean'])

    def test_actual_noise_and_training_endpoint_mismatches_refused(self):
        streams=self.streams()+self.streams('hpwl_edge')
        m.verify_pairing(streams)
        streams[3]['actual_noise_by_index']['0']='different'
        with self.assertRaisesRegex(ValueError,'actual evaluation noises'):
            m.verify_pairing(streams)
        streams=self.streams()+self.streams('hpwl_edge')
        streams[3]['training']['pairing']['fixed']='changed'
        with self.assertRaisesRegex(ValueError,'RNG endpoints'):
            m.verify_pairing(streams)
        different=[dict(streams[3],actual_noise_by_index={'0':'wrong'}),streams[4],streams[5]]
        with self.assertRaisesRegex(ValueError,'actual noise'):
            m.paired(self.streams()+different,'hpwl_edge','grt_wl')

    def test_measured_reward_uses_executed_geometry_and_train_scales(self):
        row=self.rows()[0]
        request=dict(executed_action=[[-.5,-.4],[.2,.3]],raw_action=[[.8,.8],[-.8,-.8]])
        actual=m.measured_rewards(row,request,self.context,self.specs)
        costs=np.asarray(list(actual['costs'].values()))
        for objective,spec in self.specs.items():
            c=float(np.dot(costs/np.asarray(spec['scales']),spec['weights']))
            self.assertAlmostEqual(actual['rewards'][objective],-2*c/(1+c))
        altered=copy.deepcopy(request);altered['raw_action']=[[0.,0.],[0.,0.]]
        self.assertEqual(actual,m.measured_rewards(row,altered,self.context,self.specs))
        self.assertEqual(actual['costs']['grt_wl'],1000.)

    def test_reward_missing_and_failure_not_zero_metrics(self):
        for state,expected in [('physical_failure',-4.),('geometry_failure',-4.),('missing',None),('invalid',None)]:
            result=m.measured_rewards(dict(state=state),{},self.context,self.specs)
            self.assertIsNone(result['costs'])
            self.assertTrue(all(value==expected for value in result['rewards'].values()))

    def saved_training(self):
        experiment=self.root/'actor';folder=experiment/'seed_0/round_0';folder.mkdir(parents=True)
        config=dict(m.actor.FIXED)
        config.update(dataset_sha256='d'*64,cache_sha256='c'*64,common_checkpoint_sha256={'0':'a'*64},
            rule_behavior={'fixed_targets':256},rule_common={'0':{'checkpoint':{'path':'common','sha256':'a'*64}}})
        (experiment/'config.json').write_text(json.dumps(config))
        binding=dict(replay=[],parent=None)
        (folder/'binding.json').write_text(json.dumps(binding))
        (folder/'completed.json').write_text(json.dumps(dict(offline_critic_frozen=True,online_flow_update=False)))
        common=dict(flow={'weight':torch.ones(2)},actor={'weight':torch.ones(2),
            'movable':torch.tensor([True,True]),'fixed':torch.zeros((2,2))})
        critic={'weight':torch.ones(3),'_extra_state':{'label':'bounded'}}
        refit=dict(model=SimpleNamespace(state_dict=lambda:critic),checkpoint={'path':'critic','sha256':'b'*64})
        rng=torch.Generator().manual_seed(100).get_state()
        saved=dict(critic=copy.deepcopy(critic),flow=copy.deepcopy(common['flow']),actor=copy.deepcopy(common['actor']),
            step=1000,binding=binding,rng=rng,cuda_rng=[rng],generators={k:rng for k in ('actor','q','flow')},
            optimizers={'actor':{'state':{0:{'step':torch.tensor(1000.)}}},'critic':{'state':{}},'flow':{'state':{}}})
        torch.save(saved,folder/'checkpoint.pt')
        checked=dict(completed=True,checkpoint=str(folder/'checkpoint.pt'),checkpoint_sha256=m.sha(folder/'checkpoint.pt'))
        return folder,config,refit,common,saved,checked

    def check_saved(self,values):
        folder,cfg,refit,common,saved,checked=values
        with mock.patch.object(m.native,'curve',return_value={'available':False}), \
             mock.patch.object(m.native,'check_training',return_value=checked):
            return m.checked_training(folder,cfg,0,refit,common)

    def test_saved_critic_flow_and_adam_contract_without_forward(self):
        values=self.saved_training()
        result=self.check_saved(values)
        self.assertEqual(result['adam_committed_steps'],1000)
        self.assertTrue(result['critic_frozen'] and result['flow_frozen'])
        self.assertIn('negative logged cosine',result['gradient_cosine_definition'])

    def test_saved_critic_or_flow_mutation_refused(self):
        values=self.saved_training();folder,cfg,refit,common,saved,checked=values
        saved['critic']['weight']+=1;torch.save(saved,folder/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError,'critic'):
            self.check_saved(values)
        saved['critic']['weight']-=1;saved['flow']['weight']+=1;torch.save(saved,folder/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError,'flow'):
            self.check_saved(values)

    def test_saved_budget_replay_and_optimizer_guards(self):
        values=self.saved_training();folder,cfg,refit,common,saved,checked=values
        saved['optimizers']['actor']['state'][0]['step']=torch.tensor(999.)
        torch.save(saved,folder/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError,'Adam'):
            self.check_saved(values)
        saved['optimizers']['actor']['state'][0]['step']=torch.tensor(1000.)
        saved['optimizers']['flow']['state'][0]={'step':1};torch.save(saved,folder/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError,'frozen critic/flow'):
            self.check_saved(values)

    def test_request_eval_only_header_uses_native_checker(self):
        cfg=dict(arm='rule_grt_wl',run_prefix='rrv10',weights=[1.,0.,0.,0.,0.,0.])
        training=dict(checkpoint='/saved/checkpoint.pt',checkpoint_sha256='a'*64)
        row=dict(arm=cfg['arm'],role='evaluation',round=0,terminal_mask=0,training_feedback_eligible=False,
            noise_seed=4600000,checkpoint_sha256='a'*64,source_checkpoint=training['checkpoint'],
            objective_weights=cfg['weights'],index=3,run_id='go2o_qs1_rrv10_s0_r0_evaluation_3',
            evaluation_purpose='development_validation')
        with mock.patch.object(m.backend,'checked_request',return_value=self.root) as check:
            self.assertEqual(m.verify_request(row,cfg,0,training,({},None,{})),self.root)
            self.assertEqual(check.call_count,1)
            for key,value in [('training_feedback_eligible',True),('noise_seed',1),('checkpoint_sha256','wrong'),
                              ('objective_weights',[0.,1.,0.,0.,0.,0.]),('role','collect')]:
                altered=dict(row,**{key:value})
                with self.assertRaisesRegex(ValueError,'eval-only'):
                    m.verify_request(altered,cfg,0,training,({},None,{}))

    def test_native_guard_missing_geometry_never_calls_evaluator(self):
        sample=self.root/'sample';sample.mkdir()
        (sample/'request.json').write_text('{}');(sample/'actions.npz').write_bytes(b'identity')
        run=self.root/'run';run.mkdir();(run/'manifest.json').write_text(json.dumps({'success':True}))
        row=dict(seed=0,round=0,arm='a',role='evaluation',index=0,graph='bp_fe',training_feedback_eligible=False)
        (sample/'outcome.json').write_text(json.dumps(dict(row,physical_run=str(run),status='success')))
        with mock.patch.object(m.backend,'checked_existing',side_effect=AssertionError('No automatic EDA')) as checker:
            value=m.baseline.inspect_outcome(row,sample,self.root/'tasks',({},None,{}))
        self.assertEqual(value['state'],'invalid');self.assertEqual(checker.call_count,0)
        self.assertIn('saved native geometry',value['integrity_error'])

    def test_published_plan_dataset_and_initial_refit_chain(self):
        refit=dict(spec=self.specs['grt_wl'],checkpoint={'path':str(self.root/'critic.pt'),'sha256':'q'*64})
        preparation=dict(plan={'path':'plan','sha256':'p'*64},behavior={'targets':256},
            common={'exact_common':'identity'},source_sha256={'source':'s'*64})
        inputs=dict(dataset_path='/data/dataset.npz',dataset_sha256='d'*64,cache_sha256='c'*64)
        cfg=dict(rule_plan=preparation['plan'],rule_behavior=preparation['behavior'],
            rule_common=preparation['common'],rule_sources=preparation['source_sha256'],
            rule_preparation_root=str(self.root),rule_utility_spec=refit['spec'],
            critic_checkpoint_sha256={'0':'q'*64},critic_checkpoints={'0':refit['checkpoint']['path']},
            common_checkpoint_sha256={'0':'a'*64,'1':'b'*64,'2':'c'*64},**inputs)
        m.checked_published_config(cfg,preparation,inputs,self.root,refit,0)
        for key,value in [('cache_sha256','wrong'),('rule_plan',{'path':'wrong'}),
                          ('rule_behavior',{'targets':384}),('critic_checkpoint_sha256',{'0':'wrong'})]:
            with self.assertRaisesRegex(ValueError,'Published actor'):
                m.checked_published_config(dict(cfg,**{key:value}),preparation,inputs,self.root,refit,0)


if __name__=='__main__':unittest.main()
