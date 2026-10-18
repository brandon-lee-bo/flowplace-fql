"""Saved-state, target-calibration and accounting fixtures; no model/EDA calls."""
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

from analysis import qstudy_final_feedback_actor_results as m


class FinalActorResultsTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run fixtures through CPU Slurm')
        self.temporary=tempfile.TemporaryDirectory(prefix='qstudy-final-actor-results-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.root=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.cache=dict(node_id_to_name=['a','b'],ratio_x=10/224,ratio_y=20/224,
            node_info={name:dict(x=1.,y=1.) for name in 'ab'},
            net_info={'net':dict(nodes={'a':{},'b':{}})},macro_clusters=[['a','b']],
            id2index={'a':0,'b':1},dataflow_mat=np.array([[0.,1.],[1.,0.]]))
        self.context=dict(cache=self.cache,movable=torch.tensor([True,True]))
        self.specs={name:dict(version=m.screen.SPEC_VERSION,objective=name,
            cost_names=list(m.screen.COST_NAMES),weights=list(weights),
            scales=[1000.,1100.,20.,.2,70000.,.1,.4,.2,.1],
            reference=dict(split='train',dataset_sha256='a'*64,train_rows_sha256='b'*64),
            rule_definition_sha256=m.screen.json_sha(m.screen.rule_definition()),success_cap=2.,failure_cost=4.,
            feasibility='completed_final_DRC0; final_DRCpositive=physical_failure_for_all_objectives')
            for name,weights in m.screen.OBJECTIVES.items()}

    def training(self,seed):
        return dict(completed=True,checkpoint_sha256='c'*64,
            pairing={'seed':seed,'fixed':'same'},gradient_cosine_definition='negative logged cosine aligned')

    def grt_rows(self,seed=0,delta=0.):
        rows=[]
        for index in range(8):
            metrics=dict(zip(m.native.METRICS,[1000.+100*seed+index+delta,-.4,-20.,10.,12.,.2,70000.]))
            rows.append(dict(seed=seed,index=index,noise_sha256='noise'+str(index),
                action_sha256='a'+str(index),raw_action_sha256='r'+str(index),
                request={'path':'request','sha256':'q'+str(index)},state='success',metrics=metrics,
                integrity_error=None,predicted_q=-1.,legalizer_jump_rms=.01,
                executed_action=[[-.5,-.4],[.2,.3]]))
        return rows

    def grt_streams(self,objective='grt_wl',deltas=(0.,0.,0.)):
        return [m.grt_summary(objective,seed,self.grt_rows(seed,deltas[seed]),self.training(seed),[])
                for seed in m.SEEDS]

    def final_rows(self,objective='grt_wl',seed=0,delta=0.):
        rows=[]
        for index in range(4):
            metrics=dict(zip(m.final.METRICS,[1100.+100*seed+index+delta,-.5,-25.,.21,71000.,0]))
            row=dict(objective=objective,seed=seed,index=index,noise_sha256='noise'+str(index),
                state='completed',metrics=metrics,predicted_q=-1.,legalizer_jump_rms=.01)
            row.update(m.measured_rewards(self.grt_rows(seed)[index],row,[[-.5,-.4],[.2,.3]],
                self.context,self.specs));rows.append(row)
        return rows

    def final_streams(self,objective='grt_wl',deltas=(0.,0.,0.)):
        return [m.final_summary(objective,seed,self.final_rows(objective,seed,deltas[seed]),self.training(seed),[])
                for seed in m.SEEDS]

    def test_GRTeight_DRTfour_not_eight_final_denominator(self):
        g=m.aggregate(self.grt_streams(),'GRT')[0]
        d=m.aggregate(self.final_streams(),'DRT')[0]
        self.assertEqual((g['planned'],d['planned']),(24,12))
        self.assertEqual((g['native_metrics']['rWL_um']['n'],d['native_metrics']['DRT_WL']['n']),(3,3))
        self.assertEqual(g['native_metrics']['rWL_um']['mean'],1103.5)
        self.assertEqual(d['native_metrics']['DRT_WL']['mean'],1201.5)
        self.assertEqual(g['native_metrics']['rWL_um']['sd'],100.)
        self.assertEqual(d['native_metrics']['DRT_WL']['sd'],100.)

    def test_all_six_actual_B_targets_recomputed_not_generic_backend_reward(self):
        grt=self.grt_rows()[0];grt['reward']=-123456.
        drt=self.final_rows()[0];drt['reward']=9876.
        actual=m.measured_rewards(grt,drt,[[-.5,-.4],[.2,.3]],self.context,self.specs)
        costs=np.asarray(list(actual['costs'].values()))
        self.assertEqual((actual['costs']['grt_wl'],actual['costs']['drt_wl'],
                          actual['costs']['drt_tns_violation']),(1000.,1100.,25.))
        for objective,spec in self.specs.items():
            c=float(np.dot(costs/np.asarray(spec['scales']),spec['weights']))
            self.assertAlmostEqual(actual['rewards'][objective],-2*c/(1+c))
        grt['reward']=0.;drt['reward']=0.
        self.assertEqual(actual,m.measured_rewards(grt,drt,[[-.5,-.4],[.2,.3]],self.context,self.specs))

    def test_DRCpositive_B4_all_objectives_without_discarding_measured_PPA(self):
        rows=self.final_rows();rows[0]['metrics']['DRC']=7
        rows[0].update(m.measured_rewards(self.grt_rows()[0],rows[0],[[-.5,-.4],[.2,.3]],self.context,self.specs))
        summary=m.final_summary('grt_wl',0,rows,self.training(0),[])
        self.assertTrue(summary['fully_accounted'])
        self.assertEqual((summary['completed'],summary['DRC_positive'],summary['DRC_zero']),(4,1,3))
        self.assertFalse(summary['conditional_PPA'])
        self.assertEqual(summary['completed_native_metric_means']['DRC'],1.75)
        self.assertTrue(all(v==-4. for v in rows[0]['rewards'].values()))
        self.assertIsNotNone(rows[0]['costs']);self.assertIsNotNone(rows[0]['linear_costs']['drt_wl'])

    def test_upstream_failure_requires_proof_and_gets_no_fake_native_metrics(self):
        drt=dict(state='upstream_physical_failure',metrics=None)
        actual=m.measured_rewards(dict(state='physical_failure'),drt,None,self.context,self.specs)
        self.assertIsNone(actual['costs'])
        self.assertTrue(all(v==-4. for v in actual['rewards'].values()))
        with self.assertRaisesRegex(ValueError,'Unproven'):
            m.measured_rewards(dict(state='success'),drt,None,self.context,self.specs)
        with self.assertRaisesRegex(ValueError,'Unproven'):
            m.measured_rewards(dict(state='physical_failure'),dict(drt,metrics={'DRC':0}),None,self.context,self.specs)

    def test_infrastructure_missing_invalid_is_NA_never_B4(self):
        for state in ('invalid','missing','missing_final','source_missing','source_invalid'):
            actual=m.measured_rewards({},dict(state=state),None,self.context,self.specs)
            self.assertIsNone(actual['feasible']);self.assertIsNone(actual['costs'])
            self.assertTrue(all(v is None for v in actual['rewards'].values()))

    def test_executed_rules_and_fixed_TRAINscale_no_test_refit(self):
        grt=self.grt_rows()[0];drt=self.final_rows()[0]
        with mock.patch.object(m,'rules',return_value={k:.5+i for i,k in enumerate(m.screen.EXPERT_NAMES)}) as rules:
            result=m.measured_rewards(grt,drt,[[.1,.2],[.3,.4]],self.context,self.specs)
        np.testing.assert_equal(rules.call_args[0][0],[[.1,.2],[.3,.4]])
        self.assertEqual(result['costs']['edge_clearance'],.5)
        changed=copy.deepcopy(self.specs);changed['drt_wl']['scales'][1]*=2
        after=m.measured_rewards(grt,drt,[[.1,.2],[.3,.4]],self.context,changed)
        before=m.measured_rewards(grt,drt,[[.1,.2],[.3,.4]],self.context,self.specs)
        self.assertGreater(after['rewards']['drt_wl'],before['rewards']['drt_wl'])

    def test_invalid_final_native_values_or_spec_refused(self):
        for metric,value in [('DRC',.5),('Power',-1.),('StdCellArea',float('nan')),('DRT_WL',-1.)]:
            drt=self.final_rows()[0];drt['metrics'][metric]=value
            with self.assertRaisesRegex(ValueError,'malformed'):
                m.measured_rewards(self.grt_rows()[0],drt,[[-.5,-.4],[.2,.3]],self.context,self.specs)
        spec=copy.deepcopy(self.specs);spec['drt_wl']['reference']['split']='validation'
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            m.measured_rewards(self.grt_rows()[0],self.final_rows()[0],[[-.5,-.4],[.2,.3]],self.context,spec)

    def test_incomplete_final_withholds_seedmean_no_validation_renormalization(self):
        streams=self.final_streams();rows=copy.deepcopy(streams[1]['rows'])
        rows[3].update(state='missing_final',metrics=None,feasible=None,
            rewards={o:None for o in m.screen.OBJECTIVES},linear_costs={o:None for o in m.screen.OBJECTIVES})
        streams[1]=m.final_summary('grt_wl',1,rows,self.training(1),[])
        result=m.aggregate(streams,'DRT')[0]
        self.assertFalse(result['aggregate_eligible']);self.assertIsNone(result['native_metrics'])
        self.assertEqual(result['counts']['missing_final'],1)
        self.assertIsNone(result['all_preference_rewards'])
        self.assertEqual(streams[1]['own_q_calibration']['planned'],4)
        self.assertFalse(streams[1]['own_q_calibration']['complete'])

    def test_zero_success_seed_not_silently_omitted(self):
        streams=self.final_streams();rows=copy.deepcopy(streams[2]['rows'])
        for row in rows:
            row.update(state='upstream_physical_failure',metrics=None,feasible=False,costs=None,
                rewards={o:-4. for o in m.screen.OBJECTIVES},linear_costs={o:None for o in m.screen.OBJECTIVES})
        streams[2]=m.final_summary('grt_wl',2,rows,self.training(2),[])
        result=m.aggregate(streams,'DRT')[0]
        self.assertTrue(result['aggregate_eligible']);self.assertTrue(result['conditional_PPA'])
        self.assertEqual(result['native_metrics']['DRT_WL']['n'],2)
        self.assertIsNone(result['native_metrics']['DRT_WL']['mean'])
        self.assertEqual(result['all_preference_rewards']['grt_wl']['n'],3)

    def test_ownQ_calibrated_on_executed_true_final_B_only_four(self):
        rows=self.final_rows('drt_wl')
        for row in rows:row['predicted_q']=row['rewards']['drt_wl']+.25
        summary=m.final_summary('drt_wl',0,rows,self.training(0),[])
        self.assertEqual((summary['own_q_calibration']['n'],summary['own_q_calibration']['planned']),(4,4))
        self.assertAlmostEqual(summary['own_q_calibration']['mae'],.25)
        self.assertAlmostEqual(summary['own_q_calibration']['signed_prediction_minus_measurement'],.25)
        g=self.grt_streams('drt_wl')[0]
        self.assertEqual(g['final_reward_not_planned_indices'],[4,5,6,7])
        self.assertNotIn('preference_reward_means',g)

    def test_same_actual_noise_pairs_and_three_seed_delta_SD(self):
        streams=self.final_streams()+self.final_streams('drt_wl',(-30.,-20.,-10.))
        actual=m.paired(streams,'drt_wl','grt_wl','DRT')
        self.assertEqual((actual['planned_pairs'],actual['paired']),(12,12))
        self.assertEqual(actual['three_seed_native_deltas']['DRT_WL'],dict(n=3,mean=-20.,sd=10.,eligible=True))
        self.assertIsNotNone(actual['three_seed_preference_reward_deltas']['drt_wl']['mean'])
        streams[3]['rows'][0]['noise_sha256']='different'
        with self.assertRaisesRegex(ValueError,'actual saved noise'):
            m.paired(streams,'drt_wl','grt_wl','DRT')

    def test_completed_pair_requires_actual_noise_and_failures_keep_utility(self):
        streams=self.final_streams()+self.final_streams('drt_wl')
        streams[3]['rows'][0]['noise_sha256']=None
        with self.assertRaisesRegex(ValueError,'both actual saved noise'):
            m.paired(streams,'drt_wl','grt_wl','DRT')
        streams=self.final_streams()+self.final_streams('drt_wl')
        rows=copy.deepcopy(streams[3]['rows']);rows[0].update(state='upstream_physical_failure',metrics=None,
            feasible=False,rewards={o:-4. for o in m.screen.OBJECTIVES},linear_costs={o:None for o in m.screen.OBJECTIVES})
        streams[3]=m.final_summary('drt_wl',0,rows,self.training(0),[])
        result=m.paired(streams,'drt_wl','grt_wl','DRT')
        self.assertTrue(result['fully_accounted']);self.assertTrue(result['conditional_PPA'])
        self.assertIsNone(result['pairs'][0]['deltas'])
        self.assertIsNotNone(result['pairs'][0]['preference_reward_deltas']['drt_wl'])

    def test_pairing_refuses_source_setting_or_actual_array_drift(self):
        streams=[s for o in m.screen.OBJECTIVES for s in self.grt_streams(o)]
        self.assertTrue(all(r['all_six_available'] for r in m.verify_pairing(streams)))
        streams[-1]['training']['pairing']['fixed']='changed'
        with self.assertRaisesRegex(ValueError,'RNG endpoints'):
            m.verify_pairing(streams)
        streams=[s for o in m.screen.OBJECTIVES for s in self.grt_streams(o)]
        streams[-1]['actual_noise_by_index']['7']='different'
        with self.assertRaisesRegex(ValueError,'Actual eight noise'):
            m.verify_pairing(streams)

    def saved_training(self):
        experiment=self.root/'actor';folder=experiment/'seed_0/round_0';folder.mkdir(parents=True)
        config=dict(m.actor.FIXED)
        config.update(dataset_sha256='d'*64,cache_sha256='c'*64,common_checkpoint_sha256={'0':'a'*64},
            final_feedback_behavior={'fixed_targets':256},final_feedback_critic_inputs={'rows':64,'inputs':{'collector':'bound'}},
            final_feedback_common={'0':{'checkpoint':{'path':'common','sha256':'a'*64}}})
        (experiment/'config.json').write_text(json.dumps(config))
        binding=dict(replay=[],parent=None)
        (folder/'binding.json').write_text(json.dumps(binding))
        (folder/'completed.json').write_text(json.dumps(dict(offline_critic_frozen=True,online_flow_update=False)))
        common=dict(flow={'weight':torch.ones(2)},actor={'weight':torch.ones(2),
            'movable':torch.tensor([True,True]),'fixed':torch.zeros((2,2))})
        critic={'weight':torch.ones(3),'_extra_state':{'label':'final64'}}
        refit=dict(model=SimpleNamespace(state_dict=lambda:critic),checkpoint={'path':'critic','sha256':'b'*64})
        rng=torch.Generator().manual_seed(100).get_state()
        saved=dict(critic=copy.deepcopy(critic),flow=copy.deepcopy(common['flow']),actor=copy.deepcopy(common['actor']),
            step=1000,binding=binding,rng=rng,cuda_rng=[rng],generators={k:rng for k in ('actor','q','flow')},
            optimizers={'actor':{'state':{0:{'step':torch.tensor(1000.)}}},'critic':{'state':{}},'flow':{'state':{}}})
        torch.save(saved,folder/'checkpoint.pt')
        checked=dict(completed=True,checkpoint=str(folder/'checkpoint.pt'),checkpoint_sha256=m.sha(folder/'checkpoint.pt'))
        return folder,config,refit,common,saved,checked

    def check_saved(self,values):
        folder,cfg,refit,common,_,checked=values
        with mock.patch.object(m.native,'curve',return_value={'available':False}), \
             mock.patch.object(m.native,'check_training',return_value=checked):
            return m.checked_training(folder,cfg,0,refit,common)

    def test_actual_saved_frozen_states1000_and_separate64_256_384(self):
        result=self.check_saved(self.saved_training())
        self.assertEqual((result['critic_training_rows'],result['behavior_targets'],result['physical_context_rows']),(64,256,384))
        self.assertEqual(result['adam_committed_steps'],1000)
        self.assertEqual(result['pairing']['critic_source_inputs']['rows'],64)
        self.assertIn('negative logged cosine',result['gradient_cosine_definition'])

    def test_saved_critic_flow_buffer_or_update_mutation_refused(self):
        values=self.saved_training();folder,cfg,refit,common,saved,checked=values
        original=copy.deepcopy(saved)
        for mutation in ('critic','flow','actor_buffer','step','critic_optimizer'):
            saved=copy.deepcopy(original)
            if mutation in ('critic','flow'):saved[mutation]['weight']+=1
            elif mutation=='actor_buffer':saved['actor']['fixed']+=1
            elif mutation=='step':saved['optimizers']['actor']['state'][0]['step']=torch.tensor(999.)
            else:saved['optimizers']['critic']['state'][0]={'step':1}
            torch.save(saved,folder/'checkpoint.pt')
            with self.assertRaises(ValueError):self.check_saved((folder,cfg,refit,common,saved,checked))

    def publication(self):
        inputs=dict(dataset_path='/original384/dataset.npz',dataset_sha256='d'*64,cache={'path':'cache','sha256':'c'*64})
        preparation=dict(plan={'path':'plan','sha256':'p'*64},behavior={'targets':256},common={'same':'common'},
            source_sha256={'source':'s'*64},critic_inputs={'rows':64,'inputs':inputs},evaluation=m.actor.FINAL_EVALUATION)
        refit=dict(spec=self.specs['grt_wl'],checkpoint={'path':str(self.root/'critic.pt'),'sha256':'q'*64})
        cfg=dict(final_feedback_plan=preparation['plan'],final_feedback_behavior=preparation['behavior'],
            final_feedback_common=preparation['common'],final_feedback_sources=preparation['source_sha256'],
            final_feedback_critic_inputs=preparation['critic_inputs'],final_feedback_evaluation=m.actor.FINAL_EVALUATION,
            final_feedback_preparation_root=str(self.root),dataset_path=inputs['dataset_path'],
            dataset_sha256=inputs['dataset_sha256'],cache_sha256=inputs['cache']['sha256'],
            final_feedback_utility_spec=refit['spec'],critic_checkpoint_sha256={'0':'q'*64},
            critic_checkpoints={'0':refit['checkpoint']['path']},common_checkpoint_sha256={'0':'a'*64,'1':'b'*64,'2':'c'*64})
        return cfg,preparation,inputs,refit

    def test_publication_closes_64_vs384_sources_and_full_refit(self):
        cfg,preparation,inputs,refit=self.publication()
        m.checked_published_config(cfg,preparation,inputs,self.root,refit,0)
        for field,value in [('dataset_path','/final64/cannot-evaluate'),('cache_sha256','wrong'),
                            ('critic_checkpoint_sha256',{'0':'bad'}),('final_feedback_plan',{'path':'other'})]:
            with self.assertRaisesRegex(ValueError,'binding differs'):
                m.checked_published_config(dict(cfg,**{field:value}),preparation,inputs,self.root,refit,0)
        bad=copy.deepcopy(preparation);bad['evaluation']['drt_requests']=71
        with self.assertRaisesRegex(ValueError,'binding differs'):
            m.checked_published_config(cfg,bad,inputs,self.root,refit,0)

    def source_fixture(self):
        grt=[s for o in m.screen.OBJECTIVES for s in self.grt_streams(o)]
        tasks=[];sources=[]
        for stream in grt:
            for index in m.DRT_INDICES:
                r=stream['rows'][index]
                task=dict(arm=stream['arm'],seed=stream['seed'],noise_index=index,round=0)
                tasks.append(task)
                sources.append(dict(state='resolved',binding=dict(raw_action_sha256=r['raw_action_sha256'],
                    action_sha256=r['action_sha256'],request_sha256=r['request']['sha256'],
                    actor={'checkpoint_sha256':stream['training']['checkpoint_sha256']})))
        plan={'tasks':tasks}
        def inspected(_root,_plan,task,source):
            return dict(state='completed',metrics=dict(zip(m.final.METRICS,[1100.,-.5,-25.,.21,71000.,0])),
                noise_sha256='noise'+str(task['noise_index']),physical_identity={'threads':8,'cap':20},
                physical_run='/native/'+str(tasks.index(task)))
        return grt,plan,sources,inspected

    def test72_final_sources_exact_binding_and_geometry_mapping(self):
        grt,plan,sources,inspected=self.source_fixture()
        api=SimpleNamespace(inspect_sources=mock.Mock(return_value=sources))
        with mock.patch.object(m,'route_api',return_value=api),mock.patch.object(m.final,'inspect_final',side_effect=inspected):
            streams,returned=m.collect_final(self.root,plan,grt,self.context,self.specs)
        self.assertEqual((len(streams),sum(len(s['rows']) for s in streams)),(18,72))
        self.assertTrue(all(s['fully_accounted'] for s in streams))
        self.assertEqual(returned,sources)
        self.assertFalse(api.inspect_sources.call_args.kwargs['seal'])

    def test_final_tampered_request_checkpoint_or_shared_physicalrun_invalidates(self):
        for mode in ('request','actor','run'):
            grt,plan,sources,inspected=self.source_fixture()
            if mode=='request':sources[0]['binding']['request_sha256']='tampered'
            elif mode=='actor':sources[0]['binding']['actor']['checkpoint_sha256']='tampered'
            def result(root,plan,task,source):
                row=inspected(root,plan,task,source)
                if mode=='run':row['physical_run']='/duplicate'
                return row
            with mock.patch.object(m,'route_api',return_value=SimpleNamespace(inspect_sources=lambda *a,**k:sources)), \
                 mock.patch.object(m.final,'inspect_final',side_effect=result):
                streams,_=m.collect_final(self.root,plan,grt,self.context,self.specs)
            self.assertEqual(streams[0]['rows'][0]['state'],'invalid')
            self.assertFalse(streams[0]['fully_accounted'])
            self.assertIsNone(streams[0]['rows'][0]['metrics'])

    def test_missing_native_geometry_guard_does_not_start_evaluator(self):
        sample=self.root/'sample';sample.mkdir()
        (sample/'request.json').write_text('{}');(sample/'actions.npz').write_bytes(b'identity')
        run=self.root/'run';run.mkdir();(run/'manifest.json').write_text(json.dumps({'success':True}))
        row=dict(seed=0,round=0,arm='a',role='evaluation',index=0,graph='bp_fe',training_feedback_eligible=False)
        (sample/'outcome.json').write_text(json.dumps(dict(row,physical_run=str(run),status='success')))
        with mock.patch.object(m.backend,'checked_existing',side_effect=AssertionError('No automatic EDA')) as checker:
            value=m.baseline.inspect_outcome(row,sample,self.root/'tasks',({},None,{}))
        self.assertEqual(value['state'],'invalid');self.assertEqual(checker.call_count,0)

    def collection_fixture(self):
        actors=self.root/'actors';actors.mkdir()
        route=self.root/'route';route.mkdir()
        cfg0,preparation,inputs,_=self.publication()
        common={}
        for seed in m.SEEDS:
            path=self.root/f'common_{seed}.pt'
            torch.save({'flow':{'seed':seed}},path)
            common[str(seed)]={'checkpoint':m.bind(path)}
        preparation.update(schema=m.actor.SCHEMA,status='prepared_configs_only',
            objectives=list(m.screen.OBJECTIVES),automatic_selection=False,online_implemented=False,
            backend_calls=0,actor_jobs_submitted=0,common=common,configs={})
        configs={};refits={}
        for objective in m.screen.OBJECTIVES:
            folder=actors/m.actor.ARMS[objective][0];folder.mkdir()
            cfg=copy.deepcopy(cfg0)
            cfg.update(arm=m.actor.ARMS[objective][0],final_feedback_objective=objective,
                final_feedback_common=common,final_feedback_preparation_root=str(actors),
                common_checkpoint_sha256={s:v['checkpoint']['sha256'] for s,v in common.items()},
                final_feedback_utility_spec=self.specs[objective],critic_checkpoints={},
                critic_checkpoint_sha256={},final_feedback_refits={})
            for seed in m.SEEDS:
                result=dict(spec=self.specs[objective],checkpoint={'path':str(self.root/f'critic_{objective}_{seed}.pt'),
                    'sha256':str(seed)*64},completion={'path':'completion','sha256':'e'*64},
                    normalization={'train_only':True},evidence={})
                cfg['critic_checkpoints'][str(seed)]=result['checkpoint']['path']
                cfg['critic_checkpoint_sha256'][str(seed)]=result['checkpoint']['sha256']
                cfg['final_feedback_refits'][str(seed)]=result
                refits[(objective,seed)]=result
            (folder/'config.json').write_text(json.dumps(cfg))
            preparation['configs'][cfg['arm']]=m.bind(folder/'config.json')
            configs[cfg['arm']]=cfg
        (actors/'preparation.json').write_text(json.dumps(preparation))
        tasks=[dict(arm=m.actor.ARMS[o][0],seed=s,round=0,noise_index=i)
               for o in m.screen.OBJECTIVES for s in m.SEEDS for i in m.DRT_INDICES]
        plan=dict(requested=72,tasks=tasks,evaluation_indices=list(m.DRT_INDICES),rounds=[0],
            actor_seeds=list(m.SEEDS),detailed_route_end_iteration=20,
            actor_experiments={arm:dict(path=str(actors/arm),config_sha256=m.sha(actors/arm/'config.json'),config=cfg)
                for arm,cfg in configs.items()})
        (route/'full_route_plan.json').write_text(json.dumps(plan))
        loaded=(dict(inputs=inputs),dict(train=self.context),self.context)
        api=SimpleNamespace(validate_plan=lambda *a:plan)
        def refit(_path,objective,seed,_expected,loaded=None):return refits[(objective,seed)]
        def grt(_exp,cfg,seed,*args):return self.grt_streams(cfg['final_feedback_objective'])[seed]
        def drt(*args):return [s for o in m.screen.OBJECTIVES for s in self.final_streams(o)],[]
        return actors,route,preparation,configs,loaded,api,refit,grt,drt

    def run_collection(self,values):
        actors,route,preparation,configs,loaded,api,refit,grt,drt=values
        with mock.patch.object(m.actor,'sources',return_value=preparation['source_sha256']), \
             mock.patch.object(m.actor,'validate_config',side_effect=lambda p,s:configs[p.name]), \
             mock.patch.object(m.screen,'checked_plan',return_value=loaded), \
             mock.patch.object(m.screen,'checked_refit',side_effect=refit) as check, \
             mock.patch.object(m,'route_api',return_value=api), \
             mock.patch.object(m,'collect_grt',side_effect=grt), \
             mock.patch.object(m,'collect_final',side_effect=drt), \
             mock.patch.object(m.backend,'context',return_value=({},None,{'threads':8,'config_sha256':'onlycfgdiff'})):
            result=m.collect(actors,route,m.sha(route/'full_route_plan.json'))
        return result,check

    def test_declared18actors144GRT72DRT20contrasts_and_all_refit_checks(self):
        result,check=self.run_collection(self.collection_fixture())
        self.assertEqual(check.call_count,18)
        self.assertEqual(result['status'],'completed')
        self.assertEqual((result['GRT_counts']['planned'],result['GRT_counts']['success'],
            result['DRT_counts']['completed']),(144,144,72))
        self.assertEqual((len(result['GRT_streams']),len(result['DRT_streams']),len(result['paired_vs_new_controls'])),(18,18,20))
        self.assertEqual((result['backend_calls'],result['model_inference_calls']),(0,0))
        self.assertFalse(result['actor_advanced']);self.assertFalse(result['heldout_used'])
        self.assertEqual({r['control'] for r in result['paired_vs_new_controls']},set(m.CONTROLS))

    def test_changed_route_count_or_prepared_config_refused_before_physical_data(self):
        values=self.collection_fixture();values[5].validate_plan=lambda *a:{'requested':71}
        with self.assertRaises((ValueError,KeyError)):self.run_collection(values)


if __name__=='__main__':unittest.main(verbosity=2)
