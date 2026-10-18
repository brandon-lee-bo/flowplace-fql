"""Bounded metadata/native-interface fixtures; no models, scheduler or EDA."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import qstudy_quality_weighted_route as route
from .graph_o2o_route_controller import requested, physical_run_id, route_cap


class WeightedRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'; self.source.mkdir()
        (self.source/'protocol.md').write_text('fixed protocol')
        (self.source/'p.py').write_text('captured weighted reader')
        self.output = self.root/'routes'
        self.report_path = self.root/'report/report.json'
        self.report_path.parent.mkdir()
        self.gate = dict(status=route.PASS, full48_success_numeric_gates=True)
        self.inputs = []
        self.requests = {}
        train = self.root/'training/plan.json'
        train.parent.mkdir(); train.write_text('{}')
        for arm in route.ARMS:
            experiment = self.root/arm; experiment.mkdir()
            (experiment/'config.json').write_text(json.dumps(dict(arm=arm)))
            for name in ('declaration.json','export_completed.json'):
                (experiment/name).write_text('{}')
            streams = []
            trained = {a:{str(s):dict(checkpoint=dict(path='flow_'+a+str(s),sha256='s'),
                    steps=1000,draw_chain_sha256='paired'+str(s)) for s in route.SEEDS} for a in route.ARMS}
            for seed in route.SEEDS:
                folder = experiment/f'seed_{seed}/round_0/requests'
                folder.mkdir(parents=True)
                tasks = folder/'tasks.jsonl'; tasks.write_text('fixed tasks')
                (folder/'completed.json').write_text('{}')
                rows, requests_rows = [], []
                for index in range(8):
                    sample = folder/arm/'evaluation'/str(index)
                    sample.mkdir(parents=True)
                    noise = np.full((2,2),float(index),dtype=np.float32)
                    action = noise/20
                    np.savez(sample/'actions.npz',noise=noise,raw_action=action,action=action)
                    request = dict(arm=arm,seed=seed,round=0,role='evaluation',index=index,
                        source_kind='quality_weighted_flow_development',source_sampler='Euler10',
                        sample_dir=str(sample),training_feedback_eligible=False,offline_train_eligible=False,
                        raw_action_sha256='raw'+str(index),action_sha256='action'+str(index),
                        noise_sha256='noise'+str(index),noise_seed=4600000,
                        source_training_artifacts=trained[arm][str(seed)],def_sha256='DEF')
                    (sample/'request.json').write_text(json.dumps(request))
                    grt_metrics={k:float(index+1) for k in route.weighted.native.METRICS}
                    (sample/'outcome.json').write_text(json.dumps(dict(status='success',metrics=grt_metrics)))
                    rows.append(dict(index=index,seed=seed,state='success',noise_sha256=request['noise_sha256'],
                        raw_action_sha256=request['raw_action_sha256'],action_sha256=request['action_sha256'],
                        request=route.bind(sample/'request.json'),actions=route.bind(sample/'actions.npz'),
                        outcome=route.bind(sample/'outcome.json'),metrics=grt_metrics,
                        native=dict(physical_run=str(self.root/'native'/arm/str(seed)/str(index)),
                            physical_manifest_sha256='native_manifest')))
                    requests_rows.append(request)
                streams.append(dict(seed=seed,success=8,fully_accounted=True,integrity_errors=[],rows=rows,
                    tasks=route.bind(tasks)))
                self.requests[(str(experiment),seed)] = (requests_rows,tasks)
            self.inputs.append(dict(arm=arm,experiment=str(experiment),streams=streams,
                config=route.bind(experiment/'config.json'),declaration=route.bind(experiment/'declaration.json'),
                completion=route.bind(experiment/'export_completed.json'),training_manifest=route.bind(train),
                all_paired_training=trained,physical_binding=dict(dataset_sha256='dataset',physical_flow_revision='physical'),
                producer_source_validation=dict(observer_extra_source_sha256={})))
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for item in (patch.object(route,'ROOT',self.source),
                     patch.object(route.weighted,'SOURCE_NAMES',('p.py',)),
                     patch.object(route.weighted,'PROTOCOL','protocol.md'),
                     patch.object(route.weighted,'PROTOCOL_SHA',route.sha(self.source/'protocol.md')),
                     patch.object(route.weighted,'collect_arm',side_effect=lambda p,a:copy.deepcopy(self.inputs[route.ARMS.index(a)])),
                     patch.object(route.weighted,'matched_context'),
                     patch.object(route.weighted,'protocol_gates',side_effect=lambda a,b:copy.deepcopy(self.gate)),
                     patch.object(route,'sources',return_value={'own.py':'frozen'}),
                     patch.object(route,'load_requests',side_effect=lambda p,s,r:self.requests[(str(p),s)])):
            self.stack.enter_context(item)
        self.publish_grt()

    def publish_grt(self):
        report=dict(schema='qstudy_quality_weighted_native_results_v1',status='complete',stage='global_route_only',
            development_only=True,no_replay=True,selected_winner=None,new_backend_calls=0,
            protocol=route.bind(self.source/'protocol.md'),source_sha256={'p.py':route.sha(self.source/'p.py')},
            experiments=copy.deepcopy(self.inputs),development_gates=copy.deepcopy(self.gate))
        self.report_path.write_text(json.dumps(report))
        (self.report_path.parent/'completed.json').write_text(json.dumps(dict(status='completed',report_status='complete',
            backend_calls=0,model_inference_calls=0,artifacts={'report.json':route.sha(self.report_path)})))

    def checked(self):
        return route.checked_grt_report(self.report_path,route.sha(self.report_path))

    def plan(self):
        return route.build_plan(self.output,self.report_path,route.sha(self.report_path))

    def publish(self):
        plan=self.plan()
        route.native.freeze(self.output/'full_route_plan.json',plan)
        route.native.freeze(self.output/'route_request.json',dict(schema=route.SCHEMA,
            plan=route.bind(self.output/'full_route_plan.json'),requested=24,purpose='development_validation',
            helper_sha256=route.sha(route.__file__),conditional_on_prespecified_GRT_gate=True))
        return plan

    def test_completed48_gate_and_observer_only_source_evidence(self):
        self.assertEqual(len(self.checked()[1]),2)
        self.inputs[0]['producer_source_validation']['observer_extra_source_sha256']['observer.py']='new'
        self.assertEqual(len(self.checked()[1]),2)
        self.inputs[0]['streams'][0]['rows'][0]['metrics']['rWL_um']+=1
        with self.assertRaisesRegex(ValueError,'revalidated native'):self.checked()

    def test_failed_gate_missing_results_and_report_tamper_refused(self):
        self.gate['status']='fails_declared_development_gate';self.publish_grt()
        with self.assertRaisesRegex(ValueError,'gate did not pass'):self.checked()
        self.gate['status']=route.PASS;self.publish_grt()
        self.inputs[0]['streams'][0]['success']=7
        with self.assertRaises(ValueError):self.checked()
        digest=route.sha(self.report_path);self.report_path.write_text('{}')
        with self.assertRaises(ValueError):route.checked_grt_report(self.report_path,digest)

    def test_fixed24_original_controller_contract_and_first4(self):
        plan=self.plan();tasks=requested(self.output,plan)
        self.assertEqual(len(tasks),24);self.assertEqual(route_cap(plan),20)
        self.assertEqual(len({physical_run_id(plan,t['key']) for t in tasks}),24)
        for arm in route.ARMS:
            for seed in route.SEEDS:
                self.assertEqual([t['noise_index'] for t in plan['tasks'] if t['arm']==arm and t['seed']==seed],[0,1,2,3])
        self.assertTrue(all(t['run_id'].startswith('go2o_') for t in plan['tasks']))
        self.assertFalse(plan['training_feedback_eligible']);self.assertFalse(plan['offline_train_eligible'])

    def test_budget_namespace_overlap_guards(self):
        for prefix,count in [('bad',2),('go2o_good',3),('go2o_good',0),('go2o_good',True)]:
            with self.assertRaises(ValueError):route.build_plan(self.output,self.report_path,route.sha(self.report_path),prefix,count)
        with self.assertRaises(ValueError):route.build_plan(self.root/'uniform/routes',self.report_path,route.sha(self.report_path))

    def test_plan_drop_source_and_config_tampering_refused(self):
        plan=self.publish();digest=route.sha(self.output/'full_route_plan.json')
        self.assertEqual(route.validate_plan(self.output,digest),plan)
        with self.assertRaises(ValueError):route.validate_plan(self.output,'0'*64)
        with patch.object(route,'sources',return_value={'own.py':'changed'}):
            with self.assertRaises(ValueError):route.validate_plan(self.output)
        path=self.output/'full_route_plan.json';changed=copy.deepcopy(plan);changed['tasks'].pop()
        path.write_text(json.dumps(changed))
        declaration=route.read(self.output/'route_request.json');declaration['plan']=route.bind(path)
        (self.output/'route_request.json').write_text(json.dumps(declaration))
        with self.assertRaises(ValueError):route.validate_plan(self.output)

    def test_seal_resume_exact_arrays_and_immutable_source(self):
        plan=self.publish();loaded=self.checked()
        rows=route.inspect_sources(self.output,plan,seal=True,loaded=loaded)
        self.assertEqual(CounterState(rows),{'resolved':24})
        binding=rows[0]['binding']
        self.assertEqual(set(binding['exact_coordinates_and_noise']),{'noise','raw_action','action'})
        self.assertEqual(binding['trained_flow']['steps'],1000)
        self.assertEqual(rows,route.inspect_sources(self.output,plan,seal=False,loaded=loaded))
        path=self.output/'source_bindings'/(rows[0]['key']+'.json')
        value=route.read(path);value['noise_seed']=99;path.write_text(json.dumps(value))
        failed=route.inspect_sources(self.output,plan,seal=False,loaded=loaded)
        self.assertEqual(failed[0]['state'],'invalid')

    def test_source_request_noise_tamper_cannot_change_sealed_pair(self):
        plan=self.plan();loaded=self.checked()
        self.requests[(str(self.root/'quality_rank'),0)][0][0]['noise_sha256']='wrongnoise'
        rows=route.inspect_sources(self.output,plan,seal=True,loaded=loaded)
        self.assertEqual(sum(r['state']=='invalid' for r in rows),1)
        self.assertEqual(sum(r['state']=='resolved' for r in rows),23)

    def final_rows(self):
        return [dict(key=t['key'],arm=t['arm'],seed=t['seed'],round=0,noise_index=t['noise_index'],
            state='completed',noise_sha256='n'+str(t['noise_index']),
            metrics={k:float(t['seed']+(2 if t['arm']=='quality_rank' else 1)) for k in route.native.METRICS},
            physical_identity={'native':'same'},physical_run='unique_'+t['key']) for t in self.plan()['tasks']]

    def test_three_seed_pairing_drc_measured_and_failure_denominators(self):
        rows=self.final_rows();value=route.aggregate(rows)
        self.assertEqual(value['matched_deltas']['statistics']['DRT_WL'],dict(n=3,mean=1.,sd=0.))
        self.assertEqual(value['policy_groups'][0]['statistics']['DRT_WL']['sd'],1.)
        self.assertEqual(value['policy_groups'][0]['drc_clean'],0)
        rows[0].update(state='upstream_physical_failure',metrics=None)
        value=route.aggregate(rows)
        self.assertIsNone(value['policy_groups'][0]['statistics'])
        self.assertEqual(value['policy_groups'][0]['planned'],12)
        self.assertIsNone(value['matched_deltas']['statistics'])
        self.assertEqual(value['matched_deltas']['independent_complete_seeds'],2)

    def test_original_final_inspector_delegation_and_native_call_uniqueness(self):
        plan=self.publish();rows=self.final_rows()
        with patch.object(route.native,'inspect_final',side_effect=copy.deepcopy(rows)) as inspector:
            report=route.report_value(self.output,plan,[{'state':'resolved'}]*24)
            self.assertEqual(inspector.call_count,24);self.assertEqual(report['status'],'all_requested_accounted')
        rows[1]['physical_run']=rows[0]['physical_run']
        with patch.object(route.native,'inspect_final',side_effect=rows):
            report=route.report_value(self.output,plan,[{'state':'resolved'}]*24)
        self.assertEqual(report['counts']['invalid'],2)
        self.assertEqual(report['requested'],24)
        self.assertEqual(report['status'],'partial_or_invalid')


def CounterState(rows):
    from collections import Counter
    return dict(Counter(r['state'] for r in rows))


if __name__=='__main__':
    unittest.main()
