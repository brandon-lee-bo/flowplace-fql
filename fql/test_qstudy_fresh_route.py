"""Synthetic fresh DRT contracts; no fresh policy, container or physical call."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from . import qstudy_fresh_route as route
from . import qstudy_fresh_route_report as report
from . import qstudy_fresh_eval as fresh
from . import test_qstudy_fresh_native as fixtures


class FreshRouteTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use bounded CPU Slurm')
        self.fixture=fixtures.FreshNativeTests('test_real_native_geometry_failure_has_B4_no_PPA_and_never_calls_EDA')
        self.fixture.setUp();self.root=self.fixture.root
        self.args=argparse.Namespace(selection=self.fixture.f.path,selection_sha256=self.fixture.f.digest,
            output=self.root/'fresh_drt')

    def tearDown(self):self.fixture.tearDown()

    def prepared(self):
        plan=route.prepare(self.args);path=self.args.output/'plan.json'
        return path,route.checked_plan(path,fresh.sha(path))

    def geometry_outcomes(self):
        for _,_,rows,tasks in self.fixture.exports:
            for row in rows:self.fixture.geometry_outcome(row,tasks)

    def native_final(self,plan,task,drc=3):
        """Real final-auditor files, including native DEF parse and route log."""
        grt=self.root/'grt';grt.mkdir(exist_ok=True)
        components=dict(openroad_binary_sha256='a'*64,design_config_sha256='b'*64)
        old=dict(success=True,time_end=1,stage='global_route_only',design='bp_fe',num_cores=8,
            tool_image_sha256='c'*64,physical_flow_components=components,physical_flow_revision='d'*64,
            openroad_sha256='a'*64,design_config_sha256='b'*64,def_sha256='e'*64)
        fresh.atomic_json(grt/'manifest.json',old);fresh.atomic_json(grt/'metrics.json',dict(rWL_um=100))
        geometry=dict(legal_no_halo=True,dbu_per_um=1000,
            bindings=dict(physical_manifest_sha256=fresh.sha(grt/'manifest.json')),
            actual_macros=dict(m0=dict(master='fakeram45_test',bbox_dbu=[1000,2000,3000,4000],orientation='N')))
        fresh.atomic_json(grt/'backend_macro_audit.json',geometry)
        outcome=Path(task['source_outcome']);outcome.parent.mkdir(parents=True,exist_ok=True)
        # This helper tests the final auditor, not the GRT parser (covered by
        # real geometry-failure fixtures below).
        fresh.atomic_json(outcome,dict(status='success',physical_run=str(grt)))
        binding=self.args.output/'source_bindings'/f"{task['key']}.json"
        binding.parent.mkdir(exist_ok=True);fresh.atomic_json(binding,dict(synthetic=True))
        source=dict(state='success',physical_run=str(grt),outcome=fresh.binding(outcome),
            native=dict(physical_manifest_sha256=fresh.sha(grt/'manifest.json')),
            source_binding_artifact=fresh.binding(binding))
        folder=Path(task['output']);folder.mkdir(parents=True)
        spec=route.expected_spec(task,source)
        fresh.atomic_json(folder/'fresh_invocation.json',route.invocation_binding(plan,task,source))
        fresh.atomic_json(folder/'spec.json',spec)
        run=self.root/'drt';run.mkdir()
        final=run/'orfs_work/results/nangate45/bp_fe/eval/6_final.def';final.parent.mkdir(parents=True)
        final.write_text('UNITS DISTANCE MICRONS 1000 ;\nCOMPONENTS 1 ;\n'
            '- m0 fakeram45_test + FIXED ( 1000 2000 ) N ;\nEND COMPONENTS\n')
        log=run/'orfs_work/logs/nangate45/bp_fe/eval/5_2_route.log';log.parent.mkdir(parents=True)
        log.write_text(f'Number of violations = {drc}.\n[INFO DRT-0198] Complete detail routing.\n')
        metrics=dict(DRT_WL=123.,WNS=-.3,TNS=-4.,Power=.2,StdCellArea=1000.,DRC=drc)
        fresh.atomic_json(run/'metrics.json',metrics)
        finish_components={'scripts/detail_route.tcl':'f'*64}
        identity=dict(grt_components=components,finish_components=finish_components,route_end_iteration=20,num_cores=8)
        manifest=dict(old,stage='finish_after_detailed_route',source_grt_run=str(grt),
            source_grt_manifest_sha256=fresh.sha(grt/'manifest.json'),source_grt_metrics_sha256=fresh.sha(grt/'metrics.json'),
            route_end_iteration=20,detailed_route_flow_components=finish_components,
            detailed_route_physical_revision=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
            canonical_metrics_sha256=fresh.sha(run/'metrics.json'),drc_native_final_count=drc,
            route_drc_clean=drc==0,final_macros_match_audited_grt=True,
            source_geometry_sha256=fresh.sha(grt/'backend_macro_audit.json'),final_def_sha256=fresh.sha(final))
        fresh.atomic_json(run/'manifest.json',manifest)
        out=dict(status='completed',spec=spec,spec_sha256=fresh.sha(folder/'spec.json'),physical_run=str(run),
            physical_manifest_sha256=fresh.sha(run/'manifest.json'),metrics_sha256=fresh.sha(run/'metrics.json'),
            metrics=metrics,route_drc_clean=drc==0,final_macros_match_audited_grt=True)
        fresh.atomic_json(folder/'outcome.json',out)
        return source,out,final,log

    def test_predeclared40_tasks_exact_indices_shared_noise_no_GRT_selection(self):
        with patch.object(fresh,'declare',side_effect=AssertionError('writer forbidden')), \
             patch.object(fresh,'shared_noise',side_effect=AssertionError('noise forbidden')), \
             patch.object(fresh,'build_sampler',side_effect=AssertionError('model forbidden')), \
             patch.object(route.finish.backend,'measure',side_effect=AssertionError('EDA forbidden')):
            path,(plan,checked)=self.prepared()
            self.assertEqual(plan['requested'],40);self.assertEqual(plan['route_cap'],20)
            self.assertEqual({r['index'] for r in plan['tasks']},{0,1,2,3})
            self.assertTrue(all(r['run_id'].startswith('go2o_') for r in plan['tasks']))
            self.assertTrue(all(not Path(r['source_outcome']).exists() for r in plan['tasks']))
            first=path.read_bytes();route.prepare(self.args);self.assertEqual(first,path.read_bytes())
            self.assertEqual(len(route.inspect_sources(plan,checked)),40)

    def test_route_plan_code_cap_request_and_export_mutations_refused(self):
        path,(plan,_)=self.prepared()
        for field,value in [('route_cap',21),('concurrency',3),('requested',39),('feedback_eligible',True)]:
            bad=copy.deepcopy(plan);bad[field]=value;fresh.atomic_json(path,bad)
            with self.assertRaises(ValueError):route.checked_plan(path,fresh.sha(path))
        fresh.atomic_json(path,plan)
        _,_,rows,_=self.fixture.exports[0]
        request=Path(rows[0]['sample_dir'])/'request.json';value=fresh.read(request)
        fresh.atomic_json(request,dict(value,fresh_drt_preselected=False))
        with self.assertRaises(ValueError):route.checked_plan(path,fresh.sha(path))

    def test_real40_geometry_failures_no_EDA_fake_PPA_or_final_congestion(self):
        self.geometry_outcomes();path,(plan,checked)=self.prepared()
        with patch.object(route.finish.backend,'measure',side_effect=AssertionError('EDA forbidden')), \
             patch.object(route,'audit_final_outcome',side_effect=AssertionError('No final PPA')):
            sources=route.inspect_sources(plan,checked,seal=True)
            self.assertTrue(all(r['source_binding_artifact'] for r in sources))
            result=report.collect(path,fresh.sha(path))
        self.assertEqual(result['status'],'all_requested_outcomes_accounted')
        self.assertEqual(len(result['records']),40)
        self.assertTrue(all(r['state']=='upstream_physical_failure' and r['metrics'] is None for r in result['records']))
        self.assertTrue(all(r['aggregate_eligible'] and r['complete_policy_PPA_statistics'] is None for r in result['stream_statistics']))
        self.assertEqual(result['call_accounting']['unique_physical_calls_with_manifest'],0)
        self.assertIsNone(result['final_congestion']);self.assertEqual(result['backend_calls'],0)

    def test_sealed_GRT_failure_source_cannot_disappear_or_change(self):
        self.geometry_outcomes();_,(plan,checked)=self.prepared()
        route.inspect_sources(plan,checked,seal=True)
        Path(plan['tasks'][0]['source_outcome']).unlink()
        self.assertEqual(route.inspect_sources(plan,checked)[0]['state'],'invalid')

    def test_real_native_final_auditor_retains_nonzero_DRC_and_detects_macro_change(self):
        _,(plan,_)=self.prepared();task=plan['tasks'][0]
        source,out,final,_=self.native_final(plan,task,drc=3)
        with patch.object(route.finish.backend,'measure',side_effect=AssertionError('EDA forbidden')):
            result=route.inspect_final(plan,task,source)
        self.assertEqual(result['state'],'completed');self.assertEqual(result['metrics']['DRC'],3)
        self.assertFalse(result['route_drc_clean']);self.assertTrue(result['final_native_audit']['actual_macro_DEF_reparsed'])
        final.write_text(final.read_text().replace('( 1000 2000 )','( 1001 2000 )'))
        with self.assertRaisesRegex(ValueError,'Actual final DEF changed'):route.inspect_final(plan,task,source)

    def test_invocation_and_exact_source_spec_required_before_final_audit(self):
        _,(plan,_)=self.prepared();task=plan['tasks'][0]
        source,_,_,_=self.native_final(plan,task)
        invocation=Path(task['output'])/'fresh_invocation.json';old=invocation.read_bytes()
        invocation.unlink()
        with self.assertRaisesRegex(ValueError,'invocation'):route.inspect_final(plan,task,source)
        invocation.write_bytes(old)
        spec=Path(task['output'])/'spec.json';fresh.atomic_json(spec,dict(fresh.read(spec),route_end_iteration=21))
        with self.assertRaisesRegex(ValueError,'exact locked fresh source'):route.inspect_final(plan,task,source)

    def test_worker_exact_frozen_cli_and_cpu_guard_no_new_sampler(self):
        path,(plan,checked)=self.prepared();task=plan['tasks'][0]
        # Only the physical-call subprocess is mocked; plan/export/request
        # validators above remain real. Worker dispatch is tested independently
        # of the synthetic upstream geometry failure.
        binding=self.args.output/'source_bindings'/f"{task['key']}.json";binding.parent.mkdir(exist_ok=True)
        fresh.atomic_json(binding,dict(synthetic=True));outcome=Path(task['source_outcome'])
        fresh.atomic_json(outcome,dict(synthetic=True))
        source=dict(state='success',source_binding_artifact=fresh.binding(binding),outcome=fresh.binding(outcome))
        args=argparse.Namespace(plan=path,plan_sha256=fresh.sha(path),index=0)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'2'}):
            with self.assertRaises(RuntimeError):route.worker(args)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'8'}), \
             patch.object(route,'inspect_sources',return_value=[source]), \
             patch.object(route,'inspect_final',side_effect=[dict(state='missing_DRT'),dict(state='completed')]), \
             patch.object(route.subprocess,'run') as call:
            route.worker(args)
        command=call.call_args.args[0]
        self.assertEqual(command[2:4],['-m','fql.graph_o2o_full_route'])
        self.assertEqual(command[-2:],['--end-iter','20'])
        self.assertEqual(command[command.index('--source-outcome')+1],task['source_outcome'])
        self.assertEqual(command[command.index('--storage')+1],'scratch')
        self.assertTrue((Path(task['output'])/'fresh_invocation.json').exists())
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'8'}), \
             patch.object(route,'inspect_sources',return_value=[source]), \
             patch.object(route,'inspect_final',return_value=dict(state='missing_DRT')), \
             patch.object(route.subprocess,'run'):
            with self.assertRaisesRegex(RuntimeError,'without a validated final outcome'):
                route.worker(args)

    def test_missing_only_retry_true_preemption_and_ordinary_failure_does_not_stop_others(self):
        _,(plan,_)=self.prepared();rows=[dict(state='missing_DRT') for _ in plan['tasks']]
        rows[3]=dict(state='completed');rows[4]=dict(state='upstream_physical_failure')
        state=dict(jobs={'0':['100'],'1':['101'],'2':['102']})
        def observer(job):return {'100':['FAILED'],'101':['PREEMPTED'],'102':['RUNNING']}[job]
        decision=route.next_actions(plan,state,rows,observer)
        self.assertEqual(decision['busy'],1);self.assertEqual(decision['submit_indices'],[1])
        self.assertIn('0',decision['attention']);self.assertNotIn(3,decision['unresolved'])
        rows[2]=dict(state='missing_DRT')
        decision=route.next_actions(plan,state,rows,lambda _:['COMPLETED'])
        self.assertTrue(all(str(i) in decision['attention'] for i in range(3)))
        self.assertEqual(decision['submit_indices'],[5,6])

    def test_controller_never_calls_worker_for_already_accounted_sources(self):
        self.geometry_outcomes();path,(plan,_)=self.prepared()
        with patch.object(route.finish,'submit_unique',side_effect=AssertionError('No jobs needed')):
            route.controller(argparse.Namespace(plan=path,plan_sha256=fresh.sha(path)))
        marker=fresh.read(self.args.output/'completed.json')
        self.assertEqual(marker['logical_requests'],40);self.assertFalse(marker['feedback_eligible'])

    def test_truncated_source_or_result_inventory_refused(self):
        _,(plan,_)=self.prepared()
        with self.assertRaisesRegex(ValueError,'source denominator'):
            route.inspect_finals(plan,[])
        with self.assertRaisesRegex(ValueError,'result denominator'):
            route.next_actions(plan,dict(jobs={}),[],lambda _:[])


class FreshRouteStatisticsTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use bounded CPU Slurm')
        self.streams=[dict(id='candidate',purpose='candidate',kind='one_step',seeds=[0,1,2]),
            dict(id='bc',purpose='bc',kind='one_step',seeds=[0,1,2]),
            dict(id='public',purpose='public_reference',kind='public_official20',seeds=[0])]
        self.rows=[]
        for stream in self.streams:
            for seed in stream['seeds']:
                for index in range(4):
                    value=100+10*seed+index+(-2 if stream['id']=='candidate' else 0)
                    self.rows.append(dict(stream=stream['id'],seed=seed,index=index,state='completed',noise_sha256=f'noise{index}',
                        metrics={k:float(value) if k!='DRC' else index for k in route.METRICS}))

    def summaries(self):return [report.summarize(self.rows,s,seed) for s in self.streams for seed in s['seeds']]

    def test_three_training_seed_SD_vs_one_public_checkpoint_noise_SD(self):
        values=self.summaries();current=report.aggregate(values,self.streams[0])
        self.assertEqual(current['complete_policy_PPA_statistics']['DRT_WL'],dict(n=3,mean=109.5,sd=10.))
        public=report.aggregate(values,self.streams[2]);self.assertEqual(public['independent_training_seed_n'],0)
        self.assertEqual(public['complete_policy_PPA_statistics']['DRT_WL']['n'],4)
        self.assertEqual(public['public_frozen_checkpoint_n'],1)

    def test_missing_withholds_policy_mean_failures_retained_in_denominator(self):
        self.rows[0].update(state='missing_DRT',metrics=None)
        current=report.aggregate(self.summaries(),self.streams[0])
        self.assertFalse(current['aggregate_eligible']);self.assertIsNone(current['success_conditional_PPA_statistics'])
        self.rows[0]['state']='upstream_physical_failure'
        current=report.aggregate(self.summaries(),self.streams[0])
        self.assertTrue(current['aggregate_eligible']);self.assertTrue(current['conditional_PPA'])
        self.assertIsNone(current['complete_policy_PPA_statistics'])
        self.assertEqual(current['request_accounting']['planned'],12)
        self.assertEqual(current['successful_PPA_denominator'],11)
        self.assertIsNotNone(current['success_conditional_PPA_statistics'])

    def test_same_actual_noise_paired_deltas_and_missing_denominator(self):
        pairs,seeds,stats=report.paired(self.rows,self.streams)
        bc=next(r for r in stats if r['reference']=='bc')
        self.assertEqual(bc['metric_deltas']['DRT_WL'],dict(n=3,mean=-2.,sd=0.))
        self.assertEqual(bc['planned_pairs'],12);self.assertTrue(bc['metric_deltas_conditioned_on_both_success'])
        self.rows[0].update(state='missing_DRT',metrics=None)
        self.assertIsNone(next(r for r in report.paired(self.rows,self.streams)[2] if r['reference']=='bc')['metric_deltas'])
        self.rows[0]['noise_sha256']='different'
        with self.assertRaisesRegex(ValueError,'same actual saved noise'):report.paired(self.rows,self.streams)


if __name__=='__main__':unittest.main()
