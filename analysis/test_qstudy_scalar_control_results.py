"""Small saved-only fixtures; no checkpoint, inference, EDA or scheduler use."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import qstudy_scalar_control_results as report


def metrics(value=1., final=False):
    names=report.final.METRICS if final else report.native.METRICS
    return {name:(0 if name=='DRC' else -value if name in ('WNS','TNS','WNS_ns','TNS_ns') else value)
            for name in names}


def population(success=False):
    """The declared topology, without any actual project measurements."""
    streams=[]; drt=[];specs=[]
    for stage in (0,1,2):
        keys=('bc_t0','fql_t0') if stage==0 else tuple(a+'_t'+str(stage) for a in report.ARMS)
        for key in keys:
            for seed in report.SEEDS:
                rows=[]
                for role in (('collect','evaluation') if stage<2 else ('evaluation',)):
                    for index in range(8):
                        row=dict(key=f'{key}_s{seed}_{role}_{index}',physical_key=key,arm='soc_'+key,
                            seed=seed,round=stage,role=role,index=index,
                            noise_sha256=(f'e{index}' if role=='evaluation' else f'c{stage}s{seed}n{index}'),
                            state='success' if success else 'not_exported',metrics=metrics(seed+index+1) if success else None)
                        rows.append(row);specs.append(deepcopy(row))
                streams.append(dict(physical_key=key,seed=seed,round=stage,errors=[],rows=rows))
                if stage in (0,2):
                    for index in range(4):
                        drt.append(dict(key=f'{key}_s{seed}_n{index}',group=key,arm='soc_'+key,
                            seed=seed,round=stage,noise_index=index,noise_sha256=f'e{index}' if success else None,
                            state='completed' if success else 'source_missing',metrics=metrics(seed+index+1,True) if success else None))
    return dict(physical_grt_requests=specs,physical_drt_requests=deepcopy(drt)),streams,drt


class ReductionTests(unittest.TestCase):
    def test_partial_inventory_keeps384_and72_without_fabrication(self):
        plan,streams,drt=population()
        with patch.object(report,'bind',return_value={'path':'fixture','sha256':'0'*64}):
            result=report.build_report(plan,streams,drt,None)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['GRT_counts'],{'not_exported':384})
        self.assertEqual(result['DRT_counts'],{'source_missing':72})
        self.assertEqual(result['budgets']['T0_physical_roots'],2)
        self.assertEqual(len([g for g in result['policy_groups'] if g['stage']=='GRT' and g['round']==0]),2)
        self.assertTrue(all(g['metrics'] is None for g in result['policy_groups']))

    def test_seed_mean_sd_and_no_generation_pseudoreplication(self):
        plan,streams,drt=population(True)
        with patch.object(report,'bind',return_value={'path':'fixture','sha256':'0'*64}):
            result=report.build_report(plan,streams,drt,None)
        self.assertEqual(result['status'],'complete')
        group=next(g for g in result['policy_groups'] if g['stage']=='GRT' and g['physical_key']=='bc_t0')
        self.assertEqual(group['metrics']['rWL_um'],dict(n=3,mean=5.5,sd=1.,eligible=True))
        self.assertEqual(group['planned'],24)
        drtgroup=next(g for g in result['policy_groups'] if g['stage']=='DRT' and g['physical_key']=='bc_t0')
        self.assertEqual(drtgroup['planned'],12)
        self.assertEqual(drtgroup['metrics']['DRC']['mean'],0)

    def test_physical_failure_is_accounted_but_not_zero_ppa(self):
        _,streams,_=population(True)
        source=next(s for s in streams if s['physical_key']=='bc_t0' and s['seed']==0)
        rows=[deepcopy(r) for r in source['rows'] if r['role']=='evaluation']
        for row in rows:row.update(state='physical_failure',metrics=None)
        group=report.group_summary(rows,'GRT','bc_t0',0,'evaluation')
        self.assertTrue(group['fully_accounted']);self.assertTrue(group['conditional_PPA'])
        self.assertEqual(group['success_n'],0);self.assertIsNone(group['successful_metric_means']['rWL_um'])
        groups=[group]
        for seed in (1,2):
            these=[r for s in streams if s['physical_key']=='bc_t0' and s['seed']==seed for r in s['rows'] if r['role']=='evaluation']
            groups.append(report.group_summary(these,'GRT','bc_t0',seed,'evaluation'))
        aggregate=report.aggregate_groups(groups)[0]
        self.assertTrue(aggregate['fully_accounted'])
        self.assertEqual(aggregate['metrics']['rWL_um'],dict(n=2,mean=None,sd=None,eligible=False))

    def test_wrong_noise_denominator_refused_and_drc_retained(self):
        _,_,drt=population(True)
        rows=[r for r in drt if r['group']=='bc_t0' and r['seed']==0]
        rows[0]['metrics']['DRC']=7
        result=report.group_summary(rows,'DRT','bc_t0',0,'evaluation')
        self.assertTrue(result['all_success']);self.assertEqual(result['DRC_positive'],1)
        self.assertEqual(result['successful_metric_means']['DRC'],1.75)
        rows[0]['noise_index']=1
        with self.assertRaisesRegex(ValueError,'denominator'):
            report.group_summary(rows,'DRT','bc_t0',0,'evaluation')

    def test_noise_mismatch_and_eval_collection_overlap_refused(self):
        _,streams,_=population(True)
        rows=[r for s in streams for r in s['rows']]
        self.assertTrue(report.noise_gate(rows)['eligible'])
        rows[-1]['noise_sha256']='wrong'
        self.assertFalse(report.noise_gate(rows)['eligible'])
        rows[-1]['noise_sha256']='e7'
        rows[0]['noise_sha256']='e0'
        self.assertIn('Evaluation noise entered a collection stream',report.noise_gate(rows)['errors'])

    def test_actual_same_noise_deltas_and_missing_pair_withholding(self):
        _,streams,_=population(True);rows=[r for s in streams for r in s['rows']]
        for row in rows:
            if row['physical_key']=='fql_t0':row['metrics']['rWL_um']+=row['seed']+1
        delta=report.matched_deltas(rows,'GRT')[0]
        self.assertTrue(delta['eligible'])
        self.assertEqual(delta['statistics']['rWL_um'],dict(n=3,mean=2.,sd=1.))
        row=next(r for r in rows if r['physical_key']=='fql_t0' and r['seed']==1 and r['role']=='evaluation')
        row.update(state='missing',metrics=None)
        delta=report.matched_deltas(rows,'GRT')[0]
        self.assertFalse(delta['eligible']);self.assertIsNone(delta['statistics'])
        self.assertEqual(delta['per_seed'][1]['both_success_same_noise_n'],7)

    def test_native_merge_accepts_real_duplicate_binding_fields_not_identity(self):
        expected=dict(seed=0,index=1,role='evaluation',sample_dir='/fixture',key='sealed')
        item=dict(seed=0,index=1,role='evaluation',sample_dir='/fixture',request={'old':1},actions={'old':2},state='missing')
        row=report.merge_native_row(expected,item,request={'sha256':'a'},actions={'sha256':'b'})
        self.assertEqual(row['key'],'sealed');self.assertEqual(row['request'],{'sha256':'a'})
        item['seed']=1
        with self.assertRaisesRegex(ValueError,'identity'):report.merge_native_row(expected,item)

    def test_duplicate_native_run_invalidates_every_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'run';run.mkdir();alias=Path(directory)/'alias';alias.symlink_to(run,target_is_directory=True)
            rows=[dict(state='success',metrics=metrics(),native={'physical_run':str(path)}) for path in (run,alias)]
            report.reject_aliased_grt_runs(rows)
            self.assertTrue(all(r['state']=='invalid' and r['metrics'] is None for r in rows))

    def test_original_native_geometry_guard_prevents_implicit_eda(self):
        with tempfile.TemporaryDirectory() as directory:
            sample=Path(directory)/'sample';sample.mkdir();run=Path(directory)/'native';run.mkdir()
            row=dict(seed=0,round=0,arm='a',role='evaluation',index=0,graph='bp_fe',training_feedback_eligible=False)
            for name in ('request.json','actions.npz'):(sample/name).write_bytes(b'fixture')
            (sample/'outcome.json').write_text(json.dumps(dict(row,status='success',physical_run=str(run))))
            (run/'manifest.json').write_text('{"success":true}')
            with patch('fql.qstudy_backend.checked_existing',side_effect=AssertionError('would invoke geometry')) as checker:
                result=report.baseline.inspect_outcome(row,sample,sample/'tasks.jsonl',({},None,{}))
            checker.assert_not_called()
            self.assertEqual(result['state'],'invalid');self.assertIn('Missing saved native geometry audit',result['integrity_error'])

    def test_explicit_route_plan_required_never_inspect_unbound_final(self):
        plan,streams,drt=population(True);plan['output']='/fixture'
        rows=[r for s in streams for r in s['rows']]
        with patch.object(report.final,'inspect_final',side_effect=AssertionError('unbound read')) as inspect:
            results,evidence=report.collect_drt(plan,rows,None)
        inspect.assert_not_called();self.assertIsNone(evidence)
        self.assertEqual(len(results),72);self.assertTrue(all(r['state']=='source_missing' for r in results))

    def test_route_namespace_translation_only_and_original_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); route=root/'route';route.mkdir()
            manifest=root/'manifest.json';manifest.write_text('{"fixture":true}')
            original=[dict(key='bc_t0_s0_n0',group='bc_t0',seed=0,round=0,noise_index=0,run_id='socdrt1_bc_t0_s0_n0')]
            plan=dict(output=str(root),physical_drt_requests=original)
            rows=[dict(role='evaluation',physical_key='bc_t0',seed=0,index=0,state='missing')]
            digest=report.hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            physical=dict(base_manifest=report.bind(manifest),tasks=[dict(original[0],run_id='go2o_socdrt1_bc_t0_s0_n0')],
                run_id_translation=report.RUN_ID_TRANSLATION,original_tasks_sha256=digest,
                design='bp_fe',detailed_route_end_iteration=20,backend_threads=8,backend_storage='scratch')
            path=route/'full_route_plan.json'
            with patch.object(report,'MANIFEST',manifest),patch.object(report,'MANIFEST_SHA',report.native.sha(manifest)):
                for change in (None,'untranslated','other_field','bad_digest'):
                    value=deepcopy(physical)
                    if change=='untranslated':value['tasks']=original
                    if change=='other_field':value['tasks'][0]['seed']=1
                    if change=='bad_digest':value['original_tasks_sha256']='f'*64
                    path.write_text(json.dumps(value))
                    if change is None:
                        result,_=report.collect_drt(plan,rows,report.bind(path))
                        self.assertEqual(result[0]['run_id'],'go2o_socdrt1_bc_t0_s0_n0')
                    else:
                        with self.assertRaisesRegex(ValueError,'sealed72-task'):
                            report.collect_drt(plan,rows,report.bind(path))

    def test_drt_source_binds_actual_scalar_checkpoint_and_native_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'outcome.json';path.write_text('{"failure_class":null}')
            task=dict(key='fql_t0_s0_n0',arm='soc_fql_t0',seed=0,round=0,noise_index=0,source_outcome=str(path))
            grt=dict(role='evaluation',state='success',outcome=report.bind(path),request={'sha256':'request'},
                tasks={'sha256':'tasks'},export_completion={'sha256':'complete'},actions={'sha256':'actions'},
                raw_action_sha256='raw',action_sha256='executed',noise_sha256='noise',noise_stream_seed=81400000,
                config={'sha256':'cfg'},dataset_sha256='dataset',physical_flow_revision={'revision':'fixed'},
                def_sha256='def',checkpoint_sha256='actualT2',source_checkpoint={'path':'actualT2.pt','sha256':'actualT2'},metrics=metrics())
            binding=dict(task,status='success',source_outcome_sha256=grt['outcome']['sha256'],
                request_sha256='request',tasks_sha256='tasks',export_completion_sha256='complete',actions_npz_sha256='actions',
                raw_action_sha256='raw',action_sha256='executed',noise_sha256='noise',noise_seed=81400000,
                config_sha256='cfg',dataset_sha256='dataset',physical_flow_revision={'revision':'fixed'},def_sha256='def',
                failure_class=None,native={'manifest':'native'},GRT_metrics=metrics(),
                actor={'checkpoint':{'path':'actualT2.pt','sha256':'actualT2'}})
            with patch.object(report.native,'native_fingerprints',return_value={'manifest':'native'}):
                report.check_final_source(task,grt,binding)
                for key,value in (('actor',{'checkpoint':{'path':'oldT1.pt','sha256':'oldT1'}}),('action_sha256','wrong'),('dataset_sha256','wrong')):
                    changed=deepcopy(binding);changed[key]=value
                    with self.assertRaises(ValueError):report.check_final_source(task,grt,changed)


if __name__=='__main__':unittest.main()
