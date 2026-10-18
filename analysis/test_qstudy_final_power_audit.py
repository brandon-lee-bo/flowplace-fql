"""Small saved-native fixtures; no model, STA, backend or real label changes."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from analysis import qstudy_final_power_audit as m


class PowerAuditTests(unittest.TestCase):
    def setUp(self):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run through bounded CPU Slurm')
        temporary = tempfile.TemporaryDirectory(prefix='qstudy-power-fixture-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self,path,value):
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value if isinstance(value,str) else json.dumps(value))
        return path

    def fixture(self):
        run = self.root/'run'
        logs = run/'orfs_work/logs/nangate45/bp_fe/eval'
        reports = run/'orfs_work/reports/nangate45/bp_fe/eval'
        final = run/'orfs_work/results/nangate45/bp_fe/eval'
        self.write(logs/'5_2_route.json',dict(detailedroute__route__wirelength=1000.,detailedroute__route__drc_errors=0))
        self.write(logs/'6_report.json',dict(finish__timing__setup__ws=-.4,finish__timing__setup__tns=-20.,
            finish__power__total=.3,finish__design__instance__area__stdcell=70000.))
        self.write(logs/'2_1_floorplan.json',{'run__flow__platform__time_units':'1ns','run__flow__platform__distance_units':'1um'})
        text='wns -0.40\ntns -20.0\nfinish report_power\nGroup Internal Switching Leakage Total (Watts)\nTotal 0.1000 0.2000 0.0000 0.3000 100.0%\n'
        self.write(reports/'6_finish.rpt',text)
        author=dict(DRT_WL=1000.,WNS=-.4,TNS=-20.,Power=.3,StdCellArea=70000.,DRC=0)
        self.write(run/'author_metrics.json',author)
        canonical,proof=m.canonicalize(run,author)
        self.write(run/'metrics.json',canonical);self.write(run/'metric_canonicalization.json',proof)
        self.write(final/'6_1_fill.sdc','set_units -time ns\ncreate_clock -name CLK -period 1.400 -waveform {0 .7} [get_ports clk_i]\nset_input_delay .2 [get_ports data]\n')
        source=self.root/'source';self.write(source/'manifest.json',dict(tool_image_sha256='a'*64))
        manifest=dict(canonical_metrics_sha256=m.sha(run/'metrics.json'),metric_evidence_sha256=m.sha(run/'metric_canonicalization.json'),
            design='bp_fe',openroad_sha256='b'*64,design_config_sha256='c'*64,physical_flow_components={'library':'declared'},
            detailed_route_flow_components={'scripts/final_report.tcl':'d'*64},physical_flow_revision='e'*64,
            detailed_route_physical_revision='f'*64,source_grt_run=str(source),stage='finish_after_detailed_route',design_config=str(self.root/'config.mk'))
        self.write(run/'manifest.json',manifest)
        return run

    def test_real_canonicalizer_native_power_and_components(self):
        row=m.native_power(self.fixture(),m.Capture())
        self.assertEqual(row['power_W'],.3)
        self.assertEqual(row['components']['components'],dict(internal_W=.1,switching_W=.2,leakage_W=0.))
        self.assertEqual(row['common_conditions']['tool_image_sha256'],'a'*64)
        self.assertEqual(row['common_conditions']['clock']['clocks'][0]['-period'],'1.4')

    def test_power_sum_rounding_not_exact_float_equality(self):
        rounded='finish report_power\n(Watts)\nTotal .100 .200 .001 .300 100%\n'
        self.assertTrue(m.power_components(rounded)['sum_check']['passed'])
        with self.assertRaisesRegex(ValueError,'rounding'):
            m.power_components(rounded.replace('.300','.250'))
        with self.assertRaisesRegex(ValueError,'component'):
            m.power_components(rounded.replace('.100','-.100'))
        unavailable=m.power_components(rounded.replace('.100','NA'))
        self.assertIsNone(unavailable['components'])
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            m.power_components(rounded.replace('.100','nan'))

    def test_json_components_match_printed_precision_or_remain_explicit_NA(self):
        text=m.power_components('finish report_power\n(Watts)\nTotal .100 .200 .001 .301 100%\n')
        raw={'finish__power__total':.301,'finish__power__internal':.1,
             'finish__power__switching':.2,'finish__power__leakage':.001}
        result=m.json_components(raw,text)
        self.assertEqual(result['available'],3)
        self.assertTrue(result['sum_check']['passed'])
        with self.assertRaisesRegex(ValueError,'precision'):
            m.json_components(dict(raw,finish__power__internal=.5),text)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            m.json_components(dict(raw,finish__power__internal=float('inf')),text)
        del raw['finish__power__leakage']
        result=m.json_components(raw,text)
        self.assertIsNone(result['fields']['leakage_W'])
        self.assertIsNone(result['sum_check'])

    def test_clock_semantics_ignore_layout_bytes_but_reject_changed_period(self):
        a='create_clock -name CLK -period 1.400 -waveform {0 0.700} [get_ports clk_i]\nset_input_delay 0.2 a\n'
        b='# regenerated layout\ncreate_clock -name CLK -period 1.4 -waveform {0 .7} [get_ports clk_i]\nset_input_delay 0.3 b\n'
        self.assertEqual(m.clock_signature(a),m.clock_signature(b))
        self.assertNotEqual(m.clock_signature(a),m.clock_signature(b.replace('1.4','1.5')))
        for command in ('read_saif data.saif','read_vcd data.vcd','set_power_activity -global'):
            with self.assertRaisesRegex(ValueError,'activity'):
                m.clock_signature(a+command)

    def test_no_clock_or_unresolved_generated_clock_is_not_fabricated(self):
        for text in ('set_input_delay .2 a','create_clock -period $x a','create_generated_clock -divide_by 2 a'):
            with self.assertRaises(ValueError): m.clock_signature(text)

    def test_native_raw_tamper_rejected_without_backend(self):
        run=self.fixture();path=run/'orfs_work/logs/nangate45/bp_fe/eval/6_report.json'
        raw=json.loads(path.read_text());raw['finish__power__total']=.4;self.write(path,raw)
        with self.assertRaisesRegex(ValueError,'report changed'):
            m.native_power(run,m.Capture())

    def test_post_inspector_manifest_change_cannot_rebind_to_new_power_view(self):
        run=self.fixture()
        outcome=dict(physical_run=str(run),physical_manifest_sha256=m.sha(run/'manifest.json'),
            metrics_sha256=m.sha(run/'metrics.json'),metrics=json.loads((run/'metrics.json').read_text()))
        m.checked_native_power(outcome,m.Capture())
        manifest=json.loads((run/'manifest.json').read_text());manifest['late_untracked_change']=True
        self.write(run/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'audited outcome'):
            m.checked_native_power(outcome,m.Capture())

    def test_safe_relative_and_capture_mutation(self):
        run=self.fixture()
        with self.assertRaises(ValueError): m.safe_relative(run,'../../foreign')
        capture=m.Capture();path=self.write(self.root/'mutable','before');capture.bytes(path);path.write_text('after')
        with self.assertRaisesRegex(ValueError,'changed'): capture.verify()

    def test_liberty_models_corner_and_missing_transparent(self):
        base='time_unit : "1ns"; voltage_unit : "1V"; nom_process : 1.0; nom_temperature : 25.000; nom_voltage : 1.100;'
        present=m.liberty_metadata(base+' internal_power() {} cell_leakage_power : 0.1;')
        self.assertEqual(present['corner'],dict(nom_process='1',nom_temperature='25',nom_voltage='1.1'))
        self.assertEqual(present['power_model_presence'],'present')
        self.assertEqual(m.liberty_metadata(base)['power_model_presence'],'not_established')
        self.assertIsNone(m.liberty_metadata('library(x){}')['corner']['nom_voltage'])

    def test_static_make_library_expansion_no_execution(self):
        variables=m.make_assignments('export LIB_FILES = $(PLATFORM_DIR)/typ.lib \\\n  $(ADDITIONAL_LIBS)\nexport ADDITIONAL_LIBS = ${PLATFORM_DIR}/ram.lib\n')
        variables['PLATFORM_DIR']='/explicit/platform'
        self.assertEqual(m.expand(variables['LIB_FILES'],variables).split(),['/explicit/platform/typ.lib','/explicit/platform/ram.lib'])
        with self.assertRaisesRegex(ValueError,'resolve'):
            m.expand('$(shell command)',variables)

    def host_fixture(self):
        flow=self.root/'flow';platform=flow/'platforms/nangate45'
        config=self.write(flow/'designs/config.mk','export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram.lib\n')
        self.write(platform/'config.mk','export LIB_FILES = $(PLATFORM_DIR)/lib/typ.lib $(ADDITIONAL_LIBS)\n')
        self.write(platform/'setRC.tcl','set_wire_rc -signal -layer metal3\n')
        scripts=('load.tcl','util.tcl','read_liberty.tcl','report_metrics.tcl','final_report.tcl','deleteRoutingObstructions.tcl')
        for name in scripts:self.write(flow/'scripts'/name,'# inspected source\n')
        liberty='time_unit : "1ns"; nom_process : 1; nom_temperature : 25; nom_voltage : 1.1; internal_power() {} cell_leakage_power : 1;'
        for name in ('typ.lib','fakeram.lib'):self.write(platform/'lib'/name,liberty)
        common=dict(design_config_sha256=m.sha(config),physical_flow_components={'platform_config_sha256':m.sha(platform/'config.mk')},
            detailed_route_flow_components={'scripts/final_report.tcl':m.sha(flow/'scripts/final_report.tcl')})
        manifest=self.write(self.root/'native_manifest/manifest.json',dict(stage='finish_after_detailed_route',design_config=str(config)))
        capture=m.Capture();capture.bytes(manifest)
        return flow,common,capture

    def test_host_library_metadata_not_retroactive_and_missing_models_preserved(self):
        flow,common,capture=self.host_fixture()
        with mock.patch.object(m,'FLOW',flow):
            result=m.host_observations(common,capture)
        self.assertEqual(result['critical_errors'],[])
        self.assertEqual(result['status'],'present_day_observations_only')
        self.assertEqual(sum(x['fakeram'] for x in result['libraries']),1)
        self.assertIn('not retroactive',result['limitation'])
        library=flow/'platforms/nangate45/lib/fakeram.lib'
        library.write_text(library.read_text().replace('internal_power() {}',''))
        capture=m.Capture();capture.bytes(self.root/'native_manifest/manifest.json')
        with mock.patch.object(m,'FLOW',flow):
            result=m.host_observations(common,capture)
        self.assertTrue(result['critical_errors'])
        self.assertEqual(result['libraries'][1]['power_model_presence'],'not_established')

    def test_captured_host_digest_mismatch_fails_closed(self):
        flow,common,capture=self.host_fixture()
        (flow/'scripts/final_report.tcl').write_text('changed current host source')
        with mock.patch.object(m,'FLOW',flow),self.assertRaisesRegex(ValueError,'captured native digest'):
            m.host_observations(common,capture)

    def collector_fixture(self):
        folder=self.root/'collector';folder.mkdir()
        plan=dict(output=str(folder),source_sha256={'source':'a'*64},selection={'selected_families':[]},tasks=[])
        records=[]
        for i in range(64):
            task=dict(index=i,key='k'+str(i),output=str(folder/str(i)))
            path=self.write(Path(task['output'])/'outcome.json',{'physical_run':'none'})
            plan['tasks'].append(task)
            records.append(dict(index=i,key=task['key'],state='completed',outcome=m.collector.bind(path),attempts=[]))
        self.write(folder/'plan.json',plan)
        report=dict(version=m.collector.VERSION,status='complete',requested=64,counts={'completed':64},
            plan=m.collector.bind(folder/'plan.json'),source_sha256=plan['source_sha256'],source_split='original384TRAIN_only',
            policy_evaluation_or_old_DRT_used=False,family_folds=[],records=records,DRC_zero=0)
        return plan,report

    def test_full64_collector_binding_and_row_tamper(self):
        plan,report=self.collector_fixture()
        def inspect(p,t):
            row=copy.deepcopy(report['records'][t['index']]);row.pop('attempts');return row
        with mock.patch.object(m.collector,'inspect_result',side_effect=inspect),mock.patch.object(m.collector,'attempt_evidence',return_value=[]):
            m.checked_collector(plan,report,m.Capture())
            wrong=copy.deepcopy(report);wrong['records'][5]['state']='missing'
            with self.assertRaisesRegex(ValueError,'row differs'):
                m.checked_collector(plan,wrong,m.Capture())
            wrong=copy.deepcopy(report);wrong['status']='incomplete'
            with self.assertRaisesRegex(ValueError,'complete pinned'):
                m.checked_collector(plan,wrong,m.Capture())

    def test_missing_component_not_zero_and_missing_evidence_fail_closed(self):
        text='finish report_power\n(Watts)\nTotal NA .200 .001 .300 100%\n'
        self.assertEqual(m.power_components(text)['component_status'],'not_available_in_saved_text')
        run=self.fixture();(run/'orfs_work/results/nangate45/bp_fe/eval/6_1_fill.sdc').unlink()
        with self.assertRaises(FileNotFoundError): m.native_power(run,m.Capture())

    def test_immutable_checked_report_binds_completion_and_all_evidence(self):
        folder=self.root/'audit';folder.mkdir()
        evidence=self.write(self.root/'evidence','saved')
        sources={name:m.sha(m.ROOT/name) for name in m.SOURCE_NAMES}
        for name in sources:
            path=folder/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((m.ROOT/name).read_bytes())
        plan={'path':'plan','sha256':'p'*64};collector_report={'path':'collector','sha256':'c'*64}
        value=dict(schema='qstudy_final_power_audit_v1',status='verified',admission_eligible=True,plan=plan,
            collector_report=collector_report,protocol=dict(path=str(m.PROTOCOL),sha256=m.PROTOCOL_SHA),
            planned=64,verified=64,errors=[],rows=[dict(index=i,status='verified') for i in range(64)],sources=sources,
            evidence_sha256={str(evidence):m.sha(evidence)},model_calls=0,backend_calls=0,label_mutations=0)
        report=self.write(folder/'report.json',value);digest=m.sha(report)
        done=dict(status='completed',admission_eligible=True,report_sha256=digest,sources=sources,model_calls=0,backend_calls=0)
        self.write(folder/'completed.json',done)
        checked=m.checked_report(report,digest,plan,collector_report)
        self.assertEqual(checked['audit_binding']['sha256'],digest)
        with self.assertRaisesRegex(ValueError,'mismatched'):
            m.checked_report(report,digest,dict(plan,sha256='wrong'),collector_report)
        evidence.write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            m.checked_report(report,digest,plan,collector_report)


if __name__=='__main__': unittest.main()
