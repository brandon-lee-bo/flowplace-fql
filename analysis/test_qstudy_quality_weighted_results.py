"""Saved-only fixtures for matched FM native accounting and operational gates."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import qstudy_quality_weighted_results as m


class ProducerClosureTests(unittest.TestCase):
    """Exercise the real frozen producer validator on metadata-only fixtures."""
    def setUp(self):
        from fql import qstudy_quality_weighted_eval as exporter
        self.producer = exporter
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/"source"
        self.experiment = Path(self.tmp.name)/"experiment"
        self.parent = Path(self.tmp.name)/"parent"
        for folder in (self.root,self.experiment,self.parent):
            folder.mkdir()
        self.files = {}
        for name in (*exporter.OWN_SOURCES,"fql/producer_dependency.py"):
            path = self.root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("# original captured source\n")
            self.files[name] = path
        (self.experiment/"config.json").write_text('{}\n')
        for filename in ("config.json","declaration.json"):
            (self.parent/filename).write_text('{}\n')
        self.cfg = dict(arm="uniform",backend_concurrency=2,run_prefix="fixture",
            training_manifest=dict(path="training_plan",sha256="plan"),
            flow_eval_experiment=str(self.parent))
        self.physical,self.pairs,self.plan,self.trained = {},[{"seed":0,"index":0,"noise_sha256":"noise"}],dict(population={}),{}
        self.patches = (patch.object(exporter,"ROOT",self.root),
                        patch.object(exporter,"source_files",return_value=dict(self.files)))
        for item in self.patches:
            item.start(); self.addCleanup(item.stop)
        exporter.declare(self.experiment,self.cfg,self.physical,self.pairs,self.plan,self.trained)
        self.saved = m.read(self.experiment/"declaration.json")

    def check(self):
        return m.checked_captured_declaration(self.producer,self.experiment,self.cfg,
            self.physical,self.pairs,self.plan,self.trained)

    def test_extra_reader_imports_accepted_without_changing_producer_globals(self):
        extra = self.root/"analysis/new_observer.py"
        extra.parent.mkdir(); extra.write_text('# read-only observer\n')
        self.producer.source_files.return_value = dict(self.files,**{"analysis/new_observer.py":extra})
        originals = (self.producer.source_files,self.producer.declaration_value,
                     self.producer.checked_declaration)
        with self.assertRaisesRegex(ValueError,"source closure"):
            self.producer.checked_declaration(self.experiment,self.cfg,self.physical,
                self.pairs,self.plan,self.trained)
        value,evidence = self.check()
        self.assertEqual(value,self.saved)
        self.assertEqual(evidence["observer_extra_source_sha256"],
                         {"analysis/new_observer.py":m.native.sha(extra)})
        self.assertEqual(originals,(self.producer.source_files,self.producer.declaration_value,
                                    self.producer.checked_declaration))
        self.assertIs(self.producer.declaration_value.__globals__["source_files"],originals[0])

    def test_captured_working_source_or_snapshot_tamper_refused(self):
        name = "fql/producer_dependency.py"
        for path in (self.files[name],self.experiment/"quality_flow_eval_sources"/name):
            with self.subTest(path=str(path)):
                payload = path.read_bytes()
                path.write_text('# altered recorded producer source\n')
                with self.assertRaisesRegex(ValueError,"changed"):
                    self.check()
                path.write_bytes(payload)
        self.assertEqual(self.check()[0],self.saved)

    def test_non_source_declaration_and_dispatch_tamper_still_refused(self):
        path = self.experiment/"declaration.json"
        declaration_bytes=path.read_bytes()
        for field,value in (("flow_steps",20),("selection","best measured output"),
                            ("pairs",[{"seed":2}])):
            with self.subTest(field=field):
                changed=copy.deepcopy(self.saved); changed[field]=value
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,"source closure"):
                    self.check()
                path.write_bytes(declaration_bytes)
        dispatch=self.experiment/"dispatch.json"
        payload=dispatch.read_bytes(); changed=m.read(dispatch); changed["count"]=1
        dispatch.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"dispatch"):
            self.check()
        dispatch.write_bytes(payload)
        self.assertEqual(self.check()[0],self.saved)

    def test_common_source_inventory_is_its_own_exact_snapshot(self):
        other = Path(self.tmp.name)/"common"
        for name,path in self.files.items():
            target=other/"flow_eval_sources"/name
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(path.read_bytes())
        files,evidence=m.captured_sources(self.producer,other,self.saved,"flow_eval_sources")
        self.assertEqual(files,self.files)
        self.assertTrue(evidence["captured_working_sources_and_snapshots_verified"])
        target=other/"flow_eval_sources/fql/producer_dependency.py"
        target.write_text('# changed common snapshot\n')
        with self.assertRaisesRegex(ValueError,"changed"):
            m.captured_sources(self.producer,other,self.saved,"flow_eval_sources")


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_common_checkpoint_realpath_alias_accepts_exact_original_only(self):
        original=self.root/"retained/common.pt"
        original.parent.mkdir();original.write_bytes(b"opaque immutable checkpoint")
        captured=m.artifact(original)
        direct=m.checked_common_initialization(captured,str(original),captured["sha256"])
        self.assertFalse(direct["configured_path_is_alias"])
        alias=self.root/"legacy_common.pt";alias.symlink_to(original)
        value=m.checked_common_initialization(captured,str(alias),captured["sha256"])
        self.assertTrue(value["configured_path_is_alias"])
        self.assertEqual(value["canonical_binding"],captured)
        self.assertEqual(value["configured_literal_path"],str(alias))
        duplicate=self.root/"different_common.pt";duplicate.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(ValueError,"Common reference checkpoint differs"):
            m.checked_common_initialization(captured,str(duplicate),captured["sha256"])
        alias.unlink();alias.symlink_to(duplicate)
        with self.assertRaisesRegex(ValueError,"Common reference checkpoint differs"):
            m.checked_common_initialization(captured,str(alias),captured["sha256"])

    def test_common_checkpoint_hash_or_captured_identity_change_refused(self):
        original=self.root/"common.pt";original.write_bytes(b"original checkpoint")
        captured=m.artifact(original)
        with self.assertRaisesRegex(ValueError,"Common reference checkpoint differs"):
            m.checked_common_initialization(captured,str(original),"0"*64)
        changed=dict(captured,sha256="0"*64)
        with self.assertRaisesRegex(ValueError,"Common reference checkpoint differs"):
            m.checked_common_initialization(changed,str(original),captured["sha256"])
        original.write_bytes(b"changed checkpoint bytes")
        with self.assertRaisesRegex(ValueError,"Common reference checkpoint differs"):
            m.checked_common_initialization(captured,str(original),captured["sha256"])

    def result(self, arm, seed_delta=(0., 0., 0.)):
        experiment = self.root/arm
        experiment.mkdir(exist_ok=True)
        streams = []
        for seed in m.SEEDS:
            tasks = experiment/f"tasks_{seed}.jsonl"
            tasks.write_text("fixture\n")
            rows = []
            for index in range(8):
                # Seed-level baselines differ; eight generation outputs are
                # deliberately not treated as independent model repetitions.
                wl = 1000.+100.*seed+index+seed_delta[seed]
                values = dict(zip(m.native.METRICS, (wl, -.4, -20., 10., 12., .2, 70000.)))
                rows.append(dict(seed=seed,index=index,noise_sha256="noise"+str(index),
                    action_sha256=f"action_{arm}_{seed}_{index}",raw_action_sha256="raw",
                    state="success",metrics=values,legalizer_jump_rms=.01,
                    request=dict(path="request",sha256="requestsha")))
            streams.append(m.stream(arm,seed,rows,tasks))
        return dict(arm=arm,experiment=str(experiment),
            physical_context=dict(config_sha256=arm,dataset_sha256="data",revision="physical"),
            training_manifest=dict(path=str(self.root/"training/plan.json"),sha256="plan"),
            all_paired_training={"paired_draw_chain":"fixed"},source_pairs=[{"actual":"fixed"}],
            streams=streams)

    def replace_row(self,result,seed,index,**changes):
        stream = result["streams"][seed]
        rows = copy.deepcopy(stream["rows"])
        rows[index].update(changes)
        result["streams"][seed] = m.stream(result["arm"],seed,rows,Path(stream["tasks"]["path"]))

    def test_three_seed_sd_not_twentyfour_independent_repeats(self):
        result = m.aggregate(self.result("uniform"))
        stats = result["metrics"]["rWL_um"]
        self.assertEqual(stats["n"],3)
        self.assertAlmostEqual(stats["mean"],1103.5)
        self.assertAlmostEqual(stats["sd"],100.)
        self.assertEqual(result["request_accounting"]["planned"],24)

    def test_same_noise_pairs_then_three_seed_sample_sd(self):
        uniform=self.result("uniform")
        weighted=self.result("quality_rank",(-30.,-20.,-10.))
        paired=m.paired(weighted,uniform)
        self.assertEqual(len(paired["pairs"]),24)
        self.assertEqual(paired["both_success"],24)
        self.assertEqual(paired["three_seed_paired_deltas"]["rWL_um"],dict(n=3,mean=-20.,sd=10.))
        self.assertTrue(all(r["metrics"]["rWL_um"]["n"]==8 for r in paired["seed_means"]))

    def test_missing_keeps_denominator_and_withholds_full_policy_gate(self):
        uniform=self.result("uniform")
        weighted=self.result("quality_rank",(-30.,-30.,-30.))
        self.replace_row(weighted,1,7,state="missing",metrics=None)
        summary=m.aggregate(weighted)
        self.assertFalse(summary["aggregate_eligible"])
        self.assertEqual(summary["request_accounting"]["missing"],1)
        self.assertEqual(summary["request_accounting"]["planned"],24)
        paired=m.paired(weighted,uniform)
        self.assertEqual(paired["both_success"],23)
        self.assertIsNone(paired["three_seed_paired_deltas"]["rWL_um"]["mean"])
        gates=m.protocol_gates(weighted,uniform)
        self.assertEqual(gates["status"],"inconclusive")
        self.assertFalse(gates["full48_success_numeric_gates"])

    def test_physical_failures_are_not_fake_numeric_labels(self):
        uniform=self.result("uniform")
        weighted=self.result("quality_rank",(-30.,-30.,-30.))
        self.replace_row(weighted,0,0,state="physical_failure",metrics=None)
        aggregate=m.aggregate(weighted)
        self.assertTrue(aggregate["aggregate_eligible"])
        self.assertTrue(aggregate["conditional_PPA"])
        self.assertEqual(aggregate["request_accounting"]["physical_failure"],1)
        paired=m.paired(weighted,uniform)
        self.assertIsNone(paired["pairs"][0]["deltas"])
        self.assertTrue(paired["success_conditional"])
        gates=m.protocol_gates(weighted,uniform)
        self.assertEqual(gates["status"],"fails_declared_development_gate")
        self.assertIsNone(gates["gates"]["primary_WL"])

    def test_zero_success_seed_cannot_be_dropped_from_aggregate(self):
        weighted=self.result("quality_rank")
        for index in range(8):
            self.replace_row(weighted,2,index,state="geometry_failure",metrics=None)
        aggregate=m.aggregate(weighted)
        self.assertTrue(aggregate["aggregate_eligible"])
        self.assertEqual(aggregate["metrics"]["rWL_um"]["n"],2)
        self.assertIsNone(aggregate["metrics"]["rWL_um"]["mean"])
        self.assertEqual(aggregate["request_accounting"]["geometry_failure"],8)

    def test_actual_noise_and_training_matrix_mismatch_refused(self):
        uniform=self.result("uniform");weighted=self.result("quality_rank")
        m.matched_context([uniform,weighted])
        weighted["all_paired_training"]["paired_draw_chain"]="changed"
        with self.assertRaisesRegex(ValueError,"training matrix"):
            m.matched_context([uniform,weighted])
        weighted["all_paired_training"] = copy.deepcopy(uniform["all_paired_training"])
        self.replace_row(weighted,0,0,noise_sha256="different")
        with self.assertRaisesRegex(ValueError,"actual noise"):
            m.matched_context([uniform,weighted])
        with self.assertRaisesRegex(ValueError,"actual noise"):
            m.paired(weighted,uniform)

    def test_gates_correct_signed_slacks_and_native_units(self):
        uniform=self.result("uniform")
        weighted=self.result("quality_rank",(-30.,-30.,-30.))
        self.assertEqual(m.protocol_gates(weighted,uniform)["status"],"passes_declared_development_gate")
        for seed in m.SEEDS:
            for index in range(8):
                metrics=copy.deepcopy(weighted["streams"][seed]["rows"][index]["metrics"])
                metrics.update(WNS_ns=-.43,TNS_ns=-22.)
                self.replace_row(weighted,seed,index,metrics=metrics)
        gates=m.protocol_gates(weighted,uniform)
        self.assertFalse(gates["gates"]["WNS"]["passed"])
        self.assertAlmostEqual(gates["gates"]["WNS"]["deterioration_ns"],.03)
        self.assertFalse(gates["gates"]["TNS"]["passed"])
        self.assertEqual(gates["gates"]["TNS"]["tolerance_ns"],1.)

    def test_mean_wl_gain_alone_does_not_pass_seed_or_secondary_gate(self):
        uniform=self.result("uniform")
        weighted=self.result("quality_rank",(-100.,-100.,1.))
        gates=m.protocol_gates(weighted,uniform)
        self.assertTrue(gates["gates"]["primary_WL"]["passed"])
        self.assertFalse(gates["gates"]["all_three_seed_WL_direction"]["passed"])
        self.assertEqual(gates["status"],"fails_declared_development_gate")
        weighted=self.result("quality_rank",(-30.,-30.,-30.))
        for seed in m.SEEDS:
            for index in range(8):
                metrics=copy.deepcopy(weighted["streams"][seed]["rows"][index]["metrics"])
                metrics["Power_W"] = .203
                self.replace_row(weighted,seed,index,metrics=metrics)
        self.assertFalse(m.protocol_gates(weighted,uniform)["gates"]["Power_W"]["passed"])

    def test_completion_exact_task_identity_and_marker_fields(self):
        experiment=self.root/"export";experiment.mkdir()
        for name in ("declaration.json","dispatch.json"):
            (experiment/name).write_text("{}")
        tasks={s:experiment/f"task_{s}.jsonl" for s in m.SEEDS}
        for s,path in tasks.items(): path.write_text(str(s))
        physical={"config_sha256":"config"}
        done=dict(status="completed",count=24,
            declaration_sha256=m.native.sha(experiment/"declaration.json"),config_sha256="config",
            dispatch_sha256=m.native.sha(experiment/"dispatch.json"),
            seed_tasks={str(s):m.artifact(tasks[s]) for s in m.SEEDS},
            backend_calls=0,production_native_submitted=False,evaluation_never_replayed=True)
        marker=experiment/"export_completed.json";marker.write_text(json.dumps(done))
        m.checked_completion(experiment,{},physical,tasks)
        tasks[1].write_text("changed actual task identity")
        with self.assertRaisesRegex(ValueError,"completion"):
            m.checked_completion(experiment,{},physical,tasks)
        done["seed_tasks"]={str(s):m.artifact(tasks[s]) for s in m.SEEDS}
        done["evaluation_never_replayed"]=False;marker.write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError,"completion"):
            m.checked_completion(experiment,{},physical,tasks)

    def test_saved_geometry_guard_never_calls_physical_verifier_when_missing(self):
        sample=self.root/"sample";sample.mkdir()
        (sample/"request.json").write_text("{}");(sample/"actions.npz").write_bytes(b"opaque")
        tasks=self.root/"tasks.jsonl";tasks.write_text("{}")
        run=self.root/"native";run.mkdir();(run/"manifest.json").write_text('{"success":true}')
        row=dict(seed=0,round=0,arm="uniform",role="evaluation",index=0,graph="bp_fe",
            training_feedback_eligible=False)
        (sample/"outcome.json").write_text(json.dumps(dict(row,physical_run=str(run))))
        with patch("fql.qstudy_backend.checked_existing") as verify:
            result=m.metric_row(row,sample,tasks,({},None,{}))
        verify.assert_not_called()
        self.assertEqual(result["state"],"invalid")
        self.assertIn("Missing saved native geometry audit",result["integrity_error"])

    def test_nonnegative_metrics_rejected_but_signed_slacks_retained(self):
        values=dict(zip(m.native.METRICS,(10.,-.4,-20.,0.,0.,.2,70000.)))
        row=dict(state="success",metrics=values)
        with patch.object(m.baseline,"inspect_outcome",return_value=copy.deepcopy(row)):
            self.assertEqual(m.metric_row({},None,None,None)["state"],"success")
        values["rWL_um"]=-1.
        with patch.object(m.baseline,"inspect_outcome",return_value=copy.deepcopy(row)):
            result=m.metric_row({},None,None,None)
        self.assertEqual(result["state"],"invalid")
        self.assertIsNone(result["metrics"])

    def test_immutable_snapshot_completion_hashes_and_protected_output(self):
        results=[self.result("uniform"),self.result("quality_rank",(-30.,-30.,-30.))]
        with patch.dict(os.environ,{"SLURM_JOB_ID":"fixture"}):
            snapshot,value=m.write_report(self.root/"reports",results)
            self.assertEqual(value["status"],"complete")
            done=m.read(snapshot/"completed.json")
            for name,digest in done["artifacts"].items():
                self.assertEqual(m.native.sha(snapshot/name),digest)
            latest=m.read(self.root/"reports/latest.json")
            self.assertEqual(latest["report"],m.artifact(snapshot/"report.json"))
            self.assertEqual(m.read(snapshot/"report.json")["new_backend_calls"],0)
            with self.assertRaisesRegex(ValueError,"overlaps"):
                m.write_report(self.root/"training/report",results)
        self.assertEqual(m.read(snapshot/"report.json")["aggregates"][0]["metrics"]["rWL_um"]["n"],3)


if __name__ == "__main__":
    unittest.main()
