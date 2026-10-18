"""Saved JSON fixtures only; run in a bounded CPU Slurm allocation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from . import qstudy_fresh_acceptance as acceptance


ROOT = Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+"\n")
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class FreshAcceptanceTests(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Run fixture reductions in bounded CPU Slurm")
        self.temp=tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.root=Path(self.temp.name)
        self.rules_path=ROOT/"fql/qstudy_confirmation_rules_v1.json"
        self.rules,self.rb=acceptance.capture(self.rules_path,acceptance.RULES_SHA256)
        self.streams=[dict(id="candidate",purpose="candidate",kind="one_step",seeds=[0,1,2]),
                      dict(id="bc",purpose="bc",kind="one_step",seeds=[0,1,2]),
                      dict(id="flow",purpose="behavior_flow",kind="behavior_flow",seeds=[0,1,2]),
                      dict(id="public",purpose="public_reference",kind="public_official20",seeds=[0])]
        evaluation=dict(noise_seed=85000000,samples=8,drt_indices=[0,1,2,3],drt_repair_cap=20)
        self.lock=dict(status="selection_locked",streams=self.streams,development_inventory=[self.rb],
                       design="bp_fe",physical_flow_revision="frozen_physical_revision",evaluation=evaluation,
                       output_root=str(self.root/"frozen_exports"))
        self.sb=write(self.root/"selection.json",self.lock)
        self.declaration=dict(selection_lock=self.sb,role="evaluation",terminal_mask=0,
                              evaluation_purpose="fresh_confirmation",training_feedback_eligible=False,
                              offline_train_eligible=False,streams=self.streams,evaluation=evaluation)
        self.db=write(self.root/"declaration.json",self.declaration)
        noise_path=self.root/"noise.npz";noise_path.write_bytes(b"saved-noise-fixture-not-generated")
        self.noise=dict(selection_lock_sha256=self.sb["sha256"],noise_seed=85000000,generator_device="cuda",
                        dtype="float32",no_training_or_replay=True,
                        noise_sha256=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)],
                        artifact=dict(path=str(noise_path),sha256=hashlib.sha256(noise_path.read_bytes()).hexdigest()))
        self.plan=dict(selection=self.sb,declaration=self.db,noise=self.noise,requested=40,indices=[0,1,2,3],
                       route_cap=20,route_settings="unchanged graph_o2o_full_route cap20",design="bp_fe",
                       output=str(self.root/"frozen_DRT"))
        self.pb=write(self.root/"plan.json",self.plan)
        self.grt=dict(version="fresh_native_report_v1",selection=self.sb,declaration=self.db,noise=self.noise,
                      design="bp_fe",native_revision="frozen_physical_revision",training_feedback_eligible=False,
                      offline_train_eligible=False,expected_requests=80,streams=[])
        self.drt=dict(version="fresh_detailed_route_v1",selection=self.sb,declaration=self.db,plan=self.pb,
                      requested=40,indices=[0,1,2,3],route_cap=20,backend_threads=8,
                      route_settings=self.plan["route_settings"],feedback_eligible=False,integrity_errors=[],records=[])
        for stream in self.streams:
            for seed in stream["seeds"]:
                rows=[]
                for i in range(8):
                    wl=98. if stream["id"]=="candidate" else 100.
                    metrics=dict(zip(acceptance.GRT_METRICS,[wl,-.2,-10.,10.,20.,.2,1000.]))
                    rows.append(dict(index=i,state="success",metrics=metrics,noise_sha256=self.noise["noise_sha256"][i]))
                    if i<4:
                        self.drt["records"].append(dict(stream=stream["id"],seed=seed,index=i,state="completed",
                            metrics=dict(zip(acceptance.DRT_METRICS,[wl,-.2,-10.,.2,1000.,0])),
                            noise_sha256=self.noise["noise_sha256"][i]))
                self.grt["streams"].append(dict(stream=stream["id"],purpose=stream["purpose"],kind=stream["kind"],
                                               seed=seed,planned=8,rows=rows))

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self):
        g=write(self.root/"grt.json",self.grt);d=write(self.root/"drt.json",self.drt)
        return acceptance.collect(g["path"],g["sha256"],d["path"],d["sha256"],self.rules_path,self.rb["sha256"],"candidate","bc")

    def records(self,stage="GRT",stream="candidate"):
        if stage=="GRT":
            return [row for rec in self.grt["streams"] if rec["stream"]==stream for row in rec["rows"]]
        return [r for r in self.drt["records"] if r["stream"]==stream]

    def test_complete_pass_reduces_request_then_independent_seed_and_never_selects(self):
        result=self.run_case()
        self.assertEqual(result["classification"],"overall_improvement")
        self.assertTrue(all(g["passed"] for g in result["gates"]))
        self.assertEqual(result["metrics"]["GRT"]["candidate"]["metrics"]["rWL_um"],dict(n=3,mean=98.,sd=0.))
        self.assertEqual(result["metrics"]["DRT"]["continuous_deltas"]["DRT_WL"]["delta_candidate_minus_BC"],-2.)
        self.assertFalse(result["selection_performed"])
        self.assertEqual(result["backend_calls"],0)

    def test_native_parser_metadata_does_not_hide_required_seven_numeric_metrics(self):
        for row in self.records():
            row["metrics"].update(stage="global_route_only",not_final_table2=True,parser_version="native_v1",
                worst_slack_ns=-.2,congestion_evidence="Final OpenDB markers",congestion_marker_status="/saved/status.json")
        for row in self.records("DRT"):
            row["metrics"].update(GRT_WL=98.,runtime=dict(detail_route=1.2),stage="final_detailed_route")
        self.assertEqual(self.run_case()["classification"],"overall_improvement")
        self.records()[0]["metrics"].pop("WNS_ns")
        self.assertEqual(self.run_case()["classification"],"inconclusive")

    def test_output_is_outside_immutable_export_and_route_roots(self):
        roots=[self.lock["output_root"],self.plan["output"]]
        for root in roots:
            for path in [Path(root),Path(root)/"new_report"]:
                with self.assertRaisesRegex(ValueError,"outside immutable"):
                    acceptance.checked_output_root(path,roots)
        self.assertEqual(acceptance.checked_output_root(self.root/"separate_report",roots),self.root/"separate_report")

    def test_one_worse_training_seed_fails_even_with_lower_complete_mean(self):
        for rec in self.grt["streams"]:
            if rec["stream"]=="candidate":
                for row in rec["rows"]:row["metrics"]["rWL_um"]=90. if rec["seed"]<2 else 101.
        result=self.run_case()
        self.assertEqual(result["classification"],"no_stable_gain")
        gate=next(g for g in result["gates"] if "GRT_primary" in g["name"])
        self.assertGreater(gate["value"]["reduction_fraction"],.01)
        self.assertEqual(gate["value"]["paired_seed_deltas"],[-10.,-10.,1.])

    def test_primary_failure_precedes_secondary_and_drt_failure(self):
        for row in self.records():row["metrics"]["rWL_um"]=99.5;row["metrics"]["WNS_ns"]=-1.
        for row in self.records("DRT"):row["metrics"]["DRC"]=2
        self.assertEqual(self.run_case()["classification"],"no_stable_gain")

    def test_secondary_worse_is_tradeoff_and_signed_slacks_preserved(self):
        for row in self.records():row["metrics"]["WNS_ns"]=-.23
        result=self.run_case()
        self.assertEqual(result["classification"],"tradeoff")
        gate=next(g for g in result["gates"] if g["name"]=="GRT_WNS_ns")
        self.assertFalse(gate["passed"]);self.assertAlmostEqual(gate["value"]["delta_candidate_minus_BC"],-.03)

    def test_drt_worse_or_nonzero_drc_is_tradeoff_not_physical_failure(self):
        for row in self.records("DRT"):row["metrics"]["DRT_WL"]=101.
        result=self.run_case();self.assertEqual(result["classification"],"tradeoff")
        self.assertTrue(next(g for g in result["gates"] if g["name"]=="candidate_all24_GRT_and12_DRT_success")["passed"])
        for row in self.records("DRT"):row["metrics"]["DRT_WL"]=98.;row["metrics"]["DRC"]=1
        self.assertEqual(self.run_case()["classification"],"tradeoff")

    def test_unexported_or_unresolved_reference_withholds_entire_confirmation(self):
        rec=next(r for r in self.grt["streams"] if r["stream"]=="flow")
        rec["rows"].pop()
        result=self.run_case()
        self.assertEqual(result["classification"],"inconclusive")
        self.assertFalse(next(g for g in result["gates"] if g["name"]=="all_required_requests_resolved")["passed"])
        self.assertFalse(result["integrity_errors"])

    def test_confirmed_public_failure_is_explicit_without_claiming_reference_outperformance(self):
        self.records(stream="public")[0].update(state="geometry_failure",metrics=None)
        self.records("DRT","public")[0].update(state="upstream_physical_failure",metrics=None)
        result=self.run_case()
        self.assertEqual(result["classification"],"overall_improvement")
        public=next(r for r in result["request_accounting"]["GRT"] if r["stream"]=="public")
        self.assertEqual(public["states"],dict(geometry_failure=1,success=7))
        self.assertNotIn("reference_outperformance",result)

    def test_confirmed_candidate_failure_has_no_fake_ppa_or_primary_success(self):
        self.records()[0].update(state="geometry_failure",metrics=None)
        self.records("DRT")[0].update(state="upstream_physical_failure",metrics=None)
        result=self.run_case()
        self.assertEqual(result["classification"],"no_stable_gain")
        self.assertEqual(result["metrics"]["GRT"]["candidate"]["per_seed"][0]["success"],7)
        self.assertTrue(result["metrics"]["GRT"]["candidate"]["per_seed"][0]["PPA_is_success_conditional"])

    def test_confirmed_bc_failure_is_unassessable_not_conditional_wl_pass(self):
        self.records(stream="bc")[0].update(state="physical_failure",metrics=None)
        self.records("DRT","bc")[0].update(state="upstream_physical_failure",metrics=None)
        result=self.run_case()
        self.assertEqual(result["classification"],"inconclusive")
        self.assertIn("BC physical failure",result["reason"])

    def test_same_actual_noise_and_selection_declaration_identities_required(self):
        original=copy.deepcopy(self.drt)
        self.drt["records"][0]["noise_sha256"]="bad"
        result=self.run_case();self.assertEqual(result["classification"],"inconclusive")
        self.assertTrue(result["integrity_errors"])
        self.drt=original;self.drt["selection"]=dict(self.sb,sha256="a"*64)
        self.assertEqual(self.run_case()["classification"],"inconclusive")

    def test_duplicate_or_wrong_budget_or_fake_failed_metric_refused(self):
        original=copy.deepcopy(self.drt)
        self.drt["records"][1]=copy.deepcopy(self.drt["records"][0])
        self.assertEqual(self.run_case()["classification"],"inconclusive")
        self.drt=original;self.drt["route_cap"]=21
        self.assertEqual(self.run_case()["classification"],"inconclusive")
        self.drt=copy.deepcopy(original);self.records()[0]["state"]="physical_failure"
        self.assertEqual(self.run_case()["classification"],"inconclusive")

    def test_threshold_edges_are_inclusive_but_seed_improvement_strict(self):
        for row in self.records():
            row["metrics"].update(rWL_um=99.,WNS_ns=-.2-.02,TNS_ns=-11.,horizontal_overflow_sum=11.,
                                  vertical_overflow_sum=21.,Power_W=.2*1.01,StdCellArea_um2=1010.)
        for row in self.records("DRT"):
            row["metrics"].update(DRT_WL=100.,WNS=-.2-.02,TNS=-11.,Power=.2*1.01,StdCellArea=1010.)
        result=self.run_case();self.assertEqual(result["classification"],"overall_improvement")
        for row in self.records():row["metrics"]["rWL_um"]=99.000000001
        self.assertEqual(self.run_case()["classification"],"no_stable_gain")

    def test_pinned_file_bytes_rules_and_saved_noise_tampering_rejected(self):
        g=write(self.root/"grt.json",self.grt);d=write(self.root/"drt.json",self.drt)
        (self.root/"grt.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            acceptance.collect(g["path"],g["sha256"],d["path"],d["sha256"],self.rules_path,self.rb["sha256"],"candidate","bc")
        bad_rules=self.root/"bad_rules.json";altered=copy.deepcopy(self.rules);altered["primary_endpoint"]["minimum_mean_reduction_fraction_vs_matched_bc"]=0.
        br=write(bad_rules,altered)
        with self.assertRaisesRegex(ValueError,"exact locked"):
            acceptance.checked_rules(altered,br)
        Path(self.noise["artifact"]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"noise artifact SHA"):
            self.run_case()


if __name__ == "__main__":
    unittest.main()
