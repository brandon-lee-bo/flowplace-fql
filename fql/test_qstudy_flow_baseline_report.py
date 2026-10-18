"""Bounded CPU-only report tests; no model inference or physical tools."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_flow_baseline_report as report


def metrics(wl):
    return dict(zip(report.native.METRICS,[wl,-.3,-50.,2.,3.,.2,70000.]))


def rows(wl=100., failure=None):
    return [dict(seed=0,index=i,state=failure if i==0 and failure else "success",
        metrics=None if i==0 and failure else metrics(wl),noise_sha256="noise"+str(i),
        legalizer_jump_rms=.01) for i in range(8)]


def stream(seed=0,wl=100.,failure=None):
    values=rows(wl,failure)
    return dict(seed=seed,rows=values,integrity_errors=[],
        actual_noise_by_index={str(r["index"]):r["noise_sha256"] for r in values},**report.summarize(values))


class FlowBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.folder=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def artifact(self,name,payload="payload"):
        path=self.folder/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload)
        return report.fingerprint(path)

    def test_current_equal_independent_seed_means_and_sample_SD(self):
        result=report.aggregate([stream(0,90),stream(1,100),stream(2,110)],"current")
        self.assertTrue(result["aggregate_eligible"])
        self.assertEqual(result["independent_training_seed_n"],3)
        self.assertEqual(result["metrics"]["rWL_um"],dict(n=3,mean=100.,sd=10.,eligible=True))
        self.assertEqual(result["request_accounting"]["planned"],24)

    def test_public_noise_SD_is_not_three_training_seed_SD(self):
        s=stream()
        for i,row in enumerate(s["rows"]):
            row["metrics"]["rWL_um"]=100.+i
        s.update(report.summarize(s["rows"]))
        result=report.aggregate([s],"public")
        self.assertTrue(result["aggregate_eligible"])
        self.assertEqual(result["independent_training_seed_n"],0)
        self.assertEqual(result["released_public_checkpoint_n"],1)
        self.assertEqual(result["metrics"]["rWL_um"]["n"],8)
        self.assertEqual(result["metrics"]["rWL_um"]["mean"],103.5)
        self.assertAlmostEqual(result["metrics"]["rWL_um"]["sd"],(6.)**.5)
        self.assertIn("ONE fixed",result["uncertainty_unit"])

    def test_missing_is_unresolved_and_withholds_whole_baseline_aggregate(self):
        s=stream(failure="missing")
        result=report.aggregate([s],"public")
        self.assertFalse(result["aggregate_eligible"])
        self.assertEqual(result["request_accounting"]["missing"],1)
        self.assertEqual(result["request_accounting"]["physical_failure"],0)
        self.assertIsNone(result["metrics"])

    def test_confirmed_failure_keeps_denominator_but_has_no_PPA(self):
        s=stream(failure="physical_failure")
        result=report.aggregate([s],"public")
        self.assertTrue(result["aggregate_eligible"])
        self.assertTrue(result["conditional_PPA"])
        self.assertEqual(result["request_accounting"]["planned"],8)
        self.assertEqual(result["metrics"]["rWL_um"]["n"],7)

    def test_zero_success_seed_does_not_silently_drop_independent_seed(self):
        s=stream()
        for row in s["rows"]:
            row.update(state="geometry_failure",metrics=None)
        s.update(report.summarize(s["rows"]))
        result=report.aggregate([s,stream(1),stream(2)],"current")
        self.assertTrue(result["aggregate_eligible"])
        self.assertEqual(result["metrics"]["rWL_um"],dict(n=2,mean=None,sd=None,eligible=False))

    def test_differing_actual_noise_mapping_blocks_seed_aggregation(self):
        s=stream(2)
        s["actual_noise_by_index"]["0"]="other"
        result=report.aggregate([stream(0),stream(1),s],"current")
        self.assertFalse(result["aggregate_eligible"])

    def test_forged_actual_array_audit_rejected(self):
        path=self.folder/"audit.json"
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,"actual-array"):
            report.checked_audit(path)

    def test_changed_actual_array_hash_invalidates_audited_request(self):
        evidence=dict(seed=0,index=0,noise_sha256="noise",raw_action_sha256="raw",action_sha256="action",
            request=self.artifact("sample/request.json"),actions=self.artifact("sample/actions.npz"),
            fixed_raw_and_executed_preserved=True)
        row=dict(seed=0,index=0,sample_dir=str(self.folder/"sample"),noise_sha256="noise",
            raw_action_sha256="raw",action_sha256="action")
        report.verify_request_audit(row,evidence)
        (self.folder/"sample/actions.npz").write_text("tamper")
        with self.assertRaisesRegex(ValueError,"frozen baseline artifact"):
            report.verify_request_audit(row,evidence)

    def native_request(self):
        sample=self.folder/"sample"
        sample.mkdir()
        self.artifact("sample/request.json")
        self.artifact("sample/actions.npz")
        row=dict(seed=0,round=0,arm="behavior_flow",role="evaluation",index=0,graph="bp_fe",
            training_feedback_eligible=False,sample_dir=str(sample),noise_sha256="noise")
        return sample,row

    def test_missing_geometry_audit_never_invokes_native_checker_or_EDA(self):
        sample,row=self.native_request()
        run=self.folder/"native"
        run.mkdir()
        (run/"manifest.json").write_text('{"success":true}')
        (sample/"outcome.json").write_text(json.dumps(dict(row,physical_run=str(run),status="success")))
        with patch("fql.qstudy_backend.checked_existing") as checker:
            result=report.inspect_outcome(row,sample,self.folder/"tasks",({},None,{}))
        checker.assert_not_called()
        self.assertEqual(result["state"],"invalid")
        self.assertIn("must not run OpenROAD",result["integrity_error"])

    def test_validated_native_success_preserves_all_seven_units(self):
        sample,row=self.native_request()
        out=dict(row,status="success",metrics=metrics(110.),stage="global_route_only")
        (sample/"outcome.json").write_text(json.dumps(out))
        with patch("fql.qstudy_backend.checked_existing") as checker, \
             patch.object(report.native,"native_fingerprints",return_value={"native_sha":"verified"}):
            result=report.inspect_outcome(row,sample,self.folder/"tasks",({},None,{}))
        checker.assert_called_once()
        self.assertEqual(result["state"],"success")
        self.assertEqual(result["metrics"],metrics(110.))
        self.assertIsNone(result["detailed_route_DRC"])

    def test_wrong_outcome_role_is_invalid_without_invoking_checker(self):
        sample,row=self.native_request()
        out=dict(row,status="success",metrics=metrics(110.),stage="global_route_only",role="collect")
        (sample/"outcome.json").write_text(json.dumps(out))
        with patch("fql.qstudy_backend.checked_existing") as checker:
            result=report.inspect_outcome(row,sample,self.folder/"tasks",({},None,{}))
        checker.assert_not_called()
        self.assertEqual(result["state"],"invalid")

    def test_infrastructure_error_never_becomes_physical_failure_PPA(self):
        sample,row=self.native_request()
        (sample/"outcome.json").write_text(json.dumps(dict(row,status="infrastructure_error")))
        with patch("fql.qstudy_backend.checked_existing",side_effect=ValueError("bad native evidence")):
            result=report.inspect_outcome(row,sample,self.folder/"tasks",({},None,{}))
        self.assertEqual(result["state"],"invalid")
        self.assertIsNone(result["metrics"])

    def test_fresh_process_backend_then_flow_imports_do_not_shadow_analysis(self):
        source=("from fql import qstudy_flow_baseline_report; "
                "from fql import qstudy_backend; from fql import qstudy_flow_eval; "
                "from fql import qstudy_public_flow_eval; from analysis.placement_metrics import legality; "
                "print('fresh-import-ok')")
        result=subprocess.run([str(report.ROOT/".venv/bin/python"),"-c",source],cwd=report.ROOT,
            text=True,capture_output=True,check=True,timeout=90)
        self.assertIn("fresh-import-ok",result.stdout)


if __name__=="__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run bounded CPU report tests under Slurm")
    unittest.main()
