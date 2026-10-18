"""Safety and estimand checks; synthetic evidence only, never a physical run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_final_route as m


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / "code"
        for name in m.SOURCES:
            path = self.sources / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.cfg = dict(design="bp_fe", seeds=[0, 1, 2], evaluation_samples=8,
            online_samples=8, evaluation_purpose="development_validation", backend_threads=8,
            terminal_mask=0, evaluation_noise_seed=4600000, offline_actor_steps=1000, online_updates=600)
        self.exps = {}
        for arm in ("bc", "legacy"):
            folder = self.root / arm
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps(dict(self.cfg, arm=arm,
                run_prefix=arm, q_weight=0 if arm == "bc" else 1)))
            self.exps[arm] = folder
        self.out = self.root / "route"

    def make_plan(self):
        with patch.object(m, "ROOT", self.sources):
            return m.build_plan(self.out, self.exps, "go2o_test")

    def records(self):
        result = []
        for task in self.make_plan()["tasks"]:
            v = 100 + task["seed"] * 10 + task["noise_index"] + (5 if task["arm"] == "legacy" else 0)
            result.append(dict(task, state="completed", noise_sha256="noise" + str(task["noise_index"]),
                metrics={name: v if name != "DRC" else task["noise_index"] for name in m.METRICS}))
        return result

    def test_plan_48_fixed_tasks_without_outcome_reads(self):
        # Source_outcome paths deliberately do not exist. Planning still works.
        first = self.make_plan()
        second = self.make_plan()
        self.assertEqual(first, second)
        self.assertEqual(len(first["tasks"]), 48)
        self.assertEqual(len({row["key"] for row in first["tasks"]}), 48)
        self.assertFalse(first["fresh_confirmation"])
        self.assertEqual({row["noise_index"] for row in first["tasks"]}, {0, 1, 2, 3})
        self.assertTrue(all(not Path(row["source_outcome"]).exists() for row in first["tasks"]))

    def test_config_or_source_change_refused(self):
        args = argparse.Namespace(output=self.out, bc=self.exps["bc"], legacy=self.exps["legacy"],
                                  run_prefix="go2o_test", concurrency=2)
        with patch.object(m, "ROOT", self.sources):
            m.plan_command(args)
            self.assertEqual(m.validate_plan(self.out)["requested"], 48)
            (self.sources / m.SOURCES[0]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                m.validate_plan(self.out)

    def test_unmatched_training_setting_refused(self):
        path = self.exps["legacy"] / "config.json"
        cfg = m.read(path)
        cfg["online_updates"] = 601
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            self.make_plan()

    def test_three_seed_statistics_not_twelve_noise_replicates(self):
        stats = m.aggregate(self.records())
        bc = next(row for row in stats["policy_groups"] if row["arm"] == "bc" and row["round"] == 0)
        self.assertEqual(bc["statistics"]["DRT_WL"], dict(n=3, mean=111.5, sd=10.))
        paired = next(row for row in stats["matched_deltas"] if row["comparison"] == "legacy_minus_bc_r0")
        self.assertEqual(paired["statistics"]["DRT_WL"], dict(n=3, mean=5., sd=0.))
        self.assertEqual(bc["statistics"]["DRC"]["mean"], 1.5)

    def test_missing_or_physical_failure_withholds_full_mean(self):
        rows = self.records()
        target = next(row for row in rows if row["arm"] == "legacy" and row["round"] == 0 and row["seed"] == 1)
        target.update(state="upstream_physical_failure", metrics=None)
        result = m.aggregate(rows)
        group = next(row for row in result["policy_groups"] if row["arm"] == "legacy" and row["round"] == 0)
        self.assertIsNone(group["statistics"])
        self.assertEqual(group["complete_actor_seeds"], 2)
        seed = next(row for row in result["seed_means"] if row["arm"] == "legacy" and row["round"] == 0 and row["seed"] == 1)
        self.assertIsNone(seed["mean"])
        self.assertEqual(seed["planned"], 4)
        self.assertEqual(seed["counts"]["upstream_physical_failure"], 1)
        self.assertIsNotNone(seed["successful_only_mean"])

    def test_noise_mismatch_invalidates_all_available_matched_members(self):
        plan = self.make_plan()
        def fake(plan, task, cache):
            return dict(noise_index=task["noise_index"], noise_sha256=("wrong" if task["arm"] == "legacy" else "right"))
        with patch.object(m, "inspect_source", side_effect=fake):
            records = m.inspect_sources(self.out, plan, seal=True)
        self.assertTrue(all(row["state"] == "invalid" for row in records))
        self.assertFalse((self.out / "source_bindings").exists())

    def test_previously_sealed_source_cannot_change(self):
        plan = self.make_plan()
        def fake(plan, task, cache):
            return dict(noise_index=task["noise_index"], noise_sha256="same", payload=0)
        with patch.object(m, "inspect_source", side_effect=fake):
            self.assertTrue(all(row["state"] == "resolved" for row in m.inspect_sources(self.out, plan, seal=True)))
        def changed(plan, task, cache):
            result = fake(plan, task, cache)
            result["payload"] = 1
            return result
        with patch.object(m, "inspect_source", side_effect=changed):
            records = m.inspect_sources(self.out, plan, seal=True)
        self.assertTrue(all(row["state"] == "invalid" for row in records))

    def test_final_calls_native_auditor_and_rejects_changed_spec(self):
        plan = self.make_plan()
        task = plan["tasks"][0]
        output = Path(task["output"])
        output.mkdir(parents=True)
        binding = dict(status="success", noise_sha256="noise", source_outcome_sha256="sourceSHA",
            native=dict(physical_run=str(self.root / "grt"), physical_manifest_sha256="manifestSHA"))
        spec = dict(source_run=binding["native"]["physical_run"], source_manifest_sha256="manifestSHA",
            source_outcome=task["source_outcome"], source_outcome_sha256="sourceSHA", method=task["method"],
            run_id=task["run_id"], route_end_iteration=20, storage="scratch")
        (output / "spec.json").write_text(json.dumps(spec))
        out = dict(status="completed", spec=spec, spec_sha256=m.sha(output / "spec.json"),
            physical_run=str(self.root / "drt"), physical_manifest_sha256="finalSHA", metrics_sha256="metricSHA",
            route_drc_clean=False, metrics={name: 1 for name in m.METRICS})
        (output / "outcome.json").write_text(json.dumps(out))
        source = dict(state="resolved", binding=binding, binding_sha256="bindingSHA")
        audit = dict(source_run=spec["source_run"], physical_identity=dict(native=True))
        with patch.object(m, "audit_final_outcome", return_value=audit) as validator:
            row = m.inspect_final(self.out, plan, task, source)
            validator.assert_called_once_with(out, design="bp_fe", route_cap=20)
            self.assertEqual(row["state"], "completed")
            self.assertEqual(row["metrics"]["DRC"], 1)
        changed = copy.deepcopy(out)
        changed["spec"]["run_id"] = "other"
        (output / "outcome.json").write_text(json.dumps(changed))
        with patch.object(m, "audit_final_outcome") as validator:
            with self.assertRaisesRegex(ValueError, "exact predeclared"):
                m.inspect_final(self.out, plan, task, source)
            validator.assert_not_called()

    def test_upstream_failure_never_has_synthetic_final_ppa(self):
        task = self.make_plan()["tasks"][0]
        source = dict(state="resolved", binding_sha256="bindingSHA", binding=dict(status="geometry_failure",
            noise_sha256="noise", source_outcome_sha256="outSHA", failure_class="normalized_geometry"))
        row = m.inspect_final(self.out, self.make_plan(), task, source)
        self.assertEqual(row["state"], "upstream_physical_failure")
        self.assertIsNone(row["metrics"])
        self.assertFalse(row["upstream_failure_sidecar_present"])

    def test_missing_saved_native_geometry_cannot_invoke_tool_validator(self):
        plan = self.make_plan()
        task = plan["tasks"][0]
        sample = Path(task["source_outcome"]).parent
        sample.mkdir(parents=True)
        run = self.root / "grt"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps(dict(success=True)))
        row = dict(arm="bc", seed=0, round=0, role="evaluation", index=0, graph="bp_fe",
            checkpoint_sha256="checkpoint", noise_seed=4600000,
            evaluation_purpose="development_validation", training_feedback_eligible=False)
        out = dict(row, physical_run=str(run))
        Path(task["source_outcome"]).write_text(json.dumps(out))
        cfg = dict(evaluation_noise_seed=4600000)
        rc = (dict(checkpoint_sha256="checkpoint"), [row], self.root / "tasks.jsonl", (cfg, None, {}))
        with patch.object(m, "round_context", return_value=rc), patch.object(m, "checked_request", return_value=sample), \
                patch.object(m, "checked_existing") as potentially_launching_validator:
            with self.assertRaisesRegex(ValueError, "Missing saved native geometry"):
                m.inspect_source(plan, task, {})
            potentially_launching_validator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
