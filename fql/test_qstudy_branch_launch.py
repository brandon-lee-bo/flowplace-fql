"""Cheap wait/exec gate tests; no controller jobs, tensors, or physical tools."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import qstudy_branch as branch
from . import qstudy_branch_launch as launcher


class LaunchTests(unittest.TestCase):
    def setUp(self):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run wait/import tests in bounded Slurm CPU allocation")
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.root = Path(self.temporary.name)
        self.source, self.target, self.output = (self.root/name for name in ("source", "target", "controller"))
        self.source.mkdir(); self.target.mkdir()
        self.cfg = dict(arm="frozen", run_prefix="frozenflow", online_flow_update=False,
            seeds=[0, 1, 2], online_samples=2, evaluation_samples=3, online_rounds=2)
        self.target_cfg = dict(self.cfg, arm="updated", run_prefix="updatedflow", online_flow_update=True)
        branch.atomic_json(self.source/"config.json", self.cfg)
        branch.atomic_json(self.target/"config.json", self.target_cfg)
        self.source_config = (self.source/"config.json").read_bytes()
        self.target_config = (self.target/"config.json").read_bytes()
        self.outcomes = []
        for seed in self.cfg["seeds"]:
            folder = self.source/f"seed_{seed}/round_0"
            folder.mkdir(parents=True)
            (folder/"checkpoint.pt").write_bytes(b"Never tensor-loaded by cheap gate")
            branch.atomic_json(folder/"binding.json", {})
            branch.atomic_json(folder/"completed.json", dict(status="completed"))
            rows = []
            for role, count in (("collect", 2), ("evaluation", 3)):
                for index in range(count):
                    sample = folder/"requests"/self.cfg["arm"]/role/str(index)
                    sample.mkdir(parents=True)
                    branch.atomic_json(sample/"outcome.json", dict(synthetic_presence_only=True))
                    self.outcomes.append(sample/"outcome.json")
                    rows.append(dict(role=role, index=index, arm=self.cfg["arm"], seed=seed,
                        round=0, sample_dir=str(sample)))
            (folder/"requests/tasks.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            branch.atomic_json(folder/"requests/completed.json", dict(status="completed", count=5))
        self.project = self.root/"project"
        for name in launcher.LAUNCH_SOURCES:
            path = self.project/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(launcher.ROOT/name, path)
        self.args = SimpleNamespace(source=self.source, target=self.target, controller_output=self.output,
            wait_source=False, poll_seconds=1., max_wait_seconds=None)
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(launcher, "ROOT", self.project))
        self.stack.enter_context(mock.patch.object(launcher.os, "chdir"))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch("subprocess.run", side_effect=AssertionError("No actual jobs or tools")))
        self.stack.enter_context(mock.patch.object(branch.torch, "load", side_effect=AssertionError("No tensor loading by wait gate")))
        self.stack.enter_context(mock.patch.object(branch.backend, "context", side_effect=AssertionError("No native validation by wait gate")))
        self.importer = self.stack.enter_context(mock.patch.object(branch, "import_prefix", side_effect=self.fake_import))
        self.executed = mock.Mock(return_value="exec sentinel")

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def fake_import(self, source, target):
        self.assertEqual((source, target), (self.source, self.target))
        self.assertFalse(launcher.source_presence(source, self.cfg))
        result = dict(status="completed", seeds=self.cfg["seeds"], new_train_updates=0, unique_physical_calls=0,
            all_source_and_imported_native_results_validated=True)
        branch.atomic_json(target/"branch_import.json", result)
        return result

    def run_launch(self, **kwargs):
        return launcher.launch(self.args, execv=self.executed, **kwargs)

    def test_missing_source_without_wait_never_imports_or_executes(self):
        self.outcomes[-1].unlink()
        with self.assertRaisesRegex(RuntimeError, "prefix incomplete"):
            self.run_launch()
        self.importer.assert_not_called()
        self.executed.assert_not_called()
        self.assertFalse((self.output/"branch_controller_handoff.json").exists())

    def test_cheap_wait_all_seeds_then_single_validated_import_before_controller(self):
        missing = self.outcomes[-1]
        missing.unlink(); self.args.wait_source = True
        sleeps = []
        def finish(seconds):
            sleeps.append(seconds)
            branch.atomic_json(missing, dict(synthetic_presence_only=True))
        self.assertEqual(self.run_launch(sleep=finish), "exec sentinel")
        self.assertEqual(sleeps, [1.])
        self.importer.assert_called_once_with(self.source, self.target)
        self.executed.assert_called_once()
        command = self.executed.call_args.args[1]
        self.assertEqual(command[command.index("--experiments")+1], str(self.target))
        self.assertEqual(command[-6:], ["--rounds", "2", "--gpu-concurrency", "2", "--backend-concurrency", "2"])
        self.assertEqual((self.source/"config.json").read_bytes(), self.source_config)
        self.assertEqual((self.target/"config.json").read_bytes(), self.target_config)
        handoff = branch.read(self.output/"branch_controller_handoff.json")
        self.assertTrue(handoff["no_new_round0_backend"])
        self.assertEqual(handoff["import_manifest_sha256"], branch.sha(self.target/"branch_import.json"))

    def test_invalid_or_missing_final_gate_cannot_execute_controller(self):
        def invalid(source, target):
            return dict(status="completed", seeds=self.cfg["seeds"], new_train_updates=0, unique_physical_calls=0,
                all_source_and_imported_native_results_validated=True)
        self.importer.side_effect = invalid
        with self.assertRaisesRegex(ValueError, "final branch import gate"):
            self.run_launch()
        self.executed.assert_not_called()
        self.assertFalse((self.output/"branch_controller_handoff.json").exists())

    def test_restart_after_later_round_exists_reuses_immutable_launch_plan(self):
        self.run_launch()
        plan = (self.output/"branch_launch_plan.json").read_bytes()
        (self.target/"seed_0/round_1").mkdir(parents=True)
        (self.target/"seed_0/round_1/checkpoint.pt").write_bytes(b"Later online prefix must remain untouched")
        self.run_launch()
        self.assertEqual(self.importer.call_count, 2)  # Full importer revalidates, never re-trains.
        self.assertEqual(self.executed.call_count, 2)
        self.assertEqual((self.output/"branch_launch_plan.json").read_bytes(), plan)
        self.assertEqual((self.target/"seed_0/round_1/checkpoint.pt").read_bytes(), b"Later online prefix must remain untouched")

    def test_configuration_change_during_wait_prevents_import(self):
        self.outcomes[-1].unlink(); self.args.wait_source = True
        def tamper(_):
            branch.atomic_json(self.target/"config.json", dict(self.target_cfg, changed=True))
        with self.assertRaisesRegex(ValueError, "configuration changed while waiting"):
            self.run_launch(sleep=tamper)
        self.importer.assert_not_called(); self.executed.assert_not_called()

    def test_foreign_output_and_overlapping_trees_refused(self):
        self.output.mkdir(); (self.output/"foreign.txt").write_text("unchanged")
        with self.assertRaisesRegex(ValueError, "foreign artifacts"):
            self.run_launch()
        self.assertEqual((self.output/"foreign.txt").read_text(), "unchanged")
        nested = copy.copy(self.args); nested.target = self.source/"nested"; nested.controller_output = self.source/"nested_controller"
        with self.assertRaisesRegex(ValueError, "trees must be disjoint"):
            launcher.launch(nested, execv=self.executed)
        self.assertFalse((self.source/"nested_controller").exists())
        self.importer.assert_not_called(); self.executed.assert_not_called()

    def test_wait_timeout_and_poll_bound_never_reaches_import(self):
        self.outcomes[-1].unlink()
        clock = [0.]
        def advance(seconds): clock[0] += seconds
        with self.assertRaises(TimeoutError):
            launcher.wait_source(self.source, self.target, self.cfg, branch.sha(self.source/"config.json"),
                branch.sha(self.target/"config.json"), wait=True, poll_seconds=1., max_wait_seconds=2.,
                sleep=advance, monotonic=lambda: clock[0])
        with self.assertRaises(ValueError):
            launcher.wait_source(self.source, self.target, self.cfg, "", "", wait=True, poll_seconds=61.)
        self.importer.assert_not_called(); self.executed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
