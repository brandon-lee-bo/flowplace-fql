"""Controller safety tests with a fully simulated scheduler and workers.

No test submits or changes a real experiment job. The bounded Slurm allocation
running this test suite is the only scheduler action outside the mocks.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import qstudy_controller as controller
from . import qstudy_backend as backend


class SimulatedPreemption(RuntimeError):
    pass


class FakeScheduler:
    """Tiny scheduler: writes fake request/outcome files, never real PPA labels."""

    def __init__(self, root, *, interrupt_submission=False, preempt_actor=False,
                 preempt_backend=False):
        self.root = root
        self.jobs = {}
        self.submissions = []
        self.ticks = 0
        self.next_id = 80000
        self.max_gpu = 0
        self.interrupt_submission = interrupt_submission
        self.preempt_actor = preempt_actor
        self.preempt_backend = preempt_backend
        self.interrupted = False
        self.actor_preempted = False
        self.backend_preempted = False
        self.backend_indices = []
        self.checked = []
        self.forbidden = root / "historical_table2_hold.json"
        self.forbidden.write_text('{"job_id":"1777777","state":"held"}\n')
        self.forbidden_bytes = self.forbidden.read_bytes()

    def submit(self, arguments, unique_name):
        # Simulate exact-name adoption after a successful, unjournaled submit.
        existing = [j for j, v in self.jobs.items() if v["name"] == unique_name]
        if existing:
            if len(existing) != 1:
                raise AssertionError("ambiguous stage identity")
            return existing[0]
        self.next_id += 1
        job_id = str(self.next_id)
        is_backend = any(str(a).startswith("--array=") for a in arguments)
        entry = dict(name=unique_name, arguments=list(arguments), status=["RUNNING"],
                     kind="backend" if is_backend else "actor")
        if is_backend:
            array = next(a.removeprefix("--array=") for a in arguments if a.startswith("--array="))
            indices, throttle = array.split("%")
            entry["indices"] = [int(v) for v in indices.split(",")]
            entry["throttle"] = int(throttle)
            self.backend_indices.append(entry["indices"])
            self.assert_backend_missing(entry)
        else:
            self.assert_prior_round_done(entry)
        self.jobs[job_id] = entry
        self.submissions.append((job_id, entry))
        live_gpu = sum(v["kind"] == "actor" and v["status"] == ["RUNNING"]
                       for v in self.jobs.values())
        self.max_gpu = max(self.max_gpu, live_gpu)
        if self.interrupt_submission and not self.interrupted:
            self.interrupted = True
            raise SimulatedPreemption("controller stopped after sbatch but before journal")
        return job_id

    @staticmethod
    def option(entry, flag):
        args = entry["arguments"]
        return args[args.index(flag) + 1]

    def assert_prior_round_done(self, entry):
        r = int(self.option(entry, "--round"))
        if r:
            experiment = Path(self.option(entry, "--experiment"))
            seed = int(self.option(entry, "--seed"))
            rows, _ = self.requests(experiment, seed, r - 1)
            if not all((Path(row["sample_dir"]) / "outcome.json").exists() for row in rows):
                raise AssertionError("online actor launched before its previous feedback existed")

    def matrix(self, entry):
        path = Path(self.option(entry, "--dispatch"))
        return [json.loads(line) for line in path.read_text().splitlines()]

    def assert_backend_missing(self, entry):
        rows = self.matrix(entry)
        if entry["throttle"] != 4:
            raise AssertionError("backend array must retain declared throttle four")
        for index in entry["indices"]:
            if (Path(rows[index]["sample_dir"]) / "outcome.json").exists():
                raise AssertionError("completed backend measurement resubmitted")

    def state(self, job):
        if str(job) not in self.jobs:
            raise AssertionError("queried a foreign or historical job: " + str(job))
        return self.jobs[str(job)]["status"]

    def sleep(self, seconds):
        self.ticks += 1
        if self.ticks > 100:
            raise AssertionError("controller did not converge in simulated time")
        for _, entry in list(self.jobs.items()):
            if entry["status"] != ["RUNNING"]:
                continue
            if entry["kind"] == "actor":
                if self.preempt_actor and not self.actor_preempted:
                    self.actor_preempted = True
                    entry["status"] = ["PREEMPTED"]
                    continue
                experiment = Path(self.option(entry, "--experiment"))
                seed = int(self.option(entry, "--seed"))
                round_number = int(self.option(entry, "--round"))
                self.write_requests(experiment, seed, round_number)
                entry["status"] = ["COMPLETED"]
            else:
                rows = self.matrix(entry)
                indices = entry["indices"]
                if self.preempt_backend and not self.backend_preempted:
                    self.backend_preempted = True
                    complete = indices[:entry["throttle"]]
                    entry["status"] = ["COMPLETED", "PREEMPTED"]
                else:
                    complete = indices
                    entry["status"] = ["COMPLETED"]
                for index in complete:
                    task = rows[index]
                    requests, _ = self.requests(task["experiment"], task["seed"], task["round"])
                    request = requests[task["index"]]
                    (Path(task["sample_dir"]) / "outcome.json").write_text(
                        json.dumps(dict(request, simulated_test_only=True)))

    def write_requests(self, experiment, seed, round_number):
        directory = experiment / f"seed_{seed}" / f"round_{round_number}" / "requests"
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(2):
            sample = directory / f"sample_{i}"
            sample.mkdir(exist_ok=True)
            rows.append(dict(sample_dir=str(sample), run_id=f"fake_{seed}_{round_number}_{i}",
                seed=seed, round=round_number, arm="fake_actor", role="collect" if i == 0 else "evaluation",
                index=i, graph="test", training_feedback_eligible=i == 0))
        (directory / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (directory / "completed.json").write_text(json.dumps(dict(count=len(rows))))

    @staticmethod
    def requests(experiment, seed, round_number):
        path = Path(experiment) / f"seed_{seed}" / f"round_{round_number}" / "requests/tasks.jsonl"
        return [json.loads(v) for v in path.read_text().splitlines()], path

    def checked_request(self, row, args, cfg, binding):
        return Path(row["sample_dir"])

    def checked_existing(self, out, row, sample, tasks, binding, cfg, verification):
        if not out.get("simulated_test_only"):
            raise ValueError("Corrupt fake outcome")
        self.checked.append(str(sample))


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run controller tests in their bounded Slurm allocation")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qstudy-controller-test-",
            dir=os.environ.get("TMPDIR", "/scratch/bohanli/flowplace_graph"))
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, *, rounds=1):
        plans = []
        for i in range(2):
            path = self.root / f"anchor_{i}"
            path.mkdir(exist_ok=True)
            (path / "config.json").write_text(json.dumps(dict(seeds=[1, 2, 3], design="test",
                                                            online_rounds=rounds)))
            plans.append(str(path))
        return SimpleNamespace(experiments=plans, output=self.root / "controller", rounds=rounds,
                               gpu_concurrency=2, backend_concurrency=4)

    @contextlib.contextmanager
    def simulated(self, scheduler):
        # Any subprocess accidentally escaping the scheduler mocks is a failure,
        # so the tests cannot submit, cancel, release or inspect real jobs.
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(controller, "sbatch", side_effect=scheduler.submit))
            stack.enter_context(mock.patch.object(controller, "state_of", side_effect=scheduler.state))
            stack.enter_context(mock.patch.object(controller, "load_requests", side_effect=scheduler.requests))
            stack.enter_context(mock.patch.object(controller.time, "sleep", side_effect=scheduler.sleep))
            stack.enter_context(mock.patch.object(backend, "context", return_value=({}, None, {})))
            stack.enter_context(mock.patch.object(backend, "checked_request", side_effect=scheduler.checked_request))
            stack.enter_context(mock.patch.object(backend, "checked_existing", side_effect=scheduler.checked_existing))
            stack.enter_context(mock.patch.object(subprocess, "run", side_effect=AssertionError("real scheduler access forbidden")))
            yield

    def test_six_streams_bound_concurrency_and_preemption_only_missing_rows(self):
        args = self.arguments()
        scheduler = FakeScheduler(self.root, preempt_actor=True, preempt_backend=True)
        with self.simulated(scheduler):
            controller.controller(args)
        self.assertEqual(scheduler.max_gpu, 2)
        self.assertTrue(scheduler.actor_preempted)
        self.assertTrue(scheduler.backend_preempted)
        self.assertEqual(scheduler.backend_indices[0], list(range(12)))
        self.assertEqual(scheduler.backend_indices[1], list(range(4, 12)))
        self.assertEqual(scheduler.backend_indices[2], list(range(12)))
        self.assertEqual(sum(e["kind"] == "actor" for _, e in scheduler.submissions), 13)
        state = json.loads((args.output / "state.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["gpu_jobs"]), 12)
        self.assertEqual(len(scheduler.checked), 24)
        self.assertEqual(scheduler.forbidden.read_bytes(), scheduler.forbidden_bytes)
        self.assertTrue(all(e["name"].startswith("qs_") for _, e in scheduler.submissions))

    def test_unjournaled_submission_adopted_and_completed_rerun_is_noop(self):
        args = self.arguments(rounds=0)
        scheduler = FakeScheduler(self.root, interrupt_submission=True)
        with self.simulated(scheduler):
            with self.assertRaises(SimulatedPreemption):
                controller.controller(args)
            state = json.loads((args.output / "state.json").read_text())
            self.assertTrue(all(not value for value in state["gpu_jobs"].values()))
            self.assertEqual(len(scheduler.submissions), 1)
            controller.controller(args)
            self.assertEqual(sum(e["kind"] == "actor" for _, e in scheduler.submissions), 6)
            before = len(scheduler.submissions)
            controller.controller(args)
            self.assertEqual(len(scheduler.submissions), before)
        self.assertEqual(scheduler.forbidden.read_bytes(), scheduler.forbidden_bytes)

    def test_recoverable_waits_for_accounting_and_rejects_ordinary_failures(self):
        for status in (["PENDING"], ["RUNNING"], ["ACCOUNTING_PENDING"],
                       ["COMPLETED", "ACCOUNTING_PENDING"]):
            with self.subTest(status=status), mock.patch.object(controller, "state_of", return_value=status):
                self.assertFalse(controller.recoverable("123"))
        for status in (["PREEMPTED"], ["COMPLETED", "PREEMPTED"], ["NODE_FAIL"], ["TIMEOUT"]):
            with self.subTest(status=status), mock.patch.object(controller, "state_of", return_value=status):
                self.assertTrue(controller.recoverable("123"))
        for status in (["FAILED"], ["OUT_OF_MEMORY"], ["CANCELLED"], ["COMPLETED"], ["UNKNOWN"]):
            with self.subTest(status=status), mock.patch.object(controller, "state_of", return_value=status):
                with self.assertRaisesRegex(RuntimeError, "diagnosis"):
                    controller.recoverable("123")

    def test_sbatch_adopts_only_exact_owned_job_name(self):
        name = "qs_test_anchor_s1_r0_a0"
        calls = []
        def queued(command, **kwargs):
            calls.append(command)
            self.assertEqual(command[0], "squeue")
            return SimpleNamespace(stdout="1777777|historical_table2\n81234_0|"+name+"\n"
                                   "89999|"+name+"_other\n", returncode=0)
        with mock.patch.object(controller.subprocess, "run", side_effect=queued):
            self.assertEqual(controller.sbatch(["fake.slurm"], name), "81234")
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(c[0] not in {"scancel", "scontrol", "srun"} for c in calls))

    def test_sbatch_recovers_accounted_submission_before_submitting_again(self):
        name = "qs_test_anchor_s1_r0_a0"
        replies = [SimpleNamespace(stdout="1777777|historical_table2\n", returncode=0),
                   SimpleNamespace(stdout="81235|"+name+"|\n89999|foreign\n", returncode=0)]
        with mock.patch.object(controller.subprocess, "run", side_effect=replies) as run:
            self.assertEqual(controller.sbatch(["fake.slurm"], name), "81235")
        self.assertEqual([c.args[0][0] for c in run.call_args_list], ["squeue", "sacct"])
        self.assertIn("--name="+name, run.call_args_list[1].args[0])
        self.assertIn("JobIDRaw,JobName%200", run.call_args_list[1].args[0])

    def test_sbatch_rejects_ambiguous_identity_and_never_modifies_jobs(self):
        name = "qs_test_anchor_s1_r0_a0"
        reply = SimpleNamespace(stdout="81234|"+name+"\n81235|"+name+"\n", returncode=0)
        with mock.patch.object(controller.subprocess, "run", return_value=reply) as run:
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                controller.sbatch(["fake.slurm"], name)
        self.assertEqual(len(run.call_args_list), 1)

    def test_sbatch_rejects_nonnumeric_or_refused_submission(self):
        for final in [SimpleNamespace(stdout="warning text", returncode=0),
                      SimpleNamespace(stdout="", stderr="quota denied", returncode=1)]:
            with self.subTest(final=final):
                replies = [SimpleNamespace(stdout="", returncode=0),
                           SimpleNamespace(stdout="", returncode=0), final]
                with mock.patch.object(controller.subprocess, "run", side_effect=replies) as run:
                    with self.assertRaises(RuntimeError):
                        controller.sbatch(["fake.slurm"], "qs_unique_test")
                self.assertEqual([c.args[0][0] for c in run.call_args_list], ["squeue", "sacct", "sbatch"])

    def test_invalid_rounds_concurrency_and_experiment_identities_refused_before_submission(self):
        for field, value in [("rounds", -1), ("gpu_concurrency", 0), ("gpu_concurrency", -2),
                             ("backend_concurrency", 0), ("backend_concurrency", -4)]:
            with self.subTest(field=field, value=value):
                args = self.arguments()
                setattr(args, field, value)
                with mock.patch.object(controller, "sbatch", side_effect=AssertionError("submission forbidden")):
                    with self.assertRaises(ValueError):
                        controller.controller(args)
        for same_basename in [False, True]:
            with self.subTest(same_basename=same_basename):
                args = self.arguments()
                first = Path(args.experiments[0])
                if same_basename:
                    other = self.root / "other_parent" / first.name
                    other.mkdir(parents=True)
                    (other / "config.json").write_bytes((first / "config.json").read_bytes())
                else:
                    other = first
                args.experiments = [str(first), str(other)]
                with mock.patch.object(controller, "sbatch", side_effect=AssertionError("submission forbidden")):
                    with self.assertRaisesRegex(ValueError, "unique"):
                        controller.controller(args)

    def test_invalid_seed_lists_and_actor_round_mismatch_refused_before_submission(self):
        for seeds in [[], [1, 1], [-1], [True], [1.0], ["1"]]:
            with self.subTest(seeds=seeds):
                args = self.arguments()
                first = Path(args.experiments[0]) / "config.json"
                cfg = json.loads(first.read_text())
                first.write_text(json.dumps(dict(cfg, seeds=seeds)))
                with mock.patch.object(controller, "sbatch", side_effect=AssertionError("submission forbidden")):
                    with self.assertRaisesRegex(ValueError, "seeds"):
                        controller.controller(args)
        args = self.arguments()
        args.rounds += 1
        with mock.patch.object(controller, "sbatch", side_effect=AssertionError("submission forbidden")):
            with self.assertRaisesRegex(ValueError, "rounds differ"):
                controller.controller(args)

    def test_atomic_matrix_survives_interrupted_publication_and_refuses_changed_identity(self):
        matrix = self.root / "matrix.jsonl"
        content = '{"sample":"one"}\n{"sample":"two"}\n'
        with mock.patch.object(controller.os, "replace", side_effect=SimulatedPreemption("before rename")):
            with self.assertRaises(SimulatedPreemption):
                controller.atomic_matrix(matrix, content)
        self.assertFalse(matrix.exists())
        temporary = matrix.with_name(matrix.name + ".tmp." + str(os.getpid()))
        self.assertTrue(temporary.exists())
        temporary.write_text('{"partial stale temporary')
        controller.atomic_matrix(matrix, content)
        self.assertEqual(matrix.read_text(), content)
        self.assertFalse(temporary.exists())
        controller.atomic_matrix(matrix, content)
        with self.assertRaisesRegex(ValueError, "identities changed"):
            controller.atomic_matrix(matrix, content + '{"foreign":true}\n')
        self.assertEqual(matrix.read_text(), content)

    def outcome_fixture(self):
        scheduler = FakeScheduler(self.root)
        experiment = self.root / "anchor"
        scheduler.write_requests(experiment, 2, 0)
        rows, original = scheduler.requests(experiment, 2, 0)
        tasks = []
        for i, row in enumerate(rows):
            (Path(row["sample_dir"]) / "outcome.json").write_text(json.dumps(dict(row, simulated_test_only=True)))
            tasks.append(dict(experiment=str(experiment), seed=2, round=0, index=i,
                              sample_dir=row["sample_dir"], tasks_sha256=controller.sha(original)))
        return scheduler, rows, tasks

    def test_round_outcomes_validate_existing_and_report_only_missing(self):
        scheduler, rows, tasks = self.outcome_fixture()
        (Path(rows[1]["sample_dir"]) / "outcome.json").unlink()
        with self.simulated(scheduler):
            self.assertEqual(controller.validate_round_outcomes(tasks), [1])
        self.assertEqual(scheduler.checked, [rows[0]["sample_dir"]])

    def test_round_outcomes_reject_role_and_native_validation_corruption(self):
        scheduler, rows, tasks = self.outcome_fixture()
        path = Path(rows[0]["sample_dir"]) / "outcome.json"
        original = json.loads(path.read_text())
        for field, bad in [("seed", 99), ("round", 99), ("arm", "foreign"),
                           ("role", "evaluation"), ("index", 99), ("graph", "foreign"),
                           ("training_feedback_eligible", False)]:
            with self.subTest(field=field), self.simulated(scheduler):
                path.write_text(json.dumps(dict(original, **{field: bad})))
                with self.assertRaisesRegex(ValueError, "role/identity"):
                    controller.validate_round_outcomes(tasks)
        path.write_text(json.dumps(dict(original, simulated_test_only=False)))
        with self.simulated(scheduler):
            with self.assertRaisesRegex(ValueError, "Corrupt fake outcome"):
                controller.validate_round_outcomes(tasks)
        path.write_text("{truncated")
        with self.simulated(scheduler):
            with self.assertRaises(json.JSONDecodeError):
                controller.validate_round_outcomes(tasks)

    def test_round_outcomes_require_existing_native_geometry_audit_without_launching_it(self):
        scheduler, rows, tasks = self.outcome_fixture()
        path = Path(rows[0]["sample_dir"]) / "outcome.json"
        original = json.loads(path.read_text())
        native_run = self.root / "physical_run"
        native_run.mkdir()
        (native_run / "manifest.json").write_text('{"success":true}')
        path.write_text(json.dumps(dict(original, physical_run=str(native_run))))
        with self.simulated(scheduler):
            with self.assertRaisesRegex(ValueError, "geometry audit missing"):
                controller.validate_round_outcomes(tasks)
            self.assertEqual(scheduler.checked, [])
            (native_run / "backend_macro_audit.json").write_text('{"test_only":true}')
            self.assertEqual(controller.validate_round_outcomes(tasks), [])
        self.assertEqual(scheduler.checked, [r["sample_dir"] for r in rows])

    def test_round_outcomes_reject_dispatch_mismatch_before_native_admission(self):
        scheduler, _, tasks = self.outcome_fixture()
        for update in [dict(index=-1), dict(index=2), dict(tasks_sha256="wrong"), dict(sample_dir="foreign")]:
            with self.subTest(update=update), self.simulated(scheduler):
                changed = [dict(tasks[0], **update)]
                with self.assertRaisesRegex(ValueError, "dispatch identity"):
                    controller.validate_round_outcomes(changed)

    def test_dispatch_binds_matrix_and_never_submits_scheduler_work(self):
        scheduler = FakeScheduler(self.root)
        experiment = self.root / "anchor"
        scheduler.write_requests(experiment, 2, 0)
        rows, original = scheduler.requests(experiment, 2, 0)
        matrix = self.root / "dispatch.jsonl"
        task = dict(experiment=str(experiment), seed=2, round=0, index=1,
                    sample_dir=rows[1]["sample_dir"], tasks_sha256=controller.sha(original))
        matrix.write_text(json.dumps(task) + "\n")
        args = SimpleNamespace(dispatch=matrix, dispatch_sha256=controller.sha(matrix))
        with mock.patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "0"}), \
             mock.patch.object(controller, "load_requests", side_effect=scheduler.requests), \
             mock.patch.object(backend, "measure") as measure, \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("scheduler forbidden")):
            controller.dispatch(args)
            called = measure.call_args.args[0]
            self.assertEqual(vars(called), dict(experiment=str(experiment), seed=2, round=0,
                                              index=1, storage="scratch"))
            matrix.write_text(json.dumps(dict(task, sample_dir="foreign")) + "\n")
            with self.assertRaises(ValueError):
                controller.dispatch(args)
            self.assertEqual(measure.call_count, 1)

    def test_dispatch_rejects_negative_and_out_of_range_indices(self):
        scheduler, _, tasks = self.outcome_fixture()
        matrix = self.root / "dispatch_bounds.jsonl"
        matrix.write_text(json.dumps(tasks[0]) + "\n")
        args = SimpleNamespace(dispatch=matrix, dispatch_sha256=controller.sha(matrix))
        with mock.patch.object(controller, "load_requests", side_effect=scheduler.requests), \
             mock.patch.object(backend, "measure") as measure, \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("scheduler forbidden")):
            for index in ["-1", "1", "999"]:
                with self.subTest(array_index=index), mock.patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": index}):
                    with self.assertRaises(IndexError):
                        controller.dispatch(args)
            for inner_index in [-1, 2]:
                matrix.write_text(json.dumps(dict(tasks[0], index=inner_index)) + "\n")
                args.dispatch_sha256 = controller.sha(matrix)
                with self.subTest(request_index=inner_index), mock.patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "0"}):
                    with self.assertRaises(ValueError):
                        controller.dispatch(args)
            self.assertEqual(measure.call_count, 0)


if __name__ == "__main__":
    unittest.main()
