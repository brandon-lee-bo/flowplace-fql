"""Bounded CPU tests of identical-prefix import, with no physical tool calls.

Tiny real actor updates/checkpoints and unchanged request/replay validators run.
Only graph construction/CUDA factories and native physical parsing are replaced
by explicit synthetic fixtures. These tests make no PPA or tool-quality claim.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_branch as branch
from . import qstudy_actor as actor
from . import qstudy_backend as backend
from .graph_o2o_archive import atomic_json, sha
from . import test_qstudy_actor as actor_tests


class InterruptedImport(RuntimeError):
    pass


class BranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run prefix import tests in bounded Slurm CPU allocation")
        torch.set_num_threads(1)

    def setUp(self):
        self.fixture = actor_tests.ActorRunnerTests("test_common_initialization_frozen_offline_q_and_zero_q_bc")
        self.fixture.setUp()
        self.root = self.fixture.root
        self.project = self.root/"project"
        for name in branch.IMPORT_SOURCES:
            dest = self.project/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(branch.ROOT/name, dest)
        cache = self.project/"baseline/runtime/benchmark_cache/bp_fe.pt"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"static synthetic graph cache")
        atomic_json(self.fixture.dataset.with_name("manifest.json"), dict(synthetic_test=True))
        self.fixture.cfg.update(cache_sha256=sha(cache), backend_threads=8, reward_scale=100.,
            backend_storage="scratch", seeds=[0, 1])
        for name in ("common_checkpoints", "critic_checkpoints", "common_checkpoint_sha256", "critic_checkpoint_sha256"):
            self.fixture.cfg[name]["1"] = self.fixture.cfg[name]["0"]
        self.source = self.fixture.experiment("source")
        self.target = self.root/"updated_branch"
        self.cfg = copy.deepcopy(self.fixture.cfg)
        self.cfg.update(arm="updated", run_prefix="updatedflow", online_flow_update=True)
        for seed in self.fixture.cfg["seeds"]:
            with self.fixture.cpu_fixture():
                actor.train(SimpleNamespace(experiment=self.source, seed=seed, round=0))
            self.emit(seed)
        self.before = branch.inventory(self.source)
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(branch, "ROOT", self.project))
        self.stack.enter_context(mock.patch.object(backend, "context", side_effect=self.context))
        self.verify = self.stack.enter_context(mock.patch.object(backend, "verify_run", side_effect=self.verify_run))
        # Any accidental subprocess/physical-tool action fails the test.
        self.stack.enter_context(mock.patch("subprocess.run", side_effect=AssertionError("No physical tools or jobs allowed")))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.stack.close()
        self.fixture.tearDown()

    def context(self, experiment):
        experiment = Path(experiment)
        cfg = branch.read(experiment/"config.json")
        if sha(cfg["dataset_path"]) != cfg["dataset_sha256"] or cfg["backend_threads"] != 8:
            raise ValueError("Synthetic pinned dataset/threads changed")
        binding = dict(design="bp_fe", dataset_path=cfg["dataset_path"],
            dataset_sha256=cfg["dataset_sha256"],
            dataset_manifest_sha256=sha(Path(cfg["dataset_path"]).with_name("manifest.json")),
            physical_flow_revision=backend.PHYSICAL_REVISIONS["bp_fe"], physical_reference="synthetic reference",
            reward_scale=cfg["reward_scale"], config_sha256=sha(experiment/"config.json"))
        return cfg, None, binding

    def verify_run(self, run, row, cfg, verification):
        run = Path(run)
        meta = branch.read(run/"manifest.json")
        if (meta["def_sha256"] != row["def_sha256"] or sha(meta["def_path"]) != row["def_sha256"]
                or meta["physical_flow_revision"] != backend.PHYSICAL_REVISIONS["bp_fe"]
                or meta["num_cores"] != 8 or not meta["success"]):
            raise ValueError("Synthetic native pinned DEF/revision/threads mismatch")
        geometry = branch.read(run/"backend_macro_audit.json")
        if geometry["bindings"]["input_def_sha256"] != row["def_sha256"]:
            raise ValueError("Synthetic cached geometry binding mismatch")
        return dict(status="success", metrics=branch.read(run/"metrics.json"),
            physical_manifest_sha256=sha(run/"manifest.json"),
            physical_flow_revision=meta["physical_flow_revision"])

    def emit(self, seed):
        folder = self.source/f"seed_{seed}/round_0"
        checkpoint = folder/"checkpoint.pt"
        saved = torch.load(checkpoint, weights_only=False)
        policy = actor_tests.TinyPolicy(self.fixture.problem)
        policy.load_state_dict(saved["actor"])
        cfg = self.fixture.cfg
        requests, outcomes = [], []
        for role in ("collect", "evaluation"):
            count = cfg["online_samples"] if role == "collect" else cfg["evaluation_samples"]
            noise_seed = cfg["evaluation_noise_seed"] if role == "evaluation" else cfg["collection_noise_seed"]+seed*100
            noises = 2*torch.rand(count, 3, 2, generator=torch.Generator().manual_seed(noise_seed))-1
            for index, noise in enumerate(noises):
                sample = folder/"requests"/cfg["arm"]/role/str(index)
                sample.mkdir(parents=True)
                with torch.no_grad():
                    raw = policy(noise[None])[0].numpy()
                executed = raw.copy()
                np.savez_compressed(sample/"actions.npz", noise=noise.numpy(), raw_action=raw, action=executed)
                path = sample/"layout.def"
                path.write_text("# Synthetic DEF fixture only\n"+json.dumps(executed.tolist())+"\n")
                atomic_json(path.with_suffix(".audit.json"), dict(output_sha256=sha(path), legal_no_halo=True))
                row = dict(arm=cfg["arm"], seed=seed, round=0, role=role, index=index, graph="bp_fe",
                    terminal_mask=0, training_feedback_eligible=role == "collect", sample_dir=str(sample),
                    config_sha256=sha(self.source/"config.json"), run_id=f"synthetic_{seed}_{role}_{index}",
                    raw_action=raw.tolist(), executed_action=executed.tolist(), predicted_q=[-1.],
                    raw_action_sha256=actor.action_hash(raw), action_sha256=actor.action_hash(executed),
                    noise_sha256=actor.action_hash(noise.numpy()), noise_seed=noise_seed,
                    source_checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    def_path=str(path), def_sha256=sha(path), rounded_geometry=True,
                    geometry=dict(legal=True), objective_weights=cfg["weights"], stage="global_route_only")
                atomic_json(sample/"request.json", row)
                run = self.root/"native"/f"s{seed}_{role}_{index}"
                run.mkdir(parents=True)
                metrics = dict(rWL_um=100.+index, WNS_ns=-.4, TNS_ns=-20.-index,
                    horizontal_overflow_sum=1., vertical_overflow_sum=2., Power_W=.2,
                    StdCellArea_um2=1000., runtime_seconds=1.)
                atomic_json(run/"manifest.json", dict(design="bp_fe", success=True, time_end=4., time_start=1.,
                    def_sha256=sha(path), def_path=str(path), num_cores=8,
                    physical_flow_revision=backend.PHYSICAL_REVISIONS["bp_fe"]))
                atomic_json(run/"metrics.json", metrics)
                atomic_json(run/"backend_macro_audit.json", dict(legal_no_halo=True, input_to_actual_changes=[],
                    bindings=dict(input_def_sha256=sha(path))))
                out = dict(row, request_sha256=sha(sample/"request.json"), dataset_sha256=cfg["dataset_sha256"],
                    status="success", metrics=metrics, reward=-metrics["rWL_um"]/cfg["reward_scale"],
                    physical_run=str(run), physical_manifest_sha256=sha(run/"manifest.json"),
                    metrics_sha256=sha(run/"metrics.json"), backend_macro_geometry_sha256=sha(run/"backend_macro_audit.json"),
                    physical_flow_revision=backend.PHYSICAL_REVISIONS["bp_fe"], backend_macro_legal=True,
                    backend_track_snapped_macro_count=0, physical_slurm_job_id="synthetic", detailed_route_DRC=None)
                # Production outcomes do not contain the export-only source/sample fields.
                for key in ("sample_dir", "source_checkpoint", "def_path", "def_sha256", "geometry",
                            "rounded_geometry", "objective_weights"):
                    out.pop(key)
                requests.append(row); outcomes.append((sample, out))
        tasks = folder/"requests/tasks.jsonl"
        tasks.write_text("".join(json.dumps(row)+"\n" for row in requests))
        for sample, out in outcomes:
            out["tasks_sha256"] = sha(tasks)
            atomic_json(sample/"outcome.json", out)
        atomic_json(tasks.with_name("completed.json"), dict(status="completed", count=len(requests),
            tasks_sha256=sha(tasks), config_sha256=sha(self.source/"config.json")))

    def perform(self, **kwargs):
        return branch.import_prefix(self.source, self.target, self.cfg, **kwargs)

    def source_request(self, seed=0, index=0):
        return branch.load_requests(self.source, seed, 0)[0][index]

    def test_preserves_all_tensors_optimizer_rng_steps_and_original_native_evidence(self):
        result = self.perform()
        self.assertEqual(result["new_train_updates"], 0)
        self.assertEqual(result["unique_physical_calls"], 0)
        self.assertTrue(result["no_new_independent_samples"])
        self.assertEqual(branch.inventory(self.source), self.before)
        for seed in self.cfg["seeds"]:
            source = torch.load(self.source/f"seed_{seed}/round_0/checkpoint.pt", weights_only=False)
            target = torch.load(self.target/f"seed_{seed}/round_0/checkpoint.pt", weights_only=False)
            self.fixture.assert_nested_equal(
                {k:v for k,v in source.items() if k != "binding"},
                {k:v for k,v in target.items() if k not in {"binding", "prefix_import"}})
            self.assertEqual(source["binding"]["seed"], target["binding"]["seed"])
            self.assertNotEqual(source["binding"]["config_sha256"], target["binding"]["config_sha256"])
            rows, _ = branch.load_requests(self.target, seed, 0)
            originals, _ = branch.load_requests(self.source, seed, 0)
            for row, old in zip(rows, originals):
                self.assertEqual(row["raw_action"], old["raw_action"])
                self.assertEqual(row["executed_action"], old["executed_action"])
                self.assertEqual(row["noise_sha256"], old["noise_sha256"])
                self.assertEqual(row["def_sha256"], old["def_sha256"])
                out, original = branch.read(Path(row["sample_dir"])/"outcome.json"), branch.read(Path(old["sample_dir"])/"outcome.json")
                self.assertEqual(out["metrics"], original["metrics"])
                self.assertEqual(out["physical_run"], original["physical_run"])
                self.assertIs(out["new_measurement"], False)

    def test_rejects_config_and_unsupported_flag_changes(self):
        for overrides in (dict(alpha=5.), dict(q_weight=0.), dict(online_flow_update=False),
                          dict(online_updates=5), dict(online_rounds=3), dict(seeds=[0])):
            with self.subTest(overrides=overrides):
                cfg = dict(self.cfg, **overrides)
                with self.assertRaises(ValueError):
                    branch.import_prefix(self.source, self.target, cfg)
        self.assertFalse(self.target.exists())
        cfg = dict(self.cfg, online_rounds=2)
        branch.import_prefix(self.source, self.target, cfg)
        with self.assertRaisesRegex(ValueError, "configuration is immutable"):
            branch.import_prefix(self.source, self.target, self.cfg)

    def test_current_training_source_and_initialization_tampering_refused(self):
        path = self.project/"fql/qstudy_actor.py"
        path.write_text(path.read_text()+"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "training binding"):
            self.perform()
        self.assertFalse((self.target/"branch_import.json").exists())

    def test_incomplete_export_or_native_outcome_refused_before_publication(self):
        row = self.source_request()
        outcome = Path(row["sample_dir"])/"outcome.json"
        outcome.unlink()
        with self.assertRaisesRegex(ValueError, "native terminal evidence"):
            self.perform()
        self.assertFalse((self.target/"seed_0").exists())

    def test_missing_audit_never_calls_native_validator_or_tool(self):
        row = self.source_request()
        out = branch.read(Path(row["sample_dir"])/"outcome.json")
        (Path(out["physical_run"])/"backend_macro_audit.json").unlink()
        with self.assertRaisesRegex(ValueError, "never run OpenROAD"):
            self.perform()
        self.verify.assert_not_called()

    def test_bad_native_def_threads_and_inline_action_refused(self):
        row = self.source_request()
        outcome = Path(row["sample_dir"])/"outcome.json"
        original = branch.read(outcome)
        changed = copy.deepcopy(original)
        changed["raw_action"][0][0] += .1
        atomic_json(outcome, changed)
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            self.perform()
        atomic_json(outcome, original)
        manifest = Path(original["physical_run"])/"manifest.json"
        bad = branch.read(manifest); bad["num_cores"] = 4
        atomic_json(manifest, bad)
        changed = dict(original, physical_manifest_sha256=sha(manifest))
        atomic_json(outcome, changed)
        with self.assertRaisesRegex(ValueError, "threads mismatch"):
            self.perform()

    def test_idempotence_refuses_changed_target_or_source(self):
        first = self.perform()
        before = branch.inventory(self.target)
        self.assertEqual(self.perform(), first)
        self.assertEqual(branch.inventory(self.target), before)
        path = self.target/"seed_0/round_0/checkpoint.pt"
        data = path.read_bytes(); path.write_bytes(data+b"corruption")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            self.perform()

    def test_interrupted_atomic_journal_write_recovers_without_changing_configs(self):
        self.target.mkdir()
        branch.atomic_json(self.target/"config.json", self.cfg)
        config_bytes = (self.target/"config.json").read_bytes()
        original_replace = os.replace

        def interrupt_plan(source, destination):
            if Path(destination) == self.target/"import_plan.json":
                raise InterruptedImport("Process dies after journal fsync, before publication")
            return original_replace(source, destination)

        with mock.patch.object(branch.os, "replace", side_effect=interrupt_plan):
            with self.assertRaises(InterruptedImport):
                self.perform()
        self.assertFalse((self.target/"import_plan.json").exists())
        self.assertTrue(list(self.target.glob("import_plan.json.tmp.*")))
        # Partial writes from other uncommitted root journals also carry no authority.
        (self.target/"branch_import.json.tmp.123").write_text('{"status":')
        (self.target/"config.json.tmp.123").write_text('{"arm":')
        self.perform()
        self.assertFalse(list(self.target.glob("*.tmp.*")))
        self.assertEqual((self.target/"config.json").read_bytes(), config_bytes)
        self.assertEqual(branch.inventory(self.source), self.before)

    def test_resume_after_atomic_publication_adopts_prefix_without_retraining(self):
        def stop(phase, seed):
            if phase == "published" and seed == 0:
                raise InterruptedImport()
        with self.assertRaises(InterruptedImport):
            self.perform(interrupt=stop)
        checkpoint = self.target/"seed_0/round_0/checkpoint.pt"
        digest = sha(checkpoint)
        self.assertFalse((self.target/"branch_import.json").exists())
        self.assertFalse((self.target/"seed_1").exists())
        with mock.patch.object(actor, "train", side_effect=AssertionError("No training during import")):
            result = self.perform()
        self.assertEqual(sha(checkpoint), digest)
        self.assertEqual(result["seeds"], [0, 1])
        self.assertEqual(branch.inventory(self.source), self.before)

    def test_partial_uncommitted_stage_rebuilt_but_ready_corruption_refused(self):
        def partial(stage, *args, **kwargs):
            (stage/"round_0").mkdir(parents=True)
            (stage/"round_0/checkpoint.pt").write_bytes(b"interrupted checkpoint")
            raise InterruptedImport()
        with mock.patch.object(branch, "build_seed", side_effect=partial):
            with self.assertRaises(InterruptedImport):
                self.perform()
        self.assertFalse((self.target/"seed_0").exists())
        def stop(phase, seed):
            if phase == "staged" and seed == 0:
                raise InterruptedImport()
        with self.assertRaises(InterruptedImport):
            self.perform(interrupt=stop)
        staged = self.target/".branch_staging/seed_0/round_0/checkpoint.pt"
        staged.write_bytes(staged.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "corrupted before publication"):
            self.perform()

    def test_foreign_target_and_changed_source_after_plan_are_refused(self):
        self.target.mkdir()
        (self.target/"foreign.txt").write_text("untouched")
        with self.assertRaisesRegex(ValueError, "Target must be empty"):
            self.perform()
        self.assertEqual((self.target/"foreign.txt").read_text(), "untouched")
        (self.target/"foreign.txt").unlink()
        def stop(phase, seed):
            if phase == "staged":
                raise InterruptedImport()
        with self.assertRaises(InterruptedImport):
            self.perform(interrupt=stop)
        curve = self.source/"seed_0/round_0/learning_curve.jsonl"
        curve.write_text(curve.read_text()+"{}\n")
        with self.assertRaisesRegex(ValueError, "provenance plan changed"):
            self.perform()

    def test_exact_payload_hash_detects_optimizer_rng_and_tensor_changes(self):
        saved = torch.load(self.source/"seed_0/round_0/checkpoint.pt", weights_only=False)
        before = branch.state_fingerprints(saved)
        for key in ("actor", "optimizers", "rng", "generators", "step"):
            changed = copy.deepcopy(saved)
            if key == "actor":
                changed[key]["linear.weight"][0, 0] += 1.
            elif key == "optimizers":
                changed[key]["actor"]["param_groups"][0]["lr"] *= 2
            elif key == "rng":
                changed[key][0] ^= 1
            elif key == "generators":
                changed[key]["actor"][0] ^= 1
            else:
                changed[key] += 1
            self.assertNotEqual(branch.state_fingerprints(changed)["all_payload"], before["all_payload"])

    def test_frozen_round1_actor_consumes_imported_prefix_and_exact_shared_feedback(self):
        self.perform()
        read_replay = actor.read_replay
        with self.fixture.cpu_fixture():
            with mock.patch.object(actor, "read_replay", side_effect=read_replay):
                actor.train(SimpleNamespace(experiment=self.source, seed=0, round=1))
                actor.train(SimpleNamespace(experiment=self.target, seed=0, round=1))
        a = torch.load(self.source/"seed_0/round_1/checkpoint.pt", weights_only=False)
        b = torch.load(self.target/"seed_0/round_1/checkpoint.pt", weights_only=False)
        start = torch.load(self.source/"seed_0/round_0/checkpoint.pt", weights_only=False)
        self.fixture.assert_nested_equal(a["flow"], start["flow"])
        self.assertNotEqual(branch.tree_sha(b["flow"]), branch.tree_sha(start["flow"]))
        self.fixture.assert_nested_equal(a["critic"], b["critic"])
        self.fixture.assert_nested_equal(a["generators"]["actor"], b["generators"]["actor"])
        self.fixture.assert_nested_equal(a["generators"]["q"], b["generators"]["q"])
        self.assertNotEqual(branch.tree_sha(a["actor"]), branch.tree_sha(b["actor"]))
        self.assertEqual(len(a["binding"]["replay"]), self.cfg["online_samples"])
        self.assertEqual(len(b["binding"]["replay"]), self.cfg["online_samples"])
        self.assertTrue(all("/collect/" in r["path"] for r in b["binding"]["replay"]))
        self.assertEqual(self.perform()["unique_physical_calls"], 0)


if __name__ == "__main__":
    unittest.main()
