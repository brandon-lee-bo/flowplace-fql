"""Tiny structural source-study fixtures; no real actor, EDA, or quality claim.

All36 task records pass the unchanged source prediction verifier. Selected
refits contain real, untrained critic state so normalization/cache/model loading
is exercised. CV checkpoints are synthetic bytes, never used for actor init.
"""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_source_actor_config as adapter
from . import qstudy_source as source
from . import qstudy_screen as screen
from . import test_qstudy_source as source_tests
from . import test_qstudy_models as model_tests
from .qstudy_data import load_dataset
from .qstudy_models import measured_costs


class SourceActorConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run config/checkpoint tests through bounded Slurm CPU")
        torch.set_num_threads(1)

    def setUp(self):
        self.fixture = source_tests.SourceControlSafetyTests("test_selection_is_prelabel_deterministic_and_metric_independent")
        self.fixture.setUp()
        self.root, self.study = self.fixture.root, self.fixture.root/"study"
        self.study.mkdir()
        for splits in self.fixture.sources.values():
            for rows in splits.values():
                for row in rows:
                    row["design"] = "bp_fe"
                    row["backend"]["design"] = "bp_fe"
                    row["backend"]["physical_flow_revision"] = adapter.PHYSICAL_REVISIONS["bp_fe"]
        self.cache = model_tests.QStudyCriticTests.cache()
        self.cache_path = self.root/"cache/bp_fe.pt"
        self.cache_path.parent.mkdir()
        torch.save(self.cache, self.cache_path)
        datasets = {}
        for arm in source.ARMS:
            folder = self.study/"datasets"/arm
            selected = source.arm_rows(self.fixture.sources, self.fixture.selection(arm))
            source.export_arm(folder, arm, selected, self.fixture.reference, self.fixture.parents,
                self.fixture.selection(arm), "bp_fe")
            datasets[arm] = source.dataset_binding(folder/"dataset.npz")
        self.common = {}
        for seed in (0, 1, 2):
            experiment = self.root/("common_"+str(seed))
            folder = experiment/f"seed_{seed}/round_0"
            folder.mkdir(parents=True)
            source.atomic_json(experiment/"config.json", dict(seed=seed, synthetic_initialization=True))
            (folder/"common.pt").write_bytes(("Independent synthetic init "+str(seed)).encode())
            source.atomic_json(folder/"training_status.json", dict(status="completed", seed=seed, round=0,
                graph_cache_sha256=source.sha(self.cache_path), no_heldout_labels_used=True,
                config_sha256=source.sha(experiment/"config.json")))
            self.common[str(seed)] = source.bind(folder/"common.pt")
        self.teacher = self.root/"teacher.json"
        source.atomic_json(self.teacher, dict(common_seed=0, synthetic=True))
        declaration = dict(design="bp_fe", cache=source.bind(self.cache_path),
            old_source=datasets["O64"], teacher_config=source.bind(self.teacher),
            teacher_common=self.common["0"], teacher_reference=datasets["O64"]["dataset"],
            steps=2, folds=3, batch_size=64, lr=3e-4, weight_decay=1e-4)
        source.atomic_json(self.study/"plan.json", dict(declaration=declaration,
            declaration_sha256=source.object_sha(declaration)))
        tasks = []
        for arm in source.ARMS:
            for variant in source.VARIANTS:
                for seed in (0, 1, 2):
                    tasks.append(dict(index=len(tasks), arm=arm, variant=variant, seed=seed,
                        dataset=datasets[arm]["dataset"]["path"],
                        output=str(self.study/"runs"/arm/variant/f"seed_{seed}")))
        self.matrix = dict(plan=source.bind(self.study/"plan.json"), datasets=datasets,
            implementation=source.implementation_binding(), tasks=tasks, tasks_sha256=source.object_sha(tasks))
        source.atomic_json(self.study/"matrix.json", self.matrix)
        self.declaration = declaration
        for task in tasks:
            self.emit_task(task)
        report_dir = self.study/"report/software_fixture"
        report_dir.mkdir(parents=True)
        self.report = report_dir/"summary.json"
        rows = []
        for task in tasks:
            _, status = source.checked_predictions(task, self.matrix, declaration)
            rows.append(dict(task, **status))
        source.atomic_json(report_dir/"tasks.json", rows)
        source.atomic_json(self.report, dict(status="complete", completed_tasks=36, planned_tasks=36,
            integrity_errors=[], actor_training=False, automatic_promotion=False, backend_calls=0,
            matrix=source.bind(self.study/"matrix.json"), plan=self.matrix["plan"],
            report_source_sha256=source.sha(source.__file__)))
        self.base_path = self.root/"base/config.json"
        self.base_path.parent.mkdir()
        self.base = dict(design="bp_fe", arm="reference", run_prefix="reference", critic_variant="legacy",
            seeds=[0, 1, 2], alpha=10., weights=[1, 0, 0, 0, 0], q_weight=1., uncertainty_beta=0.,
            flow_steps=10, online_flow_update=False, offline_critic_update=False, terminal_mask=0,
            backend_threads=8, physical_failure_reward=-4, backend_storage="scratch",
            dataset_path=datasets["O64"]["dataset"]["path"], dataset_sha256=datasets["O64"]["dataset"]["sha256"],
            cache_sha256=source.sha(self.cache_path), online_rounds=2, offline_actor_steps=1000,
            online_updates=600, online_samples=8, evaluation_samples=8, evaluation_noise_seed=4600000,
            collection_noise_seed=4500000, actor_lr=1e-5, critic_lr=3e-4, flow_lr=1e-5, batch_size=16,
            common_checkpoints={k:v["path"] for k,v in self.common.items()},
            common_checkpoint_sha256={k:v["sha256"] for k,v in self.common.items()},
            critic_checkpoints={}, critic_checkpoint_sha256={})
        source.atomic_json(self.base_path, self.base)
        self.diagnostic = self.root/"direction_probe.json"
        source.atomic_json(self.diagnostic, dict(synthetic_review_only=True, no_actor_performance=True))
        self.note_path = self.root/"review.json"
        self.note = dict(date="2026-10-18", reason="Synthetic explicit source contrast for software verification",
            source_report_sha256=source.sha(self.report), reference_weights=[1, 0, 0, 0, 0],
            selections=[dict(source_arm=arm, variant=variant, actor_arm=arm.lower()+"_"+variant,
                             run_prefix=arm.lower()+"_"+variant)
                        for arm, variant in (("O64", "legacy"), ("T64", "legacy"), ("T64", "huber_residual"))],
            diagnostic_review_provenance=[dict(role="direction_probe", **source.bind(self.diagnostic))])
        source.atomic_json(self.note_path, self.note)
        self.args = SimpleNamespace(study=self.study, source_report=self.report, base_config=self.base_path,
            review_note=self.note_path, output=self.root/"prepared")
        self.stack = ExitStack()
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch("subprocess.run", side_effect=AssertionError("No actor/EDA/jobs")))
        self.stack.enter_context(mock.patch("baseline.api.load_problem", return_value=
            SimpleNamespace(cond=SimpleNamespace(is_ports=torch.zeros(3, dtype=torch.bool)))))

    def tearDown(self):
        self.stack.close()
        self.fixture.tearDown()

    def emit_task(self, task):
        data, _ = load_dataset(task["dataset"])
        arrays, indices = {}, {}
        for split in ("train", "validation"):
            costs, mask = measured_costs(torch.tensor(data[split]["metrics"][:, :5], dtype=torch.float32),
                screen.METRICS, torch.tensor(data[split]["metric_mask"][:, :5]))
            indices[split] = list(range(len(costs)))
            arrays[split] = dict(costs=costs, mask=mask,
                x=torch.tensor(data[split]["raw_action"]).reshape(len(costs), -1, 2),
                family=torch.tensor(data[split]["family"]))
        groups = source.stratified_folds(data["train"]["family"], data["train"]["source"])
        folder = Path(task["output"])
        folder.mkdir(parents=True)
        cfg = dict(dataset=task["dataset"], output=task["output"], variant=task["variant"], seed=task["seed"],
            steps=2, folds=3, batch_size=64, lr=3e-4, weight_decay=1e-4)
        binding = dict(dataset_sha256=source.sha(task["dataset"]),
            manifest_sha256=source.sha(Path(task["dataset"]).with_name("manifest.json")),
            matrix_sha256=source.sha(self.study/"matrix.json"), plan_sha256=self.matrix["plan"]["sha256"],
            cache_sha256=source.sha(self.cache_path), source_sha256=self.matrix["implementation"],
            config=cfg, family_folds=groups, source_indices=indices, task=task,
            heldout_used=False, backend_calls=0, fold_seed=41307)
        source.atomic_json(folder/"binding.json", binding)
        train = arrays["train"]
        model, normalization = screen.make_model(self.cache, train["x"], train["costs"], train["mask"],
            torch.ones(3, dtype=torch.bool), task["variant"], train["family"], task["seed"])
        outcomes = []
        for name in ("cv_0", "cv_1", "cv_2", "refit"):
            part = folder/name; part.mkdir()
            refit = name == "refit"
            held = [] if refit else groups[int(name[-1])]
            ti = np.arange(len(train["x"])) if refit else np.flatnonzero(~np.isin(data["train"]["family"], held))
            target = arrays["validation"] if refit else train
            vi = np.arange(len(target["x"])) if refit else np.flatnonzero(np.isin(data["train"]["family"], held))
            if refit:
                torch.save(dict(model=model.state_dict(), step=2, train_indices=ti.tolist(), binding=binding,
                    model_config=screen.VARIANTS[task["variant"]], normalization=normalization), part/"resume.pt")
            else:
                (part/"resume.pt").write_bytes(b"Synthetic CV checkpoint; never used for actor")
            truth = target["costs"][vi].numpy()
            predictions = truth[:, :1] if task["variant"] == "legacy" else truth
            np.savez_compressed(part/"predictions.npz", truth=truth, prediction=predictions,
                family=target["family"][vi].numpy(), target_indices=vi, train_indices=ti)
            done = dict(status="completed", binding=binding, steps=2, variant=task["variant"], seed=task["seed"],
                train_rows=len(ti), target_rows=len(vi), checkpoint_sha256=source.sha(part/"resume.pt"),
                normalization=normalization)
            source.atomic_json(part/"completed.json", done); outcomes.append(done)
        source.atomic_json(folder/"completed.json", dict(status="completed", binding=binding, outcomes=outcomes))

    def perform(self):
        return adapter.prepare(self.args)

    def update_note(self):
        source.atomic_json(self.note_path, self.note)

    def test_explicit_configs_preserve_budgets_targets_and_physical_scale(self):
        before = {str(p):source.sha(p) for p in (self.base_path, Path(self.base["dataset_path"]), self.cache_path)}
        manifest = self.perform()
        self.assertEqual(manifest["actor_jobs_submitted"], 0)
        self.assertEqual(manifest["backend_calls"], 0)
        self.assertFalse(manifest["automatic_selection"])
        self.assertIn("seed0 only", manifest["teacher_source_limitation"])
        sref = manifest["reference"]["scales"][0]
        for selected in manifest["selected"]:
            cfg = source.read_json(self.args.output/selected["selection"]["actor_arm"]/"config.json")
            for key in ("offline_actor_steps", "online_updates", "online_samples", "evaluation_samples",
                        "evaluation_noise_seed", "collection_noise_seed", "common_checkpoints", "actor_lr"):
                self.assertEqual(cfg[key], self.base[key])
            self.assertEqual(cfg["alpha"], 10.)
            self.assertFalse(cfg["online_flow_update"])
            self.assertNotIn("reward_scale", cfg)  # Do not relabel native outcomes.
            self.assertEqual(cfg["physical_failure_reward"], -4)
            if cfg["critic_variant"] == "legacy":
                self.assertEqual(cfg["online_rounds"], 0)
                sj = selected["source_scale"][0]
                self.assertAlmostEqual(cfg["q_weight"]*(-1400./sj), -1400./sref)
                self.assertAlmostEqual(selected["failure_utility"], -4*cfg["q_weight"])
            else:
                self.assertEqual(cfg["online_rounds"], 2)
                self.assertEqual(cfg["q_weight"], 1.)
                self.assertAlmostEqual(cfg["weights"][0]/selected["source_scale"][0], 1./sref)
        self.assertEqual({str(p):source.sha(p) for p in (self.base_path, Path(self.base["dataset_path"]), self.cache_path)}, before)

    def test_multi_metric_weight_correction_preserves_native_utility_and_failure(self):
        self.note["selections"] = [self.note["selections"][-1]]
        self.note["reference_weights"] = [1., .15, .35, .1, .1]
        self.update_note()
        manifest = self.perform()
        selected = manifest["selected"][0]
        cfg = source.read_json(self.args.output/selected["selection"]["actor_arm"]/"config.json")
        costs = np.asarray([1200., .2, 15., 1., 3.])
        source_utility = -np.sum(np.asarray(cfg["weights"])*costs/selected["source_scale"])
        ref_utility = -np.sum(np.asarray(self.note["reference_weights"])*costs/manifest["reference"]["scales"])
        self.assertAlmostEqual(source_utility, ref_utility)
        self.assertEqual(selected["failure_utility"], -4)

    def test_report_must_be_complete_current_and_review_bound(self):
        report = source.read_json(self.report)
        source.atomic_json(self.report, dict(report, completed_tasks=35))
        with self.assertRaisesRegex(ValueError, "all36 completed"):
            self.perform()
        source.atomic_json(self.report, report)
        self.note["source_report_sha256"] = "0"*64; self.update_note()
        with self.assertRaisesRegex(ValueError, "different source report"):
            self.perform()
        self.assertFalse(self.args.output.exists())

    def test_missing_or_implicit_review_selection_refused(self):
        self.note["selections"] = []; self.update_note()
        with self.assertRaisesRegex(ValueError, "Explicit reviewed"):
            self.perform()
        self.note["selections"] = [dict(source_arm="O64", variant="legacy", actor_arm="x", run_prefix="x")]
        self.note["diagnostic_review_provenance"] = []; self.update_note()
        with self.assertRaisesRegex(ValueError, "direction-probe"):
            self.perform()
        self.assertFalse(self.args.output.exists())

    def test_selected_checkpoint_tensor_tamper_refused_even_with_consistent_hashes(self):
        task = next(t for t in self.matrix["tasks"] if (t["arm"], t["variant"], t["seed"]) == ("T64", "huber_residual", 0))
        path = Path(task["output"])/"refit/resume.pt"
        saved = torch.load(path, weights_only=False)
        saved["model"]["objective_scale"][0] *= 2
        torch.save(saved, path)
        done = source.read_json(path.with_name("completed.json")); done["checkpoint_sha256"] = source.sha(path)
        source.atomic_json(path.with_name("completed.json"), done)
        root = Path(task["output"])/"completed.json"
        marker = source.read_json(root); marker["outcomes"][-1] = done; source.atomic_json(root, marker)
        report_tasks = self.report.with_name("tasks.json")
        rows = source.read_json(report_tasks); rows[task["index"]]["root_sha256"] = source.sha(root); source.atomic_json(report_tasks, rows)
        with self.assertRaisesRegex(ValueError, "normalization buffer differs"):
            self.perform()
        self.assertFalse(self.args.output.exists())

    def test_finite_initial_labels_and_original_bc_targets_required(self):
        path = Path(self.matrix["datasets"]["T64"]["dataset"]["path"])
        data, _ = load_dataset(path)
        fake = copy.deepcopy(data); fake["train"]["metrics"][0, 0] = np.nan
        fake["train"]["metric_mask"][0, 0] = False; fake["train"]["status"][0] = "physical_failure"
        real = adapter.load_dataset
        with mock.patch.object(adapter, "load_dataset", side_effect=lambda p: (fake, load_dataset(p)[1]) if Path(p) == path else real(p)):
            with self.assertRaisesRegex(ValueError, "finite successful initial TRAIN"):
                self.perform()
        fake = copy.deepcopy(data); fake["behavior_action"][0, 0] += .01
        with mock.patch.object(adapter, "load_dataset", side_effect=lambda p: (fake, load_dataset(p)[1]) if Path(p) == path else real(p)):
            with self.assertRaisesRegex(ValueError, "BC targets/families changed"):
                self.perform()
        self.assertFalse(self.args.output.exists())

    def test_stale_refit_or_cv_checkpoint_rejected_before_config_creation(self):
        path = Path(self.matrix["tasks"][-1]["output"])/"cv_0/resume.pt"
        path.write_bytes(b"Changed completed CV checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint/completion binding mismatch"):
            self.perform()
        self.assertFalse(self.args.output.exists())

    def test_idempotence_and_immutable_review_plan(self):
        first = self.perform()
        self.assertEqual(self.perform(), first)
        path = self.args.output/"preparation.json"
        before = source.sha(path)
        self.note["reason"] = "Changed scientific decision after preparation"; self.update_note()
        with self.assertRaisesRegex(ValueError, "preparation plan is immutable"):
            self.perform()
        self.assertEqual(source.sha(path), before)


if __name__ == "__main__":
    unittest.main()
