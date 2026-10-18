"""Small CPU contracts for measured-quality FM; execute only through Slurm."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import qstudy_quality_weighted_flow as q
from .qstudy_data import METRIC_NAMES


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 2))
        self.register_buffer("movable", torch.tensor([True, False]))
        self.register_buffer("fixed", torch.tensor([[0., 0.], [.7, -.4]]))

    def restore(self, x):
        return torch.where(self.movable[None, :, None], x, self.fixed[None])

    def velocity(self, x, t):
        return self.net(torch.cat((x, t[:, None, None].expand(-1, len(self.fixed), 1)), -1))*self.movable[None, :, None]


def training_rows(n=6, family_offset=0, provenance_offset=0):
    raw = np.linspace(-.8, .8, n*4, dtype=np.float32).reshape(n, 4)
    metrics = np.ones((n, 7), np.float64)
    metrics[:, 0] = np.linspace(1e6, 3e6, n)
    metrics[:, 1:3] = -.3
    return dict(raw_action=raw, executed_action=raw+.01, metrics=metrics,
        metric_mask=np.isfinite(metrics), status=np.array(["success"]*n, dtype="U24"),
        family=np.arange(n)//2+family_offset, source_row=np.arange(n), variant=np.arange(n)%2,
        provenance_index=np.arange(n)+provenance_offset)


def nested_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b), "State tensors differ")
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            nested_equal(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x,y in zip(a,b):
            nested_equal(test, x, y)
    else:
        test.assertEqual(a,b)


class QualityWeightsTests(unittest.TestCase):
    def test_tied_ranks_direction_support_normalization_and_scale_independence(self):
        w, info = q.quality_weights([0., 2., 2., 4.])
        np.testing.assert_allclose(w, np.array([.5, 1., 1., 2.])/1.125, rtol=1e-6)
        self.assertEqual(w[1], w[2])
        self.assertGreater(w.min(), 0)
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertLessEqual(float(w.max()/w.min()), 4.)
        np.testing.assert_array_equal(w, q.quality_weights(np.array([0.,2.,2.,4.])*19-7)[0])
        self.assertEqual(info["split"], "train")
        np.testing.assert_array_equal(q.quality_weights([3.,3.,3.])[0], np.ones(3))
        for invalid in ([1.], [0., float("nan")], [[1.,2.]]):
            with self.assertRaises(ValueError):
                q.quality_weights(invalid)

    def test_train_only_failure_missing_exclusion_and_no_behavior_substitution(self):
        rows = training_rows()
        rows["status"][2:5] = ["physical_failure", "geometry_failure", "unresolved"]
        rows["metric_mask"][5, 0] = False
        rows["metrics"][5, 0] = float("nan")
        data = dict(train=rows, validation={"metrics": "MUST NOT ACCESS"}, behavior_action="MUST NOT ACCESS")
        pop = q.train_population(data)
        np.testing.assert_array_equal(pop["indices"], [0,1])
        np.testing.assert_array_equal(pop["targets"], rows["raw_action"][:2])
        self.assertEqual(pop["exclusion"], dict(total_train=6, retained=2, confirmed_failure=2,
            unsuccessful_other=1, success_missing_or_invalid_wl=1, retained_families=1))
        self.assertAlmostEqual(pop["reference_wl_um"], rows["metrics"][:2,0].mean())
        before = q.population_binding(pop)
        data["validation"] = {"metrics": np.array([1e100])}
        data["behavior_action"] = np.zeros((256,99))
        self.assertEqual(before, q.population_binding(q.train_population(data)))
        rows["provenance_index"][1] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            q.train_population(data)

    def test_actual_unreduced_fm_formula_fixed_nodes_and_detached_weights(self):
        from .graph_o2o_models import GraphPolicy
        torch.manual_seed(12)
        flow = TinyFlow()
        target, noise, t = torch.rand(3,2,2), torch.rand(3,2,2), torch.rand(3)
        uniform, _ = q.weighted_flow_loss(flow, target, noise, t, torch.ones(3))
        direct = GraphPolicy.flow_loss(flow, target, noise, t)
        torch.testing.assert_close(uniform, direct)
        weights = torch.tensor([.5,1.,1.5], requires_grad=True)
        errors = q.per_layout_flow_loss(flow, target, noise, t)
        loss, plain = q.weighted_flow_loss(flow, target, noise, t, weights)
        torch.testing.assert_close(loss, (errors*weights.detach()).mean())
        torch.testing.assert_close(plain, errors.mean())
        loss.backward()
        self.assertIsNone(weights.grad)
        changed, changed_noise = target.clone(), noise.clone()
        changed[:,1] += 30
        changed_noise[:,1] -= 30
        torch.testing.assert_close(q.per_layout_flow_loss(flow, changed, changed_noise, t), errors)
        with self.assertRaises(ValueError):
            q.weighted_flow_loss(flow, target, noise, t, torch.tensor([0.,1.,1.]))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.manual_seed(81)
        self.initial = TinyFlow().state_dict()
        self.targets = torch.linspace(-.9,.9,32).reshape(8,2,2)
        self.settings = dict(q.SETTINGS, steps=9, batch_size=4, checkpoint_every=2, lr=1e-3)
        self.binding = dict(source="same-completed-initial-flow", seed=0)

    def tearDown(self):
        self.temp.cleanup()

    def fit(self, name, weights=None, stop=lambda:False, binding=None):
        torch.manual_seed(91)
        flow = TinyFlow()
        flow.load_state_dict(self.initial)
        with redirect_stdout(io.StringIO()):
            done = q.fit(flow, self.targets, torch.ones(8) if weights is None else weights,
                self.settings, 0, binding or self.binding, self.root/name, stop)
        return done, torch.load(self.root/name/"checkpoint.pt", map_location="cpu", weights_only=False)

    def test_paired_draws_and_optimizer_reset_differ_only_in_quality_weight(self):
        _, uniform = self.fit("uniform")
        weights = torch.tensor(q.quality_weights(np.arange(8))[0])
        _, weighted = self.fit("weighted", weights)
        self.assertEqual(uniform["draw_chain_sha256"], weighted["draw_chain_sha256"])
        nested_equal(self, uniform["generators"], weighted["generators"])
        nested_equal(self, uniform["rng"], weighted["rng"])
        self.assertTrue(any(not torch.equal(uniform["flow"][k], weighted["flow"][k]) for k in uniform["flow"]))
        for saved in (uniform, weighted):
            self.assertTrue(all(float(s["step"]) == 9 for s in saved["optimizer"]["state"].values()))
        _, same = self.fit("ones")
        nested_equal(self, uniform["flow"], same["flow"])
        nested_equal(self, uniform["optimizer"], same["optimizer"])

    def test_interrupted_resume_preserves_exact_optimizer_rng_draws_and_parameters(self):
        _, uninterrupted = self.fit("whole")
        calls = [0]
        def stop():
            calls[0] += 1
            return calls[0] == 3
        with self.assertRaises(SystemExit) as caught:
            self.fit("resumed", stop=stop)
        self.assertEqual(caught.exception.code, 85)
        self.assertFalse((self.root/"resumed/completed.json").exists())
        _, resumed = self.fit("resumed")
        for key in ("flow", "optimizer", "rng", "cuda_rng", "generators", "step", "draw_chain_sha256"):
            nested_equal(self, uninterrupted[key], resumed[key])

    def test_completion_gap_idempotence_and_changed_binding_or_curve_refusal(self):
        done, _ = self.fit("finished")
        checkpoint = self.root/"finished/checkpoint.pt"
        before = q.sha(checkpoint)
        again, _ = self.fit("finished")
        self.assertEqual(done, again)
        (self.root/"finished/completed.json").unlink()
        recovered, _ = self.fit("finished")
        self.assertEqual(before, q.sha(checkpoint))
        self.assertEqual(recovered["steps"], 9)
        with self.assertRaisesRegex(ValueError, "binding"):
            self.fit("finished", binding=dict(self.binding, seed=2))
        (self.root/"finished/learning_curve.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "curve"):
            self.fit("finished")

    def test_loaded_common_or_resume_cannot_replace_fixed_graph_geometry(self):
        flow=TinyFlow()
        source=copy.deepcopy(self.initial)
        source["fixed"][1,0]+=.1
        with self.assertRaisesRegex(ValueError,"geometry"):
            q.load_flow_state(flow,source)
        source=copy.deepcopy(self.initial)
        source["movable"][1]=True
        with self.assertRaisesRegex(ValueError,"geometry"):
            q.load_flow_state(flow,source)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project, self.data, self.output = self.root/"project", self.root/"data", self.root/"new-control"
        self.data.mkdir()
        for name in q.SOURCE_NAMES:
            path = self.project/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source "+name)
        self.base = self.root/"old-bc/config.json"
        self.base.parent.mkdir()
        cache = self.project/"baseline/runtime/benchmark_cache/bp_fe.pt"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cache-context")
        public = self.project/"baseline/runtime/logs/debug/mask.61.train_mask_flowplace_uniform.61/best.ckpt"
        public.parent.mkdir(parents=True)
        public.write_bytes(b"public-constructor")
        train, val = training_rows(384), training_rows(16, family_offset=192, provenance_offset=384)
        arrays = {s+"_"+k:v for s,rows in (("train",train),("validation",val)) for k,v in rows.items()}
        arrays.update(behavior_action=np.zeros((256,4),np.float32), behavior_family=np.arange(256))
        np.savez_compressed(self.data/"dataset.npz", **arrays)
        (self.data/"rows.jsonl").write_text("synthetic-row-binding\n")
        q.atomic_json(self.data/"manifest.json", dict(design="bp_fe",stage="global_route_only",
            metric_names=list(METRIC_NAMES), dataset_sha256=q.sha(self.data/"dataset.npz"),
            rows_sha256=q.sha(self.data/"rows.jsonl"), no_test_or_heldout_exported=True,
            historical_online_included=False, split_counts=dict(train=384,validation=16)))
        checkpoints = {}
        for seed in q.SEEDS:
            checkpoint = self.root/f"common{seed}/common.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(("initial-independent-seed"+str(seed)).encode())
            checkpoints[str(seed)] = checkpoint
        cfg = dict(design="bp_fe",seeds=list(q.SEEDS),online_flow_update=False,flow_steps=10,
            q_weight=0.,offline_critic_update=False,dataset_path=str(self.data/"dataset.npz"),
            dataset_sha256=q.sha(self.data/"dataset.npz"),cache_sha256=q.sha(cache),
            common_checkpoints={k:str(v) for k,v in checkpoints.items()},
            common_checkpoint_sha256={k:q.sha(v) for k,v in checkpoints.items()})
        q.atomic_json(self.base,cfg)
        self.patches = [mock.patch.object(q,"ROOT",self.project),
                        mock.patch.object(q,"DATASET_SHA",q.sha(self.data/"dataset.npz")),
                        mock.patch.dict("os.environ",SLURM_JOB_ID="unit",SLURM_CPUS_PER_TASK="1")]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temp.cleanup()

    def test_immutable_prepare_real_loader_snapshots_and_all_six_tasks(self):
        plan = q.prepare(self.base,self.output)
        self.assertEqual(plan,q.prepare(self.base,self.output))
        checked,pop = q.checked_plan(self.output/"plan.json")
        self.assertEqual(checked,plan)
        self.assertEqual(len(pop["targets"]),384)
        self.assertEqual(len(plan["tasks"]),6)
        self.assertEqual({t["arm"] for t in plan["tasks"]},{"uniform","quality_rank"})
        self.assertEqual(plan["population"]["eligible_train_indices"],list(range(384)))
        self.assertEqual(len(set(c["sha256"] for c in plan["inputs"]["common"].values())),3)
        self.assertFalse(plan["scope"]["automatic_launch"])
        self.assertFalse(list(self.output.glob("*/seed_*")))
        snapshot=self.output/"sources"/q.SOURCE_NAMES[0]
        snapshot.write_text("changed snapshot")
        with self.assertRaisesRegex(ValueError,"snapshots"):
            q.checked_plan(self.output/"plan.json")

    def test_changed_initial_checkpoint_dataset_and_task_plan_fail_closed(self):
        plan=q.prepare(self.base,self.output)
        path=Path(plan["inputs"]["common"]["2"]["path"])
        original=path.read_bytes()
        path.write_bytes(b"seed0-standin")
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            q.checked_plan(self.output/"plan.json")
        path.write_bytes(original)
        plan["tasks"][1]["seed"]=0
        q.atomic_json(self.output/"plan.json",plan)
        with self.assertRaisesRegex(ValueError,"tasks"):
            q.checked_plan(self.output/"plan.json")
        with self.assertRaises(ValueError):
            q.prepare(self.base,self.data/"bad-output")

    def test_protocol_data_substitution_and_seed0_standin_refused(self):
        cfg=q.read(self.base)
        original=dict(cfg)
        cfg["dataset_sha256"]="1"*64
        q.atomic_json(self.base,cfg)
        with self.assertRaisesRegex(ValueError,"Initial control"):
            q.prepare(self.base,self.output)
        cfg=copy.deepcopy(original)
        cfg["common_checkpoints"]["2"]=cfg["common_checkpoints"]["0"]
        cfg["common_checkpoint_sha256"]["2"]=cfg["common_checkpoint_sha256"]["0"]
        q.atomic_json(self.base,cfg)
        with self.assertRaisesRegex(ValueError,"independently"):
            q.prepare(self.base,self.output)


if __name__ == "__main__":
    unittest.main()
