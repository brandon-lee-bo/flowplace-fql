"""Tiny Slurm CPU tests of actor orchestration, not graph/GPU performance.

Only CUDA tensor factories and the graph policy are replaced by CPU fixtures.
The actual actor train/export loops, multi-metric critic, checkpoint handling,
request hashes and replay admission execute. Native EDA execution is not tested.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import qstudy_actor as runner
from .graph_o2o_archive import atomic_json, sha
from .qstudy_models import MultiMetricCritic
from .flow_policy_config import RELEASED, PUBLIC_SHA


class TinyPolicy(nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.linear = nn.Linear(len(problem.x) * 2, len(problem.x) * 2)
        self.register_buffer("movable", ~problem.cond.is_ports.bool())
        self.register_buffer("fixed", problem.x.detach().clone())

    def restore(self, x, clip=False):
        if clip:
            x = x.clamp(-1, 1)
        return torch.where(self.movable[None, :, None], x, self.fixed[None])

    def forward(self, x, clip=True):
        return self.restore(self.linear(x.flatten(1)).reshape_as(x), clip)

    def sample_flow(self, noise, steps=10):
        return self(noise)

    def flow_loss(self, action, noise, t):
        return (self(noise, clip=False) - action).square().mean()


class ActorRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run actor orchestration tests under Slurm")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qstudy-actor-test-", dir=os.environ.get(
            "TMPDIR", "/scratch/bohanli/flowplace_graph"))
        self.root = Path(self.temp.name)
        self.cache = dict(node_id_to_name=["a", "b", "c"], ratio_x=10/224,
                          ratio_y=20/224, node_info={n: dict(x=1., y=1.) for n in "abc"},
                          net_info={"n": {"nodes": {n: {} for n in "abc"}}})
        self.problem = SimpleNamespace(x=torch.zeros(3, 2),
            cond=SimpleNamespace(is_ports=torch.zeros(3, dtype=torch.bool)), benchmark_info=self.cache)
        self.x = torch.rand((12, 3, 2), generator=torch.Generator().manual_seed(31)) * 1.2 - .6
        self.costs = torch.stack((100+self.x[:, 0, 0], .4+.1*self.x[:, 1, 0],
                                 20+self.x[:, 2, 1], 1+self.x[:, 0, 1], 1+self.x[:, 1, 1]), -1)
        self.mask = torch.ones_like(self.costs, dtype=torch.bool)
        self.names = tuple(runner.METRIC_NAMES[:5])
        self.families = torch.arange(6).repeat_interleave(2)
        self.critic_template = MultiMetricCritic(self.cache, self.x, self.costs,
            self.problem.cond.is_ports.logical_not(), self.names, width=12, seed=7)
        self.common = self.root/"common.pt"
        torch.manual_seed(12)
        flow, actor = TinyPolicy(self.problem), TinyPolicy(self.problem)
        torch.save(dict(flow=flow.state_dict(), actor=actor.state_dict()), self.common)
        self.qpath = self.root/"critic.pt"
        torch.save(dict(model=self.critic_template.state_dict()), self.qpath)
        self.dataset = self.root/"dataset.npz"
        self.dataset.write_bytes(b"synthetic-data-identity")
        self.cfg = dict(design="bp_fe", arm="test", run_prefix="unit", seeds=[0],
            dataset_path=str(self.dataset), dataset_sha256=sha(self.dataset),
            critic_variant="multi_huber", weights=[1, .15, .35, 0, 0],
            common_checkpoints={"0": str(self.common)}, critic_checkpoints={"0": str(self.qpath)},
            common_checkpoint_sha256={"0": sha(self.common)}, critic_checkpoint_sha256={"0": sha(self.qpath)},
            offline_actor_steps=7, online_updates=4, batch_size=4, online_samples=2,
            evaluation_samples=3, physical_failure_reward=-4, q_weight=1., alpha=10.,
            actor_lr=1e-3, flow_lr=1e-3, critic_lr=1e-3, online_flow_update=False,
            collection_noise_seed=123, evaluation_noise_seed=456)
        runner.STOP = False

    def tearDown(self):
        runner.STOP = False
        self.temp.cleanup()

    def experiment(self, name, **overrides):
        exp = self.root/name
        exp.mkdir()
        cfg = dict(self.cfg, **overrides)
        atomic_json(exp/"config.json", cfg)
        return exp

    def setup(self, exp, seed):
        cfg = json.loads((exp/"config.json").read_text())
        return cfg, self.problem, self.x, self.costs, self.mask, self.x, copy.deepcopy(
            self.critic_template), self.families

    @contextmanager
    def cpu_fixture(self, *, replay=None, setup=None):
        """Map orchestration tensors to CPU; no production code/device changes."""
        with ExitStack() as stack:
            for name in ("tensor", "rand", "randint", "full", "arange", "triu_indices"):
                original = getattr(torch, name)
                def redirect(*args, _original=original, **kwargs):
                    if str(kwargs.get("device", "")).startswith("cuda"):
                        kwargs["device"] = "cpu"
                    return _original(*args, **kwargs)
                stack.enter_context(mock.patch.object(torch, name, side_effect=redirect))
            original_generator, original_load = torch.Generator, torch.load
            stack.enter_context(mock.patch.object(torch, "Generator", side_effect=lambda device="cpu":
                original_generator(device="cpu")))
            def load(*args, **kwargs):
                if str(kwargs.get("map_location", "")).startswith("cuda"):
                    kwargs["map_location"] = "cpu"
                return original_load(*args, **kwargs)
            stack.enter_context(mock.patch.object(torch, "load", side_effect=load))
            stack.enter_context(mock.patch.object(torch.cuda, "get_rng_state_all", return_value=[]))
            stack.enter_context(mock.patch.object(torch.cuda, "set_rng_state_all"))
            stack.enter_context(mock.patch.object(runner, "GraphPolicy", TinyPolicy))
            stack.enter_context(mock.patch.object(runner, "setup", side_effect=setup or self.setup))
            stack.enter_context(mock.patch.object(runner, "read_replay", return_value=replay))
            stack.enter_context(mock.patch.object(runner.signal, "signal"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield

    @staticmethod
    def args(exp, round_number=0):
        return SimpleNamespace(experiment=exp, seed=0, round=round_number)

    @staticmethod
    def checkpoint(exp, round_number=0):
        return torch.load(exp/f"seed_0/round_{round_number}/checkpoint.pt", weights_only=False)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second), "checkpoint tensors differ")
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_common_initialization_frozen_offline_q_and_zero_q_bc(self):
        bc, fql = self.experiment("bc", q_weight=0.), self.experiment("fql")
        with self.cpu_fixture():
            runner.train(self.args(bc))
            runner.train(self.args(fql))
        a, b = self.checkpoint(bc), self.checkpoint(fql)
        common = torch.load(self.common, weights_only=False)
        initial_q = torch.load(self.qpath, weights_only=False)["model"]
        self.assert_nested_equal(a["flow"], common["flow"])
        self.assert_nested_equal(b["flow"], common["flow"])
        self.assert_nested_equal(a["critic"], initial_q)
        self.assert_nested_equal(b["critic"], initial_q)
        self.assertFalse(torch.equal(a["actor"]["linear.weight"], common["actor"]["linear.weight"]))
        self.assertFalse(torch.equal(a["actor"]["linear.weight"], b["actor"]["linear.weight"]))
        rows = [json.loads(line) for line in (bc/"seed_0/round_0/learning_curve.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["action_gradients"]["q_grad_norm"], 0.)
        self.assertEqual(rows[0]["parameter_gradients"]["q_grad_norm"], 0.)
        rows = [json.loads(line) for line in (fql/"seed_0/round_0/learning_curve.jsonl").read_text().splitlines()]
        self.assertGreater(rows[0]["action_gradients"]["q_grad_norm"], 0.)

    def test_released_teacher_ignores_legacy_common_flow_and_stays_frozen_online(self):
        exp = self.experiment("released", flow_policy=RELEASED, flow_steps=20,
                              flow_warmup_steps=0, flow_checkpoint_sha256=PUBLIC_SHA)
        torch.manual_seed(9876)
        public = TinyPolicy(self.problem).requires_grad_(False).eval()
        public_state = copy.deepcopy(public.state_dict())
        replay = dict(x=torch.full((4, 3, 2), .9), costs=self.costs[:4]*.9,
                      mask=self.mask[:4], success=torch.ones(4, dtype=torch.bool),
                      bindings=[dict(path="fixture", sha256="own")])
        def factory(*args):
            return copy.deepcopy(public)
        with self.cpu_fixture(), mock.patch.object(runner, "make_behavior_flow", side_effect=factory):
            runner.train(self.args(exp))
        with self.cpu_fixture(replay=replay), mock.patch.object(runner, "make_behavior_flow", side_effect=factory):
            runner.train(self.args(exp, 1))
        offline, online = self.checkpoint(exp), self.checkpoint(exp, 1)
        for saved in [offline, online]:
            self.assert_nested_equal(saved["flow"], public_state)
            self.assertFalse(saved["optimizers"]["flow"]["state"])
        self.assertFalse(torch.equal(offline["actor"]["linear.weight"], online["actor"]["linear.weight"]))
        self.assertTrue(any(not torch.equal(offline["critic"][key], value)
                            for key, value in online["critic"].items() if key.startswith("members")))

    def test_exact_resume_and_final_checkpoint_before_marker_recovery(self):
        full, interrupted = self.experiment("full"), self.experiment("interrupted")
        with self.cpu_fixture():
            runner.train(self.args(full))
            original_step = torch.optim.Adam.step
            calls = []
            def interrupt(optimizer, *args, **kwargs):
                result = original_step(optimizer, *args, **kwargs)
                calls.append(1)
                if len(calls) == 3:
                    runner.STOP = True
                return result
            with mock.patch.object(torch.optim.Adam, "step", new=interrupt), self.assertRaises(SystemExit) as error:
                runner.train(self.args(interrupted))
            self.assertEqual(error.exception.code, 85)
            runner.STOP = False
            runner.train(self.args(interrupted))
        first, second = self.checkpoint(full), self.checkpoint(interrupted)
        for key in ("flow", "actor", "critic", "optimizers", "rng", "cuda_rng", "generators", "step"):
            self.assert_nested_equal(first[key], second[key])
        marker = interrupted/"seed_0/round_0/completed.json"
        before = sha(marker.with_name("checkpoint.pt"))
        marker.unlink()
        with self.cpu_fixture():
            runner.train(self.args(interrupted))
        self.assertEqual(before, sha(marker.with_name("checkpoint.pt")))
        self.assertEqual(json.loads(marker.read_text())["checkpoint_sha256"], before)

    def test_changed_completed_and_parent_checkpoint_are_rejected(self):
        exp = self.experiment("corrupt")
        with self.cpu_fixture():
            runner.train(self.args(exp))
        checkpoint = exp/"seed_0/round_0/checkpoint.pt"
        checkpoint.write_bytes(checkpoint.read_bytes()+b"changed")
        with self.cpu_fixture(), self.assertRaisesRegex(ValueError, "Completed policy checkpoint changed"):
            runner.train(self.args(exp))
        with self.cpu_fixture(), self.assertRaisesRegex(ValueError, "parent checkpoint changed"):
            runner.train(self.args(exp, 1))

    def test_online_uses_own_balanced_feedback_and_optional_flow_updates(self):
        exp = self.experiment("online", online_flow_update=True, critic_variant="huber_local")
        with self.cpu_fixture():
            runner.train(self.args(exp))
        replay_x = torch.full((4, 3, 2), .9)
        replay = dict(x=replay_x, costs=self.costs[:4]*.9, mask=self.mask[:4],
                      success=torch.ones(4, dtype=torch.bool), bindings=[dict(path="fixture", sha256="own")])
        batches, targets = [], []
        def setup(exp, seed):
            result = list(self.setup(exp, seed))
            critic = result[6]
            original = critic.loss
            def record(x, *args, **kwargs):
                batches.append((x.detach().clone(), kwargs.copy()))
                return original(x, *args, **kwargs)
            critic.loss = record
            return tuple(result)
        original_flow_loss = TinyPolicy.flow_loss
        def flow_loss(policy, action, noise, t):
            targets.append(action.detach().clone())
            return original_flow_loss(policy, action, noise, t)
        with self.cpu_fixture(replay=replay, setup=setup), mock.patch.object(TinyPolicy, "flow_loss", new=flow_loss):
            runner.train(self.args(exp, 1))
        self.assertEqual(len(batches), self.cfg["online_updates"])
        for batch, options in batches:
            self.assertEqual(batch.shape[0], 4)
            self.assertTrue((batch[:2].abs() < .7).all())
            self.assertTrue((batch[2:] == .9).all())
            self.assertEqual(options["difference_weight"], .1)
        self.assertEqual(len(targets), self.cfg["online_updates"])
        for target in targets:
            self.assertEqual(int((target[:, 0, 0] == .9).sum()), 2)
        before, after = self.checkpoint(exp, 0), self.checkpoint(exp, 1)
        self.assertFalse(torch.equal(before["flow"]["linear.weight"], after["flow"]["linear.weight"]))
        self.assertTrue(any(not torch.equal(before["critic"][key], value) for key, value in after["critic"].items()
                            if isinstance(value, torch.Tensor) and key.startswith("members")))
        self.assert_nested_equal(before["critic"]["target_mean"], after["critic"]["target_mean"])

    def test_online_actor_noise_paired_across_q_and_flow_update_factors(self):
        replay = dict(x=torch.full((4, 3, 2), .9), costs=self.costs[:4]*.9,
                      mask=self.mask[:4], success=torch.ones(4, dtype=torch.bool),
                      bindings=[dict(path="fixture", sha256="own")])
        records = {}
        original_sample = TinyPolicy.sample_flow
        for q_weight, flow_update in ((0., False), (0., True), (1., False), (1., True)):
            with self.subTest(q_weight=q_weight, flow_update=flow_update):
                label = f"q{q_weight}_flow{flow_update}"
                exp = self.experiment(label, q_weight=q_weight, online_flow_update=flow_update)
                with self.cpu_fixture():
                    runner.train(self.args(exp))
                noises = []
                def sample(policy, noise, steps=10):
                    noises.append(noise.detach().clone())
                    return original_sample(policy, noise, steps)
                with self.cpu_fixture(replay=replay), mock.patch.object(TinyPolicy, "sample_flow", new=sample):
                    runner.train(self.args(exp, 1))
                self.assertEqual(len(noises), self.cfg["online_updates"])
                records[(q_weight, flow_update)] = dict(noises=noises, checkpoint=self.checkpoint(exp, 1))
        reference = records[(0., False)]
        for record in records.values():
            for expected, actual in zip(reference["noises"], record["noises"]):
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            torch.testing.assert_close(reference["checkpoint"]["generators"]["actor"],
                                       record["checkpoint"]["generators"]["actor"], atol=0, rtol=0)
        # Ensure this control actually changed the non-actor random streams.
        self.assertFalse(torch.equal(reference["checkpoint"]["generators"]["q"],
                                     records[(1., False)]["checkpoint"]["generators"]["q"]))
        self.assertFalse(torch.equal(reference["checkpoint"]["generators"]["flow"],
                                     records[(0., True)]["checkpoint"]["generators"]["flow"]))

    def test_online_resume_restores_all_three_advanced_generators(self):
        full = self.experiment("online_full", online_flow_update=True)
        interrupted = self.experiment("online_interrupted", online_flow_update=True)
        replay = dict(x=torch.full((4, 3, 2), .9), costs=self.costs[:4]*.9,
                      mask=self.mask[:4], success=torch.ones(4, dtype=torch.bool),
                      bindings=[dict(path="fixture", sha256="own")])
        with self.cpu_fixture():
            runner.train(self.args(full))
            runner.train(self.args(interrupted))
        original_step = torch.optim.Adam.step
        steps = []
        def interrupt_after_two_iterations(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            steps.append(1)
            # Actor and flow each use Adam once per iteration; critic uses AdamW.
            if len(steps) == 4:
                runner.STOP = True
            return result
        with self.cpu_fixture(replay=replay):
            runner.train(self.args(full, 1))
            with mock.patch.object(torch.optim.Adam, "step", new=interrupt_after_two_iterations), \
                 self.assertRaises(SystemExit) as error:
                runner.train(self.args(interrupted, 1))
            self.assertEqual(error.exception.code, 85)
            checkpoint = torch.load(interrupted/"seed_0/round_1/resume.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)
            runner.STOP = False
            runner.train(self.args(interrupted, 1))
        first, second = self.checkpoint(full, 1), self.checkpoint(interrupted, 1)
        for key in ("flow", "actor", "critic", "optimizers", "rng", "cuda_rng", "generators", "step"):
            self.assert_nested_equal(first[key], second[key])

    def request_fixture(self, exp):
        """Real request hashes and geometry-failure outcomes, no EDA fixtures."""
        cfg = json.loads((exp/"config.json").read_text())
        checkpoint = exp/"source_checkpoint.pt"
        checkpoint.write_bytes(b"frozen-source")
        dest = exp/"seed_0/round_0/requests"
        dest.mkdir(parents=True)
        requests = []
        for role, count in (("collect", cfg["online_samples"]), ("evaluation", 1)):
            for index in range(count):
                sample = dest/role/str(index)
                sample.mkdir(parents=True)
                action = np.zeros((3, 2), dtype=np.float32)+index*.01
                noise = np.ones((3, 2), dtype=np.float32)*.1
                row = dict(arm=cfg["arm"], graph="bp_fe", seed=0, round=0, role=role, index=index,
                    run_id=role+str(index), raw_action=action.tolist(), executed_action=action.tolist(),
                    source_checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), sample_dir=str(sample),
                    raw_action_sha256=runner.action_hash(action), action_sha256=runner.action_hash(action),
                    noise_sha256=runner.action_hash(noise), terminal_mask=0,
                    training_feedback_eligible=role == "collect", config_sha256=sha(exp/"config.json"))
                np.savez(sample/"actions.npz", action=action, raw_action=action, noise=noise)
                atomic_json(sample/"request.json", row)
                requests.append(row)
        tasks = dest/"tasks.jsonl"
        tasks.write_text("".join(json.dumps(row)+"\n" for row in requests))
        atomic_json(dest/"completed.json", dict(count=len(requests), tasks_sha256=sha(tasks),
                                                config_sha256=sha(exp/"config.json")))
        binding = dict(config_sha256=sha(exp/"config.json"), dataset_sha256=cfg["dataset_sha256"], reward_scale=100.)
        for row in requests:
            sample = Path(row["sample_dir"])
            out = dict(row, request_sha256=sha(sample/"request.json"), tasks_sha256=sha(tasks),
                       dataset_sha256=cfg["dataset_sha256"], status="geometry_failure", metrics=None,
                       reward=-4., failure_class="normalized_geometry")
            atomic_json(sample/"outcome.json", out)
        return cfg, binding, requests

    def test_replay_keeps_failure_metrics_missing_and_rejects_foreign_scope(self):
        exp = self.experiment("replay")
        cfg, binding, requests = self.request_fixture(exp)
        from . import qstudy_backend
        original_tensor = torch.tensor
        def tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)
        with mock.patch.object(qstudy_backend, "context", return_value=(cfg, None, binding)), \
             mock.patch.object(torch, "tensor", side_effect=tensor):
            replay = runner.read_replay(exp, 0, 1, cfg, (-1, 3, 2))
            self.assertEqual(len(replay["x"]), 2)
            self.assertTrue(torch.isnan(replay["costs"]).all())
            self.assertFalse(replay["mask"].any())
            self.assertFalse(replay["success"].any())
            path = Path(requests[0]["sample_dir"])/"outcome.json"
            original = json.loads(path.read_text())
            for field, value in (("arm", "foreign"), ("role", "evaluation"),
                                 ("training_feedback_eligible", False), ("index", 100)):
                atomic_json(path, dict(original, **{field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.read_replay(exp, 0, 1, cfg, (-1, 3, 2))
            atomic_json(path, original)
            with mock.patch.object(qstudy_backend, "checked_existing", side_effect=ValueError("native mismatch")):
                with self.assertRaisesRegex(ValueError, "native mismatch"):
                    runner.read_replay(exp, 0, 1, cfg, (-1, 3, 2))

    def test_export_fastpath_checks_task_and_checkpoint_immutability(self):
        exp = self.experiment("export")
        cfg, binding, requests = self.request_fixture(exp)
        from . import qstudy_backend
        with self.cpu_fixture(), mock.patch.object(qstudy_backend, "context", return_value=(cfg, None, binding)):
            runner.export(self.args(exp))
            source = Path(requests[0]["source_checkpoint"])
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                runner.export(self.args(exp))


if __name__ == "__main__":
    unittest.main()
