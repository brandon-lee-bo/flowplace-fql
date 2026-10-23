"""Teacher contract tests; real sampler equivalence is covered by the GPU smoke."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from . import released_flowplace_teacher as module
from .flow_policy_config import RELEASED, LEGACY, PUBLIC_SHA, STEPS, validate_teacher_config
from tools.prepare_released_flowplace import prepare


def config(**overrides):
    return dict(dict(flow_policy=RELEASED, flow_steps=STEPS, flow_warmup_steps=0,
                     flow_checkpoint_sha256=PUBLIC_SHA, online_flow_update=False), **overrides)


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Run these tests under Slurm")
        torch.set_num_threads(1)

    def setUp(self):
        self.problem = SimpleNamespace(x=torch.tensor([[0., 0.], [.2, .3], [0., 0.]]),
            cond=SimpleNamespace(is_ports=torch.tensor([False, True, False])))
        self.loader = mock.patch.object(module, "load_model", side_effect=lambda *a, **kw: (nn.Linear(2, 2), None))
        self.checksum = mock.patch.object(module, "checkpoint_sha", return_value=PUBLIC_SHA)
        self.load = self.loader.start()
        self.checksum.start()
        self.addCleanup(self.loader.stop)
        self.addCleanup(self.checksum.stop)

    def test_public_model_options_and_serial_same_noise_sampling(self):
        teacher = module.make_behavior_flow(self.problem, config(), "cpu")
        self.load.assert_called_once_with(self.problem, device="cpu", steps=20,
                                         hard_constraint=True, w_reg=.1, w_dist=1.)
        z = torch.rand(3, 3, 2)
        calls = []
        def sample(model, problem, noise):
            calls.append(noise.clone())
            result = noise+3  # Detect accidental extra clamp/projection.
            result[1] = problem.x[1]
            return result, noise, []
        with mock.patch.object(module, "generate", side_effect=sample):
            result = teacher.sample_flow(z)
        torch.testing.assert_close(torch.stack(calls), z, atol=0, rtol=0)
        torch.testing.assert_close(result[:, teacher.movable], (z+3)[:, teacher.movable], atol=0, rtol=0)
        self.assertFalse(result.requires_grad)
        teacher.train()
        self.assertFalse(teacher.training)
        self.assertFalse(teacher.model.training)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            teacher.flow_loss(None)

    def test_modified_weights_and_legacy_checkpoint_cannot_replace_teacher(self):
        teacher = module.make_behavior_flow(self.problem, config(), "cpu")
        original = copy.deepcopy(teacher.state_dict())
        teacher.load_state_dict(original)
        modified = copy.deepcopy(original)
        modified["model.weight"][0, 0] += 1
        for state in [modified, {}, {"legacy.weight": torch.zeros(2)}]:
            with self.assertRaisesRegex(ValueError, "Cannot replace"):
                teacher.load_state_dict(state)
        for name, value in original.items():
            self.assertTrue(torch.equal(value, teacher.state_dict()[name]))
        with mock.patch.object(module, "checkpoint_sha", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "pinned release"):
                module.make_behavior_flow(self.problem, config(), "cpu")

    def test_incompatible_configuration_is_rejected(self):
        validate_teacher_config(config())
        for override in [dict(flow_steps=10), dict(flow_warmup_steps=1),
                         dict(online_flow_update=True), dict(flow_checkpoint_sha256="wrong"),
                         dict(flow_policy="typo")]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                module.make_behavior_flow(self.problem, config(**override), "cpu")
        with mock.patch("fql.graph_o2o_models.GraphPolicy") as legacy:
            module.make_behavior_flow(self.problem, {}, "cpu")
            module.make_behavior_flow(self.problem, dict(flow_policy=LEGACY), "cpu")
            self.assertEqual(legacy.call_count, 2)

    def test_invalid_inputs_and_sampler_contract_violations_fail(self):
        teacher = module.make_behavior_flow(self.problem, config(), "cpu")
        for noise, steps in [(torch.zeros(1, 3, 2), 10), (torch.zeros(0, 3, 2), 20),
                             (torch.zeros(3, 2), 20), (torch.full((1, 3, 2), float("nan")), 20)]:
            with self.subTest(shape=noise.shape, steps=steps), self.assertRaises(ValueError):
                teacher.sample_flow(noise, steps)
        z = torch.zeros(1, 3, 2)
        for raw, used in [(z[0], z[0]), (self.problem.x, z[0]+1),
                          (torch.full_like(z[0], float("nan")), z[0])]:
            with mock.patch.object(module, "generate", return_value=(raw, used, [])):
                with self.assertRaises(ValueError):
                    teacher.sample_flow(z)

    def test_fresh_config_preserves_source_and_cannot_reuse_experiment(self):
        import json
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            root = Path(directory)
            source = root/"old.json"
            original = dict(design="bp_fe", flow_steps=10, flow_warmup_steps=300,
                            actor_warmup_steps=800, offline_steps=1000, alpha=10)
            source.write_text(json.dumps(original))
            cfg = prepare(source, root/"new", "released_test")
            validate_teacher_config(cfg)
            self.assertEqual(cfg["alpha"], 10)
            self.assertEqual(cfg["actor_warmup_steps"], 800)
            self.assertEqual(json.loads(source.read_text()), original)
            with self.assertRaises(FileExistsError):
                prepare(source, root/"new", "released_test")
            with self.assertRaises(ValueError):
                prepare(source, root/"other", "go2o_v1")


@unittest.skipUnless(os.environ.get("FQL_REAL_CPU_SAMPLER") == "1",
                     "Opt-in real-weight CPU sampler check")
class RealCPUSamplerTests(unittest.TestCase):
    def test_original_sampler_equivalence_and_real_actor_gradient(self):
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError("Real model checks require Slurm")
        from baseline.api import load_problem, load_model, generate
        from .graph_o2o_models import GraphPolicy, actor_objective
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261022)
        problem = load_problem("bp_fe", "cpu")
        teacher = module.make_behavior_flow(problem, config(), "cpu")
        initial = copy.deepcopy(teacher.state_dict())
        noise = 2*torch.rand(2, len(problem.x), 2)-1
        target = teacher.sample_flow(noise, 20)
        original, _ = load_model(problem, device="cpu", steps=20, hard_constraint=True)
        expected = torch.stack([generate(original, problem, noise=row)[0] for row in noise])
        torch.testing.assert_close(target, expected, atol=0, rtol=0)
        actor = GraphPolicy(problem, "cpu")
        critic = nn.Sequential(nn.Flatten(), nn.Linear(len(problem.x)*2, 2))
        before = copy.deepcopy(actor.state_dict())
        loss, _, _ = actor_objective(actor, teacher, critic, noise, flow_steps=20)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in actor.parameters()))
        self.assertFalse(any(p.grad is not None for p in teacher.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, actor.state_dict()[k]) for k, v in before.items()))
        for key, value in initial.items():
            self.assertTrue(torch.equal(value, teacher.state_dict()[key]))
        teacher.load_state_dict(initial)


if __name__ == "__main__":
    unittest.main()
