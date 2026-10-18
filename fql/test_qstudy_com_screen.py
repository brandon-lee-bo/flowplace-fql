"""Tiny real CPU fits and local COM safety tests; Slurm only, no EDA."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_com_screen as com
from . import qstudy_bounded_screen as original


class LinearCost(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def standardized(self, actions):
        return (self.weight*actions[:, 0, 0])[:, None, None]


class COMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run these tiny numerical tests through CPU Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='qstudy-com-test-', dir=os.environ['TMPDIR'])
        self.root = Path(self.temporary.name)
        cache = dict(node_id_to_name=['a', 'b', 'c'], ratio_x=10/224, ratio_y=20/224,
            node_info={n:dict(x=1., y=1.2) for n in 'abc'},
            net_info={'n':{'nodes':{'a':{'x_offset':.2}, 'b':{}, 'c':{'y_offset':-.3}}}})
        x = torch.rand((12, 3, 2), generator=torch.Generator().manual_seed(19))*1.2-.6
        native = torch.stack((100+3*x[:, 0, 0], -(.4+.1*x[:, 1, 0]), -(25+x[:, 2, 1]),
            (3*x[:, 0, 1]).clamp_min(0), (4*x[:, 1, 1]).clamp_min(0),
            torch.ones(12)*.2, torch.ones(12)*1000), -1).double().numpy()
        self.context = dict(x=x, native=native, mask=np.isfinite(native),
            executed=(x*.9).numpy().reshape(12, -1), statuses=np.full(12, 'success', dtype='<U32'),
            families=np.arange(6).repeat(2), source_ids=[f'family{i//2}_row{i}' for i in range(12)],
            movable=torch.ones(3, dtype=torch.bool), cache=cache,
            binding=dict(dataset_sha256='a'*64, manifest_sha256='b'*64, cache_sha256='c'*64,
                         source_sha256={'fixture':'d'*64}))
        fold = original.family_folds(self.context['families'])[0]
        self.train = np.asarray(fold['train_indices'])
        self.target = np.asarray(fold['target_indices'])
        com.STOP = original.STOP = False

    def tearDown(self):
        com.STOP = original.STOP = False
        com._ACTIVE_NAMESPACES.clear()
        self.temporary.cleanup()

    def args(self, name='fit', steps=6):
        return SimpleNamespace(output=self.root/name, objective='wl', seed=2,
            steps=steps, batch_size=8, lr=3e-4, weight_decay=1e-4)

    def fit(self, args, context=None, intercept=None, beta=.1, target_data=None):
        context = self.context if context is None else context
        with contextlib.redirect_stdout(io.StringIO()):
            return com.fit(args, context, 'cv_0', self.train, self.target,
                context if target_data is None else target_data, intercept=intercept,
                loss_config=dict(com.LOSS, coefficient=beta))

    def equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b), 'numerical tensor differs')
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.equal(left, right)
        else:
            self.assertEqual(a, b)

    def saved(self, args):
        return torch.load(args.output/'cv_0/resume.pt', weights_only=False)

    def test_beta_zero_exact_original_optimizer_rng_minibatches_and_predictions(self):
        baseline, contrast = self.args('original'), self.args('zero')
        first, second = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            old = original.fit(baseline, self.context, 'cv_0', self.train, self.target,
                self.context, intercept=lambda x:first.append(x.clone()))
        new = self.fit(contrast, intercept=lambda x:second.append(x.clone()), beta=0.)
        self.equal(first, second)
        for key in ['model', 'optimizer', 'rng', 'batch_rng', 'step', 'normalization', 'spec', 'utility_spec']:
            self.equal(self.saved(baseline)[key], self.saved(contrast)[key])
        for key in ['critic', 'normalization', 'constant_train_B', 'input_gradients']:
            self.equal(old[key], new[key])
        with np.load(baseline.output/'cv_0/predictions.npz') as a, np.load(contrast.output/'cv_0/predictions.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])

    def test_com_resume_matches_full_parameters_optimizer_rng_and_predictions(self):
        full, resumed = self.args('full'), self.args('resumed')
        self.fit(full)
        calls = []
        def stop_after_three(_):
            calls.append(1)
            if len(calls) == 3:
                com._stop()
        with self.assertRaises(SystemExit) as stopped:
            self.fit(resumed, intercept=stop_after_three)
        self.assertEqual(stopped.exception.code, 85)
        self.assertEqual(self.saved(resumed)['step'], 3)
        self.assertFalse(original.STOP)
        self.assertFalse(com._ACTIVE_NAMESPACES)
        com.STOP = False
        self.fit(resumed)
        for key in ['model', 'optimizer', 'rng', 'batch_rng', 'step', 'normalization', 'spec', 'utility_spec', 'binding']:
            self.equal(self.saved(full)[key], self.saved(resumed)[key])
        with np.load(full.output/'cv_0/predictions.npz') as a, np.load(resumed.output/'cv_0/predictions.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])

    def test_inner_ascent_direction_sign_detachment_and_both_outer_terms(self):
        model = LinearCost()
        actions = torch.zeros(2, 3, 2)
        movable = torch.tensor([True, True, False])
        actions[:, 2] = 2.5  # Fixed nodes need not be in actor's box.
        candidate, eligible, audit = com.adversaries(model, actions, movable)
        self.assertTrue(eligible.all())
        self.assertLess(float(candidate[0, 0, 0]), 0.)
        self.assertFalse(candidate.requires_grad)
        self.assertIsNone(model.weight.grad)
        self.assertAlmostEqual(audit['radius_rms_max'], .02, places=6)
        self.assertEqual(audit['fixed_max_abs_change'], 0.)
        self.assertGreater(audit['inner_score_gain_mean'], 0.)
        penalty = (model.standardized(actions)-model.standardized(candidate)).mean()
        derivative = torch.autograd.grad(penalty, model.weight)[0]
        expected = (actions[:, 0, 0]-candidate[:, 0, 0]).mean()
        torch.testing.assert_close(derivative, expected)
        self.assertGreater(float(derivative), 0.)
        # Descending this loss reduces the optimism slope; ascending would have the wrong sign.
        self.assertLess(float(model.weight-.1*derivative), float(model.weight))

    def test_outside_anchors_skipped_without_clamp_and_boundary_fixed_checks(self):
        actions = torch.zeros(3, 3, 2)
        actions[0, 0, 0] = 1.2
        actions[1, 0, 0] = -1.
        actions[:, 2] = 2.5
        movable = torch.tensor([True, True, False])
        candidate, eligible, audit = com.adversaries(LinearCost(), actions, movable)
        np.testing.assert_array_equal(eligible, [False, True, True])
        torch.testing.assert_close(candidate[0], actions[0], rtol=0, atol=0)
        torch.testing.assert_close(candidate[:, 2], actions[:, 2], rtol=0, atol=0)
        self.assertEqual(float(candidate[1, 0, 0]), -1.)
        self.assertEqual(audit['box_excess_max'], 0.)
        self.assertGreater(audit['visited_box_clip_rows'], 0)
        self.assertLessEqual(audit['radius_rms_max'], .020001)

    def test_zero_gradient_retains_anchor_and_noeligible_keeps_full_supervision(self):
        actions = torch.zeros(2, 3, 2)
        candidate, _, audit = com.adversaries(LinearCost(0.), actions, self.context['movable'])
        torch.testing.assert_close(candidate, actions, rtol=0, atol=0)
        self.assertEqual(audit['selected_step_mean'], 0.)
        context = copy.deepcopy(self.context)
        context['x'][:, 0, 0] = 1.2
        done = self.fit(self.args(steps=2), context)
        self.assertEqual(done['admitted_train_rows'], len(self.train))
        self.assertEqual(done['com_audit']['eligible_COM_rows'], 0)
        self.assertEqual(done['com_audit']['skipped_COM_outside_center_box'], len(self.train))
        record = json.loads((self.args().output/'cv_0/learning_curve.jsonl').read_text().splitlines()[-1])
        self.assertEqual(record['parts']['com_weighted_penalty'], 0.)
        self.assertEqual(record['parts']['com_weighted_parameter_grad_norm'], 0.)

    def test_private_function_keeps_frozen_code_globals_config_and_stop_unchanged(self):
        old_class, old_sources = original.MultiMetricCritic, original.SOURCE_NAMES
        old_config = copy.deepcopy(original.CONFIG)
        with mock.patch.object(com, 'FunctionType', wraps=FunctionType) as factory:
            self.fit(self.args(steps=2))
        call = factory.call_args.args
        self.assertIs(call[0], original.fit.__code__)
        self.assertIsNot(call[1], original.fit.__globals__)
        self.assertIsNot(call[1]['MultiMetricCritic'], old_class)
        self.assertIs(original.MultiMetricCritic, old_class)
        self.assertIs(original.SOURCE_NAMES, old_sources)
        self.assertEqual(original.CONFIG, old_config)
        self.assertFalse(original.STOP)
        self.assertFalse(com._ACTIVE_NAMESPACES)

    def test_heldout_labels_do_not_change_model_spec_normalization_or_native_evidence(self):
        first, second = self.args('original', 3), self.args('heldout', 3)
        a = self.fit(first)
        revised = copy.deepcopy(self.context)
        revised['native'][self.target] *= 1000
        b = self.fit(second, revised)
        self.equal(self.saved(first)['model'], self.saved(second)['model'])
        self.assertEqual(a['normalization'], b['normalization'])
        self.assertEqual(a['utility_spec'], b['utility_spec'])
        with np.load(second.output/'cv_0/predictions.npz') as saved:
            np.testing.assert_array_equal(saved['native_metrics'], revised['native'][self.target])
            self.assertEqual(saved['native_metrics'].shape, (len(self.target), 7))
            np.testing.assert_array_equal(saved['raw_action'], revised['x'][self.target])
        record = json.loads((first.output/'cv_0/learning_curve.jsonl').read_text().splitlines()[-1])
        for key in ['com_penalty', 'com_regression_parameter_grad_norm', 'com_weighted_parameter_grad_norm',
                    'com_gradient_ratio', 'com_eligible_rows', 'com_radius_rms_max']:
            self.assertTrue(np.isfinite(record['parts'][key]))

    def test_completed_gap_audit_resume_and_artifact_tamper_refusal(self):
        args = self.args(steps=2)
        result = self.fit(args)
        completed = args.output/'cv_0/completed.json'
        no_audit = dict(result)
        del no_audit['com_audit']
        completed.write_text(json.dumps(no_audit))
        reused = self.fit(args)
        self.assertEqual(reused['com_audit'], result['com_audit'])
        self.assertEqual(reused['checkpoint_sha256'], result['checkpoint_sha256'])
        corrupted = dict(reused, com_audit={})
        completed.write_text(json.dumps(corrupted))
        with self.assertRaisesRegex(ValueError, 'coverage audit changed'):
            self.fit(args)
        completed.write_text(json.dumps(reused))
        prediction = args.output/'cv_0/predictions.npz'
        prediction.write_bytes(prediction.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.fit(args)

    def test_changed_source_binding_resume_and_undeclared_loss_refused(self):
        args = self.args()
        with self.assertRaises(SystemExit):
            self.fit(args, intercept=lambda _:com._stop())
        com.STOP = False
        changed = copy.deepcopy(self.context)
        changed['binding']['source_sha256']['fixture'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            self.fit(args, changed)
        with self.assertRaisesRegex(ValueError, 'hyperparameter change'):
            com.fit(self.args('other'), self.context, 'cv_0', self.train, self.target,
                self.context, loss_config=dict(com.LOSS, radius_rms=.1))

    def test_control_binds_all_existing_twelve_artifacts_and_rejects_incomplete(self):
        folds = original.family_folds(self.context['families'])
        folds += [dict(name='refit', train_indices=list(range(12)), target_indices=list(range(4)))]
        inputs = dict(dataset_sha256='a'*64)
        path = self.root/'control_plan.json'
        path.write_text('{}')
        plan = dict(inputs=inputs, folds=folds, output=str(self.root/'control'), config=original.CONFIG,
            tasks=[dict(index=s, objective='wl', seed=s) for s in range(3)])
        for task in plan['tasks']:
            for fold in folds:
                folder = Path(plan['output'])/'wl'/f"seed_{task['seed']}"/fold['name']
                folder.mkdir(parents=True)
                for name in ('completed.json', 'resume.pt', 'predictions.npz'):
                    (folder/name).write_text('{}')
        with mock.patch.object(com.control_report, 'checked_plan', return_value=(plan, {})), \
             mock.patch.object(com.control_report, 'verify_fit') as verify:
            evidence = com.checked_control(path, inputs, folds)
            self.assertEqual(verify.call_count, 12)
            self.assertEqual(len(evidence['artifacts']), 12)
            self.assertEqual(evidence['prediction_or_optimization_calls'], 0)
            first = Path(next(iter(evidence['artifacts'][0]['files'])))
            first.write_text('{"changed":true}')
            self.assertNotEqual(evidence, com.checked_control(path, inputs, folds))
            first.unlink()
            with self.assertRaises(FileNotFoundError):
                com.checked_control(path, inputs, folds)
            with self.assertRaisesRegex(ValueError, 'inputs/folds differ'):
                com.checked_control(path, {}, folds)

    def test_predeclared_budget_source_snapshot_plan_and_run_guard(self):
        def expanded(count, offset):
            result = {}
            for key, value in self.context.items():
                if key in ['cache', 'movable', 'binding']:
                    continue
                ix = np.arange(count) % 12
                result[key] = value[ix] if isinstance(value, (np.ndarray, torch.Tensor)) else [value[i] for i in ix]
            result['families'] = np.arange(count)//2+offset
            result['source_ids'] = [f'source{offset}_row{i}' for i in range(count)]
            return result
        data = dict(train=expanded(384, 0), validation=expanded(16, 1000))
        inputs = dict(dataset_path=str(self.root/'data.npz'))
        control = dict(plan_path=str(self.root/'old_plan.json'), plan_sha256='f'*64)
        args = SimpleNamespace(dataset=self.root/'data.npz', output=self.root/'new', control_plan=self.root/'old_plan.json')
        with mock.patch.object(com.supervised, 'load_inputs', return_value=(data, {'design':'bp_fe'}, inputs)), \
             mock.patch.object(com, 'checked_control', return_value=control), contextlib.redirect_stdout(io.StringIO()):
            com.prepare(args)
            path = args.output/'plan.json'
            plan = json.loads(path.read_text())
            self.assertEqual(plan['maximum_new_fits'], 12)
            self.assertEqual(len(plan['tasks']), 3)
            self.assertEqual(plan['supervised_control'], control)
            self.assertEqual(set(plan['source_sha256']), set(com.SOURCE_NAMES))
            for name, digest in plan['source_sha256'].items():
                self.assertEqual(com.sha(args.output/'source_snapshot'/name), digest)
            com.prepare(args)
            com.checked_control.return_value = dict(control, plan_sha256='a'*64)
            with self.assertRaisesRegex(ValueError, 'Immutable COM plan changed'):
                com.prepare(args)
        run_args = SimpleNamespace(plan=path, plan_sha256='0'*64, index=0)
        with self.assertRaisesRegex(ValueError, 'plan SHA changed'):
            com.run(run_args)
        plan['config']['loss']['coefficient'] = .2
        path.write_text(json.dumps(plan))
        run_args.plan_sha256 = com.sha(path)
        with self.assertRaisesRegex(ValueError, 'Undeclared COM stage'):
            com.run(run_args)


if __name__ == '__main__':
    unittest.main()
