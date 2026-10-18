"""Safety tests for the frozen Q-data-source experiment.

All data/native files below are tiny software fixtures in task-local scratch;
no real request, layout, measurement, checkpoint, or source plan is modified.
The small normalization test executes the real source runner and screen.fit.
Run under a bounded Slurm CPU allocation.
"""
from __future__ import annotations

import contextlib
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

from . import qstudy_screen as screen
from . import qstudy_source as source
from .qstudy_data import METRIC_NAMES, load_dataset


class SourceControlSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run source-control tests under Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        scratch = os.environ.get('TMPDIR', '/scratch/bohanli/flowplace_graph')
        self.temp = tempfile.TemporaryDirectory(prefix='qstudy-source-software-test-', dir=scratch)
        self.root = Path(self.temp.name)
        self.sources = {}
        index = 0
        for name in source.SOURCE_NAMES:
            self.sources[name] = {}
            base = 0 if name == 'old' else 5300000
            for split, families in [('train', range(base, base+32)),
                                     ('validation', range(192, 200) if name == 'old' else range(base+32, base+40))]:
                rows = []
                for family in families:
                    for variant in (0, 1):
                        rows.append(self.row(index, name, split, family, variant))
                        index += 1
                self.sources[name][split] = rows
        self.reference = dict(
            behavior_action=np.random.default_rng(101).uniform(-.8, .8, (256, 6)).astype(np.float32),
            behavior_family=np.repeat(np.arange(128, dtype=np.int64), 2))
        self.parents = {'software_fixture_only': True, 'declaration': {'fixture_sha': 'unchanged'}}
        screen.STOP = False

    def tearDown(self):
        screen.STOP = False
        self.temp.cleanup()

    @staticmethod
    def row(index, name, split, family, variant):
        raw = np.asarray([-.8+.003*index+.0002*k for k in range(6)], dtype=np.float32)
        executed = raw+.00001
        values = [1000+index, -.1-.001*index, -10-.1*index,
                  index % 5, index % 7, .1+.00001*index, 100+index]
        return dict(design='software_fixture', split=split, family=int(family), variant=variant,
            source_row=index, collection_kind='software_fixture_only', status='success',
            raw_action=raw.tolist(), executed_action=executed.tolist(),
            raw_action_sha256=source.action_sha(raw), action_sha256=source.action_sha(executed),
            metrics=dict(zip(METRIC_NAMES, values)), terminal_mask=0,
            backend=dict(design='software_fixture', num_cores='8', physical_flow_revision='fixture_revision'),
            actual_backend_geometry=None, bindings={'software_fixture_only': True},
            qstudy_source=name)

    def selection(self, arm):
        return dict(families=dict(
            old=[] if arm.startswith('T') else list(range(16 if arm.startswith('M') else 32)),
            teacher=[] if arm.startswith('O') else list(range(5300000, 5300000+(16 if arm.startswith('M') else 32)))),
            variants=[0] if 'base' in arm else [0, 1], train_requests=32 if 'base' in arm else 64)

    def export(self, arm='M64', *, rows=None, parents=None):
        selection = self.selection(arm)
        chosen = source.arm_rows(self.sources, selection) if rows is None else rows
        path = self.root/('dataset_'+arm)
        source.export_arm(path, arm, chosen, self.reference,
            self.parents if parents is None else parents, selection, 'software_fixture')
        return path

    def plan_fixture(self):
        old = self.root/'old/dataset.npz'
        old.parent.mkdir()
        old.write_bytes(b'software-fixture-only-no-metric-arrays')
        source.atomic_json(old.with_name('manifest.json'), {'design': 'software_fixture'})
        rows = self.sources['old']['train']+self.sources['old']['validation']
        source.atomic_text(old.with_name('rows.jsonl'), ''.join(json.dumps(r)+'\n' for r in rows))
        common = self.root/'common.fixture'
        common.write_bytes(b'software-common-checkpoint-binding-only')
        cache = self.root/'baseline/runtime/benchmark_cache/software_fixture.pt'
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b'software-cache-binding-only')
        teacher = self.root/'teacher'
        teacher.mkdir()
        source.atomic_json(teacher/'config.json', dict(design='software_fixture', train_families=32,
            val_families=8, noise_seed=5300000, common_state_key='flow', flow_steps=10,
            common_checkpoint=str(common), common_checkpoint_sha256=source.sha(common),
            dataset_path=str(old), dataset_sha256=source.sha(old), cache_sha256=source.sha(cache)))
        return old, teacher, common

    def test_selection_is_prelabel_deterministic_and_metric_independent(self):
        old, teacher, _ = self.plan_fixture()
        first, second, third = [self.root/name for name in ('plan_a', 'plan_b', 'plan_c')]
        for folder in (first, second, third):
            folder.mkdir()
        def create(folder):
            args = SimpleNamespace(old_dataset=old, teacher_experiment=teacher, output=folder, steps=2)
            with mock.patch.object(source, 'ROOT', self.root), \
                 mock.patch.object(source, 'load_dataset', side_effect=AssertionError('planning read numeric arrays')), \
                 contextlib.redirect_stdout(io.StringIO()):
                return source.plan(args)
        self.assertFalse((teacher/'dataset/dataset.npz').exists())
        a, b = create(first), create(second)
        self.assertEqual(a['declaration'], b['declaration'])
        self.assertTrue(a['labels_not_used_for_selection'])
        # Reorder every old row and reverse its apparent quality. Family choices
        # must remain unchanged although the input provenance fingerprint changes.
        altered = source.read_rows(old)[::-1]
        for row in altered:
            row['metrics']['rWL_um'] = 1e12-row['metrics']['rWL_um']
        source.atomic_text(old.with_name('rows.jsonl'), ''.join(json.dumps(r)+'\n' for r in altered))
        c = create(third)
        self.assertEqual(a['declaration']['selections'], c['declaration']['selections'])
        mixed = a['declaration']['selections']['M64']
        self.assertEqual({k: len(v) for k, v in mixed['families'].items()}, {'old': 16, 'teacher': 16})
        self.assertEqual(mixed['families'], a['declaration']['selections']['M32_base']['families'])
        self.assertNotEqual(a['declaration']['old_source'], c['declaration']['old_source'])

    def test_changed_parent_file_invalidates_frozen_plan(self):
        old, teacher, common = self.plan_fixture()
        folder = self.root/'plan'
        folder.mkdir()
        args = SimpleNamespace(old_dataset=old, teacher_experiment=teacher, output=folder, steps=2)
        with mock.patch.object(source, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()):
            source.plan(args)
            source.load_plan(folder)
            common.write_bytes(b'changed-parent')
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                source.load_plan(folder)

    def test_mixed_folds_are_balanced_disjoint_and_atomic_for_both_variants(self):
        for arm in source.ARMS:
            with self.subTest(arm=arm):
                rows = source.arm_rows(self.sources, self.selection(arm))
                train = [r for r in rows if r['split'] == 'train']
                families = np.asarray([r['family'] for r in train])
                origins = np.asarray([r['qstudy_source'] for r in train])
                folds = source.stratified_folds(families, origins)
                self.assertEqual(folds, source.stratified_folds(families, origins))
                self.assertEqual(len(families), self.selection(arm)['train_requests'])
                self.assertEqual(sorted(sum(folds, [])), sorted(np.unique(families).tolist()))
                for name in source.SOURCE_NAMES:
                    groups = set(families[origins == name])
                    sizes = [len(groups & set(fold)) for fold in folds]
                    self.assertLessEqual(max(sizes)-min(sizes), 1)
                for family in np.unique(families):
                    self.assertEqual(sum(family in fold for fold in folds), 1)
                    assignments = [i for i, fold in enumerate(folds) if family in fold]
                    self.assertEqual(len(assignments), 1)
                for fold in folds:
                    target = np.isin(families, fold)
                    self.assertTrue(set(families[target]).isdisjoint(set(families[~target])))
        with self.assertRaisesRegex(ValueError, 'namespace collision'):
            source.stratified_folds(np.array([0, 0]), np.array(['old', 'teacher']))

    def test_crosssplit_coordinate_and_family_collisions_are_refused(self):
        original = source.arm_rows(self.sources, self.selection('M64'))
        for key, sha_key in [('raw_action', 'raw_action_sha256'), ('executed_action', 'action_sha256')]:
            with self.subTest(coordinates=key):
                rows = copy.deepcopy(original)
                training = next(r for r in rows if r['split'] == 'train')
                validation = next(r for r in rows if r['split'] == 'validation')
                validation[key], validation[sha_key] = training[key], training[sha_key]
                with self.assertRaisesRegex(ValueError, 'Coordinate leakage'):
                    self.export(rows=rows)
                self.assertFalse((self.root/'dataset_M64').exists())
        rows = copy.deepcopy(original)
        next(r for r in rows if r['split'] == 'validation')['family'] = rows[0]['family']
        with self.assertRaisesRegex(ValueError, 'family crosses'):
            self.export(rows=rows)

    def test_failed_requests_keep_denominator_missing_mask_and_original_provenance(self):
        rows = source.arm_rows(self.sources, self.selection('M64'))
        failed = rows[0]
        failed.update(status='physical_failure', metrics={m: None for m in METRIC_NAMES}, backend=None)
        snapshot = copy.deepcopy(failed)
        folder = self.export(rows=rows)
        data, manifest = load_dataset(folder)
        self.assertEqual(len(data['train']['metrics']), 64)
        self.assertEqual(data['train']['status'][0], 'physical_failure')
        self.assertTrue(np.isnan(data['train']['metrics'][0]).all())
        self.assertFalse(data['train']['metric_mask'][0].any())
        self.assertEqual(manifest['source_counts']['train_old']['failures'], 1)
        self.assertEqual(manifest['summary']['train']['successful'], 63)
        self.assertEqual(manifest['summary']['train']['count'], 64)
        self.assertEqual(source.read_rows(folder/'dataset.npz')[0], snapshot)
        self.assertTrue(np.isfinite(data['train']['metrics'][1:]).all())

    def test_audit_rejects_failure_with_numeric_ppa(self):
        row = self.row(0, 'old', 'train', 0, 0)
        row.update(status='physical_failure', backend=None)
        raw = np.asarray(row['raw_action'], dtype=np.float32)[None]
        arrays = dict(raw_action=raw, executed_action=np.asarray(row['executed_action'], dtype=np.float32)[None],
            family=np.array([0]), variant=np.array([0]), source_row=np.array([0]),
            status=np.array(['physical_failure']), metrics=np.asarray([list(row['metrics'].values())]))
        native_manifest = dict(stage='global_route_only', not_final_table2=True, design='software_fixture')
        with mock.patch.object(source, 'load_dataset', return_value=({'train': arrays}, native_manifest)), \
             mock.patch.object(source, 'read_rows', return_value=[row]), \
             mock.patch.object(source, 'check_binding_tree'):
            with self.assertRaisesRegex(ValueError, 'Failure PPA'):
                source.audited_source(self.root/'mocked-native.npz', 'old', set())

    def test_all_six_arms_keep_exact_behavior_and_native_fields(self):
        expected = copy.deepcopy(self.reference)
        for arm in source.ARMS:
            folder = self.export(arm)
            data, manifest = load_dataset(folder)
            np.testing.assert_array_equal(data['behavior_action'], expected['behavior_action'])
            np.testing.assert_array_equal(data['behavior_family'], expected['behavior_family'])
            self.assertEqual(manifest['behavior_targets'], 256)
            self.assertEqual(manifest['split_counts'], {'train': 32 if 'base' in arm else 64, 'validation': 32})
            self.assertEqual(set(data['validation']['source']), {'old', 'teacher'})
            self.assertTrue(set(data['train']['family']).isdisjoint(set(data['validation']['family'])))
            self.assertEqual(len(manifest['metric_units']), 7)
            self.assertEqual(set(manifest['physical_flow_revisions']), {'fixture_revision'})
            self.assertGreater(manifest['summary']['train']['metrics']['rWL_um']['mean'], 0)
            if 'base' in arm:
                self.assertEqual(set(data['train']['variant']), {0})
            original = {(r['qstudy_source'], r['split'], r['family'], r['variant']): r
                        for name in self.sources.values() for split in name.values() for r in split}
            for row in source.read_rows(folder/'dataset.npz'):
                native = original[(row['qstudy_source'], row['split'], row['family'], row['variant'])]
                for key in ('metrics', 'raw_action', 'executed_action', 'bindings'):
                    self.assertEqual(row[key], native[key])
        np.testing.assert_array_equal(self.reference['behavior_action'], expected['behavior_action'])

    def test_existing_arm_rejects_changed_parent_binding(self):
        folder = self.export()
        before = source.dataset_binding(folder/'dataset.npz')
        changed = copy.deepcopy(self.parents)
        changed['declaration']['fixture_sha'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'immutable parents'):
            self.export(parents=changed)
        self.assertEqual(before, source.dataset_binding(folder/'dataset.npz'))

    def test_assembly_refuses_changing_behavior_targets_or_family_identity(self):
        declaration = dict(old_source={'dataset': {'path': 'mocked-old'}},
                           expected_teacher_dataset='mocked-teacher', design='software_fixture')
        for field in ('behavior_action', 'behavior_family'):
            teacher = copy.deepcopy(self.reference)
            teacher[field][0] += 1
            side_effect = [(self.reference, {'design': 'software_fixture'}, self.sources['old']),
                           (teacher, {'design': 'software_fixture'}, self.sources['teacher'])]
            with mock.patch.object(source, 'load_plan', return_value={'declaration': declaration}), \
                 mock.patch.object(source, 'audited_source', side_effect=side_effect):
                with self.assertRaisesRegex(ValueError, 'Changing behavior targets'):
                    source.assemble(SimpleNamespace(output=self.root))

    def test_actual_runner_fits_normalizers_and_residual_only_on_each_train_fold(self):
        cache = dict(node_id_to_name=['a', 'b', 'c'], ratio_x=10/224, ratio_y=20/224,
            node_info={name: dict(x=1., y=1.2) for name in 'abc'},
            net_info={'n': {'nodes': {'a': {'x_offset': .2}, 'b': {}, 'c': {'y_offset': -.3}}}})
        x = torch.rand((12, 3, 2), generator=torch.Generator().manual_seed(19))*1.2-.6
        costs = torch.stack((100+3*x[:, 0, 0], .4+.1*x[:, 1, 0],
                             25+x[:, 2, 1], (3*x[:, 0, 1]).clamp_min(0), (4*x[:, 1, 1]).clamp_min(0)), -1)
        families = np.repeat([0, 1, 2, 5300000, 5300001, 5300002], 2)
        names = np.repeat(['old', 'teacher'], 6)
        native = np.column_stack((costs.numpy(), np.ones((12, 2))))
        native[:, 1:3] *= -1
        validation = native[:4].copy()
        validation[:, :3] *= 1e6  # A visible sentinel if validation leaks into normalizers.
        data = dict(train=dict(raw_action=x.flatten(1).numpy(), metrics=native,
                metric_mask=np.ones_like(native, dtype=bool), family=families, source=names),
            validation=dict(raw_action=x[:4].flatten(1).numpy()+20, metrics=validation,
                metric_mask=np.ones_like(validation, dtype=bool), family=np.repeat([100, 5300032], 2),
                source=np.repeat(['old', 'teacher'], 2)))
        dataset = self.root/'train_fixture.npz'
        dataset.write_bytes(b'mocked-loader-software-fixture-only')
        source.atomic_json(dataset.with_name('manifest.json'), {'software_fixture_only': True})
        source.atomic_json(self.root/'plan.json', {'software_fixture_only': True})
        source.atomic_json(self.root/'matrix.json', {'software_fixture_only': True})
        task = dict(index=0, arm='M64', variant='huber_residual', seed=2,
                    dataset=str(dataset), output=str(self.root/'fit'))
        matrix = dict(tasks=[task], implementation=source.implementation_binding())
        d = dict(steps=2, folds=3, batch_size=8, lr=3e-4, weight_decay=1e-4,
                 cache={'path': 'mocked-cache', 'sha256': 'software-fixture'}, design='software_fixture')
        make = screen.make_model
        captures = []
        def capture(*args, **kwargs):
            model, normalization = make(*args, **kwargs)
            captures.append((args[1].clone(), args[2].clone(), copy.deepcopy(normalization)))
            return model, normalization
        problem = SimpleNamespace(cond=SimpleNamespace(is_ports=torch.zeros(3, dtype=torch.bool)))
        with mock.patch.object(source, 'load_matrix', return_value=matrix), \
             mock.patch.object(source, 'load_plan', return_value={'declaration': d}), \
             mock.patch.object(source, 'load_dataset', return_value=(data, {'design': 'software_fixture'})), \
             mock.patch('torch.load', return_value=cache), \
             mock.patch('baseline.api.load_problem', return_value=problem), \
             mock.patch.object(screen, 'make_model', side_effect=capture), \
             mock.patch.object(source.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
            source.train(SimpleNamespace(output=self.root, task_index=0))
        self.assertEqual(len(captures), 4)
        binding = source.read_json(self.root/'fit/binding.json')
        self.assertEqual(binding['family_folds'], source.stratified_folds(families, names))
        for fold, (actions, labels, normalization) in enumerate(captures):
            selected = np.ones(12, dtype=bool) if fold == 3 else ~np.isin(families, binding['family_folds'][fold])
            torch.testing.assert_close(actions, x[selected])
            torch.testing.assert_close(labels, costs[selected])
            np.testing.assert_allclose(normalization['target_mean'], costs[selected].mean(0).numpy(), rtol=1e-6)
            name = 'refit' if fold == 3 else 'cv_'+str(fold)
            saved = torch.load(self.root/('fit/'+name+'/resume.pt'), weights_only=False)
            self.assertEqual(saved['normalization'], normalization)
            self.assertEqual(saved['train_indices'], np.flatnonzero(selected).tolist())
        self.assertTrue((self.root/'fit/completed.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
