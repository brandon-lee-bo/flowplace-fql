"""Tiny software-only tests for TRAIN-only analytic-prior reporting.

No real dataset, checkpoint, layout, label or backend run is modified. Run the
test module through a bounded Slurm CPU allocation.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from . import qstudy_prior_report as prior
from .qstudy_models import MultiMetricCritic


def fixture_cache():
    # Anisotropic canvas [224,448] makes unit/order mistakes visible.
    return dict(node_id_to_name=['a', 'b', 'c'], ratio_x=1., ratio_y=2.,
        node_info={name: dict(x=10.+i, y=12.+i) for i, name in enumerate(['a', 'b', 'c'])},
        net_info={'ab': dict(nodes={'a': dict(x_offset=3., y_offset=5.),
                                   'b': dict(x_offset=-2., y_offset=-7.)}),
                  'bc': dict(nodes={'b': dict(x_offset=1., y_offset=-2.),
                                   'c': dict(x_offset=-4., y_offset=3.)})})


class PriorReportSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run prior tests through Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_smooth_calibration_matches_zero_neural_residual_exactly(self):
        rng = np.random.default_rng(32)
        x = rng.uniform(-.8, .8, (12, 3, 2)).astype(np.float32)
        feature, _ = prior.analytic_features(fixture_cache(), x)
        costs = np.asarray(1200+150*feature['smooth_feature']+np.arange(12)*3, dtype=np.float32)
        calibration = prior.calibrate(feature['smooth_feature'], costs)
        model = MultiMetricCritic(fixture_cache(), torch.tensor(x), torch.tensor(costs[:, None]),
            [True, True, True], ['rWL_um'], hpwl_residual=True, seed=91)
        for parameter in model.parameters():
            parameter.data.zero_()
        target = rng.uniform(-1., 1., (7, 3, 2)).astype(np.float32)
        tf, _ = prior.analytic_features(fixture_cache(), target)
        with torch.no_grad():
            expected = model(torch.tensor(target)).mean(1).numpy()[:, 0]
        np.testing.assert_array_equal(prior.calibrated_prediction(calibration, tf['smooth_feature']), expected)
        self.assertEqual(calibration['slope'], float(model.baseline_slope[0]))
        self.assertEqual(calibration['intercept'], float(model.baseline_intercept[0]))
        self.assertEqual(calibration['hpwl_std'], float(model.hpwl_std))

    def test_validation_and_held_family_cannot_fit_calibration(self):
        family = np.repeat(np.arange(6), 2)
        feature = dict(exact_feature=np.arange(12, dtype=float)/10,
                       smooth_feature=np.arange(12, dtype=float)/8,
                       exact_hpwl=np.arange(12, dtype=float)*100,
                       smooth_hpwl=np.arange(12, dtype=float)*125)
        validation = {k: np.asarray([.2, .8, 1.1]) for k in feature}
        costs = np.asarray(100+np.arange(12)*20, dtype=float)
        folds = [np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4, 5])]
        _, c1, _ = prior.prior_predictions(feature, validation, costs, family, folds)
        changed = {k: v.copy() for k, v in feature.items()}
        for value in changed.values():
            value[:4] = 1e7
        changed_costs = costs.copy()
        changed_costs[:4] = 2e9
        extreme_validation = {k: v*1e10 for k, v in validation.items()}
        _, c2, _ = prior.prior_predictions(changed, extreme_validation, changed_costs, family, folds)
        self.assertEqual([c for c in c1 if c['fit'] == 'cv_0'], [c for c in c2 if c['fit'] == 'cv_0'])
        _, c3, _ = prior.prior_predictions(feature, extreme_validation, costs, family, folds)
        self.assertEqual(c1, c3)

    def test_exact_units_pin_offsets_and_raw_order(self):
        cache = fixture_cache()
        x = np.asarray([[[-.5, -.5], [.5, .5], [0., .75]],
                        [[-.2, .1], [.1, -.2], [.7, .3]]], dtype=np.float32)
        features, metadata = prior.analytic_features(cache, x)
        expected = []
        canvas = np.asarray([224., 448.])
        for layout in x:
            positions = (layout+1)*canvas/2
            a = positions[0]+[3., 5.]
            b = positions[1]+[-2., -7.]
            c = positions[1]+[1., -2.]
            d = positions[2]+[-4., 3.]
            expected.append(np.abs(a-b).sum()+np.abs(c-d).sum())
        np.testing.assert_allclose(features['exact_hpwl'], expected, rtol=2e-7)
        np.testing.assert_allclose(features['exact_feature'], np.asarray(expected)/672, rtol=2e-7)
        self.assertTrue(np.all(features['smooth_hpwl'] >= features['exact_hpwl']-1e-4))
        reordered = dict(cache, node_id_to_name=['c', 'a', 'b'])
        reordered_features, _ = prior.analytic_features(reordered, x[:, [2, 0, 1]])
        np.testing.assert_allclose(features['exact_hpwl'], reordered_features['exact_hpwl'], rtol=2e-7)
        np.testing.assert_allclose(features['smooth_hpwl'], reordered_features['smooth_hpwl'], rtol=2e-7)
        self.assertEqual(metadata['included_nets'], 2)

    def test_known_linear_native_cost_and_rank_direction(self):
        h = np.asarray([0., 1., 2., 3.])
        y = 500+200*h
        calibration = prior.calibrate(h, y, ridge=0.)
        predicted = prior.calibrated_prediction(calibration, [4., 5.])
        np.testing.assert_allclose(predicted, [1300., 1500.], rtol=2e-7)
        negative = prior.calibrate(h, 1200-100*h, ridge=0.)
        self.assertLess(negative['slope'], 0.)
        self.assertLess(prior.calibrated_prediction(negative, [3.])[0], prior.calibrated_prediction(negative, [1.])[0])

    def test_constant_feature_remains_train_mean_and_no_nan(self):
        calibration = prior.calibrate([1., 1., 1.], [10., 20., 60.])
        self.assertEqual(calibration['hpwl_std'], 1.)
        self.assertEqual(calibration['slope'], 0.)
        np.testing.assert_allclose(prior.calibrated_prediction(calibration, [-1e6, 1e6]), [30., 30.])
        with self.assertRaises(ValueError):
            prior.calibrate([1., np.nan], [1., 2.])

    def test_raw_scores_no_native_mae_and_priors_not_training_seeds(self):
        rows = []
        prior.add_metrics(rows, dataset='fixture', model='smooth_hpwl_rank_only', seed=None,
            scope='historical_validation', prediction=np.asarray([10., 20., 30., 40.]),
            truth=np.asarray([1000., 2000., 1500., 4000.]), families=np.asarray([0, 0, 1, 1]),
            reporting_scale=1000., native_prediction=False)
        for row in rows:
            self.assertIsNone(row['mae'])
            self.assertIsNone(row['normalized_mae'])
            self.assertEqual(row['independent_training_runs'], 0)
        self.assertEqual(rows[0]['family_pair_accuracy'], 1.)
        self.assertEqual(rows[0]['top_regret'], 0.)

    def test_missing_or_duplicate_oof_family_rejected(self):
        f = dict(exact_feature=np.arange(8, dtype=float), smooth_feature=np.arange(8, dtype=float),
                 exact_hpwl=np.arange(8, dtype=float), smooth_hpwl=np.arange(8, dtype=float))
        v = {k: np.asarray([2., 3.]) for k in f}
        families = np.repeat(np.arange(4), 2)
        with self.assertRaises(ValueError):
            prior.prior_predictions(f, v, np.arange(8)+1., families, [np.asarray([0]), np.asarray([1])])
        with self.assertRaises(ValueError):
            prior.prior_predictions(f, v, np.arange(8)+1., families,
                [np.asarray([0, 1]), np.asarray([1, 2]), np.asarray([3])])


if __name__ == '__main__':
    unittest.main(verbosity=2)
