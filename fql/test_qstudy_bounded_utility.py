"""Small, CPU-only checks for prospective measured bounded target semantics."""
from copy import deepcopy
import unittest

import torch

from .qstudy_bounded_utility import (
    FAILURE_COST, METRIC_NAMES, VERSION, assert_common_scale, derive_cost_utility,
    make_spec, scale_identity, utility_identity, validate_spec,
)


class BoundedUtilityTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(split="train", dataset_sha256="a" * 64,
                              train_rows_sha256="b" * 64)
        self.spec = make_spec([1, 0, 0, 0, 0], [10, 1, 1, 1, 1], self.reference)

    def test_native_slack_conversion_and_exact_scalar_units(self):
        spec = make_spec([1, .15, .35, .1, .1], [10, 2, 4, 5, 10], self.reference)
        native = torch.tensor([[10., -2., -4., 5., 10.],
                               [0., 2., 4., 0., 0.]], dtype=torch.float64)
        result = derive_cost_utility(native, ["success"] * 2, spec)
        c = 1 + .15 + .35 + .1 + .1
        torch.testing.assert_close(result["cost"], torch.tensor([2 * c / (1 + c), 0.], dtype=native.dtype))
        torch.testing.assert_close(result["utility"], -result["cost"])
        self.assertEqual(result["cost"].dtype, torch.float64)
        self.assertEqual(result["valid"].tolist(), [True, True])
        self.assertEqual(result["success"].tolist(), [True, True])
        self.assertEqual(result["confirmed_failure"].tolist(), [False, False])

    def test_success_order_and_failure_gap_with_huge_float32(self):
        # Every value is a finite measured float32, but float32 C would overflow.
        largest = torch.finfo(torch.float32).max
        native = torch.tensor([[0., 0., 0., 0., 0.], [10., 0., 0., 0., 0.],
                               [largest, -largest, -largest, largest, largest],
                               [float("nan")] * 5])
        spec = make_spec([1e30] * 5, [1e-30] * 5, self.reference)
        result = derive_cost_utility(native, ["success"] * 3 + ["physical_failure"], spec)
        self.assertTrue(bool(torch.isfinite(result["cost"]).all()))
        self.assertTrue(bool((result["utility"][:3] > result["utility"][3]).all()))
        self.assertEqual(result["cost"].tolist(), [0., 2., 2., FAILURE_COST])
        # Even a positive float64 weighted sum overflow safely saturates at 2.
        huge64 = native[2:3].double()
        spec64 = make_spec([1e308] * 5, [1e-308] * 5, self.reference)
        result64 = derive_cost_utility(huge64, ["success"], spec64)
        self.assertTrue(bool(torch.isfinite(result64["cost"]).all()))
        self.assertEqual(result64["cost"].item(), 2.)

    def test_extreme_intermediates_do_not_change_finite_final_cost(self):
        # Multiplication overflows, although the final normalized C is ~1.176.
        cases = [
            (1e200, 2e108, 1.7e308),
            # Multiplication rounds to zero; division rescues a nonzero C.
            (1e-200, 2e-200, 1.7e-308),
            # A nonzero subnormal product also needs a precise fallback.
            (1e-200, 2e-108, 1.7e-308),
            # The final B is representable although sigmoid(log(C)) can zero.
            (1e-200, 1e-200, 1e-90),
        ]
        for cost, weight, scale in cases:
            with self.subTest(cost=cost, weight=weight, scale=scale):
                spec = make_spec([weight, 0, 0, 0, 0], [scale, 1, 1, 1, 1], self.reference)
                native = torch.tensor([[cost, 0., 0., 0., 0.]], dtype=torch.float64)
                result = derive_cost_utility(native, ["success"], spec)
                # Alternate association is finite/accurate for these fixtures.
                c = cost / scale * weight
                expected = 2 * c / (1 + c)
                self.assertGreater(result["cost"].item(), 0.)
                torch.testing.assert_close(result["cost"], torch.tensor([expected], dtype=torch.float64),
                                           rtol=1e-12, atol=0.)
        # Multiple active heads require log-SUM-exp, not a max approximation;
        # zero-cost active heads must not introduce NaN into that sum.
        spec = make_spec([2e108, 1, 1, 1, 0], [1.7e308, 1, 1, 1, 1], self.reference)
        native = torch.tensor([[1e200, -.5, 0., 0., float("nan")]], dtype=torch.float64)
        result = derive_cost_utility(native, ["success"], spec)
        c = (1e200 / 1.7e308 * 2e108) + .5
        torch.testing.assert_close(result["cost"], torch.tensor([2*c/(1+c)], dtype=torch.float64),
                                   rtol=1e-12, atol=0.)

    def test_inactive_missing_metrics_and_power_area_are_ignored(self):
        native = torch.tensor([[10., float("nan"), float("inf"), -100., float("nan"),
                                float("nan"), -20.]])
        mask = torch.tensor([[True, False, False, False, False, False, False]])
        result7 = derive_cost_utility(native, ["success"], self.spec, mask)
        result5 = derive_cost_utility(native[:, :5], ["success"], self.spec, mask[:, :5])
        self.assertEqual(result7["cost"].item(), 1.)
        for key in result7:
            torch.testing.assert_close(result7[key], result5[key])
        # A declared active missing label is never interpreted as zero.
        timing = make_spec([1, .15, .35, 0, 0], self.spec["scales"], self.reference)
        invalid = derive_cost_utility(native, ["success"], timing, mask)
        self.assertFalse(invalid["valid"].item())
        self.assertTrue(torch.isnan(invalid["cost"]).item())

    def test_status_is_required_and_only_native_failures_are_targets(self):
        statuses = ["success", "geometry_failure", "physical_failure", "infrastructure_failure",
                    "unresolved", "missing", "", "Success", "unknown"]
        result = derive_cost_utility(None, statuses, self.spec)
        self.assertEqual(result["valid"].tolist(), [False, True, True] + [False] * 6)
        self.assertEqual(result["confirmed_failure"].tolist(), [False, True, True] + [False] * 6)
        self.assertFalse(bool(result["success"].any()))
        self.assertEqual(result["cost"][1:3].tolist(), [4., 4.])
        self.assertTrue(bool(torch.isnan(result["cost"][~result["valid"]]).all()))
        # Finite metrics do not turn an infrastructure failure into a success.
        unresolved = derive_cost_utility(torch.zeros(1, 5), ["infrastructure_failure"], self.spec)
        self.assertFalse(unresolved["valid"].item())
        self.assertTrue(torch.isnan(unresolved["utility"]).item())

    def test_negative_native_costs_missing_masks_and_nonfinite_active_values(self):
        spec = make_spec([1, 0, 0, 1, 1], [1] * 5, self.reference)
        native = torch.tensor([[-1., 0., 0., 0., 0.], [1., 0., 0., -1., 0.],
                               [1., 0., 0., 0., -1.], [float("inf"), 0., 0., 0., 0.],
                               [float("nan"), 0., 0., 0., 0.], [1., 0., 0., 0., 0.]])
        mask = torch.ones_like(native, dtype=torch.bool)
        mask[-1, 0] = False
        result = derive_cost_utility(native, ["success"] * len(native), spec, mask)
        self.assertFalse(bool(result["valid"].any()))
        self.assertTrue(bool(torch.isnan(result["utility"]).all()))
        # There is deliberately no clipping of a predicted negative cost into
        # a perfect-success label and no singular rational transform at -1.
        rejected_prediction = derive_cost_utility(torch.tensor([[-10., 0., 0., 0., 0.]]),
                                                   ["success"], self.spec)
        self.assertTrue(torch.isnan(rejected_prediction["cost"]).item())

    def test_normal_success_target_gradient_and_monotone_order(self):
        native = torch.tensor([[2., 0., 0., 0., 0.], [10., 0., 0., 0., 0.],
                               [30., 0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        result = derive_cost_utility(native, ["success"] * 3, self.spec)
        self.assertTrue(bool((result["cost"][1:] > result["cost"][:-1]).all()))
        result["cost"].sum().backward()
        c = native.detach()[:, 0] / 10
        torch.testing.assert_close(native.grad[:, 0], 2 / (1 + c).square() / 10)
        torch.testing.assert_close(native.grad[:, 1:], torch.zeros(3, 4, dtype=torch.float64))

    def test_common_scale_and_full_objective_have_distinct_exact_identities(self):
        different_weights = make_spec([1, .15, .35, 0, 0], self.spec["scales"], self.reference)
        self.assertEqual(scale_identity(self.spec), scale_identity(different_weights))
        self.assertNotEqual(utility_identity(self.spec), utility_identity(different_weights))
        assert_common_scale(self.spec, different_weights)
        for key in ("scales", "reference"):
            other = deepcopy(self.spec)
            if key == "scales":
                other[key][0] += 1e-12
            else:
                other[key]["train_rows_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "common TRAIN scale"):
                assert_common_scale(self.spec, other)

    def test_spec_is_versioned_strict_and_does_not_fit_or_alias_reference(self):
        expected = deepcopy(self.spec)
        supplied = deepcopy(self.spec)
        canonical = validate_spec(supplied)
        supplied["reference"]["train_rows_sha256"] = "c" * 64
        supplied["weights"][0] = 99
        self.assertEqual(canonical, expected)
        self.assertEqual(canonical["version"], VERSION)
        self.assertEqual(canonical["metric_names"], list(METRIC_NAMES))
        mutations = [
            ("version", "bounded_terminal_utility_v2"), ("metric_names", list(reversed(METRIC_NAMES))),
            ("success_cap", 3.), ("failure_cost", 2.), ("weights", [0] * 5),
            ("weights", [-1, 0, 0, 0, 0]), ("weights", [float("nan")] * 5),
            ("weights", [float("inf")] * 5), ("weights", [True, 0, 0, 0, 0]),
            ("scales", [0, 1, 1, 1, 1]), ("scales", [float("inf")] * 5),
            ("scales", [1] * 4),
            ("reference", dict(self.reference, split="validation")),
            ("reference", dict(self.reference, dataset_sha256="")),
            ("reference", dict(self.reference, train_rows_sha256="not-a-hash")),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bad = deepcopy(expected)
                bad[key] = value
                validate_spec(bad)
        with self.assertRaises(ValueError):
            validate_spec(dict(expected, extra_semantics="unreviewed"))
        with self.assertRaises(ValueError):
            make_spec([1] * 5, [1] * 5, dict(self.reference, extra="unbound"))

    def test_input_shapes_dtypes_and_mask_contract(self):
        bad_cases = [
            (torch.zeros(5), ["success"], None),
            (torch.zeros(1, 6), ["success"], None),
            (torch.zeros(1, 5), [], None),
            (torch.zeros(1, 5), "success", None),
            (torch.zeros(1, 5), [True], None),
            (torch.zeros(1, 5), ["success"], torch.ones(1, 5)),
            (torch.zeros(1, 5), ["success"], torch.ones(1, 4, dtype=torch.bool)),
            (torch.zeros(1, 5), ["success"], torch.ones(2, 5, dtype=torch.bool)),
            (torch.zeros(1, 5, dtype=torch.complex64), ["success"], None),
        ]
        for metrics, statuses, mask in bad_cases:
            with self.subTest(shape=metrics.shape, statuses=statuses), self.assertRaises(ValueError):
                derive_cost_utility(metrics, statuses, self.spec, mask)
        integer = derive_cost_utility([[10, 0, 0, 0, 0]], ["success"], self.spec)
        self.assertEqual(integer["cost"].dtype, torch.float32)
        self.assertEqual(integer["cost"].item(), 1.)
        empty = derive_cost_utility(torch.empty(0, 7), [], self.spec)
        self.assertEqual(empty["cost"].shape, (0,))

    def test_native_metrics_and_original_reward_are_untouched(self):
        native = torch.tensor([[10., -2., -4., 5., 10., .2, 70000.]])
        before = native.clone()
        outcome = dict(status="success", reward=-1.2345)
        original = deepcopy(outcome)
        result = derive_cost_utility(native, [outcome["status"]], self.spec)
        torch.testing.assert_close(native, before)
        self.assertEqual(outcome, original)
        self.assertEqual(result["utility"].item(), -1.)


if __name__ == "__main__":
    unittest.main()
