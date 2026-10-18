import copy
import statistics
import unittest

from analysis.qstudy_policy_curves import METRICS, extract


def fixture():
    bindings, rounds = {}, []
    for arm, offset in (("bc", 0), ("fql", -10)):
        experiment = "/data/" + arm
        bindings[experiment] = dict(config=dict(arm=arm, seeds=[0, 1, 2], evaluation_samples=8), report_round_endpoint=1)
        for seed in range(3):
            for original_round in range(2):
                rows = [dict(role="evaluation", state="success", noise_sha256=str(index),
                    metrics={metric:100 + offset + seed + original_round + index for metric in METRICS}) for index in range(8)]
                summary = dict(planned=8, success=8, fully_accounted=True, metrics_conditional_on_success=False,
                    successful_native_metric_means={metric:statistics.fmean(r["metrics"][metric] for r in rows) for metric in METRICS})
                rounds.append(dict(experiment=experiment, arm=arm, seed=seed, round=original_round,
                    status="complete", roles=dict(evaluation=summary), rows=rows))
    return dict(stage="global_route_only", status="complete", experiment_bindings=bindings, rounds=rounds)


class PolicyCurveTest(unittest.TestCase):
    def test_equal_seed_means_and_sample_sd(self):
        result = extract(fixture())
        row = next(r for r in result["curves"] if r["arm"] == "bc" and r["round"] == 0 and r["metric"] == METRICS[0])
        self.assertEqual(row["mean"], 104.5)
        self.assertEqual(row["seed_sample_sd"], 1)
        self.assertEqual(len(result["curves"]), 28)
        self.assertEqual(len(result["per_seed"]), 12)

    def assert_rejected(self, mutate):
        report = copy.deepcopy(fixture()); mutate(report)
        with self.assertRaises(ValueError): extract(report)

    def test_missing_training_seed(self):
        self.assert_rejected(lambda r:r["rounds"].pop())

    def test_duplicate_policy_round(self):
        self.assert_rejected(lambda r:r["rounds"].append(r["rounds"][0]))

    def test_noise_mismatch(self):
        self.assert_rejected(lambda r:r["rounds"][0]["rows"][0].update(noise_sha256="different"))

    def test_physical_failure_not_dropped(self):
        self.assert_rejected(lambda r:r["rounds"][0]["rows"][0].update(state="physical_failure"))

    def test_conditional_mean_not_used(self):
        self.assert_rejected(lambda r:r["rounds"][0]["roles"]["evaluation"].update(metrics_conditional_on_success=True))

    def test_saved_summary_disagrees(self):
        self.assert_rejected(lambda r:r["rounds"][0]["roles"]["evaluation"]["successful_native_metric_means"].update(rWL_um=0))

    def test_nonfinite_native_metric(self):
        self.assert_rejected(lambda r:r["rounds"][0]["rows"][0]["metrics"].update(WNS_ns=float("nan")))

    def test_no_partial_report_or_fake_seed_count(self):
        self.assert_rejected(lambda r:r.update(status="partial_or_integrity_invalid"))
        self.assert_rejected(lambda r:r["experiment_bindings"]["/data/bc"]["config"].update(seeds=[0, 1]))


if __name__ == "__main__": unittest.main()
