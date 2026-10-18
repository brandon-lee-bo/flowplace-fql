"""Synthetic selection/audit tests; no model, native run or learning curve."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import qstudy_selection_report as m


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def rows():
        return [dict(index=i, state="success", predicted_q=float(i), macro_hpwl=float(8-i),
            common_utility=-(100.+i)/100, noise_sha256="noise"+str(i),
            metrics={name: (100.+i if name == "rWL_um" else -1. if name in ("WNS_ns", "TNS_ns") else 1.)
                     for name in m.METRICS}) for i in range(8)]

    def test_rules_keep_distinct_fixed_proxy_q_and_backend_choices(self):
        chosen = m.select_candidates(self.rows())
        self.assertEqual({k:v["index"] for k,v in chosen.items()}, dict(fixed_index0=0,
            min_executed_macro_hpwl=7, max_recorded_current_q=7, oracle_min_grt_wl=0,
            oracle_max_common_utility=0))

    def test_no_backend_failure_filter_or_fallback_for_free_selectors(self):
        rows = self.rows()
        rows[7].update(state="physical_failure", metrics=None, common_utility=-4.)
        chosen = m.select_candidates(rows)
        self.assertEqual(chosen["min_executed_macro_hpwl"]["index"], 7)
        self.assertEqual(chosen["max_recorded_current_q"]["index"], 7)
        self.assertIsNone(chosen["max_recorded_current_q"]["metrics"])
        self.assertEqual(chosen["oracle_min_grt_wl"]["index"], 0)

    def test_oracle_failure_and_utility_conventions_are_explicit(self):
        rows = self.rows()
        for row in rows:
            row.update(state="geometry_failure", metrics=None, common_utility=-4.)
        chosen = m.select_candidates(rows)
        self.assertEqual(chosen["oracle_min_grt_wl"]["index"], 0)
        self.assertIsNone(chosen["oracle_min_grt_wl"]["metrics"])
        # Fixed -4 utility may outrank an extremely poor but successful design.
        rows[1].update(state="success", metrics={name:600. for name in m.METRICS}, common_utility=-6.)
        chosen = m.select_candidates(rows)
        self.assertEqual(chosen["oracle_min_grt_wl"]["index"], 1)
        self.assertEqual(chosen["oracle_max_common_utility"]["index"], 0)

    def test_exact_ties_choose_lowest_declared_index(self):
        rows = self.rows()
        for row in rows:
            row.update(predicted_q=1., macro_hpwl=1., common_utility=-1.)
            row["metrics"]["rWL_um"] = 100.
        self.assertTrue(all(row["index"] == 0 for row in m.select_candidates(rows[::-1]).values()))

    def test_partial_duplicate_unaccounted_or_nonfinite_pool_refused(self):
        variants = [self.rows()[:7], self.rows(), self.rows(), self.rows()]
        variants[1][7]["index"] = 0
        variants[2][7]["state"] = "missing"
        variants[3][7]["predicted_q"] = float("nan")
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                m.select_candidates(rows)

    def test_common_utility_uses_fixed_reference_train_scale(self):
        rows = self.rows()
        weights, scales = [1, 0, 0, 0, 0], [200., 1., 1., 1., 1.]
        self.assertEqual(m.common_utility(rows[0]["metrics"], "success", weights, scales), -.5)
        self.assertEqual(m.common_utility(None, "geometry_failure", weights, scales), -4.)
        with self.assertRaises(ValueError):
            m.common_utility(None, "missing", weights, scales)

    def test_selected_aggregation_has_three_seed_replicates(self):
        selected = []
        for seed in (0, 1, 2):
            row = self.rows()[0]
            row.update(experiment="a", seed=seed, round=0, rule=m.RULES[0], common_utility=float(seed))
            row["metrics"]["rWL_um"] = 100.+10*seed
            selected.append(row)
        result = m.aggregate_selected(selected, {"a":{}}, {"a":0})[0]
        self.assertEqual(result["full_three_seed_summary"]["native_full"]["rWL_um"], dict(n=3, mean=110., sd=10.))
        self.assertEqual(result["full_three_seed_summary"]["utility"], dict(n=3, mean=1., sd=1.))
        selected[-1].update(state="physical_failure", metrics=None, common_utility=-4.)
        result = m.aggregate_selected(selected, {"a":{}}, {"a":0})[0]
        self.assertIsNone(result["full_three_seed_summary"]["native_full"])
        self.assertEqual(result["full_three_seed_summary"]["utility"]["n"], 3)
        result = m.aggregate_selected(selected[:2], {"a":{}}, {"a":0})[0]
        self.assertIsNone(result["full_three_seed_summary"])

    def test_policy_mean_averages_noise_before_three_seed_statistics(self):
        pools = []
        for seed in (0, 1, 2):
            rows = self.rows()
            for row in rows:
                row["metrics"]["rWL_um"] += 10*seed
            pools.append(dict(experiment="a", seed=seed, round=0, state="complete", policy_pool_mean=m.pool_mean(rows)))
        result = m.aggregate_pool_means(pools, {"a":{}}, {"a":0})[0]
        self.assertEqual(result["native_full"]["rWL_um"], dict(n=3, mean=113.5, sd=10.))

    def test_pairs_match_entire_pool_not_selected_noise(self):
        bindings = {exp:dict(config=dict(design="bp_fe", dataset_sha256="data", cache_sha256="cache")) for exp in ("a", "b")}
        pools = [dict(experiment=exp, seed=0, round=0, state="complete", noise_pool=list(range(8))) for exp in ("a", "b")]
        selected = []
        for exp, index in (("a", 0), ("b", 7)):
            for rule in m.RULES:
                selected.append(dict(self.rows()[index], experiment=exp, seed=0, round=0, rule=rule))
        result = m.paired_selection(pools, selected, "a", bindings)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["selected_index"], 7)
        self.assertEqual(result[0]["reference_selected_index"], 0)
        pools[1]["noise_pool"] = list(reversed(range(8)))
        with self.assertRaisesRegex(ValueError, "eight-noise"):
            m.paired_selection(pools, selected, "a", bindings)

    def fixture(self):
        exp = self.root / "actor"
        exp.mkdir()
        initial = self.root / "initial.pt"
        initial.write_bytes(b"not a model; hashed fixture only")
        cfg = dict(arm="test", design="bp_fe", evaluation_noise_seed=123,
            common_checkpoints={"0":str(initial)}, critic_checkpoints={"0":str(initial)},
            common_checkpoint_sha256={"0":m.sha(initial)}, critic_checkpoint_sha256={"0":m.sha(initial)})
        training = dict(completed=True, checkpoint_sha256="checkpoint", completion_sha256="done", binding_sha256="binding", steps=1)
        folder = exp / "seed_0/round_0/requests"
        folder.mkdir(parents=True)
        tasks = folder / "tasks.jsonl"
        requests, saved = [], []
        for i in range(8):
            sample = folder / "test/evaluation" / str(i)
            sample.mkdir(parents=True)
            row = dict(arm="test", seed=0, round=0, role="evaluation", index=i, graph="bp_fe",
                sample_dir=str(sample), checkpoint_sha256="checkpoint", noise_sha256="noise"+str(i), noise_seed=123,
                raw_action_sha256="raw", action_sha256="executed", predicted_q=[float(i)], macro_hpwl=100.+i,
                training_feedback_eligible=False, evaluation_purpose="development_validation")
            out = dict(row, status="geometry_failure", metrics=None)
            (sample / "request.json").write_text(json.dumps(row))
            (sample / "outcome.json").write_text(json.dumps(out))
            (sample / "actions.npz").write_bytes(b"arrays mocked; identity still checked")
            old = dict(row, predicted_q=float(i), state="geometry_failure", metrics=None, native={},
                request_sha256=m.sha(sample / "request.json"), outcome_sha256=m.sha(sample / "outcome.json"),
                actions_npz_sha256=m.sha(sample / "actions.npz"))
            requests.append(row); saved.append(old)
        tasks.write_text("".join(json.dumps(row)+"\n" for row in requests))
        (folder / "completed.json").write_text(json.dumps(dict(selection="all predetermined noises; no Q-selection", evaluation_never_replayed=True)))
        record = dict(experiment=str(exp), seed=0, round=0, status="complete", errors=[], rows=saved,
            training=training, tasks_sha256=m.sha(tasks), export_completion_sha256=m.sha(folder / "completed.json"))
        binding = dict(config=cfg, config_sha256="cfg", backend_binding={})
        return record, binding, requests, tasks, training

    def test_audit_revalidates_bindings_without_reading_learning_curve(self):
        record, binding, requests, tasks, training = self.fixture()
        with patch.object(m, "check_training", return_value=training) as check, \
                patch.object(m, "load_requests", return_value=(requests, tasks)), \
                patch.object(m, "checked_request", side_effect=lambda row,*args:Path(row["sample_dir"])), \
                patch.object(m, "checked_existing") as native:
            rows = m.audit_pool(record, binding, (binding["config"], None, {}), [1,0,0,0,0], [100,1,1,1,1])
            self.assertEqual(len(rows), 8)
            self.assertEqual(native.call_count, 8)
            self.assertEqual(check.call_args.kwargs["curve_snapshot"], {})

    def test_changed_request_and_missing_geometry_never_reach_native_validator(self):
        for mode in ("request", "geometry"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.root) as temp:
                old_root, self.root = self.root, Path(temp)
                record, binding, requests, tasks, training = self.fixture()
                self.root = old_root
                sample = Path(requests[0]["sample_dir"])
                if mode == "request":
                    (sample / "request.json").write_text("changed")
                else:
                    run = Path(temp) / "grt"
                    run.mkdir()
                    (run / "manifest.json").write_text(json.dumps(dict(success=True)))
                    out = m.read(sample / "outcome.json")
                    out["physical_run"] = str(run)
                    (sample / "outcome.json").write_text(json.dumps(out))
                    record["rows"][0]["outcome_sha256"] = m.sha(sample / "outcome.json")
                with patch.object(m, "check_training", return_value=training), \
                        patch.object(m, "load_requests", return_value=(requests, tasks)), \
                        patch.object(m, "checked_request", side_effect=lambda row,*args:Path(row["sample_dir"])), \
                        patch.object(m, "checked_existing") as native:
                    with self.assertRaises(ValueError):
                        m.audit_pool(record, binding, (binding["config"], None, {}), [1,0,0,0,0], [100,1,1,1,1])
                    native.assert_not_called()

    def test_parent_snapshot_sha_and_source_mutations_refused(self):
        code = self.root / "code"
        source = code / "source.py"
        code.mkdir(); source.write_text("original")
        exp = self.root / "exp"
        exp.mkdir()
        cfg = dict(seeds=[0,1,2], evaluation_samples=8, physical_failure_reward=-4)
        (exp / "config.json").write_text(json.dumps(cfg))
        report = dict(stage="global_route_only", not_final_table2=True, source_sha256={"source.py":m.sha(source)},
            experiment_bindings={str(exp):dict(config=cfg, config_sha256=m.sha(exp / "config.json"))}, rounds=[])
        path = self.root / "report.json"
        path.write_text(json.dumps(report))
        digest = m.sha(path)
        (self.root / "completed.json").write_text(json.dumps(dict(status="completed", artifacts={"report.json":digest})))
        with patch.object(m, "ROOT", code):
            m.audit_snapshot(path, digest)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                m.audit_snapshot(path, digest)
        with self.assertRaisesRegex(ValueError, "report SHA"):
            m.audit_snapshot(path, "wrong")


if __name__ == "__main__":
    unittest.main()
