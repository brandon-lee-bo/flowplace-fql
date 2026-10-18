"""Fixed-denominator metadata, tamper and generic native-route interface tests."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fql import qstudy_final_feedback_route as route
from fql.graph_o2o_route_controller import requested, physical_run_id, route_cap


class FinalFeedbackRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.actors = self.root/'actors'
        self.output = self.root/'routes'
        self.actors.mkdir()
        (self.actors/'preparation.json').write_text('{}')
        for objective in route.actor.OBJECTIVES:
            arm = route.actor.ARMS[objective][0]
            (self.actors/arm).mkdir()
            (self.actors/arm/'config.json').write_text(json.dumps(self.cfg(objective)))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(route.actor, 'validate_config', side_effect=
            lambda exp, seed, r: route.read(Path(exp)/'config.json')))
        self.stack.enter_context(patch.object(route, 'sources', return_value={'frozen.py':'a'*64}))

    @staticmethod
    def cfg(objective):
        return dict(final_feedback_objective=objective,
            final_feedback_evaluation=copy.deepcopy(route.actor.FINAL_EVALUATION),
            evaluation_purpose='development_validation', backend_threads=8,
            backend_storage='scratch')

    def plan(self):
        return route.build_plan(self.output, self.actors)

    def publish(self):
        plan = self.plan()
        self.output.mkdir()
        route.native.freeze(self.output/'full_route_plan.json', plan)
        route.native.freeze(self.output/'route_request.json', dict(schema=route.SCHEMA,
            plan=route.bind(self.output/'full_route_plan.json'), requested=72,
            frozen_without_reading_evaluation_metrics=True,
            purpose='development_validation', helper_sha256=route.sha(route.__file__)))
        return plan

    def test_fixed72_all_arms_seeds_and_first4(self):
        plan = self.plan()
        self.assertEqual(len(plan['tasks']), 72)
        self.assertEqual(len({t['key'] for t in plan['tasks']}), 72)
        for objective in route.actor.OBJECTIVES:
            arm = route.actor.ARMS[objective][0]
            for seed in route.SEEDS:
                rows = [t for t in plan['tasks'] if t['arm']==arm and t['seed']==seed]
                self.assertEqual([r['noise_index'] for r in rows], [0,1,2,3])
                self.assertEqual({r['round'] for r in rows}, {0})
        self.assertFalse(plan['training_feedback_eligible'])

    def test_unchanged_route_controller_compatibility(self):
        plan = self.plan()
        tasks = requested(self.output, plan)
        self.assertEqual(len(tasks), 72)
        self.assertEqual(len({physical_run_id(plan,t['key']) for t in tasks}), 72)
        self.assertEqual(route_cap(plan),20)
        self.assertTrue(all(Path(t['output']).parent==self.output/'full_route' for t in tasks))

    def test_namespace_and_concurrency_guards(self):
        for prefix, concurrency in [('a/b',2),('okay',0),('okay',3),('okay',True)]:
            with self.assertRaises(ValueError):
                route.build_plan(self.output,self.actors,prefix,concurrency)

    def test_disjoint_output(self):
        with self.assertRaises(ValueError):
            route.build_plan(self.actors/'routes',self.actors)

    def test_config_different_physical_settings_refused(self):
        path=self.actors/'ff_drt_power/config.json'
        cfg=route.read(path);cfg['backend_threads']=4;path.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):self.plan()

    def test_validated_frozen_plan(self):
        plan=self.publish()
        digest=route.sha(self.output/'full_route_plan.json')
        self.assertEqual(route.validate_plan(self.output,digest),plan)

    def test_dropped_task_refused_even_when_declaration_rebound(self):
        self.publish()
        path=self.output/'full_route_plan.json';plan=route.read(path)
        plan['tasks'].pop();path.write_text(json.dumps(plan))
        declaration=route.read(self.output/'route_request.json')
        declaration['plan']=route.bind(path)
        (self.output/'route_request.json').write_text(json.dumps(declaration))
        with self.assertRaises(ValueError):route.validate_plan(self.output)

    def test_changed_source_or_config_refused(self):
        self.publish()
        with patch.object(route,'sources',return_value={'frozen.py':'b'*64}):
            with self.assertRaises(ValueError):route.validate_plan(self.output)
        path=self.actors/'ff_grt_wl/config.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):route.validate_plan(self.output)

    def test_pinned_plan_sha_refused(self):
        self.publish()
        with self.assertRaises(ValueError):route.validate_plan(self.output,'0'*64)

    def test_original_native_source_inspection_is_delegated(self):
        plan=self.plan()
        with patch.object(route.native,'inspect_sources',return_value=[{'state':'resolved'}]) as inspect:
            self.assertEqual(route.inspect_sources(self.output,plan,seal=True),[{'state':'resolved'}])
            inspect.assert_called_once_with(self.output.resolve(),plan,seal=True)
        plan['tasks'].pop()
        with self.assertRaises(ValueError):route.inspect_sources(self.output,plan)


if __name__ == '__main__':
    unittest.main()
