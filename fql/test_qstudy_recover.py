"""Recovery safety tests, mocked scheduler/native interfaces, no EDA."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from .qstudy_recover import disposition, guarded_validate, parse_known


class RecoveryTests(unittest.TestCase):
    def test_preempted_and_node_failure_can_retry(self):
        self.assertEqual(disposition([['PREEMPTED'], ['NODE_FAIL']]), 'retry')

    def test_live_attempt_always_prevents_duplicate(self):
        for live in ['PENDING', 'RUNNING', 'COMPLETING', 'SUSPENDED']:
            self.assertEqual(disposition([['PREEMPTED'], [live]]), 'wait')

    def test_unknown_accounting_cannot_retry(self):
        self.assertEqual(disposition([['PREEMPTED'], ['ACCOUNTING_PENDING']]), 'wait')
        self.assertEqual(disposition([]), 'wait')

    def test_completed_missing_outcome_is_integrity_error(self):
        with self.assertRaises(RuntimeError):
            disposition([['PREEMPTED'], ['COMPLETED']])

    def test_code_failure_not_relabelled_or_retried(self):
        for failure in ['FAILED', 'OUT_OF_MEMORY', 'CANCELLED']:
            with self.assertRaises(RuntimeError):
                disposition([[failure]])

    def test_known_index_bounds_and_nonnumeric_jobs_rejected(self):
        self.assertEqual(parse_known(['6=123', '6=124'], 52), {'6': ['123', '124']})
        for value in ['52=123', '-1=123', '6=1_6']:
            with self.assertRaises(ValueError):
                parse_known([value], 52)

    def fixture(self, root, outcome=True, successful=True):
        sample, run = root/'sample', root/'native'
        sample.mkdir(); run.mkdir()
        (run/'manifest.json').write_text(json.dumps(dict(success=successful)))
        if outcome:
            (sample/'outcome.json').write_text(json.dumps(dict(physical_run=str(run))))
        tasks = root/'tasks.jsonl'; tasks.write_text('{}\n')
        return [dict(sample_dir=str(sample))], tasks, run

    def test_missing_saved_geometry_never_calls_native_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows, tasks, _ = self.fixture(Path(temporary))
            validator = Mock()
            with self.assertRaisesRegex(ValueError, 'geometry audit missing'):
                guarded_validate(rows, tasks, Path(temporary), validator)
            validator.assert_not_called()

    def test_complete_native_validation_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows, tasks, run = self.fixture(root)
            (run/'backend_macro_audit.json').write_text('{}')
            validator = Mock(return_value=[])
            outputs = guarded_validate(rows, tasks, root, validator)
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0]['index'], 0)
            self.assertEqual(len(outputs[0]['sha256']), 64)
            validator.assert_called_once()

    def test_missing_is_not_failure_and_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows, tasks, _ = self.fixture(root, outcome=False)
            validator = Mock()
            with self.assertRaisesRegex(ValueError, 'all predeclared outcomes'):
                guarded_validate(rows, tasks, root, validator)
            validator.assert_not_called()


if __name__ == '__main__':
    unittest.main()
