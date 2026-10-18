import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from analysis import collect_costs


class CollectCostsTest(unittest.TestCase):
    def test_numeric_array_child_coverage_retains_both_requeue_attempt_costs(self):
        parent, child, missing = '6100000', '6100002', '6100999'
        # Numeric JobIDRaw differs from formatted parent_index. Both allocations
        # share a raw child ID but have distinct starts after requeue.
        preempted = '|'.join((child, parent+'_7', 'array_worker', 'PREEMPTED', '0:0',
            '600', '2', 'cpu=2,mem=8G,gres/gpu=1', '1200', 'test_node',
            '2026-10-18T00:00:00', '2026-10-18T00:01:00', '2026-10-18T00:11:00'))
        completed = '|'.join((child, parent+'_7', 'array_worker', 'COMPLETED', '0:0',
            '1800', '2', 'cpu=2,mem=8G,gres/gpu=1', '3600', 'test_node',
            '2026-10-18T00:00:00', '2026-10-18T00:13:00', '2026-10-18T00:43:00'))
        # Listing parent and child can return overlapping rows. Only exact
        # repeated attempts should be removed, never the distinct earlier run.
        raw = '\n'.join((preempted, completed, preempted, completed))+'\n'
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = directory/'jobs.json'
            manifest.write_text(json.dumps(dict(job_ids=[parent, child, missing], scope='unit_test')))
            output = directory/'accounting'
            argv = ['collect_costs.py', '--output', str(output), '--jobs-file', str(manifest)]
            with mock.patch.dict(os.environ, {'SLURM_JOB_ID':'9999999'}), \
                    mock.patch('sys.argv', argv), \
                    mock.patch.object(collect_costs.subprocess, 'check_output', return_value=raw) as sacct, \
                    contextlib.redirect_stdout(io.StringIO()):
                collect_costs.main()
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['unobserved_requested_job_ids'], [missing])
            self.assertEqual(report['requested_job_ids'], [parent, child, missing])
            self.assertEqual(report['attempt_counts_by_job_id_raw'], {child:2})
            self.assertEqual(report['duplicate_attempt_rows_skipped'], 2)
            self.assertEqual(len(report['allocations']), 2)
            self.assertEqual({r['State'] for r in report['allocations']}, {'PREEMPTED', 'COMPLETED'})
            self.assertEqual(len({r['allocation_attempt_key'] for r in report['allocations']}), 2)
            self.assertEqual(report['by_state']['PREEMPTED']['allocation_rows'], 1)
            self.assertEqual(report['by_state']['COMPLETED']['allocation_rows'], 1)
            self.assertAlmostEqual(report['reserved_cpu_hours'], (1200+3600)/3600)
            self.assertAlmostEqual(report['reserved_gpu_hours'], (600+1800)/3600)
            self.assertAlmostEqual(report['by_state']['PREEMPTED']['reserved_cpu_hours'], 1200/3600)
            self.assertAlmostEqual(report['by_state']['COMPLETED']['reserved_gpu_hours'], 1800/3600)
            self.assertEqual((output/'sacct.txt').read_text(), raw)
            with (output/'allocations.csv').open(newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            sacct.assert_called_once()
            command = sacct.call_args[0][0]
            self.assertIn('-D', command)
            self.assertIn('-X', command)
            self.assertEqual(command[command.index('-j')+1], ','.join((parent, child, missing)))


if __name__ == '__main__':
    unittest.main()
