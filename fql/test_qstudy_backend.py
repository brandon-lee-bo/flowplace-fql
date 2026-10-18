"""Read-only real-context and mocked failure/recovery checks; run through Slurm."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import qstudy_backend as backend
from .qstudy_data import action_sha


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='backend_check_',dir=os.environ['TMPDIR'])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        sample=self.root/'sample';sample.mkdir()
        checkpoint=self.root/'checkpoint.fixture';checkpoint.write_bytes(b'identity fixture, not model weights')
        raw=np.asarray([[.1,.2],[.3,.4]],dtype=np.float32)
        action=raw.copy();noise=-raw
        np.savez(sample/'actions.npz',raw_action=raw,action=action,noise=noise)
        config=self.root/'config.json';config.write_text('{}\n')
        row=dict(graph='bp_fe',seed=0,round=0,arm='fql',role='collect',index=0,
            raw_action=raw.tolist(),executed_action=action.tolist(),terminal_mask=0,
            training_feedback_eligible=True,source_checkpoint=str(checkpoint),
            checkpoint_sha256=backend.sha(checkpoint),sample_dir=str(sample),
            raw_action_sha256=action_sha(raw),action_sha256=action_sha(action),noise_sha256=action_sha(noise),
            config_sha256=backend.sha(config),run_id='unit_no_physical_runner')
        (sample/'request.json').write_text(json.dumps(row))
        tasks=self.root/'tasks.jsonl';tasks.write_text(json.dumps(row)+'\n')
        binding=dict(config_sha256=row['config_sha256'],dataset_sha256='datasetfixture',
            dataset_manifest_sha256='manifestfixture',physical_reference={},reward_scale=100.)
        cfg=dict(design='bp_fe',physical_failure_reward=-4,backend_storage='scratch')
        args=SimpleNamespace(experiment=str(self.root),seed=0,round=0,index=0,storage=None)
        return sample,row,tasks,binding,cfg,args

    def test_real_pinned_contexts(self):
        for design,dataset in [('bp_fe','bp_fe384'),('bp_be','bp_be64')]:
            exp=self.root/design;exp.mkdir()
            cfg=dict(design=design,dataset_path=str(backend.ROOT.parent.parent.parent/
                'unused'))
            cfg['dataset_path']=f'/fact_data/bohanli/flowplace_fql/qstudy_v1/data_audit/{dataset}/dataset.npz'
            cfg['physical_failure_reward']=-4
            cfg['backend_storage']='scratch'
            (exp/'config.json').write_text(json.dumps(cfg))
            loaded,verification,binding=backend.context(exp)
            self.assertEqual(binding['physical_flow_revision'],backend.PHYSICAL_REVISIONS[design])
            self.assertEqual(loaded['design'],design)
            self.assertEqual(verification is not None,design=='bp_be')

    def test_noise_and_raw_hash_tampering_rejected(self):
        sample,row,_,binding,cfg,args=self.fixture()
        backend.checked_request(row,args,cfg,binding)
        with np.load(sample/'actions.npz') as a:
            raw=a['raw_action'];action=a['action'];noise=a['noise'].copy()
        noise[0,0]+=.125
        np.savez(sample/'actions.npz',raw_action=raw,action=action,noise=noise)
        with self.assertRaisesRegex(ValueError,'Noise identity'):
            backend.checked_request(row,args,cfg,binding)
        row['raw_action_sha256']='0'*64
        (sample/'request.json').write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError,'Raw proposal'):
            backend.checked_request(row,args,cfg,binding)

    def recovery_setup(self,terminal=True):
        sample,row,tasks,binding,cfg,args=self.fixture()
        run=self.root/'old_physical_run';run.mkdir()
        if terminal:
            (run/'manifest.json').write_text(json.dumps(dict(time_end=1)))
        attempt=sample/'backend_attempts'/'old_job';attempt.mkdir(parents=True)
        (attempt/'attempt.json').write_text(json.dumps(dict(physical_run=str(run),
            request_sha256=backend.sha(sample/'request.json'),config_sha256=binding['config_sha256'],
            job_id='previous_completed_fixture_job',restart_count=0)))
        measured=dict(status='success',metrics=dict(rWL_um=200.),physical_run=str(run),
            physical_manifest_sha256='physicalfixture',physical_flow_revision=backend.PHYSICAL_REVISIONS['bp_fe'])
        return sample,row,tasks,binding,cfg,args,measured

    def test_completed_attempt_recovered_without_physical_rerun(self):
        sample,row,tasks,binding,cfg,args,measured=self.recovery_setup()
        with patch.object(backend,'context',return_value=(cfg,None,binding)), \
             patch.object(backend,'load_requests',return_value=([row],tasks)), \
             patch.object(backend,'checked_request',return_value=sample), \
             patch.object(backend,'verify_run',return_value=measured), \
             patch.object(backend,'archive_run',return_value={'status':'mock_archive'}) as archive, \
             patch.object(backend.subprocess,'run',side_effect=AssertionError('EDA must not rerun')):
            backend.measure(args)
            out=json.loads((sample/'outcome.json').read_text())
            self.assertEqual(out['reward'],-2.)
            self.assertIsNotNone(out['recovered_completed_attempt'])
            self.assertTrue((sample/'measurement.json').exists())
            self.assertEqual(archive.call_count,1)
            backend.measure(args)
            self.assertEqual(archive.call_count,1)

    def test_live_attempt_not_duplicated(self):
        sample,row,tasks,binding,cfg,args,_=self.recovery_setup(terminal=False)
        with patch.object(backend,'context',return_value=(cfg,None,binding)), \
             patch.object(backend,'load_requests',return_value=([row],tasks)), \
             patch.object(backend,'checked_request',return_value=sample), \
             patch.object(backend,'states',return_value=['RUNNING']), \
             patch.object(backend.subprocess,'run',side_effect=AssertionError('Live EDA must not duplicate')):
            with self.assertRaisesRegex(RuntimeError,'unresolved/live'):
                backend.measure(args)
            self.assertFalse((sample/'outcome.json').exists())

    def test_unknown_failure_never_becomes_numeric_reward(self):
        sample,row,tasks,binding,cfg,args,_=self.recovery_setup()
        with patch.object(backend,'context',return_value=(cfg,None,binding)), \
             patch.object(backend,'load_requests',return_value=([row],tasks)), \
             patch.object(backend,'checked_request',return_value=sample), \
             patch.object(backend,'verify_run',side_effect=RuntimeError('infrastructure failure')), \
             patch.object(backend,'archive_run',side_effect=AssertionError('Invalid label cannot archive')):
            with self.assertRaisesRegex(RuntimeError,'infrastructure'):
                backend.measure(args)
            self.assertFalse((sample/'outcome.json').exists())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Read-only native-context tests must use Slurm')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),slurm_job_id=os.environ['SLURM_JOB_ID'],
        physical_measurements_launched=False,backend_source_sha256=backend.sha(backend.__file__),
        test_source_sha256=backend.sha(__file__))
    (output/'backend_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
