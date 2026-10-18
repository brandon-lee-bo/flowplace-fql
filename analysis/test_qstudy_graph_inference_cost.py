"""Small CPU contracts; execute only inside a bounded Slurm allocation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from analysis import qstudy_graph_inference_cost as cost
from analysis.placement_metrics import legality


class Backbone(nn.Module):
    def forward(self, x, *_):
        return .1*x


class Outer(nn.Module):
    def __init__(self):
        super().__init__(); self._reverse_model = Backbone()

    def __call__(self, x, *_):
        return self._reverse_model(x)


class Policy(cost.runtime().GraphPolicy):
    def __init__(self, fixed=True):
        # Exercise the real GraphPolicy methods without constructing/loading a
        # large released checkpoint in these small CPU fixtures.
        nn.Module.__init__(self); self.model = Outer(); self.cond = object()
        self.register_buffer('movable', torch.tensor([True, not fixed]))
        self.register_buffer('fixed', torch.tensor([[-.5, 0], [.5, 0]]))


def noise():
    return [torch.tensor([[[-.5+i*.001, 0], [.5, 0]]]) for i in range(cost.COUNT)]


class CostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def test_real_hook_survives_outer_call_override_and_counts_actor_teacher(self):
        p = Policy(); n = noise()[0]
        outer_count = []
        outer_hook = p.model.register_forward_hook(lambda *_: outer_count.append(1))
        try:
            _, n1 = cost.count_nfe(p.model._reverse_model, lambda x: p(x), n, 1)
            _, n10 = cost.count_nfe(p.model._reverse_model, lambda x: p.sample_flow(x, 10), n, 10)
        finally: outer_hook.remove()
        self.assertEqual((n1, n10), (1, 10)); self.assertEqual(outer_count, [])
        with self.assertRaisesRegex(ValueError, 'NFE'):
            cost.count_nfe(p.model._reverse_model, lambda x: p(x), n, 10)
        self.assertEqual(len(p.model._reverse_model._forward_hooks), 0)

    def test_full_stage_and_fresh_legalizer_calls_are_real_and_not_summed(self):
        calls = []
        def legalizer(x, problem):
            calls.append(1); return x.clone()
        p = Policy(); case = cost.make_case(p, noise(), cost.LABELS[0], legalizer, None)
        saved = []
        for n in case['noises']:
            r = p(n)[0].numpy(); saved.append(dict(raw_action=r, action=r.copy()))
        audit = cost.audit_case(case, saved, legality, np.zeros((2, 2)))
        self.assertEqual(len(calls), 16)
        self.assertTrue(all(a['nfe'] == dict(model_only=1, legalizer_only=0, full_generation=1) for a in audit))
        before = len(calls)
        ticks = iter(i*.01 for i in range(1000))
        rows = cost.benchmark_cases([case], warmup=0, repeats=32, device='cpu', clock=lambda: next(ticks))
        self.assertEqual(len(calls)-before, 64)
        self.assertEqual(len(rows), 96)
        for scope in cost.SCOPES:
            local = [r for r in rows if r['scope']==scope]
            self.assertEqual([sum(r['noise_index']==i for r in local) for i in range(8)], [4]*8)
            self.assertTrue(all(abs(r['seconds']-.01)<1e-10 for r in local))
        self.assertEqual(sum(r['expected_backbone_nfe'] for r in rows), 64)

    def test_explicit_sync_brackets_the_timed_call(self):
        events = []
        ticks = iter([1., 1.25])
        def clock(): events.append('clock'); return next(ticks)
        def fn(x): events.append('function'); return x
        with patch.object(cost, 'synchronize', side_effect=lambda _: events.append('sync')):
            out, elapsed = cost.measured_call(fn, 42, 'cuda', clock)
        self.assertEqual((out, elapsed), (42, .25))
        self.assertEqual(events, ['sync', 'clock', 'function', 'sync', 'clock'])

    def test_fixed_zero_count_and_illegal_geometry_not_filtered(self):
        for fixed in (True, False):
            p = Policy(fixed); case = cost.make_case(p, noise(), cost.LABELS[0], lambda x, _: x.clone(), None)
            saved = [dict(raw_action=p(n)[0].numpy(), action=p(n)[0].numpy()) for n in case['noises']]
            rows = cost.audit_case(case, saved, legality, np.full((2, 2), 1.5))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]['fixednode_count'], int(fixed))
            self.assertFalse(rows[0]['executed_geometry']['legal'])
        p = Policy(); case = cost.make_case(p, noise(), cost.LABELS[0], lambda x, _: x+1, None)
        with self.assertRaisesRegex(ValueError, 'fixed node'):
            cost.audit_case(case, saved, legality, np.zeros((2, 2)))

    def test_model_state_equality_and_hash_include_buffers(self):
        p = Policy(); state = p.state_dict(); clone = copy.deepcopy(state)
        self.assertTrue(cost.equal_state(state, clone))
        self.assertEqual(cost.state_sha(state), cost.state_sha(clone))
        clone['fixed'][0, 0] += .1
        self.assertFalse(cost.equal_state(state, clone))
        self.assertNotEqual(cost.state_sha(state), cost.state_sha(clone))

    def test_loaded_actual_noise_and_coordinates_must_match_bound_hashes(self):
        rows = []
        for i, n in enumerate(noise()):
            folder = self.root/str(i); folder.mkdir()
            a = n[0].numpy()
            np.savez_compressed(folder/'actions.npz', noise=a, raw_action=a, action=a)
            (folder/'request.json').write_text('{}')
            rows.append(dict(index=i, request=cost.bind(folder/'request.json'), actions=cost.bind(folder/'actions.npz'),
                noise_sha256=cost.array_sha(a), raw_action_sha256=cost.array_sha(a), action_sha256=cost.array_sha(a)))
        (self.root/'tasks.jsonl').write_text('fixture\n'); (self.root/'completed.json').write_text('{}')
        export=dict(rows=rows,tasks=cost.bind(self.root/'tasks.jsonl'), completion=cost.bind(self.root/'completed.json'))
        self.assertEqual(len(cost.load_arrays(export)), 8)
        wrong = copy.deepcopy(export); wrong['rows'][0]['noise_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'arrays differ'): cost.load_arrays(wrong)
        p = Path(rows[0]['actions']['path']); p.write_bytes(p.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError, 'Changed'): cost.load_arrays(export)

    def test_checkpoint_prepare_uses_producer_sha_without_bulk_read(self):
        p = self.root/'checkpoint.pt'; p.write_bytes(b'not loaded as a model')
        with patch.object(cost, 'sha', side_effect=AssertionError('bulk checkpoint hashed during prepare')):
            declared = cost.declared_checkpoint(p, 'a'*64)
        self.assertEqual(declared['sha256'], 'a'*64)
        with self.assertRaisesRegex(ValueError, 'SHA'): cost.declared_checkpoint(p, 'not-a-digest')

    def test_actor_metadata_checks_real_parent_and_own_collect_without_loading(self):
        cfg = dict(dataset_sha256='d'*64, common_checkpoint_sha256={'0': 'c'*64},
            critic_checkpoint_sha256={'0': 'e'*64}, arm='bc')
        path = self.root/'config.json'; path.write_text(json.dumps(cfg))
        parent = None
        for r in range(3):
            folder = self.root/f'seed_0/round_{r}'; folder.mkdir(parents=True, exist_ok=True)
            replay = []
            for prior in range(r):
                for index in range(8):
                    p = self.root/f'seed_0/round_{prior}/requests/bc/collect/{index}/outcome.json'
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(dict(status='success', arm='bc', seed=0,
                        role='collect', training_feedback_eligible=True, round=prior)))
                    replay.append(dict(path=str(p), sha256=cost.sha(p), status='success'))
            binding = dict(config_sha256=cost.sha(path), dataset_sha256='d'*64,
                common_sha256='c'*64, initial_critic_sha256='e'*64,
                seed=0, round=r, parent=parent, replay=replay)
            (folder/'checkpoint.pt').write_bytes(b'fixture; never torch.load')
            (folder/'binding.json').write_text(json.dumps(binding))
            (folder/'completed.json').write_text(json.dumps(dict(status='completed', binding=binding,
                steps=1000 if r==0 else 600, checkpoint_sha256=str(r)*64)))
            parent = dict(checkpoint_sha256=str(r)*64,
                completed_sha256=cost.sha(folder/'completed.json'), binding_sha256=cost.sha(folder/'binding.json'))
        actual_sha = cost.sha
        def no_bulk(p):
            if str(p).endswith('.pt'): raise AssertionError('Bulk read during prepare')
            return actual_sha(p)
        with patch.object(cost, 'sha', side_effect=no_bulk), patch.object(cost.torch, 'load', side_effect=AssertionError('load')):
            result = cost.actor_metadata(path, cfg, 0)
        self.assertEqual([r['round'] for r in result], [0, 1, 2])
        changed = cost.read(folder/'binding.json'); changed['parent']['checkpoint_sha256']='f'*64
        done = cost.read(folder/'completed.json'); done['binding']=changed
        (folder/'binding.json').write_text(json.dumps(changed)); (folder/'completed.json').write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError, 'parent chain'): cost.actor_metadata(path, cfg, 0)

    def test_three_seed_summary_does_not_pool_repetitions_as_training_seeds(self):
        rows = [dict(seed=s, label=l, scope=c, repeat=i, noise_index=i%8, seconds=(s+1)*.001)
                for s in cost.SEEDS for l in cost.LABELS for c in cost.SCOPES for i in range(32)]
        out = cost.summarize(rows)
        self.assertEqual(len(out), 9)
        self.assertEqual(out[0]['three_seed_mean_ms']['n'], 3)
        self.assertAlmostEqual(out[0]['three_seed_mean_ms']['mean'], 2)
        self.assertAlmostEqual(out[0]['three_seed_mean_ms']['sample_sd'], 1)
        with self.assertRaisesRegex(ValueError, 'denominator'): cost.summarize(rows[:-1])

    def test_python_patch_is_recorded_but_abi_and_packages_are_frozen(self):
        prepared = dict(python='3.9.21', python_abi='3.9', torch='2.5.1', numpy='1.24',
            torch_geometric='2.6', cuda='12.4', executable='/pinned/.venv/bin/python')
        runtime = dict(prepared, python='3.9.25')
        self.assertNotEqual(prepared, runtime)
        self.assertEqual(cost.software_contract(prepared), cost.software_contract(runtime))
        self.assertNotEqual(cost.software_contract(prepared), cost.software_contract(dict(runtime, python_abi='3.10')))
        self.assertNotEqual(cost.software_contract(prepared), cost.software_contract(dict(runtime, torch='2.6.0')))


if __name__ == '__main__': unittest.main()
