"""Synthetic CPU contracts; never construct a real fresh CUDA noise/policy/EDA run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import qstudy_fresh_eval as q


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.output = self.root / "fresh"
        self.output.mkdir()
        self.data = self.root / "dataset.npz"
        np.savez(self.data, synthetic_fixture_only=np.zeros(2))
        revision = q.backend.PHYSICAL_REVISIONS["bp_fe"]
        q.atomic_json(self.root / "manifest.json", dict(design="bp_fe", dataset_sha256=q.sha(self.data),
            stage="global_route_only", no_test_or_heldout_exported=True, historical_online_included=False,
            physical_flow_revisions=[revision], summary=dict(train=dict(metrics=dict(rWL_um=dict(mean=100.))))))
        self.runtime = self.root / "runtime"
        (self.runtime / "benchmark_cache").mkdir(parents=True)
        self.cache = self.runtime / "benchmark_cache/bp_fe.pt"
        self.cache.write_bytes(b"synthetic graph cache: never passed to a model")
        self.reference = self.root / "physical_reference.json"
        q.atomic_json(self.reference, dict(physical_flow_revision=revision))
        self.public = self.root / "official_public.ckpt"
        self.public.write_bytes(b"synthetic official checkpoint")
        self.commons = {}
        self.critics = {}
        for seed in range(3):
            common = self.root / f"common_{seed}.pt"
            common.write_bytes(f"synthetic common{seed}".encode())
            self.commons[str(seed)] = str(common)
            critic = self.root / f"critic_{seed}.pt"
            critic.write_bytes(f"synthetic critic{seed}".encode())
            self.critics[str(seed)] = str(critic)
        self.cfg_base = dict(design="bp_fe", dataset_path=str(self.data), dataset_sha256=q.sha(self.data),
            cache_sha256=q.sha(self.cache), physical_failure_reward=-4, backend_threads=8,
            common_checkpoints=self.commons, common_checkpoint_sha256={s:q.sha(p) for s,p in self.commons.items()},
            critic_checkpoints=self.critics,
            critic_checkpoint_sha256={s:q.sha(p) for s,p in self.critics.items()},
            offline_actor_steps=1000, online_updates=1000, online_samples=0)
        streams = []
        for name, purpose, kind, qweight in (("candidate", "candidate", "one_step", 1),
                ("bc", "bc", "one_step", 0), ("flow", "behavior_flow", "behavior_flow", 0),
                ("public", "public_reference", "public_official20", 0)):
            seeds = [0] if kind == "public_official20" else [0, 1, 2]
            original = self.root / ("original_" + name)
            original.mkdir()
            pcfg = dict(self.cfg_base, q_weight=qweight)
            q.atomic_json(original / "config.json", pcfg)
            checkpoints, chain = {}, []
            for seed in seeds:
                if kind == "one_step":
                    for r in range(2):
                        folder = original / f"seed_{seed}/round_{r}"
                        folder.mkdir(parents=True)
                        ckpt = folder / "checkpoint.pt"
                        ckpt.write_bytes(f"synthetic frozen{name}{seed}round{r}".encode())
                        previous = original / f"seed_{seed}/round_0"
                        parent = None if r == 0 else dict(checkpoint_sha256=q.sha(previous / "checkpoint.pt"),
                            completed_sha256=q.sha(previous / "completed.json"), binding_sha256=q.sha(previous / "binding.json"))
                        binding = dict(config_sha256=q.sha(original / "config.json"),
                            dataset_sha256=pcfg["dataset_sha256"], seed=seed, round=r, parent=parent, replay=[],
                            common_sha256=pcfg["common_checkpoint_sha256"][str(seed)],
                            initial_critic_sha256=pcfg["critic_checkpoint_sha256"][str(seed)], source_sha256={})
                        q.atomic_json(folder / "binding.json", binding)
                        q.atomic_json(folder / "completed.json", dict(status="completed", binding=binding,
                            checkpoint_sha256=q.sha(ckpt), steps=1000))
                        chain.extend(q.binding(folder / file) for file in ("checkpoint.pt","completed.json","binding.json"))
                else:
                    ckpt = self.public if kind == "public_official20" else Path(self.commons[str(seed)])
                checkpoints[str(seed)] = q.binding(ckpt)
            experiment = self.output / name
            experiment.mkdir()
            ecfg = dict(pcfg, arm=name, run_prefix="fresh"+name, seeds=seeds,
                evaluation_noise_seed=q.NOISE_SEED, evaluation_samples=8,
                evaluation_purpose="fresh_confirmation", online_rounds=0)
            q.atomic_json(experiment / "config.json", ecfg)
            streams.append(dict(id=name, purpose=purpose, kind=kind, seeds=seeds,
                checkpoint_stage="trained" if kind == "one_step" else "public" if kind == "public_official20" else "common",
                selected_round=1 if kind == "one_step" else 0,
                source_config=q.binding(original / "config.json"), evaluation_config=q.binding(experiment / "config.json"),
                checkpoints=checkpoints, training_chain=sorted(chain,key=lambda x:x["path"])))
        note = self.root / "synthetic_development.json"
        q.atomic_json(note, dict(synthetic=True, inspected=[900000,4300000,4500000,4600000,5300000]))
        scales = self.root / "train_utility_reference.json"
        q.atomic_json(scales, dict(synthetic=True, TRAIN_scales=[100.,1.,10.,3.,4.], weights=[1.,0,0,0,0]))
        self.lock = dict(version=q.VERSION, status="selection_locked", scientific_reason="synthetic test only",
            primary_endpoint="rWL", secondary_acceptance_rules="declared before export", failure_accounting="all8",
            improvement_or_tradeoff_rule="synthetic preregistration", hardware_software=q.environment_spec("synthetic GPU name"),
            training_budget_comparison="same budgets", common_train_utility_reference=q.binding(scales),
            development_inventory=[q.binding(note)], inspected_noise_seeds=[900000,4300000,4500000,4600000,5300000],
            evaluation=dict(noise_seed=q.NOISE_SEED, samples=8, generator_device="cuda", dtype="float32",
                prior="uniform[-1,1]", shared_across_all_streams=True, drt_indices=[0,1,2,3], drt_repair_cap=20),
            output_root=str(self.output), design="bp_fe", dataset_sha256=q.sha(self.data), cache_sha256=q.sha(self.cache),
            physical_flow_revision=revision, streams=streams, candidate_is_legacy=True,
            source_sha256={name:q.sha(path) for name,path in q.required_sources().items()},
            geometry_inputs=[q.binding(p) for p in sorted([
                q.ROOT / "backend/releases/flowplace_DEFS/openroad/bp_fe.def",
                *(q.ROOT / "backend/thirdparty/OpenROAD-flow-scripts/platforms/nangate45/lef").glob("*.lef")])])
        self.path = self.root / "selection.json"
        self.save()

    def save(self):
        q.atomic_json(self.path, self.lock)
        self.digest = q.sha(self.path)

    def rebind_cfg(self, stream, changes):
        path = Path(stream["evaluation_config"]["path"])
        q.atomic_json(path, dict(q.read(path), **changes))
        stream["evaluation_config"] = q.binding(path)
        self.save()

    def noisy_fixture(self, n=3):
        # Hand-made synthetic arrays, not a generation of the reserved CUDA stream.
        a = np.arange(q.COUNT*n*2, dtype=np.float32).reshape(q.COUNT,n,2) / (q.COUNT*n*2)
        return a

    def saved_noise(self, n=3):
        folder = self.output / "shared_noise"
        folder.mkdir(exist_ok=True)
        a = self.noisy_fixture(n)
        np.savez(folder / "noises.npz", noise=a, selection_lock_sha256=np.asarray(self.digest))
        return q.shared_noise(self.output, self.digest, n)

    def native_row(self, index=0):
        lock, configs, physicals, declaration = q.declare(self.path, self.digest)
        stream = lock["streams"][0]
        noises, info = self.saved_noise()
        cfg, physical = configs[stream["id"]], physicals[stream["id"]]
        sample = Path(stream["evaluation_config"]["path"]).parent / "seed_0/round_0/requests/candidate/evaluation" / str(index)
        sample.mkdir(parents=True)
        raw = np.zeros((3,2), np.float32)
        action = raw.copy()
        np.savez(sample / "actions.npz", noise=noises[index], raw_action=raw, action=action)
        row = dict(q.row_identity(stream,0,index,cfg,physical,self.digest,declaration,info),
            sample_dir=str(sample), raw_action=raw.tolist(), executed_action=action.tolist(),
            raw_action_sha256=q.action_sha(raw), action_sha256=q.action_sha(action))
        q.atomic_json(sample / "request.json",row)
        return row, stream,cfg,physical,declaration,noises,info


class FreshEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self.temp.name)
        f = self.fixture
        self.patches = [patch.object(q, "RUNTIME", f.runtime), patch.object(q, "CHECKPOINT", f.public),
                        patch.object(q, "PUBLIC_SHA", q.sha(f.public)), patch.object(q.backend, "REFERENCE", f.reference)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def test_validate_declare_no_tensor_noise_policy_or_backend_calls(self):
        f=self.fixture
        with patch.object(q,"build_sampler",side_effect=AssertionError("model forbidden")), \
             patch.object(torch,"rand",side_effect=AssertionError("RNG forbidden")), \
             patch.object(q.backend,"measure",side_effect=AssertionError("EDA forbidden")):
            lock,cfg,physical,d = q.declare(f.path,f.digest)
            q.declare(f.path,f.digest)
        self.assertFalse((f.output/"shared_noise").exists())
        self.assertEqual([x["expected_requests"] for x in d["streams"]],[24,24,24,8])
        self.assertEqual(d["independent_public_checkpoints"],1)
        self.assertEqual(d["evaluation"]["drt_indices"],[0,1,2,3])
        self.assertEqual(d["evaluation"]["drt_repair_cap"],20)

    def test_unlocked_or_changed_selection_hash_refused(self):
        f=self.fixture
        with self.assertRaises(ValueError):
            q.validate_lock(f.path,"0"*64)
        for key,value in (("status","draft"),("primary_endpoint",None),("candidate_is_legacy",None)):
            old=copy.deepcopy(f.lock)
            f.lock[key]=value;f.save()
            with self.assertRaises(ValueError):q.validate_lock(f.path,f.digest)
            f.lock=old;f.save()

    def test_code_dataset_cache_backend_and_config_tamper_refused(self):
        f=self.fixture
        old=copy.deepcopy(f.lock)
        f.lock["source_sha256"]["baseline/api.py"]="0"*64;f.save()
        with self.assertRaisesRegex(ValueError,"source code"):q.validate_lock(f.path,f.digest)
        f.lock=old;f.save()
        cache_bytes=f.cache.read_bytes();f.cache.write_bytes(b"different cache")
        with self.assertRaisesRegex(ValueError,"locked file"):q.validate_lock(f.path,f.digest)
        f.cache.write_bytes(cache_bytes)
        f.data.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"dataset"):q.validate_lock(f.path,f.digest)

    def test_fresh_native_config_gate_is_real(self):
        f=self.fixture;s=f.lock["streams"][0]
        f.rebind_cfg(s,dict(backend_threads=4))
        with self.assertRaisesRegex(ValueError,"eight threads"):q.validate_lock(f.path,f.digest)
        f.rebind_cfg(s,dict(backend_threads=8,evaluation_noise_seed=4600000))
        with self.assertRaisesRegex(ValueError,"Fresh evaluation config"):q.validate_lock(f.path,f.digest)
        f.rebind_cfg(s,dict(evaluation_noise_seed=q.NOISE_SEED))
        f.lock["physical_flow_revision"]="0"*64;f.save()
        with self.assertRaisesRegex(ValueError,"physical revision"):q.validate_lock(f.path,f.digest)

    def test_selected_original_round_is_separate_from_logical_round(self):
        f=self.fixture
        row,s,cfg,p,d,noises,info=f.native_row()
        q.verify_row(row,s,0,0,cfg,p,f.digest,d,noises,info)
        self.assertEqual(row["round"],0)
        self.assertEqual(row["source_original_round"],1)
        f.lock["streams"][0]["selected_round"]=0;f.save()
        with self.assertRaisesRegex(ValueError,"training round"):q.validate_lock(f.path,f.digest)

    def test_public_cannot_duplicate_training_seeds_or_change_drt_subset(self):
        f=self.fixture
        old=copy.deepcopy(f.lock)
        f.lock["streams"][-1]["seeds"]=[0,1,2];f.save()
        with self.assertRaisesRegex(ValueError,"training-seed"):q.validate_lock(f.path,f.digest)
        f.lock=old;f.lock["evaluation"]["drt_indices"]=[3,4,5,6];f.save()
        with self.assertRaisesRegex(ValueError,"DRT"):q.validate_lock(f.path,f.digest)

    def test_native_checked_request_accepts_fresh_rejects_changed_arrays(self):
        f=self.fixture
        row,s,cfg,p,d,noises,info=f.native_row()
        sample=q.verify_row(row,s,0,0,cfg,p,f.digest,d,noises,info)
        self.assertEqual(sample,Path(row["sample_dir"]))
        changed=noises[0].copy();changed[0,0]+=.02
        np.savez(sample/"actions.npz",noise=changed,raw_action=np.zeros((3,2),np.float32),action=np.zeros((3,2),np.float32))
        with self.assertRaisesRegex(ValueError,"Noise identity"):q.verify_row(row,s,0,0,cfg,p,f.digest,d,noises,info)

    def test_native_feedback_source_and_prediction_identity_fail_closed(self):
        f=self.fixture
        row,s,cfg,p,d,noises,info=f.native_row()
        for key,value in (("training_feedback_eligible",True),("offline_train_eligible",True),
                          ("noise_seed",4600000),("predicted_q",1.),("fresh_drt_preselected",False)):
            changed=dict(row,**{key:value})
            with self.assertRaisesRegex(ValueError,"locked source"):q.verify_row(changed,s,0,0,cfg,p,f.digest,d,noises,info)
        checkpoint=Path(row["source_checkpoint"]);checkpoint.write_bytes(b"changed policy")
        with self.assertRaisesRegex(ValueError,"checkpoint identity"):q.verify_row(row,s,0,0,cfg,p,f.digest,d,noises,info)

    def test_actual_shared_noise_byte_identity_partial_publication_resume(self):
        f=self.fixture
        arrays,info=f.saved_noise()
        self.assertEqual(len(set(info["noise_sha256"])),8)
        again,info2=q.shared_noise(f.output,f.digest,3)
        self.assertTrue(np.array_equal(arrays,again));self.assertEqual(info,info2)
        self.assertTrue((f.output/"shared_noise/manifest.json").exists())
        with self.assertRaisesRegex(ValueError,"different selection"):q.shared_noise(f.output,"f"*64,3)
        with self.assertRaisesRegex(ValueError,"actual shared"):q.shared_noise(f.output,f.digest,4)

    def test_noise_refuses_duplicates_nonfinite_range_dtype_and_no_gpu_creation(self):
        f=self.fixture;a=f.noisy_fixture()
        invalid=[a.astype(np.float64),np.full_like(a,np.nan),np.full_like(a,2.),np.zeros_like(a)]
        for x in invalid:
            with self.assertRaises(ValueError):q.noise_manifest(x,f.digest,3)
        with patch.object(torch.cuda,"is_available",return_value=False):
            with self.assertRaisesRegex(RuntimeError,"Slurm CUDA"):q.shared_noise(f.output,f.digest,3,create=True)

    def test_immutable_declaration_and_source_config_integrity(self):
        f=self.fixture
        q.declare(f.path,f.digest)
        f.lock["primary_endpoint"]="after-the-fact change";f.save()
        with self.assertRaisesRegex(ValueError,"immutable"):q.declare(f.path,f.digest)
        source=Path(f.lock["streams"][0]["source_config"]["path"])
        q.atomic_json(source,dict(q.read(source),q_weight=0.1))
        with self.assertRaisesRegex(ValueError,"locked file"):q.validate_lock(f.path,f.digest)

    def test_original_parent_chain_and_locked_completion_are_checked(self):
        f=self.fixture
        s=f.lock["streams"][0]
        s["training_chain"]=[];f.save()
        with self.assertRaisesRegex(ValueError,"training/replay chain"):q.validate_lock(f.path,f.digest)
        folder=Path(s["checkpoints"]["0"]["path"]).parent
        b=q.read(folder/"binding.json");b["parent"]["checkpoint_sha256"]="f"*64
        q.atomic_json(folder/"binding.json",b)
        done=q.read(folder/"completed.json");done["binding"]=b;q.atomic_json(folder/"completed.json",done)
        with self.assertRaisesRegex(ValueError,"parent chain"):q.validate_lock(f.path,f.digest)

    def test_completed_marker_and_official_trajectory_are_bound(self):
        f=self.fixture
        row,s,cfg,p,d,noises,info=f.native_row()
        folder=Path(row["sample_dir"]).parents[2]
        (folder/"tasks.jsonl").write_text(json.dumps(row)+"\n")
        marker=dict(status="completed",count=8,config_sha256=p["config_sha256"],tasks_sha256=q.sha(folder/"tasks.jsonl"),
            selection_lock_sha256=f.digest,fresh_declaration_sha256=q.object_sha(d),backend_calls=0,
            optimizer_updates=0,evaluation_never_replayed=True)
        q.atomic_json(folder/"completed.json",marker)
        q.verify_completion(folder,cfg,f.digest,d)
        for key,value in (("status","partial"),("selection_lock_sha256","f"*64),("optimizer_updates",1),
                          ("backend_calls",1),("evaluation_never_replayed",False)):
            q.atomic_json(folder/"completed.json",dict(marker,**{key:value}))
            with self.assertRaisesRegex(ValueError,"marker changed"):q.verify_completion(folder,cfg,f.digest,d)
        public=f.lock["streams"][-1]
        proot=Path(public["evaluation_config"]["path"]).parent
        pcfg,_,phys=q.backend.context(proot)
        sample=proot/"seed_0/round_0/requests/public/evaluation/0";sample.mkdir(parents=True)
        a=np.zeros((3,2),np.float32)
        np.savez(sample/"actions.npz",noise=noises[0],raw_action=a,action=a)
        (sample/"trajectory.pt").write_bytes(b"synthetic trajectory")
        pubrow=dict(q.row_identity(public,0,0,pcfg,phys,f.digest,d,info),sample_dir=str(sample),
            raw_action=a.tolist(),executed_action=a.tolist(),raw_action_sha256=q.action_sha(a),action_sha256=q.action_sha(a),
            trajectory_sha256=q.sha(sample/"trajectory.pt"))
        q.atomic_json(sample/"request.json",pubrow)
        q.verify_row(pubrow,public,0,0,pcfg,phys,f.digest,d,noises,info)
        (sample/"trajectory.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"trajectory changed"):q.verify_row(pubrow,public,0,0,pcfg,phys,f.digest,d,noises,info)

    def test_hardware_precision_declaration_is_checked_without_gpu_inference(self):
        f=self.fixture
        f.lock["hardware_software"]["allow_tf32"]=True;f.save()
        with self.assertRaisesRegex(ValueError,"software versions"):q.validate_lock(f.path,f.digest)
        with patch.object(torch.cuda,"get_device_name",return_value="different GPU"), \
             patch.object(torch.cuda,"current_device",return_value=0):
            with self.assertRaisesRegex(ValueError,"Actual CUDA"):q.configure_export_environment(q.environment_spec("synthetic GPU name"))

    def test_sampler_dispatch_shapes_no_updates_and_explicit_public_noise(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
            def forward(self,x):return x*self.weight
            def sample_flow(self,x,steps):
                self.steps=steps;return x*.5
        model=Model().eval().requires_grad_(False)
        noise=torch.tensor([[.1,.2],[.3,.4]])
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            raw,t=q.generate_raw(dict(kind="one_step"),model,None,noise)
            self.assertEqual(tuple(raw.shape),(2,2));self.assertIsNone(t)
            raw,t=q.generate_raw(dict(kind="behavior_flow"),model,None,noise)
            self.assertEqual(model.steps,10)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with patch.object(q,"generate",return_value=(noise,noise.clone(),[noise])) as official:
            raw,t=q.generate_raw(dict(kind="public_official20"),model,None,noise)
            official.assert_called_once_with(model,None,noise=noise,intermediate_every=1)
            self.assertTrue(torch.equal(raw,noise))

    def test_actual_graph_policy_restore_retains_fixed_nodes_and_clips_movables(self):
        # Exercise the real wrapper without loading an upstream model/checkpoint.
        policy=q.GraphPolicy.__new__(q.GraphPolicy)
        torch.nn.Module.__init__(policy)
        policy.register_buffer("movable",torch.tensor([True,False,True]))
        policy.register_buffer("fixed",torch.tensor([[0.,0.],[.35,-.7],[0.,0.]]))
        x=torch.tensor([[[2.,-3.],[.9,.9],[-4.,5.]]])
        restored=policy.restore(x,clip=True)
        self.assertTrue(torch.equal(restored[0,1],policy.fixed[1]))
        self.assertTrue(torch.equal(restored[0,0],torch.tensor([1.,-1.])))
        self.assertTrue(torch.equal(restored[0,2],torch.tensor([-1.,1.])))


if __name__=="__main__":unittest.main()
