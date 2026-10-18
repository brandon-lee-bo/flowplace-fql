"""Synthetic-only public-release source, sampler and eval-safety checks."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf
import torch

from . import qstudy_public_flow_eval as q


def sources():
    return [dict(seed=seed, index=index, noise_sha256=f"noise{index}", noise_shape=[2,2],
        source_bc=dict(path=f"bc{seed}/{index}"), source_fql=dict(path=f"fql{seed}/{index}"))
        for seed in range(3) for index in range(8)]


def config():
    return dict(design="bp_fe", dataset_path="data.npz", dataset_sha256="data", cache_sha256="cache",
        evaluation_noise_seed=4600000, evaluation_samples=8, physical_failure_reward=-4, backend_threads=8,
        evaluation_purpose="development_validation", arm="public_flowplace", seeds=[0], flow_steps=20,
        native_budget=8, online_flow_update=False, backend_concurrency=2, public_checkpoint_sha256=q.PUBLIC_SHA,
        run_prefix="pubv1", source_control_experiment="controls", source_control_config_sha256="sourcecfg",
        source_control_declaration_sha256="sourcedecl")


class PublicFlowTests(unittest.TestCase):
    def test_source_context_keeps_20hard_distinct_from_current_10step_teacher(self):
        cfg = config()
        parent = dict(cfg, seeds=[0,1,2], flow_steps=10, native_budget=24, arm="behavior_flow")
        physical = dict(config_sha256="new", physical_flow_revision="pinned")
        with patch.object(q, "context", return_value=(cfg,None,physical)), \
                patch.object(q, "file_binding"), patch.object(q, "read", return_value=dict(source_sha256={},pairs=sources(),expected_requests=24)), \
                patch.object(q, "control_settings", return_value=(parent,dict(physical,config_sha256="old"),[])), \
                patch.object(q, "source_pairs", return_value=sources()):
            result, _, pairs = q.settings(Path("public"))
            self.assertEqual(result["flow_steps"],20)
            self.assertEqual(len(pairs),24)
            for key,value in (("seeds",[0,1,2]),("flow_steps",10),("public_checkpoint_sha256","wrong"),
                              ("native_budget",24),("online_flow_update",True)):
                with self.subTest(key=key), patch.dict(cfg,{key:value}):
                    with self.assertRaisesRegex(ValueError,"one released"):
                        q.settings(Path("public"))

    def test_metadata_declaration_deduplicates_noise_but_retains_all_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            exp=Path(directory)
            source=exp/"source.py";source.write_text("immutable source")
            with patch.object(q,"source_files",return_value={"source.py":source}), \
                    patch.object(q,"file_binding",side_effect=lambda p,*a:dict(path=str(p),sha256="binding")), \
                    patch.object(q,"ROOT",exp), patch.object(q,"load_model") as load, patch.object(q,"generate") as generate:
                # Supply only tiny fixture snapshots, never inspect real runtime weights.
                (exp/"fql").mkdir()
                (exp/"fql/qstudy_public_flow_eval.slurm").write_text("wrapper")
                declaration=q.declare(exp,config(),dict(config_sha256="config"),sources())
                q.declare(exp,config(),dict(config_sha256="config"),sources())
            self.assertEqual(declaration["seeds"],[0]);self.assertEqual(declaration["native_budget"],8)
            self.assertEqual(len(declaration["noises"]),8)
            self.assertEqual(sum(len(n["matched_pairs"]) for n in declaration["noises"]),24)
            self.assertTrue(all(not declaration[k] for k in ("training","training_feedback_eligible","offline_train_eligible")))
            self.assertEqual(q.read(exp/"dispatch.json")["count"],8)
            load.assert_not_called();generate.assert_not_called()

    def test_source_seed_or_noise_collisions_are_refused_before_export(self):
        bad=sources();bad[-1]["seed"]=1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"all actor seeds"):
                q.declare(Path(directory),config(),{},bad)
        bad=sources()
        for p in bad:
            p["noise_sha256"]="identical"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"distinct"):
                q.declare(Path(directory),config(),{},bad)

    def fixture(self, directory, index=0):
        exp=Path(directory);checkpoint=exp/"public.ckpt";checkpoint.write_bytes(b"synthetic model")
        cfg=config();physical=dict(config_sha256="config",dataset_sha256="data")
        noise=np.array([[.1,.2],[.3,.4]],dtype=np.float32)
        item=dict(index=index,noise_sha256=q.action_sha(noise),noise_shape=[2,2],matched_pairs=[dict(seed=0)])
        raw=torch.tensor([[-.6,-.6],[.4,.4]])
        problem=SimpleNamespace(x=raw.clone(),cond=SimpleNamespace(is_ports=torch.tensor([True,False])))
        metric=Mock(return_value=10.);metric.sizes=np.full((2,2),.1)
        return exp,checkpoint,cfg,physical,noise,item,raw,problem,metric

    def test_exact_official_generation_then_external_legalizer_and_backend_compatible_request(self):
        with tempfile.TemporaryDirectory() as directory:
            exp,checkpoint,cfg,physical,noise,item,raw,problem,metric=self.fixture(directory)
            declaration=dict(noises=[item])
            moved=torch.tensor([[.9,.9],[.5,.5]])
            def sample_save(folder,*args):
                (folder/"macro_physical.json").write_text("{}")
            def def_convert(design,source,output):
                output.write_text("synthetic DEF")
                (output.with_suffix(".audit.json")).write_text(json.dumps(dict(output_sha256=q.sha(output),legal_no_halo=True)))
                return dict(legal_no_halo=True)
            with patch.object(q,"CHECKPOINT",checkpoint),patch.object(q,"PUBLIC_SHA",q.sha(checkpoint)), \
                    patch.object(q,"noise_array",return_value=noise),patch.object(q,"generate",return_value=(raw,torch.from_numpy(noise),[raw])) as gen, \
                    patch.object(q,"legalize",return_value=moved),patch.object(q,"save_sample",side_effect=sample_save), \
                    patch.object(q,"convert",side_effect=def_convert):
                row=q.emit(exp,cfg,physical,declaration,item,problem,SimpleNamespace(device="cpu"),metric)
                self.assertEqual(gen.call_count,2) # one fixed-index reproducibility check
                self.assertEqual(gen.call_args_list[0].kwargs["intermediate_every"],1)
                self.assertTrue(torch.equal(gen.call_args_list[0].kwargs["noise"],torch.from_numpy(noise)))
                self.assertEqual(row["raw_action_sha256"],q.action_sha(raw.numpy()))
                self.assertEqual(row["executed_action"][0],raw[0].tolist()) # explicit fixed-port restoration
                self.assertEqual(row["executed_action"][1],moved[1].tolist())
                self.assertEqual(row["source_sampler_steps"],20)
                self.assertFalse(row["training"]);self.assertFalse(row["offline_train_eligible"])
                self.assertIn("def_sha256",row)
                original=(Path(row["sample_dir"])/"request.json").read_bytes()
                gen.reset_mock()
                q.emit(exp,cfg,physical,declaration,item,problem,SimpleNamespace(device="cpu"),metric)
                gen.assert_not_called()
                self.assertEqual((Path(row["sample_dir"])/"request.json").read_bytes(),original)
                for key,value in (("training_feedback_eligible",True),("offline_train_eligible",True),
                                  ("source_sampler_steps",10),("noise_sha256","forged")):
                    with self.subTest(key=key),patch.object(q,"checked_request",return_value=Path(row["sample_dir"])):
                        with self.assertRaisesRegex(ValueError,"binding changed"):
                            q.verify_row(dict(row,**{key:value}),cfg,physical,declaration,item)

    def test_non_deterministic_public_sampler_records_actual_error_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            exp,checkpoint,cfg,physical,noise,item,raw,problem,metric=self.fixture(directory)
            changed=raw.clone();changed[1,0]+=.01
            with patch.object(q,"noise_array",return_value=noise),patch.object(q,"generate",side_effect=[
                    (raw,torch.from_numpy(noise),[raw]),(changed,torch.from_numpy(noise),[changed])]), \
                    patch.object(q,"legalize",return_value=raw):
                with self.assertRaisesRegex(ValueError,"no fallback"):
                    q.emit(exp,cfg,physical,{},item,problem,SimpleNamespace(device="cpu"),metric)
            sample=exp/"seed_0/round_0/requests/public_flowplace/evaluation/0"
            self.assertGreater(q.read(sample/"determinism.json")["raw_max_abs_difference"],0)
            self.assertFalse((sample/"request.json").exists())

    def test_export_loads_public_model_once_with_exact_release_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            exp=Path(directory);cfg=config();declaration=dict(noises=[dict(index=i) for i in range(8)])
            folder=exp/"seed_0/round_0/requests";folder.mkdir(parents=True)
            sampler_cfg=OmegaConf.create(dict(model=dict(max_diffusion_steps=20,use_hard_constraint_in_sampling=True)))
            emitted=[]
            def emit(*args):
                item=args[4];emitted.append(item["index"])
                return dict(index=item["index"],run_id=f"unique{item['index']}")
            with patch.dict(q.os.environ,SLURM_JOB_ID="synthetic"),patch.object(q.torch.cuda,"is_available",return_value=True), \
                    patch.object(q,"load_problem",return_value=SimpleNamespace(benchmark_info={})), \
                    patch.object(q,"load_model",return_value=(object(),sampler_cfg)) as load, \
                    patch.object(q,"MacroHPWL"),patch.object(q,"emit",side_effect=emit):
                q.export(exp,cfg,dict(config_sha256="config"),declaration)
            load.assert_called_once()
            self.assertEqual(load.call_args.kwargs,dict(device="cuda",steps=20,hard_constraint=True,w_reg=.1,w_dist=1.))
            self.assertEqual(emitted,list(range(8)))
            marker=q.read(folder/"completed.json")
            self.assertEqual(marker["count"],8);self.assertEqual(marker["independent_public_checkpoints"],1)
            self.assertEqual(marker["backend_calls"],0)
            self.assertFalse((exp/"seed_1").exists())
            with patch.object(q,"load_requests",return_value=([dict(index=i) for i in range(8)],folder/"tasks.jsonl")), \
                    patch.object(q,"verify_row") as verify,patch.object(q,"load_model") as load:
                q.export(exp,cfg,dict(config_sha256="config"),declaration)
                load.assert_not_called();self.assertEqual(verify.call_count,8)
                (exp/"public_sampler_config.json").write_text("{}")
                with self.assertRaisesRegex(ValueError,"sampler binding"):
                    q.export(exp,cfg,dict(config_sha256="config"),declaration)

    def test_orphan_outcome_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            exp,_,cfg,physical,_,item,_,problem,metric=self.fixture(directory)
            folder=exp/"seed_0/round_0/requests/public_flowplace/evaluation/0";folder.mkdir(parents=True)
            path=folder/"outcome.json";path.write_text('{"immutable":true}')
            with patch.object(q,"generate") as generate:
                with self.assertRaisesRegex(ValueError,"orphan"):
                    q.emit(exp,cfg,physical,{},item,problem,object(),metric)
            generate.assert_not_called();self.assertEqual(path.read_text(),'{"immutable":true}')


if __name__=="__main__":
    unittest.main()
