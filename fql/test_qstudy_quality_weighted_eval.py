"""Small software checks; no public model, CUDA inference, or native EDA."""
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

from . import qstudy_quality_weighted_eval as q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TinyFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("movable", torch.tensor([True, False]))
        self.register_buffer("fixed", torch.tensor([[0., 0.], [.25, .25]]))
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.calls = 0

    def sample_flow(self, noise, steps):
        if steps != 10:
            raise ValueError("not Euler10")
        self.calls += 1
        return torch.where(self.movable[None, :, None], noise.clamp(-1, 1), self.fixed[None])


class EvalTests(unittest.TestCase):
    def training_fixture(self, root, arm="uniform"):
        train = root/"training"
        common = {}
        for seed in q.SEEDS:
            p = root/f"common{seed}.pt"
            p.write_bytes(b"opaque initial flow"+bytes([seed]))
            common[str(seed)] = q.file_binding(p)
        plan = dict(output=str(train), settings=dict(steps=1000, model_mode="eval_with_gradients"),
            arms=list(q.ARMS), seeds=list(q.SEEDS), population=dict(weights=[.5, 1., 1.5]),
            inputs=dict(common=common), source_sha256={"frozen/source.py":"sha"})
        write(train/"plan.json", plan)
        manifest = q.file_binding(train/"plan.json")
        artifacts = {}
        for a in q.ARMS:
            for seed in q.SEEDS:
                folder = train/a/f"seed_{seed}"
                folder.mkdir(parents=True)
                (folder/"checkpoint.pt").write_bytes(b"opaque trained flow"+a.encode()+bytes([seed]))
                chain = "same_stream_seed"+str(seed)
                w = np.ones(3, np.float32) if a == "uniform" else np.asarray(plan["population"]["weights"], np.float32)
                binding = dict(plan=manifest, task=dict(index=q.ARMS.index(a)*3+seed, arm=a, seed=seed),
                    common=common[str(seed)], source_sha256=plan["source_sha256"],
                    population_sha256=q.object_sha(plan["population"]), effective_weight_sha256=q.training.array_sha(w),
                    initial_state_key="flow", model_mode="eval_with_gradients")
                write(folder/"binding.json", binding)
                write(folder/"learning_curve.json", [dict(step=1000, draw_chain_sha256=chain)])
                done = dict(status="completed", binding=binding, steps=1000,
                    checkpoint_sha256=q.sha(folder/"checkpoint.pt"), curve_sha256=q.sha(folder/"learning_curve.json"),
                    draw_chain_sha256=chain, flow_updates=1000, actor_updates=0, critic_updates=0,
                    layout_generation_calls=0, backend_calls=0)
                write(folder/"completed.json", done)
                if a == arm:
                    artifacts[str(seed)] = {k:q.file_binding(folder/f) for k,f in (
                        ("checkpoint","checkpoint.pt"),("completion","completed.json"),
                        ("binding","binding.json"),("curve","learning_curve.json"))}
        cfg = dict(arm=arm, training_manifest=manifest, training_root=str(train), trained_artifacts=artifacts)
        return cfg, plan

    def test_all_six_completions_and_same_seed_streams_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, plan = self.training_fixture(Path(tmp))
            with patch.object(q.training, "checked_plan", return_value=(plan, {})):
                _, result = q.training_artifacts(cfg)
            self.assertEqual(sum(map(len,result.values())),6)
            self.assertEqual(result["uniform"]["2"]["draw_chain_sha256"], result["quality_rank"]["2"]["draw_chain_sha256"])

    def test_changed_other_arm_stream_refused_even_when_own_checkpoint_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, plan = self.training_fixture(Path(tmp))
            folder = Path(cfg["training_root"])/"quality_rank/seed_1"
            done = q.read(folder/"completed.json"); done["draw_chain_sha256"] = "different"
            curve = [dict(step=1000, draw_chain_sha256="different")]
            write(folder/"learning_curve.json", curve)
            done["curve_sha256"] = q.sha(folder/"learning_curve.json")
            write(folder/"completed.json", done)
            with patch.object(q.training, "checked_plan", return_value=(plan, {})):
                with self.assertRaisesRegex(ValueError,"identical row/noise/time"):
                    q.training_artifacts(cfg)

    def test_partial_or_actor_trained_completions_refused(self):
        for field, bad in (("steps",999),("actor_updates",1),("backend_calls",1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                cfg, plan = self.training_fixture(Path(tmp))
                path=Path(cfg["training_root"])/"quality_rank/seed_2/completed.json"
                done=q.read(path);done[field]=bad;write(path,done)
                with patch.object(q.training,"checked_plan",return_value=(plan,{})):
                    with self.assertRaisesRegex(ValueError,"Incomplete/changed"):
                        q.training_artifacts(cfg)

    def test_declared_checkpoint_or_completion_mutation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, plan = self.training_fixture(Path(tmp))
            Path(cfg["trained_artifacts"]["1"]["checkpoint"]["path"]).write_bytes(b"changed")
            with patch.object(q.training,"checked_plan",return_value=(plan,{})):
                with self.assertRaisesRegex(ValueError,"Incomplete/changed"):
                    q.training_artifacts(cfg)

    def test_model_load_uses_flow_state_and_checks_fixed_node_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, plan = self.training_fixture(Path(tmp))
            with patch.object(q.training,"checked_plan",return_value=(plan,{})):
                _, artifacts=q.training_artifacts(cfg)
            a=artifacts["uniform"]["0"]
            tiny=TinyFlow()
            saved=dict(step=1000,binding=q.read(a["binding"]["path"]),
                       draw_chain_sha256=a["draw_chain_sha256"],flow=tiny.state_dict())
            with patch.object(q.torch.cuda,"is_available",return_value=True), \
                    patch.object(q.torch,"load",return_value=saved), patch.object(q,"load_problem",return_value=object()), \
                    patch.object(q,"GraphPolicy",side_effect=lambda _:TinyFlow()):
                _,flow=q.trained_flow(dict(cfg,design="bp_fe"),dict(all_paired_training=artifacts),0)
                self.assertTrue(all(not p.requires_grad for p in flow.parameters()))
                saved["flow"]["fixed"]=saved["flow"]["fixed"]+.1
                with self.assertRaisesRegex(ValueError,"fixed-node buffers"):
                    q.trained_flow(dict(cfg,design="bp_fe"),dict(all_paired_training=artifacts),0)

    def export_fixture(self, root, index=0):
        exp=root/"eval";exp.mkdir()
        checkpoint=root/"flow.pt";checkpoint.write_bytes(b"trained flow")
        source=root/"source.npz"
        noise=np.array([[.1,.2],[.4,.5]],np.float32)
        np.savez(source,noise=noise)
        cfg=dict(arm="uniform",design="bp_fe",run_prefix="qwev1",dataset_sha256="dataset",
            cache_sha256="cache",training_manifest=dict(path="plan.json",sha256="plan"))
        write(exp/"config.json",cfg)
        physical=dict(config_sha256=q.sha(exp/"config.json"),dataset_sha256="dataset")
        pair=dict(seed=0,index=index,noise_sha256=q.action_sha(noise),noise_shape=[2,2],
            source_bc=dict(actions_npz=q.file_binding(source)))
        artifact=dict(checkpoint=q.file_binding(checkpoint),draw_chain_sha256="paired_stream")
        declaration=dict(all_paired_training={"uniform":{"0":artifact}},pairs=[pair])
        metric=type("Metric",(),{"sizes":np.ones((2,2)),"__call__":lambda self,x:float(np.asarray(x).sum())})()
        return exp,cfg,physical,pair,declaration,metric

    def test_export_is_real_backend_request_compatible_and_resumes_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp,cfg,physical,pair,declaration,metric=self.export_fixture(Path(tmp))
            flow=TinyFlow()
            with patch.object(q,"legalize",side_effect=lambda raw,_:raw), \
                    patch.object(q,"legality",return_value=dict(legal=False)):
                row=q.emit(exp,cfg,physical,declaration,pair,object(),flow,metric)
                self.assertEqual(flow.calls,2) # deterministic check, never a noise draw
                again=q.emit(exp,cfg,physical,declaration,pair,object(),flow,metric)
            self.assertEqual(row,again)
            self.assertEqual(flow.calls,2)
            self.assertFalse(row["training_feedback_eligible"])
            self.assertFalse(row["offline_train_eligible"])
            self.assertEqual(row["pre_backend_failure"],"normalized_geometry")
            self.assertIsNone(row["predicted_q"])
            self.assertEqual(row["executed_action"][1],[.25,.25])
            q.checked_request(row,argparse.Namespace(seed=0,round=0),cfg,physical)
            # Actual load_requests validator accepts our frozen per-seed manifest.
            folder=Path(row["sample_dir"]).parents[2]
            tasks=folder/"tasks.jsonl";tasks.write_text(json.dumps(row)+"\n")
            write(folder/"completed.json",dict(count=1,tasks_sha256=q.sha(tasks),config_sha256=physical["config_sha256"]))
            rows,_=q.load_requests(exp,0,0)
            self.assertEqual(rows,[row])

    def test_noise_or_replay_or_trained_source_tamper_refused_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp,cfg,physical,pair,declaration,metric=self.export_fixture(Path(tmp))
            with patch.object(q,"legalize",side_effect=lambda raw,_:raw), \
                    patch.object(q,"legality",return_value=dict(legal=False)):
                row=q.emit(exp,cfg,physical,declaration,pair,object(),TinyFlow(),metric)
            for key,bad in (("offline_train_eligible",True),("source_state_key","actor"),("source_sampler_steps",20)):
                changed=dict(row,**{key:bad})
                write(Path(row["sample_dir"])/"request.json",changed)
                with self.assertRaisesRegex(ValueError,"checkpoint/stream/noise/eligibility"):
                    q.verify_row(changed,cfg,physical,declaration,pair,exp)
            write(Path(row["sample_dir"])/"request.json",row)
            source=Path(pair["source_bc"]["actions_npz"]["path"])
            np.savez(source,noise=np.zeros((2,2),np.float32))
            with self.assertRaisesRegex(ValueError,"input changed"):
                q.verify_row(row,cfg,physical,declaration,pair,exp)

    def test_legal_DEF_is_bound_by_unchanged_backend_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp,cfg,physical,pair,declaration,metric=self.export_fixture(Path(tmp))
            def fake_save(sample,*_):
                write(sample/"macro_physical.json",{})
            def fake_convert(design,coordinates,output):
                output.write_text("fixture DEF")
                audit=dict(output_sha256=q.sha(output),legal_no_halo=True)
                write(output.with_suffix(".audit.json"),audit)
                return audit
            with patch.object(q,"legalize",side_effect=lambda raw,_:raw), \
                    patch.object(q,"legality",return_value=dict(legal=True)), \
                    patch.object(q,"save_sample",side_effect=fake_save),patch.object(q,"convert",side_effect=fake_convert):
                row=q.emit(exp,cfg,physical,declaration,pair,object(),TinyFlow(),metric)
            self.assertNotIn("pre_backend_failure",row)
            q.verify_row(row,cfg,physical,declaration,pair,exp)
            Path(row["def_path"]).write_text("changed DEF")
            with self.assertRaisesRegex(ValueError,"DEF/export audit"):
                q.verify_row(row,cfg,physical,declaration,pair,exp)

    def test_sampler_moving_fixed_ports_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp,cfg,physical,pair,declaration,metric=self.export_fixture(Path(tmp))
            flow=TinyFlow()
            with patch.object(flow,"sample_flow",side_effect=lambda x,steps:x), \
                    patch.object(q,"legalize",side_effect=lambda raw,_:raw):
                with self.assertRaisesRegex(ValueError,"moved fixed ports"):
                    q.emit(exp,cfg,physical,declaration,pair,object(),flow,metric)

    def test_orphan_export_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp,cfg,physical,pair,declaration,metric=self.export_fixture(Path(tmp))
            sample=exp/"seed_0/round_0/requests/uniform/evaluation/0"
            sample.mkdir(parents=True); (sample/"outcome.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"orphan"):
                q.emit(exp,cfg,physical,declaration,pair,object(),TinyFlow(),metric)

    def test_immutable_dispatch_has_exact_ordered_24_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);write(exp/"declaration.json",{})
            pairs=[dict(seed=s,index=i,noise_sha256=f"noise{i}") for s in q.SEEDS for i in range(8)]
            cfg=dict(arm="quality_rank",run_prefix="qwer1")
            d=q.dispatch_value(exp,cfg,pairs)
            self.assertEqual(len(d["requests"]),24)
            self.assertEqual([(r["seed"],r["index"]) for r in d["requests"]],[(s,i) for s in q.SEEDS for i in range(8)])
            q.immutable(exp/"dispatch.json",d)
            bad=copy.deepcopy(d);bad["requests"][0]["noise_sha256"]="other"
            with self.assertRaisesRegex(ValueError,"immutable"):
                q.immutable(exp/"dispatch.json",bad)

    def test_declare_cannot_follow_undeclared_layout_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);(exp/"seed_0").mkdir()
            with patch.object(q,"declaration_value",return_value={"source_sha256":{}}):
                with self.assertRaisesRegex(ValueError,"after any evaluation"):
                    q.declare(exp,{}, {},[],{}, {})

    def test_seed_completion_semantics_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);write(exp/"config.json",{})
            folder=exp/"seed_0/round_0/requests";folder.mkdir(parents=True)
            rows=[dict(index=i,run_id=f"run{i}") for i in range(8)]
            tasks=folder/"tasks.jsonl";tasks.write_text("".join(json.dumps(r)+"\n" for r in rows))
            declaration=dict(selection="all fixed",pairs=[dict(seed=0,index=i) for i in range(8)])
            marker=dict(status="completed",count=8,tasks_sha256=q.sha(tasks),
                config_sha256=q.sha(exp/"config.json"),evaluation_never_replayed=True,
                selection=declaration["selection"],quality_flow_declaration_sha256=q.object_sha(declaration))
            with patch.object(q,"verify_row"):
                write(folder/"completed.json",marker)
                q.checked_seed(exp,{}, {},declaration,0)
                for key,bad in (("status","pending"),("evaluation_never_replayed",False),("selection","best1")):
                    write(folder/"completed.json",dict(marker,**{key:bad}))
                    with self.assertRaisesRegex(ValueError,"fixed8 quality-FM export manifest"):
                        q.checked_seed(exp,{}, {},declaration,0)


if __name__ == "__main__":
    unittest.main()
