"""Metadata-only fixtures: no production selection, tensor load, RNG or EDA."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import qstudy_fresh_lock as builder
from .test_qstudy_fresh_eval import Fixture


class FreshLockBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/scratch/bohanli")
        self.fixture = f = Fixture(self.temp.name)
        self.patches = [patch.object(builder.fresh,"RUNTIME",f.runtime),
            patch.object(builder.fresh,"CHECKPOINT",f.public),
            patch.object(builder.fresh,"PUBLIC_SHA",builder.fresh.sha(f.public)),
            patch.object(builder.fresh.backend,"REFERENCE",f.reference),
            patch.object(torch,"load",side_effect=AssertionError("tensor load forbidden")),
            patch.object(torch,"rand",side_effect=AssertionError("RNG forbidden")),
            patch.object(builder.fresh,"declare",side_effect=AssertionError("declaration forbidden")),
            patch.object(builder.fresh,"build_sampler",side_effect=AssertionError("model forbidden")),
            patch.object(builder.fresh.backend,"measure",side_effect=AssertionError("EDA forbidden"))]
        for p in self.patches: p.start()
        self.request = dict(version=builder.VERSION,scientific_selection_approved_by_caller=True,
            output_root=str(f.root/"metadata_only"),run_namespace="fixture",native_reference_config=f.lock["streams"][1]["source_config"],
            gpu_name="synthetic GPU",candidate_is_legacy=True,legacy_anchor=None,
            common_train_utility_reference=f.lock["common_train_utility_reference"],
            development_inventory=f.lock["development_inventory"],
            inspected_noise_seeds=f.lock["inspected_noise_seeds"],
            **{key:f.lock[key] for key in builder.RULES})
        for stream in f.lock["streams"]:
            self.request[stream["purpose"]] = dict(source_config=stream["source_config"],selected_round=stream["selected_round"])
        self.path = f.root/"approved_request.json"

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def build(self):
        builder.fresh.atomic_json(self.path,self.request)
        return builder.build(self.path,builder.fresh.sha(self.path))

    def test_exact_config_chain_and_no_runtime_calls(self):
        result=self.build(); output=Path(self.request["output_root"])
        lock=builder.fresh.read(result["selection"]["path"])
        self.assertEqual(result["status"],"metadata_validated_not_declared")
        self.assertFalse((output/"declaration.json").exists())
        self.assertFalse((output/"shared_noise").exists())
        self.assertEqual([s["seeds"] for s in lock["streams"]],[[0,1,2]]*3+[[0]])
        self.assertEqual(lock["streams"][0]["training_chain"],self.fixture.lock["streams"][0]["training_chain"])
        self.assertEqual(builder.fresh.read(output/"candidate/config.json")["evaluation_noise_seed"],85000000)
        self.assertEqual(lock["evaluation"]["drt_indices"],[0,1,2,3])
        self.assertTrue(set(builder.route.SOURCE_NAMES)<=set(lock["source_sha256"]))
        self.assertEqual(self.build(),result)

    def test_no_implicit_approval_round_or_scientific_rule(self):
        original=copy.deepcopy(self.request)
        for key,value in (("scientific_selection_approved_by_caller",False),("candidate_is_legacy",None),("primary_endpoint",None)):
            self.request=copy.deepcopy(original); self.request[key]=value
            with self.assertRaises(ValueError): self.build()
        self.request=copy.deepcopy(original); del self.request["candidate"]["selected_round"]
        with self.assertRaisesRegex(ValueError,"round automatically"): self.build()
        self.assertFalse(Path(self.request["output_root"]).exists())

    def test_new_method_requires_explicit_legacy_anchor(self):
        self.request["candidate_is_legacy"]=False
        self.request["legacy_anchor"]=copy.deepcopy(self.request["candidate"])
        result=self.build(); lock=builder.fresh.read(result["selection"]["path"])
        self.assertEqual(len(lock["streams"]),5)
        self.assertEqual(lock["streams"][-1]["purpose"],"legacy_anchor")

    def test_native_reference_scale_overrides_source_evaluator_not_source_identity(self):
        source=Path(self.request["candidate"]["source_config"]["path"])
        original=builder.fresh.read(source)
        # A different source reward scale is metadata-only in this fixture;
        # update its own immutable chain config binding consistently first.
        original["reward_scale"]=50.
        source_data=self.fixture.root/"source_only_dataset.npz"
        source_data.write_bytes(b"synthetic different source: no arrays loaded")
        original.update(dataset_path=str(source_data),dataset_sha256=builder.fresh.sha(source_data))
        builder.fresh.atomic_json(source,original)
        for seed in range(3):
            for r in range(2):
                folder=source.parent/f"seed_{seed}/round_{r}"
                value=builder.fresh.read(folder/"binding.json"); value["config_sha256"]=builder.fresh.sha(source)
                value["dataset_sha256"]=original["dataset_sha256"]
                if r:
                    previous=folder.parent/"round_0"
                    value["parent"]={key:builder.fresh.sha(previous/name) for key,name in
                        (("checkpoint_sha256","checkpoint.pt"),("completed_sha256","completed.json"),("binding_sha256","binding.json"))}
                builder.fresh.atomic_json(folder/"binding.json",value)
                completed=builder.fresh.read(folder/"completed.json"); completed["binding"]=value
                builder.fresh.atomic_json(folder/"completed.json",completed)
        self.request["candidate"]["source_config"]=builder.fresh.binding(source)
        self.build()
        cfg=builder.fresh.read(Path(self.request["output_root"])/"candidate/config.json")
        self.assertEqual(cfg["reward_scale"],100.)
        self.assertEqual(cfg["dataset_path"],str(self.fixture.data))
        self.assertEqual(builder.fresh.read(source)["dataset_path"],str(source_data))
        self.assertEqual(builder.fresh.read(source)["reward_scale"],50.)

    def test_changed_binding_and_started_output_refused(self):
        self.request["candidate"]["source_config"]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"locked file"): self.build()
        self.request["candidate"]["source_config"]=builder.fresh.binding(self.fixture.root/"original_candidate/config.json")
        output=Path(self.request["output_root"]); output.mkdir()
        (output/"declaration.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"non-builder artifacts"): self.build()


if __name__=="__main__": unittest.main()
