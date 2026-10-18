"""Small real critic fits and objective contracts; CPU Slurm only."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_rule_reward_screen as screen
from . import qstudy_bounded_screen as bounded
from .qstudy_data import sha


class RuleRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run rule reward tests in CPU Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='qstudy-rule-screen-',
            dir=os.environ.get('TMPDIR', '/scratch/bohanli/flowplace_graph'))
        self.root = Path(self.temporary.name)
        self.cache = dict(node_id_to_name=['a', 'b', 'c'], ratio_x=10/224, ratio_y=20/224,
            node_info={n:dict(x=1., y=1.2) for n in 'abc'},
            net_info={'n':dict(nodes={'a':dict(x_offset=.2), 'b':{}, 'c':dict(y_offset=-.3)})},
            macro_clusters=[['a', 'b']], id2index={n:i for i,n in enumerate('abc')},
            dataflow_mat=np.asarray([[0.,1.,2.],[1.,0.,0.],[2.,0.,0.]]))
        gen = torch.Generator().manual_seed(19)
        x = torch.rand((12,3,2), generator=gen)*1.2-.6
        native = torch.stack((100+3*x[:,0,0], -(.4+.1*x[:,1,0]), -(25+x[:,2,1]),
            (3*x[:,0,1]).clamp_min(0), (4*x[:,1,1]).clamp_min(0),
            torch.ones(12)*.2, torch.ones(12)*1000), -1).double().numpy()
        base = dict(x=x, native=native, mask=np.isfinite(native),
            executed=(x*.9).numpy().reshape(12,-1), statuses=np.full(12,'success',dtype='<U32'),
            families=np.arange(6).repeat(2), source_ids=[f'train_family{i//2}_row{i}' for i in range(12)],
            movable=torch.ones(3,dtype=torch.bool), cache=self.cache,
            binding=dict(dataset_sha256='a'*64, cache_sha256='b'*64))
        self.data = {'train':base, 'validation':copy.deepcopy(base)}
        screen.add_costs(self.data,self.cache)
        self.context = self.data['train']
        # Avoid making narrow-channel variation depend on incidental random
        # geometry in the tiny fit fixture; geometry definition has own tests.
        self.context['cost_values'][:,-1] = np.arange(12)/10+.1
        self.data['validation']['cost_values'][:,-1] = np.arange(12)/10+.2
        fold = screen.family_folds(self.context['families'],3)[0]
        self.train, self.target = np.asarray(fold['train_indices']), np.asarray(fold['target_indices'])
        screen.STOP = False

    def tearDown(self):
        screen.STOP = False
        self.temporary.cleanup()

    def args(self, name='fit', objective='hpwl_expert_composite', steps=4):
        return SimpleNamespace(output=self.root/name, objective=objective, seed=2,
            steps=steps, batch_size=8, lr=3e-4, weight_decay=1e-4)

    def fit(self, args, context=None, intercept=None):
        context = self.context if context is None else context
        with contextlib.redirect_stdout(io.StringIO()):
            return screen.fit(args,context,'cv_0',self.train,self.target,context,intercept=intercept)

    def nested_equal(self, a, b):
        if isinstance(a,torch.Tensor):
            self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.nested_equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):self.nested_equal(left,right)
        else:self.assertEqual(a,b)

    def test_exact_eight_objectives_and_lambdas(self):
        self.assertEqual(list(screen.OBJECTIVES),['grt_wl','macro_hpwl','hpwl_edge','hpwl_group',
            'hpwl_dataflow','hpwl_channel','hpwl_expert_composite','grt_expert_composite'])
        self.assertEqual(len(screen.tasks()),24)
        self.assertEqual(screen.CONFIG,bounded.CONFIG)
        for values in screen.OBJECTIVES.values():self.assertAlmostEqual(sum(values),1.)
        self.assertEqual(screen.OBJECTIVES['hpwl_edge'],(0.,1/1.25,.25/1.25,0.,0.,0.))
        self.assertEqual(screen.OBJECTIVES['hpwl_expert_composite'],(0.,.8,.05,.05,.05,.05))
        self.assertNotIn('port_blockage_fraction',screen.rule_definition()['rules'])

    def test_scales_train_only_and_shared_across_all_objectives(self):
        specs = [screen.specification(self.context,self.train,o)[0] for o in screen.OBJECTIVES]
        for spec in specs:self.assertEqual(screen.scale_identity(specs[0]),screen.scale_identity(spec))
        altered = copy.deepcopy(self.context)
        altered['cost_values'][self.target] *= 1000
        for objective,spec in zip(screen.OBJECTIVES,specs):
            self.assertEqual(screen.specification(altered,self.train,objective)[0],spec)
        np.testing.assert_allclose(specs[0]['scales'],self.context['cost_values'][self.train].mean(0))

    def test_composite_is_equal_expert_mean_and_no_B_renormalization(self):
        spec,_ = screen.specification(self.context,self.train,'hpwl_expert_composite')
        out = screen.labels(self.context,self.train,spec)
        norm = self.context['cost_values'][self.train]/np.asarray(spec['scales'])
        c = (norm[:,1]+.25*norm[:,2:].mean(1))/1.25
        np.testing.assert_allclose(out['composite_cost'],c,rtol=1e-14)
        np.testing.assert_allclose(out['cost'],2*c/(1+c),rtol=1e-14)
        self.assertAlmostEqual(c.mean(),1.)
        self.assertFalse(np.isclose(float(out['cost'].mean()),1.,rtol=1e-10,atol=1e-10))
        self.assertTrue((out['utility']==-out['cost']).all())

    def test_executed_proxies_do_not_change_when_only_raw_changes(self):
        altered = copy.deepcopy(self.data)
        altered['train']['x'] *= -.4
        screen.add_costs(altered,self.cache)
        reference = copy.deepcopy(self.data)
        screen.add_costs(reference,self.cache)
        np.testing.assert_array_equal(altered['train']['cost_values'],reference['train']['cost_values'])
        altered['train']['executed'] = altered['train']['executed']*.5
        screen.add_costs(altered,self.cache)
        self.assertFalse(np.array_equal(altered['train']['cost_values'][:,1:],reference['train']['cost_values'][:,1:]))
        np.testing.assert_array_equal(altered['train']['cost_values'][:,0],reference['train']['cost_values'][:,0])

    def test_active_constant_or_unavailable_refused_without_changing_other_arms(self):
        context = copy.deepcopy(self.context)
        context['cost_values'][:,-1]=.1
        with self.assertRaisesRegex(ValueError,'constant'):
            screen.specification(context,self.train,'hpwl_channel')
        screen.specification(context,self.train,'grt_wl')
        context['cost_values'][:,-1]=np.nan
        context['cost_mask'][:,-1]=False
        with self.assertRaisesRegex(ValueError,'unavailable'):
            screen.specification(context,self.train,'grt_expert_composite')
        screen.specification(context,self.train,'macro_hpwl')

    def test_physical_failures_and_missing_feedback_not_fake_PPA(self):
        spec,_ = screen.specification(self.context,self.train,'hpwl_expert_composite')
        context = copy.deepcopy(self.context)
        rows = self.target[:3]
        context['statuses'][rows]=['physical_failure','infrastructure_failure','success']
        context['cost_values'][rows]=np.nan
        context['cost_mask'][rows]=False
        out = screen.labels(context,rows,spec)
        np.testing.assert_array_equal(out['valid'],[True,False,False])
        self.assertEqual(out['cost'][0],4.)
        self.assertTrue(torch.isnan(out['cost'][1:]).all())
        self.assertTrue(torch.isnan(out['composite_cost']).all())
        self.assertTrue(np.isfinite(context['native']).all())

    def test_pure_GRT_control_matches_original_bounded_labels(self):
        spec,_=screen.specification(self.context,self.train,'grt_wl')
        original,_=bounded.specification(self.context,self.train,'wl')
        np.testing.assert_allclose(screen.labels(self.context,self.target,spec)['cost'],
            bounded.labels(self.context,self.target,original)['cost'],rtol=1e-14,atol=1e-14)

    def test_private_engine_keeps_frozen_globals_and_same_minibatches(self):
        original_spec, original_labels, original_stop = bounded.specification,bounded.labels,bounded.STOP
        batches = {}
        for objective in screen.OBJECTIVES:
            batches[objective]=[]
            args=self.args(objective,objective,steps=2)
            done=self.fit(args,intercept=lambda x:batches[objective].append(x.clone()))
            self.assertEqual(done['backend_calls'],0)
            saved=torch.load(args.output/'cv_0/resume.pt',weights_only=False)
            self.assertEqual((saved['spec']['width'],saved['spec']['depth'],saved['spec']['ensemble_size']),(128,2,2))
            self.assertEqual(saved['actor_contract'],screen.ACTOR_CONTRACT)
            self.assertIs(bounded.specification,original_spec)
            self.assertIs(bounded.labels,original_labels)
            self.assertEqual(bounded.STOP,original_stop)
        for arm in list(screen.OBJECTIVES)[1:]:
            for a,b in zip(batches['grt_wl'],batches[arm]):self.assertTrue(torch.equal(a,b))

    def test_target_changes_do_not_affect_parameters_or_normalization(self):
        first=self.args('original',steps=3);done=self.fit(first)
        altered=copy.deepcopy(self.context);altered['cost_values'][self.target]*=1000
        second=self.args('altered',steps=3);other=self.fit(second,context=altered)
        self.assertEqual(done['normalization'],other['normalization'])
        self.assertEqual(done['utility_spec'],other['utility_spec'])
        a=torch.load(first.output/'cv_0/resume.pt',weights_only=False)
        b=torch.load(second.output/'cv_0/resume.pt',weights_only=False)
        self.nested_equal(a['model'],b['model'])
        with np.load(first.output/'cv_0/predictions.npz') as aa,np.load(second.output/'cv_0/predictions.npz') as bb:
            np.testing.assert_array_equal(aa['prediction'],bb['prediction'])
            self.assertFalse(np.array_equal(aa['truth'],bb['truth']))

    def test_exact_resume_and_changed_binding_refusal(self):
        full=self.args('full');self.fit(full)
        resumed=self.args('resumed');calls=[]
        def interrupt(_):
            calls.append(1)
            if len(calls)==2:screen._stop()
        with self.assertRaises(SystemExit) as stopped:self.fit(resumed,intercept=interrupt)
        self.assertEqual(stopped.exception.code,85)
        self.assertFalse((resumed.output/'cv_0/completed.json').exists())
        screen.STOP=False
        self.fit(resumed)
        a=torch.load(full.output/'cv_0/resume.pt',weights_only=False)
        b=torch.load(resumed.output/'cv_0/resume.pt',weights_only=False)
        for key in ['model','optimizer','rng','batch_rng','step','normalization','binding','spec','utility_spec']:
            self.nested_equal(a[key],b[key])
        broken=self.args('broken')
        with self.assertRaises(SystemExit):self.fit(broken,intercept=lambda _:screen._stop())
        screen.STOP=False
        altered=copy.deepcopy(self.context);altered['source_ids'][self.train[0]]='changed'
        with self.assertRaisesRegex(ValueError,'binding changed'):self.fit(broken,context=altered)

    def test_checked_model_last_step_buffers_and_finite_raw_gradient(self):
        args=self.args(steps=3);done=self.fit(args)
        checkpoint=args.output/'cv_0/resume.pt'
        model=screen.checked_model(self.context,done['utility_spec'],args.seed,self.train,checkpoint,done,expected_steps=3)
        raw=self.context['x'][self.target].clone().requires_grad_(True)
        q=screen.scalar_reward(model,raw)
        torch.testing.assert_close(q,-model(raw).mean((1,2)))
        gradient,=torch.autograd.grad(q.sum(),raw)
        self.assertTrue(torch.isfinite(gradient).all())
        with self.assertRaisesRegex(ValueError,'step'):
            screen.checked_model(self.context,done['utility_spec'],args.seed,self.train,checkpoint,done)
        saved=torch.load(checkpoint,weights_only=False)
        saved['model']['feature_mean']+=1
        torch.save(saved,checkpoint);done['checkpoint_sha256']=sha(checkpoint)
        with self.assertRaisesRegex(ValueError,'buffer changed'):
            screen.checked_model(self.context,done['utility_spec'],args.seed,self.train,checkpoint,done,expected_steps=3)

    def test_families_and_completed_tamper_stay_strict(self):
        altered=copy.deepcopy(self.context)
        altered['families'][self.train[0]]=altered['families'][self.target[0]]
        with self.assertRaisesRegex(ValueError,'family crosses'):self.fit(self.args(),context=altered)
        args=self.args();self.fit(args)
        path=args.output/'cv_0/predictions.npz';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.fit(args)

    def test_full_four_fit_refit_interface_and_strict_completed_binding(self):
        task=screen.tasks()[0]
        config=dict(screen.CONFIG,steps=3)
        folds=screen.folds_for(self.data)
        output=self.root/'study'
        plan=dict(schema=screen.SCHEMA,output=str(output),inputs=dict(dataset_sha256='a'*64),
            tasks=screen.tasks(),folds=folds,source_sha256={},fold_plan_sha256=screen.json_sha(folds))
        path=self.root/'plan.json';path.write_text(json.dumps(plan))
        context=copy.deepcopy(self.context)
        args=SimpleNamespace(**config,**task,output=output/task['objective']/f"seed_{task['seed']}")
        args.output.mkdir(parents=True)
        context['binding']=screen.run_binding(plan,path,task)
        outcomes=[]
        with mock.patch.dict(screen.CONFIG,config,clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                for fold in folds:
                    target=self.data['validation'] if fold['name']=='refit' else context
                    outcomes.append(screen.fit(args,context,fold['name'],fold['train_indices'],fold['target_indices'],target))
            summary=dict(status='completed',binding=context['binding'],objective=task['objective'],seed=task['seed'],
                outcomes=outcomes,**screen.summarize(outcomes))
            screen.bounded.atomic_json(args.output/'summary.json',summary)
            screen.bounded.atomic_json(args.output/'completed.json',dict(status='completed',binding=context['binding'],
                summary_sha256=sha(args.output/'summary.json'),backend_calls=0,actor_advanced=False))
            result=screen.checked_refit(path,task['objective'],task['seed'],loaded=(plan,self.data,context))
            self.assertTrue({'model','spec','context','checkpoint','completion','evidence','normalization'}<=result.keys())
            self.assertEqual(result['spec'],screen.specification(context,list(range(12)),'grt_wl')[0])
            self.assertEqual(len(result['evidence']),15)
            completed=args.output/'cv_0/completed.json'
            value=json.loads(completed.read_text());value['steps']=1
            completed.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'fit TRAIN/target/step/source'):
                screen.checked_refit(path,task['objective'],task['seed'],loaded=(plan,self.data,context))


if __name__=='__main__':unittest.main()
