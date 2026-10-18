"""Bounded Slurm-only tests: exact comparison rows, executed inputs and resume."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import qstudy_executed_control as control
from . import qstudy_screen as screen


class ExecutedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run synthetic model tests through Slurm')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='qstudy-executed-test-',
            dir=os.environ.get('TMPDIR','/scratch/bohanli/flowplace_graph'))
        self.root = Path(self.temporary.name)
        self.cache = dict(node_id_to_name=['a','b','c'], ratio_x=10/224, ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.2) for n in 'abc'},
            net_info={'n':{'nodes':{'a':{'x_offset':.2},'b':{},'c':{'y_offset':-.3}}}})
        raw = torch.rand((16,3,2),generator=torch.Generator().manual_seed(19))*.9-.45
        executed = raw.clone()
        executed[:,0,0] = .35*raw[:,0,0] + .1
        executed[:,1,1] = -.2*raw[:,1,1] - .1
        costs = torch.stack((100+3*executed[:,0,0], .4+.1*executed[:,1,0],
            25+executed[:,2,1], 1+(3*executed[:,0,1]).clamp_min(0),
            1+(4*executed[:,1,1]).clamp_min(0)), -1)
        native = np.column_stack((costs.numpy(),np.ones((16,2))))
        native[:,1:3] *= -1
        self.data = {}
        for split, indices, families in [('train',np.arange(12),np.repeat(np.arange(6),2)),
                ('validation',np.arange(12,16),np.repeat([100,101],2))]:
            self.data[split] = dict(raw_action=raw[indices].flatten(1).numpy(),
                executed_action=executed[indices].flatten(1).numpy(), metrics=native[indices],
                metric_mask=np.ones((len(indices),7),dtype=bool),family=families,
                status=np.full(len(indices),'success'))
        self.arrays,self.accounting = control.assemble_arrays(self.data)
        self.dataset = self.root/'data_audit/bp_fe384/dataset.npz'
        self.dataset.parent.mkdir(parents=True)
        self.dataset.write_bytes(b'synthetic-audited-dataset')
        self.dataset.with_name('manifest.json').write_text('{"synthetic":true}')
        self.dataset.with_name('rows.jsonl').write_text('{"synthetic_provenance":true}\n')
        self.sources = control.source_bindings()
        self.config = dict(steps=6,folds=3,batch_size=8,lr=3e-4,weight_decay=1e-4)
        self.matrix = self.root/'screen'
        self.raw_output = self.matrix/'bp_fe384/huber_residual/seed_0'
        self.task = dict(dataset=str(self.dataset),dataset_sha256=control.sha(self.dataset),
            manifest_sha256=control.sha(self.dataset.with_name('manifest.json')),
            rows_sha256=control.sha(self.dataset.with_name('rows.jsonl')),
            cache_sha256=control.sha(control.ROOT/'baseline/runtime/benchmark_cache/bp_fe.pt'),
            design='bp_fe',seed=0,variant='huber_residual',config=self.config,
            input_representation=control.INPUT,folds=control.folds(self.arrays),accounting=self.accounting,
            output=str(self.root/'executed'),raw_output=str(self.raw_output))
        self.plan_path=self.root/'plan.json'
        self.plan_path.write_text('{"synthetic_plan":true}')
        self.plan=dict(source_sha256=self.sources)
        self.movable=torch.ones(3,dtype=torch.bool)
        screen.STOP=False

    def tearDown(self):
        screen.STOP=False
        self.temporary.cleanup()

    def comparator(self, capture=None):
        args=SimpleNamespace(output=self.raw_output,variant='huber_residual',seed=0,**self.config)
        binding=dict(dataset_sha256=self.task['dataset_sha256'],manifest_sha256=self.task['manifest_sha256'],
            cache_sha256=self.task['cache_sha256'],config={k:str(v) if isinstance(v,Path) else v
                for k,v in vars(args).items()},source_sha256={k:v for k,v in self.sources.items()
                if k in ('fql/qstudy_screen.py','fql/qstudy_models.py','fql/qstudy_data.py',
                         'fql/graph_o2o_models.py','fql/physical_ridge.py','fql/analytic.py')},
            fold_seed=41307,heldout_used=False,backend_calls=0)
        binding['config']['dataset']=str(self.dataset)
        raw_arrays={split:dict(values,x=values['raw_x']) for split,values in self.arrays.items()}
        context=dict(**raw_arrays['train'],cache=self.cache,movable=self.movable,binding=binding)
        outcomes=[]
        self.raw_output.mkdir(parents=True,exist_ok=True)
        control.atomic_json(self.raw_output/'binding.json',binding)
        with contextlib.redirect_stdout(io.StringIO()),self.capture_batches(capture):
            for fold in self.task['folds']:
                outcomes.append(screen.fit(args,context,fold['name'],np.asarray(fold['train_indices']),
                    np.asarray(fold['target_indices']),raw_arrays[fold['split']]))
        control.atomic_json(self.raw_output/'completed.json',dict(status='completed',binding=binding,outcomes=outcomes))
        self.task['raw_comparison']=control.raw_comparisons(self.task,self.arrays,self.sources)

    @contextlib.contextmanager
    def capture_batches(self, capture=None, interrupt_after=None):
        original=screen.make_model
        def make(*args,**kwargs):
            model,normalization=original(*args,**kwargs)
            loss=model.loss
            def recorded(x,*rest,**options):
                if capture is not None:
                    capture.append(x.detach().clone())
                    if interrupt_after is not None and len(capture)==interrupt_after:
                        screen._stop(signal.SIGUSR1,None)
                return loss(x,*rest,**options)
            model.loss=recorded
            return model,normalization
        with mock.patch.object(screen,'make_model',side_effect=make):
            yield

    def run_task(self, task=None, capture=None, interrupt_after=None):
        Path((task or self.task)['output']).mkdir(parents=True,exist_ok=True)
        original=torch.load
        def load(path,*args,**kwargs):
            if str(path).endswith('/baseline/runtime/benchmark_cache/bp_fe.pt'):
                return self.cache
            return original(path,*args,**kwargs)
        problem=SimpleNamespace(cond=SimpleNamespace(is_ports=~self.movable))
        with mock.patch.object(control.torch,'load',side_effect=load), \
             mock.patch('baseline.api.load_problem',return_value=problem), \
             self.capture_batches(capture,interrupt_after),contextlib.redirect_stdout(io.StringIO()):
            control._run_task(self.plan_path,self.plan,task or self.task,self.arrays)

    def nested_equal(self,a,b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a,b),'Checkpoint tensor differs')
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.nested_equal(a[key],b[key])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.nested_equal(x,y)
        else:self.assertEqual(a,b)

    def test_assembly_uses_executed_coordinates_and_preserves_failure_accounting(self):
        data=copy.deepcopy(self.data)
        failed={k:np.concatenate((v,v[:1]),axis=0) for k,v in data['train'].items()}
        failed['metrics'][-1]=np.nan
        failed['metric_mask'][-1]=False
        failed['status']=failed['status'].astype('<U20')
        failed['status'][-1]='physical_failure'
        data['train']=failed
        arrays,account=control.assemble_arrays(data)
        torch.testing.assert_close(arrays['train']['x'].flatten(1),torch.tensor(self.data['train']['executed_action']))
        self.assertFalse(torch.equal(arrays['train']['x'],arrays['train']['raw_x']))
        self.assertTrue((arrays['train']['costs'][:,1:3]>0).all())
        self.assertEqual(account['train']['original_rows'],13)
        self.assertEqual(account['train']['excluded_source_indices'],[12])
        self.assertEqual(account['train']['status_counts']['physical_failure'],1)
        self.assertEqual(len(arrays['train']['source_indices']),12)

    def test_nonfinite_inputs_and_family_leak_are_refused(self):
        for key in ('raw_action','executed_action'):
            changed=copy.deepcopy(self.data)
            changed['train'][key][0,0]=np.nan
            with self.assertRaisesRegex(ValueError,'must be finite'):control.assemble_arrays(changed)
        changed=copy.deepcopy(self.data)
        changed['validation']['family'][0]=0
        with self.assertRaisesRegex(ValueError,'family leakage'):control.assemble_arrays(changed)

    def test_folds_and_all_four_raw_prediction_bindings_are_exact(self):
        self.comparator()
        self.assertEqual(set(self.task['raw_comparison']['fits']),{'cv_0','cv_1','cv_2','refit'})
        seen=[]
        for fold in self.task['folds'][:3]:
            self.assertFalse(set(self.arrays['train']['families'][fold['train_indices']].tolist()) &
                set(self.arrays['train']['families'][fold['target_indices']].tolist()))
            seen+=fold['target_indices']
        self.assertEqual(sorted(seen),list(range(12)))
        control.raw_comparisons(self.task,self.arrays,self.sources)
        changed=copy.deepcopy(self.arrays)
        changed['train']['costs'][0,0]+=1
        with self.assertRaises(ValueError):control.raw_comparisons(self.task,changed,self.sources)
        changed=copy.deepcopy(self.task)
        changed['config']['steps']+=1
        with self.assertRaisesRegex(ValueError,'budget/fold/configuration'):control.raw_comparisons(changed,self.arrays,self.sources)
        changed=dict(self.sources)
        changed['fql/qstudy_screen.py']='wrong'
        with self.assertRaisesRegex(ValueError,'frozen source'):control.raw_comparisons(self.task,self.arrays,changed)

    def test_raw_artifact_tampering_is_refused(self):
        self.comparator()
        path=self.raw_output/'cv_0/resume.pt'
        path.write_bytes(path.read_bytes()+b'tampered')
        with self.assertRaisesRegex(ValueError,'checkpoint SHA'):control.raw_comparisons(self.task,self.arrays,self.sources)

    def test_changed_target_normalization_is_refused(self):
        self.comparator()
        path=self.raw_output/'cv_0/completed.json'
        done=control.read_json(path)
        done['normalization']['target_mean'][0]+=1
        control.atomic_json(path,done)
        overall=control.read_json(self.raw_output/'completed.json')
        overall['outcomes'][0]=done
        control.atomic_json(self.raw_output/'completed.json',overall)
        with self.assertRaisesRegex(ValueError,'TRAIN-only target normalization'):
            control.raw_comparisons(self.task,self.arrays,self.sources)

    def test_paired_batches_and_train_only_target_scales_match_raw_control(self):
        raw_batches=[]
        self.comparator(raw_batches)
        executed_batches=[]
        self.run_task(capture=executed_batches)
        self.assertEqual(len(raw_batches),24)
        self.assertEqual(len(executed_batches),24)
        raw=self.arrays['train']['raw_x']
        executed=self.arrays['train']['x']
        for before,after in zip(raw_batches,executed_batches):
            for r,e in zip(before,after):
                row=torch.nonzero((raw==r).flatten(1).all(1)).item()
                torch.testing.assert_close(e,executed[row],atol=0,rtol=0)
        for fold in self.task['folds']:
            original=torch.load(self.raw_output/fold['name']/'resume.pt',weights_only=False)
            new=torch.load(Path(self.task['output'])/fold['name']/'resume.pt',weights_only=False)
            torch.testing.assert_close(original['model']['target_mean'],new['model']['target_mean'],atol=0,rtol=0)
            torch.testing.assert_close(original['model']['objective_scale'],new['model']['objective_scale'],atol=0,rtol=0)
            self.assertFalse(torch.equal(original['model']['feature_mean'],new['model']['feature_mean']))
            self.assertEqual(new['binding']['input_representation'],'legalizer_output_pre_DEF')

    def test_signal_interruption_resume_matches_uninterrupted_tensors_optimizer_rng(self):
        self.comparator()
        self.run_task()
        resumed=copy.deepcopy(self.task)
        resumed['output']=str(self.root/'resumed')
        with self.assertRaises(SystemExit) as stopped:
            self.run_task(resumed,capture=[],interrupt_after=3)
        self.assertEqual(stopped.exception.code,85)
        checkpoint=torch.load(Path(resumed['output'])/'cv_0/resume.pt',weights_only=False)
        self.assertEqual(checkpoint['step'],3)
        self.assertFalse((Path(resumed['output'])/'cv_0/completed.json').exists())
        screen.STOP=False
        self.run_task(resumed)
        for fold in self.task['folds']:
            a=torch.load(Path(self.task['output'])/fold['name']/'resume.pt',weights_only=False)
            b=torch.load(Path(resumed['output'])/fold['name']/'resume.pt',weights_only=False)
            for key in ('model','optimizer','rng','batch_rng','step','normalization','train_indices'):
                self.nested_equal(a[key],b[key])

    def test_completed_reentry_is_immutable_and_rejects_changed_prediction_bytes(self):
        self.comparator()
        self.run_task()
        folder=Path(self.task['output'])
        digest=control.sha(folder/'completed.json')
        self.run_task()
        self.assertEqual(control.sha(folder/'completed.json'),digest)
        path=folder/'cv_0/predictions.npz'
        path.write_bytes(path.read_bytes()+b'tampered')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.run_task()

    def test_make_task_binds_exact_source_rows_and_detects_prediction_changes(self):
        self.comparator()
        with mock.patch.object(control,'load_dataset',return_value=(self.data,{'design':'bp_fe'})):
            task,_=control.make_task(self.dataset,0,self.task['output'],self.matrix,self.sources,self.config)
            self.assertEqual(task,self.task)
            path=self.raw_output/'cv_0/predictions.npz'
            with np.load(path,allow_pickle=False) as p:values={k:p[k].copy() for k in p.files}
            values['prediction'][0,0]+=1
            np.savez_compressed(path,**values)
            changed,_=control.make_task(self.dataset,0,self.task['output'],self.matrix,self.sources,self.config)
            self.assertNotEqual(changed['raw_comparison'],task['raw_comparison'])

    def test_public_run_forwards_usr1_and_refuses_postbackend_input(self):
        self.comparator()
        task=copy.deepcopy(self.task)
        task['config']=control.CONFIG
        tasks=[]
        for dataset in control.DATASETS:
            for seed in (0,1,2):
                item=copy.deepcopy(task)
                item['seed']=seed
                item['dataset']=str(self.root/'data_audit'/dataset/'dataset.npz')
                item['output']=str(self.root/dataset/f'seed_{seed}')
                item['raw_output']=str(self.matrix/dataset/control.VARIANT/f'seed_{seed}')
                tasks.append(item)
        task=tasks[0]
        control.atomic_json(self.root/'tasks.json',tasks)
        plan=dict(task_count=6,datasets=list(control.DATASETS),seeds=[0,1,2],variant=control.VARIANT,
            config=control.CONFIG,input_representation=control.INPUT,source_sha256=self.sources,
            raw_matrix=str(self.matrix),data_root=str(self.root/'data_audit'),
            tasks_sha256=control.sha(self.root/'tasks.json'))
        control.atomic_json(self.plan_path,plan)
        handlers={}
        def register(sig,handler):handlers[sig]=handler
        def inspect(*_):
            self.assertIs(handlers[signal.SIGUSR1],screen._stop)
            handlers[signal.SIGUSR1](signal.SIGUSR1,None)
            self.assertTrue(screen.STOP)
        with mock.patch.object(control,'make_task',return_value=(task,self.arrays)), \
             mock.patch.object(control.signal,'signal',side_effect=register), \
             mock.patch.object(control,'_run_task',side_effect=inspect):
            control.run(self.plan_path,0)
        changed=copy.deepcopy(task)
        changed['accounting']['train']['array_sha256']['x']='changed_executed_pre_DEF_coordinates'
        with mock.patch.object(control,'make_task',return_value=(changed,self.arrays)), \
             mock.patch.object(control,'_run_task') as fitter:
            with self.assertRaisesRegex(ValueError,'Declared data/folds/labels'):
                control.run(self.plan_path,0)
            fitter.assert_not_called()
        tasks[0]['input_representation']='backend_actual_coordinates'
        control.atomic_json(self.root/'tasks.json',tasks)
        plan['tasks_sha256']=control.sha(self.root/'tasks.json')
        control.atomic_json(self.plan_path,plan)
        with self.assertRaisesRegex(ValueError,'pre-DEF'):control.run(self.plan_path,0)
        with self.assertRaisesRegex(ValueError,'outside declared bounds'):control.run(self.plan_path,6)


if __name__=='__main__':
    unittest.main()
