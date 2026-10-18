"""Small executable-contract fixtures; no model forward, training or backend."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

PATH=Path(__file__).with_name('qstudy_scalar_online_control_continue.py')
spec=importlib.util.spec_from_file_location('control_continuation_under_test',PATH)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.old=dict(output=str(self.root/'base'),arms={a:driver.base.settings(a) for a in driver.base.ARMS},
            diagnostic_interval=100,streams={str(r):driver.base.streams(r) for r in (0,1,2)})
        self.plan=dict(output=str(self.root/'addendum'))
    def tearDown(self):self.tmp.cleanup()
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return driver.bound(path)
    def physical(self,key,seed,round_number):
        exp=driver.base.experiment(self.old['output'],key)
        self.write(exp/'config.json',dict(key=key));self.write(exp/'declaration.json',{})
        tasks=exp/f'seed_{seed}/round_{round_number}/requests/tasks.jsonl';tasks.parent.mkdir(parents=True,exist_ok=True)
        tasks.write_text('')
        self.write(tasks.with_name('completed.json'),dict(tasks_sha256=driver.sha(tasks),
            config_sha256=driver.sha(exp/'config.json'),count=0))
        return exp
    def test_private_bookkeeping_does_not_mutate_stage0(self):
        self.assertEqual(driver.base.PHASES,('prepare0','export0','native0'))
        self.assertEqual(driver.dispatch['PHASES'],driver.PHASES)
        self.assertIsNot(driver.base.stage_submit.__globals__,driver.dispatch['stage_submit'].__globals__)
        self.assertEqual(driver.dispatch['stage_submit'].__code__,driver.base.stage_submit.__code__)
        self.assertNotEqual(driver.base.VERSION,driver.VERSION)
        self.assertEqual(len(driver.PHASES),9)
        self.assertFalse(set(driver.PHASES)&set(driver.FUTURE_PHASES))
        self.assertEqual(driver.FUTURE_PHASES,('route_prepare','route'))
    def test_true_shared_T0_requests_for_three_FQL_branches_and_ownBC(self):
        for arm in driver.base.ARMS:
            key=driver.base.parent_key(arm,1);self.physical(key,0,0)
            with patch.object(driver,'original',return_value=self.old):
                binding=driver.training_request(self.plan,arm,0,1)
            request=driver.read(binding['path'])
            self.assertEqual(Path(request['experiment']).name,'bc_t0' if arm=='bc' else 'fql_t0')
            self.assertEqual(request['settings'],self.old['arms'][arm])
            self.assertEqual(request['settings']['total_updates'],600)
            self.assertNotIn('cycle',request)
    def test_second_request_points_to_genuine_branch_child(self):
        for arm in driver.base.ARMS:
            self.physical(arm+'_t1',2,1)
            with patch.object(driver,'original',return_value=self.old):
                binding=driver.training_request(self.plan,arm,2,2)
            request=driver.read(binding['path'])
            self.assertEqual(request['cycle'],2)
            self.assertEqual(Path(request['experiment']).name,arm+'_t1')
            self.assertEqual(Path(request['output']).parts[-3:],('cycle_2',arm,'seed_2'))
    def test_child_prepare_contracts_use_real_checkpoint_paths_and_roles(self):
        for stage in (1,2):
            for arm in driver.base.ARMS:
                def parents(plan,name,seed,cycle):
                    folder=self.root/f'{name}_{cycle}_{seed}';folder.mkdir()
                    request=self.write(folder/'request.json',{})
                    checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'actual file identity fixture')
                    return dict(request=request,checkpoint=driver.bound(checkpoint))
                with patch.object(driver,'original',return_value=self.old),patch.object(driver,'parent_training',side_effect=parents):
                    identity=driver.prepare_request(self.plan,arm+'_t'+str(stage))
                request=driver.read(identity['path'])
                self.assertEqual(request['logical_round'],stage)
                self.assertEqual(set(request['parent_training']),{'0','1','2'})
                self.assertEqual(set(request['streams']['0']),{'collect','evaluation'} if stage==1 else {'evaluation'})
                self.assertEqual(request['streams']['0']['evaluation'],dict(seed=81400000,count=8))
    def pair(self):
        root=dict(weight=torch.tensor([1.]),step=1000)
        replay=dict(count=8,source='same actual own C0')
        payload=dict(step=600,parent_steps=1000,critic=dict(w=torch.tensor([2.])),flow=dict(w=torch.tensor([3.])),
            optimizers=dict(critic=dict(state={1:dict(step=torch.tensor(600.),exp_avg=torch.tensor([.1]))}),flow=dict(state={})),
            rng=torch.tensor([3],dtype=torch.uint8),cuda_rng=[torch.tensor([4],dtype=torch.uint8)],
            generators={k:torch.tensor([i],dtype=torch.uint8) for i,k in enumerate(('actor','q','flow'))},
            history=[dict(critic_updates=1,teacher_updates=0,qloss=float(i)) for i in range(600)])
        a=dict(payload=payload,source=dict(payload=root),identity=dict(continuation_binding=dict(replay=replay,
            settings=driver.base.settings('updated_q10')),state_audit=dict(continued_actor_updates=600,continued_critic_updates=600)))
        b=copy.deepcopy(a);b['identity']['continuation_binding']['settings']=driver.base.settings('updated_q30')
        return a,b
    def check_pair(self,pair):
        from fql import qstudy_scalar_online_child_export as child
        with patch.object(driver,'original',return_value=self.old),patch.object(driver,'parent_training',return_value={}),\
            patch.object(child,'checked_parent',side_effect=pair):
            return driver.gate1(self.plan,0)
    def test_exact_critic_Adam_RNG_gate_allows_actor_contrast(self):
        pair=self.pair();pair[0]['payload']['actor']={'w':torch.tensor([1.])};pair[1]['payload']['actor']={'w':torch.tensor([9.])}
        result=self.check_pair(pair);self.assertEqual(driver.read(result['path'])['actor_adam_steps'],1600)
    def test_gate_refuses_different_critic_Adam_or_named_Q_stream(self):
        for field in ('critic','adam','rng'):
            pair=self.pair()
            if field=='critic':pair[1]['payload']['critic']['w']+=1
            elif field=='adam':pair[1]['payload']['optimizers']['critic']['state'][1]['exp_avg']+=1
            else:pair[1]['payload']['generators']['q']+=1
            with self.assertRaises((ValueError,AssertionError)):self.check_pair(pair)
    def test_gate_refuses_same_final_state_but_different_Q_trajectory(self):
        pair=self.pair();pair[1]['payload']['history'][25]['qloss']+=1
        with self.assertRaises(ValueError):self.check_pair(pair)
    def test_gate_refuses_wrong_update_count_or_replay(self):
        for field in ('count','replay'):
            pair=self.pair()
            if field=='count':pair[1]['payload']['step']=599
            else:pair[1]['identity']['continuation_binding']['replay']['source']='different collect'
            with self.assertRaises((ValueError,AssertionError)):self.check_pair(pair)
    def test_gate_refuses_two_aliased_or_swapped_alpha_arms(self):
        pair=self.pair();pair[1]['identity']['continuation_binding']['settings']=driver.base.settings('updated_q10')
        with self.assertRaises(ValueError):self.check_pair(pair)
        pair=self.pair()
        pair[0]['identity']['continuation_binding']['settings'],pair[1]['identity']['continuation_binding']['settings']=\
            pair[1]['identity']['continuation_binding']['settings'],pair[0]['identity']['continuation_binding']['settings']
        with self.assertRaises(ValueError):self.check_pair(pair)


if __name__=='__main__':unittest.main(verbosity=2)
