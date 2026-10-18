"""Check pin geometry/units and action derivatives, not physical quality."""
import copy
import os
import unittest

import torch

from .qstudy_pin_control import PinFeatureCritic


class PinFeatureTests(unittest.TestCase):
    @staticmethod
    def cache(offset=.2):
        return dict(node_id_to_name=['a','b','c'], ratio_x=10/224, ratio_y=20/224,
            node_info={n:dict(x=1.,y=1.) for n in 'abc'},
            net_info={'ab':{'nodes':{'a':dict(x_offset=offset),'b':{}}},
                      'bc':{'nodes':{'b':{},'c':dict(y_offset=-.3)}}})

    def setUp(self):
        self.x = torch.rand(8,3,2,generator=torch.Generator().manual_seed(19),dtype=torch.float64)-.5
        self.y = torch.stack((10+self.x[:,0,0], .4+.1*self.x[:,1,0],
                              30+2*self.x[:,2,1]), -1)

    def model(self, cache=None):
        return PinFeatureCritic(cache or self.cache(), self.x, self.y,
            torch.ones(3,dtype=torch.bool), ['rWL_um','WNS_ns','TNS_ns'],
            width=12, depth=2, seed=9)

    def test_pin_offsets_and_normalized_canvas_units(self):
        model=self.model()
        spans=model.pin_spans(self.x)
        expected=torch.stack(((self.x[:,0]+torch.tensor([.04,0.])-self.x[:,1]).abs(),
            (self.x[:,1]-self.x[:,2]-torch.tensor([0.,-.03])).abs()), dim=1)
        torch.testing.assert_close(spans,expected)
        physical_hpwl=(spans*torch.tensor([5.,10.])).sum((1,2))
        torch.testing.assert_close(physical_hpwl,model.extractor.hpwl(self.x))
        self.assertEqual(model.features(self.x).shape,(8,17+6+8))
        self.assertEqual(model.spec['added_feature_count'],8)

    def test_net_identity_survives_equal_total_hpwl(self):
        cache=self.cache(0.)
        cache['net_info']['bc']['nodes']['c']={}
        model=self.model(cache)
        positions=torch.tensor([[[-.8,0.],[0.,0.],[.8,0.]],
                                [[-.8,0.],[-.4,0.],[.8,0.]]],dtype=torch.float64)
        spans=model.pin_spans(positions)
        torch.testing.assert_close(spans.sum((1,2))[0],spans.sum((1,2))[1])
        self.assertFalse(torch.equal(spans[0],spans[1]))

    def test_coordinate_gradient_matches_finite_difference(self):
        model=self.model()
        value=self.x[:1].clone().requires_grad_()
        loss=model.pin_spans(value).square().sum()
        gradient=torch.autograd.grad(loss,value)[0]
        epsilon=1e-6
        plus=value.detach().clone();minus=plus.clone()
        plus[0,0,0]+=epsilon;minus[0,0,0]-=epsilon
        difference=(model.pin_spans(plus).square().sum()-model.pin_spans(minus).square().sum())/(2*epsilon)
        torch.testing.assert_close(gradient[0,0,0],difference,rtol=1e-5,atol=1e-7)
        prediction=model(value).sum()
        self.assertTrue(torch.isfinite(torch.autograd.grad(prediction,value)[0]).all())

    def test_stats_frozen_and_checkpoint_spec_reloads(self):
        model=self.model()
        old=copy.deepcopy(model.normalization_metadata())
        model(self.x+5)
        self.assertEqual(old,model.normalization_metadata())
        repeated=self.model()
        repeated.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(self.x),repeated(self.x))

    def test_empty_netlist_has_zero_extra_features(self):
        cache=self.cache();cache['net_info']={}
        model=self.model(cache)
        self.assertEqual(model.pin_spans(self.x).shape,(8,0,2))
        self.assertEqual(model.features(self.x).shape,(8,23))
        self.assertTrue(torch.isfinite(model(self.x)).all())


if __name__=='__main__':
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run numerical tests via Slurm')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    unittest.main()
