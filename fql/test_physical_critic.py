import unittest
import numpy as np
import torch
from .physical_critic_diagnostic import fit_ridge, predict, choose_penalty
from .physical_critic_diagnostic import physical_features
from .physical_ridge import PhysicalRidgeCritic
from analysis.placement_metrics import MacroHPWL


class PhysicalCriticTests(unittest.TestCase):
    def test_torch_features_and_coordinate_gradient(self):
        cache={'node_id_to_name':['a','b','c'],'ratio_x':1.,'ratio_y':1.,
            'node_info':{n:{'x':10.+i,'y':12.+i} for i,n in enumerate(['a','b','c'])},
            'net_info':{'net':{'nodes':{'a':{'x_offset':2.},'b':{},'c':{}}}}}
        rng=np.random.default_rng(9)
        actions=rng.uniform(-.7,.7,(8,6))
        features,_=physical_features(actions,MacroHPWL(cache))
        fitted=fit_ridge(features,rng.normal(size=8),1.)
        critic=PhysicalRidgeCritic(cache,fitted)
        x=torch.tensor(actions,dtype=torch.float64,requires_grad=True)
        np.testing.assert_allclose(critic.features(x).detach().numpy(),features,atol=1e-12)
        np.testing.assert_allclose(critic(x).detach().numpy()[:,0],predict(fitted,features),atol=1e-12)
        derivative=torch.autograd.grad(critic(x).sum(),x)[0].numpy()
        eps=1e-6;plus=actions.copy();minus=actions.copy();plus[0,0]+=eps;minus[0,0]-=eps
        fp=predict(fitted,physical_features(plus,MacroHPWL(cache))[0])[0]
        fm=predict(fitted,physical_features(minus,MacroHPWL(cache))[0])[0]
        self.assertAlmostEqual(derivative[0,0],(fp-fm)/(2*eps),places=6)

    def test_train_only_standardization_and_constant_feature(self):
        x=np.column_stack([np.arange(10),np.ones(10)])
        y=2*np.arange(10)+3
        model=fit_ridge(x,y,1e-8)
        np.testing.assert_allclose(predict(model,x),y,atol=1e-6)
        np.testing.assert_allclose(model['mean'],[4.5,1.])
        self.assertTrue(np.isfinite(predict(model,np.array([[1000.,1.]]))).all())

    def test_folds_keep_each_family_together(self):
        groups=np.repeat(np.arange(10),2)
        x=np.arange(20)[:,None].astype(float)
        _,_,folds=choose_penalty(x,x[:,0],groups,[.1,1.])
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups==group])),1)


if __name__=='__main__':
    unittest.main()
