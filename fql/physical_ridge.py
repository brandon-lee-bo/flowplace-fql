"""Differentiable raw-proposal physical-feature critic, without legalization.

The fitted ridge predicts the measured reward after external legalization.
Its derivatives concern the raw features only, not a legalizer Jacobian.
"""
import numpy as np
import torch
from torch import nn
from .analytic import DifferentiableMacroHPWL


class PhysicalRidgeCritic(nn.Module):
    def __init__(self, cache, model):
        super().__init__()
        self.hpwl = DifferentiableMacroHPWL(cache)
        for key in ['mean','std','coeff','intercept']:
            self.register_buffer('ridge_'+key,torch.as_tensor(model[key],dtype=torch.float64))
        pairs=torch.triu_indices(len(self.hpwl.names),len(self.hpwl.names),offset=1)
        self.register_buffer('pair_i',pairs[0]);self.register_buffer('pair_j',pairs[1])
        self.register_buffer('grid_centers',torch.tensor([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]],dtype=torch.float64))

    def features(self, action):
        x=action.reshape(len(action),-1,2)
        sizes=self.hpwl.sizes.to(x.dtype)
        area=sizes.prod(-1)
        centroid=(x*area[None,:,None]).sum(1)/area.sum()
        span=x.amax(1)-x.amin(1)
        radius=((x-centroid[:,None]).square().sum(-1)*area).sum(-1)/area.sum()
        gaps=torch.minimum(x-sizes/2+1,1-x-sizes/2)
        dist=(x[:,self.pair_i]-x[:,self.pair_j]).abs()
        l1=dist.sum(-1)
        separation=dist-(sizes[self.pair_i]+sizes[self.pair_j])/2
        overlap=(-separation).clamp_min(0).prod(-1).sum(-1)
        density=(torch.exp(-(x[:,:,None]-self.grid_centers.to(x.dtype)).square().sum(-1)/.5)*area[None,:,None]).sum(1)/area.sum()
        hpwl=self.hpwl(action)/self.hpwl.canvas.to(x.dtype).sum()
        return torch.cat([hpwl[:,None],centroid,span,radius[:,None],
            gaps.mean((1,2))[:,None],gaps.amin((1,2))[:,None],(-gaps).clamp_min(0).sum((1,2))[:,None],
            l1.mean(1)[:,None],l1.std(1,unbiased=False)[:,None],l1.amin(1)[:,None],overlap[:,None],density],dim=-1)

    def forward(self, action):
        f=self.features(action)
        mean,std,coeff=[getattr(self,'ridge_'+k).to(f.dtype) for k in ['mean','std','coeff']]
        return (((f-mean)/std*coeff).sum(-1)+self.ridge_intercept.to(f.dtype))[:,None]
