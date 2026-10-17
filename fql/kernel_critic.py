"""Small-data kernel critic with explicit TRAIN-only hyperparameter selection.

The posterior uncertainty is a model quantity, not a guaranteed error bound.
Both mean and mean-minus-standard-deviation support action gradients; neither
claims a derivative through the external placement legalizer.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
import torch
from torch import nn

from .core import critic_diagnostics
from .physical_ridge import PhysicalRidgeCritic


def kernel(a, b, length):
    distance = np.square(a[:, None] - b[None]).mean(-1)
    return np.exp(-distance/(2*length**2))


def fit(features, reward, length, noise):
    features = np.asarray(features, dtype=float)
    reward = np.asarray(reward, dtype=float)
    mean, std = features.mean(0), features.std(0)
    keep = np.flatnonzero(std > 1e-5)
    if not len(keep):
        raise ValueError('Kernel critic requires variable TRAIN features')
    z = (features[:, keep]-mean[keep])/std[keep]
    center, scale = float(reward.mean()), max(float(reward.std()), .03)
    matrix = kernel(z, z, length) + (noise+1e-8)*np.eye(len(z))
    cholesky, _ = cho_factor(matrix, lower=True)
    cholesky = np.tril(cholesky)
    coeff = cho_solve((cholesky, True), (reward-center)/scale)
    return dict(mean=mean, std=std, keep=keep, train_z=z, coeff=coeff,
        cholesky=cholesky, reward_center=np.asarray(center), reward_scale=np.asarray(scale),
        length=np.asarray(length), noise=np.asarray(noise))


def predict(model, features):
    keep = model['keep']
    z = (features[:, keep]-model['mean'][keep])/model['std'][keep]
    k = kernel(z, model['train_z'], float(model['length']))
    mean = float(model['reward_center']) + float(model['reward_scale'])*(k@model['coeff'])
    projected = solve_triangular(model['cholesky'], k.T, lower=True)
    variance = np.maximum(1+float(model['noise'])-np.square(projected).sum(0), 1e-12)
    return mean, float(model['reward_scale'])*np.sqrt(variance)


def select(feature_sets, reward, families):
    unique = np.unique(families)
    if len(unique) < 5:
        raise ValueError('At least five independent TRAIN families required')
    shuffled = np.random.default_rng(20261018).permutation(unique)
    mapping = {group:i % 5 for i, group in enumerate(shuffled)}
    folds = np.asarray([mapping[group] for group in families])
    scores = []
    for name, x in feature_sets.items():
        for length in [.5, 1., 2., 4.]:
            for noise in [.01, .1, 1.]:
                prediction, uncertainty = np.empty(len(reward)), np.empty(len(reward))
                for fold in range(5):
                    train, valid = folds != fold, folds == fold
                    if set(families[train]) & set(families[valid]):
                        raise ValueError('Noise-family leakage in kernel TRAIN CV')
                    model = fit(x[train], reward[train], length, noise)
                    prediction[valid], uncertainty[valid] = predict(model, x[valid])
                score = dict(critic_diagnostics(prediction[:, None], reward),
                    optimism_bias=float((prediction-reward).mean()),
                    within_one_model_std_fraction=float((np.abs(prediction-reward)<=uncertainty).mean()))
                scores.append(dict(features=name, length=length, noise=noise, group_cv=score))
    selected = min(scores, key=lambda row:(row['group_cv']['mae'], row['features'], row['length'], row['noise']))
    model = fit(feature_sets[selected['features']], reward, selected['length'], selected['noise'])
    return selected, model, scores, folds


class KernelCritic(nn.Module):
    def __init__(self, cache, model, feature_kind, pessimism=0.):
        super().__init__()
        self.feature_kind = feature_kind
        self.pessimism = float(pessimism)
        zero = dict(mean=np.zeros(17), std=np.ones(17), coeff=np.zeros(17), intercept=0.)
        self.extractor = PhysicalRidgeCritic(cache, zero)
        for name, value in model.items():
            dtype = torch.long if name == 'keep' else torch.float64
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))

    def features(self, action):
        action = action.double()
        raw = action.reshape(len(action), -1)
        if self.feature_kind == 'coordinates':
            return raw
        physical = self.extractor.features(raw)
        if self.feature_kind == 'physical':
            return physical
        if self.feature_kind != 'physical_coordinates':
            raise ValueError('Unknown kernel feature set')
        return torch.cat([physical, raw], -1)

    def posterior(self, action):
        f = self.features(action)
        z = (f[:, self.keep]-self.mean[self.keep])/self.std[self.keep]
        distance = (z[:, None]-self.train_z[None]).square().mean(-1)
        k = torch.exp(-distance/(2*self.length.square()))
        mean = self.reward_center+self.reward_scale*(k@self.coeff)
        projected = torch.linalg.solve_triangular(self.cholesky, k.T, upper=False)
        variance = (1+self.noise-projected.square().sum(0)).clamp_min(1e-12)
        return mean, self.reward_scale*variance.sqrt()

    def forward(self, action):
        mean, uncertainty = self.posterior(action)
        return (mean-self.pessimism*uncertainty)[:, None].to(action.dtype)
