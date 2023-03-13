"""
Implementation of spectral normalization for GANs.
"""
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .svd import get_sing_vals


class NormEstimation(object):
    r"""
    Spectral Normalization for GANs (Miyato 2018).

    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.

    Details: See Algorithm 1 of Appendix A (Miyato 2018).

    Attributes:
        n_dim (int): Number of dimensions.
        num_iters (int): Number of iterations for power iter.
        eps (float): Epsilon for zero division tolerance when normalizing.
    """

    def __init__(
        self, n_dim=0, upd_gamma_every=1000, n_samp=10, upd_est_every=1, denom=0.5
    ):
        self.n_dim = n_dim
        self.upd_gamma_every = upd_gamma_every
        self.upd_est_every = upd_est_every
        self.gamma_cnt = 0
        self.est_cnt = 0
        self.pad_to = (0, 0)
        self.n_samp = n_samp
        self.denom = denom

        # Register a singular vector for each gamma
        self.register_buffer("sn_gamma", torch.ones(1, requires_grad=False))
        # self.register_buffer("sn_estimate", torch.ones(1, requires_grad=False))
        self.register_forward_pre_hook(
            lambda m, i: setattr(m, "pad_to", i[0].shape[2:])
        )

    @property
    def gamma(self):
        return getattr(self, "sn_gamma")

    @torch.no_grad()
    @abstractmethod
    def _compute_gamma(W, pad_to, stride):
        sing_vals = torch.sort(
            get_sing_vals(W, pad_to, stride).flatten(), descending=True
        )[0]
        second_norm = sing_vals[0]
        frob_norm = torch.sqrt((sing_vals**2).sum())
        n_dim = len(sing_vals)
        gamma = frob_norm**2 / second_norm**2 / n_dim
        return gamma, n_dim

    @abstractmethod
    def _estimate(module: nn.Module, pad_to, stride: int, n_samp: int = 100):
        W = module.weight
        gamma, n_dim = NormEstimation._compute_gamma(W, pad_to, stride)
        device = W.device
        samp = torch.randn([n_samp, W.shape[1]] + list(pad_to)).to(device)
        out = module.forward(samp)
        normsq = ((out**2).sum(list(range(len(out.shape)))[1:])).mean(0)
        estimate = normsq / (gamma * n_dim)
        return estimate

    def estimate_norm(self):
        W = self.weight

        if self.training:
            self.gamma_cnt += 1
            if self.gamma_cnt % self.upd_gamma_every == 1:
                gamma, n_dim = NormEstimation._compute_gamma(
                    W, self.pad_to, self.stride
                )
                with torch.no_grad():
                    self.gamma[:] = gamma
                    self.n_dim = n_dim

        # self.est_cnt += 1
        # if self.est_cnt % self.upd_est_every == 1:
        samp = torch.randn(
            [self.n_samp, W.shape[1]] + list(self.pad_to), device=W.device
        )
        out = self._forward(samp)
        normsq = ((out**2).sum(list(range(len(out.shape)))[1:])).mean(0)
        estimate = normsq / (self.gamma * self.n_dim)
        # self.estimate = estimate #.detach()

        return estimate

    def sn_weights(self):
        return self.weight / self.estimate_norm() ** 0.5 / self.denom


class NEConv2d(nn.Conv2d, NormEstimation):
    r"""
    Spectrally normalized layer for Conv2d.

    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)

        NormEstimation.__init__(self, n_dim=out_channels)

    def _forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.sn_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
