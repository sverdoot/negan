"""
Implementation of spectral normalization for GANs.
"""
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .svd import get_ready_for_svd, get_sing_vals, get_sing_vals_simple


class SpectralNorm(object):
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

    def __init__(self, n_dim, num_iters=1, eps=1e-12):
        self.num_iters = num_iters
        self.eps = eps

        # Register a singular vector for each sigma
        self.register_buffer("sn_u", torch.randn(1, n_dim))
        self.register_buffer("sn_sigma", torch.ones(1))
        self.register_forward_pre_hook(
            lambda m, i: setattr(m, "pad_to", i[0].shape[2:])
        )

    @property
    def u(self):
        return getattr(self, "sn_u")

    @property
    def sigma(self):
        return getattr(self, "sn_sigma")

    def _power_iteration(self, W, u, num_iters, eps=1e-12):
        with torch.no_grad():
            for _ in range(num_iters):
                v = F.normalize(torch.matmul(u, W), eps=eps)
                u = F.normalize(torch.matmul(v, W.t()), eps=eps)

        # Note: must have gradients, otherwise weights do not get updated!
        sigma = torch.mm(u, torch.mm(W, v.t()))

        return sigma, u, v

    def sn_weights(self):
        r"""
        Spectrally normalize current weights of the layer.
        """
        W = self.weight.view(self.weight.shape[0], -1)

        # Power iteration
        sigma, u, v = self._power_iteration(
            W=W, u=self.u, num_iters=self.num_iters, eps=self.eps
        )

        # Update only during training
        if self.training:
            with torch.no_grad():
                self.sigma[:] = sigma
                self.u[:] = u

        return self.weight / sigma


class EffSpectralNorm(object):
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

    def __init__(self, n_dim=0, num_iters=1, eps=1e-12, update_every=100, n_samp=10):
        self.num_iters = num_iters
        self.eps = eps
        self.n_dim = n_dim
        self.update_every = update_every
        self.cnt = 0
        self.pad_to = (0, 0)
        self.n_samp = n_samp

        # Register a singular vector for each gamma
        self.register_buffer("sn_gamma", torch.ones(1, requires_grad=False))
        self.register_forward_pre_hook(
            lambda m, i: setattr(m, "pad_to", i[0].shape[2:])
        )

    @property
    def gamma(self):
        return getattr(self, "sn_gamma")

    @torch.no_grad()
    @abstractmethod
    def _compute_gamma(W, pad_to, stride):
        # sing_vals = np.sort(get_sing_vals(W, x.shape[2:], stride).flatten())[::-1]
        sing_vals = torch.sort(
            get_sing_vals(W, pad_to, stride).flatten(), descending=True
        )[0]
        second_norm = sing_vals[0]
        frob_norm = torch.sqrt((sing_vals**2).sum())
        # print(frob_norm, second_norm)
        n_dim = len(sing_vals)
        # self.n_dim = n_dim
        gamma = frob_norm**2 / second_norm**2 / n_dim
        return gamma, n_dim

    @abstractmethod
    def _estimate(module: nn.Module, pad_to, stride: int, n_samp: int = 100):
        W = module.weight
        gamma, n_dim = EffSpectralNorm._compute_gamma(W, pad_to, stride)
        device = W.device
        samp = torch.randn([n_samp, W.shape[1]] + list(pad_to)).to(device)
        out = module.forward(samp)
        normsq = ((out**2).sum(list(range(len(out.shape)))[1:])).mean(0)
        estimate = normsq / (gamma * n_dim)
        return estimate

    def estimate_norm(self):
        W = self.weight

        if self.training:
            self.cnt += 1
            if self.cnt % self.update_every == 1:
                gamma, n_dim = EffSpectralNorm._compute_gamma(
                    W, self.pad_to, self.stride
                )
                with torch.no_grad():
                    self.gamma[:] = gamma
                    self.n_dim = n_dim

        device = next(self.parameters()).device
        samp = torch.randn([self.n_samp, W.shape[1]] + list(self.pad_to)).to(device)
        out = self.forward(samp)
        normsq = ((out**2).sum(list(range(len(out.shape)))[1:])).mean(0)
        estimate = normsq / (self.gamma * self.n_dim)
        # if self.cnt % self.update_every == 1:
        #     print(normsq **.5, estimate **.5)
        return estimate

    def sn_weights(self):
        return self.weight


class SNConv2d(nn.Conv2d, EffSpectralNorm):
    r"""
    Spectrally normalized layer for Conv2d.

    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)

        EffSpectralNorm.__init__(
            self, n_dim=out_channels, num_iters=kwargs.get("num_iters", 1)
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


class SNLinear(nn.Linear, SpectralNorm):
    r"""
    Spectrally normalized layer for Linear.

    Attributes:
        in_features (int): Input feature dimensions.
        out_features (int): Output feature dimensions.
    """

    def __init__(self, in_features, out_features, *args, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, *args, **kwargs)

        SpectralNorm.__init__(
            self, n_dim=out_features, num_iters=kwargs.get("num_iters", 1)
        )

    def forward(self, x):
        return F.linear(input=x, weight=self.sn_weights(), bias=self.bias)


class SNEmbedding(nn.Embedding, SpectralNorm):
    r"""
    Spectrally normalized layer for Embedding.

    Attributes:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimensions of each embedding vector
    """

    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, *args, **kwargs)

        SpectralNorm.__init__(self, n_dim=num_embeddings)

    def forward(self, x):
        return F.embedding(input=x, weight=self.sn_weights())
