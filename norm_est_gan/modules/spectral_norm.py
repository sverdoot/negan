"""
Implementation of spectral normalization for GANs.
"""
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNormStateDictHook

from .svd import get_sing_vals


class SpectralNorm:
    def __init__(
        self,
        upd_gamma_every: int=1000,
        n_samp: int=10,
        denom: Optional[float]=None,
        default: bool=False,
    ):
        self.upd_gamma_every = upd_gamma_every
        self.n_samp = n_samp
        self.denom = denom
        self.default = default

    def __call__(self, module: nn.Module, name: str = "weight") -> nn.Module:
        if self.default == True:
            module = nn.utils.spectral_norm(module, name=name)
            # save sigma
            module.register_buffer(f'{name}_sigma', torch.ones(1, requires_grad=False))
            
            def save_sigma(module: nn.Module, inputs: Any, outputs: Any):
                u = getattr(module, name + '_u')
                v = getattr(module, name + '_v')
                weight = getattr(module, name + '_orig')
                # weight = weight.permute(0,
                #                             *[d for d in range(weight.dim()) if d != 1])
                height = weight.size(0)
                weight = weight.reshape(height, -1)
                sigma = torch.dot(u, torch.mv(weight, v))
                setattr(module, f'{name}_sigma', sigma.data)
                
            module.register_forward_hook(save_sigma)
            
        else:
            NormEstimation.apply(
                module,
                name,
                self.upd_gamma_every,
                self.n_samp,
                self.denom,
            )
        return module


class NormEstimation:
    r"""
    Attributes:
        # n_dim (int): Number of dimensions.
        # num_iters (int): Number of iterations for power iter.
        # eps (float): Epsilon for zero division tolerance when normalizing.
    """
    weight: str = "weight"
    upd_gamma_every: int = 1000
    n_samp: int = 10
    upd_est_every: int = 1
    denom: Optional[float] = None

    def __init__(
        self,
        name: str = "weight",
        upd_gamma_every: int = 1000,
        n_samp: int = 10,
        denom: Optional[float] = None,
    ):
        self.name = name
        self.upd_gamma_every = upd_gamma_every
        self.n_samp = n_samp
        self.denom = denom
        
    def forward(self, module, x) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            return F.conv2d(
                input=x,
                weight=getattr(module, f"{self.name}_orig"),
                bias=None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        elif isinstance(module, nn.Linear):
            return F.linear(x, getattr(module, f"{self.name}_orig"), bias=None)
        elif isinstance(module, nn.Embedding):
            return F.embedding(x, getattr(module, f"{self.name}_orig"))
        else:
            RuntimeError("")

    @staticmethod
    @torch.no_grad()
    def compute_eff_rank(W: torch.Tensor, pad_to: Optional[Tuple[int, int]]=None, stride: Optional[int]=None):
        if W.ndim == 2:
            sing_vals = torch.linalg.svdvals(W)
        elif W.ndim == 4:
            sing_vals = torch.sort(
                get_sing_vals(W, pad_to, stride).flatten(), descending=True
            )[0]
        else:
            ValueError("")
        second_norm = sing_vals[0]
        frob_norm = torch.sqrt((sing_vals**2).sum())
        n_dim = len(sing_vals)
        eff_rank = frob_norm ** 2 / second_norm ** 2
        return eff_rank, n_dim

    def upd_eff_rank(self, module: nn.Module):
        W = getattr(module, f"{self.name}_orig")
        cnt = getattr(module, f"{self.name}_cnt").item()
        setattr(module, f"{self.name}_cnt", torch.tensor([cnt + 1], device=W.device))
        if cnt % self.upd_gamma_every == 0:
            eff_rank, n_dim = NormEstimation.compute_eff_rank(
                W, getattr(module, f"{self.name}_pad_to", None), getattr(module, "stride", None)
            )
            with torch.no_grad():
                setattr(module, f"{self.name}_gamma", torch.tensor([eff_rank.item() / n_dim], device=W.device))
                setattr(module, f"{self.name}_ndim", torch.tensor([n_dim], device=W.device))
                
    def estimate_norm(self, module: nn.Module):
        W = getattr(module, f"{self.name}_orig")
        samp = torch.randn(
            [self.n_samp, W.shape[1]] + list(getattr(module, f"{self.name}_pad_to")),
            device=W.device,
        )
        out = self.forward(module, samp)
        normsq = ((out ** 2).sum(list(range(len(out.shape)))[1:])).mean(0)
        estimate = normsq / (
            getattr(module, f"{self.name}_gamma") * getattr(module, f"{self.name}_ndim")
        )
        return estimate

    def compute_weight(self, module):
        weight = getattr(module, f"{self.name}_orig")
        if module.training:
            self.upd_eff_rank(module)
        if self.denom is not None:
            sigma = self.estimate_norm(module) ** .5
            setattr(module, f'{self.name}_sigma', sigma.clone().data * self.denom)
            weight = weight / sigma / self.denom
        return weight

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module: nn.Module, name, upd_gamma_every, n_samp, denom):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, NormEstimation) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = NormEstimation(name, upd_gamma_every, n_samp, denom)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(
                f"`SpectralNorm` cannot be applied as parameter `{name}` is None"
            )
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying spectral normalization"
            )

        delattr(module, fn.name)
        module.register_parameter(f"{fn.name}_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        # setattr(module, f"{fn.name}_cnt", 0)
        # Register a singular vector for each gamma
        module.register_buffer(f"{fn.name}_gamma", torch.ones(1, requires_grad=False))
        module.register_buffer(f"{fn.name}_sigma", torch.ones(1, requires_grad=False))
        module.register_buffer(f"{fn.name}_ndim", torch.ones(1, dtype=torch.long, requires_grad=False))
        module.register_buffer(f"{fn.name}_cnt", torch.zeros(1, dtype=torch.long, requires_grad=False))

        module.register_forward_pre_hook(
            lambda m, i: setattr(m, f"{fn.name}_pad_to", i[0].shape[2:] if isinstance(m, nn.Conv2d) else [])
        )
        module.register_forward_pre_hook(fn)
        # module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        # module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

