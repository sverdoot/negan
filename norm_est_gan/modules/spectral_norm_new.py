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
        upd_gamma_every=1000,
        n_samp=10,
        upd_est_every=1,
        denom=None,
        default=False,
    ):
        self.upd_gamma_every = upd_gamma_every
        self.upd_est_every = upd_est_every
        self.n_samp = n_samp
        self.denom = denom
        self.default = default

    def apply(self, module: nn.Module, name: str = "weight") -> nn.Module:
        if self.default == True:
            return nn.utils.spectral_norm(module, name=name)
        else:
            NormEstimation.apply(
                module,
                name,
                self.upd_gamma_every,
                self.n_samp,
                self.upd_est_every,
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
        upd_est_every: int = 1,
        denom: Optional[float] = None,
    ):
        self.name = name
        self.upd_gamma_every = upd_gamma_every
        self.upd_est_every = upd_est_every
        self.n_samp = n_samp
        self.denom = denom

    @torch.no_grad()
    @staticmethod
    def _compute_gamma(W: torch.Tensor, pad_to: Tuple[int, int], stride: int):
        sing_vals = torch.sort(
            get_sing_vals(W, pad_to, stride).flatten(), descending=True
        )[0]
        second_norm = sing_vals[0]
        frob_norm = torch.sqrt((sing_vals**2).sum())
        n_dim = len(sing_vals)
        gamma = frob_norm**2 / second_norm**2 / n_dim
        return gamma, n_dim

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

    def upd_gamma(self, module: nn.Module):
        W = getattr(module, f"{self.name}_orig")
        cnt = getattr(module, f"{self.name}_cnt")
        setattr(module, f"{self.name}_cnt", cnt + 1)
        if cnt % self.upd_gamma_every == 1:
            gamma, n_dim = NormEstimation._compute_gamma(
                W, getattr(module, f"{self.name}_pad_to"), module.stride
            )
            with torch.no_grad():
                setattr(module, f"{self.name}_gamma", gamma)
                setattr(module, f"{self.name}_ndim", n_dim)

    def estimate_norm(self, module: nn.Module):
        W = getattr(module, f"{self.name}_orig")
        samp = torch.randn(
            [self.n_samp, W.shape[1]] + list(getattr(module, f"{self.name}_pad_to")),
            device=W.device,
        )
        out = self.forward(module, samp)
        normsq = ((out**2).sum(list(range(len(out.shape)))[1:])).mean(0)
        estimate = normsq / (
            getattr(module, f"{self.name}_gamma") * getattr(module, f"{self.name}_ndim")
        )
        return estimate

    def compute_weight(self, module):
        weight = getattr(module, f"{self.name}_orig")
        if module.training:
            self.upd_gamma(module)
        if self.denom is not None:
            weight = weight / self.estimate_norm(module) ** 0.5 / self.denom
        return weight

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module: nn.Module, name, upd_gamma_every, n_samp, upd_est_every, denom):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, NormEstimation) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = NormEstimation(
            module, name, 0, upd_gamma_every, n_samp, upd_est_every, denom
        )
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
        setattr(module, f"{fn.name}_cnt", 0)
        # Register a singular vector for each gamma
        module.register_buffer(f"{fn.name}_gamma", torch.ones(1, requires_grad=False))

        module.register_forward_pre_hook(
            lambda m, i: setattr(m, f"{fn.name}_pad_to", i[0].shape[2:])
        )
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        # module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


class SpectralNormLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        fn = self.fn
        version = local_metadata.get("spectral_norm", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if (
                version is None
                and all(weight_key + s in state_dict for s in ("_orig", "_gamma"))
                and weight_key not in state_dict
            ):
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ("_orig", "", "_gamma"):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            # with torch.no_grad():
            #     weight_orig = state_dict[weight_key + '_orig']
            #     weight = state_dict.pop(weight_key)
            #     sigma = (weight_orig / weight).mean()
            #     weight_mat = fn.reshape_weight_to_matrix(weight_orig)
            #     u = state_dict[weight_key + '_u']
            #     v = fn._solve_v_and_rescale(weight_mat, u, sigma)
            #     state_dict[weight_key + '_v'] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
# class SpectralNormStateDictHook:
#     # See docstring of SpectralNorm._version on the changes to spectral_norm.
#     def __init__(self, fn) -> None:
#         self.fn = fn

#     def __call__(self, module, state_dict, prefix, local_metadata) -> None:
#         if "spectral_norm" not in local_metadata:
#             local_metadata["spectral_norm"] = {}
#         key = self.fn.name + ".version"
#         if key in local_metadata["spectral_norm"]:
#             raise RuntimeError(
#                 "Unexpected key in metadata['spectral_norm']: {}".format(key)
#             )
#         local_metadata["spectral_norm"][key] = self.fn._version
