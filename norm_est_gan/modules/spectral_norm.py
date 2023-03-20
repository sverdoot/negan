from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.nn.utils.spectral_norm import SpectralNorm as TorchSN
from torch.nn.utils.spectral_norm import (
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)

from .svd import get_sing_vals, prepare_matrix


class SpectralNorm:
    def __init__(
        self,
        upd_gamma_every: int = 1000,
        n_samp: int = 10,
        denom: Optional[float] = None,
        power_method: bool = False,
        fft: bool = True,
    ):
        self.upd_gamma_every = upd_gamma_every
        self.n_samp = n_samp
        self.denom = denom
        self.power_method = power_method
        self.fft = fft

    def __call__(self, module: nn.Module, name: str = "weight") -> nn.Module:
        if self.power_method:
            # if dim is None:
            if isinstance(
                module,
                (
                    torch.nn.ConvTranspose1d,
                    torch.nn.ConvTranspose2d,
                    torch.nn.ConvTranspose3d,
                ),
            ):
                dim = 1
            else:
                dim = 0
            PowerMethodSN.apply(
                module,
                name,
                dim=dim,
                eps=1e-12,
                n_power_iterations=1,
                fft=self.fft,
            )
            # save sigma
            # module.register_buffer(f"{name}_sigma", torch.ones(1,
            # requires_grad=False))

            # def save_sigma(module: nn.Module, inputs: Any, outputs: Any):
            #     u = getattr(module, name + "_u")
            #     v = getattr(module, name + "_v")
            #     weight = getattr(module, name + "_orig")
            #     # weight = weight.permute(0,
            #     #                             *[d for d in range(weight.dim())
            #     # if d != 1])
            #     height = weight.size(0)
            #     weight = weight.reshape(height, -1)
            #     sigma = torch.dot(u, torch.mv(weight, v))
            #     setattr(module, f"{name}_sigma", sigma.data)

            # module.register_forward_hook(save_sigma)

        else:
            NormEstimateSN.apply(
                module,
                name,
                self.upd_gamma_every,
                self.n_samp,
                self.denom,
                self.fft,
            )
        return module


class PowerMethodSN(TorchSN):
    def __init__(
        self,
        stride,
        pad_to,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        fft: bool = False,
    ) -> None:
        super().__init__(name, n_power_iterations, dim, eps)
        self.fft = fft
        self.stride = stride
        self.pad_to = pad_to

    def compute_weight(
        self,
        module: nn.Module,
        do_power_iteration: bool,
    ) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    if weight_mat.ndim == 2:
                        v = normalize(
                            torch.mv(weight_mat.t(), u),
                            dim=0,
                            eps=self.eps,
                            out=v,
                        )
                        u = normalize(
                            torch.mv(weight_mat, v),
                            dim=0,
                            eps=self.eps,
                            out=u,
                        )
                    elif weight_mat.ndim == 4:
                        v = normalize(
                            torch.einsum("abcd,abc->abd", weight_mat, u),
                            dim=-1,
                            eps=self.eps,
                            out=v,
                        )
                        u = normalize(
                            torch.einsum("abcd,abd->abc", weight_mat, v),
                            dim=-1,
                            eps=self.eps,
                            out=u,
                        )
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        if weight_mat.ndim == 2:
            sigma = torch.dot(u, torch.mv(weight_mat, v))
        elif weight_mat.ndim == 4:
            sigma = torch.einsum(
                "abc,abc->ab",
                u,
                torch.einsum("abcd,abd->abc", weight_mat, v),
            )
            sigma = ((sigma.real**2 + sigma.imag**2) ** 0.5).max()
        weight = weight / sigma
        setattr(
            module,
            f"{self.name}_sigma",
            torch.tensor([sigma.item()], requires_grad=False, device=weight.device),
        )
        return weight

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        if weight_mat.ndim == 2:
            v = torch.linalg.multi_dot(
                [
                    weight_mat.t().mm(weight_mat).pinverse(),
                    weight_mat.t(),
                    u.unsqueeze(1),
                ],
            ).squeeze(1)
            return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))
        elif weight_mat.ndim == 4:
            wtw = torch.einsum("abcd,abcd->abdd", weight_mat, weight_mat)
            wtu = torch.einsum("abcd,abc->abd", weight_mat, u)
            v = torch.einsum("abcd,abd->abc", wtw, wtu)
            return v.mul_(
                target_sigma
                / torch.einsum(
                    "abc,abc->",
                    u,
                    torch.einsum("abcd,abd->abc", weight_mat, v),
                ),
            )

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim,
                *[d for d in range(weight_mat.dim()) if d != self.dim],
            )

        if self.dim == 0 and self.fft and weight.ndim > 2:
            weight_mat = weight_mat.permute([2, 3, 0, 1])
            weight_mat = prepare_matrix(weight_mat, self.pad_to, self.stride)[-1]
            # weight_mat = weight_mat.permute(2, 3, 0, 1)
        else:
            height = weight_mat.size(0)
            weight_mat = weight_mat.reshape(height, -1)
        return weight_mat

    @staticmethod
    def apply(
        module: nn.Module,
        name: str,
        n_power_iterations: int,
        dim: int,
        eps: float,
        fft: bool,
    ) -> "SpectralNorm":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, NormEstimateSN) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name),
                )

        fn = PowerMethodSN(
            module.stride,
            getattr(module, f"{name}_pad_to", (1, 1)),
            name,
            n_power_iterations,
            dim,
            eps,
            fft,
        )
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(
                f"`SpectralNorm` cannot be applied as parameter `{name}` is None",
            )
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized \
                parameters. "
                "Make sure to run the dummy forward before applying spectral \
                    normalization",
            )

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.shape[-2:]
            # randomly initialize `u` and `v`
            u = normalize(
                weight.new_empty(
                    *weight_mat.shape[:-2],
                    h,
                    dtype=weight_mat.dtype,
                ).normal_(0, 1),
                dim=0,
                eps=fn.eps,
            )
            v = normalize(
                weight.new_empty(
                    *weight_mat.shape[:-2],
                    w,
                    dtype=weight_mat.dtype,
                ).normal_(0, 1),
                dim=0,
                eps=fn.eps,
            )

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1, requires_grad=False))

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


class NormEstimateSN:
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
    fft: bool = True

    def __init__(
        self,
        name: str = "weight",
        upd_gamma_every: int = 1000,
        n_samp: int = 10,
        denom: Optional[float] = None,
        fft: bool = True,
    ):
        self.name = name
        self.upd_gamma_every = upd_gamma_every
        self.n_samp = n_samp
        self.denom = denom
        self.fft = fft

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
    def compute_eff_rank(
        W: torch.Tensor,
        pad_to: Optional[Tuple[int, int]] = None,
        stride: Optional[int] = None,
    ):
        if W.ndim == 2:
            sing_vals = torch.linalg.svdvals(W)
        elif W.ndim == 4 and pad_to is not None and stride is not None:
            sing_vals = torch.sort(
                get_sing_vals(W, pad_to, stride).flatten(),
                descending=True,
            )[0]
        else:
            ValueError("")
        second_norm = sing_vals[0]
        frob_norm = torch.sqrt((sing_vals**2).sum())
        n_dim = len(sing_vals)
        eff_rank = frob_norm**2 / second_norm**2
        return eff_rank, n_dim

    def upd_eff_rank(self, module: nn.Module):
        W = getattr(module, f"{self.name}_orig")
        cnt = getattr(module, f"{self.name}_cnt").item()
        setattr(module, f"{self.name}_cnt", torch.tensor([cnt + 1], device=W.device))
        if cnt % self.upd_gamma_every == 0:
            eff_rank, n_dim = NormEstimateSN.compute_eff_rank(
                W,
                getattr(module, f"{self.name}_pad_to", None),
                getattr(module, "stride", None),
            )
            with torch.no_grad():
                setattr(
                    module,
                    f"{self.name}_gamma",
                    torch.tensor([eff_rank.item() / n_dim], device=W.device),
                )
                setattr(
                    module,
                    f"{self.name}_ndim",
                    torch.tensor([n_dim], device=W.device),
                )

    def estimate_norm(self, module: nn.Module):
        W = getattr(module, f"{self.name}_orig")
        samp = torch.randn(
            [self.n_samp, W.shape[1]] + list(getattr(module, f"{self.name}_pad_to")),
            device=W.device,
            dtype=W.dtype,
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
            self.upd_eff_rank(module)
        if self.denom is not None:
            sigma = self.estimate_norm(module) ** 0.5
            setattr(module, f"{self.name}_sigma", sigma.clone().data * self.denom)
            weight = weight / sigma / self.denom
        return weight

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(
        module: nn.Module,
        name,
        upd_gamma_every,
        n_samp,
        denom,
        fft: bool = True,
    ):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, NormEstimateSN) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name),
                )

        fn = NormEstimateSN(name, upd_gamma_every, n_samp, denom)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(
                f"`SpectralNorm` cannot be applied as parameter `{name}` is None",
            )
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized \
                    parameters. "
                "Make sure to run the dummy forward before applying spectral \
                    normalization",
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
        module.register_buffer(
            f"{fn.name}_ndim",
            torch.ones(1, dtype=torch.long, requires_grad=False),
        )
        module.register_buffer(
            f"{fn.name}_cnt",
            torch.zeros(1, dtype=torch.long, requires_grad=False),
        )

        module.register_forward_pre_hook(
            lambda m, i: setattr(
                m,
                f"{fn.name}_pad_to",
                i[0].shape[2:] if isinstance(m, nn.Conv2d) else [],
            ),
        )
        module.register_forward_pre_hook(fn)
        return fn
