from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from norm_est_gan.modules.spectral_norm import NormEstimation
from norm_est_gan.modules.svd import get_sing_vals
from norm_est_gan.registry import Registry


class Callback(ABC):
    cnt: int = 0

    @abstractmethod
    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        raise NotImplementedError

    def reset(self):
        self.cnt = 0
        

@torch.no_grad()
def get_spectr(model):
    spectr = dict()
    sigmas = dict()
    for i, mod in enumerate(model.modules()):
        # if hasattr(mod, "weight_orig") and hasattr(
        #     mod, "stride"
        # ):
        if hasattr(mod, "weight_orig"):
            if mod.weight_orig.ndim == 2:
                sing_vals = torch.linalg.svdvals(mod.weight_orig).detach().cpu()
            elif mod.weight_orig.ndim == 4:
                sing_vals = get_sing_vals(mod.weight_orig, getattr(mod, "weight_pad_to", None), getattr(mod, "stride", None))
            sing_vals = torch.sort(sing_vals.flatten(), descending=True)[0]
            spectr[i] = sing_vals.detach()
            sigmas[i] = mod.weight_sigma.detach()

    return spectr, sigmas


@Registry.register()
class LogSigularVals(Callback):
    def __init__(
        self,
        net,
        *,
        invoke_every: int = 1,
        init_params: Optional[Dict] = None,
        keys: Optional[List[str]] = None,
        save_dir=None,
    ):
        self.init_params = init_params if init_params else {}

        self.invoke_every = invoke_every
        self.keys = keys
        self.net = net
        self.save_dir = save_dir

    @torch.no_grad()
    def invoke(self, info: Dict[str, Any], log_data):
        step = info.get("step", self.cnt)
        if step % self.invoke_every == 0:
            singular_vals, sigmas = get_spectr(self.net)

            gammas = dict()
            for n, v in singular_vals.items():
                second_norm = v[0]
                frob_norm = np.sqrt((v**2).sum())
                n_dim = len(v)
                gamma = frob_norm**2 / second_norm**2 / n_dim
                log_data.add_metric(f"sec_norm_{n}", second_norm.item())
                log_data.add_metric(f"frob_norm_{n}", frob_norm.item())
                log_data.add_metric(f"gamma_{n}", gamma.item())
                log_data.add_metric(f"sigma_{n}", sigmas[n].item())

                gammas[n] = gamma

            torch.save(singular_vals, Path(self.save_dir, f"spec{step:05d}.pt"))
            torch.save(gammas, Path(self.save_dir, f"gamma{step:05d}.pt"))

        self.cnt += 1
        return 1


# @Registry.register()
# class WandbCallback(Callback):
#     def __init__(
#         self,
#         *,
#         invoke_every: int = 1,
#         init_params: Optional[Dict] = None,
#         keys: Optional[List[str]] = None,
#     ):
#         self.init_params = init_params if init_params else {}
#         import wandb

#         self.wandb = wandb
#         wandb.init(**self.init_params)

#         self.invoke_every = invoke_every
#         self.keys = keys

#         self.img_transform = transforms.Resize(
#             128, interpolation=transforms.InterpolationMode.LANCZOS
#         )

#     def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
#         step = info.get("step", self.cnt)
#         if step % self.invoke_every == 0:
#             wandb = self.wandb
#             if not self.keys:
#                 self.keys = info.keys()
#             log = dict()
#             for key in self.keys:
#                 if key not in info:
#                     continue
#                 if isinstance(info[key], np.ndarray):
#                     log[key] = wandb.Image(
#                         make_grid(
#                             self.img_transform(
#                                 torch.clip(torch.from_numpy(info[key][:25]), 0, 1)
#                             ),
#                             nrow=5,
#                         ),
#                         caption=key,
#                     )
#                 else:
#                     log[key] = info[key]
#             log["step"] = step
#             wandb.log(log)
#         self.cnt += 1
#         return 1

#     def reset(self):
#         super().reset()
#         self.wandb.init(**self.init_params)
