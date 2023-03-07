import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import scipy as sp
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid


class Callback(ABC):
    cnt: int = 0

    @abstractmethod
    def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
        raise NotImplementedError

    def reset(self):
        self.cnt = 0


class CallbackRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: Callback) -> Callback:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> Callback:
        model = cls.registry[name]
        model = model(**kwargs)
        return model
    
    
from modules.svd import get_sing_vals
from modules.spectral_norm import SpectralNorm, EffSpectralNorm


@torch.no_grad()
def get_spectr(model):
    spectr = dict()
    for i, mod in enumerate(model.modules()):
        if EffSpectralNorm in type(mod).__bases__ or SpectralNorm in type(mod).__bases__:
            singular_vals = get_sing_vals(mod.weight, mod.pad_to, mod.stride)
            spectr[i] = singular_vals.detach()
            
    return spectr
            

@CallbackRegistry.register()
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
    def invoke(self, info: Dict[str, Any]):
        step = info.get("step", self.cnt)
        print(step)
        if step % self.invoke_every == 0:
            singular_vals = get_spectr(self.net)
            
            torch.save(singular_vals, Path(self.save_dir, f'spec{step:05d}.pt'))
            
            
        self.cnt += 1
        return 1
    

@CallbackRegistry.register()
class LogGamma(Callback):
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
    def invoke(self, info: Dict[str, Any]):
        step = info.get("step", self.cnt)
        print(step)
        if step % self.invoke_every == 0:
            gammas = dict()
            for i, mod in enumerate(self.net.modules()):
                if EffSpectralNorm in type(mod).__bases__:
                    gammas[i] = mod.gamma.detach().data
            
            torch.save(gammas, Path(self.save_dir, f'gamma{step:05d}.pt'))
            
        self.cnt += 1
        return 1
        
        
# @CallbackRegistry.register()
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
#             128, interpolation=transforms.InterpolationMode.NEAREST
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