"""
Implementation of residual blocks for discriminator and generator.
We follow the official SNGAN Chainer implementation as closely as possible:
https://github.com/pfnet-research/sngan_projection
"""
import math

import torch.nn as nn
import torch.nn.functional as F
from torch_mimicry.modules import ConditionalBatchNorm2d  # , SNConv2d
from torch_mimicry.modules import SNConv2d
from torch_mimicry.modules.resblocks import DBlock, DBlockOptimized, GBlock

from .spectral_norm import NEConv2d


class GBlock(GBlock):
    r"""
    Residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        upsample=False,
        num_classes=0,
        spectral_norm=False,
        norm="sn",
        convl=nn.Conv2d,
    ):
        super().__init__(
            in_channels,
            out_channels,
            hidden_channels,
            upsample,
            num_classes,
            spectral_norm,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else out_channels
        )
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        # Build the layers
        # Note: Can't use something like self.conv = SNConv2d to save code length
        # this results in somehow spectral norm working worse consistently.
        if norm == "norm_est":
            self.c1 = NEConv2d(self.in_channels, self.hidden_channels, 3, 1, padding=1)
            self.c2 = NEConv2d(self.hidden_channels, self.out_channels, 3, 1, padding=1)
        elif self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, padding=1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, padding=1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, padding=1)
            self.c2 = nn.Conv2d(
                self.hidden_channels, self.out_channels, 3, 1, padding=1
            )

        if self.num_classes == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = ConditionalBatchNorm2d(self.in_channels, self.num_classes)
            self.b2 = ConditionalBatchNorm2d(self.hidden_channels, self.num_classes)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, padding=0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)


class DBlock(DBlock):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        downsample=False,
        spectral_norm=True,
        norm="sn",
    ):
        super().__init__(
            in_channels, out_channels, hidden_channels, downsample, spectral_norm
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels
        )
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        # Build the layers
        if norm == "norm_est":
            self.c1 = NEConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = NEConv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        elif self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if norm == "norm_est":
                self.c_sc = NEConv2d(in_channels, out_channels, 1, 1, 0)
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)


class DBlockOptimized(DBlockOptimized):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        spectral_norm=True,
        norm="sn",
    ):
        super().__init__(in_channels, out_channels, spectral_norm)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build the layers
        if norm == "norm_est":
            self.c1 = NEConv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.c2 = NEConv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.c_sc = NEConv2d(self.in_channels, self.out_channels, 1, 1, 0)
        elif self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.c_sc = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.c2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)
