"""
https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/sngan/sngan_32.py
"""
import torch
import torch.nn as nn
from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.nets.sngan import SNGANDiscriminator32, SNGANGenerator32

from norm_est_gan.nets.our_gan import base
from norm_est_gan.modules.resblocks import GBlock, DBlock, DBlockOptimized

# def SNConv2d(*args, default=True, **kwargs):
#     r"""
#     Wrapper for applying spectral norm on conv2d layer.
#     """
#     if default:
#         return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))

#     else:
#         return spectral_norm.SNConv2d(*args, **kwargs)
    
class SNGANGenerator32(SNGANGenerator32, base.BaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """

    def __init__(self, nz=128, ngf=256, bottom_width=4, **kwargs):
        norm = kwargs.pop("norm", "norm_est")
        super(SNGANGenerator32, self).__init__(
            nz=nz, ngf=ngf, bottom_width=bottom_width
        )  # , **kwargs)
        base.BaseGenerator.__post_init__(self, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)#, norm=norm)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)#, norm=norm)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)#, norm=norm)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(self.ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def train_step(
        self, real_batch, netD, optG, log_data, device=None, global_step=None, **kwargs
    ):
        return base.BaseGenerator.train_step(
            self, real_batch, netD, optG, log_data, device, global_step, **kwargs
        )


class SNGANDiscriminator32(SNGANDiscriminator32, base.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """

    def __init__(self, ndf=128, **kwargs):
        norm = kwargs.pop("norm", "norm_est")
        super(SNGANDiscriminator32, self).__init__(ndf=ndf)  # , **kwargs)
        base.BaseDiscriminator.__post_init__(self, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf, norm=norm)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True, norm=norm)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False, norm=norm)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False, norm=norm)
        self.l5 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def train_step(
        self, real_batch, netG, optD, log_data, device=None, global_step=None, **kwargs
    ):
        return base.BaseDiscriminator.train_step(
            self, real_batch, netG, optD, log_data, device, global_step, **kwargs
        )
