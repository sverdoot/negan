"""
https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/sngan/sngan_32.py
"""
import torch.nn as nn
from torch_mimicry.nets.sngan import SNGANDiscriminator32 as SNGAND32
from torch_mimicry.nets.sngan import SNGANGenerator32 as SNGANG32

from norm_est_gan.modules.resblocks import DBlock, DBlockOptimized, GBlock
from norm_est_gan.nets import base


# from torch_mimicry.modules.layers import SNLinear # isort: ignore


class SNGANGenerator32(SNGANG32, base.BaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """

    def __init__(self, nz=128, ngf=256, bottom_width=4, **kwargs):
        kwargs.pop("spectral_norm", None)
        super().__init__(
            nz=nz,
            ngf=ngf,
            bottom_width=bottom_width,
        )  # , **kwargs)
        base.BaseGenerator.__post_init__(self, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(self.ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def train_step(
        self,
        real_batch,
        netD,
        optG,
        log_data,
        device=None,
        global_step=None,
        **kwargs,
    ):
        return base.BaseGenerator.train_step(
            self,
            real_batch,
            netD,
            optG,
            log_data,
            device,
            global_step,
            **kwargs,
        )


class SNGANDiscriminator32(SNGAND32, base.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    im_size = 32

    def __init__(self, ndf=128, **kwargs):
        spectral_norm = kwargs.pop("spectral_norm", None)
        super().__init__(ndf=ndf)  # , **kwargs)
        base.BaseDiscriminator.__post_init__(self, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(
            3,
            self.ndf,
            sn=spectral_norm,
            im_size=self.im_size,
        )
        self.block2 = DBlock(
            self.ndf,
            self.ndf,
            downsample=True,
            sn=spectral_norm,
            im_size=self.im_size // (2**1),
        )
        self.block3 = DBlock(
            self.ndf,
            self.ndf,
            downsample=False,
            sn=spectral_norm,
            im_size=self.im_size // (2**2),
        )
        self.block4 = DBlock(
            self.ndf,
            self.ndf,
            downsample=False,
            sn=spectral_norm,
            im_size=self.im_size // (2**2),
        )

        # self.l5 = SNLinear(self.ndf, 1)
        self.l5 = nn.Linear(self.ndf, 1)
        if spectral_norm is not None:
            # pass
            self.l5.weight_pad_to = (1, 1)
            self.l5.stride = None
            self.l5 = spectral_norm(self.l5)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def train_step(
        self,
        real_batch,
        netG,
        optD,
        log_data,
        device=None,
        global_step=None,
        **kwargs,
    ):
        return base.BaseDiscriminator.train_step(
            self,
            real_batch,
            netG,
            optD,
            log_data,
            device,
            global_step,
            **kwargs,
        )
