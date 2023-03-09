"""
Implementation of Base GAN models.
"""
import torch
from torch_mimicry.nets.gan import BaseDiscriminator, BaseGenerator

from modules.spectral_norm import EffSpectralNorm


# from torch_mimicry.nets.basemodel import basemodel
# from torch_mimicry.modules import losses


def compute_norm_penalty(model, scale=1.0):
    loss = 0
    for p in model.modules():
        if EffSpectralNorm in type(p).__bases__:
            loss += p.estimate_norm().squeeze()
    return loss * scale


class BaseGenerator(BaseGenerator):
    r"""
    Base class for a generic unconditional generator model.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """

    def __post_init__(self, np_scale=1.0):
        self.np_scale = np_scale

    def compute_gan_loss(self, output):
        r"""
        Computes GAN loss for generator.

        Args:
            output (Tensor): A batch of output logits from the discriminator of shape (N, 1).

        Returns:
            Tensor: A batch of GAN losses for the generator.
        """
        # Compute loss and backprop
        errG = super().compute_gan_loss(output)
        errG += compute_norm_penalty(self, scale=self.np_scale)

        return errG

    # def train_step(self, real_batch, netG, optD, log_data, device=None, global_step=None, **kwargs):
    #     log_data = super().train_step(real_batch, netG, optD, log_data, device, global_step, **kwargs)
    #     for i, p in enumerate(self.modules()):
    #         if EffSpectralNorm in type(p).__bases__:
    #             log_data.add_metric(f'g_gamma_{i}', p.gamma.item())
    #     return log_data


class BaseDiscriminator(BaseDiscriminator):
    r"""
    Base class for a generic unconditional discriminator model.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """

    def __post_init__(self, np_scale=1.0):
        self.np_scale = np_scale

    def compute_gan_loss(self, output_real, output_fake):
        r"""
        Computes GAN loss for discriminator.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.

        Returns:
            errD (Tensor): A batch of GAN losses for the discriminator.
        """
        errD = super().compute_gan_loss(output_real, output_fake)
        errD += compute_norm_penalty(self, scale=self.np_scale)

        return errD

    # def train_step(self, real_batch, netG, optD, log_data, device=None, global_step=None, **kwargs):
    #     log_data = super().train_step(real_batch, netG, optD, log_data, device, global_step, **kwargs)
    #     for i, p in enumerate(self.modules()):
    #         if EffSpectralNorm in type(p).__bases__:
    #             log_data.add_metric(f'd_gamma_{i}', p.gamma.item())
    #     return log_data
