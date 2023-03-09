import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch import nn

from nets.our_gan import sngan_32
from training.callback import LogSigularVals  # , LogGamma
from training.trainer import CustomTrainer


def parse_argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--norm",
        type=str,
        choices=["norm_est", "sn"],
        default="norm_est",
        help="whether to use our norm estimate or spectral norm",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--np_scale", type=float, default=0.01)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    return args


def main(args):
    # Data handling objects
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root="./datasets", name=args.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4
    )

    # Define models and optimizers
    netG = sngan_32.SNGANGenerator32(norm=args.norm, np_scale=args.np_scale).to(device)
    netD = sngan_32.SNGANDiscriminator32(norm=args.norm, np_scale=args.np_scale).to(
        device
    )
    for mod in netG.modules():
        mod.register_forward_pre_hook(lambda m, i: setattr(m, "pad_to", i[0].shape[2:]))
    for mod in netD.modules():
        mod.register_forward_pre_hook(lambda m, i: setattr(m, "pad_to", i[0].shape[2:]))

    log_dir = Path(f"./log/sn32_{args.norm}{args.suffix}")
    if not args.eval:
        log_dir.mkdir(exist_ok=True, parents=True)
        optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
        optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

        # Start training
        sv_dir = Path(log_dir, "sv_d")
        sv_dir.mkdir(exist_ok=True)
        log_sv = LogSigularVals(netD, invoke_every=100, save_dir=sv_dir)
        # log_gamma = LogGamma(netD, invoke_every=100, save_dir=sv_dir)

        trainer = CustomTrainer(
            netD=netD,
            netG=netG,
            optD=optD,
            optG=optG,
            n_dis=5,
            num_steps=args.num_steps,
            lr_decay="linear",
            dataloader=dataloader,
            log_dir=log_dir.as_posix(),
            device=device,
            callbacks=[log_sv],
            print_steps=10,
        )
        trainer.train()
    # else:
    for metric_name in ["fid"]:  # , 'kid', 'inception_score']:
        mmc.metrics.evaluate(
            metric=metric_name,
            log_dir=log_dir.as_posix(),
            netG=netG,
            dataset=args.dataset,
            num_real_samples=50000,
            num_fake_samples=50000,
            # num_samples=50000,
            evaluate_range=(args.num_steps, 0, -5000),
            device=device,
        )


if __name__ == "__main__":
    args = parse_argumets()
    main(args)
