import argparse
import shutil
from pathlib import Path

import torch
import torch.optim as optim
import torch_mimicry as mmc
from munch import Munch
from ruamel import yaml

from norm_est_gan.metrics.compute_metrics import evaluate
from norm_est_gan.modules.spectral_norm import SpectralNorm
from norm_est_gan.nets import sngan_32
from norm_est_gan.training.callback import LogSigularVals
from norm_est_gan.training.trainer import CustomTrainer


def parse_argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="configs/sn32/nesn.yml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--np_scale", type=float, default=0.01)
    # parser.add_argument("--denom", type=float, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
    else:
        prefix = "_".join(
            list(map(lambda x: Path(x).stem, Path(args.config).parts))[1:],
        )
        suffix = "_" + args.suffix if args.suffix is not None else ""
        log_dir = Path(
            f"./log/{prefix}{suffix}",
        )  # _{config.np_scale:.03f}_{config.denom:.03f}_

    if args.eval:
        config = yaml.safe_load(Path(log_dir, Path(args.config).name).open("r"))
    else:
        config = yaml.safe_load(Path(args.config).open("r"))
    config = Munch(config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root="./datasets", name=args.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )

    spectral_norm = SpectralNorm(
        upd_gamma_every=config.upd_gamma_every,
        denom=config.denom,
        power_method=config.power_method,
        fft=config.fft,
    )
    netG = sngan_32.SNGANGenerator32(
        spectral_norm=spectral_norm,
        np_scale=config.np_scale,
    ).to(device)
    netD = sngan_32.SNGANDiscriminator32(
        spectral_norm=spectral_norm,
        np_scale=config.np_scale,
    ).to(device)

    if not args.eval:
        log_dir.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(Path(args.config), Path(log_dir, Path(args.config).name))

        optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
        optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

        # Start training
        sv_dir = Path(log_dir, "sv_d")
        sv_dir.mkdir(exist_ok=True)
        log_sv = LogSigularVals(netD, invoke_every=100, save_dir=sv_dir)

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
        evaluate(
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
