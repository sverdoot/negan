import torch
import torch.optim as optim
import torch_mimicry as mmc
from nets.our_gan import sngan_32
import argparse


def parse_argumets():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=str, choices=['norm_est', 'sn'], default='norm_est', help='whether to use our norm estimate or spectral norm')
    args = parser.parse_args()
    return args


def main(args):
    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=True, num_workers=0) #4)

    # Define models and optimizers
    netG = sngan_32.SNGANGenerator32(norm='norm_est').to(device) #norm='sn'
    netD = sngan_32.SNGANDiscriminator32(norm='norm_est').to(device) #norm='sn'
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=5,
        num_steps=100000,
        lr_decay='linear',
        dataloader=dataloader,
        log_dir=f'./log/example_{args.norm}',
        device=device)
    trainer.train()
    

if __name__ == '__main__':
    args = parse_argumets()
    main(args)
