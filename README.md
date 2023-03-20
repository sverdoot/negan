# GAN with operator norm regularization

## Setup

Create environment:
```bash
conda create -n norm_est_gan python=3.8
conda activate norm_est_gan
```

One of the following:
* For using poetry (more isolated):
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.create false
poetry install
```
* Pure conda:
```bash
pip install git+https://github.com/kwotsin/mimicry.git@1.0.16
pip install einops
```

## Usage
Configs:
* nosn - default GAN
* sn - default SN-GAN
* sn_prop - normalization of block-diagonal matrix
* ne - our spectral norm regularization
* nesn - our spectral normalization

Train:
```bash
python main.py \
    configs/sn32/{sn,ne,nesn,nosn}.yml \
    --suffix exp_name
```

Evaluate:
```bash
python main.py configs/sn32/{sn,ne,nesn,nosn}.yml --suffix exp_name  --eval
```

Tensorboard:
```
tensorboard --logdir_spec=exp1_name:./log/exp1,exp2_name:./log/exp2  --port 8888
```

## Results
