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

Ttain:
```bash
python main.py \
    --norm {norm_est/sn} \ # our penalty vs spectral norm
    --suffix exp_name \
    --np_scale 1.0 # penalty scale
```

Evaluate:
```bash
python main.py --norm {norm_est/sn} --eval --suffix exp_name --num_steps 55000
```

Tensorboard:
```
tensorboard --logdir=./log/exp1:exp1_name,./log/exp2:exp2_name  --port 8888
```

## Results