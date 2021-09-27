#!/usr/bin/env python

import torch

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import info, debug, error

from ddpm.unet import UNet
from ddpm.diffusion import Diffuser

from utils import set_seed
from dataset import get_dataset

import argparse
import toml
from pathlib import Path
from types import SimpleNamespace

def main(args):
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    cfg = SimpleNamespace(**toml.load(args.cfg_path))

    train_dataset, test_dataset = get_dataset(cfg.data['name'])

    diffuser = Diffuser(**cfg.diffuser) 
    net = UNet(**cfg.unet)

    # def loss_fn(net, x):
        # pass

    trainer_cfg = TrainerConfig(
        **cfg.trainer,
        nb_workers = args.nb_workers,
        save_outputs = not args.no_save,
        use_cuda = not args.no_cuda,
        use_amp = not args.no_amp,
    )

    trainer = Trainer(
        net = net,
        loss_fn = loss_fn,
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        cfg = trainer_cfg,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/debug.toml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=8)
    parser.add_argument('--detect-anomaly', action='store_true') # menacing aura!
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)
