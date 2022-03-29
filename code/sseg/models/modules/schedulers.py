# -*- coding: utf-8 -*
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

__all__ = ['build_scheduler']


def build_scheduler(cfg, optimizer):
    if cfg.train.lr_scheduler.type == 'Cosine':
        return CosineAnnealingLR(optimizer, T_max=cfg.train.total_iter, eta_min=cfg.train.lr * 0.001)
    elif cfg.train.lr_scheduler.type == 'Poly':
        lr_lambda = lambda iter: (1 - (iter / cfg.train.total_iter)) ** cfg.train.lr_scheduler.poly.power
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError('{} is not a valid scheduler'.format(cfg.train.lr_scheduler.type))
