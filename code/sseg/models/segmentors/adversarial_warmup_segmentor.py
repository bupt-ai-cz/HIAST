# -*- coding: utf-8 -*
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from sseg.models.modules.seg_models import build_seg_model
from sseg.models.modules.discriminator import build_discriminator
from utils.registry.registries import LOSS, MODEL


@MODEL.register('AdversarialWarmupSegmentor')
class AdversarialWarmupSegmentor(nn.Module):

    def __init__(self, cfg):
        super(AdversarialWarmupSegmentor, self).__init__()
        self.cfg = cfg
        self.seg_model = build_seg_model(cfg)

        assert cfg.model.discriminator.is_enabled
        self.D = build_discriminator(cfg.dataset.num_classes)

        self.seg_loss_fun = LOSS[cfg.model.predictor.seg_loss.type]
        self.D_loss_fun = LOSS[cfg.model.discriminator.D_loss.type]

        if cfg.model.discriminator.is_entropy_input:  # AdvEnt
            self.D_preprocess_fun = lambda x: prob_2_entropy(F.softmax(x, dim=1))
        else:  # AdaptSegNet
            self.D_preprocess_fun = lambda x: F.softmax(x, dim=1)

        if cfg.model.predictor.ent_loss.weight > 0:  # MinEnt
            self.ent_loss_fun = lambda x: entropy_loss(F.softmax(x, dim=1))  # mean entropy for each pixel

    def forward(self, s_img, t_img=None, s_lbl=None):
        # get prediction output of source image
        s_logits, _ = self.seg_model(s_img)
        s_logits = F.interpolate(s_logits, size=s_img.shape[2:], mode='bilinear', align_corners=True)

        if self.training:
            # get prediction output of target image
            t_logits, _ = self.seg_model(t_img)
            t_logits = F.interpolate(t_logits, size=t_img.shape[2:], mode='bilinear', align_corners=True)

            losses = {}
            # loss for training G
            # source seg loss
            losses['source_seg_loss'] = self.cfg.model.predictor.seg_loss.source_weight * self.seg_loss_fun(s_logits, s_lbl)

            # adv loss
            t_D_logits = self.D(self.D_preprocess_fun(t_logits))
            is_source = torch.zeros_like(t_D_logits).cuda()
            losses['adv_loss'] = self.cfg.model.discriminator.D_loss.adv_weight * self.D_loss_fun(t_D_logits, is_source)

            # loss for training D
            # discriminator loss
            s_D_logits = self.D(self.D_preprocess_fun(s_logits.detach()))
            t_D_logits = self.D(self.D_preprocess_fun(t_logits.detach()))
            is_source = torch.zeros_like(s_D_logits).cuda()
            is_target = torch.ones_like(t_D_logits).cuda()
            losses['D_loss'] = self.cfg.model.discriminator.D_loss.weight * (self.D_loss_fun(s_D_logits, is_source) +
                                                                             self.D_loss_fun(t_D_logits, is_target)) / 2

            # entropy loss of target in all region
            if self.cfg.model.predictor.ent_loss.weight > 0:
                losses['target_ent_loss'] = self.cfg.model.predictor.ent_loss.weight * self.ent_loss_fun(t_logits)

            return losses
        else:
            return {'logits': s_logits}


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps"""
    _, c, _, _ = prob.size()

    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))
