# -*- coding: utf-8 -*
from torch import nn
from torch.nn import functional as F
import torch
from sseg.models.modules.seg_models import build_seg_model
from utils.registry.registries import LOSS, MODEL


@MODEL.register('SelfTrainingSegmentor')
class SelfTrainingSegmentor(nn.Module):

    def __init__(self, cfg):
        super(SelfTrainingSegmentor, self).__init__()
        self.cfg = cfg
        self.seg_model = build_seg_model(cfg)

        self.seg_loss_fun = LOSS[cfg.model.predictor.seg_loss.type]
        self.kld_loss_fun = _kld
        self.ent_loss_fun = _entropy
        if cfg.cst_training.is_enabled:
            self.cst_loss_fun = LOSS[cfg.cst_training.cst_loss.type]

    def forward(self, t_img):
        t_logits = self.seg_model(t_img)
        t_logits = F.interpolate(t_logits, size=t_img.shape[2:], mode='bilinear', align_corners=True)
        return {'logits': t_logits}

    def compute_loss(self, t_logits, t_plbl, t_cst_lbl=None):
        losses = {}

        # target seg loss with pseudo label
        losses['target_seg_loss'] = self.cfg.model.predictor.seg_loss.target_pseudo_weight * self.seg_loss_fun(t_logits, t_plbl)

        reg_weight_confident, reg_weight_ignored = build_region_weight(t_logits, t_plbl)
        # kl div loss (confident region)
        if self.cfg.model.predictor.kld_loss.weight > 0:
            losses['kld_confident_loss'] = self.cfg.model.predictor.kld_loss.weight * self.kld_loss_fun(t_logits, reg_weight_confident)

        # entropy loss (ignored region)
        if self.cfg.model.predictor.ent_loss.weight > 0:
            losses['ent_ignored_loss'] = self.cfg.model.predictor.ent_loss.weight * self.ent_loss_fun(t_logits, reg_weight_ignored)

        # consistency loss
        if t_cst_lbl is not None and self.cfg.cst_training.is_enabled and self.cfg.cst_training.cst_loss.weight > 0:
            losses['cst_loss'] = self.cfg.cst_training.cst_loss.weight * \
                                 self.cst_loss_fun(t_logits, t_cst_lbl, refer_labels=t_plbl, region=self.cfg.cst_training.cst_loss.region)

        return losses


def build_region_weight(t_logits, t_plbl):
    reg_val_matrix = torch.ones_like(t_plbl).type_as(t_logits)  # [B, H, W]
    reg_val_matrix[t_plbl == 255] = 0
    reg_val_matrix = reg_val_matrix.unsqueeze(dim=1)  # [B, 1, H, W]
    reg_ignore_matrix = 1 - reg_val_matrix  # [B, 1, H, W]
    reg_weight = torch.ones_like(t_logits)  # [B, C, H, W]
    reg_weight_confident = reg_weight * reg_val_matrix  # [B, C, H, W]
    reg_weight_ignored = reg_weight * reg_ignore_matrix  # [B, C, H, W]

    return reg_weight_confident, reg_weight_ignored


def _entropy(logits, weight):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight > 0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classed = logits.size()[1]
    entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
    entropy_reg = torch.sum(entropy) / val_num
    return entropy_reg


def _kld(logits, weight):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight > 0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classes = logits.size()[1]
    kld = - 1 / num_classes * weight * logits_log_softmax
    kld_reg = torch.sum(kld) / val_num
    return kld_reg
