# -*- coding: utf-8 -*
from torch import nn
from torch.nn import functional as F
from sseg.models.modules.seg_models import build_seg_model
from utils.registry.registries import LOSS, MODEL


@MODEL.register('SourceOnlySegmentor')
class SourceOnlySegmentor(nn.Module):

    def __init__(self, cfg):
        super(SourceOnlySegmentor, self).__init__()
        self.cfg = cfg
        self.seg_model = build_seg_model(cfg)
        self.seg_loss_fun = LOSS[cfg.model.predictor.seg_loss.type]

    def forward(self, img, lbl=None):
        logits, _ = self.seg_model(img)
        logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear', align_corners=True)

        if self.training:
            return {'seg_loss': self.cfg.model.predictor.seg_loss.source_weight * self.seg_loss_fun(logits, lbl)}
        else:
            return {'logits': logits}
