# -*- coding: utf-8 -*
from utils.registry.registries import SEG_MODEL


def build_seg_model(cfg):
    print('%% use segment model: {}'.format(cfg.model.seg_model.type))
    assert cfg.model.seg_model.type == 'DeepLab_V2'
    return SEG_MODEL[cfg.model.seg_model.type](num_classes=cfg.dataset.num_classes, output_dim=cfg.model.seg_model.output_dim)
