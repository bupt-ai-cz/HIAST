# -*- coding: utf-8 -*
from utils.registry.registries import SEG_MODEL


def build_seg_model(cfg):
    print('%% use segment model: {}'.format(cfg.model.seg_model.type))
    return SEG_MODEL[cfg.model.seg_model.type](num_classes=cfg.dataset.num_classes)
