# -*- coding: utf-8 -*
# Import the needed package here for registration
from sseg.models.modules import losses
from sseg.datasets.loader import gtav_dataset, synthia_dataset, cityscapes_dataset, oxford_dataset
from sseg.models.modules.seg_models.deeplab_v2 import DeepLab_V2
from sseg.models.segmentors import source_only_segmentor, adversarial_warmup_segmentor, self_training_segmentor
from workflows import pseudo_label_generator
from sseg.datasets import preprocessor
from workflows.trainer import source_only_trainer, adversarial_warmup_trainer, self_training_trainer, consistency_self_training_trainer
