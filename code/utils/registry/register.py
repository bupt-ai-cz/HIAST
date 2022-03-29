# -*- coding: utf-8 -*
# Import the needed package here for registration
from sseg.models.modules import losses
from sseg.datasets.loader import gtav_dataset, synthia_dataset, cityscapes_dataset, oxford_dataset
from sseg.models.modules.seg_models import deeplab_v2, deeplab_v2_advent
from sseg.models.segmentors import source_only_segmentor, adversarial_warmup_segmentor, self_training_segmentor
from sseg.datasets import preprocessor
