# -*- coding: utf-8 -*
from sseg.datasets.loader.base_dataset import BaseDataset
import numpy as np
from PIL import Image
from sseg.datasets import augmentations, utils
from utils.registry.registries import DATASET


@DATASET.register('Cityscapes')
class CityscapesDataset(BaseDataset):
    # Only for 9 classes (Cityscapes to Oxford RobotCar)
    __id_map = {0: 8, 1: 7, 2: 6, 3: 255, 4: 255, 5: 255, 6: 5, 7: 4, 8: 255, 9: 255, 10: 0, 11: 1, 12: 1, 13: 3, 14: 3, 15: 3, 16: 255, 17: 2, 18: 2}

    def read_label(self, path):
        assert self.num_classes in [9, 19], 'num_classes of Cityscapes is only valid for 9 (Cityscapes to Oxford RobotCar) and 19 (GTAV/SYNTHIA to Cityscapes)'

        lbl = np.array(Image.open(path), dtype=np.uint8)
        if self.num_classes == 9:  # convert 19 classes into 9 classes
            lbl = utils.preprocess_label(lbl, self.__id_map)
        return lbl

    def build_aug_fun(self, aug_type):
        if aug_type is None or aug_type == '':
            return None
        elif aug_type == 'MS':
            return augmentations.flip_crop_resize(512, 1024, min_max_height=(341, 1000), w2h_ratio=2)
        elif aug_type == 'OMS':  # Cityscapes to Oxford RobotCar
            return augmentations.flip_crop_resize(768, 1024, min_max_height=(341, 1000), w2h_ratio=1280 / 960)
        elif aug_type == 'DACS':
            return augmentations.resize_crop(512, 1024, 512, 512)
        elif aug_type == 'SCA':  # simple color augmentation
            return augmentations.simple_color_aug()
        elif aug_type == 'CCA':  # complex color augmentation
            return augmentations.complex_color_aug()
        elif 'PRS' in aug_type:  # resize for generating pseudo labels, PRS-640-1280, PRS-768-1536, PRS-960-1280
            h, w = utils.parse_resize_params(aug_type)
            return augmentations.resize(h, w)
        elif aug_type == 'FDA-Source':  # transfer style to source domain
            assert self.cfg.dataset.source.type == 'GTAV' or self.cfg.dataset.source.type == 'SYNTHIA', 'FDA-Source for Cityscapes is only valid for GTAV/SYNTHIA to Cityscapes'
            return augmentations.fda(self.cfg.dataset.source.json_path, self.cfg.dataset.source.image_dir)
        elif aug_type == 'FDA-Target':  # transfer style to target domain
            assert self.cfg.dataset.source.type == 'Oxford', 'FDA-Target for Cityscapes is only valid for Cityscapes to Oxford RobotCar'
            return augmentations.fda(self.cfg.dataset.target.json_path, self.cfg.dataset.target.image_dir)
        else:
            raise ValueError('aug_type is not valid')
