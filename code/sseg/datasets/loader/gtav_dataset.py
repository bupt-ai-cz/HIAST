# -*- coding: utf-8 -*
from sseg.datasets.loader.base_dataset import BaseDataset
from PIL import Image
import numpy as np
from sseg.datasets import augmentations, utils
from utils.registry.registries import DATASET


@DATASET.register('GTAV')
class GTAVDataset(BaseDataset):
    __id_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def read_label(self, path):
        lbl = np.asarray(Image.open(path), dtype=np.uint8)
        lbl = utils.preprocess_label(lbl, self.__id_map)
        return lbl

    def build_aug_fun(self, aug_type):
        if aug_type is None or aug_type == '':
            return None
        elif aug_type == 'MS':
            return augmentations.flip_crop_resize(512, 1024, min_max_height=(341, 950), w2h_ratio=2)
        elif aug_type == 'DACS':
            return augmentations.resize_crop(720, 1280, 512, 512)
        elif 'PRS' in aug_type:  # resize for generating pseudo labels, PRS-640-1280, PRS-768-1536, PRS-960-1280
            h, w = utils.parse_resize_params(aug_type)
            return augmentations.resize(h, w)
        elif aug_type == 'FDA-Target':  # transfer style to target domain
            return augmentations.fda(self.cfg.dataset.target.json_path, self.cfg.dataset.target.image_dir)
        else:
            raise ValueError('aug_type is not valid')
