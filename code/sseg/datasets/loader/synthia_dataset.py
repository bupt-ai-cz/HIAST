# -*- coding: utf-8 -*
from sseg.datasets.loader.base_dataset import BaseDataset
import imageio
import numpy as np
from sseg.datasets import augmentations, utils
from utils.registry.registries import DATASET


@DATASET.register('SYNTHIA')
class SYNTHIADataset(BaseDataset):
    __id_map = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5, 15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12, 8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}

    def read_label(self, path):
        assert self.num_classes == 19, 'num classes should be 19 for SYNTHIA to Cityscapes (actually it is 16)'

        lbl = np.asarray(imageio.imread(path, format='PNG-FI'))[:, :, 0]
        lbl = utils.preprocess_label(lbl, self.__id_map)
        return lbl

    def build_aug_fun(self, aug_type):
        if aug_type is None or aug_type == '':
            return None
        elif aug_type == 'MS':
            return augmentations.flip_crop_resize(512, 1024, min_max_height=(341, 640), w2h_ratio=2)
        elif aug_type == 'DACS':
            return augmentations.resize_crop(760, 1280, 512, 512)
        elif 'PRS' in aug_type:  # resize for generating pseudo labels, PRS-640-1280, PRS-768-1536, PRS-960-1280
            h, w = utils.parse_resize_params(aug_type)
            return augmentations.resize(h, w)
        elif aug_type == 'FDA-Target':  # transfer style to target domain
            return augmentations.fda(self.cfg.dataset.target.json_path, self.cfg.dataset.target.image_dir)
        else:
            raise ValueError('aug_type is not valid')
