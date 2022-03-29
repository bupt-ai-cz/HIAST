# -*- coding: utf-8 -*
from sseg.datasets.loader.base_dataset import BaseDataset
import numpy as np
from PIL import Image
from sseg.datasets import augmentations, utils
from utils.registry.registries import DATASET


@DATASET.register('Oxford')
class OxfordDataset(BaseDataset):
    # https://github.com/layumi/Seg-Uncertainty/blob/b5fff9ceb84fa0294955c8340ef2b5e9c9bb93d7/dataset/robot_dataset.py#L39-L44
    __id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 11: 8, 12: 8, 13: 8, 14: 8, 17: 8}

    def read_label(self, path):
        assert self.num_classes == 9, 'num_classes of Oxford RobotCar is only valid for 9 (Cityscapes to Oxford RobotCar)'

        if not path.endswith('.png'):  # There are not labels for training set of Oxford RobotCar
            lbl = None
        else:
            lbl = np.asarray(Image.open(path), dtype=np.uint8)[:, :, 0]  # convert [H, W, 4] into [H, W]
            lbl = utils.preprocess_label(lbl, self.__id_map)
        return lbl

    def build_aug_fun(self, aug_type):
        if aug_type is None or aug_type == '':
            return None
        elif aug_type == 'OMS':  # Cityscapes to Oxford RobotCar
            return augmentations.flip_crop_resize(768, 1024, min_max_height=(341, 900), w2h_ratio=1280 / 960)
        elif aug_type == 'SCA':  # simple color augmentation
            return augmentations.simple_color_aug()
        elif aug_type == 'CCA':  # complex color augmentation
            return augmentations.complex_color_aug()
        elif 'PRS' in aug_type:  # resize for generating pseudo labels, PRS-640-1280, PRS-768-1536
            h, w = utils.parse_resize_params(aug_type)
            return augmentations.resize(h, w)
        elif aug_type == 'FDA-Source':  # transfer style to source domain
            return augmentations.fda(self.cfg.dataset.source.json_path, self.cfg.dataset.source.image_dir)
        else:
            raise ValueError('aug_type is not valid')
