# -*- coding: utf-8 -*
from torch.utils.data import Dataset
from sseg.datasets import augmentations, utils
import os
from PIL import Image
import numpy as np
import cv2
from sseg.datasets.preprocessor import CopyPaste


class BaseDataset(Dataset):

    def __init__(self, cfg, json_path, image_dir, pseudo_dir=None, aug_type=(), num_classes=19, is_debug=False):
        self.cfg = cfg
        self.assert_cfg()
        self.pseudo_dir = pseudo_dir
        self.num_classes = num_classes
        self.is_debug = is_debug
        self.preprocessor = None
        print('%% transform style: {}'.format(self.cfg.dataset.transform_style))

        self.img_path_list, self.lbl_path_list = utils.get_path_list(json_path, image_dir)
        self.aug_fun = utils.get_aug_fun(aug_type, self.build_aug_fun)  # aug_fun or [aug_fun_0, aug_fun_1]

        assert len(self.img_path_list) == len(self.lbl_path_list), 'images and labels should have the same number'

    def assert_cfg(self):
        pass

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        try:
            img, lbl, img_path = self.load_data(index)

            # preprocess (Copy Paste, ClassMix, CutMix)
            if self.preprocessor is not None:
                img, lbl = self.preprocessor.run(img, lbl)

            # augment
            img, lbl = self.aug(img, lbl)

            # transform
            if not self.is_debug:
                img, lbl = utils.transform(img, lbl, style=self.cfg.dataset.transform_style)
        except Exception as e:
            from sseg.datasets.loader.gtav_dataset import GTAVDataset
            if isinstance(self, GTAVDataset):  # 只有在数据集为GTAVDataset的时候才处理异常
                print('## File: {}, Line number: {}, {} in loading {}: {}'.format(e.__traceback__.tb_frame.f_globals["__file__"], e.__traceback__.tb_lineno,
                                                                                  repr(e), index, self.img_path_list[index]))
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)
            else:
                raise e

        return {
            'images': img,  # img or [img_0, img_1] (multiple augmented images)
            'labels': lbl,  # lbl or [lbl_0, lbl_1] (multiple augmented labels)
            'image_paths': img_path
        }

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        print('%% use {}'.format(self.preprocessor.__class__.__name__))
        if isinstance(self.preprocessor, CopyPaste):
            print('%% copy from {} to {}, mode: {}'.format(self.preprocessor.dataset_copy_from.__class__.__name__, self.__class__.__name__,
                                                           self.cfg.preprocessor.copy_paste.mode))

    def read_label(self, path):
        """Read and transform, different datasets have different methods to read label."""
        raise NotImplementedError

    def aug(self, img, lbl):
        img, lbl = augmentations.aug(self.aug_fun, img, lbl)
        return img, lbl

    def build_aug_fun(self, aug_type):
        """Parse and build function for augmentation."""
        raise NotImplementedError

    def load_data(self, index):
        """Load an image with corresponding label."""
        img_path = self.img_path_list[index]
        lbl_path = self.lbl_path_list[index]
        # convert label path into pseudo label path
        if self.pseudo_dir is not None:
            pseudo_label_name = os.path.splitext(os.path.basename(img_path))[0] + '_pseudo_label.png'
            lbl_path = os.path.join(self.pseudo_dir, pseudo_label_name)

        img = np.array(Image.open(img_path), dtype=np.uint8)
        if self.pseudo_dir is not None:  # read pseudo label (gray mask)
            lbl = np.array(Image.open(lbl_path), dtype=np.uint8)
        else:
            lbl = self.read_label(lbl_path)  # read and preprocess label
        if lbl is None:  # label for training set of Oxford RobotCar
            lbl = 255 * np.ones(img.shape[:-1], dtype=np.uint8)

        # ensure the same shape
        lbl = cv2.resize(lbl, img.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST)

        return img, lbl, img_path
