# -*- coding: utf-8 -*
import torch
from torch.utils.data import Dataset
from sseg.datasets import augmentations, utils
import os
import json
from PIL import Image
import numpy as np
import cv2
from sseg.datasets.preprocessor import CopyPaste
import os.path as osp

class BaseDataset(Dataset):

    def __init__(self, cfg, json_path, image_dir, pseudo_dir=None, aug_type=[], num_classes=19):
        self.cfg = cfg
        self.pseudo_dir = pseudo_dir
        self.num_classes = num_classes
        self.preprocessor = None
        self.aug_fun = utils.get_aug_fun(aug_type, self.build_aug_fun)  # aug_fun or [aug_fun_0, aug_fun_1]

        # get the path list of images and labels
        self.img_path_list, self.lbl_path_list, self.city_list = utils.get_path_list(json_path, image_dir)
        assert len(self.img_path_list) == len(self.lbl_path_list), 'images and labels should have the same number'
        self.file_to_idx = {} #  get idx by file name
        for idx in range(len(self.img_path_list)):
            img_idx_name = self.img_path_list[idx].split('/')[-1]
            self.file_to_idx[img_idx_name] = idx

        # stat the images in which each class exists for subsequent copy paste
        if self.pseudo_dir is not None:
            self.samples_with_class = self.stat_samples_with_class(self.pseudo_dir.split(self.pseudo_dir.split('/')[-1])[0])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        # prevent deadlock when multithreading
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        if self.preprocessor is not None:
            if isinstance(self.preprocessor, CopyPaste):
                return self.get_item_with_copy_paste(index)
            else:
                raise NotImplementedError
        else:
            return self.original_get_item(index)

    def get_file_to_idx(self, file_name):
        return self.file_to_idx[file_name]

    def get_samples_with_class(self):
        return self.samples_with_class

    def get_aug(self):
        return self.aug_fun

    def get_city_list(self):
        return self.city_list
    
    def stat_samples_with_class(self, data_root):
        """Stat the images in which each class exists"""
        with open(osp.join(data_root, 'samples_with_class.json'), 'r') as of:
            samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v  for k, v in samples_with_class_and_n.items()
            }

        samples_with_class = {}
        for c in range(self.cfg.dataset.num_classes):
            samples_with_class[c] = []
            for file, pixels in sorted(samples_with_class_and_n[c], key=lambda item: item[1]):
                samples_with_class[c].append(file.split('/')[-1])
            samples_with_class[c] = samples_with_class[c][round(len(samples_with_class[c])*0.1):] # filter samples with too small pixels
            samples_with_class = {k: v for k, v in sorted(samples_with_class.items(), key=lambda item: item[0])}

        return samples_with_class

    def original_get_item(self, index):
        """Loading data and performing data augment and transform"""
        try:
            img, lbl, img_path = self.load_data(index)
        except Exception as e:
            print('## {} in loading {}: {}'.format(repr(e), index, self.img_path_list[index]))
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)

        # augment
        img, lbl = augmentations.aug(self.aug_fun, img, lbl, index)
        # transform
        img, lbl = utils.transform(img, lbl)

        return {
            'images': img,  # img or [img_0, img_1] (multiple augmented images)
            'labels': lbl,  # lbl or [lbl_0, lbl_1] (multiple augmented labels)
            'image_paths': img_path
        }

    def get_item_with_copy_paste(self, index):
        """Loading data and performing data augment and transform with copy paste"""
        result = {}
        try:
            img, lbl, img_path = self.load_data(index)
        except Exception as e:
            print('## {} in loading {}: {}'.format(repr(e), index, self.img_path_list[index]))
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)

        # preprocess (Copy Paste)
        img, lbl, copy_paste_mask = self.preprocessor.run(img, lbl)
        # augment
        img, lbl = augmentations.aug(self.aug_fun, img, lbl)
        # transform
        img, lbl = utils.transform(img, lbl)
        if copy_paste_mask is not None:
            copy_paste_mask = torch.from_numpy(copy_paste_mask).long()

        result['images'] = img  # img or [img_0, img_1]
        result['labels'] = lbl  # lbl or [lbl_0, lbl_1]
        result['image_paths'] = img_path
        if copy_paste_mask is not None:
            result['copy_paste_mask'] = copy_paste_mask

        return result

    def set_preprocessor(self, preprocessor):
        """Set preprocessor"""
        self.preprocessor = preprocessor
        print('%% use {}'.format(self.preprocessor.__class__.__name__))
        if isinstance(self.preprocessor, CopyPaste):
            print('%% copy from {} to {}, mode: {}'.format(self.preprocessor.dataset_copy_from.__class__.__name__, self.__class__.__name__,
                                                           self.cfg.preprocessor.copy_paste.mode))

    def read_label(self, path):
        """Read and transform, different datasets have different methods to read label."""
        raise NotImplementedError

    def build_aug_fun(self, aug_type):
        """Parse and build function for augmentation."""
        raise NotImplementedError

    def load_data_real_lbl(self, index):
        """Load an image with corresponding label."""
        img_path = self.img_path_list[index]
        lbl_path = self.lbl_path_list[index]
        # convert label path into pseudo label path
        img = np.array(Image.open(img_path), dtype=np.uint8)
        lbl = self.read_label(lbl_path)  # read and preprocess label
        
        if lbl is None:  # label for training set of Oxford RobotCar
            lbl = 255 * np.ones(img.shape[:-1], dtype=np.uint8)

        # ensure the same shape
        lbl = cv2.resize(lbl, img.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST)

        return img, lbl, img_path

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
