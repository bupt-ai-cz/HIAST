# -*- coding: utf-8 -*
import numpy as np
import cv2
import torch
from numpy.lib.type_check import common_type
from utils.registry.registries import PREPROCESSOR
import os
import albumentations as A
from datetime import datetime

@PREPROCESSOR.register('CopyPaste')
class CopyPaste:

    def __init__(self, cfg, dataset_copy_from, init_class_value):
        self.cfg = cfg
        self.dataset_copy_from = dataset_copy_from  # copy data from this dataset

        if self.cfg.dataset.source.type == 'SYNTHIA':
            self.ignored_classes = [9, 14, 16]
        else:
            self.ignored_classes = None

        # get hard classes and sampling probability
        print('%% Init copy paste with class_value: {}'.format(init_class_value))
        self.class_value, self.hard_classes = self.get_hard_classes(init_class_value)
        self.samples_with_class = self.dataset_copy_from.get_samples_with_class()
        self.class_probs = self.calculate_class_probs()

    def calculate_class_probs(self):
        """Calculate sampling probability based on class threshold"""
        probs = torch.tensor(self.class_value)
        probs = (1 - probs) ** 2
        probs = probs / torch.sum(probs)
        return probs.numpy()

    def get_hard_classes(self, class_value):
        """Get hard classes based on class threshold"""
        if self.ignored_classes is not None:
            for c in self.ignored_classes:
                class_value[c] = np.inf
        
        hard_classes = np.argsort(class_value)[:self.cfg.preprocessor.copy_paste.selected_num_classes]  # select classes with smaller class value
        print('%% Update copy paste setting with class_value: {}'.format(class_value))
        return class_value, hard_classes

    def resize(self, img, lbl, target_shape):
        if len(target_shape.shape) > 2:
            target_shape = (target_shape[0], target_shape[1])  # [H, W]

        img = cv2.resize(img, target_shape[::-1], interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, target_shape[::-1], interpolation=cv2.INTER_NEAREST)
        return img, lbl

    def aug(self, aug_fun, img, lbl):
        if aug_fun is None:
            return img, lbl
        elif isinstance(lbl, (list, tuple)):
            aug_result = aug_fun(image=img, masks=lbl)
            return aug_result['image'], aug_result['masks']
        else:
            aug_result = aug_fun(image=img, mask=lbl)
            return aug_result['image'], aug_result['mask']

    def run(self, img, lbl):
        if self.cfg.preprocessor.copy_paste.mode == 'original':
            return self.run_original(img, lbl)
        else:
            return NotImplementedError

    def random_select(self, selected_classes):
        """Randomly choose a hard class based on sampling probability"""
        while True:
            select_c = np.random.choice([i for i in range(self.cfg.dataset.num_classes)], size=1, replace=False, p=self.class_probs)[0]
            if select_c in selected_classes:
                break
                
        return select_c

    def run_original(self, img, lbl):
        """
        # CopyPaste
        # Specific process:
        #   1. Input one image
        #   2. Randomly select another image c
        #   3. Copy all the required categories contained in image c to the input image
        """
        copy_paste_mask = np.ones_like(lbl, dtype=np.uint8) * 255  # record which region is copied from which class
        selected_classes = self.hard_classes  # used to select hard classes for copy paste
        exist_classes = []  # record which classes are already copied

        for _ in range(3):
            # randomly select a hard class according to sampling probability
            select_c = self.random_select(selected_classes)
            # randomly select a sample from the dataset to copy
            file_name = np.random.choice(self.samples_with_class[select_c])
            tmp_idx = self.dataset_copy_from.get_file_to_idx(file_name)
            img_, lbl_, path_ = self.dataset_copy_from.load_data(tmp_idx)

            if img.shape != img_.shape:
                img_, lbl_ = self.resize(img_, lbl_, lbl.shape)

            # find the hard classes region mask
            selected_mask = np.zeros_like(lbl, dtype=np.bool)
            for c in self.hard_classes:
                if c in selected_classes and c not in exist_classes:
                    exist_classes.append(c)  # record which classes are already copied
                selected_mask[lbl_ == c] = True
                copy_paste_mask[lbl_ == c] = c

            # copy the selected region to the input image
            img[selected_mask] = img_[selected_mask]
            lbl[selected_mask] = lbl_[selected_mask]

            # if the number of classes that have been copied reaches half, exit;
            # otherwise select a new image from the hard classes that has not been selected and repeat the above operation
            non_exist_classes = [c for c in self.hard_classes if c not in exist_classes]            
            if len(exist_classes) >= len(self.hard_classes) * 0.5:
                break
            else:
                selected_classes = non_exist_classes

        return img, lbl, copy_paste_mask