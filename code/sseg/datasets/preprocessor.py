# -*- coding: utf-8 -*
import numpy as np
import cv2
import torch
from utils.registry.registries import PREPROCESSOR


@PREPROCESSOR.register('CopyPaste')
class CopyPaste:

    def __init__(self, cfg, dataset_copy_from, init_class_value=None, selected_classes=None):
        """
        :param dataset_copy_from: 表示从哪一个Dataset内CP内容
        :param init_class_value: 使用class_value以及copy_paste.selected_num_classes初始化需要CP的类别
        :param selected_classes: 直接设置需要CP的类别
        """
        self.cfg = cfg
        self.dataset_copy_from = dataset_copy_from  # copy data from this dataset
        self.samples_with_class = self.dataset_copy_from.get_samples_class()

        if self.cfg.dataset.source.type == 'SYNTHIA':
            self.ignored_classes = [9, 14, 16]
        else:
            self.ignored_classes = None

        # get the hard classes and sampling probability
        assert init_class_value is not None
        self.update_setting(init_class_value)
        print('%% Init copy paste with class_value: {}'.format(init_class_value))

        self.class_probs = self.get_hard_class_probs(init_class_value)

    def update_setting(self, class_value):
        if self.ignored_classes is not None:
            for c in self.ignored_classes:
                class_value[c] = np.inf

        self.selected_classes = np.argsort(class_value)[:self.cfg.preprocessor.copy_paste.selected_num_classes]  # select classes with smaller class value
        print('%% Update copy paste setting with class_value: {}'.format(class_value))


    def get_hard_class_probs(self, init_class_value):
        probs = np.power(init_class_value, 2)
        class_probs = (1 - probs) / np.sum(probs)
        return class_probs

    def resize(self, img, lbl, target_shape):
        if len(target_shape) > 2:
            target_shape = (target_shape[0], target_shape[1])  # [H, W]

        img = cv2.resize(img, target_shape[::-1], interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, target_shape[::-1], interpolation=cv2.INTER_NEAREST)
        return img, lbl

    def run(self, img, lbl):
        """
            普通的CopyPaste
            具体过程：
                1. 输入1张图像img
                2. 根据采样率随机选择一个类别c
                2. 随机选择一张包含类别c的图像img_
                3. 将图像img_中含有的所有困难类别给Copy到输入图像img中
            """
        select_c = np.random.choice([i for i in range(self.cfg.dataset.num_classes)], size=1, replace=False ,p=self.class_probs)
        while True:
            if select_c[0] in self.selected_classes:
                break
            else:
                select_c = np.random.choice([i for i in range(self.cfg.dataset.num_classes)], size=1, replace=False ,p=self.class_probs)
        
        for c_dix in select_c:
            while True:
                file_name = np.random.choice(self.samples_with_class[c_dix])
                if file_name in self.dataset_copy_from.file_to_idx.keys():
                    tmp_idx = self.dataset_copy_from.get_file_to_idx(file_name)
                    break

           # 如果读取图像有问题，则跳过这一次CP
            try:
                img_, lbl_, path_ = self.dataset_copy_from.load_data(tmp_idx)
            except Exception as e:
                print('## {} in loading {}: {}'.format(repr(e), tmp_idx, self.dataset_copy_from.img_path_list[tmp_idx]))
                continue

            if img.shape != img_.shape:
                img_, lbl_ = self.resize(img_, lbl_, w=lbl.shape[1], h=lbl.shape[0])

            selected_mask = np.zeros_like(lbl, dtype=np.bool_)
            for c in self.selected_classes:
                selected_mask[lbl_ == c] = True

            img[selected_mask] = img_[selected_mask]
            lbl[selected_mask] = lbl_[selected_mask]

        return img, lbl
