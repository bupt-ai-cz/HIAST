# -*- coding: utf-8 -*
import numpy as np
import cv2
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

        if self.cfg.dataset.source.type == 'SYNTHIA':
            self.ignored_classes = [9, 14, 16]
        else:
            self.ignored_classes = None

        # init_class_value和selected_classes必须给一个值
        assert (init_class_value is not None and selected_classes is None) or (init_class_value is None and selected_classes is not None)
        if init_class_value is not None:
            self.update_setting(init_class_value)
            print('%% Init copy paste with class_value: {}'.format(init_class_value))
        if selected_classes is not None:
            self.selected_classes = selected_classes
            print('%% Init copy paste with selected_classes: {}'.format(selected_classes))

    def update_setting(self, class_value):
        if self.ignored_classes is not None:
            for c in self.ignored_classes:
                class_value[c] = np.inf

        self.selected_classes = np.argsort(class_value)[:self.cfg.preprocessor.copy_paste.selected_num_classes]  # select classes with smaller class value
        print('%% Update copy paste setting with class_value: {}'.format(class_value))

    def resize(self, img, lbl, w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, lbl

    def run(self, img, lbl):
        """
            普通的CopyPaste
            具体过程：
                1. 输入1张图像
                2. 随机选择一张图像c
                3. 将图像c中含有的所有需要Copy的类别给Copy到输入图像中
            """
        for tmp_idx in np.random.randint(0, len(self.dataset_copy_from), self.cfg.preprocessor.copy_paste.times):  # randomly select some images to copy
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

            # print('copy class: {} from img: {}, {}'.format([c for c in self.selected_classes if c in lbl_], tmp_idx, path_))

        return img, lbl
