# -*- coding: utf-8 -*
import albumentations as A
from albumentations.core.composition import BaseCompose
import numpy as np
import random
import cv2
import json
import os


def aug(aug_fun, img, lbl, seed=None):
    if seed is not None:
        random.seed(seed)
    if isinstance(aug_fun, (list, tuple)):
        return __aug_all(aug_fun, img, lbl)
    else:
        return __aug_one(aug_fun, img, lbl)


def __aug_one(aug_fun, img, lbl):
    if aug_fun is None:  # no operation
        return img, lbl
    elif isinstance(lbl, (list, tuple)):  # multiple labels
        aug_result = aug_fun(image=img, masks=lbl)
        return aug_result['image'], aug_result['masks']
    else:  # single label
        aug_result = aug_fun(image=img, mask=lbl)
        return aug_result['image'], aug_result['mask']


def __aug_all(aug_fun, img, lbl, mode='serial'):
    assert isinstance(aug_fun, (list, tuple)) and len(aug_fun) >= 2

    augmented_imgs = []
    augmented_lbls = []

    img_init = img.copy()
    lbl_init = lbl.copy()
    for i in aug_fun:
        if mode == 'serial':  # serial
            img, lbl = __aug_one(i, img, lbl)
        else:  # parallel
            img, lbl = __aug_one(i, img_init, lbl_init)
        augmented_imgs.append(img)
        augmented_lbls.append(lbl)

    return augmented_imgs, augmented_lbls


def resize(h, w):
    """Resize to fixed size."""
    return A.Resize(h, w, p=1.0)


def flip_crop_resize(h, w, min_max_height, w2h_ratio):
    """Used in IAST and HIAST. Randomly flip, randomly crop, and resize to fixed size."""
    return A.Compose([A.HorizontalFlip(p=0.5),
                      A.RandomSizedCrop(height=h, width=w, min_max_height=min_max_height, w2h_ratio=w2h_ratio)])


def resize_crop(h, w, h_c, w_c):
    """Used in DSP, CAMix, and DACS. Resize to fixed size, and randomly crop. """
    return A.Compose([A.Resize(h, w, p=1.0),
                      A.RandomCrop(h_c, w_c, p=1.0)])


def simple_color_aug():
    """Used in DACS, DSP, CAMix. Color jittering and gaussian blurring."""
    return A.Compose([A.ColorJitter(p=0.5),
                      A.GaussianBlur(blur_limit=(3, 41), p=0.5)])


def complex_color_aug(selected_num=3):
    """Used in HIAST."""
    aug_fun_pool = [A.ColorJitter(p=0.5),
                    A.GaussianBlur(blur_limit=(3, 41), p=0.5),
                    A.RandomContrast(limit=(0, 3), p=0.5),
                    A.RandomBrightness(limit=0.5, p=0.5),
                    A.Posterize(num_bits=4, p=0.5),
                    A.Equalize(p=0.5),
                    A.Solarize(p=0.5),
                    A.ToGray(p=0.5)]

    # https://github.com/albumentations-team/albumentations/blob/0173bfa4956780ed9b523447487233018fd25433/albumentations/core/composition.py#L313
    # Select N transforms to apply. Selected transforms will be called with `force_apply=True`.
    # Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.
    # return A.SomeOf(transforms=aug_fun_pool, n=selected_num)

    return SomeOf(transforms=aug_fun_pool, n=selected_num)


def fda(target_json_path, target_image_dir, beta_limit=0.001):
    """Fourier Domain Adaptation. http://arxiv.org/abs/2004.05498"""
    with open(target_json_path) as f:
        json_data = json.load(f)
    target_img_path_list = [os.path.join(target_image_dir, i['image_name']) for i in json_data]

    # read_fn在多进程运行时不能使用lambda表达式，将会报无法序列化的错误
    return A.FDA(reference_images=target_img_path_list, beta_limit=beta_limit, read_fn=__imread, p=1.0)


def __imread(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


class SomeOf(BaseCompose):
    """
    SomeOf rewritten, randomly select transforms by uniform distribution, and apply them with their probability parameters
    https://github.com/albumentations-team/albumentations/blob/5a3b2301f6ebfbece7a2e8c1e134fb08cfdfac86/albumentations/core/composition.py#L313-L350
    """

    def __init__(self, transforms, n, replace=False, p=1):
        super(SomeOf, self).__init__(transforms, p)
        self.n = n
        self.replace = replace

    def __call__(self, force_apply=False, **data):
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if force_apply or random.random() < self.p:
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            # randomly choice by uniform distribution without replacement
            transforms = random_state.choice(self.transforms.transforms, size=self.n, replace=self.replace)
            for t in transforms:
                data = t(**data)
        return data

    def _to_dict(self):
        dictionary = super(SomeOf, self)._to_dict()
        dictionary.update({"n": self.n, "replace": self.replace})
        return dictionary
