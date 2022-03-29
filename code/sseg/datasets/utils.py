# -*- coding: utf-8 -*
import json
import os
import torchvision
import torch
import numpy as np


def get_path_list(json_path, image_dir):
    with open(json_path) as f:
        json_data = json.load(f)

    img_path_list = [os.path.join(image_dir, i['image_name']) for i in json_data]
    lbl_path_list = [os.path.join(image_dir, i['mask_name']) for i in json_data]

    return img_path_list, lbl_path_list


def transform(img, lbl, style='iast'):
    """
    Image: Normalize + ToTensor
    Label: torch.from_numpy()
    """
    if style == 'iast':
        img_trans_fun = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        lbl_trans_fun = lambda x: torch.from_numpy(x).long()
    elif style == 'advent':
        img_trans_fun = lambda x: torchvision.transforms.ToTensor()(
            x[:, :, ::-1].astype(np.float32) - np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32))
        lbl_trans_fun = lambda x: torch.from_numpy(x).long()
    else:
        raise ValueError

    if isinstance(img, (list, tuple)):
        img_transformed = [img_trans_fun(i) for i in img]
    else:
        img_transformed = img_trans_fun(img)
    if isinstance(lbl, (list, tuple)):
        lbl_transformed = [lbl_trans_fun(i) for i in lbl]
    else:
        lbl_transformed = lbl_trans_fun(lbl)

    return img_transformed, lbl_transformed


def preprocess_label(lbl, id_map, ignored_index=255):
    """Convert label to gray map"""
    assert len(lbl.shape) == 2, 'Only label with shape of [H, W] is valid'
    lbl_processed = ignored_index * np.ones(lbl.shape, dtype=np.uint8)
    for k, v in id_map.items():
        lbl_processed[lbl == k] = v
    return lbl_processed


def parse_resize_params(aug_type):
    """Return height and width for Resizing, eg. 'PRS-512-1024'."""
    args = aug_type.split('-')
    assert len(args) == 3, 'aug_type should be as "PRS-512-1024"'
    return [int(args[1]), int(args[2])]


def get_aug_fun(aug_type, build_aug_fun):
    """Parse and build one or multiple aug_fun."""
    assert isinstance(aug_type, (list, tuple))
    if len(aug_type) >= 2:  # aug_type=['AA', 'BB', ...]
        return [build_aug_fun(i) for i in aug_type]
    elif len(aug_type) == 1:  # aug_type=['AA']
        return build_aug_fun(aug_type[0])
    else:  # aug_type=[]
        return build_aug_fun(None)
