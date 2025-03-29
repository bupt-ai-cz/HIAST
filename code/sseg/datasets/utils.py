# -*- coding: utf-8 -*
import json
import os
import torchvision
import torch
import numpy as np


def str_to_num(list):
    """Convert string list into number list"""
    list_unique = np.unique(list)
    list_unique_len = len(list_unique)
    new_list = np.array([], dtype=int)
    for i in list:
        for j in range(list_unique_len):
            if list_unique[j] == i:
                new_list = np.append(new_list, int(j))
    return new_list


def get_path_list(json_path, image_dir):
    """Get path lists from json file."""
    with open(json_path) as f:
        json_data = json.load(f)

    img_path_list = [os.path.join(image_dir, i['image_name']) for i in json_data]
    lbl_path_list = [os.path.join(image_dir, i['mask_name']) for i in json_data]
    if json_path.split("/")[-1].split("_")[0] == "cityscapes":
        city_list = [i.split("/")[4] for i in img_path_list]
        city_list = str_to_num(city_list)
    else:
        city_list = [0 for i in img_path_list]

    return img_path_list, lbl_path_list, city_list


def transform(img, lbl, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Image: Normalize + ToTensor
    Label: torch.from_numpy()
    """
    img_trans_fun = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean, std)])
    lbl_trans_fun = lambda x: torch.from_numpy(x).long()

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
