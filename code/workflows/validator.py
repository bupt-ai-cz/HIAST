# -*- coding: utf-8 -*
import torch
import tqdm
from torch.nn import functional as F
from utils import metrics, utils
import numpy as np
import os
from utils.registry.registries import DATASET
from torch.utils.data import DataLoader
from PIL import Image


class Validator():

    def __init__(self, cfg):
        self.cfg = cfg
        self.initialize()

    def initialize(self):
        # init model
        self.model = utils.load_model(self.cfg, resume_from=self.cfg.validate.resume_from).cuda()

        # init dataset for validation
        v_dataset = DATASET[self.cfg.dataset.val.type](self.cfg,
                                                       self.cfg.dataset.val.json_path,
                                                       self.cfg.dataset.val.image_dir,
                                                       num_classes=self.cfg.dataset.num_classes)
        self.v_loader = DataLoader(v_dataset, self.cfg.validate.batch_size, num_workers=self.cfg.dataset.num_workers, pin_memory=True)

        if self.cfg.validate.color_mask_dir_path is not None:
            assert not os.path.exists(self.cfg.validate.color_mask_dir_path) or len(os.listdir(self.cfg.validate.color_mask_dir_path)) == 0
            utils.create_dir(self.cfg.validate.color_mask_dir_path)

    def get_multi_scale_and_flip_logits(self, imgs, is_softmax=True):
        pred_result_list = []
        if is_softmax:
            pred_fun = lambda x: F.softmax(self.model(x)['logits'], dim=1)
        else:
            pred_fun = lambda x: self.model(x)['logits']

        for size in self.cfg.validate.resize_sizes:
            assert len(size) == 2 and size[0] <= size[1], \
                'Please input right format of each resize_size: [height, width] and height <= width, such as [512, 1024]'

            tmp_imgs = F.interpolate(imgs, size, mode='bilinear', align_corners=True)
            pred_result = pred_fun(tmp_imgs)

            if self.cfg.validate.is_flip:  # 水平翻折
                flip_logits = pred_fun(torch.flip(tmp_imgs, dims=[3]))
                pred_result += torch.flip(flip_logits, dims=[3])

            pred_result = F.interpolate(pred_result, imgs.size()[2:], mode='bilinear', align_corners=True)
            pred_result_list.append(pred_result)

        return sum(pred_result_list)

    def colorize_mask(self, mask):
        # the palette here is actually composed of RGB values of colors of each category, such as [c1_r, c1_g, c1_b, c2_r, c2_g, c2_b, ......]
        if self.cfg.dataset.num_classes == 19:  # GTAV/SYN-to-Cityscapes
            palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                       220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
                       0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        elif self.cfg.dataset.num_classes == 9:  # Cityscapes-to-Oxford
            palette = [70, 130, 180, 220, 20, 60, 119, 11, 32, 0, 0, 142, 220, 220, 0, 250, 170, 30, 70, 70, 70, 244, 35, 232, 128, 64, 128]
        else:
            raise NotImplementedError
        # mask: numpy array of the mask
        color_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        color_mask.putpalette(palette)
        return color_mask

    def save_color_mask(self, lbls_pred, img_paths):
        for lbl_pred, img_path in zip(lbls_pred, img_paths):
            color_mask = self.colorize_mask(lbl_pred)
            save_path = os.path.join(self.cfg.validate.color_mask_dir_path, os.path.basename(img_path))
            color_mask.save(save_path)

    def run(self):
        print('%% batch_size: {}'.format(self.cfg.validate.batch_size))
        print('%% num_classes: {}'.format(self.cfg.dataset.num_classes))
        print('%% resize_sizes: {}'.format(self.cfg.validate.resize_sizes))
        print('%% is_flip: {}'.format(self.cfg.validate.is_flip))
        print('%% color_mask_dir_path: {}'.format(self.cfg.validate.color_mask_dir_path))

        intersection_sum = 0
        union_sum = 0

        self.model.eval()
        with torch.no_grad():
            for data in tqdm.tqdm(self.v_loader, desc='Validation', ncols=160):
                imgs = data['images'].cuda()
                lbls = data['labels'].cuda()
                img_paths = data['image_paths']

                results = self.get_multi_scale_and_flip_logits(imgs)
                lbls_pred = results.argmax(dim=1)

                intersection, union = metrics.intersectionAndUnionGPU(lbls_pred, lbls, self.cfg.dataset.num_classes)
                intersection_sum += intersection
                union_sum += union

                if self.cfg.validate.color_mask_dir_path is not None:
                    self.save_color_mask(lbls_pred.cpu().numpy(), img_paths)

        iou = intersection_sum.cpu().numpy() / (union_sum.cpu().numpy() + 1e-10)
        miou = np.mean(iou)

        if 'SYNTHIA' in self.cfg.dataset.source.type:  # SYNTHIA to Cityscapes (16 and 13 classes)
            miou *= 19 / 16
            iu_13 = iou.copy()
            iu_13[3:6] = 0
            miou_13 = np.mean(iu_13) * 19 / 13
            print('miou_16: {:.4f}, miou_13: {:.4f}, iou: {}'.format(miou, miou_13, {c: round(v, 4) for c, v in enumerate(iou)}))
        else:
            print('miou: {:.4f}, iou: {}'.format(miou, {c: round(v, 4) for c, v in enumerate(iou)}))
