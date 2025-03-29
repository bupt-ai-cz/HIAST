# -*- coding: utf-8 -*
import numpy as np
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
from utils.registry.registries import DATASET, PSEUDO_POLICY
from utils import utils
import cv2
import json


class BasePseudoGenerator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.statics_class = np.array([0] * self.cfg.dataset.num_classes)
        self.sample_stats = []
        self.samples_class = {i:[] for i in range(self.cfg.dataset.num_classes)}
        self.class_mean_probs = np.zeros(self.cfg.dataset.num_classes)
        
        self.initialize()

    def initialize(self):
        # init model
        self.model = utils.load_model(self.cfg, resume_from=self.cfg.pseudo_policy.resume_from).cuda()

        # init target dataset for pseudo label generation
        aug_type = ['PRS-{}-{}'.format(self.cfg.pseudo_policy.resize_size[0], self.cfg.pseudo_policy.resize_size[1])]  # ['PRS-height-width']
        self.t_dataset = DATASET[self.cfg.dataset.target.type](self.cfg,
                                                               self.cfg.dataset.target.json_path,
                                                               self.cfg.dataset.target.image_dir,
                                                               aug_type=aug_type,
                                                               num_classes=self.cfg.dataset.num_classes)
        self.t_loader = DataLoader(self.t_dataset, self.cfg.pseudo_policy.batch_size, shuffle=True, num_workers=self.cfg.dataset.num_workers, pin_memory=True)

        self.pseudo_label_save_dir = self.cfg.pseudo_policy.save_dir
        assert self.pseudo_label_save_dir is not None and \
               (not os.path.exists(self.pseudo_label_save_dir) or len(os.listdir(self.pseudo_label_save_dir)) == 0)
        utils.create_dir(self.pseudo_label_save_dir)

    def save_pseudo_label(self, plbl, img_path):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        plbl_save_path = os.path.join(self.pseudo_label_save_dir, '{}_pseudo_label.png'.format(img_name))
        cv2.imwrite(plbl_save_path, plbl.astype(np.uint8))

    def save_data(self):
        if self.class_threshold is not None:
            print('class threshold: {}'.format(self.class_threshold))
            np.save(os.path.join(self.pseudo_label_save_dir, '..', 'class_threshold.npy'), self.class_threshold)
        print('class statics number: {}'.format(self.statics_class))
        np.save(os.path.join(self.pseudo_label_save_dir, '..', 'statics_class.npy'), self.statics_class)
        print('class mean probabilities: {}'.format(self.class_mean_probs))
        np.save(os.path.join(self.pseudo_label_save_dir, '..', 'class_mean_probabilities.npy'), self.class_mean_probs)

        stats_json = json.dumps(self.sample_stats)
        with open(os.path.join(self.pseudo_label_save_dir, '..', 'sample_class_stats.json'), 'a') as f:
            f.write(stats_json)
        class_json = json.dumps(self.samples_class)
        with open(os.path.join(self.pseudo_label_save_dir, '..', 'samples_with_class.json'), 'a') as f:
            f.write(class_json)
        
    def run(self):
        raise NotImplementedError

    def select_and_save_confident_label(self, probs_pred, lbls_pred, img_paths):
        confident_label_list = []

        # select confident regions as pseudo label according to class threshold
        for prob_pred, lbl_pred, img_path in zip(probs_pred, lbls_pred, img_paths):
            current_stats = {}
            if self.class_threshold is not None:
                lbl_class_threshold = np.apply_along_axis(lambda x: [self.class_threshold[e] for e in x], 1, lbl_pred)
                ignored_index = prob_pred < lbl_class_threshold

                plbl = lbl_pred.copy()
                plbl[ignored_index] = 255
            else:  # argmax for no threshold policy
                plbl = lbl_pred

            for i in range(self.cfg.dataset.num_classes):
                pix_num = int(sum(sum((plbl == i))))
                if pix_num != 0:
                    current_stats[i] = pix_num
                    self.samples_class[i].append([img_path, current_stats[i]])
                    self.statics_class[i] += current_stats[i]
            current_stats['file'] = img_path
            self.sample_stats.append(current_stats)

            self.save_pseudo_label(plbl, img_path)

            confident_label_list.append(np.expand_dims(plbl, 0))  # [H, W] to [1, H, W]

        # update mean class probabilities
        confident_labels = np.concatenate(confident_label_list)  # [B, H, W]
        for c in range(self.cfg.dataset.num_classes):
            mask_ = (confident_labels == c)
            mean_value = np.mean(probs_pred[mask_])
            if not np.isnan(mean_value) and not np.isinf(mean_value):
                if self.class_mean_probs[c] == 0:
                    self.class_mean_probs[c] = mean_value
                else:
                    self.class_mean_probs[c] = self.class_mean_probs[c] * self.cfg.preprocessor.copy_paste.gamma + \
                                               mean_value * (1 - self.cfg.preprocessor.copy_paste.gamma)
        return plbl


@PSEUDO_POLICY.register('CT')
class ConstantThresholdPseudoGenerator(BasePseudoGenerator):

    def get_constant_threshold(self):
        return self.cfg.pseudo_policy.ct.threshold * np.ones(self.cfg.dataset.num_classes)

    def run(self):
        if len(os.listdir(self.pseudo_label_save_dir)) >= len(self.t_dataset):
            print('%% pseudo labels have existed')
        else:
            self.class_threshold = self.get_constant_threshold()
            self.model.eval()
            with torch.no_grad():
                for data in tqdm.tqdm(self.t_loader, desc='Generating pseudo labels (policy: {})'.format(self.cfg.pseudo_policy.type), ncols=160):
                    imgs = data['images'].cuda()
                    probs = F.softmax(self.model(imgs)['logits'], dim=1)

                    probs_pred, lbls_pred = probs.max(dim=1)
                    probs_pred = probs_pred.cpu().numpy()
                    lbls_pred = lbls_pred.cpu().numpy()

                    self.select_and_save_confident_label(probs_pred, lbls_pred, data['image_paths'])

            self.save_data()


@PSEUDO_POLICY.register('NT')
class NoThresholdPseudoGenerator(ConstantThresholdPseudoGenerator):

    def get_constant_threshold(self):
        return None


@PSEUDO_POLICY.register('CBST')
class CBSTPseudoGenerator(ConstantThresholdPseudoGenerator):

    def get_constant_threshold(self):
        class_probs_dict = {c: [] for c in range(self.cfg.dataset.num_classes)}  # store the prediction probability of all points in each category
        self.model.eval()
        with torch.no_grad():
            for data in tqdm.tqdm(self.t_loader, desc='Computing thresholds of CBST', ncols=160):
                imgs = data['images'].cuda()
                probs = F.softmax(self.model(imgs)['logits'], dim=1)

                probs_pred, lbls_pred = probs.max(dim=1)
                probs_pred = probs_pred.cpu().numpy()
                lbls_pred = lbls_pred.cpu().numpy()

                for c in range(self.cfg.dataset.num_classes):
                    temp_prob_list = probs_pred[lbls_pred == c].astype(np.float16)  # prediction probability of all pixels predicted as cls in the batch
                    class_probs_dict[c].extend(temp_prob_list[0:len(temp_prob_list):self.cfg.pseudo_policy.cbst.sample_interval])  # equal interval sampling save

        class_threshold = np.ones(self.cfg.dataset.num_classes)
        for c in range(self.cfg.dataset.num_classes):
            class_threshold[c] = np.quantile(class_probs_dict[c], 1 - self.cfg.pseudo_policy.cbst.p)  # for each category, select the upper p-score of the prediction probability as the threshold

        return class_threshold


@PSEUDO_POLICY.register('IAS')
class IASPseudoGenerator(BasePseudoGenerator):

    def get_ias_threshold(self, class_probs_dict, num_classes, alpha, old_thresholds=None, gamma=1.0):
        if old_thresholds is None:
            old_thresholds = np.ones(num_classes)

        class_threshold = np.ones(num_classes, dtype=np.float32)
        for c in range(num_classes):
            if class_probs_dict[c] is not None:
                class_threshold[c] = np.quantile(class_probs_dict[c], 1 - alpha * old_thresholds[c] ** gamma)
        return class_threshold

    def run(self):
        if len(os.listdir(self.pseudo_label_save_dir)) >= len(self.t_dataset):
            print('%% pseudo labels have existed')
        else:
            self.class_threshold = 0.9 * np.ones(self.cfg.dataset.num_classes)
            self.model.eval()
                
            with torch.no_grad():
                for data in tqdm.tqdm(self.t_loader, desc='Generating pseudo labels (policy: {})'.format(self.cfg.pseudo_policy.type), ncols=160):
                    imgs = data['images'].cuda()
                    result = self.model(imgs)
                    probs = F.softmax(result['logits'], dim=1)
                    probs_pred, lbls_pred = probs.max(dim=1)
                    probs_pred = probs_pred.cpu().numpy()
                    lbls_pred = lbls_pred.cpu().numpy()

                    # record the probability of pixels predicted as each category in the current batch
                    class_probs_dict = {c: [self.class_threshold[c]] for c in range(self.cfg.dataset.num_classes)}
                    print(len(class_probs_dict[0]))
                    for c in range(self.cfg.dataset.num_classes):
                        class_probs_dict[c].extend(probs_pred[lbls_pred == c].astype(np.float16))

                    # update IAS threshold
                    temp_class_threshold = self.get_ias_threshold(class_probs_dict, self.cfg.dataset.num_classes, self.cfg.pseudo_policy.ias.alpha,
                                                                  self.class_threshold, self.cfg.pseudo_policy.ias.gamma)
                    print(temp_class_threshold)
                    self.class_threshold = self.cfg.pseudo_policy.ias.beta * self.class_threshold + \
                                           (1 - self.cfg.pseudo_policy.ias.beta) * temp_class_threshold
                    self.class_threshold[self.class_threshold >= 1] = 0.999

                    select_plbl = self.select_and_save_confident_label(probs_pred, lbls_pred, data['image_paths'])

            self.save_data()
