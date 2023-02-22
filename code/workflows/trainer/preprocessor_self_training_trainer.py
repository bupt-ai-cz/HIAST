# -*- coding: utf-8 -*
from torch.utils.data import DataLoader, DistributedSampler
from workflows.trainer.self_training_trainer import BaseTrainer
from utils.registry.registries import TRAINER, DATASET
from sseg.datasets.preprocessor import CopyPaste
import numpy as np
import os
import torch
from torch.nn import functional as F
import torch.distributed as dist


@TRAINER.register('PreprocessorSelfTrainingTrainer')
class PreprocessorSelfTrainingTrainer(BaseTrainer):
    """Self Training with preprocessor"""

    def assert_cfg(self):
        assert self.cfg.dataset.target.pseudo_dir is not None, 'directory of pseudo labels should be given for self training'
        assert self.cfg.train.resume_from is not None, 'self-training should resume_from one state_dict'
        assert self.cfg.preprocessor.type is not None, 'preprocessor should be not None'

    def init_preprocessor(self):
        if self.cfg.preprocessor.type is None:
            self.preprocessor = None
            print('%% not use preprocessor')
        elif self.cfg.preprocessor.type == 'CopyPaste':
            self.init_copy_paste()
        else:
            raise NotImplementedError

    def init_copy_paste(self):
        # init copy and paste，必须在生成iter之前设置，不然没有效果
        init_class_value_path = os.path.join(self.cfg.dataset.target.pseudo_dir, '..', 'class_mean_probabilities.npy')
        self.class_value = np.load(init_class_value_path)

        if self.cfg.preprocessor.copy_paste.copy_from == 'target':  # cp from target
            dataset_copy_from = self.t_dataset
        else:  # cp from source
            dataset_copy_from = DATASET[self.cfg.dataset.source.type](self.cfg,
                                                                      self.cfg.dataset.source.json_path,
                                                                      self.cfg.dataset.source.image_dir,
                                                                      aug_type=self.cfg.dataset.source.aug_type,
                                                                      num_classes=self.cfg.dataset.num_classes)

        self.preprocessor = CopyPaste(self.cfg, dataset_copy_from, self.class_value)
        self.t_dataset.set_preprocessor(self.preprocessor)

    def build_train_data_reader(self):
        self.t_dataset = DATASET[self.cfg.dataset.target.type](self.cfg,
                                                               self.cfg.dataset.target.json_path,
                                                               self.cfg.dataset.target.image_dir,
                                                               pseudo_dir=self.cfg.dataset.target.pseudo_dir,
                                                               aug_type=self.cfg.dataset.target.aug_type,
                                                               num_classes=self.cfg.dataset.num_classes)

        self.init_preprocessor()

        self.t_sampler = DistributedSampler(self.t_dataset, num_replicas=self.cfg.train.gpu_num, rank=self.gpu_index, shuffle=True)
        self.t_loader = DataLoader(self.t_dataset, self.cfg.train.batch_size, sampler=self.t_sampler, num_workers=self.cfg.dataset.num_workers,
                                   pin_memory=True, drop_last=True)
        self.t_iter = iter(self.t_loader)

    def train(self, current_iter):
        try:
            t = next(self.t_iter)
        except StopIteration:
            if self.t_sampler.shuffle:  # if shuffle, reset epoch as random seed
                self.t_sampler.set_epoch(self.t_sampler.epoch + 1)
            self.t_iter = iter(self.t_loader)
            t = next(self.t_iter)
        t_img = t['images'].cuda()
        t_plbl = t['labels'].cuda()

        self.model.train()
        t_logits = self.model(t_img)['logits']
        losses = self.model.module.compute_loss(t_logits, t_plbl)

        return losses
