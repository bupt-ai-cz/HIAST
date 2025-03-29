# -*- coding: utf-8 -*
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import functional as F
from workflows.trainer.base_trainer import BaseTrainer
from utils import utils
from utils.result_recorder import ResultRecorder
from apex import amp
from sseg.datasets.preprocessor import CopyPaste
import numpy as np
import torch.distributed as dist
import os
from utils.registry.registries import TRAINER, DATASET


@TRAINER.register('ConsistencySelfTrainingTrainer')
class ConsistencySelfTrainingTrainer(BaseTrainer):

    def assert_cfg(self):
        assert self.cfg.dataset.target.pseudo_dir is not None, 'directory of pseudo labels should be given for self training'
        assert self.cfg.cst_training.is_enabled, 'consistency training should be enabled'
        # assert self.cfg.cst_training.cst_loss.weight > 0, 'consistency loss should be larger than 0'
        assert len(self.cfg.dataset.target.aug_type) == 1 or len(self.cfg.dataset.target.aug_type) == 2, \
            'target domain dataset should have 1 or 2 augmentations for consistency training'
        assert self.cfg.preprocessor.type == 'CopyPaste'

    def build_train_data_reader(self):        
        # init copy paste
        init_class_value_path = os.path.join(self.cfg.dataset.target.pseudo_dir, '..', 'class_mean_probabilities.npy')
        self.class_value = np.load(init_class_value_path)

        self.t_dataset = DATASET[self.cfg.dataset.target.type](self.cfg,
                                                               self.cfg.dataset.target.json_path,
                                                               self.cfg.dataset.target.image_dir,
                                                               pseudo_dir=self.cfg.dataset.target.pseudo_dir,
                                                               aug_type=self.cfg.dataset.target.aug_type,
                                                               num_classes=self.cfg.dataset.num_classes)
        self.preprocessor = CopyPaste(self.cfg, self.t_dataset, self.class_value)
        self.t_dataset.set_preprocessor(self.preprocessor)

        self.t_sampler = DistributedSampler(self.t_dataset, num_replicas=self.cfg.train.gpu_num, rank=self.gpu_index, shuffle=True)
        self.t_loader = DataLoader(self.t_dataset, self.cfg.train.batch_size, sampler=self.t_sampler, num_workers=self.cfg.dataset.num_workers,
                                   pin_memory=True, drop_last=True)
        self.t_iter = iter(self.t_loader)

    def build_all_model(self):
        super(ConsistencySelfTrainingTrainer, self).build_all_model()
        self.ema_model = utils.init_model(self.cfg, student_model=self.model).cuda()
        self.ema_model = amp.initialize(self.ema_model, verbosity=0)
        self.ema_model_recorder = ResultRecorder(self.cfg, self.gpu_index, None, None, 'ema_model',
                                                 self.logger if self.gpu_index == 0 else None, self.writer if self.gpu_index == 0 else None)

    def run(self):
        if self.gpu_index == 0:
            self.logger.info('=' * 160)
            self.logger.info(self.cfg)
            self.logger.info('=' * 160)

        self.model_recorder.reset_time_and_losses()
        self.ema_model_recorder.reset_time_and_losses()

        for current_iter in range(1, self.cfg.train.total_iter + 1):
            # get training loss
            temp_losses = self.train()

            # loss backward and update model
            self.update_model(self.g_optimizer, self.d_optimizer, temp_losses)
            # update ema model
            if current_iter % self.cfg.cst_training.ema_model.iter_update == 0:
                # https://github.com/microsoft/ProDA/blob/9ba80c7dbbd23ba1a126e3f4003a72f27d121a1f/models/adaptation_modelv2.py#L261-L265
                self.ema_model = utils.update_ema_model(self.ema_model, self.model, self.cfg.cst_training.ema_model.gamma)

            # update scheduler
            for s in self.schedulers:
                s.step()

            # record loss
            self.model_recorder.record_losses(temp_losses)

            # report loss and lr
            if current_iter % self.cfg.train.iter_report == 0:
                self.model_recorder.report_losses(current_iter)

            # validate
            if current_iter % self.cfg.train.iter_val == 0:
                self.validate(self.model, self.model_recorder, current_iter)
                self.validate(self.ema_model, self.ema_model_recorder, current_iter, True)

        self.model_recorder.report_end_info()
        self.ema_model_recorder.report_end_info()

    def train(self):
        try:
            t = next(self.t_iter)
        except StopIteration:
            if self.t_sampler.shuffle:  # if shuffle, reset epoch as random seed
                self.t_sampler.set_epoch(self.t_sampler.epoch + 1)
            self.t_iter = iter(self.t_loader)
            t = next(self.t_iter)
        t_img = t['images']
        t_plbl = t['labels']

        if isinstance(t_img, (list, tuple)):
            assert len(t_img) == 2 and torch.equal(t_plbl[0], t_plbl[1])
            t_weak_img = t_img[0].cuda()
            t_strong_img = t_img[1].cuda()
            t_plbl = t_plbl[0].cuda()
        else:
            t_weak_img = t_img.cuda()
            t_strong_img = t_weak_img
            t_plbl = t_plbl.cuda()

        self.ema_model.eval()
        with torch.no_grad():
            t_weak_logits = self.ema_model(t_weak_img)['logits']
            if self.cfg.cst_training.cst_loss.type == 'CE':  # label for CrossEntropy should be hard label
                t_cst_lbl = t_weak_logits.argmax(dim=1)
            else:  # soft label
                t_cst_lbl = F.softmax(t_weak_logits, dim=1)

        self.model.train()
        t_strong_logits = self.model(t_strong_img)['logits']

        losses = self.model.module.compute_loss(t_strong_logits, t_plbl, t_cst_lbl)

        return losses
