# -*- coding: utf-8 -*
from workflows.trainer.base_trainer import BaseTrainer
from utils.registry.registries import TRAINER


@TRAINER.register('AdversarialWarmupTrainer')
class AdversarialWarmupTrainer(BaseTrainer):

    def assert_cfg(self):
        assert self.cfg.model.discriminator.is_enabled, 'discriminator should be enabled for adversarial warmup training'
        assert self.cfg.train.resume_from is not None, 'adversarial warmup training should resume_from one state_dict'

    def train(self, current_iter):
        try:
            s = next(self.s_iter)
        except StopIteration:
            if self.s_sampler.shuffle:  # if shuffle, reset epoch as random seed
                self.s_sampler.set_epoch(self.s_sampler.epoch + 1)
            self.s_iter = iter(self.s_loader)
            s = next(self.s_iter)
        s_img = s['images'].cuda()
        s_lbl = s['labels'].cuda()

        try:
            t = next(self.t_iter)
        except StopIteration:
            if self.t_sampler.shuffle:  # if shuffle, reset epoch as random seed
                self.t_sampler.set_epoch(self.t_sampler.epoch + 1)
            self.t_iter = iter(self.t_loader)
            t = next(self.t_iter)
        t_img = t['images'].cuda()

        self.model.train()
        losses = self.model(s_img, t_img, s_lbl)

        return losses
