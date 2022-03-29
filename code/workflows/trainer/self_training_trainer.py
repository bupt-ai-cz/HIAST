# -*- coding: utf-8 -*
from workflows.trainer.base_trainer import BaseTrainer
from utils.registry.registries import TRAINER


@TRAINER.register('SelfTrainingTrainer')
class SelfTrainingTrainer(BaseTrainer):

    def assert_cfg(self):
        assert self.cfg.dataset.target.pseudo_dir is not None, 'directory of pseudo labels should be given for self training'
        assert self.cfg.train.resume_from is not None, 'self-training should resume_from one state_dict'

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
