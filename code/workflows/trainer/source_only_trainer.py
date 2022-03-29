# -*- coding: utf-8 -*
from workflows.trainer.base_trainer import BaseTrainer
from utils.registry.registries import TRAINER


@TRAINER.register('SourceOnlyTrainer')
class SourceOnlyTrainer(BaseTrainer):

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

        self.model.train()
        losses = self.model(s_img, s_lbl)

        return losses
