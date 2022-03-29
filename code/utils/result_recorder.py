# -*- coding: utf-8 -*
import numpy as np
import time
from utils import utils


class ResultRecorder():
    """Record losses and validation metrics of an model during training"""

    def __init__(self, cfg, gpu_index, g_optimizer=None, d_optimizer=None, model_name='model', logger=None, writer=None):
        self.cfg = cfg
        self.gpu_index = gpu_index
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.model_name = model_name
        self.logger = logger
        self.writer = writer

        self.is_synthia = 'SYNTHIA' in self.cfg.dataset.source.type

        self.best_miou = 0
        self.best_iter = 0
        if self.is_synthia:
            self.miou_13_when_16_best = 0

        self.losses_recorded = {'total_loss': 0}
        self.start_time = time.time()

        if self.gpu_index == 0:
            assert self.logger is not None and self.writer is not None

    def record_and_report_metrics(self, miou, iou, current_iter):
        if self.gpu_index == 0:
            if self.is_synthia:
                miou *= 19 / 16
                iu_13 = iou.copy()
                iu_13[3:6] = 0
                miou_13 = np.mean(iu_13) * 19 / 13

            if miou > self.best_miou:
                self.best_miou = miou
                self.best_iter = current_iter
                if self.is_synthia:
                    self.miou_13_when_16_best = miou_13

            # report miou and iou
            if self.is_synthia:
                self.logger.info('{}, iter: {}, miou_16: {:.4f}({:.4f}), miou_13: {:.4f}, iou: {}'.format(
                    self.model_name, current_iter, miou, self.best_miou, miou_13, {i: round(value, 3) for i, value in enumerate(iou)}))
                self.writer.add_scalar('val_{}/miou_16'.format(self.model_name), miou, current_iter)
                self.writer.add_scalar('val_{}/miou_13'.format(self.model_name), miou_13, current_iter)
            else:
                self.logger.info('{}, iter: {}, miou: {:.4f}({:.4f}), iou: {}'.format(
                    self.model_name, current_iter, miou, self.best_miou, {i: round(value, 3) for i, value in enumerate(iou)}))
                self.writer.add_scalar('val_{}/miou'.format(self.model_name), miou, current_iter)
            self.writer.add_scalars('val_{}/iou'.format(self.model_name), {str(i): value for i, value in enumerate(iou)}, current_iter)

    def record_losses(self, loss_dict):
        """Record losses from multiple gpu"""
        for name, value in loss_dict.items():
            average_value = utils.all_reduce_average(value.detach(), self.cfg.train.gpu_num).item()  # compute average loss in multiple gpu
            if name not in self.losses_recorded:
                self.losses_recorded[name] = 0
            self.losses_recorded[name] += average_value

            if 'D_' not in name:  # compute total generator loss
                self.losses_recorded['total_loss'] += average_value

    def report_losses(self, current_iter):
        if self.gpu_index == 0:
            # compute average loss in multiple iterations
            for name, value in self.losses_recorded.items():
                self.losses_recorded[name] = round(value / self.cfg.train.iter_report, 6)

            # estimate left training time
            time_spent = time.time() - self.start_time
            speed = time_spent / self.cfg.train.iter_report  # s/iter
            eta = utils.itv2time((self.cfg.train.total_iter - current_iter) * speed)  # left time of training

            if self.d_optimizer is None:
                self.logger.info('{}, eta: {}, iter: [{}/{}], speed: {:.3f} s/iter, g_lr: {:.2e}, loss: {}'.format(
                    self.model_name, eta, current_iter, self.cfg.train.total_iter, speed, self.g_optimizer.param_groups[-1]['lr'], self.losses_recorded))
            else:
                self.logger.info('{}, eta: {}, iter: [{}/{}], speed: {:.3f} s/iter, g_lr: {:.2e}, d_lr: {:.2e}, loss: {}'.format(
                    self.model_name, eta, current_iter, self.cfg.train.total_iter, speed, self.g_optimizer.param_groups[-1]['lr'],
                    self.d_optimizer.param_groups[-1]['lr'], self.losses_recorded))
            self.writer.add_scalars('train_{}/loss'.format(self.model_name), self.losses_recorded, current_iter)

            g_lr_recorded = {str(i): p_group['lr'] for i, p_group in enumerate(self.g_optimizer.param_groups)}
            self.writer.add_scalars('train_{}/g_lr'.format(self.model_name), g_lr_recorded, current_iter)
            if self.d_optimizer is not None:
                d_lr_recorded = {str(i): p_group['lr'] for i, p_group in enumerate(self.d_optimizer.param_groups)}
                self.writer.add_scalars('train_{}/d_lr'.format(self.model_name), d_lr_recorded, current_iter)

            self.reset_time_and_losses()

    def reset_time_and_losses(self):
        # reset recorder
        self.start_time = time.time()
        self.losses_recorded = {'total_loss': 0}  # record losses between reported iterations

    def report_end_info(self):
        if self.gpu_index == 0:
            if self.is_synthia:
                self.logger.info('End, {}, best_miou_16: {:.4f}, miou_13: {:.4f}, best_iter: {}'.format(
                    self.model_name, self.best_miou, self.miou_13_when_16_best, self.best_iter))
            else:
                self.logger.info('End, {}, best_miou: {:.4f}, best_iter: {}'.format(self.model_name, self.best_miou, self.best_iter))
            self.writer.close()
