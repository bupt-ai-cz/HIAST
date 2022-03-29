# -*- coding: utf-8 -*
import torch
from torch.nn import functional as F
from utils import metrics, utils
from utils.result_recorder import ResultRecorder
import torch.distributed as dist
import numpy as np
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import os
from utils.registry.registries import DATASET
from torch.utils.data import DataLoader, DistributedSampler


class BaseTrainer:

    def __init__(self, cfg, gpu_index):
        self.cfg = cfg
        self.gpu_index = gpu_index

        self.assert_cfg()
        self.initialize()
        self.build_all_model()
        self.build_train_data_reader()
        self.build_val_data_reader()

    def assert_cfg(self):
        pass

    def initialize(self):
        # init random seed
        utils.seed_everything(self.cfg.train.random_seed)

        # init logger and writer
        if self.gpu_index == 0:
            self.logger, self.writer = utils.init_logger_and_writer(log_path=os.path.join(self.cfg.work_dir, 'train.log'),
                                                                    tensorboard_dir_path=os.path.join(self.cfg.work_dir, 'tensorboard'))
            self.checkpoint_dir_path = os.path.join(self.cfg.work_dir, 'checkpoints')
            if not os.path.exists(self.checkpoint_dir_path):
                utils.create_dir(self.checkpoint_dir_path)

        # init distributed environment
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:{}'.format(self.cfg.train.port),
                                world_size=self.cfg.train.gpu_num,
                                rank=self.gpu_index)
        torch.cuda.set_device(self.gpu_index)

    def build_all_model(self):
        """Build all used models with corresponding optimizers and schedulers"""
        print('%% initialize main model')
        self.model = utils.init_model(self.cfg, resume_from=self.cfg.train.resume_from).cuda()
        self.g_optimizer, self.d_optimizer = utils.init_optimizers(self.cfg, self.model)
        self.schedulers = utils.init_schedulers(self.cfg, self.g_optimizer, self.d_optimizer)
        self.model, self.g_optimizer, self.d_optimizer = utils.init_amp_setting(self.cfg, self.model, self.g_optimizer, self.d_optimizer)
        self.model = DDP(self.model, delay_allreduce=True)
        self.model_recorder = ResultRecorder(self.cfg, self.gpu_index, self.g_optimizer, self.d_optimizer, 'model',
                                             self.logger if self.gpu_index == 0 else None, self.writer if self.gpu_index == 0 else None)

    def build_train_data_reader(self):
        # source domain dataset for training
        if self.cfg.dataset.source.type is not None and self.cfg.dataset.source.json_path is not None and self.cfg.dataset.source.image_dir is not None:
            self.s_dataset = DATASET[self.cfg.dataset.source.type](self.cfg,
                                                                   self.cfg.dataset.source.json_path,
                                                                   self.cfg.dataset.source.image_dir,
                                                                   aug_type=self.cfg.dataset.source.aug_type,
                                                                   num_classes=self.cfg.dataset.num_classes)
            self.s_sampler = DistributedSampler(self.s_dataset, num_replicas=self.cfg.train.gpu_num, rank=self.gpu_index, shuffle=True)
            self.s_loader = DataLoader(self.s_dataset, self.cfg.train.batch_size, sampler=self.s_sampler, num_workers=self.cfg.dataset.num_workers,
                                       pin_memory=True, drop_last=True)
            self.s_iter = iter(self.s_loader)

        # target domain dataset for training
        if self.cfg.dataset.target.type is not None and self.cfg.dataset.target.json_path and self.cfg.dataset.target.image_dir is not None:
            self.t_dataset = DATASET[self.cfg.dataset.target.type](self.cfg,
                                                                   self.cfg.dataset.target.json_path,
                                                                   self.cfg.dataset.target.image_dir,
                                                                   pseudo_dir=self.cfg.dataset.target.pseudo_dir,
                                                                   aug_type=self.cfg.dataset.target.aug_type,
                                                                   num_classes=self.cfg.dataset.num_classes)
            self.t_sampler = DistributedSampler(self.t_dataset, num_replicas=self.cfg.train.gpu_num, rank=self.gpu_index, shuffle=True)
            self.t_loader = DataLoader(self.t_dataset, self.cfg.train.batch_size, sampler=self.t_sampler, num_workers=self.cfg.dataset.num_workers,
                                       pin_memory=True, drop_last=True)
            self.t_iter = iter(self.t_loader)

    def build_val_data_reader(self):
        # target domain dataset for validation
        v_dataset = DATASET[self.cfg.dataset.val.type](self.cfg,
                                                       self.cfg.dataset.val.json_path,
                                                       self.cfg.dataset.val.image_dir,
                                                       num_classes=self.cfg.dataset.num_classes)
        v_sampler = DistributedSampler(v_dataset, num_replicas=self.cfg.train.gpu_num, rank=self.gpu_index)
        self.v_loader = DataLoader(v_dataset, self.cfg.train.batch_size, sampler=v_sampler, num_workers=self.cfg.dataset.num_workers, pin_memory=True)

    def run(self):
        if self.gpu_index == 0:
            self.logger.info('=' * 160)
            self.logger.info(self.cfg)
            self.logger.info('=' * 160)

        self.model_recorder.reset_time_and_losses()

        for current_iter in range(1, self.cfg.train.total_iter + 1):
            # get training loss
            temp_losses = self.train(current_iter)

            # loss backward and update model
            self.update_model(self.g_optimizer, self.d_optimizer, temp_losses)

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

        self.model_recorder.report_end_info()

    def update_model(self, g_optimizer, d_optimizer, losses):
        # backward generator loss
        g_loss = sum(torch.mean(value) for name, value in losses.items() if 'D_' not in name)
        g_optimizer.zero_grad()
        with amp.scale_loss(sum(g_loss) if g_loss.size() else g_loss, g_optimizer) as scaled_loss:
            scaled_loss.backward()
        g_optimizer.step()

        # backward discriminator loss
        if 'D_loss' in losses:
            d_loss = losses['D_loss']
            d_optimizer.zero_grad()
            with amp.scale_loss(sum(d_loss) if d_loss.size() else d_loss, d_optimizer) as scaled_loss:
                scaled_loss.backward()
            d_optimizer.step()

    def train(self, current_iter):
        raise NotImplementedError

    def validate(self, model, recorder, current_iter, is_ema=False):
        # validate in multiple gpu
        iou, miou = self.get_validate_result(model)

        # report validate result
        recorder.record_and_report_metrics(miou, iou, current_iter)

        # save model
        if self.gpu_index == 0:
            if not is_ema:
                self.save_checkpoint(model, current_iter, recorder.model_name, (miou == recorder.best_miou))
            else:  # ema model only saves last model
                torch.save(model.state_dict(), os.path.join(self.checkpoint_dir_path, '{}_last.pth'.format(recorder.model_name)))

    def get_validate_result(self, model):
        model.eval()
        intersection_sum = 0
        union_sum = 0
        with torch.no_grad():
            for data in self.v_loader:
                img = data['images'].cuda()
                lbl = data['labels'].cuda()

                tmp_img = F.interpolate(img, self.cfg.dataset.val.resize_size, mode='bilinear', align_corners=True)
                logits = model(tmp_img)['logits']
                logits = F.interpolate(logits, lbl.size()[1:], mode='bilinear', align_corners=True)

                lbl_pred = logits.argmax(dim=1)

                intersection, union = metrics.intersectionAndUnionGPU(lbl_pred, lbl, self.cfg.dataset.num_classes)
                intersection_sum += intersection
                union_sum += union

        # synchronize in multiple gpu (ReduceOp.SUM)
        dist.all_reduce(intersection_sum)
        dist.all_reduce(union_sum)

        iou = intersection_sum.cpu().numpy() / (union_sum.cpu().numpy() + 1e-10)
        miou = np.mean(iou)

        return iou, miou

    def save_checkpoint(self, model, iter, model_name, is_best=False):
        checkpoint = model.module.state_dict()
        if self.cfg.train.is_save_all:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir_path, '{}_iter_{}.pth'.format(model_name, iter)))
        torch.save(checkpoint, os.path.join(self.checkpoint_dir_path, '{}_last.pth'.format(model_name)))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir_path, '{}_best.pth'.format(model_name)))
        if iter >= self.cfg.train.total_iter // 2:
            mid_pth_path = os.path.join(self.checkpoint_dir_path, '{}_mid.pth'.format(model_name))
            if not os.path.exists(mid_pth_path):
                torch.save(checkpoint, mid_pth_path)
