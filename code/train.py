# -*- coding: utf-8 -*
from utils.registry import register
import torch
import torch.multiprocessing as mp
import argparse
import os
from utils.default_config import cfg
from utils import utils
from utils.registry.registries import TRAINER, SEG_MODEL
from utils.utils import gen_code_archive


def main_worker(proc_idx, cfg):
    trainer = TRAINER[cfg.trainer](cfg, proc_idx)
    trainer.run()


def parse_args():
    parser = argparse.ArgumentParser(description='UDA-Experiment Training')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--setting_file')
    parser.add_argument('--resume_from')
    parser.add_argument('--pseudo_save_dir')
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--seg_model', choices=list(SEG_MODEL.keys()))
    args = parser.parse_args()

    return args


def update_cfg(cfg, args):
    """update cfg, priority: config_file < setting_file < args"""

    # update cfg with according files
    cfg.merge_from_file(args.config_file)
    if args.setting_file:
        cfg.merge_from_file(args.setting_file)

    if args.work_dir:
        cfg.work_dir = args.work_dir

    if args.resume_from:
        cfg.train.resume_from = args.resume_from

    if args.pseudo_save_dir:
        cfg.dataset.target.pseudo_dir = args.pseudo_save_dir

    if args.seg_model:
        cfg.model.seg_model.type = args.seg_model

    # update batch size for training on multiple gpu
    cfg.train.gpu_num = torch.cuda.device_count()
    cfg.train.batch_size //= cfg.train.gpu_num
    assert cfg.train.batch_size > 0
    print('%% total gpu: {}, batch size (each gpu): {}'.format(cfg.train.gpu_num, cfg.train.batch_size))

    # search an available port
    while utils.is_port_used(cfg.train.port):
        cfg.train.port += 1
        
    cfg.freeze()

    return cfg


if __name__ == '__main__':
    args = parse_args()
    cfg = update_cfg(cfg, args)

    # create work directory
    utils.create_dir(cfg.work_dir)

    # backup config file
    cfg_backup_path = os.path.join(cfg.work_dir, os.path.basename(args.config_file))
    with open(cfg_backup_path, 'w') as f:
        f.write(cfg.dump())

    # backup code
    gen_code_archive(cfg.work_dir)

    # run training pipeline with distributed setting
    mp.spawn(main_worker, nprocs=cfg.train.gpu_num, args=(cfg,))
