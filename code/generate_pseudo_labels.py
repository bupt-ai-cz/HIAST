# -*- coding: utf-8 -*
from utils.registry import register
import argparse
from utils.default_config import cfg
from utils.registry.registries import PSEUDO_POLICY, SEG_MODEL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--setting_file')
    parser.add_argument('--pseudo_resume_from')
    parser.add_argument('--pseudo_save_dir')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seg_model', choices=list(SEG_MODEL.keys()))
    args = parser.parse_args()

    return args


def update_cfg(cfg, args):
    cfg.merge_from_file(args.config_file)
    if args.setting_file:
        cfg.merge_from_file(args.setting_file)

    if args.pseudo_resume_from:
        cfg.pseudo_policy.resume_from = args.pseudo_resume_from

    if args.batch_size:
        cfg.pseudo_policy.batch_size = cfg.batch_size

    if args.pseudo_save_dir:
        cfg.pseudo_policy.save_dir = args.pseudo_save_dir

    if args.seg_model:
        cfg.model.seg_model.type = args.seg_model

    cfg.freeze()

    return cfg


if __name__ == '__main__':
    args = parse_args()
    cfg = update_cfg(cfg, args)

    pseudo_generator = PSEUDO_POLICY[cfg.pseudo_policy.type](cfg)
    pseudo_generator.run()
