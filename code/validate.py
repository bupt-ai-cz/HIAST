# -*- coding: utf-8 -*
from utils.registry import register
import argparse
from utils.default_config import cfg
from workflows.validator import Validator
from utils.registry.registries import SEG_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description='UDA-Experiment Validation')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--setting_file')
    parser.add_argument('--resume_from')
    parser.add_argument('--color_mask_dir_path')
    parser.add_argument('--seg_model', choices=list(SEG_MODEL.keys()))
    parser.add_argument('--transform_style', choices=['advent', 'iast'])
    args = parser.parse_args()
    return args


def update_cfg(cfg, args):
    """update cfg, priority: config_file < setting_file < args"""

    # update cfg with according files
    cfg.merge_from_file(args.config_file)
    if args.setting_file:
        cfg.merge_from_file(args.setting_file)

    if args.resume_from:
        cfg.validate.resume_from = args.resume_from

    if args.color_mask_dir_path:
        cfg.validate.color_mask_dir_path = args.color_mask_dir_path

    if args.seg_model:
        cfg.model.seg_model.type = args.seg_model

    if args.transform_style:
        cfg.dataset.transform_style = args.transform_style

    cfg.freeze()

    return cfg


if __name__ == '__main__':
    args = parse_args()
    cfg = update_cfg(cfg, args)

    validator = Validator(cfg)
    validator.run()
