# -*- coding: utf-8 -*
import os
from torch import nn
from apex.parallel import SyncBatchNorm, convert_syncbn_model
from apex import amp
import random
import numpy as np
import torch
import socket
from sseg.models.modules.schedulers import build_scheduler
import torch.distributed as dist
import logging
from tensorboardX import SummaryWriter
from utils.registry.registries import MODEL
import warnings
import tarfile


def seed_everything(seed=888):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        warnings.warn('{} has existed'.format(dir_path))


def is_port_used(port, host='127.0.0.1'):
    """Check port status"""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, int(port)))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()


def itv2time(iItv):
    h = int(iItv // 3600)
    sUp_h = iItv - 3600 * h
    m = int(sUp_h // 60)
    sUp_m = sUp_h - 60 * m
    s = int(sUp_m)
    return "{}h {:0>2d}min".format(h, m, s)


def freeze_bn(model):
    """Freeze BatchNorm layers"""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm, SyncBatchNorm)):
            for i in m.parameters():
                i.requires_grad = False


def load_model(cfg, resume_from=None, student_model=None):
    assert not (resume_from is not None and student_model is not None)

    model = MODEL[cfg.model.type](cfg)
    if student_model is not None:  # load model from student model (DDP)
        model.load_state_dict(student_model.module.state_dict().copy())
        print('%% load model from student model')
    elif resume_from is not None:  # load model from resume_from
        model_state_dict = model.state_dict()
        resume_from_state_dict = torch.load(resume_from, map_location='cpu')
        if 'module' in list(resume_from_state_dict.keys())[0]:  # model saved with DDP
            temp_state_dict = {k[7:]: v for k, v in resume_from_state_dict.items() if k[7:] in model_state_dict}
        else:
            temp_state_dict = {k: v for k, v in resume_from_state_dict.items() if k in model_state_dict}

        model_state_dict.update(temp_state_dict)
        model.load_state_dict(model_state_dict)
        print('%% load model from {}'.format(resume_from))
    else:  # load model from scratch
        warnings.warn('not load model')

    return model


def init_model(cfg, resume_from=None, student_model=None):
    """Init model or ema model (loading state_dict from student model)"""
    assert not (resume_from is not None and student_model is not None)

    if student_model is not None:  # load model from student model
        model = load_model(cfg, student_model=student_model)
    elif resume_from is not None:  # load model from resume_from
        model = load_model(cfg, resume_from=resume_from)
    else:  # load model from scratch
        model = load_model(cfg)

    if cfg.train.gpu_num > 1:
        model = convert_syncbn_model(model)  # this operation will activate batch norm layers
        print('%% convert BN to SyncBN')

    # freeze all bn layers in model
    if cfg.model.is_freeze_bn:
        freeze_bn(model)
        print('%% freeze all BN layers')

    return model


def update_ema_model(ema_model, model, gamma):
    # https://github.com/microsoft/ProDA/blob/9ba80c7dbbd23ba1a126e3f4003a72f27d121a1f/models/adaptation_modelv2.py#L261-L265
    for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
        param_k.data = param_k.data.clone() * gamma + \
                       param_q.data.clone() * (1 - gamma)
    for buffer_q, buffer_k in zip(model.buffers(), ema_model.buffers()):
        buffer_k.data = buffer_q.data.clone()

    return ema_model


def init_amp_setting(cfg, model, g_optimizer, d_optimizer):
    if d_optimizer is not None:
        model, optimizers = amp.initialize(model, [g_optimizer, d_optimizer], opt_level=cfg.train.apex_opt, verbosity=0)
        g_optimizer, d_optimizer = optimizers[0], optimizers[1]
    else:
        model, g_optimizer = amp.initialize(model, g_optimizer, opt_level=cfg.train.apex_opt, verbosity=0)
    return model, g_optimizer, d_optimizer


def init_optimizers(cfg, model):
    # generator optimizer
    g_param = model.seg_model.get_optimizer_params(cfg.train.lr)  # set weights of parameters with get_optimizer_params() in Segment Model Class

    if cfg.train.optimizer == 'SGD':
        g_optimizer = torch.optim.SGD(g_param, momentum=0.9, weight_decay=0.0005)
    elif cfg.train.optimizer == 'Adam':
        g_optimizer = torch.optim.Adam(g_param, betas=(0.9, 0.999), weight_decay=0.0005)
    elif cfg.train.optimizer == 'AdamW':
        g_optimizer = torch.optim.AdamW(g_param, betas=(0.9, 0.999), weight_decay=0.0005)
    else:
        raise ValueError('{} is not a valid optimizer'.format(cfg.train.optimizer))

    # discriminator optimizer
    if cfg.model.discriminator.is_enabled:
        d_optimizer = torch.optim.Adam(model.D.parameters(), lr=cfg.model.discriminator.lr, betas=(0.9, 0.999))  # optimizer of discriminator is fixed to Adam
    else:
        d_optimizer = None

    return g_optimizer, d_optimizer


def init_schedulers(cfg, g_optimizer, d_optimizer=None):
    schedulers = []
    schedulers.append(build_scheduler(cfg, g_optimizer))
    if d_optimizer is not None:
        schedulers.append(build_scheduler(cfg, d_optimizer))

    return schedulers


def all_reduce_average(tensor, world_size):
    """Compute the average tensor of all gpu"""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def init_logger_and_writer(log_path, tensorboard_dir_path):
    logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                        filename=log_path,
                        filemode='a',  # append mode
                        level=logging.INFO)
    logger = logging.getLogger("UDA.trainer")
    logger.addHandler(logging.StreamHandler())

    writer = SummaryWriter(tensorboard_dir_path, flush_secs=10)

    return logger, writer


def is_source_file(x):
    if x.isdir() or x.name.endswith(('.py', '.sh', '.yml', '.json', '.txt')) \
            and '.mim' not in x.name and 'jobs/' not in x.name:
        # print(x.name)
        return x
    else:
        return None


def gen_code_archive(out_dir, file='code.tar.gz'):
    archive = os.path.join(out_dir, file)
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive
