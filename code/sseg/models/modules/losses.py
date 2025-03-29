# -*- coding: utf-8 -*
import torch.nn as nn
import torch
from torch.nn import functional as F
import warnings
from utils.registry.registries import LOSS


@LOSS.register('MSE')
def mse(logits, labels, weights=None, ignore_index=255, refer_labels=None, region='ignore'):
    if weights is not None:
        warnings.warn('Weights is not available for MSE Loss')
    return compute_loss(nn.MSELoss(), nn.MSELoss(reduction='none'), logits, labels, ignore_index, refer_labels, region)


@LOSS.register('KLDIV')
def kl_div(input_logits, target_logits, weights=None, ignore_index=255, refer_labels=None, region='confident'):
    if weights is not None:
        warnings.warn('Weights is not available for KLDiv Loss')
    # Pytorch的KLDivLoss，对于input要求是log_softmax()的形式，对于target要求是softmax()的形式
    input = F.log_softmax(input_logits, dim=1)
    target = F.softmax(target_logits, dim=1)
    return compute_loss(nn.KLDivLoss(), nn.KLDivLoss(reduction='none'), input, target, ignore_index, refer_labels, region)


@LOSS.register('BCEWithLogits')
def bce_with_logits(logits, labels):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, labels)


@LOSS.register('CE')
def ce(logits, labels, weights=None, ignore_index=255, refer_labels=None, region='confident'):
    # 只有CrossEntropy使用Hard Label，也只有CrossEntropy才能直接使用ignore_index进行指定像素点损失的计算，其他的都需要ignore_index、refer_labels、region来辅助
    return compute_loss(nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights), nn.CrossEntropyLoss(weight=weights, reduction='none'), logits, labels,
                        ignore_index, refer_labels, region)


@LOSS.register('SoftCE')
def soft_ce(logits, labels, weights=None, ignore_index=255, refer_labels=None, region='confident'):
    return compute_loss(SoftCELoss(weights=weights), SoftCELoss(weights=weights, reduction='none'), logits, labels, ignore_index, refer_labels, region)


class SoftCELoss(torch.nn.Module):
    def __init__(self, weights=None, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape == target.shape  # [B, C, H, W]
        assert target.min().item() >= 0 and target.max().item() <= 1, 'target should be softmax and between 0 and 1'
        nll = - F.log_softmax(input, dim=1)

        if self.weights is not None:
            assert len(self.weights) == target.shape[1]
            for c in range(target.shape[1]):
                target[:, c, :, :] *= self.weights[c]

        if self.reduction == 'none':
            return nll * target  # [B, C, H, W]
        elif self.reduction == 'sum':
            return (nll * target).sum()
        elif self.reduction == 'mean':
            return (nll * target).sum() / target.numel()


def compute_loss(loss_fun_mean, loss_fun_none, input, target, ignore_index, refer_labels, region):
    if refer_labels is None:  # 在所有区域计算损失，获得对于CrossEntropy自动根据ignore_index计算损失
        return loss_fun_mean(input, target)  # reduction='mean'
    else:  # 根据refer_labels和ignore_index来选择指定的区域计算损失，refer_labels不一定等于target
        return compute_loss_by_selected_pixel(loss_fun_none(input, target), refer_labels, ignore_index, region)  # reduction='none'


def compute_loss_by_selected_pixel(loss_tensor, refer_labels, ignore_index, region):
    """根据另一个标签矩阵refer_labels，来选择指定区域参与损失的计算"""
    if region == 'ignored':
        mask_ = (refer_labels == ignore_index)  # [B, H, W]
    elif region == 'confident':
        mask_ = (refer_labels != ignore_index)  # [B, H, W]
    elif region == 'all':
        mask_ = torch.ones_like(refer_labels, dtype=torch.bool)  # [B, H, W]
    else:
        raise ValueError('{} is not a valid region'.format(region))

    mask_ = mask_.unsqueeze(dim=1)  # [B, 1, H, W]
    loss_tensor = loss_tensor * mask_  # 将不需要计算的地方置为0

    return loss_tensor.sum() / (loss_tensor != 0).sum()
