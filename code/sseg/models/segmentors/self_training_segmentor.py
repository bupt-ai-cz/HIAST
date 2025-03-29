# -*- coding: utf-8 -*
from torch import nn
from torch.nn import functional as F
import torch
from sseg.models.modules.seg_models import build_seg_model
from utils.registry.registries import LOSS, MODEL


@MODEL.register('SelfTrainingSegmentor')
class SelfTrainingSegmentor(nn.Module):

    def __init__(self, cfg):
        super(SelfTrainingSegmentor, self).__init__()
        self.cfg = cfg
        self.seg_model = build_seg_model(cfg)

        self.seg_loss_fun = LOSS[cfg.model.predictor.seg_loss.type]
        self.kld_loss_fun = _kld
        self.ent_loss_fun = _entropy
        if cfg.cst_training.is_enabled:
            self.cst_loss_fun = LOSS[cfg.cst_training.cst_loss.type]
        if cfg.mut_training.is_enabled:  # mutual learning的损失暂时使用一致性损失，因为本质上也是一致性训练
            self.mut_loss_fun = self.cst_loss_fun

    def forward(self, t_img):
        t_logits, backbone = self.seg_model(t_img)
        t_logits = F.interpolate(t_logits, size=t_img.shape[2:], mode='bilinear', align_corners=True)
        return {'logits': t_logits, 'backbone': backbone}

    def compute_loss(self, t_logits, t_plbl, t_cst_lbl=None, s_logits=None, s_lbl=None):
        losses = {}
        if s_lbl is not None:
        # source seg loss
            losses['source_seg_loss'] = self.seg_loss_fun(s_logits, s_lbl)

        # target seg loss with pseudo label
        losses['target_seg_loss'] = self.cfg.model.predictor.seg_loss.target_pseudo_weight * self.seg_loss_fun(t_logits, t_plbl)

        reg_weight_confident, reg_weight_ignored = build_region_weight(t_logits, t_plbl)
        # kl div loss (confident region)
        if self.cfg.model.predictor.kld_loss.weight > 0:
            losses['kld_confident_loss'] = self.cfg.model.predictor.kld_loss.weight * self.kld_loss_fun(t_logits, reg_weight_confident)

        # entropy loss (ignored region)
        if self.cfg.model.predictor.ent_loss.weight > 0:
            losses['ent_ignored_loss'] = self.cfg.model.predictor.ent_loss.weight * self.ent_loss_fun(t_logits, reg_weight_ignored)

        # consistency loss
        if t_cst_lbl is not None and self.cfg.cst_training.is_enabled and self.cfg.cst_training.cst_loss.weight > 0:
            losses['cst_loss'] = self.cfg.cst_training.cst_loss.weight * \
                                 self.cst_loss_fun(t_logits, t_cst_lbl, refer_labels=t_plbl, region=self.cfg.cst_training.cst_loss.region)

        return losses

    # def compute_mutual_loss(self, t_logits, t_plbl, t_cst_lbl):
    #     """compute mutual learning loss"""
    #     losses = {}
    #     if self.cfg.mut_training.is_enabled and self.cfg.mut_training.mut_loss.weight > 0:
    #         losses['mut_loss'] = self.cfg.mut_training.mut_loss.weight * \
    #                              self.mut_loss_fun(t_logits, t_cst_lbl, refer_labels=t_plbl, region=self.cfg.mut_training.mut_loss.region)
    #     return losses

    # def compute_directional_consistency_loss(self, t_logits_0, t_logits_1, copy_paste_masks, weight=1.0):
    #     """根据2张图像计算带有方向性的Consistency Loss"""
    #     cst_loss_fun = LOSS['SoftCE']
    #     dcst_loss = torch.tensor(0).float().cuda()  # directional consistency loss
    #     batch_size = t_logits_0.shape[0]

    #     if weight == 0:
    #         return {'dcst_loss': dcst_loss / batch_size}

    #     # region-level direction
    #     # for logit_0, logit_1, cp_mask in zip(t_logits_0, t_logits_1, copy_paste_masks):
    #     #     # logit_0, logit_1: [C, H, W]
    #     #     # cp_mask: [H, W]
    #     #     for c in torch.unique(cp_mask):  # 针对每一个batch内的每个类别的数据，计算directional consistency loss
    #     #         if c.item() == 255:
    #     #             continue
    #     #         # 计算每张图像在粘贴的类别c区域内的平均预测概率
    #     #         probs_pred_0, lbls_pred_0 = F.softmax(logit_0, dim=0).max(dim=0)
    #     #         probs_pred_1, lbls_pred_1 = F.softmax(logit_1, dim=0).max(dim=0)
    #     #
    #     #         sel_mask = (cp_mask == c)
    #     #         avg_probs_0 = torch.mean(probs_pred_0[sel_mask * (lbls_pred_0 == c)])  # NOTE: 在粘贴过来且被预测正确的区域内计算平均的预测概率
    #     #         avg_probs_1 = torch.mean(probs_pred_1[sel_mask * (lbls_pred_1 == c)])
    #     #         if avg_probs_0 < avg_probs_1:  # 差的向好的对齐
    #     #             source_logit = logit_0.unsqueeze(dim=0)
    #     #             target_prob = F.softmax(logit_1.unsqueeze(dim=0), dim=1)
    #     #         else:
    #     #             source_logit = logit_1.unsqueeze(dim=0)
    #     #             target_prob = F.softmax(logit_0.unsqueeze(dim=0), dim=1)
    #     #
    #     #         new_cst_mask = torch.ones_like(cp_mask).long().cuda() * 255
    #     #         new_cst_mask[cp_mask == c] = c
    #     #         new_cst_mask = new_cst_mask.unsqueeze(dim=0)  # [B, C, H, W]
    #     #
    #     #         dcst_loss += weight * cst_loss_fun(source_logit, target_prob, refer_labels=new_cst_mask, region='confident')

    #     # pixel-level direction
    #     # logits: [B, C, H, W]
    #     # copy_paste_masks: [B, H, W]
    #     # t_logits_0 -> t_logits_1
    #     t_softmax_0 = F.softmax(t_logits_0, dim=1)  # [B, C, H, W]
    #     t_softmax_1 = F.softmax(t_logits_1, dim=1)  # [B, C, H, W]
    #     t_max_probs_0 = t_softmax_0.max(dim=1)[0]  # [B, H, W]
    #     t_max_probs_1 = t_softmax_1.max(dim=1)[0]  # [B, H, W]

    #     temp_mask = (copy_paste_masks != 255) * (t_max_probs_0 < t_max_probs_1)  # [B, H, W]
    #     # temp_mask = temp_mask.unsqueeze(dim=1).expand_as(t_logits_0).long().cuda()  # [B, C, H, W]
    #     temp_mask = temp_mask.long().cuda()  # [B, H, W]
    #     temp_dcst_loss = weight * cst_loss_fun(t_logits_0, t_softmax_1, refer_labels=temp_mask, region='confident', ignore_index=0)
    #     if not torch.isnan(temp_dcst_loss):
    #         dcst_loss += temp_dcst_loss

    #     # t_logits_1 -> t_logits_0
    #     temp_mask = (copy_paste_masks != 255) * (t_max_probs_1 < t_max_probs_0)
    #     # temp_mask = temp_mask.unsqueeze(dim=1).expand_as(t_logits_0).long().cuda()  # [B, C, H, W]
    #     temp_mask = temp_mask.long().cuda()  # [B, H, W]
    #     temp_dcst_loss = weight * cst_loss_fun(t_logits_1, t_softmax_0, refer_labels=temp_mask, region='confident', ignore_index=0)
    #     if not torch.isnan(temp_dcst_loss):
    #         dcst_loss += temp_dcst_loss

    #     # print('t_logits_1 -> t_logits_0: {}'.format(dcst_loss))

    #     return {'dcst_loss': dcst_loss / batch_size}


def build_region_weight(t_logits, t_plbl):
    reg_val_matrix = torch.ones_like(t_plbl).type_as(t_logits)  # [B, H, W]
    reg_val_matrix[t_plbl == 255] = 0
    reg_val_matrix = reg_val_matrix.unsqueeze(dim=1)  # [B, 1, H, W]
    reg_ignore_matrix = 1 - reg_val_matrix  # [B, 1, H, W]
    reg_weight = torch.ones_like(t_logits)  # [B, C, H, W]
    reg_weight_confident = reg_weight * reg_val_matrix  # [B, C, H, W]
    reg_weight_ignored = reg_weight * reg_ignore_matrix  # [B, C, H, W]

    return reg_weight_confident, reg_weight_ignored


def _entropy(logits, weight):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight > 0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classed = logits.size()[1]
    entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
    entropy_reg = torch.sum(entropy) / val_num
    return entropy_reg


def _kld(logits, weight):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight > 0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classes = logits.size()[1]
    kld = - 1 / num_classes * weight * logits_log_softmax
    kld_reg = torch.sum(kld) / val_num
    return kld_reg
