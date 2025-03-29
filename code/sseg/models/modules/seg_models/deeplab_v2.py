# -*- coding: utf-8 -*
from functools import partial
from torch import nn
from sseg.models.modules.resnet import build_resnet101
from utils.registry.registries import SEG_MODEL


class ASPP_V2(nn.Module):
    """ASPP of DeepLab V2"""

    def __init__(self, dilation_series, padding_series, num_classes):
        super(ASPP_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


@SEG_MODEL.register('DeepLab_V2')
class DeepLab_V2(nn.Module):
    def __init__(self, num_classes=19, output_dim=256):
        super(DeepLab_V2, self).__init__()

        # resnet-101, modify layer3 and layer4 with dilated convolutions
        self.backbone = build_resnet101(pretrained=True)
        self.backbone.layer3.apply(partial(self._nostride_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self._nostride_dilate, dilate=4))

        self.aspp = ASPP_V2([6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.representation = nn.Sequential(
            nn.Conv2d(2048, output_dim, 1),
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convolutions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.backbone(x)  # [B, 2048, H/8, W/8]

        prediction = self.aspp(x)  # [B, 19, H/8, W/8]

        representation = self.representation(x)  # [B, 256, H/8, W/8]
        return prediction, x

    def get_optimizer_params(self, lr):
        return [{'params': self.backbone.parameters(), "lr": lr},
                {'params': self.aspp.parameters(), "lr": lr * 10},  # other sub-blocks have greater lr
                {'params': self.representation.parameters(), "lr": lr * 10}]
