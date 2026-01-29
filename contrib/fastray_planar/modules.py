import math
import os
from functools import reduce

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torchvision.models import resnet18, ResNet18_Weights
from prefusion.registry import MODELS


__all__ = [
    "ConvBN",
    "Concat",
    "EltwiseAdd",
    "OSABlock",
    "VoVNetFPN",
    "ResNet18LiteFPN",
    "FastRaySpatialTransform",
    "VoxelTemporalAlign",
    "VoxelStreamFusion",
    "VoVNetEncoder",
    "PlanarHead"
]

import torch
import torch.nn as nn
from mmengine.model import BaseModule

# ---------------------------
# 基础组件：Conv -> BN -> ReLU6
# ---------------------------
class ConvBNReLU6(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_var, 1)
                nn.init.constant_(m.running_mean, 0)



    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------
# 无残差 PlainBlock（两层卷积）
# ---------------------------
class PlainBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU6(in_ch, out_ch, k=3, s=stride, p=1)
        self.conv2 = ConvBNReLU6(out_ch, out_ch, k=3, s=1, p=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ---------------------------
# 可配置通道的 Plain-18 骨干（无残差）
# 输出：C2(/4), C3(/8), C4(/16), C5(/32)
# ---------------------------
class Plain18Backbone(nn.Module):
    def __init__(self, c2_channels=64, c3_channels=128, c4_channels=256, c5_channels=512):
        super().__init__()
        self.c2_channels = c2_channels
        self.c3_channels = c3_channels
        self.c4_channels = c4_channels
        self.c5_channels = c5_channels

        # Stem：对齐到 c2_channels
        self.conv1 = nn.Conv2d(3, c2_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(c2_channels)
        self.act1  = nn.ReLU6(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> /4

        self.layer1 = self._make_stage(c2_channels, c2_channels, blocks=2, downsample=False)     # /4 => C2
        self.layer2 = self._make_stage(c2_channels, c3_channels, blocks=3, downsample=True)      # /8 => C3
        self.layer3 = self._make_stage(c3_channels, c4_channels, blocks=3, downsample=True)      # /16 => C4
        self.layer4 = self._make_stage(c4_channels, c5_channels, blocks=2, downsample=True)      # /32 => C5

    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, downsample):
        layers = [PlainBlock(in_ch, out_ch, stride=(2 if downsample else 1))]
        for _ in range(blocks - 1):
            layers.append(PlainBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x  = self.conv1(x); x = self.bn1(x); x = self.act1(x)
        x  = self.pool(x)                 # /4
        c2 = self.layer1(x)               # /4
        c3 = self.layer2(c2)              # /8
        c4 = self.layer3(c3)              # /16
        c5 = self.layer4(c4)              # /32
        return c2, c3, c4, c5


# ---------------------------
# 仅为 P2 服务的 Lite FPN 层（Top-down：P5→P4→P3→P2）
# 无加权融合；每步融合后接 3×3 ConvBNReLU6
# ---------------------------
class LiteFPNLayerP2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C = channels
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False)
        self.conv_p4_td = ConvBNReLU6(C, C, k=3, s=1, p=1)
        self.conv_p3_td = ConvBNReLU6(C, C, k=3, s=1, p=1)
        self.conv_p2_td = ConvBNReLU6(C, C, k=3, s=1, p=1)
        self.conv_p2_out = ConvBNReLU6(C, C, k=3, s=1, p=1)

    def forward(self, feats):
        P2, P3, P4, P5 = feats['P2'], feats['P3'], feats['P4'], feats['P5']
        P4_td = self.conv_p4_td(P4 + self.upsample(P5))         # /16
        P3_td = self.conv_p3_td(P3 + self.upsample(P4_td))      # /8
        P2_td = self.conv_p2_td(P2 + self.upsample(P3_td))      # /4
        P2_out = self.conv_p2_out(P2_td)                        # /4
        # 传回 P2_out 作为下一层的 P2；将 P3 也替换为 P3_td 以形成逐层精炼路径；P4/P5 保持（避免冗余额外计算分支）
        return {'P2': P2_out, 'P3': P3_td, 'P4': P4, 'P5': P5}


# ---------------------------
# 主模块：ResNet18LiteFPN（无残差骨干 + 单尺度输出 P2）
# ---------------------------
@MODELS.register_module()
class ResNet18LiteFPN(BaseModule):
    """
    • Backbone: Plain-18（无残差），各 stage 通道可配：c2/c3/c4/c5
    • Neck: 轻量 FPN，仅做 P5→P4→P3→P2 的 top-down 融合（逐元素相加，无加权）
    • 激活：所有卷积 Conv -> BN -> ReLU6
    • 输出：仅 P2，shape = (B, final_out_channels, H/4, W/4)
    """
    def __init__(self,
                 c2_channels=64,
                 c3_channels=128,
                 c4_channels=192,
                 c5_channels=256,
                 out_channels=80,          # FPN 内部统一通道
                 final_out_channels=None,   # 若为 None，则等于 out_channels
                 num_layers=3,
                 freeze_at=0,               # 0-4：冻结 stem+layer1..4
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        self.final_out_channels = final_out_channels or out_channels

        # --- 无残差骨干，通道完全可配
        self.backbone = Plain18Backbone(
            c2_channels=c2_channels,
            c3_channels=c3_channels,
            c4_channels=c4_channels,
            c5_channels=c5_channels
        )

        # --- 冻结（与 ResNet 语义一致）
        if freeze_at >= 1:
            for p in self.backbone.conv1.parameters(): p.requires_grad = False
            for p in self.backbone.bn1.parameters():  p.requires_grad = False
            for p in self.backbone.layer1.parameters(): p.requires_grad = False
        if freeze_at >= 2:
            for p in self.backbone.layer2.parameters(): p.requires_grad = False
        if freeze_at >= 3:
            for p in self.backbone.layer3.parameters(): p.requires_grad = False
        if freeze_at >= 4:
            for p in self.backbone.layer4.parameters(): p.requires_grad = False

        # --- C2..C5 → 统一到 out_channels（保证四路都在最终 P2 路径上被使用）
        self.c2_proj = ConvBNReLU6(c2_channels, out_channels, k=1, s=1, p=0)
        self.c3_proj = ConvBNReLU6(c3_channels, out_channels, k=1, s=1, p=0)
        self.c4_proj = ConvBNReLU6(c4_channels, out_channels, k=1, s=1, p=0)
        self.c5_proj = ConvBNReLU6(c5_channels, out_channels, k=1, s=1, p=0)

        # --- 3×3 对齐（参与训练）
        self.p2_align = ConvBNReLU6(out_channels, out_channels, k=3, s=1, p=1)
        self.p3_align = ConvBNReLU6(out_channels, out_channels, k=3, s=1, p=1)
        self.p4_align = ConvBNReLU6(out_channels, out_channels, k=3, s=1, p=1)
        self.p5_align = ConvBNReLU6(out_channels, out_channels, k=3, s=1, p=1)


        self.upsample_p3 = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False)
        self.upsample_p4 = nn.Upsample(scale_factor=4,mode='bilinear', align_corners=False)
        self.upsample_p5 = nn.Upsample(scale_factor=8,mode='bilinear', align_corners=False)

        self.feat_conv = self._make_stage(out_channels, out_channels, blocks=3, downsample=False)     # /4 => C2


    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, downsample):
        layers = [PlainBlock(in_ch, out_ch, stride=(2 if downsample else 1))]
        for _ in range(blocks - 1):
            layers.append(PlainBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

        
        # --- 仅 P2 路径的 Lite FPN 层
        # self.fpn_layers = nn.ModuleList([LiteFPNLayerP2(out_channels) for _ in range(num_layers)])

        # --- 可选最终 1×1 调整输出通道（仍接 BN+ReLU6 以满足“每个卷积后 ReLU6”）
        # if self.final_out_channels == out_channels:
        #     self.head_out = nn.Identity()
        # else:
        #     self.head_out = ConvBNReLU6(out_channels, self.final_out_channels, k=1, s=1, p=0)

    def forward(self, x):
        # Backbone：得到 C2..C5
        c2, c3, c4, c5 = self.backbone(x)

        # 统一通道 + 对齐（确保四路都进到最终 P2 的梯度路径）
        p2 = self.p2_align(self.c2_proj(c2))  # /4
        p3 = self.upsample_p3(self.p3_align(self.c3_proj(c3)))  # /8
        p4 = self.upsample_p4(self.p4_align(self.c4_proj(c4)))  # /16
        p5 = self.upsample_p5(self.p5_align(self.c5_proj(c5)))  # /32

        feats = p2 + p3 + p4 + p5  # 直接融合（无加权）
        # feats = {'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5}
        # for layer in self.fpn_layers:
        #     feats = layer(feats)

        feats = self.feat_conv(feats)   # (B, final_out_channels, H/4, W/4)
        return feats




class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 has_relu=True,
                 relu6=False):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.001)
        self.has_relu = has_relu
        if self.has_relu:
            if relu6:
                self.relu = nn.ReLU6(inplace=True)
            else:
                self.relu = nn.ReLU(inplace=True)
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.has_relu:
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_var, 1)
                nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        if hasattr(self, 'bn'):  # IMPORTANT! PREPARED FOR BN FUSION, SINCE BN WILL BE DELETED AFTER FUSED
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()

        kernel_size = 3
        padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=True)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU6()
        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


@MODELS.register_module()
class EltwiseAdd(nn.Module):
    def __init__(self):
        super(EltwiseAdd, self).__init__()

    def forward(self, *inputs):
        # return torch.add(*inputs)
        return reduce(lambda x, y: torch.add(x, y), [i for i in inputs])


class OSABlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels=None,
                 stride=2,
                 dilation=1,
                 repeat=5,
                 final_dilation=1,
                 with_reduce=True,
                 has_bn=True,
                 relu6=False):
        super(OSABlock, self).__init__()
        assert stride in [1, 2]
        assert repeat >= 2
        self.repeat = repeat
        self.with_reduce = with_reduce

        self.conv1 = ConvBN(in_channels, mid_channels, stride=stride, padding=dilation, dilation=dilation, relu6=relu6)

        for i in range(repeat - 2):
            self._modules['conv{}'.format(i + 2)] = ConvBN(
                mid_channels, mid_channels, padding=dilation, dilation=dilation, relu6=relu6
            )

        self._modules['conv{}'.format(repeat)] = ConvBN(
            mid_channels, mid_channels, padding=final_dilation, dilation=final_dilation, relu6=relu6
        )

        self.concat = Concat()
        if with_reduce:
            assert out_channels is not None
            if has_bn:
                self.reduce = ConvBN(mid_channels * repeat, out_channels, kernel_size=1, padding=0, relu6=relu6)
            else:
                self.reduce = nn.Conv2d(mid_channels * repeat, out_channels, kernel_size=1, padding=0)

    #     self.init_params()
    # def init_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        layers = []
        for i in range(self.repeat):
            x = self._modules['conv{}'.format(i + 1)](x)
            layers.append(x)
        x = self.concat(*layers)
        if self.with_reduce:
            x = self.reduce(x)
        return x


@MODELS.register_module()
class VoVNetFPN(BaseModule):
    def __init__(self, out_stride=8, out_channels=128, init_cfg=None, relu6=False, last_bn=False):
        super().__init__(init_cfg=init_cfg)
        self.strides = [4, 8, 16, 32]
        assert out_stride in self.strides
        self.out_stride = out_stride

        # BACKBONE
        self.stem1 = ConvBN(3, 64, stride=2, relu6=relu6)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3, relu6=relu6)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2, relu6=relu6)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2, relu6=relu6)

        # NECK
        if self.out_stride <= 16:
            self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
            self.p4_fusion = Concat()
        if self.out_stride <= 8:
            self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0, relu6=relu6)
            self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
            self.p3_fusion = Concat()
        if self.out_stride <= 4:
            self.p2_linear = ConvBN(256, 96, kernel_size=1, padding=0, relu6=relu6)
            self.p2_up = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2, padding=0, bias=False)
            self.p2_fusion = Concat()
        
        in_channels = {4: 192, 8: 256, 16: 384, 32: 192}
        mid_channels = {4: 96, 8: 96, 16: 128, 32: 192}
        self.out = OSABlock(
            in_channels[self.out_stride], mid_channels[self.out_stride], out_channels,
            stride=1, repeat=3, has_bn=last_bn, relu6=relu6
        )
        
        
    def forward(self, x):  # x: (N, 3, H, W)
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1) # shape: (bs, 96, 64, 176)
        osa3 = self.osa3(osa2) # shape: (bs, 128, 32, 88)
        osa4 = self.osa4(osa3) # shape: (bs, 192, 16, 44)
        osa5 = self.osa5(osa4) # shape: (bs, 192, 8, 22)

        if self.out_stride <= 32:
            out = osa5
        if self.out_stride <= 16:
            out = self.p4_fusion(self.p4_up(out), osa4)
        if self.out_stride <= 8:
            out = self.p3_fusion(self.p3_up(self.p3_linear(out)), osa3)
        if self.out_stride <= 4:
            out = self.p2_fusion(self.p2_up(self.p2_linear(out)), osa2)
        
        out = self.out(out)
        
        return out



# 原始backbone
@MODELS.register_module()
class VoVNetSlimFPN(BaseModule):
    def __init__(self, out_channels=80, init_cfg=None, relu6=False, hwc_out=False):
        super().__init__(init_cfg=init_cfg)

        # BACKBONE
        self.stem1 = ConvBN(3, 64, stride=2, relu6=relu6)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3, relu6=relu6)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2, relu6=relu6)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2, relu6=relu6)

        # NECK
        self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
        self.p4_fusion = Concat()
        self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0, relu6=relu6)
        self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.p3_fusion = Concat()
        
        self.out = OSABlock(256, 96, stride=1, repeat=3, has_bn=False, with_reduce=False, relu6=relu6)
        self.up_linear = ConvBN(288, out_channels, kernel_size=1, padding=0, relu6=relu6)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)

        self.hwc_out = hwc_out
        
        self.relu6 = relu6
        if self.relu6:
            self.out_bn = nn.BatchNorm2d(out_channels)
            self.out_relu = nn.ReLU6(inplace=True)
    def forward(self, x):  # x: (N, 3, H, W)
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1)
        osa3 = self.osa3(osa2)
        osa4 = self.osa4(osa3)
        osa5 = self.osa5(osa4)

        p4 = self.p4_fusion(self.p4_up(osa5), osa4)
        p3 = self.p3_fusion(self.p3_up(self.p3_linear(p4)), osa3)
        
        out = self.up(self.up_linear(self.out(p3)))

        if self.relu6:
            out = self.out_bn(out)
            out = self.out_relu(out)
            if self.hwc_out:
                out = out.permute(0, 2, 3, 1).contiguous()
        return out


@MODELS.register_module()
class VoVNetLiteFPN(BaseModule):
    def __init__(self, out_channels=80, init_cfg=None, relu6=False, hwc_out=False):
        super().__init__(init_cfg=init_cfg)

        # BACKBONE
        self.stem1 = ConvBN(3, 64, stride=2, relu6=relu6)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3, relu6=relu6)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2, relu6=relu6)
        # self.osa4_1 = OSABlock(192, 192, 192, stride=1, repeat=5, final_dilation=2, relu6=relu6)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2, relu6=relu6)
        # self.osa5_1 = OSABlock(192, 192, 192, stride=1, repeat=4, final_dilation=2, relu6=relu6)
        # NECK
        self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
        self.p4_fusion = Concat()
        self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0, relu6=relu6)
        self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.p3_fusion = Concat()
        
        self.out = OSABlock(256, 96, stride=1, repeat=3, has_bn=False, with_reduce=False, relu6=relu6)
        self.up_linear = ConvBN(288, out_channels, kernel_size=1, padding=0, relu6=relu6)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)

        self.hwc_out = hwc_out
        
        self.relu6 = relu6
        if self.relu6:
            self.out_bn = nn.BatchNorm2d(out_channels)
            self.out_relu = nn.ReLU6(inplace=True)
            
    def forward(self, x):  # x: (N, 3, H, W)
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1)
        osa3 = self.osa3(osa2)
        osa4 = self.osa4(osa3)
        osa5 = self.osa5(osa4)

        p4 = self.p4_fusion(self.p4_up(osa5), osa4)
        p3 = self.p3_fusion(self.p3_up(self.p3_linear(p4)), osa3)
        
        out = self.up(self.up_linear(self.out(p3)))

        if self.relu6:
            out = self.out_bn(out)
            out = self.out_relu(out)
            if self.hwc_out:
                out = out.permute(0, 2, 3, 1).contiguous()
        return out


@MODELS.register_module()
class ResNetFPN(BaseModule):
    def __init__(
        self, 
        out_stride=8, 
        out_channels=128, 
        fpn_lateral_channel=64, 
        fpn_in_channels=(256, 512, 1024, 2048), 
        init_cfg=None, 
        **backbone_kwargs
    ):
        super().__init__(init_cfg=init_cfg)
        self.strides = [4, 8, 16, 32]
        assert out_stride in self.strides
        self.out_stride = out_stride

        # BACKBONE
        backbone_kwargs["type"] = "mmdet.ResNet"
        self.backbone = MODELS.build(backbone_kwargs)

        # NECK (FPN)
        if self.out_stride <= 16:
            self.p4_up = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0, bias=False)
            self.p4_fusion = Concat()
        if self.out_stride <= 8:
            self.p3_linear = ConvBN(2048, 512, kernel_size=1, padding=0)
            self.p3_up = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0, bias=False)
            self.p3_fusion = Concat()
        if self.out_stride <= 4:
            self.p2_linear = ConvBN(1024, 256, kernel_size=1, padding=0)
            self.p2_up = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False)
            self.p2_fusion = Concat()
        
        in_channels = {4: 512, 8: 1024, 16: 2048, 32: 2048}
        mid_channels = {4: 128, 8: 256, 16: 512, 32: 1024}
        self.out = OSABlock(
            in_channels[self.out_stride], mid_channels[self.out_stride], out_channels,
            stride=1, repeat=3, has_bn=False
        )

        
    def forward(self, x):  # x: (N, 3, H, W)
        feats = self.backbone(x) # feats shape: [bs, 256, 64, 176], [bs, 512, 32, 88], [bs, 1024, 16, 44], [bs, 2048, 8, 22]
        # fpn_feats = self.fpn(feats)
        # out_feat_layer_idx = int(math.log2(self.out_stride) - 2)
        # out = fpn_feats[out_feat_layer_idx]

        if self.out_stride <= 32:
            out = feats[3]
        if self.out_stride <= 16:
            out = self.p4_fusion(self.p4_up(out), feats[2])
        if self.out_stride <= 8:
            out = self.p3_fusion(self.p3_up(self.p3_linear(out)), feats[1])
        if self.out_stride <= 4:
            out = self.p2_fusion(self.p2_up(self.p2_linear(out)), feats[0])
        
        out = self.out(out)
        
        return out



@MODELS.register_module()
class FastRaySpatialTransform(BaseModule):
    
    def __init__(self, 
                 voxel_shape, 
                 fusion_mode='weighted', 
                 bev_mode=False, 
                 reduce_channels=False,
                 in_channels=None,
                 out_channels=None,
                 dump_voxel_feats=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_shape = voxel_shape
        assert fusion_mode in ['weighted', 'sampled', 'bilinear_weighted']
        self.fusion_mode = fusion_mode
        self.bev_mode = bev_mode
        self.dump_voxel_feats = dump_voxel_feats
        self.reduce_channels = reduce_channels and bev_mode
        if self.reduce_channels:
            # self.channel_reduction =  ConvBNReLU6(in_channels, out_channels, k=1, s=1, p=0)
            self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, camera_feats_dict, camera_lookups):
        '''Output a 3d voxel tensor from 2d image features
        
        Parameters
        ----------
        camera_feats_dict : Dict[str, torch.Tensor]
            camera feature tensor of shape (N, C, H, W)
        camera_lookups : List[Dict[str, torch.Tensor]]
            camera lookup tensors of shape Z*X*Y
        
        Returns
        -------
        voxel_feats : torch.Tensor
            voxel features of shape (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        '''
        cam_ids = list(camera_feats_dict.keys())
        # cat camera features
        camera_feats = []
        for cam_id in cam_ids:
            camera_feats.append(camera_feats_dict[cam_id])
        # get sizes
        Z, X, Y = self.voxel_shape
        N, C = camera_feats[0].shape[0:2]
        # initialize voxel features, (N, C, Z*X*Y)
        voxel_feats = torch.zeros(N, C, Z*X*Y, 
                                  dtype=camera_feats[0].dtype,
                                  device=camera_feats[0].device)
        # iterate over batch
        for n in range(N):
            # iterate over cameras
            for b, cam_id in enumerate(cam_ids):
                camera_feat = camera_feats[b][n]  # C, H, W
                camera_lookup = camera_lookups[n][cam_id]
                if self.fusion_mode == 'weighted':
                    valid_map = camera_lookup['valid_map']
                    valid_uu = camera_lookup['uu'][valid_map]
                    valid_vv = camera_lookup['vv'][valid_map]
                    valid_norm_density_map = camera_lookup['norm_density_map'][valid_map]
                    voxel_feats[n, :, valid_map] += valid_norm_density_map[None] * camera_feat[:, valid_vv, valid_uu]
                if self.fusion_mode == 'sampled':
                    valid_map = camera_lookup['valid_map_sampled']
                    valid_uu = camera_lookup['uu'][valid_map]
                    valid_vv = camera_lookup['vv'][valid_map]
                    voxel_feats[n, :, valid_map] = camera_feat[:, valid_vv, valid_uu]
                if self.fusion_mode == 'bilinear_weighted':
                    valid_map = camera_lookup['valid_map_bilinear']
                    valid_norm_density_map = camera_lookup['norm_density_map'][valid_map]
                    valid_uu_floor = camera_lookup['uu_floor'][valid_map]
                    valid_uu_ceil = camera_lookup['uu_ceil'][valid_map]
                    valid_vv_floor = camera_lookup['vv_floor'][valid_map]
                    valid_vv_ceil = camera_lookup['vv_ceil'][valid_map]
                    valid_uu_bilinear_weight = camera_lookup['uu_bilinear_weight'][valid_map].to(camera_feat.dtype)
                    valid_vv_bilinear_weight = camera_lookup['vv_bilinear_weight'][valid_map].to(camera_feat.dtype)
                    interpolated_camera_feat = valid_uu_bilinear_weight[None] * (
                        camera_feat[:, valid_vv_floor, valid_uu_floor] * valid_vv_bilinear_weight[None] +
                        camera_feat[:, valid_vv_ceil, valid_uu_floor] * (1 - valid_vv_bilinear_weight)[None]
                    ) + (1 - valid_uu_bilinear_weight[None]) * (
                        camera_feat[:, valid_vv_floor, valid_uu_ceil] * valid_vv_bilinear_weight[None] +
                        camera_feat[:, valid_vv_ceil, valid_uu_ceil] * (1 - valid_vv_bilinear_weight)[None]
                    )
                    voxel_feats[n, :, valid_map] += valid_norm_density_map[None] * interpolated_camera_feat
                    
        # reshape voxel_feats
        if self.bev_mode:
            voxel_feats = voxel_feats.reshape(N, C*Z, X, Y)
            if self.reduce_channels:
                bev_feats = self.channel_reduction(voxel_feats)
                if self.dump_voxel_feats:
                    return bev_feats, voxel_feats
                return bev_feats
            return voxel_feats
        else:
            return voxel_feats.reshape(N, C, Z, X, Y)



@MODELS.register_module()
class FastRaySpatialTransformRelu6(BaseModule):
    
    def __init__(self, 
                 voxel_shape, 
                 fusion_mode='weighted', 
                 bev_mode=False, 
                 reduce_channels=False,
                 in_channels=None,
                 out_channels=None,
                 dump_voxel_feats=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_shape = voxel_shape
        assert fusion_mode in ['weighted', 'sampled', 'bilinear_weighted']
        self.fusion_mode = fusion_mode
        self.bev_mode = bev_mode
        self.dump_voxel_feats = dump_voxel_feats
        self.reduce_channels = reduce_channels and bev_mode
        if self.reduce_channels:
            self.channel_reduction =  ConvBNReLU6(in_channels, out_channels, k=1, s=1, p=0)
            # self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, camera_feats_dict, camera_lookups):
        '''Output a 3d voxel tensor from 2d image features
        
        Parameters
        ----------
        camera_feats_dict : Dict[str, torch.Tensor]
            camera feature tensor of shape (N, C, H, W)
        camera_lookups : List[Dict[str, torch.Tensor]]
            camera lookup tensors of shape Z*X*Y
        
        Returns
        -------
        voxel_feats : torch.Tensor
            voxel features of shape (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        '''
        cam_ids = list(camera_feats_dict.keys())
        # cat camera features
        camera_feats = []
        for cam_id in cam_ids:
            camera_feats.append(camera_feats_dict[cam_id])
        # get sizes
        Z, X, Y = self.voxel_shape
        N, C = camera_feats[0].shape[0:2]
        # initialize voxel features, (N, C, Z*X*Y)
        voxel_feats = torch.zeros(N, C, Z*X*Y, 
                                  dtype=camera_feats[0].dtype,
                                  device=camera_feats[0].device)
        # iterate over batch
        for n in range(N):
            # iterate over cameras
            for b, cam_id in enumerate(cam_ids):
                camera_feat = camera_feats[b][n]  # C, H, W
                camera_lookup = camera_lookups[n][cam_id]
                if self.fusion_mode == 'weighted':
                    valid_map = camera_lookup['valid_map']
                    valid_uu = camera_lookup['uu'][valid_map]
                    valid_vv = camera_lookup['vv'][valid_map]
                    valid_norm_density_map = camera_lookup['norm_density_map'][valid_map]
                    voxel_feats[n, :, valid_map] += valid_norm_density_map[None] * camera_feat[:, valid_vv, valid_uu]
                if self.fusion_mode == 'sampled':
                    valid_map = camera_lookup['valid_map_sampled']
                    valid_uu = camera_lookup['uu'][valid_map]
                    valid_vv = camera_lookup['vv'][valid_map]
                    voxel_feats[n, :, valid_map] = camera_feat[:, valid_vv, valid_uu]
                if self.fusion_mode == 'bilinear_weighted':
                    valid_map = camera_lookup['valid_map_bilinear']
                    valid_norm_density_map = camera_lookup['norm_density_map'][valid_map]
                    valid_uu_floor = camera_lookup['uu_floor'][valid_map]
                    valid_uu_ceil = camera_lookup['uu_ceil'][valid_map]
                    valid_vv_floor = camera_lookup['vv_floor'][valid_map]
                    valid_vv_ceil = camera_lookup['vv_ceil'][valid_map]
                    valid_uu_bilinear_weight = camera_lookup['uu_bilinear_weight'][valid_map].to(camera_feat.dtype)
                    valid_vv_bilinear_weight = camera_lookup['vv_bilinear_weight'][valid_map].to(camera_feat.dtype)
                    interpolated_camera_feat = valid_uu_bilinear_weight[None] * (
                        camera_feat[:, valid_vv_floor, valid_uu_floor] * valid_vv_bilinear_weight[None] +
                        camera_feat[:, valid_vv_ceil, valid_uu_floor] * (1 - valid_vv_bilinear_weight)[None]
                    ) + (1 - valid_uu_bilinear_weight[None]) * (
                        camera_feat[:, valid_vv_floor, valid_uu_ceil] * valid_vv_bilinear_weight[None] +
                        camera_feat[:, valid_vv_ceil, valid_uu_ceil] * (1 - valid_vv_bilinear_weight)[None]
                    )
                    voxel_feats[n, :, valid_map] += valid_norm_density_map[None] * interpolated_camera_feat
                    
        # reshape voxel_feats
        if self.bev_mode:
            voxel_feats = voxel_feats.reshape(N, C*Z, X, Y)
            if self.reduce_channels:
                bev_feats = self.channel_reduction(voxel_feats)
                if self.dump_voxel_feats:
                    return bev_feats, voxel_feats
                return bev_feats
            return voxel_feats
        else:
            return voxel_feats.reshape(N, C, Z, X, Y)
            
@MODELS.register_module()
class VoxelTemporalAlign(BaseModule):
    
    def __init__(self,
                 voxel_shape,
                 voxel_range,
                 bev_mode=False,
                 interpolation='bilinear',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range
        self.bev_mode = bev_mode
        self.interpolation = interpolation
        self.voxel_intrinsics = self._get_voxel_intrinsics(voxel_shape, voxel_range)
    
    @staticmethod
    def _get_voxel_intrinsics(voxel_shape, voxel_range):
        Z, X, Y = voxel_shape    
        fx = X / (voxel_range[1][1] - voxel_range[1][0])
        fy = Y / (voxel_range[2][1] - voxel_range[2][0])
        fz = Z / (voxel_range[0][1] - voxel_range[0][0])
        cx = - voxel_range[1][0] * fx - 0.5
        cy = - voxel_range[2][0] * fy - 0.5
        cz = - voxel_range[0][0] * fz - 0.5
        return cx, cy, cz, fx, fy, fz


    def _unproject_points_from_voxel_to_ego(self):
        Z, X, Y = self.voxel_shape
        cx, cy, cz, fx, fy, fz = self.voxel_intrinsics
        if self.bev_mode:
            xx, yy = torch.meshgrid(torch.arange(X), torch.arange(Y), indexing='ij')
            xx_ego = (xx - cx) / fx
            yy_ego = (yy - cy) / fy
            return torch.stack([
                xx_ego.reshape(-1), 
                yy_ego.reshape(-1),
            ], dim=0)
        else:
            zz, xx, yy = torch.meshgrid(torch.arange(Z), torch.arange(X), torch.arange(Y), indexing='ij')
            xx_ego = (xx - cx) / fx
            yy_ego = (yy - cy) / fy
            zz_ego = (zz - cz) / fz
            return torch.stack([
                xx_ego.reshape(-1), 
                yy_ego.reshape(-1),
                zz_ego.reshape(-1),
            ], dim=0)


    def _project_points_from_ego_to_voxel(self, ego_points, normalize=True):
        """Convert points from ego coords to voxel coords.

        Parameters
        ----------
        ego_points : torch.Tensor
            shape should be (N, 2, X * Y) or (N, 3, Z * X * Y)
        
        normalize : bool
            whether to normalize the projected points to [-1, 1]
            
        Return
        ------
        output : torch.Tensor
            shape should be (N, X, Y, 2) or (N, Z, X, Y, 3)
        
        """
        Z, X, Y = self.voxel_shape
        N, _, _ = ego_points.shape
        cx, cy, cz, fx, fy, fz = self.voxel_intrinsics
        if self.bev_mode:
            xx_egos = ego_points[:, 0]
            yy_egos = ego_points[:, 1]
            xx_ = xx_egos * fx + cx
            yy_ = yy_egos * fy + cy
            if normalize:
                xx_ = 2 * xx_ / (X - 1) - 1
                yy_ = 2 * yy_ / (Y - 1) - 1
            grid = torch.stack([
                yy_.reshape(N, X, Y),
                xx_.reshape(N, X, Y) 
            ], dim=-1)
        else:
            xx_egos = ego_points[:, 0]
            yy_egos = ego_points[:, 1]
            zz_egos = ego_points[:, 2]
            xx_ = xx_egos * fx + cx
            yy_ = yy_egos * fy + cy
            zz_ = zz_egos * fz + cz
            if normalize:
                xx_ = 2 * xx_ / (X - 1) - 1
                yy_ = 2 * yy_ / (Y - 1) - 1
                zz_ = 2 * zz_ / (Z - 1) - 1
            grid = torch.stack([
                yy_.reshape(N, Z, X, Y),
                xx_.reshape(N, Z, X, Y),
                zz_.reshape(N, Z, X, Y)
            ], dim=-1)
        return grid
        
    
    def forward(self, voxel_feats_pre, delta_poses):
        """
        Output a time-aligned voxel tensor from previous voxel features.

        Parameters
        ----------
        voxel_feats_pre : torch.Tensor
            shape should be (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        delta_poses : torch.Tensor
            shape should be (N, 4, 4)
        
        Return
        ------
        output : torch.Tensor
            shape should be (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        """
        # gen ego_points from voxel
        ego_points = self._unproject_points_from_voxel_to_ego().to(
            voxel_feats_pre, non_blocking=True)[None]
        # get projection matrix
        if self.bev_mode:
            assert len(voxel_feats_pre.shape) == 4, 'must be 4-D Tensor'
            rotations = delta_poses[:, :2, :2]
            translations = delta_poses[:, :2, [3]]
        else:
            assert len(voxel_feats_pre.shape) == 5, 'must be 5-D Tensor'
            rotations = delta_poses[:, :3, :3]
            translations = delta_poses[:, :3, [3]]
        # project to previous ego coords and get grid
        ego_points_projected = rotations @ ego_points + translations
        grid = self._project_points_from_ego_to_voxel(ego_points_projected)
        # apply grid sampling
        voxel_feats_pre_aligned = nn.functional.grid_sample(
            input=voxel_feats_pre, grid=grid, mode=self.interpolation, align_corners=False
        )
        return voxel_feats_pre_aligned


@MODELS.register_module()
class VoxelStreamFusion(BaseModule):
    def __init__(self, in_channels, mid_channels=128, bev_mode=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_mode = bev_mode
        if bev_mode:
            self.gain = nn.Sequential(
                ConvBN(in_channels, mid_channels),
                ConvBN(mid_channels, mid_channels),
                nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.gain = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(),
                nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(),
                nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        self.add = EltwiseAdd()
            
    
    def forward(self, voxel_feats_cur, voxel_feats_pre):
        """Temporal fusion of voxel current and previous features.

        Parameters
        ----------
        voxel_feats_cur : torch.Tensor
            current voxel features for measurement
        voxel_feats_pre : torch.Tensor
            previously predicted voxel features

        Returns
        -------
        voxel_feats_updated : torch.Tensor
            updated voxel features
        """
        assert voxel_feats_cur.shape == voxel_feats_pre.shape
        # get gain
        gain = self.gain(voxel_feats_cur)
        # update feats
        voxel_feats_updated = self.add(
            gain * voxel_feats_cur, (1 - gain) * voxel_feats_pre
        )
        return voxel_feats_updated
            

@MODELS.register_module()
class VoxelConcatFusion(BaseModule):
    def __init__(self, in_channels, pre_nframes, bev_mode=False, dilation=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_mode = bev_mode
        self.cat = Concat()
        if bev_mode:
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels * (pre_nframes + 1), in_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.fuse = nn.Sequential(
                nn.Conv3d(in_channels * (pre_nframes + 1), in_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(in_channels)
            )
    
    def forward(self, *aligned_voxel_feats_cat):
        """Temporal fusion of voxel current and previous features.

        Parameters
        ----------
        aligned_voxel_feats_cat : List[torch.Tensor]
            concated nframes of voxel features

        Returns
        -------
        voxel_feats_fused : torch.Tensor
            updated voxel features
        """
        cat = self.cat(*aligned_voxel_feats_cat)
        voxel_feats_fused = self.fuse(cat)
        return voxel_feats_fused


@MODELS.register_module()
class FeatureConcatFusion(BaseModule):
    def __init__(self, in_channels, out_channels, bev_mode=False, dilation=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.bev_mode = bev_mode
        self.cat = Concat()
        if bev_mode:
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation,
                          dilation=dilation),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.fuse = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, *aligned_voxel_feats_cat):
        """Temporal fusion of voxel current and previous features.

        Parameters
        ----------
        aligned_voxel_feats_cat : List[torch.Tensor]
            concated nframes of voxel features

        Returns
        -------
        voxel_feats_fused : torch.Tensor
            updated voxel features
        """
        cat = self.cat(*aligned_voxel_feats_cat)
        voxel_feats_fused = self.fuse(cat)
        return voxel_feats_fused


@MODELS.register_module()
class VoVNetEncoder(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 out_channels,
                 repeat=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.enc_tower = OSABlock(in_channels, 
                                  mid_channels, 
                                  out_channels, 
                                  stride=1, 
                                  repeat=repeat)
    
    def forward(self, x):
        return self.enc_tower(x)


class ResModule(BaseModule):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.convbn1 = ConvBN(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, has_relu=True)
        self.convbn2 = ConvBN(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, has_relu=False)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        x = self.convbn1(x)
        x = self.convbn2(x)
        x = identity + x
        x = self.activation(x)
        return x


@MODELS.register_module()
class M2BevEncoder(BaseModule):
    def __init__(self, 
                 in_channels, 
                 shrink_channels,
                 out_channels,
                 repeat=7,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
    
        # if fuse is not None:
        #     self.fuse = nn.Conv2d(fuse["in_channels"], fuse["out_channels"], kernel_size=1)
        # else:
        #     self.fuse = None
        self.shrink = nn.Conv2d(in_channels, shrink_channels, kernel_size=1)
        model = nn.ModuleList()
        model.append(ResModule(in_channels=shrink_channels, mid_channels=shrink_channels, out_channels=shrink_channels))
        model.append(ConvBN(
                in_channels=shrink_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                has_relu=True))
        for i in range(repeat - 1):
            model.append(ResModule(in_channels=out_channels, mid_channels=out_channels, out_channels=out_channels))
            model.append(ConvBN(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    has_relu=True))
        self.model = nn.Sequential(*model)
        self.init_params()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        x = self.shrink(x)
        out = self.model.forward(x)
        return out
    
    def init_params(self):
        nn.init.xavier_normal_(self.shrink.weight.data)





class ResModule2D(nn.Module):
    def __init__(self, n_channels, norm_cfg=dict(type='BN2d'), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x


@MODELS.register_module()
class M2BevNeck(nn.Module):
    """Neck for M2BEV.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers=2,
                 norm_cfg=dict(type='BN2d'),
                 stride=2,
                 is_transpose=True,
                 fuse=None,
                 with_cp=False):
        super().__init__()

        self.is_transpose = is_transpose
        self.with_cp = with_cp
        for i in range(3):
            print('neck transpose: {}'.format(is_transpose))

        if fuse is not None:
            self.fuse = nn.Conv2d(fuse["in_channels"], fuse["out_channels"], kernel_size=1)
        else:
            self.fuse = None

        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        for i in range(num_layers):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        if bool(os.getenv("DEPLOY", False)):
            N, X, Y, Z, C = x.shape
            x = x.reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)
        else:
            # N, C*T, X, Y, Z -> N, X, Y, Z, C -> N, X, Y, Z*C*T -> N, Z*C*T, X, Y
            N, C, X, Y, Z = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(N, X, Y, Z*C).permute(0, 3, 1, 2)

        if self.fuse is not None:
            x = self.fuse(x)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            return [x.transpose(-1, -2)]
        else:
            return [x]

# @MODELS.register_module()
# class VoxelEncoderLiteFPN(BaseModule):
#     """
#     VGG-style backbone + top-down FPN（只下采样两次）：
#       Backbone:
#         stage0: PlainBlock × repeats[0], first_stride=1  -> C0 (/1), ch = mid[0]
#         stage1: PlainBlock × repeats[1], first_stride=2  -> C1 (/2), ch = mid[1]
#         stage2: PlainBlock × repeats[2], first_stride=2  -> C2 (/4), ch = mid[2]

#       FPN（逐元素相加，无加权）:
#         P1 (/2) = refine_p1( C1 + up(1x1(C2)->ch=C1) )
#         P0 (/1) = refine_p0( C0 + up(1x1(P1)->ch=C0) )

#       输出：
#         out (/1) = ConvBNReLU6(P0, out_channels, 3×3)

#     """
#     def __init__(self, 
#                  in_channels, 
#                  mid_channels_list,   # [c0, c1, c2]
#                  out_channels,
#                  repeats=3,
#                  init_cfg=None,
#                  relu6=False):        # 仅保留兼容；ConvBNReLU6 固定 ReLU6
#         super().__init__(init_cfg=init_cfg)
#         assert len(mid_channels_list) == 3, "mid_channels_list 必须为 [c0, c1, c2]"
#         if isinstance(repeats, int):
#             repeats = (repeats, repeats, repeats)
#         else:
#             assert len(repeats) == 3, "repeats 必须为长度为3的 tuple/list"

#         c0, c1, c2 = mid_channels_list

#         # -------- Backbone：仅两次下采样 (/1 -> /2 -> /4) --------
#         self.stage0 = self._make_stage(in_channels, c0, repeats[0], first_stride=1)  # /1
#         self.stage1 = self._make_stage(c0,         c1, repeats[1], first_stride=2)  # /2
#         self.stage2 = self._make_stage(c1,         c2, repeats[2], first_stride=2)  # /4

#         # -------- FPN：通道对齐 + 上采样(UPSAMPLE) + 融合 + 3×3 refine --------
#         self.lat_2to1  = ConvBNReLU6(c2, c1, k=1, s=1, p=0)   # C2->c1
#         self.refine_p1 = ConvBNReLU6(c1, c1, k=3, s=1, p=1)

#         self.lat_1to0  = ConvBNReLU6(c1, c0, k=1, s=1, p=0)   # P1->c0
#         self.refine_p0 = ConvBNReLU6(c0, c0, k=3, s=1, p=1)

#         # 统一的 2x 上采样（替换 ConvTranspose2d）
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

#         # -------- Output head --------
#         self.out = ConvBNReLU6(c0, out_channels, k=3, s=1, p=1)

#     @staticmethod
#     def _make_stage(in_ch, out_ch, repeat, first_stride=1):
#         blocks = [PlainBlock(in_ch, out_ch, stride=first_stride)]
#         for _ in range(repeat - 1):
#             blocks.append(PlainBlock(out_ch, out_ch, stride=1))
#         return nn.Sequential(*blocks)

#     def forward(self, x):
#         # Backbone
#         c0 = self.stage0(x)    # /1, ch=c0
#         c1 = self.stage1(c0)   # /2, ch=c1
#         c2 = self.stage2(c1)   # /4, ch=c2

#         # FPN: C2 -> P1 -> P0 （使用 Upsample）
#         p1 = self.refine_p1(c1 + self.upsample2(self.lat_2to1(c2)))  # /2, ch=c1
#         p0 = self.refine_p0(c0 + self.upsample2(self.lat_1to0(p1)))  # /1, ch=c0

#         return self.out(p0)    # /1, ch=out_channels



# 原始bev特征融合
@MODELS.register_module()
class VoxelEncoderFPN(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels_list, 
                 out_channels,
                 repeats=3,
                 init_cfg=None,
                 relu6=False):
        super().__init__(init_cfg=init_cfg)
        assert len(mid_channels_list) == 3
        if type(repeats) is int:
            repeats = (repeats, repeats, repeats)
        else:
            assert len(repeats) == 3
        self.osa0 = OSABlock(in_channels, mid_channels_list[0], mid_channels_list[0], stride=1, repeat=repeats[0], final_dilation=2, relu6=relu6)
        self.osa1 = OSABlock(mid_channels_list[0], mid_channels_list[1], mid_channels_list[1], stride=2, repeat=repeats[1], final_dilation=2, relu6=relu6)
        self.osa2 = OSABlock(mid_channels_list[1], mid_channels_list[2], mid_channels_list[2], stride=2, repeat=repeats[2], final_dilation=2, relu6=relu6)

        self.p1_linear = ConvBN(mid_channels_list[2], mid_channels_list[1], kernel_size=1, padding=0, relu6=relu6)
        self.p1_up = nn.ConvTranspose2d(mid_channels_list[1], mid_channels_list[1], kernel_size=2, stride=2, padding=0, bias=False)
        self.p1_fusion = Concat()
        self.p0_linear = ConvBN(mid_channels_list[1] * 2, mid_channels_list[0], kernel_size=1, padding=0, relu6=relu6)
        self.p0_up = nn.ConvTranspose2d(mid_channels_list[0], mid_channels_list[0], kernel_size=2, stride=2, padding=0, bias=False)
        self.p0_fusion = Concat()

        self.out = ConvBN(mid_channels_list[0] * 2, out_channels, relu6=relu6)
    
    def forward(self, x):
        osa0 = self.osa0(x)
        osa1 = self.osa1(osa0)
        osa2 = self.osa2(osa1)

        p1 = self.p1_fusion(osa1, self.p1_up(self.p1_linear(osa2)))
        p0 = self.p0_fusion(osa0, self.p0_up(self.p0_linear(p1)))

        return self.out(p0)


@MODELS.register_module()
class VoxelEncoderLiteFPN(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels_list, 
                 out_channels,
                 repeats=3,
                 init_cfg=None,
                 relu6=False):
        super().__init__(init_cfg=init_cfg)
        assert len(mid_channels_list) == 3
        if type(repeats) is int:
            repeats = (repeats, repeats, repeats)
        else:
            assert len(repeats) == 3
        self.osa0 = OSABlock(in_channels, mid_channels_list[0], mid_channels_list[0], stride=1, repeat=repeats[0], final_dilation=2, relu6=relu6)
        self.osa1 = OSABlock(mid_channels_list[0], mid_channels_list[1], mid_channels_list[1], stride=2, repeat=repeats[1], final_dilation=2, relu6=relu6)
        self.osa1_1 = OSABlock(mid_channels_list[0], mid_channels_list[1], mid_channels_list[1], stride=1, repeat=repeats[1], final_dilation=2, relu6=relu6)
        self.osa2 = OSABlock(mid_channels_list[1], mid_channels_list[2], mid_channels_list[2], stride=2, repeat=repeats[2], final_dilation=2, relu6=relu6)
        self.osa2_1 = OSABlock(mid_channels_list[1], mid_channels_list[2], mid_channels_list[2], stride=1, repeat=repeats[2], final_dilation=2, relu6=relu6)

        self.p1_linear = ConvBN(mid_channels_list[2], mid_channels_list[1], kernel_size=1, padding=0, relu6=relu6)
        self.p1_up = nn.ConvTranspose2d(mid_channels_list[1], mid_channels_list[1], kernel_size=2, stride=2, padding=0, bias=False)
        self.p1_fusion = Concat()
        self.p0_linear = ConvBN(mid_channels_list[1] * 2, mid_channels_list[0], kernel_size=1, padding=0, relu6=relu6)
        self.p0_up = nn.ConvTranspose2d(mid_channels_list[0], mid_channels_list[0], kernel_size=2, stride=2, padding=0, bias=False)
        self.p0_fusion = Concat()

        self.out = ConvBN(mid_channels_list[0] * 2, out_channels, relu6=relu6)
    
    def forward(self, x):
        osa0 = self.osa0(x)
        osa1 = self.osa1_1(self.osa1(osa0))
        osa2 = self.osa2_1(self.osa2(osa1))

        p1 = self.p1_fusion(osa1, self.p1_up(self.p1_linear(osa2)))
        p0 = self.p0_fusion(osa0, self.p0_up(self.p0_linear(p1)))

        return self.out(p0)
    


@MODELS.register_module()
class PlanarHead(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
                 reg_scales=None,
                 repeat=3,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.seg_tower = OSABlock(in_channels, 
                                  mid_channels, 
                                  stride=1, 
                                  repeat=repeat, 
                                  with_reduce=False)
        self.reg_tower = OSABlock(in_channels, 
                                  mid_channels, 
                                  stride=1, 
                                  repeat=repeat, 
                                  with_reduce=False)
        self.cen_seg = nn.Conv2d(mid_channels * repeat, 
                                 cen_seg_channels, 
                                 kernel_size=1)
        self.reg = nn.Conv2d(mid_channels * repeat, 
                             reg_channels, 
                             kernel_size=1)
        self.has_reg_scale = reg_scales is not None
        if self.has_reg_scale:
            assert len(reg_scales) == reg_channels
            self.reg_scales = torch.tensor(reg_scales, dtype=torch.float32)[None, :, None, None]
    
    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        if self.has_reg_scale:
            reg = reg * self.reg_scales.to(reg.device)
        return cen_seg, reg


@MODELS.register_module()
class PlanarHeadSimple(BaseModule):
    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
                 reg_scales=None,
                 repeat=1,
                 init_cfg=None,
                 relu6=False):
        super().__init__(init_cfg=init_cfg)
        # self.seg_tower = nn.Sequential(
        #     ConvBN(in_channels, mid_channels, relu6=relu6),
        #     * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        # )
        self.seg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            *[ConvBN(mid_channels, mid_channels, relu6=relu6) for _ in range(repeat)],
        )

        # self.reg_tower = nn.Sequential(
        #     ConvBN(in_channels, mid_channels, relu6=relu6),
        #     * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        # )
        self.reg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            *[ConvBN(mid_channels, mid_channels, relu6=relu6) for _ in range(repeat)],
        )   

        self.cen_seg = nn.Conv2d(mid_channels, cen_seg_channels, kernel_size=1)
        self.reg = nn.Conv2d(mid_channels, reg_channels, kernel_size=1)
        self.has_reg_scale = reg_scales is not None
        if self.has_reg_scale:
            assert len(reg_scales) == reg_channels
            self.reg_scales = torch.tensor(reg_scales, dtype=torch.float32)[None, :, None, None]
    
    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        if self.has_reg_scale:
            reg = reg * self.reg_scales.to(reg.device)
        return cen_seg, reg


@MODELS.register_module()
class ParkingSlotHead(BaseModule):

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
                 reg_scales=None,
                 repeat=1,
                 init_cfg=None,
                 relu6=False):
        super().__init__(init_cfg=init_cfg)
        self.seg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        )
        self.reg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        )
        self.cen_seg = nn.Conv2d(mid_channels, cen_seg_channels, kernel_size=1)
        self.reg = nn.Conv2d(mid_channels, reg_channels, kernel_size=1)
        self.has_reg_scale = reg_scales is not None
        if self.has_reg_scale:
            assert len(reg_scales) == reg_channels
            self.reg_scales = torch.tensor(reg_scales, dtype=torch.float32)[None, :, None, None]

        self.up_sample= nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False)
        self.rates_res4 = [1, 2, 4, 8]
        # self.rates_res4 = [1, 1, 2, 2]
        self.aspp1 = ASPP_module(in_channels, 32, rate=self.rates_res4[0])
        self.aspp2 = ASPP_module(in_channels, 32, rate=self.rates_res4[1])
        self.aspp3 = ASPP_module(in_channels, 32, rate=self.rates_res4[2])
        self.aspp4 = ASPP_module(in_channels, 32, rate=self.rates_res4[3])
        self.concat = Concat()
        self.concat_conv  = nn.Conv2d(128, 64, kernel_size=3,stride=1, padding=1, bias=True)
        self.concat_bn    = nn.BatchNorm2d(64)
        self.concat_relu  = nn.ReLU6()

        self.last_conv= nn.Conv2d(64, 36, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv_bn=nn.BatchNorm2d(36)
        self.last_conv_relu  = nn.ReLU6()
        # self.last_up_sample= nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False)
        self.last_conv1_seg= nn.Conv2d(36, 24, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv1_seg_bn=nn.BatchNorm2d(24)
        self.last_conv1_seg_relu  = nn.ReLU6()
        self.last_conv_linear_seg= nn.Conv2d(24, 16, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv_linear_seg_bn=nn.BatchNorm2d(16)
        self.last_conv_linear_seg_relu  = nn.ReLU6()
        self.last_conv_seg = nn.Conv2d(16, 4, kernel_size=1, stride=1)

        self.last_conv1_pin= nn.Conv2d(36, 24, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv1_pin_bn=nn.BatchNorm2d(24)
        self.last_conv1_pin_relu  = nn.ReLU6()
        self.last_conv_linear_pin= nn.Conv2d(24, 16, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv_linear_pin_bn=nn.BatchNorm2d(16)
        self.last_conv_linear_pin_relu  = nn.ReLU6()
        self.last_conv_pin = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        

    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        if self.has_reg_scale:
            reg = reg * self.reg_scales.to(reg.device)

        # bev_feat=self.up_sample(x)
        # x1 = self.aspp1(bev_feat)
        # x2 = self.aspp2(bev_feat)
        # x3 = self.aspp3(bev_feat)
        # x4 = self.aspp4(bev_feat)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        parking_feat = self.concat(x1, x2, x3, x4)
        parking_feat = self.concat_conv(parking_feat)
        parking_feat = self.concat_bn(parking_feat)
        parking_feat = self.concat_relu(parking_feat)

        parking_feat=self.up_sample(parking_feat)

        parking_feat = self.last_conv(parking_feat)
        parking_feat = self.last_conv_bn(parking_feat)
        parking_feat = self.last_conv_relu(parking_feat)

        # parking_feat=self.last_up_sample(parking_feat)
        parking_feat_seg = self.last_conv1_seg(parking_feat)
        parking_feat_seg = self.last_conv1_seg_bn(parking_feat_seg)
        parking_feat_seg = self.last_conv1_seg_relu(parking_feat_seg)
        parking_feat_seg = self.last_conv_linear_seg(parking_feat_seg)
        parking_feat_seg = self.last_conv_linear_seg_bn(parking_feat_seg)
        parking_feat_seg = self.last_conv_linear_seg_relu(parking_feat_seg)
        pts = self.last_conv_seg(parking_feat_seg)
        # pts = torch.clamp(pts, min=-8, max=8)

        parking_feat_pin = self.last_conv1_pin(parking_feat)
        parking_feat_pin = self.last_conv1_pin_bn(parking_feat_pin)
        parking_feat_pin = self.last_conv1_pin_relu(parking_feat_pin)
        parking_feat_pin = self.last_conv_linear_pin(parking_feat_pin)
        parking_feat_pin = self.last_conv_linear_pin_bn(parking_feat_pin)
        parking_feat_pin = self.last_conv_linear_pin_relu(parking_feat_pin)
        pin = self.last_conv_pin(parking_feat_pin)
        # pin = torch.clamp(pin, min=-8, max=8)

        return cen_seg,reg,pts,pin
    

@MODELS.register_module()
class ParkingSlotHead_ven(BaseModule):

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def __init__(self, 
                 in_channels, 
                 mid_channels, 
                 cen_seg_channels,
                 reg_channels,
                 reg_scales=None,
                 repeat=1,
                 init_cfg=None,
                 relu6=False):
        super().__init__(init_cfg=init_cfg)
        # self.seg_tower = nn.Sequential(
        #     ConvBN(in_channels, mid_channels, relu6=relu6),
        #     * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        # )
        # self.reg_tower = nn.Sequential(
        #     ConvBN(in_channels, mid_channels, relu6=relu6),
        #     * ([ConvBN(mid_channels, mid_channels, relu6=relu6)] * repeat),
        # )
        self.seg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            *[ConvBN(mid_channels, mid_channels, relu6=relu6) for _ in range(repeat)],
        )
        self.reg_tower = nn.Sequential(
            ConvBN(in_channels, mid_channels, relu6=relu6),
            *[ConvBN(mid_channels, mid_channels, relu6=relu6) for _ in range(repeat)],
        )


        self.cen_seg = nn.Conv2d(mid_channels, cen_seg_channels, kernel_size=1)
        self.reg = nn.Conv2d(mid_channels, reg_channels, kernel_size=1)
        self.has_reg_scale = reg_scales is not None
        if self.has_reg_scale:
            assert len(reg_scales) == reg_channels
            self.reg_scales = torch.tensor(reg_scales, dtype=torch.float32)[None, :, None, None]

        
        self.rates_res4 = [1, 2, 4, 8]

        self.aspp1 = ASPP_module(in_channels, 32, rate=self.rates_res4[0])
        self.aspp2 = ASPP_module(in_channels, 32, rate=self.rates_res4[1])
        self.aspp3 = ASPP_module(in_channels, 32, rate=self.rates_res4[2])
        # self.aspp4 = ASPP_module(in_channels, 32, rate=self.rates_res4[3])
        self.concat = Concat()
        self.concat_conv  = nn.Conv2d(96, 32, kernel_size=3,stride=1, padding=1, bias=True)
        self.concat_bn    = nn.BatchNorm2d(32)
        self.concat_relu  = nn.ReLU6()

        self.last_conv= nn.Conv2d(32, 12, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv_bn=nn.BatchNorm2d(12)
        self.last_conv_relu  = nn.ReLU6()
        # self.up_sample= nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False)
        self.up_sample= nn.ConvTranspose2d(12, 12, kernel_size=2, stride=2, padding=0, bias=True)
        self.last_conv1_seg= nn.Conv2d(12, 12, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv1_seg_bn=nn.BatchNorm2d(12)
        self.last_conv1_seg_relu  = nn.ReLU6()
        self.last_conv_linear_seg= nn.Conv2d(12, 12, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv_linear_seg_bn=nn.BatchNorm2d(12)
        self.last_conv_linear_seg_relu  = nn.ReLU6()
        self.last_conv_seg = nn.Conv2d(12, 4, kernel_size=1, stride=1)

        self.last_conv1_pin= nn.Conv2d(12, 12, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv1_pin_bn=nn.BatchNorm2d(12)
        self.last_conv1_pin_relu  = nn.ReLU6()
        self.last_conv_linear_pin= nn.Conv2d(12, 12, kernel_size=3,stride=1, padding=1, bias=True)
        self.last_conv_linear_pin_bn=nn.BatchNorm2d(12)
        self.last_conv_linear_pin_relu  = nn.ReLU6()
        self.last_conv_pin = nn.Conv2d(12, 1, kernel_size=1, stride=1)
        

    def forward(self, x):
        seg_feat = self.seg_tower(x)
        cen_seg = self.cen_seg(seg_feat)
        reg_feat = self.reg_tower(x)
        reg = self.reg(reg_feat)
        if self.has_reg_scale:
            reg = reg * self.reg_scales.to(reg.device)

        # bev_feat=self.up_sample(x)
        # x1 = self.aspp1(bev_feat)
        # x2 = self.aspp2(bev_feat)
        # x3 = self.aspp3(bev_feat)
        # x4 = self.aspp4(bev_feat)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        # x4 = self.aspp4(x)

        # parking_feat = x1 + x2 + x3 + x4
        # parking_feat = x1 + x2 + x3
        parking_feat = self.concat(x1, x2, x3)
        parking_feat = self.concat_conv(parking_feat)
        parking_feat = self.concat_bn(parking_feat)
        parking_feat = self.concat_relu(parking_feat)

        parking_feat = self.last_conv(parking_feat)
        parking_feat = self.last_conv_bn(parking_feat)
        parking_feat = self.last_conv_relu(parking_feat)

        parking_feat=self.up_sample(parking_feat)



        # parking_feat=self.last_up_sample(parking_feat)
        parking_feat_seg = self.last_conv1_seg(parking_feat)
        parking_feat_seg = self.last_conv1_seg_bn(parking_feat_seg)
        parking_feat_seg = self.last_conv1_seg_relu(parking_feat_seg)
        parking_feat_seg = self.last_conv_linear_seg(parking_feat_seg)
        parking_feat_seg = self.last_conv_linear_seg_bn(parking_feat_seg)
        parking_feat_seg = self.last_conv_linear_seg_relu(parking_feat_seg)
        pts = self.last_conv_seg(parking_feat_seg)
        # pts = torch.clamp(pts, min=-8, max=8)

        parking_feat_pin = self.last_conv1_pin(parking_feat)
        parking_feat_pin = self.last_conv1_pin_bn(parking_feat_pin)
        parking_feat_pin = self.last_conv1_pin_relu(parking_feat_pin)
        parking_feat_pin = self.last_conv_linear_pin(parking_feat_pin)
        parking_feat_pin = self.last_conv_linear_pin_bn(parking_feat_pin)
        parking_feat_pin = self.last_conv_linear_pin_relu(parking_feat_pin)
        pin = self.last_conv_pin(parking_feat_pin)
        # pin = torch.clamp(pin, min=-8, max=8)

        return cen_seg,reg,pts,pin