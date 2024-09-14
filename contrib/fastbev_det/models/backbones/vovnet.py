import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer, build_activation_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from prefusion.registry import MODELS
import torch
import collections
from torch import nn

# The paper is unclear as to where to downsample, so the downsampling was
# derived from the pretrained model graph as visualized by Netron. V2 simply
# enables ESE and identity connections here, nothing else changes.
CONFIG = {
    # Introduced in V2. Difference is 3 repeats instead of 5 within each block.
    "vovnet19": [
        # kernel size, inner channels, layer repeats, output channels, downsample
        [3, 64, 3, 128, True],
        [3, 80, 3, 256, True],
        [3, 96, 3, 348, True],
        [3, 112, 3, 512, True],
    ],
    "vovnet27_slim": [
        [3, 64, 5, 128, True],
        [3, 80, 5, 256, True],
        [3, 96, 5, 348, True],
        [3, 112, 5, 512, True],
    ],
    "vovnet39": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],
        [3, 192, 5, 768, True],  # x2
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x2
        [3, 224, 5, 1024, False],
    ],
    "vovnet57": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],
        [3, 192, 5, 768, True],  # x4
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x3
        [3, 224, 5, 1024, False],
        [3, 224, 5, 1024, False],
    ],
    "vovnet99": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],  # x3
        [3, 160, 5, 512, False],
        [3, 160, 5, 512, False],
        [3, 192, 5, 768, True],  # x9
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x3
        [3, 224, 5, 1024, False],
        [3, 224, 5, 1024, False],
    ],
}

class ESE(BaseModule):
    def __init__(self, channels,conv_cfg=None, init_cfg: dict | torch.List[dict] | None = None):
        super().__init__(init_cfg)
        self.conv = build_conv_layer(conv_cfg, channels, channels, 1, bias=True)

    def forward(self, x):
        y = x.mean([2, 3], keepdim=True)
        y = self.conv(y)
        # Hard sigmoid multiplied by input.
        return x * (nn.functional.relu6(y + 3, inplace=True) / 6.0)  # type: ignore


class ConvBnRelu(BaseModule):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.conv = build_conv_layer(conv_cfg, in_ch, out_ch, kernel_size, stride=stride,
                                     padding=kernel_size//2, bias=False)
        self.norm_name, self.norm  = build_norm_layer(norm_cfg, out_ch)
        self.relu = build_activation_layer(act_cfg)

    @property
    def norm(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm_name)

    def forward(self, x):
        """Forward function."""
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        return out


class OSA(BaseModule):
    def __init__(
        self,
        in_ch: int,
        inner_ch: int,
        out_ch: int,
        repeats: int = 5,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: bool = False,
        init_cfg=None    # type: dict   
    ) -> None:
        super().__init__(init_cfg)
        self.downsample = downsample
        self.layers = nn.ModuleList(
            [
                ConvBnRelu(
                    in_ch if r == 0 else inner_ch,
                    inner_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for r in range(repeats)
            ]
        )
        self.exit_conv = ConvBnRelu(
            in_ch + repeats * inner_ch, out_ch, kernel_size=1)
        self.ese = ESE(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through all modules, but retain outputs.
        input = x
        if self.downsample:
            x = nn.functional.max_pool2d(x, 3, stride=2, padding=1)
        features = [x]
        for l in self.layers:
            features.append(l(x))
            x = features[-1]
        x = torch.cat(features, dim=1)
        x = self.exit_conv(x)
        x = self.ese(x)
        # All non-downsampling V2 layers have a residual. They also happen to
        # not change the number of channels.
        if not self.downsample:
            x += input
        return x



@MODELS.register_module()
class VoVNet(BaseModule):
    def __init__(
        self,
        model_type: str = "vovnet39",
        out_indices=(0, 1, 2, 3),
        in_ch: int = 3,
        base_channels=64,
        init_cfg=None
    ):
        """ Usage:
        >>> net = VoVNet(3, 1000)
        >>> net = net.eval()
        >>> with torch.no_grad():
        ...     y = net(torch.rand(2, 3, 64, 64))
        >>> print(list(y.shape))
        [2, 1000]
        """
        super().__init__(init_cfg)

        # Input stage.
        conf = CONFIG[model_type]
        self.out_indices = out_indices
        assert max(out_indices) < len(conf)
        self.stem = nn.Sequential(
            ConvBnRelu(in_ch, base_channels, kernel_size=3, stride=2),
            ConvBnRelu(base_channels, base_channels, kernel_size=3, stride=1),
            ConvBnRelu(base_channels, base_channels*2, kernel_size=3, stride=1),
        )

        body_layers = collections.OrderedDict()
        
        in_ch = base_channels * 2
        for idx, block in enumerate(conf):
            kernel_size, inner_ch, repeats, out_ch, downsample = block
            body_layers[f"osa{idx}"] = OSA(
                in_ch,
                inner_ch,
                out_ch,
                repeats=repeats,
                kernel_size=kernel_size,
                downsample=downsample,
            )
            in_ch = out_ch
        self.body = nn.Sequential(body_layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        outs = []
        for idx, module in enumerate(self.body):
            y = module(y)
            if idx in self.out_indices:
                outs.append(y)

        return outs
