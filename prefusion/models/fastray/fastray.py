
import torch
import torch.nn as nn

from functools import reduce

from mmengine.registry import MODELS
from mmengine.model import BaseDataPreprocessor, BaseModel, BaseModule


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 has_relu=True):
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


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


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
                 out_channels,
                 stride=2,
                 dilation=1,
                 repeat=5,
                 final_dilation=1,
                 with_reduce=True,
                 has_bn=True):
        super(OSABlock, self).__init__()
        assert stride in [1, 2]
        assert repeat >= 2
        self.repeat = repeat
        self.with_reduce = with_reduce

        self.conv1 = ConvBN(in_channels, mid_channels, stride=stride, padding=dilation, dilation=dilation)

        for i in range(repeat - 2):
            self._modules['conv{}'.format(i + 2)] = ConvBN(
                mid_channels, mid_channels, padding=dilation, dilation=dilation
            )

        self._modules['conv{}'.format(repeat)] = ConvBN(
            mid_channels, mid_channels, padding=final_dilation, dilation=final_dilation
        )

        self.concat = Concat()
        if with_reduce:
            if has_bn:
                self.reduce = ConvBN(mid_channels * repeat, out_channels, kernel_size=1, padding=0)
            else:
                self.reduce = nn.Conv2d(mid_channels * repeat, out_channels, kernel_size=1, padding=0)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

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
    def __init__(self, out_stride=8, out_feature_channels=128, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.strides = [8, 16, 32]
        assert out_stride in self.strides
        self.out_stride = out_stride

        # BACKBONE
        self.stem1 = ConvBN(3, 64, stride=2)
        self.osa2 = OSABlock(64, 64, 96, stride=2, repeat=3)
        self.osa3 = OSABlock(96, 96, 128, stride=2, repeat=4, final_dilation=2)
        self.osa4 = OSABlock(128, 128, 192, stride=2, repeat=5, final_dilation=2)
        self.osa5 = OSABlock(192, 192, 192, stride=2, repeat=4, final_dilation=2)

        # NECK
        if self.out_stride <= 16:
            self.p4_up = nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, padding=0, bias=False)
            self.p4_fusion = Concat()
        elif self.out_stride <= 8:
            self.p3_linear = ConvBN(384, 128, kernel_size=1, padding=0)
            self.p3_up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
            self.p3_fusion = Concat()
        
        in_channels = {8: 256, 16: 384, 32: 192}
        mid_channels = {8: 96, 16: 128, 32: 192}
        self.out = OSABlock(
            in_channels[self.out_stride], mid_channels[self.out_stride], out_feature_channels,
            stride=1, repeat=3, has_bn=False
        )
        
        
    def forward(self, x):  # x: (N, 3, H, W)
        stem1 = self.stem1(x)
        osa2 = self.osa2(stem1)
        osa3 = self.osa3(osa2)
        osa4 = self.osa4(osa3)
        osa5 = self.osa5(osa4)

        if self.out_stride <= 32:
            out = osa5
        if self.out_stride <= 16:
            out = self.p4_fusion(self.p4_up(out), osa4)
        if self.out_stride <= 8:
            out = self.p3_fusion(self.p3_up(self.p3_linear(out)), osa3)
        
        out = self.out(out)
        
        return out



@MODELS.register_module()
class FastRaySpatialTransform(BaseModule):
    
    def forward(image_feats, LUTS):
        '''
        output a 3d voxel tensor from 2d image features
        '''
        pass



@MODELS.register_module()
class TemporalTransform(BaseModule):
    
    def forward(self, hidden_feats, ego_poses):
        '''
        output a time-aligned voxel tensor from previous hidden voxel features.
        '''
        # TODO: USE GRID-SAMPLING TO GET TIME-ALIGNED FEATURES!
        pass





@MODELS.register_module()
class VoxelHead(BaseModule):
    def __init__(self, voxel_feature_config, init_cfg=None):
        '''
        voxel_feature_config = dict(
            voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
            voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
            ego_distance_max=40,
            ego_distance_step=2
        )
        '''
        super().__init__(init_cfg=init_cfg)
        self.voxel_feature_config = voxel_feature_config
    
    def forward(curr_feats, prev_feats):
        '''
        prev_feats: time-aligned history hidden features
        will output hidden features
        '''
        pass
        




@MODELS.register_module()
class FastRayDataProcessor(BaseDataPreprocessor):

    def forward(self, frame_batch: list, training: bool = False) -> dict | list:
        """
        Parameters
        ----------
        frame_batch : list
            list of input_dicts
        training : bool, optional
            _description_, by default False

        Returns
        -------
        dict | list
            _description_

        Notes
        -----
        ```
        input_dict = {
            'scene_id': scene_id,
            'frame_id': frame_id,
            'prev_exists': True,
            'next_exists': True,
            'transformables': {}
        }
        batch_input_dict ={
            'scene_ids': [scene_id, scene_id, ...],
            'frame_ids': [frame_id, frame_id, ...],
            'prev_exists': [True, True, ...],
            'next_exists': [True, True, ...],
            'transformables': {
                'lidar_points': [lidar_points, lidar_points, ...],
                'camera_image_batches': {
                    cam_id: (N, 3, H, W),
                    cam_id: (N, 3, H, W)
                }
            }
            'LUTS': [LUT, LUT, ...],
        }
        ```
        """
        for input_dict in frame_batch:
            pass




@MODELS.register_module()
class FastRaySpatialTemporalFusion(BaseModel):
    
    def __init__(self,
                 backbones,
                 heads,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(FastRaySpatialTemporalFusion, self).__init__(data_preprocessor, init_cfg)
        
        self.hidden_feats = None

        self.backbone_pv_front = MODELS.build(backbones['pv_front'])
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        self.backbone_fisheyes = MODELS.build(backbones['fisheyes'])

        self.head_unified = MODELS.build(heads['unified'])
