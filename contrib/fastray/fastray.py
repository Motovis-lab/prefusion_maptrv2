
import torch
import torch.nn as nn

from functools import reduce

from torch import Tensor
from typing import Union, List, Dict, Optional

from mmengine.model import BaseModel, BaseModule

from prefusion.registry import MODELS
from prefusion.loss import PlanarLoss


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
    
    def forward(self, camera_feats, camera_lookups):
        '''Output a 3d voxel tensor from 2d image features
        
        Parameters
        ----------
        camera_feats : Dict[str, torch.Tensor]
            camera features of shape (N, C, H, W)
        camera_lookups : Dict[str, torch.Tensor]
            camera lookup tensors of shape (N, Z*X*Y)
        
        Returns
        -------
        voxel_feats : torch.Tensor
            voxel features of shape (N, C*Z, X, Y)
        
        '''
        pass



@MODELS.register_module()
class VoxelTemporalAlign(BaseModule):
    
    def __init__(self,
                 voxel_shape,
                 voxel_range,
                 approx_2d=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range
        self.approx_2d = approx_2d
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
        if self.approx_2d:
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
        if self.approx_2d:
            xx_egos = ego_points[:, 0]
            yy_egos = ego_points[:, 1]
            xx_ = xx_egos * fx + cx
            yy_ = yy_egos * fy + cy
            if normalize:
                xx_ = 2 * xx_ / X - 1
                yy_ = 2 * yy_ / Y - 1
            grid = torch.stack([
                xx_.reshape(N, X, Y), 
                yy_.reshape(N, X, Y)
            ], dim=-1)
        else:
            xx_egos = ego_points[:, 0]
            yy_egos = ego_points[:, 1]
            zz_egos = ego_points[:, 2]
            xx_ = xx_egos * fx + cx
            yy_ = yy_egos * fy + cy
            zz_ = zz_egos * fz + cz
            if normalize:
                xx_ = 2 * xx_ / X - 1
                yy_ = 2 * yy_ / Y - 1
                zz_ = 2 * zz_ / Z - 1
            grid = torch.stack([
                zz_.reshape(N, Z, X, Y),
                xx_.reshape(N, Z, X, Y),
                yy_.reshape(N, Z, X, Y)
            ], dim=-1)
        return grid
        
    
    def forward(self, voxel_feats_pre, delta_poses):
        '''
        Output a time-aligned voxel tensor from previous hidden voxel features.

        Parameters
        ----------
        voxel_feats_pre : torch.Tensor
            in shape of (N, C*Z, X, Y) or (N, C, Z, X, Y)
        
        delta_poses : torch.Tensor
            in shape of (N, 4, 4)
        '''
        # TODO: USE GRID-SAMPLING TO GET TIME-ALIGNED FEATURES!
        # gen grids
        if self.approx_2d:
            assert len(voxel_feats_pre.shape) == 4, 'must be 4-D Tensor'
            ego_points = self._unproject_points_from_voxel_to_ego()
            ego_points.to(voxel_feats_pre, non_blocking=True)
            rotation_2d = delta_poses[:, :2, :2]
            translation_2d = delta_poses[:, :2, 3]
            ego_points_ = rotation_2d @ ego_points[None] + translation_2d
            grid_2d = self._project_points_from_ego_to_voxel(ego_points_)
            voxel_feats_pre_aligned = nn.functional.grid_sample(
                input=voxel_feats_pre, 
                grid=grid_2d, 
                mode='bilinear', align_corners=False
            )
        else:
            assert len(voxel_feats_pre.shape) == 5, 'must be 5-D Tensor'
            ego_points



@MODELS.register_module()
class VoxelTemporalFusion(BaseModule):
    
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




@MODELS.register_module()
class VoxelHead(BaseModule):
    def __init__(self, voxel_feature_config, init_cfg=None):
        '''
        >>> voxel_feature_config = dict(
            voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
            voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
            ego_distance_max=40,
            ego_distance_step=2
        )
        '''
        super().__init__(init_cfg=init_cfg)
        self.voxel_feature_config = voxel_feature_config
    
    def forward(voxel_feats):
        """_summary_

        Parameters
        ----------
        curr_feats : _type_
            _description_
        prev_feats : _type_
            _description_
        """
        '''
        prev_feats: time-aligned history hidden features
        will output hidden features
        '''
        pass
        




@MODELS.register_module()
class FastRayPlanarStreamModel(BaseModel):
    
    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 heads,
                 loss_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_front = MODELS.build(backbones['pv_front'])
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        self.backbone_fisheyes = MODELS.build(backbones['fisheyes'])
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        self.voxel_fusion = MODELS.build(voxel_fusion)
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        self.head_polyline_3d = MODELS.build(heads['polyline_3d'])
        self.head_parkingslot_3d = MODELS.build(heads['parkingslot_3d'])
        self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # hidden voxel features for temporal fusion
        self.voxel_feats_pre = None
        # init losses, may use loss_cfg in the future
        # self.losses = MODELS.build(loss_cfg)
        self.loss_bbox_3d=PlanarLoss(
            loss_name_prefix='bbox_3d'
        )
        self.loss_polyline_3d=PlanarLoss(
            loss_name_prefix='polyline_3d'
        )
        self.loss_parkingslot_3d=PlanarLoss(
            loss_name_prefix='parkingslot_3d'
        )

    
    def forward(self, batched_input_dict, mode='tensor'):
        """
        >>> processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': {
                    'cam_0': {
                        uu:, (N, Z*X*Y),
                        vv:, (N, Z*X*Y),
                        dd:, (N, Z*X*Y),
                        ...
                    ...
                },
                'delta_poses': [],
                'annotations': {
                    'bbox_3d_0': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, C, X, Y)
                        'reg': (N, C, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        delta_poses = batched_input_dict['delta_poses']
        # backbone
        camera_feats = {}
        for cam_id in camera_tensors:
            if cam_id in self.camera_groups['pv_front']:
                camera_feats[cam_id] = self.backbone_pv_front(camera_tensors[cam_id])
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats[cam_id] = self.backbone_pv_sides(camera_tensors[cam_id])
            if cam_id in self.camera_groups['fisheyes']:
                camera_feats[cam_id] = self.backbone_fisheyes(camera_tensors[cam_id])
        # spatial transform
        voxel_feats_cur = self.spatial_transform(camera_feats, camera_lookups)
        # temporal transform
        if batched_input_dict['index_infos'][0].prev is None:
            self.voxel_feats_pre = voxel_feats_cur
        voxel_feats_pre_aligned = self.temporal_transform(self.voxel_feats_pre, delta_poses)
        # voxel fusion
        voxel_feats_updated = self.voxel_fusion(voxel_feats_cur, voxel_feats_pre_aligned)
        self.voxel_feats_pre = voxel_feats_updated
        # heads
        pred_bbox_3d = self.head_bbox_3d(voxel_feats_updated)
        pred_polyline_3d = self.head_polyline_3d(voxel_feats_updated)
        pred_parkingslot_3d = self.head_parkingslot_3d(voxel_feats_updated)
        
        if mode == 'tensor':
            return dict(hidden_feats=self.voxel_feats_pre,
                        pred_bbox_3d=pred_bbox_3d,
                        pred_polyline_3d=pred_polyline_3d,
                        pred_parkingslot_3d=pred_parkingslot_3d)
        if mode == 'loss':
            gt_bbox_3d = batched_input_dict['annotations']['bbox_3d']
            gt_polyline_3d = batched_input_dict['annotations']['polyline_3d']
            gt_parkingslot_3d = batched_input_dict['annotations']['parkingslot_3d']
            losses = {}
            losses.update(self.loss_bbox_3d(pred_bbox_3d, gt_bbox_3d))
            losses.update(self.loss_polyline_3d(pred_polyline_3d, gt_polyline_3d))
            losses.update(self.loss_parkingslot_3d(pred_parkingslot_3d, gt_parkingslot_3d))
            return losses
        
        if mode == 'predict':
            raise NotImplementedError

