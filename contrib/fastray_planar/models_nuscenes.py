import torch

from torch import Tensor
from typing import Union, List, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import BaseDataElement

from prefusion import BaseModel
from prefusion.registry import MODELS
from .model_utils import *


@MODELS.register_module()
class FastBEVModel(BaseModel):
    def __init__(self,
                 camera_groups: List,
                 backbones,
                 neck,
                 neck_fuse,
                 neck_3d,
                 spatial_transform,
                 heads,
                 loss_cfg=None,
                 debug_mode=False,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone = MODELS.build(backbones)
        self.neck = MODELS.build(neck)
        self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        # view transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.neck_3d = MODELS.build(neck_3d)
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # init losses
        self.losses_dict = {}
        for branch in loss_cfg:
            self.losses_dict[branch] = MODELS.build(loss_cfg[branch])

    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        # backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups:
                mlvl_feats = self.backbone(camera_tensors_dict[cam_id])
                mlvl_feats = self.neck(mlvl_feats) # normally it is FPN
                camera_feats_dict[cam_id] = self.fuse_multi_level_feats(mlvl_feats)

        # spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats = self.spatial_transform(camera_feats_dict, camera_lookups)

        # if self.debug_mode:
        #     draw_aligned_voxel_feats([voxel_feats])

        # voxel encoder
        voxel_feats = voxel_feats.permute(0, 1, 3, 4, 2)
        bev_feats = self.neck_3d(voxel_feats)[0]
        # heads
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        # outputs
        pred_dict = dict(
            bbox_3d = dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:9],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_cylinder = dict(
                cen=out_bbox_3d[0][:, 9:10],
                seg=out_bbox_3d[0][:, 10:12],
                reg=out_bbox_3d[1][:, 20:28]),
            bbox_3d_oriented_cylinder = dict(
                cen=out_bbox_3d[0][:, 12:13],
                seg=out_bbox_3d[0][:, 13:15],
                reg=out_bbox_3d[1][:, 28:41]),
            bbox_3d_rect_cuboid = dict(
                cen=out_bbox_3d[0][:, 15:16],
                seg=out_bbox_3d[0][:, 16:18],
                reg=out_bbox_3d[1][:, 41:55]),
        )

        if self.debug_mode:
            draw_outputs(pred_dict, batched_input_dict)

        if mode == 'tensor':
            return pred_dict
        if mode == 'loss':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict, batched_input_dict['index_infos'][0].frame_id)
            return losses

        if mode == 'predict':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def fuse_multi_level_feats(self, mlvl_feats, target_lvl=0):
        if target_lvl != 0:
            raise NotImplementedError

        fuse_feats = [mlvl_feats[target_lvl]]
        for i in range(target_lvl + 1, len(mlvl_feats)):
            resized_feat = F.interpolate(
                mlvl_feats[i], 
                size=mlvl_feats[target_lvl].size()[2:], 
                mode="bilinear", 
                align_corners=False)
            fuse_feats.append(resized_feat)
    
        if len(fuse_feats) > 1:
            fuse_feats = torch.cat(fuse_feats, dim=1)
        else:
            fuse_feats = fuse_feats[0]

        fuse_feats = self.neck_fuse(fuse_feats)
        return fuse_feats

    def compute_losses(self, gt_dict: Dict, pred_dict: Dict, ts='-1'):
        losses = dict(loss=0)
        for branch in self.losses_dict:
            losses_branch = self.losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            # losses.update(losses_branch)
            for loss_item, loss_value in losses_branch.items():
                if loss_item == branch + '_loss':
                    new_loss_name = f"{branch}/{loss_item[len(branch) + 1:]}"
                else:
                    parts = loss_item[len(branch) + 1:].split("_")
                    new_loss_name = f"{branch}/{parts[0]}/{'_'.join(parts[1:])}"
                losses.update({new_loss_name: loss_value})
        return losses



@MODELS.register_module()
class NuscenesFastRayPlanarSingleFrameModel(BaseModel):

    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 heads,
                 loss_cfg=None,
                 debug_mode=False,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        # view transform
        self.spatial_transform = MODELS.build(spatial_transform)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # init losses
        self.losses_dict = {}
        for branch in loss_cfg:
            self.losses_dict[branch] = MODELS.build(loss_cfg[branch])


    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        # backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
        # spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats = self.spatial_transform(camera_feats_dict, camera_lookups)

        # if self.debug_mode:
        #     draw_aligned_voxel_feats([voxel_feats])

        # voxel encoder
        if len(voxel_feats.shape) == 5:
            N, C, Z, X, Y = voxel_feats.shape
            voxel_feats = voxel_feats.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats)
        # heads
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        # outputs
        pred_dict = dict(
            bbox_3d = dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:9],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_cylinder = dict(
                cen=out_bbox_3d[0][:, 9:10],
                seg=out_bbox_3d[0][:, 10:12],
                reg=out_bbox_3d[1][:, 20:28]),
            bbox_3d_oriented_cylinder = dict(
                cen=out_bbox_3d[0][:, 12:13],
                seg=out_bbox_3d[0][:, 13:15],
                reg=out_bbox_3d[1][:, 28:41]),
            bbox_3d_rect_cuboid = dict(
                cen=out_bbox_3d[0][:, 15:16],
                seg=out_bbox_3d[0][:, 16:18],
                reg=out_bbox_3d[1][:, 41:55]),
        )

        if self.debug_mode:
            draw_outputs(pred_dict, batched_input_dict)

        if mode == 'tensor':
            return pred_dict
        if mode == 'loss':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict, batched_input_dict['index_infos'][0].frame_id)
            return losses

        if mode == 'predict':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt_dict: Dict, pred_dict: Dict, ts='-1'):
        losses = dict(loss=0)
        for branch in self.losses_dict:
            losses_branch = self.losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            # losses.update(losses_branch)
            for loss_item, loss_value in losses_branch.items():
                if loss_item == branch + '_loss':
                    new_loss_name = f"{branch}/{loss_item[len(branch) + 1:]}"
                else:
                    parts = loss_item[len(branch) + 1:].split("_")
                    new_loss_name = f"{branch}/{parts[0]}/{'_'.join(parts[1:])}"
                losses.update({new_loss_name: loss_value})
        return losses


@MODELS.register_module()
class NuscenesFastRayPlanarMultiFrameModel(BaseModel):

    def __init__(self,
                 camera_groups,
                 backbones,
                 spatial_transform,
                 temporal_transform,
                 voxel_fusion,
                 heads,
                 debug_mode=False,
                 loss_cfg=None,
                 pre_nframes=1,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor, init_cfg)
        self.debug_mode = debug_mode
        # backbone
        self.camera_groups = camera_groups
        self.backbone_pv_sides = MODELS.build(backbones['pv_sides'])
        # view transform and temporal transform
        self.spatial_transform = MODELS.build(spatial_transform)
        self.temporal_transform = MODELS.build(temporal_transform)
        # voxel fusion
        # self.voxel_fusion = EltwiseAdd()
        self.voxel_fusion = MODELS.build(voxel_fusion)
        # voxel encoder
        self.voxel_encoder = MODELS.build(heads['voxel_encoder'])
        # voxel heads
        self.head_bbox_3d = MODELS.build(heads['bbox_3d'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # hidden voxel features for temporal fusion
        self.pre_nframes = pre_nframes
        self.cached_voxel_feats = {}
        self.cached_delta_poses = {}
        # init losses
        self.losses_dict = {}
        for branch in loss_cfg:
            self.losses_dict[branch] = MODELS.build(loss_cfg[branch])


    def forward(self, mode='tensor', **batched_input_dict):
        """
        >>> batched_input_dict = processed_frame_batch = {
                'index_infos': [index_info, index_info, ...],
                'camera_images': {
                    'cam_0': (N, 3, H, W),
                    ...
                },
                'camera_lookups': [
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                    {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                ],
                'delta_poses': (N, 4, 4),
                'annotations': {
                    'bbox_3d': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, V, X, Y)
                        'reg': (N, V, X, Y)
                    },
                    ...
                },
                ...
            }
        """
        camera_tensors_dict = batched_input_dict['camera_tensors']
        camera_lookups = batched_input_dict['camera_lookups']
        delta_poses = batched_input_dict['delta_poses']
        ## backbone
        camera_feats_dict = {}
        for cam_id in camera_tensors_dict:
            if cam_id in self.camera_groups['pv_sides']:
                camera_feats_dict[cam_id] = self.backbone_pv_sides(camera_tensors_dict[cam_id])
        ## spatial transform: output shape can be 4D or 5D (N, C*Z, X, Y) or (N, C, Z, X, Y)
        voxel_feats_cur = self.spatial_transform(camera_feats_dict, camera_lookups)
        ## temporal transform
        cur_first_index_info = batched_input_dict['index_infos'][0]
        # init tmp pre_0 cache
        self.cached_voxel_feats[f'pre_0'] = voxel_feats_cur
        self.cached_delta_poses[f'pre_0'] = delta_poses
        for pre_i in range(self.pre_nframes):
            index_info_prev_str = 'cur_first_index_info' + ''.join(['.prev'] * (pre_i+1))
            if eval(index_info_prev_str) is None:
                for pre_j in range(pre_i, self.pre_nframes):
                    self.cached_voxel_feats[f'pre_{pre_j+1}'] = self.cached_voxel_feats[f'pre_{pre_j}'].clone().detach()
                    self.cached_delta_poses[f'pre_{pre_j+1}'] = self.cached_delta_poses[f'pre_{pre_j}'].clone().detach()
                break
        # cache delta_poses
        for pre_i in range(self.pre_nframes, 1, -1):
            self.cached_delta_poses[f'pre_{pre_i}'] = (self.cached_delta_poses[f'pre_{pre_i-1}'] @ delta_poses).clone().detach()
        self.cached_delta_poses['pre_1'] = delta_poses.clone().detach()
        # align all history frames to current frame
        aligned_voxel_feats_cat = [voxel_feats_cur]
        for pre_i in range(self.pre_nframes):
            aligned_voxel_feats_cat.append(self.temporal_transform(
                self.cached_voxel_feats[f'pre_{pre_i+1}'], self.cached_delta_poses[f'pre_{pre_i+1}']
            ))
        # if self.debug_mode:
        #     draw_aligned_voxel_feats(aligned_voxel_feats_cat)
            # draw_aligned_voxel_feats(list(self.cached_voxel_feats.values()))
        # cache voxel features
        for pre_i in range(self.pre_nframes, 0, -1):
            self.cached_voxel_feats[f'pre_{pre_i}'] = (self.cached_voxel_feats[f'pre_{pre_i-1}']).clone().detach()
        # pop tmp pre_0 cache
        self.cached_voxel_feats.pop('pre_0')
        self.cached_delta_poses.pop('pre_0')
        ## voxel fusion
        voxel_feats_fused = self.voxel_fusion(*aligned_voxel_feats_cat)
        ## voxel encoder
        if len(voxel_feats_fused.shape) == 5:
            N, C, Z, X, Y = voxel_feats_fused.shape
            voxel_feats_fused = voxel_feats_fused.reshape(N, C*Z, X, Y)
        bev_feats = self.voxel_encoder(voxel_feats_fused)
        ## heads & outputs
        out_bbox_3d = self.head_bbox_3d(bev_feats)
        pred_dict = dict(
            bbox_3d = dict(
                cen=out_bbox_3d[0][:, 0:1],
                seg=out_bbox_3d[0][:, 1:9],
                reg=out_bbox_3d[1][:, 0:20]),
            bbox_3d_cylinder = dict(
                cen=out_bbox_3d[0][:, 9:10],
                seg=out_bbox_3d[0][:, 10:12],
                reg=out_bbox_3d[1][:, 20:28]),
            bbox_3d_oriented_cylinder = dict(
                cen=out_bbox_3d[0][:, 12:13],
                seg=out_bbox_3d[0][:, 13:15],
                reg=out_bbox_3d[1][:, 28:41]),
            bbox_3d_rect_cuboid = dict(
                cen=out_bbox_3d[0][:, 15:16],
                seg=out_bbox_3d[0][:, 16:18],
                reg=out_bbox_3d[1][:, 41:55]),
        )
        

        if self.debug_mode:
            draw_outputs(pred_dict, batched_input_dict)

        if mode == 'tensor':
            return pred_dict
        if mode == 'loss':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict, batched_input_dict['index_infos'][0].frame_id)
            return losses

        if mode == 'predict':
            losses = self.compute_losses(batched_input_dict['annotations'], pred_dict)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred_dict.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt_dict: Dict, pred_dict: Dict, ts='-1'):
        losses = dict(loss=0)
        for branch in self.losses_dict:
            losses_branch = self.losses_dict[branch](pred_dict[branch], gt_dict[branch])
            losses['loss'] += losses_branch[branch + '_loss']
            # losses.update(losses_branch)
            for loss_item, loss_value in losses_branch.items():
                if loss_item == branch + '_loss':
                    new_loss_name = f"{branch}/{loss_item[len(branch) + 1:]}"
                else:
                    parts = loss_item[len(branch) + 1:].split("_")
                    new_loss_name = f"{branch}/{parts[0]}/{'_'.join(parts[1:])}"
                losses.update({new_loss_name: loss_value})
            
                # compute the norm of grad
                # _grad_norm = compute_grad_norm(self, loss_value)
                # losses.update({f"grad_norm/{new_loss_name}": _grad_norm})
        
        # import pandas as pd
        # loss_list = sorted([(k, v.item()) for k, v in losses.items() if k != "loss" and not k.startswith("grad_norm")], key=lambda x: x[0])
        # grad_list = sorted([(k, v.item()) for k, v in losses.items() if k.startswith("grad_norm")], key=lambda x: x[0])
        # df = pd.DataFrame([(l[0], l[1], g[1]) for l, g in zip(loss_list, grad_list)], columns=['item_name', 'loss', 'gradnorm'])
        # df.to_csv(f"loss_and_grad_{ts}.csv", index=False)
        # print(f"loss_and_grad.csv saved.")
        return losses


def compute_grad_norm(model, loss):
    loss.backward(retain_graph=True)
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    model.zero_grad()
    return torch.tensor(total_norm)