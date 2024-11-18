
import torch
import torch.nn as nn

from functools import reduce

from torch import Tensor
from typing import Union, List, Dict, Optional

from mmengine.model import BaseModel, BaseModule
from mmengine.structures import BaseDataElement

from prefusion.registry import MODELS

from .model_utils import *



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
        self.head_bbox_3d_cylinder = MODELS.build(heads['bbox_3d_cylinder'])
        self.head_bbox_3d_oriented_cylinder = MODELS.build(heads['bbox_3d_oriented_cylinder'])
        self.head_bbox_3d_rect_cuboid = MODELS.build(heads['bbox_3d_rect_cuboid'])
        # self.head_occ_sdf = MODELS.build(heads['occ_sdf'])
        # init losses
        self.loss_bbox_3d = MODELS.build(loss_cfg['bbox_3d'])
        self.loss_bbox_3d_cylinder = MODELS.build(loss_cfg['bbox_3d_cylinder'])
        self.loss_bbox_3d_oriented_cylinder = MODELS.build(loss_cfg['bbox_3d_oriented_cylinder'])
        self.loss_bbox_3d_rect_cuboid = MODELS.build(loss_cfg['bbox_3d_rect_cuboid'])

    
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
        out_bbox_3d_cylinder = self.head_bbox_3d_cylinder(bev_feats)
        out_bbox_3d_oriented_cylinder = self.head_bbox_3d_oriented_cylinder(bev_feats)
        out_bbox_3d_rect_cuboid = self.head_bbox_3d_rect_cuboid(bev_feats)
        # outputs
        pred_bbox_3d = dict(
            cen=out_bbox_3d[0][:, 0:1],
            seg=out_bbox_3d[0][:, 1:],
            reg=out_bbox_3d[1])
        pred_bbox_3d_cylinder = dict(
            cen=out_bbox_3d_cylinder[0][:, 0:1],
            seg=out_bbox_3d_cylinder[0][:, 1:],
            reg=out_bbox_3d_cylinder[1])
        pred_bbox_3d_oriented_cylinder = dict(
            cen=out_bbox_3d_oriented_cylinder[0][:, 0:1],
            seg=out_bbox_3d_oriented_cylinder[0][:, 1:],
            reg=out_bbox_3d_oriented_cylinder[1])
        pred_bbox_3d_rect_cuboid = dict(
            cen=out_bbox_3d_rect_cuboid[0][:, 0:1],
            seg=out_bbox_3d_rect_cuboid[0][:, 1:],
            reg=out_bbox_3d_rect_cuboid[1])

        if self.debug_mode:
            draw_out_feats(batched_input_dict, 
                           camera_tensors_dict,
                           pred_bbox_3d,)
        
        if mode == 'tensor':
            return dict(
                hidden_feats=self.voxel_feats_pre,
                pred_bbox_3d=pred_bbox_3d,
                pred_bbox_3d_cylinder=pred_bbox_3d_cylinder,
                pred_bbox_3d_oriented_cylinder=pred_bbox_3d_oriented_cylinder,
                pred_bbox_3d_rect_cuboid=pred_bbox_3d_rect_cuboid,
            )
        if mode == 'loss':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return losses

        if mode == 'predict':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt: Dict, pred: Dict):
        loss_bbox_3d = self.loss_bbox_3d(pred["bbox_3d"], gt['bbox_3d'])
        loss_bbox_3d_cylinder = self.loss_bbox_3d_cylinder(pred["bbox_3d_cylinder"], gt["bbox_3d_cylinder"])
        loss_bbox_3d_oriented_cylinder = self.loss_bbox_3d_oriented_cylinder(pred["bbox_3d_oriented_cylinder"], gt["bbox_3d_oriented_cylinder"])
        loss_bbox_3d_rect_cuboid = self.loss_bbox_3d_rect_cuboid(pred["bbox_3d_rect_cuboid"], gt["bbox_3d_rect_cuboid"])

        total_loss = sum([
            loss_bbox_3d['bbox_3d_loss'],
            loss_bbox_3d_cylinder['bbox_3d_cylinder_loss'],
            loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_loss'],
            loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_loss'],
        ])

        losses = dict(
            loss=total_loss,
            seg_iou_loss_bbox_3d=loss_bbox_3d['bbox_3d_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_cylinder=loss_bbox_3d_cylinder['bbox_3d_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_oriented_cylinder=loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_rect_cuboid=loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_seg_iou_0_loss'],
        )

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
        self.loss_bbox_3d = MODELS.build(loss_cfg['bbox_3d'])
        self.loss_bbox_3d_cylinder = MODELS.build(loss_cfg['bbox_3d_cylinder'])
        self.loss_bbox_3d_oriented_cylinder = MODELS.build(loss_cfg['bbox_3d_oriented_cylinder'])
        self.loss_bbox_3d_rect_cuboid = MODELS.build(loss_cfg['bbox_3d_rect_cuboid'])

    
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
        if self.debug_mode:
            draw_aligned_voxel_feats(aligned_voxel_feats_cat)
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
        pred_bbox_3d = dict(
            cen=out_bbox_3d[0][:, 0:1],
            seg=out_bbox_3d[0][:, 1:9],
            reg=out_bbox_3d[1][:, 0:20])
        pred_bbox_3d_cylinder = dict(
            cen=out_bbox_3d[0][:, 9:10],
            seg=out_bbox_3d[0][:, 10:12],
            reg=out_bbox_3d[1][:, 20:28])
        pred_bbox_3d_oriented_cylinder = dict(
            cen=out_bbox_3d[0][:, 12:13],
            seg=out_bbox_3d[0][:, 13:15],
            reg=out_bbox_3d[1][:, 28:41])
        pred_bbox_3d_rect_cuboid = dict(
            cen=out_bbox_3d[0][:, 15:16],
            seg=out_bbox_3d[0][:, 16:18],
            reg=out_bbox_3d[1][:, 41:55])

        if self.debug_mode:
            draw_out_feats(batched_input_dict, 
                           camera_tensors_dict,
                           pred_bbox_3d)
        
        if mode == 'tensor':
            return dict(
                pred_bbox_3d=pred_bbox_3d,
                pred_bbox_3d_cylinder=pred_bbox_3d_cylinder,
                pred_bbox_3d_oriented_cylinder=pred_bbox_3d_oriented_cylinder,
                pred_bbox_3d_rect_cuboid=pred_bbox_3d_rect_cuboid,
            )
        if mode == 'loss':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return losses

        if mode == 'predict':
            gt = batched_input_dict['annotations']
            pred = {
                "bbox_3d": pred_bbox_3d,
                "bbox_3d_cylinder": pred_bbox_3d_cylinder,
                "bbox_3d_oriented_cylinder": pred_bbox_3d_oriented_cylinder,
                "bbox_3d_rect_cuboid": pred_bbox_3d_rect_cuboid,
            }
            losses = self.compute_losses(gt, pred)
            return (
                *[{trsfmbl_name: {t: v.cpu() for t, v in _pred.items()}} for trsfmbl_name, _pred in pred.items()],
                BaseDataElement(loss=losses),
            )

    def compute_losses(self, gt: Dict, pred: Dict):
        loss_bbox_3d = self.loss_bbox_3d(pred["bbox_3d"], gt['bbox_3d'])
        loss_bbox_3d_cylinder = self.loss_bbox_3d_cylinder(pred["bbox_3d_cylinder"], gt["bbox_3d_cylinder"])
        loss_bbox_3d_oriented_cylinder = self.loss_bbox_3d_oriented_cylinder(pred["bbox_3d_oriented_cylinder"], gt["bbox_3d_oriented_cylinder"])
        loss_bbox_3d_rect_cuboid = self.loss_bbox_3d_rect_cuboid(pred["bbox_3d_rect_cuboid"], gt["bbox_3d_rect_cuboid"])

        total_loss = sum([
            loss_bbox_3d['bbox_3d_loss'],
            loss_bbox_3d_cylinder['bbox_3d_cylinder_loss'],
            loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_loss'],
            loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_loss'],
        ])

        losses = dict(
            loss=total_loss,
            seg_iou_loss_bbox_3d=loss_bbox_3d['bbox_3d_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_cylinder=loss_bbox_3d_cylinder['bbox_3d_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_oriented_cylinder=loss_bbox_3d_oriented_cylinder['bbox_3d_oriented_cylinder_seg_iou_0_loss'],
            seg_iou_loss_bbox_3d_rect_cuboid=loss_bbox_3d_rect_cuboid['bbox_3d_rect_cuboid_seg_iou_0_loss'],
        )

        return losses
    
