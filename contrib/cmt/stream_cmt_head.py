# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from distutils.command.build import build
import enum
from turtle import down
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import NormedLinear
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
from mmdet3d.models import builder
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from einops import rearrange
import collections

from functools import reduce, partial

from scipy.optimize import curve_fit
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .cmt_head import CmtFisheyeHead, pos2embed
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out


@HEADS.register_module()
class StreamCmtFisheyeHead(CmtFisheyeHead):

    def __init__(self,
                 in_channels,
                 num_query=900,
                 hidden_dim=128,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),
                 tasks=[
                     dict(num_class=1, class_names=['car']),
                     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                     dict(num_class=2, class_names=['bus', 'trailer']),
                     dict(num_class=1, class_names=['barrier']),
                     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                     type="L1Loss",
                     reduction="mean",
                     loss_weight=0.25,
                 ),
                 loss_heatmap=dict(
                     type="GaussianFocalLoss",
                     reduction="mean"
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None
        super().__init__(in_channels, num_query, hidden_dim, depth_num, norm_bbox, downsample_scale, scalar, noise_scale, noise_trans,
                         dn_weight, split, train_cfg, test_cfg, common_heads, tasks, transformer, bbox_coder, loss_cls, loss_bbox,
                         loss_heatmap, separate_head, init_cfg, **kwargs)

        self.num_propagated = 256
        self.topk_proposals = 256
        self.memory_len = 1024
        self.embed_dims = 256
        if self.num_propagated > 0:
            # self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)
            # self.pseudo_reference_points = nn.Parameter(torch.ones(self.num_propagated, 3), requires_grad=False)
            # nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            # self.pseudo_reference_points.weight.requires_grad = False

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.with_ego_pos = True
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )
        # self.init_weights()
        self.reset_memory()

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def init_weights(self):
        super().init_weights()
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
        self.pseudo_reference_points.weight.requires_grad = False

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]  # 这个multi_apply时序吗
        forward_function = self.forward_single
        return multi_apply(forward_function, pts_feats, img_feats, img_metas)

    def temporal_alignment(self, query_pos, tgt, reference_points):  # ONLY DO MLN
        B = query_pos.size(0)
        # not clean
        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (
                self.pc_range[3:6] - self.pc_range[0:3])  # 1024, memory
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point))  # 1024, memory
        temp_memory = self.memory_embedding  # 1024, memory
        # rec中包含上一帧的256吗。在最后处理
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)

        if True:  # 当前帧的query做MLN
            rec_ego_motion = torch.cat(
                [torch.zeros_like(reference_points[..., :3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)  # 180
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)  # todo 鬼打墙了，问题在tgt上。tgt有啥问题
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)

            memory_ego_motion = torch.cat(
                [self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)  # temporal的MLN
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)
        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[..., :1])))  # rec 时序加0
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())  # query pos 时序加时间戳的变化

        # TODO: tgt加上上一帧的变化
        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[
                1] + self.num_propagated, 1, 1)  # 为什么还要多个256
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose


    def forward_single(self, x, x_img, img_metas):
        # 这里头还是有一个batch_size
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        use_a = True
        if use_a:
            self.pre_update_memory(img_metas, x)

        ret_dicts = []
        x = self.shared_conv(x)

        # generate reference point from ori 900 + 10 * max_num_of_bbx
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)

        # ----- img feat pos_emb
        rv_pos_embeds, feat_valid = self._rv_pe_fisheye(x_img, img_metas)  # 相机的position_embedding 主要修改应该在这边
        # ------ lidar feat pos_emb
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))

        # query 从reference point中得到query的信息
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds  # query_positions
        query_tgt = torch.zeros_like(query_embeds).to(query_embeds)
        if use_a:
            # 将 query: query_embeds; query_pos
            tgt, tgt_query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_embeds, query_tgt,
                                                                                                                reference_points)
            if self.training:
                num_mask = mask_dict['pad_size']
                attn_mask_expand = torch.zeros(tgt.shape[1], tgt.shape[1] + temp_memory.shape[1]).to(attn_mask)
                attn_mask_expand[:attn_mask.shape[0], :attn_mask.shape[1]] = attn_mask
                attn_mask_expand[attn_mask.shape[0]:, :num_mask] = 1  # 这一部分都要被mask掉
            else:
                attn_mask_expand = None

            outs_dec, _ = self.transformer(
                x, x_img, tgt_query_pos, tgt_query_pos,
                bev_pos_embeds, rv_pos_embeds,
                temp_memory=temp_memory, temp_pos=temp_pos,
                attn_masks=attn_mask_expand
            )
        else:
            outs_dec, _ = self.transformer(
                x, x_img, query_tgt,
                bev_pos_embeds, rv_pos_embeds,
                attn_masks=attn_mask
            )
        # 6 2 1260 256
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())

        flag = 0
        outputs_classes, outputs_coords = [], []
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label = task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            outputs_classes.append(outs['cls_logits'])
            outputs_coords.append(torch.cat([outs['center'], outs['height'], outs['vel']], dim=-1))  # 'dim', 'rot', 'vel'
            ret_dicts.append(outs)
        all_cls_scores = outputs_classes[0]
        all_bbox_preds = outputs_coords[0]
        all_bbox_preds[..., 0:3] = (
                all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        if use_a:
            # rec_reference_points = all_bbox_preds[..., :3][-1]
            # rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            # _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
            # rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
            # self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)[:, :1024]
            # # self.memory_reference_point = transform_reference_points(self.memory_reference_point, ego_pose)
            # rec_ego_pose = torch.eye(4, device=all_cls_scores.device).unsqueeze(0).unsqueeze(0).repeat(2, 2000, 1, 1)  # 为什么还要多个256
            self.post_update_memory(img_metas, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)
        return ret_dicts

    def forward_single_val(self, x, x_img, img_metas):
        # 这里头还是有一个batch_size
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        use_a = True
        if use_a:
            self.pre_update_memory(img_metas, x)

        ret_dicts = []
        x = self.shared_conv(x)

        # generate reference point from ori 900 + 10 * max_num_of_bbx
        # reference_points = torch.stack([self.reference_points.weight, self.memory_reference_point[:256]], dim=1)
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)
        # 然后在这里更新reference_points和query相关信息

        #  -------------------------
        # debug add
        #  -------------------------
        # if self.memory_reference_point is None:
        #     self.memory_reference_point = x.new_zeros(2, self.memory_len, 3)
        # if use_a:
        #     a, b = mask_dict['pad_size'], attn_mask.shape[0]
        #     # reference_points = torch.cat([reference_points, self.memory_reference_point[:, :256]], dim=1)
        #     attn_mask1 = attn_mask.new_zeros(b + 256, b + 256)
        #     attn_mask1[:b, :b] = attn_mask
        #     attn_mask1[b:, :a] = 1

        # --------------------------

        # ----- img feat pos_emb
        rv_pos_embeds, feat_valid = self._rv_pe_fisheye(x_img, img_metas)  # 相机的position_embedding 主要修改应该在这边
        # ------ lidar feat pos_emb
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))

        # query 从reference point中得到query的信息
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds  # query_positions
        query_tgt = torch.zeros_like(query_embeds).to(query_embeds)
        if use_a:
            # 将 query: query_embeds; query_pos
            tgt, tgt_query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_embeds, query_tgt,
                                                                                                                reference_points)
            #
            num_mask = mask_dict['pad_size']
            attn_mask_expand = torch.zeros(tgt.shape[1], tgt.shape[1] + temp_memory.shape[1]).to(attn_mask)
            attn_mask_expand[:attn_mask.shape[0], :attn_mask.shape[1]] = attn_mask
            attn_mask_expand[attn_mask.shape[0]:, :num_mask] = 1  # 这一部分都要被mask掉

            outs_dec, _ = self.transformer(
                x, x_img, tgt_query_pos, tgt_query_pos,
                bev_pos_embeds, rv_pos_embeds,
                temp_memory=temp_memory, temp_pos=temp_pos,
                attn_masks=attn_mask_expand
            )
        else:
            outs_dec, _ = self.transformer(
                x, x_img, query_tgt,
                bev_pos_embeds, rv_pos_embeds,
                attn_masks=attn_mask
            )
        # 6 2 1260 256
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())

        flag = 0
        outputs_classes, outputs_coords = [], []
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label = task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)

                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            outputs_classes.append(outs['cls_logits'])
            outputs_coords.append(torch.cat([outs['center'], outs['height'], outs['vel']], dim=-1))  # 'dim', 'rot', 'vel'
            ret_dicts.append(outs)
        all_cls_scores = outputs_classes[0]
        all_bbox_preds = outputs_coords[0]
        all_bbox_preds[..., 0:3] = (
                all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        if use_a:
            # rec_reference_points = all_bbox_preds[..., :3][-1]
            # rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            # _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
            # rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
            # self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)[:, :1024]
            # # self.memory_reference_point = transform_reference_points(self.memory_reference_point, ego_pose)
            # rec_ego_pose = torch.eye(4, device=all_cls_scores.device).unsqueeze(0).unsqueeze(0).repeat(2, 2000, 1, 1)  # 为什么还要多个256
            self.post_update_memory(img_metas, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)
        return ret_dicts

    def pre_update_memory(self, data, device):
        x = torch.tensor([i['prev_exists'] for i in data]).to(device).float()
        B = len(x)
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims).detach()
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3).detach()
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1).detach()
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4).detach()
            self.memory_velo = x.new_zeros(B, self.memory_len, 2).detach()
        else:
            timestamps = torch.tensor([i['timestamp'] for i in data]).to(device).double()
            ego_pose_inv = torch.tensor([i['ego_pose'] for i in data]).inverse().to(device).float()

            self.memory_timestamp += timestamps.unsqueeze(-1).unsqueeze(-1)  # 原始t (s)
            self.memory_egopose = ego_pose_inv.unsqueeze(1) @ self.memory_egopose  # Tete0
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, ego_pose_inv)  # 从world 到egot

            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)

        if self.num_propagated > 0:  # 一次时间256个，剩下的是再之前的 可以存4个时间节点
            pseudo_reference_points = self.pseudo_reference_points.weight * (
                # pseudo_reference_points = self.pseudo_reference_points * (
                    self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated] = self.memory_reference_point[:, :self.num_propagated] + (
                    1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated] = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(
                B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        device = rec_ego_pose.device
        # 先去掉dn query n*10 = mask_dict['pad_size']
        # 这里好像用不着pad_size
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
        else:
            rec_memory = outs_dec[-1]

        rec_reference_points = all_bbox_preds[..., :3][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()
        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        ego_pose = torch.tensor([i['ego_pose'] for i in data]).to(device).float()
        timestamps = torch.tensor([i['timestamp'] for i in data]).to(device).double()

        self.memory_reference_point = transform_reference_points(self.memory_reference_point, ego_pose)
        self.memory_timestamp -= timestamps.unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = ego_pose.unsqueeze(1) @ self.memory_egopose.float()


def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:])
    return memory * prev_exist


def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape

        view_shape = [1 for _ in range(len(feat_shape))]
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)

        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)

    return pos_x


def nerf_positional_encoding(
        tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:  # True
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)
