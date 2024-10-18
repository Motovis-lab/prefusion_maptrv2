# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from collections import abc
import functools

import mmcv
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmcv.runner import force_fp32, auto_fp16
# from mmdet.core import multi_apply
# from mmdet.models import DETECTORS
# from mmdet.models.builder import build_backbone
# from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
#                           merge_aug_bboxes_3d, show_result)
from mmdet3d.models import HardSimpleVFE
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmengine import MODELS, digit_version
from mmengine.model import BaseDataPreprocessor
from mmengine.utils.dl_utils import TORCH_VERSION
from torch import autocast

from contrib.cmt.grid_mask import GridMask
from contrib.cmt.spconv_voxelize import SPConvVoxelization

from inspect import getfullargspec
from typing import Callable, Iterable, List, Optional, Dict, Any


def cast_tensor_type(inputs, src_type: torch.dtype, dst_type: torch.dtype):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Note:
        In v1.4.4 and later, ``cast_tersor_type`` will only convert the
        torch.Tensor which is consistent with ``src_type`` to the ``dst_type``.
        Before v1.4.4, it ignores the ``src_type`` argument, leading to some
        potential problems. For example,
        ``cast_tensor_type(inputs, torch.float, torch.half)`` will convert all
        tensors in inputs to ``torch.half`` including those originally in
        ``torch.Int`` or other types, which is not expected.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        # we need to ensure that the type of inputs to be casted are the same
        # as the argument `src_type`.
        return inputs.to(dst_type) if inputs.dtype == src_type else inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({  # type: ignore
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(  # type: ignore
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def auto_fp16(
        apply_to: Optional[Iterable] = None,
        out_fp32: bool = False,
        supported_types: tuple = (nn.Module,),
) -> Callable:
    def auto_fp16_wrapper(old_func: Callable) -> Callable:

        @functools.wraps(old_func)
        def new_func(*args, **kwargs) -> Callable:
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], supported_types):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                f'method of those classes {supported_types}')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=True):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper


def force_fp32(apply_to: Optional[Iterable] = None,
               out_fp16: bool = False) -> Callable:
    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs) -> Callable:
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=False):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


@MODELS.register_module()
class FrameBatchMerger(BaseDataPreprocessor):
    def __init__(self, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self._device = device

    def forward(self, data: List[Dict[str, Any]], training: bool = False) -> Dict[str, List[Any]]:
        merged = {}
        for key in data[0].keys():
            merged[key] = [self._cast_data(i[key]) for i in data]
        return merged

    def _cast_data(self, data: Any):
        if isinstance(data, torch.Tensor):
            _dtype = torch.float32 if "float" in str(data.dtype) else data.dtype
            return data.to(dtype=_dtype, device=self._device)
        if isinstance(data, dict):
            return {k: self._cast_data(data[k]) for k in data}
        return data
        # return self.cast_data(merged)  # type: ignore


@MODELS.register_module()
class CmtDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 **kwargs):
        pts_voxel_cfg = kwargs.pop('pts_voxel_layer')
        super(CmtDetector, self).__init__(**kwargs)
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # if pts_voxel_cfg:
        if True:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    @auto_fp16(apply_to=('img', 'points'))
    # def forward(self, inputs, data_samples, **kwargs):
    def forward(self, *args, mode="loss", **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            # return self.forward_test(*args, **kwargs['inputs'][0][0])
            return self.forward_test(*args, **kwargs)

    def init_weights(self):
        """Initialize model weights."""
        super().init_weights()

    def extract_feat(self, voxel_dict, imgs, img_metas):
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """

        img_feats = self.extract_img_feat(imgs, img_metas) if imgs is not None else None
        pts_feats = self.extract_pts_feat(voxel_dict) if voxel_dict is not None else None

        return (img_feats, pts_feats)

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    # @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, voxel_dicts):

        """

                    voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas)

        :param pts:
        :param img_feats:
        :param img_metas:
        :return:
        """
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        batch_size = len(voxel_dicts)
        coors = []
        voxel_features = []
        for val, voxel in enumerate(voxel_dicts):
            voxel_features += [voxel['res_voxels'].clone()]
            coors += [F.pad(voxel['res_coors'].clone(), (1, 0), mode='constant', value=val)]
        voxels_features = torch.cat(voxel_features, dim=0)
        coors_batch = torch.cat(coors, dim=0)
        x = self.pts_middle_encoder(voxels_features, coors_batch, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      *,
                      index_info=None, camera_images=None, bbox_3d=None, ego_poses=None, meta_info=None,
                      lidar_points=None,
                      # points=None,
                      # img_metas=None,
                      # gt_bboxes_3d=None,
                      # gt_labels_3d=None,
                      # img=None,
                      # gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # transform meta_info to img_metas.
        img_metas = [i for i in meta_info]
        imgs = torch.stack(camera_images)
        img_feats, pts_feats = self.extract_feat(lidar_points, imgs=imgs, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            for meta, bbox, img in zip(img_metas, bbox_3d, imgs):
                meta['gt_bboxes_3d'] = BaseInstance3DBoxes(bbox[:, :9], box_dim=9)
                meta['gt_labels_3d'] = meta['bbox_3d']['classes']
                meta['pad_shape'] = [[img.shape[-2], img.shape[-1], 3]]
                meta['cam_inv_poly'] = meta['camera_images']['cam_inv_poly']
                meta['cam_intrinsic'] = meta['camera_images']['intrinsic']
                meta['lidar2cam'] = meta['camera_images']['extrinsic_inv']
            losses_pts = self.forward_pts_train(pts_feats, img_feats, None, None, img_metas)
            losses.update(losses_pts)
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        gt_bboxes_3d = [meta['gt_bboxes_3d'] for meta in img_metas]
        gt_labels_3d = [meta['gt_labels_3d'] for meta in img_metas]
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        # losses = self.pseudo_loss(outs)
        return losses

    def pseudo_loss(self, outs):
        loss = 0
        for k, v in list(outs[0][0].items())[:12]:
            loss += v.sum()
        return {'easy_loss': loss}

    def forward_test(self, *,
                     index_info=None, camera_images=None, bbox_3d=None, ego_poses=None, meta_info=None,
                     lidar_points=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """

        # lidar_points = lidar_points
        # camera_images = camera_images
        # bbox_3d = bbox_3d
        # meta_info = meta_info
        # lidar_points = [lidar_poitns]
        imgs = torch.stack(camera_images)
        img_metas = [i for i in meta_info]
        for meta, bbox, img in zip(img_metas, bbox_3d, imgs):
            meta['gt_bboxes_3d'] = BaseInstance3DBoxes(bbox[:, :9], box_dim=9)
            meta['gt_labels_3d'] = meta['bbox_3d']['classes']
            meta['pad_shape'] = [[img.shape[-2], img.shape[-1], 3]]
            meta['cam_inv_poly'] = meta['camera_images']['cam_inv_poly']
            meta['cam_intrinsic'] = meta['camera_images']['intrinsic']
            meta['lidar2cam'] = meta['camera_images']['extrinsic_inv']

        return self.simple_test(lidar_points, img_metas, imgs, **kwargs)

    # @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        # box score label
        return bbox_list
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        img_feats, pts_feats = self.extract_feat( points,img, img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)  # box, score label
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        if img_feats and self.with_img_bbox:  # False
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
