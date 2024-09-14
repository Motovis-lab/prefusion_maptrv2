"""Inherited from `https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/dense_heads/centerpoint_head.py`"""  # noqa
import numba
import numpy as np
import torch
from mmdet3d.models import draw_heatmap_gaussian, gaussian_radius
from prefusion.registry import MODELS
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead, circle_nms
from mmdet3d.models.utils import clip_sigmoid
from mmdet.utils import reduce_mean
from torch.cuda.amp import autocast
from mmdet3d.models.layers import nms_bev
from mmdet3d.structures import xywhr2xyxyr
from mmdet.models.utils import multi_apply
from scipy.spatial.transform import Rotation
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import copy
from pyquaternion import Quaternion
from contrib.fastbev_det.utils.utils import get_cam_corners, intrinsics_matrix, get_3d_lines, get_bev_lines, get_corners_with_angles, get_bev_lines_cylinder
import cv2


@numba.jit(nopython=True)
def size_aware_circle_nms(dets, thresh_scale, post_max_size=83):
    """Circular NMS.
    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.
    Args:
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept. Defaults
            to 83
    Returns:
        torch.Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    dx1 = dets[:, 2]
    dy1 = dets[:, 3]
    yaws = dets[:, 4]
    scores = dets[:, -1]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist_x = abs(x1[i] - x1[j])
            dist_y = abs(y1[i] - y1[j])
            dist_x_th = (abs(dx1[i] * np.cos(yaws[i])) +
                         abs(dx1[j] * np.cos(yaws[j])) +
                         abs(dy1[i] * np.sin(yaws[i])) +
                         abs(dy1[j] * np.sin(yaws[j])))
            dist_y_th = (abs(dx1[i] * np.sin(yaws[i])) +
                         abs(dx1[j] * np.sin(yaws[j])) +
                         abs(dy1[i] * np.cos(yaws[i])) +
                         abs(dy1[j] * np.cos(yaws[j])))
            # ovr = inter / areas[j]
            if dist_x <= dist_x_th * thresh_scale / 2 and \
               dist_y <= dist_y_th * thresh_scale / 2:
                suppressed[j] = 1
    return keep[:post_max_size]

@MODELS.register_module()
class BEVDepthHeadV1(CenterHead):
    """Head for BevDepth.

    Args:
        in_channels(int): Number of channels after bev_neck.
        tasks(dict): Tasks for head.
        bbox_coder(dict): Config of bbox coder.
        common_heads(dict): Config of head for each task.
        loss_cls(dict): Config of classification loss.
        loss_bbox(dict): Config of regression loss.
        gaussian_overlap(float): Gaussian overlap used for `get_targets`.
        min_radius(int): Min radius used for `get_targets`.
        train_cfg(dict): Config used in the training process.
        test_cfg(dict): Config used in the test process.
        bev_backbone_conf(dict): Cnfig of bev_backbone.
        bev_neck_conf(dict): Cnfig of bev_neck.
    """

    def __init__(
        self,
        in_channels=256,
        tasks=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        gaussian_overlap=0.1,
        min_radius=2,
        train_cfg=None,
        test_cfg=None,
        bev_backbone_conf=None,
        bev_neck_conf=None,
        separate_head=dict(type='SeparateHead',
                           init_bias=-2.19,
                           final_kernel=3),
    ):
        super(BEVDepthHeadV1, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
        )
        self.trunk = MODELS.build(bev_backbone_conf)
        self.trunk.init_weights()
        self.neck = MODELS.build(bev_neck_conf)
        self.neck.init_weights()
        del self.trunk.maxpool
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.tasks = tasks
        self.all_classes = []
        for sub_classes in self.class_names:
            self.all_classes.extend(sub_classes)

    @autocast(False)
    def forward(self, x):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        x = x.float()
        # FPN
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)
        ret_values = super().forward(fpn_output)
        return ret_values

    def get_targets(self, batch_data):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which
                        boxes are valid.
        """
        gt_bboxes_3d, gt_labels_3d = [], []
        bev_front = self.train_cfg['to_mv_coord'][0] / self.train_cfg['voxel_size'][0]
        bev_left = self.train_cfg['to_mv_coord'][1] / self.train_cfg['voxel_size'][1]
        for data in batch_data:
            bboxes_3d = []
            labels_3d = []
            for task_name in self.tasks:
                for box_id, boxes in enumerate(data['transformables'][task_name['label_type']].data['elements']):
                    labels_3d.append(self.all_classes.index(boxes['class']))
                    tmp_box = [*boxes['translation'].tolist(), *boxes['size'], Rotation.from_matrix(boxes['rotation']).as_euler("xyz", degrees=False).tolist()[-1], *boxes['velocity'].tolist()[:2]]
                    corners = get_corners_with_angles(np.array(tmp_box)[None], boxes['rotation'].T)[0][:4, :3]
                    front_center_z = corners[:2, 2].mean(0)
                    back_center_z = corners[2:, 2].mean(0)
                    bboxes_3d.append([*boxes['translation'].tolist(), *boxes['size'], Rotation.from_matrix(boxes['rotation']).as_euler("xyz", degrees=False).tolist()[-1], *boxes['velocity'].tolist()[:2], front_center_z, back_center_z])
            bboxes_3d = torch.as_tensor(bboxes_3d, device='cuda')
            labels_3d = torch.as_tensor(labels_3d, device='cuda')
            gt_bboxes_3d.append(bboxes_3d)
            gt_labels_3d.append(labels_3d)
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks


    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['bev_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(
                torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
            task_classes.append(
                torch.cat(task_class).long().to(gt_bboxes_3d.device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = torch.full(
                (len(self.class_names[idx]), feature_map_size[1],
                feature_map_size[0]),
                fill_value=0.0,
                dtype=torch.float32,
                device='cuda')

            anno_box = torch.full(
                (max_objs, len(self.train_cfg['code_weights'])),
                fill_value=0.0,
                dtype=torch.float32,
                device='cuda')

            ind = torch.zeros((max_objs),
                              dtype=torch.int64,
                              device='cuda')
            mask = torch.zeros((max_objs),
                             dtype=torch.uint8,
                             device='cuda')

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    # z is box center not box planer bev center box[2] -= box[5]/2
                    # TODO add box front center and box back center z reg
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        -x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        -y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device='cuda')
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[1]
                            and 0 <= center_int[1] < feature_map_size[0]):
                        continue

                    draw_gaussian(heatmap[cls_id], [center_int[1], center_int[0]], radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert x * feature_map_size[0] + y < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = x * feature_map_size[0] + y
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    if len(task_boxes[idx][k]) > 7:
                        vx, vy = task_boxes[idx][k][7:9]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    z -= box_dim[-1]/2  # z reg the box bev planer height
                    front_center_z = task_boxes[idx][k][9]
                    back_center_z = task_boxes[idx][k][10]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if len(task_boxes[idx][k]) > 7:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0),
                            front_center_z.unsqueeze(0),
                            back_center_z.unsqueeze(0)
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            front_center_z.unsqueeze(0),
                            back_center_z.unsqueeze(0)
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def loss(self, targets, preds_dicts, **kwargs):
        """Loss function for BEVDepthHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = targets
        return_loss = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor(num_pos)),
                                         min=1).item()
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps[task_id],
                                         avg_factor=cls_avg_factor)
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict[0].keys():
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['vel'], preds_dict[0]['fb_z']),
                    dim=1,
                )
            else:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['fb_z']),
                    dim=1,
                )
            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(reduce_mean(target_box.new_tensor(num)),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(pred,
                                       target_box,
                                       bbox_weights,
                                       avg_factor=num)
            return_loss += loss_bbox
            return_loss += loss_heatmap
        return return_loss, dict()

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']
            batch_hei += batch_dim[:, [-1], ...] / 2
            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(batch_heatmap.permute(0,1,3,2).contiguous(),
                                          batch_rots.permute(0,1,3,2).contiguous(),
                                          batch_rotc.permute(0,1,3,2).contiguous(),
                                          batch_hei.permute(0,1,3,2).contiguous(),
                                          batch_dim.permute(0,1,3,2).contiguous(),
                                          batch_vel.permute(0,1,3,2).contiguous(),
                                          reg=batch_reg.permute(0,1,3,2).contiguous(),
                                          task_id=task_id)
            assert self.test_cfg['nms_type'] in [
                'size_aware_circle', 'circle', 'rotate'
            ]
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(circle_nms(
                        boxes.detach().cpu().numpy(),
                        self.test_cfg['min_radius'][task_id],
                        post_max_size=self.test_cfg['post_max_size']),
                                        dtype=torch.long,
                                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            elif self.test_cfg['nms_type'] == 'size_aware_circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    boxes_2d = boxes3d[:, [0, 1, 3, 4, 6]]
                    boxes = torch.cat([boxes_2d, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        size_aware_circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['thresh_scale'][task_id],
                            post_max_size=self.test_cfg['post_max_size'],
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list
    
    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
    
    def show_results(self, all_pred, data, frame_ids, **kwargs):
        """Visualize the results to an image.

        Args:
            results (list[dict]): Predicted results.
            img_metas (list[dict]): Point cloud and image's meta info.
            out_dir (str): Path of output directory, None means not saving
                the image.
        """
        for b in range(len(all_pred)):
            gt_corners = []
            gt_cylinders = []
            # for i,sub_boxes in enumerate(all_pred[b]):
            # for i in range(all_pred[b].shape[0]):
            #     sub_cla = all_classes[b]
            #     sub_boxes = all_pred[b][i]
            for j in range(len(all_pred[b][0])):
                # class_cat = sub_cla[j]
                pred_box = all_pred[b][0][j].tolist()
                # tmp_center = [-(pred_box[1] - 36), -(pred_box[0] - 12), pred_box[2]]
                tmp_center = pred_box[:3]
                tmp_box = np.array(tmp_center + pred_box[3:6] + [0] + [0, 0])
                tmp_r = Rotation.from_euler('xyz', angles=[0, 0, pred_box[6]], degrees=False).as_matrix()
                corners = get_corners_with_angles(tmp_box[None], tmp_r.T)[0][:, :3]
                # corners[:4, 2] += np.array(pred_box[6:10])
                # corners[4:, 2] += np.array(pred_box[6:10])
                gt_corners.append(corners)
            # else:
            #     for j in range(len(sub_boxes)):
            #         class_cat = sub_cla[j]
            #         pred_box = sub_boxes[j]
            #         tmp_center = [-(pred_box[1] - 36), -(pred_box[0] - 12), pred_box[2]]
            #         tmp_r = Rotation.from_euler('xyz', angles=[0, 0, pred_box[5]], degrees=False).as_matrix()
                    
            #         corners = plot_cylinder(tmp_center, np.array([0,0,1]), radius=pred_box[3], height=pred_box[4])
            #         lines = get_bev_lines_cylinder(corners)
            #         gt_cylinders.append(corners)
                
            # gt_corners = [] 
            # gt_cylinders = []
            # for label_type in ['bbox_3d', 'bbox_bev', 'square_3d']:
            #     label_data = ori_data[b]['transformables'][label_type].data['elements']
            #     for box_3d in label_data:
            #         box = np.array(box_3d['translation'].reshape(-1).tolist() + box_3d['size'] + [Quaternion(matrix=box_3d['rotation']).yaw_pitch_roll[0]] + [0, 0])
            #         if np.linalg.norm(box[:2]) <= 1e5:
            #             corners = get_corners_with_angles(box[None], box_3d['rotation'].T)[0]
            #             gt_corners.append(corners)
            
            plt.figure(figsize=(24, 8))
            row = 5 

            # for i, k in enumerate(IMG_KEYS):
            img_num = 0
            for key_camera in data.keys():
                imgs = torch.split(data[key_camera]['imgs'], self.each_camera_nums[key_camera.split('_')[0]])[b]
                extrinsics = torch.split(data[key_camera]['extrinsic'], self.each_camera_nums[key_camera.split('_')[0]])[b]
                intrinsics = torch.split(data[key_camera]['intrinsic'], self.each_camera_nums[key_camera.split('_')[0]])[b]
                for idx in range(len(imgs)):
                    # Draw camera views
                    fig_idx = img_num + 1 if img_num < row else img_num + 2
                    img_num += 1
                    plt.subplot(2, 6, fig_idx)

                    # Set camera attributes
                    plt.title(f"{key_camera} {idx}")
                    plt.axis('off')
                    H, W = imgs.shape[-2:]
                    plt.xlim(0, W)
                    plt.ylim(H, 0)
                    img = imgs[idx].cpu().numpy().transpose(1, 2, 0) - (imgs[idx].min().cpu().numpy())
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # plt.imsave(f"{key_camera}_{idx}.jpg",img)

                    # Draw images
                    plt.imshow(img)
                    if 'fish' in key_camera:
                        continue
                    # Draw 3D gt
                    for corners in gt_corners:
                        cam_corners = get_cam_corners(
                            corners,
                            extrinsics[idx][:3, 3].cpu().numpy(),
                            Quaternion(matrix=extrinsics[idx][:3, :3].cpu().numpy(), atol=1e-06),
                            intrinsics_matrix(intrinsics[idx][:4].cpu().numpy()))
                        lines = get_3d_lines(cam_corners)
                        for line in lines:
                            plt.plot(line[0],
                                    line[1],
                                    c=cm.get_cmap('tab10')(4)
                                    )
                    for corners in gt_cylinders:
                        cam_corners = get_cam_corners(
                            corners,
                            extrinsics[idx][:3, 3].cpu().numpy(),
                            Quaternion(matrix=extrinsics[idx][:3, :3].cpu().numpy(), atol=1e-06),
                            intrinsics_matrix(intrinsics[idx][:4].cpu().numpy()))
                        
                        bottom_cam_corner = cam_corners[:100, :]
                        up_cam_corner = cam_corners[100:, :]

                        plt.plot(bottom_cam_corner[:, 0],
                                bottom_cam_corner[:, 1],
                                c=cm.get_cmap('tab10')(4)
                                )
                        plt.plot(up_cam_corner[:, 0],
                                up_cam_corner[:, 1],
                                c=cm.get_cmap('tab10')(4)
                                )
                        for i in range(0, 100, 10):
                            plt.plot([bottom_cam_corner[i, 0], up_cam_corner[i, 0]], [bottom_cam_corner[i, 1], up_cam_corner[i, 1]], 'g-')
                    # for box in info['box_2d'][k]['box']:
                    #     x1, y1, x2, y2 = box
                    #     w = x2 - x1
                    #     h = y2 - y1
                    #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    #     plt.gca().add_patch(rect)
            # Draw BEV
            plt.subplot(1, 6, 6)

            # Set BEV attributes
            plt.title('LIDAR_TOP')
            plt.axis('equal')
            plt.xlim(-15, 15)
            plt.ylim(-15, 40)

            # BEV box ego 是x朝前 y朝左,  可视化出来到图上是x朝右，y朝前，对应到图上x=-y,y=x 
            # Draw BEV GT boxes
            for corners in gt_corners:
                lines = get_bev_lines(corners)
                for line in lines:
                    plt.plot([-x for x in line[1]],
                            line[0],
                            c='r',
                            label='ground truth')
            for corners in gt_cylinders:
                lines = get_bev_lines_cylinder(corners)
                for line in lines:
                    plt.plot([-x for x in line[1]],
                            line[0],
                            c='r',
                            label='ground truth')
            # Set legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(),
                    by_label.keys(),
                    loc='upper right',
                    framealpha=1)

            # Save figure
            plt.tight_layout(w_pad=0, h_pad=2)
            # plt.show()
            plt.savefig(f"work_dirs/vis_data/{frame_ids[b]}.jpg")
            plt.close()