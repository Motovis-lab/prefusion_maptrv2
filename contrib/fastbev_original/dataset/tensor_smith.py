import cv2
import torch
import numpy as np

from typing import Tuple, Dict, Union, Iterable
from torch import Tensor

from prefusion.registry import TENSOR_SMITHS
from prefusion.dataset.utils import (
    expand_line_2d, _sign, INF_DIST,
    vec_point2line_along_direction, 
    dist_point2line_along_direction,
    choose_index
)
from prefusion.dataset.transform import Bbox3D
from prefusion.dataset.tensor_smith import TensorSmith, is_in_bbox3d, get_bev_intrinsics

__all__ = ["XyzLwhYawVeloBbox3D"]


@TENSOR_SMITHS.register_module()
class XyzLwhYawVeloBbox3D(TensorSmith):
    
    def __init__(self, 
                 voxel_shape: tuple, 
                 voxel_range: Tuple[list, list, list], 
                 use_bottom_center=False,
                 has_velocity: bool=True,
                 reverse_pre_conf: float=0.1,
                 reverse_nms_ratio: float=0.7,
                 reverse_nms_min_radius: float=0.25):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]
        use_bottom_center : bool
        has_velocity : bool
        reverse_pre_conf : float
        reverse_nms_ratio : float
        reverse_nms_min_radius : float

        Examples
        --------
        - voxel_shape=(6, 320, 160)
        - voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        - Z, X, Y = voxel_shape

        """
        self.voxel_shape = tuple(voxel_shape)
        self.voxel_range = tuple(voxel_range)
        self.bev_intrinsics = get_bev_intrinsics(voxel_shape, voxel_range)
        Z, X, Y = voxel_shape
        xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
        self.points_grid_bev = np.float32([yy, xx])
        self.use_bottom_center = use_bottom_center
        self.has_velocity = has_velocity
        self.reverse_pre_conf = reverse_pre_conf
        self.reverse_nms_ratio = reverse_nms_ratio
        self.reverse_nms_min_radius = reverse_nms_min_radius

    
    @staticmethod
    def _get_roll_from_xyvecs(xvec, yvec):
        xvec_bev_vertical = np.array([-xvec[1], xvec[0], 0])
        if np.linalg.norm(xvec_bev_vertical) < 1e-3:
            xvec_bev_vertical = np.array([0, 1, 0])
        a = xvec_bev_vertical / np.linalg.norm(xvec_bev_vertical)
        b = yvec / np.linalg.norm(yvec)
        cos_roll = np.dot(a, b)
        cross = np.cross(a, b)
        sin_sign = _sign(np.dot(cross, xvec))
        sin_roll = sin_sign * np.linalg.norm(cross)
        return cos_roll, sin_roll
    
    
    def __call__(self, transformable: Bbox3D):
        """
        transformable Bbox3D
        ---------
        elements: List[dict]
        a list of boxes. Each element is a dict of box having the following format
        ```
        elements[0] = {
            'class': 'class.vehicle.passenger_car',
            'attr': [
                'attr.time_varying.object.state.stationary',
                'attr.vehicle.is_trunk_open.false',
                'attr.vehicle.is_door_open.false'
            ],
            'size': [4.6486, 1.9505, 1.5845],
            'rotation': array([
                [ 0.93915682, -0.32818596, -0.10138267],
                [ 0.32677338,  0.94460343, -0.03071667],
                [ 0.1058472 , -0.00428138,  0.99437319]
            ]),
            'translation': array([[-15.70570354], [ 11.88484971], [ -0.61029085]]), # NOTE: it is a column vector
            'track_id': '10035_0', # NOT USED
            'velocity': array([[0.], [0.], [0.]]) # NOTE: it is a column vector
        }
        ```
        dictionary: dict
            following format
            ```dictionary = {'classes': ['car', 'bus', 'pedestrain', ...], 'attrs': []}
            ```
        flip_aware_class_pairs: List[tuple]
            list of class pairs that are flip-aware
            flip_aware_class_pairs = [('left_arrow', 'right_arrow')]
        
        Return
        ------
        tensor_smith: TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        Z, X, Y = self.voxel_shape
        cx, cy, fx, fy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev

        unit_xvecs = []
        unit_yvecs = []
        centers = []
        all_points_bev = []
        box_sizes = []
        velocities = []
        class_inds = []
        attr_lists = []
        attr_available = 'attrs' in transformable.dictionary
        for element in transformable.elements:
            if element['class'] in transformable.dictionary['classes']:
                # unit vector of x_axis/y_axis of the box
                unit_xvecs.append(element['rotation'][:, 0])
                unit_yvecs.append(element['rotation'][:, 1])
                # get the position of the box
                center = np.float32(element['translation'][:, 0])
                if self.use_bottom_center:
                    center -= 0.5 * element['size'][2] * element['rotation'][:, 2]
                centers.append(np.array([
                    center[1] * fy + cy,
                    center[0] * fx + cx,
                    center[2]
                ], dtype=np.float32))
                xvec = element['size'][0] * element['rotation'][:, 0]
                yvec = element['size'][1] * element['rotation'][:, 1]
                corner_points = np.array([
                    center + 0.5 * xvec - 0.5 * yvec,
                    center + 0.5 * xvec + 0.5 * yvec,
                    center - 0.5 * xvec + 0.5 * yvec,
                    center - 0.5 * xvec - 0.5 * yvec
                ], dtype=np.float32)
                points_bev = np.array([
                    corner_points[..., 1] * fy + cy,
                    corner_points[..., 0] * fx + cx
                ], dtype=np.float32).T
                all_points_bev.append(points_bev)
                box_sizes.append(element['size'])
                velocities.append(element['velocity'][:, 0])
                # get class and attr index
                class_inds.append(transformable.dictionary['classes'].index(element['class']))
                attr_ind_list = []
                if 'attr' in element and attr_available:
                    element_attrs = element['attr']
                    for attr in element_attrs:
                        if attr in transformable.dictionary['attrs']:
                            attr_ind_list.append(transformable.dictionary['attrs'].index(attr))
                attr_lists.append(attr_ind_list)
                # TODO: add ignore_mask according to ignore classes and attrs
        num_class_channels = len(transformable.dictionary['classes'])
        if attr_available:
            num_attr_channels = len(transformable.dictionary['attrs'])
        else:
            num_attr_channels = 0
        # maybe later, think about overlapped objects
        seg_im = np.zeros((1 + num_class_channels + num_attr_channels, X, Y), dtype=np.float32)
        cen_im = np.zeros((1, X, Y), dtype=np.float32)
        if self.has_velocity:
            reg_im = np.zeros((20, X, Y), dtype=np.float32)
        else:
            reg_im = np.zeros((17, X, Y), dtype=np.float32)
        for unit_xvec, unit_yvec, points_bev, center, size, velo, class_ind, attr_list in zip(
            unit_xvecs, unit_yvecs, all_points_bev, centers, box_sizes, velocities, class_inds, attr_lists
        ):
            points_bev_int = np.round(points_bev).astype(int)
            region_box = cv2.fillPoly(np.zeros((X, Y), dtype=np.float32), [points_bev_int], 1)
            ## gen segmentation
            cv2.fillPoly(seg_im[0], [points_bev_int], 1)
            cv2.fillPoly(seg_im[1 + class_ind], [points_bev_int], 1)
            for attr_ind in attr_list:
                cv2.fillPoly(seg_im[1 + num_class_channels + attr_ind], [points_bev_int], 1)
            ## gen regression
            # 0, 1: center x y, in bev coord
            vec2center = center[:2][..., None, None] - points_grid_bev
            reg_im[[0, 1]] = reg_im[[0, 1]] * (1 - region_box) + vec2center * region_box
            # 2: center z
            reg_im[2] = reg_im[2] * (1 - region_box) + center[2] * region_box
            # 3, 4, 5: size l, w, h
            reg_im[3] = reg_im[3] * (1 - region_box) + size[0] * region_box
            reg_im[4] = reg_im[4] * (1 - region_box) + size[1] * region_box
            reg_im[5] = reg_im[5] * (1 - region_box) + size[2] * region_box
            # 6, 7, 8: unit xvec
            reg_im[6] = reg_im[6] * (1 - region_box) + unit_xvec[0] * region_box
            reg_im[7] = reg_im[7] * (1 - region_box) + unit_xvec[1] * region_box
            reg_im[8] = reg_im[8] * (1 - region_box) + unit_xvec[2] * region_box
            # 9, 10, 11, 12, 13: absolute xvec
            reg_im[9] = reg_im[9] * (1 - region_box) + np.abs(unit_xvec[0]) * region_box
            reg_im[10] = reg_im[10] * (1 - region_box) + np.abs(unit_xvec[1]) * region_box
            reg_im[11] = reg_im[11] * (1 - region_box) + np.abs(unit_xvec[2]) * region_box
            reg_im[12] = reg_im[12] * (1 - region_box) + unit_xvec[0] * unit_xvec[1] * region_box
            reg_im[13] = reg_im[13] * (1 - region_box) + unit_xvec[0] * unit_xvec[2] * region_box
            # 14, 15, 16: abs roll angle of yvec and xvec_bev_vertial                
            cos_roll, sin_roll = self._get_roll_from_xyvecs(unit_xvec, unit_yvec)
            reg_im[14] = reg_im[14] * (1 - region_box) + abs(cos_roll) * region_box
            reg_im[15] = reg_im[15] * (1 - region_box) + abs(sin_roll) * region_box
            reg_im[16] = reg_im[16] * (1 - region_box) + cos_roll * sin_roll * region_box
            # 17, 18, 19: velocity
            if self.has_velocity:
                reg_im[17] = reg_im[17] * (1 - region_box) + velo[0] * region_box
                reg_im[18] = reg_im[18] * (1 - region_box) + velo[1] * region_box
                reg_im[19] = reg_im[19] * (1 - region_box) + velo[2] * region_box
            ## gen centerness
            center_line_l = np.float32([points_bev[[0, 1]].mean(0), points_bev[[2, 3]].mean(0)])
            center_line_w = np.float32([points_bev[[1, 2]].mean(0), points_bev[[0, 3]].mean(0)])
            direction_l = np.float32(center_line_l[0] - center_line_l[1])
            direction_w = np.float32(center_line_w[0] - center_line_w[1])
            dist2front = dist_point2line_along_direction(points_grid_bev, points_bev[[0, 1]], direction_l)
            dist2left  = dist_point2line_along_direction(points_grid_bev, points_bev[[1, 2]], direction_w)
            dist2rear  = dist_point2line_along_direction(points_grid_bev, points_bev[[2, 3]], direction_l)
            dist2right = dist_point2line_along_direction(points_grid_bev, points_bev[[3, 0]], direction_w)
            min_ds = np.minimum(dist2front, dist2rear) * region_box
            min_ds /= (min_ds.max() + 1e-3)
            min_dl = np.minimum(dist2left, dist2right) * region_box
            min_dl /= (min_dl.max() + 1e-3)
            centerness = min_ds * min_dl
            centerness /= (centerness.max() + 1e-3)
            cen_im[0] = cen_im[0] * (1 - region_box) + centerness * region_box
        ## tensor
        tensor_data = {
            'seg': torch.tensor(seg_im, dtype=torch.float32),
            'cen': torch.tensor(cen_im, dtype=torch.float32),
            'reg': torch.tensor(reg_im, dtype=torch.float32)
        }
        return tensor_data
        
    
    
    @staticmethod
    def _get_yzvec_from_xvec_and_roll(xvecs, roll_vecs):
        xvecs_bev_vertical = np.array([-xvecs[1], xvecs[0], np.zeros_like(xvecs[0])])
        if np.linalg.norm(xvecs_bev_vertical) < 1e-3:
            xvecs_bev_vertical = np.array([
                np.zeros_like(xvecs[0]), 
                np.ones_like(xvecs[0]), 
                np.zeros_like(xvecs[0])
            ])
        xvecs /= np.linalg.norm(xvecs, axis=0)
        xvecs_bev_vertical /= np.linalg.norm(xvecs_bev_vertical, axis=0)        
        cos_roll = roll_vecs[[0]]
        sin_roll = roll_vecs[[1]]
        yvecs = xvecs_bev_vertical * cos_roll + np.cross(xvecs, xvecs_bev_vertical, axis=0) * sin_roll
        yvecs /= np.linalg.norm(yvecs, axis=0)
        zvecs = np.cross(xvecs, yvecs, axis=0)
        return yvecs, zvecs
        
    
    
    def _group_nms(self, seg_scores, cen_scores, seg_classes, centers, sizes, unit_xvecs, roll_vecs, velocities):
        scores = cen_scores * seg_scores
        ranked_inds = np.argsort(scores)[::-1]
        kept_groups = []
        kept_inds = []
        # group inds
        for i in ranked_inds:
            if i not in kept_inds:
                center_i = centers[:, i]
                sizes_i = sizes[:, i]
                unit_xvec_i = unit_xvecs[:, i]
                roll_vec_i = roll_vecs[:, i]
                unit_yvec_i, unit_zvec_i = self._get_yzvec_from_xvec_and_roll(unit_xvec_i, roll_vec_i)
                kept_inds.append(i)
                grouped_inds = [i]
                kept_groups.append(grouped_inds)
                for j in ranked_inds:
                    if j not in kept_inds:
                        center_j = centers[:, j]
                        delta_ij = center_i - center_j
                        if is_in_bbox3d(
                            delta_ij, sizes_i * self.reverse_nms_ratio, 
                            unit_xvec_i, unit_yvec_i, unit_zvec_i,
                            self.reverse_nms_min_radius
                        ):
                            kept_inds.append(j)
                            grouped_inds.append(j)
        _, _, fx, fy = self.bev_intrinsics
        ## get mean bbox in group
        pred_bboxes_3d = []
        for group in kept_groups:
            # use score weighted mean
            score_sum = scores[group].sum() + 1e-6
            mean_classes = (seg_classes[:, group] * scores[group][None]).sum(1) / score_sum
            # get mean_unit_xvec
            # average all xvecs, xvecs may not have same directions
            reference_unit_xvec = (unit_xvecs[:, group] * scores[group][None]).sum(1) / score_sum
            # align all xvecs to the first one, then average all xvecs again
            unit_xvec_0 = unit_xvecs[:, group[0]]
            for i in group[1:]:
                if np.sum(unit_xvec_0 * unit_xvecs[:, i]) < 0:
                    unit_xvecs[:, i] *= -1
            mean_unit_xvec = (unit_xvecs[:, group] * scores[group][None]).sum(1) / score_sum
            # check whether reference_unit_xvec and mean_unit_xvec have same direction
            sign_direction = _sign(np.sum(mean_unit_xvec * reference_unit_xvec))
            mean_unit_xvec *= sign_direction
            mean_roll_vec = (roll_vecs[:, group] * scores[group][None]).sum(1) / score_sum
            unit_yvec, unit_zvec = self._get_yzvec_from_xvec_and_roll(mean_unit_xvec, mean_roll_vec)
            # rotation matrix
            mean_rmat = np.float32([mean_unit_xvec, unit_yvec, unit_zvec]).T
            mean_size = (sizes[:, group] * scores[group][None]).sum(1) / score_sum
            # get area score
            mean_count = seg_classes[0, group].sum()
            mean_area = mean_size[0] * mean_size[1] * fx * fy * min(self.reverse_nms_ratio ** 2, 1)
            area_score = min(1.0, mean_count / mean_area)
            # get center
            mean_center = (centers[:, group] * scores[group][None]).sum(1) / score_sum
            if self.use_bottom_center:
                mean_center = mean_center + 0.5 * unit_zvec * mean_size[2]
            bbox_3d = {
                'confs': mean_classes,
                'area_score': area_score,
                'score': mean_classes[0] * area_score,
                'size': mean_size,
                'rotation': mean_rmat,
                'translation': mean_center
            }
            if self.has_velocity: 
                bbox_3d['velocity'] = (velocities[:, group] * scores[group][None]).sum(1) / score_sum
            pred_bboxes_3d.append(bbox_3d)
        return pred_bboxes_3d
    
    
    def reverse(self, tensor_dict):
        """
        Parameters
        ----------
        tensor_dict : dict
        ```
        {
            'seg': torch.Tensor,
            'cen': torch.Tensor,
            'reg': torch.Tensor
        }
        ```

        Notes
        -----
        ```
        seg_im  # 分割图
        cen_im  # 中心图
        reg_im  # 回归图
            0, 1: center x y, bev coords
            2: center z, ego coords
            3, 4, 5: size (l, w, h), ego coords
            6, 7, 8: unit xvec, ego coords
            9, 10, 11, 12, 13: absolute xvec, ego coords
            14, 15, 16: abs roll angle of yvec and xvec_bev_vertial, intrinsic rotation
            17, 18, 19: velocity, ego coords, TODO: should be converted to ego coords, currently world coords            
        ```
        """
        seg_pred = tensor_dict['seg'].detach().cpu().numpy()
        cen_pred = tensor_dict['cen'].detach().cpu().numpy()
        reg_pred = tensor_dict['reg'].detach().cpu().numpy()
        ## pickup bbox points
        valid_points_map = seg_pred[0] > self.reverse_pre_conf 
        # valid postions
        valid_points_bev = self.points_grid_bev[:, valid_points_map]
        # pickup scores and classes
        seg_scores = seg_pred[0][valid_points_map]
        cen_scores = cen_pred[0][valid_points_map]
        seg_classes = seg_pred[:, valid_points_map]
        # pickup regressions
        reg_values = reg_pred[:, valid_points_map]
        # 0, 1, 2: centers in ego coords
        cx, cy, fx, fy = self.bev_intrinsics
        center_xs = (valid_points_bev[1] + reg_values[1] - cx) / fx
        center_ys = (valid_points_bev[0] + reg_values[0] - cy) / fy
        center_zs = reg_values[2]
        centers = np.array([center_xs, center_ys, center_zs])
        # 3, 4, 5: sizes in ego coords
        sizes = reg_values[[3, 4, 5]]
        # infer unit xvecs from absolute xvecs and ref_unit xvecs
        # 6, 7, 8: reference unit xvecs in ego coords
        ref_unit_xvecs = reg_values[[6, 7, 8]]
        # 9, 10, 11, 12, 13: absolute unit xvecs in ego coords
        abs_unit_xvecs = reg_values[[9, 10, 11, 12, 13]]
        # calculte unit xvecs
        unit_xvecs_x = abs_unit_xvecs[0]
        unit_xvecs_y = abs_unit_xvecs[1] * _sign(abs_unit_xvecs[3])
        unit_xvecs_z = abs_unit_xvecs[2] * _sign(abs_unit_xvecs[4])
        unit_xvecs = np.array([unit_xvecs_x, unit_xvecs_y, unit_xvecs_z])
        sign_xvecs = _sign(unit_xvecs * ref_unit_xvecs)
        unit_xvecs *= sign_xvecs
        # 14, 15, 16: roll angles
        cos_rolls = reg_values[14]
        sin_rolls = reg_values[15] * _sign(reg_values[16])
        roll_vecs = np.array([cos_rolls, sin_rolls])
        roll_vecs /= np.maximum(np.linalg.norm(roll_vecs), 1e-6)
        # 17, 18, 19: velocities
        if self.has_velocity:
            velocities = reg_values[[17, 18, 19]]
        else:
            velocities = None
        
        ## group nms
        boxes_3d = self._group_nms(
            seg_scores, cen_scores, seg_classes, 
            centers, sizes, unit_xvecs, roll_vecs, velocities,
        )

        return boxes_3d
