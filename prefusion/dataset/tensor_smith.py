import cv2
import torch
import numba
import numpy as np

from typing import Tuple, Dict, Union, Iterable
from torch import Tensor

from prefusion.registry import TENSOR_SMITHS
from .utils import (
    expand_line_2d, _sign, INF_DIST,
    vec_point2line_along_direction, 
    dist_point2line_along_direction,
    choose_index
)
from .transform import (
    CameraImage, CameraSegMask, CameraDepth,
    Bbox3D, Polyline3D, SegBev, ParkingSlot3D
)

__all__ = [
    "CameraImageTensor", 
    "CameraDepthTensor", 
    "CameraSegTensor",
    "PlanarBbox3D", 
    "PlanarRectangularCuboid",
    "PlanarSquarePillar", 
    "PlanarCylinder3D", 
    "PlanarOrientedCylinder3D",
    "PlanarSegBev", 
    "PlanarPolyline3D", 
    "PlanarPolygon3D", 
    "PlanarParkingSlot3D",
]


class TensorSmith:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self, tensor_dict):
        raise NotImplementedError




@TENSOR_SMITHS.register_module()
class CameraImageTensor(TensorSmith):
    def __init__(self, 
            means: Union[list[float, float, float], tuple[float, float, float], float] = 128, 
            stds: Union[list[float, float, float], tuple[float, float, float], float] = 255
        ):
        if isinstance(means, Iterable):
            means = np.array(means)[..., None, None]
        if isinstance(stds, Iterable):
            stds = np.array(stds)[..., None, None]
        self.means = means
        self.stds = stds

    def __call__(self, transformable: CameraImage):
        tensor_dict = dict(
            img=torch.tensor((np.float32(transformable.img.transpose(2, 0, 1)) - self.means) / self.stds),
            ego_mask=torch.tensor(transformable.ego_mask),
        )
        return tensor_dict



@TENSOR_SMITHS.register_module()
class CameraDepthTensor(TensorSmith):
    def __init__(self, channels):
        pass

    def __call__(self, transformable: CameraDepth):
        tensor_dict = dict(
            img=torch.tensor(transformable.img),
            ego_mask=torch.tensor(transformable.ego_mask),
        )
        return tensor_dict



@TENSOR_SMITHS.register_module()
class CameraSegTensor(TensorSmith):
    def __init__(self, class_sequence, class_combines):
        self.class_sequence = class_sequence

    def __call__(self, transformable: CameraSegMask):
        raise NotImplementedError



def get_bev_intrinsics(voxel_shape, voxel_range):
    _, X, Y = voxel_shape    
    fx = X / (voxel_range[1][1] - voxel_range[1][0])
    fy = Y / (voxel_range[2][1] - voxel_range[2][0])
    cx = - voxel_range[1][0] * fx - 0.5
    cy = - voxel_range[2][0] * fy - 0.5

    return cx, cy, fx, fy


class PlanarTensorSmith(TensorSmith):
    def __init__(self, voxel_shape: tuple, voxel_range: Tuple[list, list, list]):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[list]

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



@TENSOR_SMITHS.register_module()
class PlanarBbox3D(PlanarTensorSmith):
    
    def __init__(self, 
                 voxel_shape: tuple, 
                 voxel_range: Tuple[list, list, list], 
                 use_bottom_center=False,
                 reverse_pre_conf: float=0.1,
                 reverse_nms_ratio: float=0.7):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]
        use_bottom_center : bool
        reverse_pre_conf : float
        reverse_nms_ratio : float

        Examples
        --------
        - voxel_shape=(6, 320, 160)
        - voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        - Z, X, Y = voxel_shape

        """
        super().__init__(voxel_shape, voxel_range)
        self.use_bottom_center = use_bottom_center
        self.reverse_pre_conf = reverse_pre_conf
        self.reverse_nms_ratio = reverse_nms_ratio

    
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
                center = element['translation'][:, 0]
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
        reg_im = np.zeros((20, X, Y), dtype=np.float32)
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
        
    
    @staticmethod
    def _is_in_bbox3d(delta_ij, sizes, xvec, yvec, zvec):
        return all([
            np.linalg.norm(delta_ij * xvec) < 0.5 * sizes[0],
            np.linalg.norm(delta_ij * yvec) < 0.5 * sizes[1],
            np.linalg.norm(delta_ij * zvec) < 0.5 * sizes[2]
        ])
    
    
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
                        if self._is_in_bbox3d(delta_ij, sizes_i * self.reverse_nms_ratio, unit_xvec_i, unit_yvec_i, unit_zvec_i):
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
            area_score = min(1, mean_count / mean_area)
            # get center
            mean_center = (centers[:, group] * scores[group][None]).sum(1) / score_sum
            if self.use_bottom_center:
                mean_center = mean_center + 0.5 * unit_zvec * mean_size[2]
            mean_velocity = (velocities[:, group] * scores[group][None]).sum(1) / score_sum
            bbox_3d = {
                'confs': mean_classes,
                'area_score': area_score,
                'size': mean_size,
                'rotation': mean_rmat,
                'translation': mean_center,
                'velocity': mean_velocity,
            }
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
        velocities = reg_values[[17, 18, 19]]
        
        ## group nms
        boxes_3d = self._group_nms(
            seg_scores, cen_scores, seg_classes, 
            centers, sizes, unit_xvecs, roll_vecs, velocities,
        )

        return boxes_3d



@TENSOR_SMITHS.register_module()
class PlanarRectangularCuboid(PlanarBbox3D):
    """For no exact heading angle rectangle objects like speedbumps.
    """
    
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
        class_inds = []
        attr_lists = []
        attr_available = 'attrs' in transformable.dictionary
        for element in transformable.elements:
            if element['class'] in transformable.dictionary['classes']:
                # unit vector of x_axis/y_axis of the box
                unit_xvecs.append(element['rotation'][:, 0])
                unit_yvecs.append(element['rotation'][:, 1])
                # get the position of the box
                center = element['translation'][:, 0]
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
        reg_im = np.zeros((14, X, Y), dtype=np.float32)
        for unit_xvec, unit_yvec, points_bev, center, size, class_ind, attr_list in zip(
            unit_xvecs, unit_yvecs, all_points_bev, centers, box_sizes, class_inds, attr_lists
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
            # 6, 7, 8, 9, 10: absolute xvec
            reg_im[6] = reg_im[6] * (1 - region_box) + np.abs(unit_xvec[0]) * region_box
            reg_im[7] = reg_im[7] * (1 - region_box) + np.abs(unit_xvec[1]) * region_box
            reg_im[8] = reg_im[8] * (1 - region_box) + np.abs(unit_xvec[2]) * region_box
            reg_im[9] = reg_im[9] * (1 - region_box) + unit_xvec[0] * unit_xvec[1] * region_box
            reg_im[10] = reg_im[10] * (1 - region_box) + unit_xvec[0] * unit_xvec[2] * region_box
            # 11, 12, 13: abs roll angle of yvec and xvec_bev_vertial                
            cos_roll, sin_roll = self._get_roll_from_xyvecs(unit_xvec, unit_yvec)
            reg_im[11] = reg_im[11] * (1 - region_box) + abs(cos_roll) * region_box
            reg_im[12] = reg_im[12] * (1 - region_box) + abs(sin_roll) * region_box
            reg_im[13] = reg_im[13] * (1 - region_box) + cos_roll * sin_roll * region_box
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

    def _group_nms(self, seg_scores, cen_scores, seg_classes, centers, sizes, unit_xvecs, roll_vecs):
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
                        if self._is_in_bbox3d(delta_ij, sizes_i * self.reverse_nms_ratio, unit_xvec_i, unit_yvec_i, unit_zvec_i):
                            kept_inds.append(j)
                            grouped_inds.append(j)
        ## get mean bbox in group
        _, _, fx, fy = self.bev_intrinsics
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
            area_score = min(1, mean_count / mean_area)
            mean_center = (centers[:, group] * scores[group][None]).sum(1) / score_sum
            if self.use_bottom_center:
                mean_center = mean_center + 0.5 * unit_zvec * mean_size[2]
            bbox_3d = {
                'confs': mean_classes,
                'area_score': area_score,
                'size': mean_size,
                'rotation': mean_rmat,
                'translation': mean_center
            }
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
            6, 7, 8, 9, 10: absolute xvec, ego coords
            11, 12, 13: abs roll angle of yvec and xvec_bev_vertial, intrinsic rotation
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
        # 6, 7, 8, 9, 10: absolute xvec, ego coords
        abs_unit_xvecs = reg_values[[6, 7, 8, 9, 10]]
        # calculte unit xvecs
        unit_xvecs_x = abs_unit_xvecs[0]
        unit_xvecs_y = abs_unit_xvecs[1] * _sign(abs_unit_xvecs[3])
        unit_xvecs_z = abs_unit_xvecs[2] * _sign(abs_unit_xvecs[4])
        unit_xvecs = np.array([unit_xvecs_x, unit_xvecs_y, unit_xvecs_z])
        # 11, 12, 13: abs roll angle of yvec and xvec_bev_vertial, intrinsic rotation   
        cos_rolls = reg_values[11]
        sin_rolls = reg_values[12] * _sign(reg_values[13])
        roll_vecs = np.array([cos_rolls, sin_rolls])
        roll_vecs /= np.maximum(np.linalg.norm(roll_vecs), 1e-6)
        
        ## group nms
        boxes_3d = self._group_nms(
            seg_scores, cen_scores, seg_classes, centers, sizes, unit_xvecs, roll_vecs
        )

        return boxes_3d




@TENSOR_SMITHS.register_module()
class PlanarSquarePillar(PlanarTensorSmith):
    def __init__(self, 
                 voxel_shape: tuple, 
                 voxel_range: Tuple[list, list, list], 
                 use_bottom_center: bool=True,
                 reverse_pre_conf: float=0.1,
                 reverse_nms_ratio: float=1):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]
        use_bottom_center : bool, optional
        reverse_pre_conf : float, optional
        reverse_nms_ratio : float, optional

        Examples
        --------
        - voxel_shape=(6, 320, 160)
        - voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        - Z, X, Y = voxel_shape

        """
        super().__init__(voxel_shape, voxel_range)
        self.use_bottom_center = use_bottom_center
        self.reverse_pre_conf = reverse_pre_conf
        self.reverse_nms_ratio = reverse_nms_ratio
    
    
    @staticmethod
    def _get_yaw_from_zxvecs(zvec, xvec):
        """
        Parameters
        ----------
        zvec : array(3,)
            zvec of object
        xvec : array(3,)
            xvec of object

        Returns
        -------
        yaw : float
            intrinsical yaw
        """
        # intrinsics yaw or extrinsic yaw?
        # project zvec to xz_plane
        zvec_vert_xz_plane = np.array([zvec[2], 0, -zvec[0]])
        assert np.linalg.norm(zvec_vert_xz_plane) > 1e-3
        # unit zvec_vert_xz_plane
        a = zvec_vert_xz_plane / np.linalg.norm(zvec_vert_xz_plane)
        b = xvec / np.linalg.norm(xvec)
        cos_yaw = np.dot(a, b)
        cross = np.cross(a, b)
        sin_sign = _sign(np.dot(cross, zvec))
        sin_yaw = sin_sign * np.linalg.norm(cross)
        return np.arctan2(sin_yaw, cos_yaw)

        
    def __call__(self, transformable: Bbox3D):
        Z, X, Y = self.voxel_shape
        cx, cy, fx, fy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev
        
        unit_xvecs = []
        unit_zvecs = []
        centers = []
        all_points_bev = []
        box_sizes = []
        class_inds = []
        attr_lists = []
        attr_available = 'attrs' in transformable.dictionary
        for element in transformable.elements:
            if element['class'] in transformable.dictionary['classes']:
                # unit vector of x_axis/z_axis of the box
                unit_xvecs.append(element['rotation'][:, 0])
                unit_zvecs.append(element['rotation'][:, 2])
                # get the position of the box
                center = element['translation'][:, 0]
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
        seg_im = np.zeros((1 + num_class_channels + num_attr_channels, X, Y), dtype=np.float32)
        cen_im = np.zeros((1, X, Y), dtype=np.float32)
        reg_im = np.zeros((20, X, Y), dtype=np.float32)
        for unit_xvec, unit_zvec, points_bev, center, size, class_ind, attr_list in zip(
            unit_xvecs, unit_zvecs, all_points_bev, centers, box_sizes, class_inds, attr_lists
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
            # 6, 7, 8: unit zvec
            reg_im[6] = reg_im[6] * (1 - region_box) + unit_zvec[0] * region_box
            reg_im[7] = reg_im[7] * (1 - region_box) + unit_zvec[1] * region_box
            reg_im[8] = reg_im[8] * (1 - region_box) + unit_zvec[2] * region_box
            # 9, 10: yaw angle of zvec, intrinsical
            yaw_angle = self._get_yaw_from_zxvecs(unit_zvec, unit_xvec)
            reg_im[9] = reg_im[9] * (1 - region_box) + np.cos(4 * yaw_angle) * region_box
            reg_im[10] = reg_im[10] * (1 - region_box) + np.sin(4 * yaw_angle) * region_box
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
    def _get_xyvec_from_zvec_and_yaw(zvecs, vecs_4yaw):
        zvecs_vert_xz_plane = np.array([zvecs[2], 0, -zvecs[0]])
        if np.linalg.norm(zvecs_vert_xz_plane) < 1e-3:
            zvecs_vert_xz_plane = np.array([
                np.ones_like(zvecs[0]), 
                np.zeros_like(zvecs[0]), 
                np.zeros_like(zvecs[0]),
            ])
        zvecs /= np.linalg.norm(zvecs, axis=0)
        zvecs_vert_xz_plane /= np.linalg.norm(zvecs_vert_xz_plane, axis=0)        
        yaws = np.arctan2(vecs_4yaw[[1]], vecs_4yaw[[0]]) / 4
        cos_yaws = np.cos(yaws)
        sin_yaws = np.sin(yaws)
        xvecs = zvecs_vert_xz_plane * cos_yaws + np.cross(zvecs, zvecs_vert_xz_plane, axis=0) * sin_yaws
        xvecs /= np.linalg.norm(xvecs, axis=0)
        yvecs = np.cross(zvecs, xvecs, axis=0)
        return xvecs, yvecs
        
    
    @staticmethod
    def _is_in_bbox3d(delta_ij, sizes, xvec, yvec, zvec):
        return all([
            np.linalg.norm(delta_ij * xvec) < 0.5 * sizes[0],
            np.linalg.norm(delta_ij * yvec) < 0.5 * sizes[1],
            np.linalg.norm(delta_ij * zvec) < 0.5 * sizes[2]
        ])

    
    def _group_nms(self, seg_scores, cen_scores, seg_classes, centers, sizes, unit_zvecs, vecs_4yaw):
        scores = seg_scores * cen_scores
        ranked_inds = np.argsort(scores)[::-1]
        kept_groups = []
        kept_inds = []
        # group inds
        for i in ranked_inds:
            if i not in kept_inds:
                center_i = centers[:, i]
                sizes_i = sizes[:, i]
                unit_zvec_i = unit_zvecs[:, i]
                vec_4yaw_i = vecs_4yaw[:, i]
                unit_xvec_i, unit_yvec_i = self._get_xyvec_from_zvec_and_yaw(unit_zvec_i, vec_4yaw_i)
                kept_inds.append(i)
                grouped_inds = [i]
                kept_groups.append(grouped_inds)
                for j in ranked_inds:
                    if j not in kept_inds:
                        center_j = centers[:, j]
                        delta_ij = center_i - center_j
                        if self._is_in_bbox3d(delta_ij, sizes_i * self.reverse_nms_ratio, unit_xvec_i, unit_yvec_i, unit_zvec_i):
                            kept_inds.append(j)
                            grouped_inds.append(j)
        ## get mean bbox in group
        _, _, fx, fy = self.bev_intrinsics
        pred_pillars = []
        for group in kept_groups:
            # use score weighted mean
            score_sum = scores[group].sum() + 1e-6
            mean_classes = (seg_classes[:, group] * scores[group][None]).sum(1) / score_sum
            # get mean_unit_zvec
            # average all zvecs, zvecs may not have same directions
            mean_unit_zvec = (unit_zvecs[:, group] * scores[group][None]).sum(1) / score_sum
            # # align all zvecs to the first one, then average all zvecs again; maybe unnecessary
            # unit_zvec_0 = unit_zvecs[:, group[0]]
            # for i in group[1:]:
            #     if np.sum(unit_zvec_0 * unit_zvecs[:, i]) < 0:
            #         unit_zvecs[:, i] *= -1
            # mean_unit_zvec = (unit_zvecs[:, group] * scores[group][None]).sum(1) / score_sum
            mean_vec_4yaw = (vecs_4yaw[:, group] * scores[group][None]).sum(1) / score_sum
            unit_xvec, unit_yvec = self._get_xyvec_from_zvec_and_yaw(mean_unit_zvec, mean_vec_4yaw)
            # rotation matrix
            mean_rmat = np.float32([unit_xvec, unit_yvec, mean_unit_zvec]).T
            mean_size = (sizes[:, group] * scores[group][None]).sum(1) / score_sum
            # get area score
            mean_count = seg_classes[0, group].sum()
            mean_area = mean_size[0] * mean_size[1] * fx * fy * min(self.reverse_nms_ratio ** 2, 1)
            area_score = min(1, mean_count / mean_area)
            mean_center = (centers[:, group] * scores[group][None]).sum(1) / score_sum
            if self.use_bottom_center:
                mean_center = mean_center + 0.5 * mean_unit_zvec * mean_size[2]
            pillar_3d = {
                'confs': mean_classes,
                'area_score': area_score,
                'size': mean_size,
                'rotation': mean_rmat,
                'translation': mean_center
            }
            pred_pillars.append(pillar_3d)
        return pred_pillars
    
    
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
            0, 1: center x y, in bev coord
            2: center z
            3, 4, 5: size l, w, h
            6, 7, 8: unit zvec
            9, 10: yaw angle of zvec
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
        # 6, 7, 8: unit zvecs in ego coords
        unit_zvecs = reg_values[[6, 7, 8]]
        # 9, 10: yaw angles
        cos4yaws = reg_values[9]
        sin4yaws = reg_values[10]
        vecs_4yaw = np.array([cos4yaws, sin4yaws])
        vecs_4yaw /= np.maximum(np.linalg.norm(vecs_4yaw), 1e-6)
        
        ## group nms
        pillars_3d = self._group_nms(
            seg_scores, cen_scores, seg_classes,
            centers, sizes, unit_zvecs, vecs_4yaw
        )
        
        return pillars_3d
        
    


@TENSOR_SMITHS.register_module()
class PlanarCylinder3D(PlanarTensorSmith):

    def __init__(self, 
                 voxel_shape: tuple, 
                 voxel_range: Tuple[list, list, list], 
                 use_bottom_center: bool=False,
                 reverse_pre_conf: float=0.1,
                 reverse_nms_ratio: float=1):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]
        use_bottom_center : bool
        reverse_pre_conf : float
        reverse_nms_ratio : float

        Examples
        --------
        - voxel_shape=(6, 320, 160)
        - voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        - Z, X, Y = voxel_shape

        """
        super().__init__(voxel_shape, voxel_range)
        self.use_bottom_center = use_bottom_center
        self.reverse_pre_conf = reverse_pre_conf
        self.reverse_nms_ratio = reverse_nms_ratio
        
    
    def __call__(self, transformable: Bbox3D):
        Z, X, Y = self.voxel_shape
        cx, cy, fx, fy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev
        
        unit_zvecs = []
        centers = []
        radii = []
        heights = []
        class_inds = []
        attr_lists = []
        attr_available = 'attrs' in transformable.dictionary
        for element in transformable.elements:
            if element['class'] in transformable.dictionary['classes']:
                # unit vector of z_axis of the box
                unit_zvecs.append(element['rotation'][:, 2])
                # get the position of the box
                center = element['translation'][:, 0]
                if self.use_bottom_center:
                    center -= 0.5 * element['size'][2] * element['rotation'][:, 2]
                centers.append(np.array([
                    center[1] * fy + cy,
                    center[0] * fx + cx,
                    center[2]
                ], dtype=np.float32))
                l, w, h = element['size']
                radius = 0.5 * max(l, w)  # l, w should be the same
                radii.append(radius)
                heights.append(h)
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
        seg_im = np.zeros((1 + num_class_channels + num_attr_channels, X, Y), dtype=np.float32)
        cen_im = np.zeros((1, X, Y), dtype=np.float32)
        reg_im = np.zeros((8, X, Y), dtype=np.float32)
        for unit_zvec, center, radius, height, class_ind, attr_list in zip(
            unit_zvecs, centers, radii, heights, class_inds, attr_lists
        ):
            center_bev = np.round(center[:2]).astype(int)
            radius_bev = np.round(max(1, radius * max(fx, fy))).astype(int)  # fx, fy should be the same
            region_obj = cv2.circle(np.zeros((X, Y), dtype=np.float32), center_bev, radius_bev, 1, -1)
            ## gen segmentation
            cv2.circle(seg_im[0], center_bev, radius_bev, 1, -1)
            cv2.circle(seg_im[1 + class_ind], center_bev, radius_bev, 1, -1)
            for attr_ind in attr_list:
                cv2.circle(seg_im[1 + num_class_channels + attr_ind], center_bev, radius_bev, 1, -1)
            ## gen regression
            # 0, 1: center x y, in bev coord
            vec2center = center[:2][..., None, None] - points_grid_bev
            reg_im[[0, 1]] = reg_im[[0, 1]] * (1 - region_obj) + vec2center * region_obj
            # 2: center z
            reg_im[2] = reg_im[2] * (1 - region_obj) + center[2] * region_obj
            # 3, 4: size radius, height
            reg_im[3] = reg_im[3] * (1 - region_obj) + radius * region_obj
            reg_im[4] = reg_im[4] * (1 - region_obj) + height * region_obj
            # 5, 6, 7: unit zvec
            reg_im[5] = reg_im[5] * (1 - region_obj) + unit_zvec[0] * region_obj
            reg_im[6] = reg_im[6] * (1 - region_obj) + unit_zvec[1] * region_obj
            reg_im[7] = reg_im[7] * (1 - region_obj) + unit_zvec[2] * region_obj
            ## gen centerness
            centerness = (radius_bev ** 2 - (reg_im[0] ** 2 + reg_im[1] ** 2)) / radius ** 2
            cen_im[0] = cen_im[0] * (1 - region_obj) + centerness * region_obj
        ## tensor
        tensor_data= {
            'seg': torch.tensor(seg_im, dtype=torch.float32),
            'cen': torch.tensor(cen_im, dtype=torch.float32),
            'reg': torch.tensor(reg_im, dtype=torch.float32)
        }
        return tensor_data
        
    
    @staticmethod
    def _is_in_cylinder3d(delta_ij, sizes, zvec):
        zvec_vertical = np.array([0, zvec[2], -zvec[1]])
        zvec_vertical /= np.linalg.norm(zvec_vertical)
        return all([
            np.linalg.norm(delta_ij * zvec_vertical) < sizes[0],
            np.linalg.norm(delta_ij * zvec) < 0.5 * sizes[1]
        ])

    
    def _group_nms(self, seg_scores, cen_scores, seg_classes, centers, sizes, unit_zvecs):
        scores = seg_scores * cen_scores
        ranked_inds = np.argsort(scores)[::-1]
        kept_groups = []
        kept_inds = []
        # group inds
        for i in ranked_inds:
            if i not in kept_inds:
                center_i = centers[:, i]
                sizes_i = sizes[:, i]
                unit_zvec_i = unit_zvecs[:, i]
                kept_inds.append(i)
                grouped_inds = [i]
                kept_groups.append(grouped_inds)
                for j in ranked_inds:
                    if j not in kept_inds:
                        center_j = centers[:, j]
                        delta_ij = center_i - center_j
                        if self._is_in_cylinder3d(delta_ij, sizes_i * self.reverse_nms_ratio, unit_zvec_i):
                            kept_inds.append(j)
                            grouped_inds.append(j)
        ## get mean bbox in group
        _, _, fx, fy = self.bev_intrinsics
        pred_cylinders = []
        for group in kept_groups:
            # use score weighted mean
            score_sum = scores[group].sum() + 1e-6
            mean_classes = (seg_classes[:, group] * scores[group][None]).sum(1) / score_sum
            # get mean_unit_zvec
            mean_unit_zvec = (unit_zvecs[:, group] * scores[group][None]).sum(1) / score_sum
            mean_size = (sizes[:, group] * scores[group][None]).sum(1) / score_sum
            # get area score
            mean_count = seg_classes[0, group].sum()
            mean_area = mean_size[0] * mean_size[0] * fx * fy * min(self.reverse_nms_ratio ** 2, 1) * np.pi
            area_score = min(1, mean_count / mean_area)
            mean_center = (centers[:, group] * scores[group][None]).sum(1) / score_sum
            if self.use_bottom_center:
                mean_center = mean_center + 0.5 * mean_unit_zvec * mean_size[1]
            cylinder_3d = {
                'confs': mean_classes,
                'area_score': area_score,
                'radius': mean_size[0],
                'height': mean_size[1],
                'zvec': mean_unit_zvec,
                'translation': mean_center
            }
            pred_cylinders.append(cylinder_3d)
        return pred_cylinders
    
    
    
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
            0, 1: center x y, in bev coord
            2: center z
            3, 4: size radius, height
            5, 6, 7: unit zvec
        ```
        """
        seg_pred = tensor_dict['seg'].detach().cpu().numpy()
        cen_pred = tensor_dict['cen'].detach().cpu().numpy()
        reg_pred = tensor_dict['reg'].detach().cpu().numpy()
        ## pickup obj points
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
        # 3, 4: size radius, height in ego coords
        sizes = reg_values[[3, 4]]
        # 5, 6, 7: unit zvecs in ego coords
        unit_zvecs = reg_values[[5, 6, 7]]
        
        ## group nms
        cylinders_3d = self._group_nms(
            seg_scores, cen_scores, seg_classes, centers, sizes, unit_zvecs
        )
        return cylinders_3d




@TENSOR_SMITHS.register_module()
class PlanarOrientedCylinder3D(PlanarTensorSmith):

    def __init__(self, 
                 voxel_shape: tuple, 
                 voxel_range: Tuple[list, list, list], 
                 use_bottom_center: bool=False,
                 reverse_pre_conf: float=0.1,
                 reverse_nms_ratio: float=1):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]
        use_bottom_center : bool
        reverse_pre_conf : float
        reverse_nms_ratio : float

        Examples
        --------
        - voxel_shape=(6, 320, 160)
        - voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        - Z, X, Y = voxel_shape

        """
        super().__init__(voxel_shape, voxel_range)
        self.use_bottom_center = use_bottom_center
        self.reverse_pre_conf = reverse_pre_conf
        self.reverse_nms_ratio = reverse_nms_ratio
        
    
    @staticmethod
    def _get_yaw_from_zxvecs(zvec, xvec):
        """
        Parameters
        ----------
        zvec : array(3,)
            zvec of object
        xvec : array(3,)
            xvec of object

        Returns
        -------
        yaw : float
            intrinsical yaw
        """
        # project zvec to xz_plane
        zvec_vert_xz_plane = np.array([zvec[2], 0, -zvec[0]])
        assert np.linalg.norm(zvec_vert_xz_plane) > 1e-3
        # unit zvec_vert_xz_plane
        a = zvec_vert_xz_plane / np.linalg.norm(zvec_vert_xz_plane)
        b = xvec / np.linalg.norm(xvec)
        cos_yaw = np.dot(a, b)
        cross = np.cross(a, b)
        sin_sign = _sign(np.dot(cross, zvec))
        sin_yaw = sin_sign * np.linalg.norm(cross)
        return np.arctan2(sin_yaw, cos_yaw)

        
    def __call__(self, transformable: Bbox3D):
        Z, X, Y = self.voxel_shape
        cx, cy, fx, fy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev
        
        unit_xvecs = []
        unit_zvecs = []
        centers = []
        radii = []
        heights = []
        velocities = []
        class_inds = []
        attr_lists = []
        attr_available = 'attrs' in transformable.dictionary
        for element in transformable.elements:
            if element['class'] in transformable.dictionary['classes']:
                # unit vector of x_axis/z_axis of the box
                unit_xvecs.append(element['rotation'][:, 0])
                unit_zvecs.append(element['rotation'][:, 2])
                # get the position of the box
                center = element['translation'][:, 0]
                if self.use_bottom_center:
                    center -= 0.5 * element['size'][2] * element['rotation'][:, 2]
                centers.append(np.array([
                    center[1] * fy + cy,
                    center[0] * fx + cx,
                    center[2]
                ], dtype=np.float32))
                l, w, h = element['size']
                radius = 0.5 * max(l, w)  # l, w should be the same
                radii.append(radius)
                heights.append(h)
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
        seg_im = np.zeros((1 + num_class_channels + num_attr_channels, X, Y), dtype=np.float32)
        cen_im = np.zeros((1, X, Y), dtype=np.float32)
        reg_im = np.zeros((13, X, Y), dtype=np.float32)
        for unit_xvec, unit_zvec, center, radius, height, velocity, class_ind, attr_list in zip(
            unit_xvecs, unit_zvecs, centers, radii, heights, velocities, class_inds, attr_lists
        ):
            center_bev = np.round(center[:2]).astype(int)
            radius_bev = np.round(max(1, radius * max(fx, fy))).astype(int)  # fx, fy should be the same
            region_obj = cv2.circle(np.zeros((X, Y), dtype=np.float32), center_bev, radius_bev, 1, -1)
            ## gen segmentation
            cv2.circle(seg_im[0], center_bev, radius_bev, 1, -1)
            cv2.circle(seg_im[1 + class_ind], center_bev, radius_bev, 1, -1)
            for attr_ind in attr_list:
                cv2.circle(seg_im[1 + num_class_channels + attr_ind], center_bev, radius_bev, 1, -1)
            ## gen regression
            # 0, 1: center x y, in bev coord
            vec2center = center[:2][..., None, None] - points_grid_bev
            reg_im[[0, 1]] = reg_im[[0, 1]] * (1 - region_obj) + vec2center * region_obj
            # 2: center z
            reg_im[2] = reg_im[2] * (1 - region_obj) + center[2] * region_obj
            # 3, 4: size radius, height
            reg_im[3] = reg_im[3] * (1 - region_obj) + radius * region_obj
            reg_im[4] = reg_im[4] * (1 - region_obj) + height * region_obj
            # 5, 6, 7: unit zvec
            reg_im[5] = reg_im[5] * (1 - region_obj) + unit_zvec[0] * region_obj
            reg_im[6] = reg_im[6] * (1 - region_obj) + unit_zvec[1] * region_obj
            reg_im[7] = reg_im[7] * (1 - region_obj) + unit_zvec[2] * region_obj
            # 8, 9: yaw angle of zvec, intrinsical
            yaw_angle = self._get_yaw_from_zxvecs(unit_zvec, unit_xvec)
            reg_im[8] = reg_im[8] * (1 - region_obj) + np.cos(yaw_angle) * region_obj
            reg_im[9] = reg_im[9] * (1 - region_obj) + np.sin(yaw_angle) * region_obj
            # 10, 11, 12: velocity
            reg_im[10] = reg_im[10] * (1 - region_obj) + velocity[0] * region_obj
            reg_im[11] = reg_im[11] * (1 - region_obj) + velocity[1] * region_obj
            reg_im[12] = reg_im[12] * (1 - region_obj) + velocity[2] * region_obj
            ## gen centerness
            centerness = (radius_bev ** 2 - (reg_im[0] ** 2 + reg_im[1] ** 2)) / radius ** 2
            cen_im[0] = cen_im[0] * (1 - region_obj) + centerness * region_obj
        ## tensor
        tensor_data = {
            'seg': torch.tensor(seg_im, dtype=torch.float32),
            'cen': torch.tensor(cen_im, dtype=torch.float32),
            'reg': torch.tensor(reg_im, dtype=torch.float32)
        }
        return tensor_data
    

    @staticmethod
    def _get_xyvec_from_zvec_and_yaw(zvecs, vecs_yaw):
        zvecs_vert_xz_plane = np.array([zvecs[2], 0, -zvecs[0]])
        if np.linalg.norm(zvecs_vert_xz_plane) < 1e-3:
            zvecs_vert_xz_plane = np.array([
                np.ones_like(zvecs[0]), 
                np.zeros_like(zvecs[0]), 
                np.zeros_like(zvecs[0]),
            ])
        zvecs /= np.linalg.norm(zvecs, axis=0)
        zvecs_vert_xz_plane /= np.linalg.norm(zvecs_vert_xz_plane, axis=0)
        cos_yaws, sin_yaws = vecs_yaw
        xvecs = zvecs_vert_xz_plane * cos_yaws + np.cross(zvecs, zvecs_vert_xz_plane, axis=0) * sin_yaws
        xvecs /= np.linalg.norm(xvecs, axis=0)
        yvecs = np.cross(zvecs, xvecs, axis=0)
        return xvecs, yvecs
        
    
    @staticmethod
    def _is_in_cylinder3d(delta_ij, sizes, xvec, yvec, zvec):
        return all([
            np.linalg.norm(delta_ij * xvec) < sizes[0],
            np.linalg.norm(delta_ij * yvec) < sizes[0],
            np.linalg.norm(delta_ij * zvec) < 0.5 * sizes[1]
        ])

    
    def _group_nms(self, seg_scores, cen_scores, seg_classes, centers, sizes, unit_zvecs, vecs_yaw, velocities):
        scores = seg_scores * cen_scores
        ranked_inds = np.argsort(scores)[::-1]
        kept_groups = []
        kept_inds = []
        # group inds
        for i in ranked_inds:
            if i not in kept_inds:
                center_i = centers[:, i]
                sizes_i = sizes[:, i]
                unit_zvec_i = unit_zvecs[:, i]
                vec_yaw_i = vecs_yaw[:, i]
                unit_xvec_i, unit_yvec_i = self._get_xyvec_from_zvec_and_yaw(unit_zvec_i, vec_yaw_i)
                kept_inds.append(i)
                grouped_inds = [i]
                kept_groups.append(grouped_inds)
                for j in ranked_inds:
                    if j not in kept_inds:
                        center_j = centers[:, j]
                        delta_ij = center_i - center_j
                        if self._is_in_cylinder3d(delta_ij, sizes_i * self.reverse_nms_ratio, unit_xvec_i, unit_yvec_i, unit_zvec_i):
                            kept_inds.append(j)
                            grouped_inds.append(j)
        ## get mean bbox in group
        _, _, fx, fy = self.bev_intrinsics
        pred_cylinders = []
        for group in kept_groups:
            # use score weighted mean
            score_sum = scores[group].sum() + 1e-6
            mean_classes = (seg_classes[:, group] * scores[group][None]).sum(1) / score_sum
            # get mean_unit_zvec
            mean_unit_zvec = (unit_zvecs[:, group] * scores[group][None]).sum(1) / score_sum
            mean_vec_yaw = (vecs_yaw[:, group] * scores[group][None]).sum(1) / score_sum
            unit_xvec, unit_yvec = self._get_xyvec_from_zvec_and_yaw(mean_unit_zvec, mean_vec_yaw)
            # rotation matrix
            mean_rmat = np.float32([unit_xvec, unit_yvec, mean_unit_zvec]).T
            mean_size = (sizes[:, group] * scores[group][None]).sum(1) / score_sum
            # get area score
            mean_count = seg_classes[0, group].sum()
            mean_area = mean_size[0] * mean_size[0] * fx * fy * min(self.reverse_nms_ratio ** 2, 1) * np.pi
            area_score = min(1, mean_count / mean_area)
            mean_center = (centers[:, group] * scores[group][None]).sum(1) / score_sum
            if self.use_bottom_center:
                mean_center = mean_center + 0.5 * mean_unit_zvec * mean_size[1]
            mean_velocity = (velocities[:, group] * scores[group][None]).sum(1) / score_sum
            cylinder_3d = {
                'confs': mean_classes,
                'area_score': area_score,
                'size': mean_size[[0, 0, 1]] * [2, 2, 1],
                'rotation': mean_rmat,
                'translation': mean_center,
                'velocity': mean_velocity
            }
            pred_cylinders.append(cylinder_3d)
        return pred_cylinders
    
    
    
    def reverse(self, tensor_dict):
        """
        Parameters
        ----------
        tensor_dict : dict

        Notes
        -----
        ```
        seg_im  # 分割图
        cen_im  # 中心图
        reg_im  # 回归图
            0, 1: center x y, in bev coord
            2: center z
            3, 4: size radius, height
            5, 6, 7: unit zvec
            8, 9: yaw angle of zvec, intrinsical
            10, 11, 12: velocity, ego coords
        ```
        """
        seg_pred = tensor_dict['seg'].detach().cpu().numpy()
        cen_pred = tensor_dict['cen'].detach().cpu().numpy()
        reg_pred = tensor_dict['reg'].detach().cpu().numpy()
        ## pickup obj points
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
        # 3, 4: size radius, height in ego coords
        sizes = reg_values[[3, 4]]
        # 5, 6, 7: unit zvecs in ego coords
        unit_zvecs = reg_values[[5, 6, 7]]
        # 8, 9: yaw angles
        cos_yaws = reg_values[8]
        sin_yaws = reg_values[9]
        vecs_yaw = np.array([cos_yaws, sin_yaws])
        vecs_yaw /= np.maximum(np.linalg.norm(vecs_yaw), 1e-6)
        # 10, 11, 12: velocities
        velocities = reg_values[[10, 11, 12]]
        
        ## group nms
        cylinders_3d = self._group_nms(
            seg_scores, cen_scores, seg_classes,
            centers, sizes, unit_zvecs, vecs_yaw, velocities
        )
        return cylinders_3d





@TENSOR_SMITHS.register_module()
class PlanarSegBev(PlanarTensorSmith):
    def __call__(self, transformable: SegBev):
        raise NotImplementedError




@TENSOR_SMITHS.register_module()
class PlanarPolyline3D(PlanarTensorSmith):
    
    def __init__(self, 
                 voxel_shape: tuple, 
                 voxel_range: Tuple[list, list, list],
                 reverse_pre_conf: float=0.1,
                 reverse_group_dist_thresh: float=0.2,
                 reverse_link_max_adist: float=1.5):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]
        reverse_pre_conf : float
        reverse_group_dist_thresh : float
        reverse_link_max_adist : float

        Examples
        --------
        - voxel_shape=(6, 320, 160)
        - voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        - Z, X, Y = voxel_shape

        """
        super().__init__(voxel_shape, voxel_range)
        self.reverse_pre_conf = reverse_pre_conf
        self.reverse_group_dist_thresh = reverse_group_dist_thresh
        self.reverse_link_max_adist = reverse_link_max_adist
        
    def __call__(self, transformable: Polyline3D):
        """
        Parameters
        ----------
        transformable : Polyline3D

        Returns
        -------
        tensor_dict : dict

        Notes
        -----
        ```
        seg_im  # 分割图
        reg_im  # 回归图
            0: dist_im  # 每个分割图上的点到最近线段的垂直距离
            1,2: vert_vec_im  # 每个分割图上的点到最近线段的向量
            3,4,5: abs_dir_im  # 每个分割图上的点所在线段的广义单位方向，|nx|, |ny|, nx * ny
            6 height_im # 每个分割图上的点的高度分布图
        ```
        """
        Z, X, Y = self.voxel_shape
        cx, cy, fx, fy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev

        polylines = []
        class_inds = []
        attr_lists = []
        attr_available = 'attrs' in transformable.dictionary
        for element in transformable.elements:
            if element['class'] in transformable.dictionary['classes']:
                points = element['points']
                polylines.append(np.array([
                    points[..., 1] * fy + cy,
                    points[..., 0] * fx + cx,
                    points[..., 2]
                ]).T)
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
        seg_im = np.zeros((1 + num_class_channels + num_attr_channels, X, Y))

        line_ims = []
        dist_ims = []
        vec_ims = []
        dir_ims = []
        height_ims = []

        linewidth = int(max(0.51, abs(fx * 0.2)))

        for polyline, class_ind, attr_list in zip(polylines, class_inds, attr_lists):
            for line_3d in zip(polyline[:-1], polyline[1:]):
                line_3d = np.float32(line_3d)
                line_bev = line_3d[:, :2]
                polygon = expand_line_2d(line_bev, radius=linewidth)
                polygon_int = np.round(polygon).astype(int)
                # seg_bev_im
                cv2.fillPoly(seg_im[0], [polygon_int], 1)
                cv2.fillPoly(seg_im[1 + class_ind], [polygon_int], 1)
                for attr_ind in attr_list:
                    cv2.fillPoly(seg_im[1 + num_class_channels + attr_ind], [polygon_int], 1)
                # line segment
                line_im = cv2.fillPoly(np.zeros((X, Y)), [polygon_int], 1)
                line_ims.append(line_im)
                # line direction regressions
                line_dir = line_bev[1] - line_bev[0]
                line_length = np.linalg.norm(line_dir)
                line_dir /= line_length
                line_dir_vert = line_dir[::-1] * [1, -1]
                vec_map = vec_point2line_along_direction(points_grid_bev, line_bev, line_dir_vert)
                dist_im = line_im * np.linalg.norm(vec_map, axis=0) + (1 - line_im) * INF_DIST
                vec_im = line_im * vec_map
                abs_dir_im = line_im * np.float32([
                    np.abs(line_dir[0]),
                    np.abs(line_dir[1]),
                    line_dir[0] * line_dir[1]
                ])[..., None, None]
                dist_ims.append(dist_im)
                vec_ims.append(vec_im)
                dir_ims.append(abs_dir_im)
                # height map
                h2s = (line_3d[1, 2] - line_3d[0, 2]) / max(1e-3, line_length)
                height_im =  line_3d[0, 2] + h2s * line_im * np.linalg.norm(
                    points_grid_bev + vec_map - line_bev[0][..., None, None], axis=0
                )
                height_ims.append(height_im)
        if len(dist_ims) > 0:
            index_im = np.argmin(np.array(dist_ims), axis=0)
            vec_im = choose_index(index_im, vec_ims)
            dist_im = choose_index(index_im, dist_ims)
            dir_im = choose_index(index_im, dir_ims)
            height_im = choose_index(index_im, height_ims)
        else:
            vec_im = np.zeros((2, X, Y))
            dist_im = np.zeros((X, Y))
            dir_im = np.zeros((3, X, Y))
            height_im = np.zeros((X, Y))

        reg_im = np.concatenate([
            seg_im[:1] * dist_im[None],
            vec_im,
            dir_im,
            height_im[None]
        ], axis=0)
        # TODO: add ignore_seg_mask according to ignore classes and attrs

        tensor_data = {
            'seg': torch.tensor(seg_im, dtype=torch.float32),
            'reg': torch.tensor(reg_im, dtype=torch.float32)
        }
        
        return tensor_data


    @staticmethod
    @numba.njit
    def _group_points(dst_points, seg_scores, dist_thresh):
        ranked_ind = np.argsort(seg_scores)[::-1]
        kept_groups = []
        kept_inds = []

        for i in ranked_ind:
            if i not in kept_inds:
                point_i = dst_points[:, i]
                kept_inds.append(i)
                grouped_inds = [i]
                kept_groups.append(grouped_inds)
                for j in ranked_ind:
                    if j not in kept_inds:
                        point_j = dst_points[:, j]
                        dist_ij = np.linalg.norm(point_i - point_j)
                        if dist_ij <= dist_thresh:
                            kept_inds.append(j)
                            grouped_inds.append(j)

        return kept_groups

    def _link_line_points(self, fused_points, fused_vecs, max_adist):
        points_ind = np.arange(fused_points.shape[1])
        # get groups of line segments
        line_segments = []
        kept_inds = {}

        for i in points_ind:
            if i not in kept_inds:
                kept_inds[i] = 1
                grouped_inds = [i]
                line_segments.append(grouped_inds)
                # go forward if direction is similar
                point_i_forward = fused_points[:, i]
                abs_vec_i_forward = fused_vecs[:, i]
                oriented_point_i_forward = [point_i_forward, abs_vec_i_forward]
                vec_i_forward = np.float32([abs_vec_i_forward[0], abs_vec_i_forward[1] * abs_vec_i_forward[2], 0])
                adist_forward_max_ind = i
                while True:
                    adist_forward_max = max_adist
                    for j in points_ind:
                        if j not in kept_inds:
                            point_j = fused_points[:, j]
                            abs_vec_j = fused_vecs[:, j]
                            oriented_point_j = [point_j, abs_vec_j]
                            point_vec = point_j - point_i_forward
                            ward = fast_inner_product(point_vec, vec_i_forward) # np.sum(point_vec * vec_i_forward)
                            if ward > 0:
                                adist_ij = _angle_hook_dist(oriented_point_i_forward, oriented_point_j, disth_thresh=max_adist)
                                if adist_ij < adist_forward_max:
                                    adist_forward_max = adist_ij
                                    adist_forward_max_ind = j
                    if adist_forward_max >= max_adist:
                        break
                    grouped_inds.append(adist_forward_max_ind)
                    kept_inds[adist_forward_max_ind] = 1
                    point_i_forward_old = point_i_forward
                    point_i_forward = fused_points[:, adist_forward_max_ind]
                    abs_vec_i_forward = fused_vecs[:, adist_forward_max_ind]
                    oriented_point_i_forward = [point_i_forward, abs_vec_i_forward]
                    vec_i_forward = np.float32([abs_vec_i_forward[0], abs_vec_i_forward[1] * abs_vec_i_forward[2], 0])
                    # if np.sum(vec_i_forward * (point_i_forward - point_i_forward_old)) <= 0:
                    if fast_inner_product(vec_i_forward, (point_i_forward - point_i_forward_old)) <= 0:
                        vec_i_forward *= -1
                    
                # go backward if direction is opposite
                point_i_backward = fused_points[:, i]
                abs_vec_i_backward = fused_vecs[:, i]
                oriented_point_i_backward = [point_i_backward, abs_vec_i_backward]
                vec_i_backward = np.float32([abs_vec_i_backward[0], abs_vec_i_backward[1] * abs_vec_i_backward[2], 0])
                adist_backward_max_ind = i
                while True:
                    adist_backward_max = max_adist
                    for j in points_ind:
                        if j not in kept_inds:
                            point_j = fused_points[:, j]
                            abs_vec_j = fused_vecs[:, j]
                            oriented_point_j = [point_j, abs_vec_j]
                            point_vec = point_j - point_i_backward
                            ward = fast_inner_product(point_vec, vec_i_backward) # np.sum(point_vec * vec_i_backward)
                            if ward <= 0:
                                adist_ij = _angle_hook_dist(oriented_point_i_backward, oriented_point_j, disth_thresh=max_adist)
                                if adist_ij < adist_backward_max:
                                    adist_backward_max = adist_ij
                                    adist_backward_max_ind = j
                    if adist_backward_max >= max_adist:
                        break
                    grouped_inds.insert(0, adist_backward_max_ind)
                    kept_inds[adist_backward_max_ind] = 1
                    point_i_backward_old = point_i_backward
                    point_i_backward = fused_points[:, adist_backward_max_ind]
                    abs_vec_i_backward = fused_vecs[:, adist_backward_max_ind]
                    oriented_point_i_backward = [point_i_backward, abs_vec_i_backward]
                    vec_i_backward = np.float32([abs_vec_i_backward[0], abs_vec_i_backward[1] * abs_vec_i_backward[2], 0])
                    # if np.sum(vec_i_backward * (point_i_backward - point_i_backward_old)) > 0:
                    if fast_inner_product(vec_i_backward, (point_i_backward - point_i_backward_old)) > 0:
                        vec_i_backward *= -1
        
        # remove isolated points
        selected_line_segments = []
        for segment in line_segments:
            if len(segment) > 2:
                selected_line_segments.append(segment)

        return line_segments


    def reverse(self, tensor_dict):
        """
        Parameters
        ----------
        tensor_dict : dict

        Notes
        -----
        ```
        seg_im  # 分割图
        reg_im  # 回归图
            0: dist_im  # 每个分割图上的点到最近线段的垂直距离
            1,2: vert_vec_im  # 每个分割图上的点到最近线段的向量
            3,4,5: abs_dir_im  # 每个分割图上的点所在线段的广义单位方向，|nx|, |ny|, nx * ny
            6 height_im # 每个分割图上的点的高度分布图
        ```
        """
        
        seg_pred = tensor_dict['seg'].detach().cpu().numpy()
        reg_pred = tensor_dict['reg'].detach().cpu().numpy()

        ## pickup line points
        valid_points_map = seg_pred[0] > self.reverse_pre_conf
        # line point postions
        valid_points_bev = self.points_grid_bev[:, valid_points_map]
        # pickup scores
        seg_scores = seg_pred[0][valid_points_map]
        seg_classes = seg_pred[:, valid_points_map]
        # pickup regressions
        reg_values = reg_pred[:, valid_points_map]
        # get vertical vectors of line points
        vert_vecs = np.float32([reg_values[4], - reg_values[3] * _sign(reg_values[5])])
        vert_vecs *= _sign(np.sum(vert_vecs * reg_values[1:3], axis=0))
        # get precise points, vector of directions, and heights of line
        # dst_points = valid_points_bev + reg_values[0] * vert_vecs
        dst_points = np.concatenate([
            valid_points_bev + reg_values[0] * vert_vecs,
            reg_values[6][None]
        ])
        dst_vecs =  reg_values[3:6]
        
        ## fuse points, group nms
        # group points
        kept_groups = self._group_points(dst_points, seg_scores, self.reverse_group_dist_thresh)
        # fuse points in one group
        fused_classes = []
        fused_points = []
        fused_vecs = []
        for g in kept_groups:
            weights = seg_scores[g]
            total_weight = weights.sum()
            mean_classes = (seg_classes[:, g] * weights[None]).sum(axis=-1) / total_weight
            mean_point = (dst_points[:, g] * weights[None]).sum(axis=-1) / total_weight
            mean_vec = (dst_vecs[:, g] * weights[None]).sum(axis=-1) / total_weight
            fused_classes.append(mean_classes)
            fused_points.append(mean_point)
            fused_vecs.append(mean_vec)
        fused_classes = np.float32(fused_classes).T
        fused_points = np.float32(fused_points).T
        fused_vecs = np.float32(fused_vecs).T
        
        ## link all points and get 3d polylines
        line_segments = self._link_line_points(fused_points, fused_vecs, self.reverse_link_max_adist)
        cx, cy, fx, fy = self.bev_intrinsics
        polylines_3d = []
        for g in line_segments:
            polyline_3d = np.concatenate([
                np.stack([(fused_points[1, g] - cx) / fx,
                          (fused_points[0, g] - cy) / fy,
                          fused_points[2, g]]),
                fused_classes[:, g],
            ], axis=0)
            polylines_3d.append(polyline_3d.T)
        return polylines_3d


@numba.jit(nopython=True)
def fast_norm(x):
    return np.sqrt(np.sum(x * x))


@numba.jit(nopython=True)
def fast_inner_product(x, y):
    return (x[None, :] @ y[:, None])[0, 0]


def _angle_hook_dist(oriented_point_1, oriented_point_2, disth_thresh=2, z_weight=10):
    point_1, abs_dir_1 = oriented_point_1
    point_2, abs_dir_2 = oriented_point_2
    
    vec_1 = np.float32([abs_dir_1[0], abs_dir_1[1] * abs_dir_1[2]])
    vec_2 = np.float32([abs_dir_2[0], abs_dir_2[1] * abs_dir_2[2]])
    vec_1 /= fast_norm(vec_1)
    vec_2 /= fast_norm(vec_2)
    # vec_mean = 0.5 * (vec_1 + vec_2)
    
    dir_12 = point_2 - point_1
    dist_12 = fast_norm(dir_12)
    unit_12 = dir_12 / dist_12

    if dist_12 < disth_thresh:
        # calc link_dir vs vec similarity
        link_12 = dir_12[:2] / max(1e-5, fast_norm(dir_12[:2]))
        cos_link = fast_inner_product(link_12, vec_1) # np.sum(link_12 * vec_1)
        # calc vec_1 vs vec_2 similarity
        cos_12 = fast_inner_product(vec_1, vec_2) # np.sum(vec_1 * vec_2)
        # z distance
        dist_z = np.abs(unit_12[2])
        dir_gain = 0.5 * disth_thresh * (cos_link + cos_12)
        dist = dist_12 + z_weight * dist_z - dir_gain
    else:
        dist = dist_12
    
    return dist


@TENSOR_SMITHS.register_module()
class PlanarPolygon3D(PlanarTensorSmith):
    def __init__(self, voxel_shape: list, voxel_range: Tuple[list, list, list]):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]

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

    def __call__(self, transformable: Polyline3D):
        Z, X, Y = self.voxel_shape
        cx, cy, fx, fy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev

        polylines = []
        class_inds = []
        attr_lists = []
        attr_available = 'attrs' in transformable.dictionary
        for element in transformable.elements:
            if element['class'] in transformable.dictionary['classes']:
                points = element['points']
                polylines.append(np.array([
                    points[..., 1] * fy + cy,
                    points[..., 0] * fx + cx,
                    points[..., 2]
                ]).T)
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
        seg_im = np.zeros((2 + num_class_channels + num_attr_channels, X, Y))

        line_ims = []
        dist_ims = []
        vec_ims = []
        dir_ims = []
        height_ims = []

        for polyline, class_ind, attr_list in zip(polylines, class_inds, attr_lists):
            polyline_int = np.round(polyline).astype(int)
            # seg_bev_im
            cv2.fillPoly(seg_im[2 + class_ind], [polyline_int], 1)
            for attr_ind in attr_list:
                cv2.fillPoly(seg_im[2 + num_class_channels + attr_ind], [polyline_int], 1)
            for line_3d in zip(polyline[:-1], polyline[1:]):
                line_3d = np.float32(line_3d)
                line_bev = line_3d[:, :2]
                polygon = expand_line_2d(line_bev, radius=0.5)
                polygon_int = np.round(polygon).astype(int)
                # edge_seg_bev_im
                cv2.fillPoly(seg_im[0], [polygon_int], 1)
                # line segment
                line_im = cv2.fillPoly(np.zeros((X, Y)), [polygon_int], 1)
                line_ims.append(line_im)
                # line direction regressions
                line_dir = line_bev[1] - line_bev[0]
                line_length = np.linalg.norm(line_dir)
                line_dir /= line_length
                line_dir_vert = line_dir[::-1] * [1, -1]
                vec_map = vec_point2line_along_direction(points_grid_bev, line_bev, line_dir_vert)
                dist_im = line_im * np.linalg.norm(vec_map, axis=0) + (1 - line_im) * INF_DIST
                vec_im = line_im * vec_map
                abs_dir_im = line_im * np.float32([
                    np.abs(line_dir[0]),
                    np.abs(line_dir[1]),
                    line_dir[0] * line_dir[1]
                ])[..., None, None]
                dist_ims.append(dist_im)
                vec_ims.append(vec_im)
                dir_ims.append(abs_dir_im)
                # height map
                h2s = (line_3d[1, 2] - line_3d[0, 2]) / max(1e-3, line_length)
                height_im =  line_3d[0, 2] + h2s * line_im * np.linalg.norm(
                    points_grid_bev + vec_map - line_bev[0][..., None, None], axis=0
                )
                height_ims.append(height_im)

        if len(dist_ims) > 0:
            index_im = np.argmin(np.array(dist_ims), axis=0)
            vec_im = choose_index(index_im, vec_ims)
            dist_im = choose_index(index_im, dist_ims)
            dir_im = choose_index(index_im, dir_ims)
            height_im = choose_index(index_im, height_ims)
        else:
            vec_im = np.zeros((2, X, Y))
            dist_im = np.zeros((X, Y))
            dir_im = np.zeros((3, X, Y))
            height_im = np.zeros((X, Y))

        reg_im = np.concatenate([
            seg_im * dist_im[None],
            vec_im,
            dir_im,
            height_im[None]
        ], axis=0)
        # TODO: add ignore_seg_mask according to ignore classes and attrs

        tensor_data = {
            'seg': torch.tensor(seg_im, dtype=torch.float32),
            'reg': torch.tensor(reg_im, dtype=torch.float32)
        }
        
        return tensor_data





@TENSOR_SMITHS.register_module()
class PlanarParkingSlot3D(PlanarTensorSmith):
    
    def __init__(self, 
                 voxel_shape: tuple, 
                 voxel_range: Tuple[list, list, list],
                 reverse_pre_conf: float=0.3,
                 reverse_dist_thresh: float=1):
        """
        Parameters
        ----------
        voxel_shape : tuple
        voxel_range : Tuple[List]
        reverse_pre_conf : float
        reverse_dist_thresh : float

        Examples
        --------
        - voxel_shape=(6, 320, 160)
        - voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        - Z, X, Y = voxel_shape

        """
        super().__init__(voxel_shape, voxel_range)
        self.reverse_pre_conf = reverse_pre_conf
        self.reverse_dist_thresh = reverse_dist_thresh
    
    @staticmethod
    def _get_height_map(
            slot_points_3d: np.ndarray[Tuple[int, int]], 
            points_grid_bev: np.ndarray[Tuple[int, int, int]]
        ):
        """_summary_

        Parameters
        ----------
        slot_points_3d : np.ndarray[Tuple[int, int]]
            of shape (4, 3)
        points_grid_bev : np.ndarray[Tuple[2, int, int]]
            of shape (2, _, _)

        Returns
        -------
        _type_
            _description_
        """
        vec_01 = slot_points_3d[1] - slot_points_3d[0]
        vec_12 = slot_points_3d[2] - slot_points_3d[1]
        vec_23 = slot_points_3d[3] - slot_points_3d[2]
        vec_30 = slot_points_3d[0] - slot_points_3d[3]
        X, Y = points_grid_bev[0].shape
        # plane based on triangle_012
        a_012, b_012, c_012 = norm_vec_012 = np.cross(vec_01, vec_12)
        d_012 = - (norm_vec_012 * slot_points_3d[1]).sum()
        h_012 = - ((points_grid_bev[::-1] * [[[a_012]], [[b_012]]]).sum(axis=0) + d_012) / c_012
        mask_012 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[0, 1, 2], :2][..., [1, 0]].astype(int)], 1)
        # plane based on triangle_230
        a_230, b_230, c_230 = norm_vec_230 = np.cross(vec_23, vec_30)
        d_230 = - (norm_vec_230 * slot_points_3d[3]).sum()
        h_230 = - ((points_grid_bev[::-1] * [[[a_230]], [[b_230]]]).sum(axis=0) + d_230) / c_230
        mask_230 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[2, 3, 0], :2][..., [1, 0]].astype(int)], 1)
        # plane based on triangle_013
        a_013, b_013, c_013 = norm_vec_013 = np.cross(vec_01, vec_30)
        d_013 = - (norm_vec_013 * slot_points_3d[0]).sum()
        h_013 = - ((points_grid_bev[::-1] * [[[a_013]], [[b_013]]]).sum(axis=0) + d_013) / c_013
        mask_013 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[0, 1, 3], :2][..., [1, 0]].astype(int)], 1)
        # plane based on triangle_123
        a_123, b_123, c_123 = norm_vec_123 = np.cross(vec_12, vec_23)
        d_123 = - (norm_vec_123 * slot_points_3d[2]).sum()
        h_123 = - ((points_grid_bev[::-1] * [[[a_123]], [[b_123]]]).sum(axis=0) + d_123) / c_123
        mask_123 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[1, 2, 3], :2][..., [1, 0]].astype(int)], 1)
        # average on all four planes
        overlaps = mask_012 + mask_230 + mask_013 + mask_123 + 1e-3
        height_map = (mask_012 * h_012 + mask_230 * h_230 + mask_013 * h_013 + mask_123 * h_123) / overlaps
        return height_map
 

    def __call__(self, transformable: ParkingSlot3D) -> dict:
        Z, X, Y = self.voxel_shape
        cx, cy, fx, fy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev

        
        seg_im = np.zeros((4, X, Y), dtype=np.float32)
        height_map = np.zeros((1, X, Y), dtype=np.float32)

        reg_im = np.zeros((14, X, Y), dtype=np.float32)
        cen_im = np.zeros((1, X, Y), dtype=np.float32)

        for element in transformable.elements:
            slot_points_3d = (np.float32(element['points']) * [[fx, fy, 1]] + [[cx, cy, 0]])
            entrance_length = np.linalg.norm(element['points'][1] - element['points'][0])
            side_length = np.linalg.norm(element['points'][3] - element['points'][0])
            if entrance_length > side_length:
                slot_points_3d = slot_points_3d[[1, 2, 3, 0]]
            slot_points_bev = slot_points_3d[..., [1, 0]]
            slot_points_bev_int = np.round(slot_points_bev).astype(int)

            # define short side and long side on bev
            short_sides = np.float32([
                np.float32([slot_points_bev[0], slot_points_bev[1]]), 
                np.float32([slot_points_bev[2], slot_points_bev[3]])
            ])
            long_sides = np.float32([
                np.float32([slot_points_bev[1], slot_points_bev[2]]),
                np.float32([slot_points_bev[3], slot_points_bev[0]])
            ])
            center_long_line = short_sides.mean(1)
            center_short_line = long_sides.mean(1)

            ## gen segmentation and height map
            # 0: full slot seg
            cv2.fillPoly(seg_im[0], [slot_points_bev_int], 1)
            # 1: slot line
            linewidth = int(max(1, abs(fx * 0.15)))
            cv2.polylines(seg_im[1], [slot_points_bev_int], linewidth, 1)
            # 2: corners
            radius = int(max(1, abs(fx * 0.5)))
            for corner in slot_points_bev_int:
                cv2.circle(seg_im[2], corner, radius, 1, -1)
            # 3: entrance _seg
            if entrance_length > side_length:
                cv2.circle(seg_im[3], slot_points_bev[[0, 3]].mean(0).astype(int), radius, 1, -1)
            else:
                cv2.circle(seg_im[3], slot_points_bev[[0, 1]].mean(0).astype(int), radius, 1, -1)
            # add height map
            height_map[0] = self._get_height_map(slot_points_3d, points_grid_bev)
            for point_3d in slot_points_3d:
                cv2.circle(height_map[0], point_3d[[1, 0]].astype(int), radius, point_3d[2], -1)
            
            ## preparations for generating regressions
            # gen four regions, front, left, bottom, right
            region_front = cv2.fillPoly(np.zeros((X, Y)), [np.concatenate([
                slot_points_bev_int[[0, 1]], np.round(center_short_line).astype(int)
            ])], 1)
            region_left = cv2.fillPoly(np.zeros((X, Y)), [np.concatenate([
                slot_points_bev_int[[1, 2]], np.round(center_long_line).astype(int)[::-1]
            ])], 1)
            region_bottom = cv2.fillPoly(np.zeros((X, Y)), [np.concatenate([
                slot_points_bev_int[[2, 3]], np.round(center_short_line).astype(int)[::-1]
            ])], 1)
            region_right = cv2.fillPoly(np.zeros((X, Y)), [np.concatenate([
                slot_points_bev_int[[3, 0]], np.round(center_long_line).astype(int)
            ])], 1)
            # handle minor overlaps
            overlap_short = region_front + region_bottom - np.clip(region_front + region_bottom, 0, 1)
            overlap_long = region_left + region_right - np.clip(region_left + region_right, 0, 1)
            region_front -= overlap_short
            region_left -= overlap_long
            # total region of the slot
            region_slot = cv2.fillPoly(np.zeros((X, Y)), [slot_points_bev_int], 1)
            # side directions, general directions
            side_direction_front = np.float32(slot_points_bev[1] - slot_points_bev[0])
            side_direction_front /= np.linalg.norm(side_direction_front)
            abs_side_direction_front = np.concatenate([
                np.abs(side_direction_front), 
                side_direction_front[0, None] * side_direction_front[1, None]
            ], axis=-1)
            side_direction_left = np.float32(slot_points_bev[2] - slot_points_bev[1])
            side_direction_left /= np.linalg.norm(side_direction_left)
            abs_side_direction_left = np.concatenate([
                np.abs(side_direction_left), 
                side_direction_left[0, None] * side_direction_left[1, None]
            ], axis=-1)
            side_direction_bottom = np.float32(slot_points_bev[3] - slot_points_bev[2])
            side_direction_bottom /= np.linalg.norm(side_direction_bottom)
            abs_side_direction_bottom = np.concatenate([
                np.abs(side_direction_bottom), 
                side_direction_bottom[0, None] * side_direction_bottom[1, None]
            ], axis=-1)
            side_direction_right = np.float32(slot_points_bev[0] - slot_points_bev[3])
            side_direction_right /= np.linalg.norm(side_direction_right)
            abs_side_direction_right = np.concatenate([
                np.abs(side_direction_right), 
                side_direction_right[0, None] * side_direction_right[1, None]
            ], axis=-1)
            # calc distances along long to short and short to long sides
            dist_along_left_to_front = dist_point2line_along_direction(
                points_grid_bev, short_sides[0], side_direction_left
            ) * region_left
            dist_along_left_to_bottom = dist_point2line_along_direction(
                points_grid_bev, short_sides[1], side_direction_left
            ) * region_left
            dist_along_right_to_front = dist_point2line_along_direction(
                points_grid_bev, short_sides[0], side_direction_right
            ) * region_right
            dist_along_right_to_bottom = dist_point2line_along_direction(
                points_grid_bev, short_sides[1], side_direction_right
            ) * region_right
            dist_along_front_to_left = dist_point2line_along_direction(
                points_grid_bev, long_sides[0], side_direction_front
            ) * region_front
            dist_along_front_to_right = dist_point2line_along_direction(
                points_grid_bev, long_sides[1], side_direction_front
            ) * region_front
            dist_along_bottom_to_left = dist_point2line_along_direction(
                points_grid_bev, long_sides[0], side_direction_bottom
            ) * region_bottom
            dist_along_bottom_to_right = dist_point2line_along_direction(
                points_grid_bev, long_sides[1], side_direction_bottom
            ) * region_bottom
            # get min and max distances, like distance field
            min_dist_along_left = np.minimum(dist_along_left_to_front, dist_along_left_to_bottom)
            max_dist_along_left = np.maximum(dist_along_left_to_front, dist_along_left_to_bottom)
            min_dist_along_right = np.minimum(dist_along_right_to_front, dist_along_right_to_bottom)
            max_dist_along_right = np.maximum(dist_along_right_to_front, dist_along_right_to_bottom)
            min_dist_along_front = np.minimum(dist_along_front_to_left, dist_along_front_to_right)
            max_dist_along_front = np.maximum(dist_along_front_to_left, dist_along_front_to_right)
            min_dist_along_bottom = np.minimum(dist_along_bottom_to_left, dist_along_bottom_to_right)
            max_dist_along_bottom = np.maximum(dist_along_bottom_to_left, dist_along_bottom_to_right)
            # merge long and short
            min_dist_along_long = np.maximum(min_dist_along_left, min_dist_along_right)
            max_dist_along_long = np.maximum(max_dist_along_left, max_dist_along_right)
            min_dist_along_short = np.maximum(min_dist_along_front, min_dist_along_bottom)
            max_dist_along_short = np.maximum(max_dist_along_front, max_dist_along_bottom)
            # vector from side to center line
            vec_left_short_center = vec_point2line_along_direction(
                points_grid_bev, center_short_line, side_direction_left
            ) * region_left[None]
            vec_right_short_center = vec_point2line_along_direction(
                points_grid_bev, center_short_line, side_direction_right
            ) * region_right[None]
            vec_front_long_center = vec_point2line_along_direction(
                points_grid_bev, center_long_line, side_direction_front
            ) * region_front[None]
            vec_bottom_long_center = vec_point2line_along_direction(
                points_grid_bev, center_long_line, side_direction_bottom
            ) * region_bottom[None]
            ## generate regressions
            # 0, 1: min, max distance to short side along long side
            reg_im[0] = reg_im[0] * (1 - region_slot) + region_slot * min_dist_along_long
            reg_im[1] = reg_im[1] * (1 - region_slot) + region_slot * max_dist_along_long
            # 2, 3: min, max distance to long side along short side
            reg_im[2] = reg_im[2] * (1 - region_slot) + region_slot * min_dist_along_short
            reg_im[3] = reg_im[3] * (1 - region_slot) + region_slot * max_dist_along_short
            # 4, 5, 6: (|nl_x|, |nl_y|, nl_x * nl_y) normalized absolute long side direction
            reg_im[4:7] = reg_im[4:7] * (1 - region_slot) + region_slot * (np.full(
                (3, X, Y), abs_side_direction_left[..., None, None]
            ) * region_left + np.full(
                (3, X, Y), abs_side_direction_right[..., None, None]
            ) * region_right)
            # 7, 8, 9: (|ns_x|, |ns_y|, ns_x * ns_y) normalized absolute short side direction
            reg_im[7:10] = reg_im[7:10] * (1 - region_slot) + region_slot * (np.full(
                (3, X, Y), abs_side_direction_front[..., None, None]
            ) * region_front + np.full(
                (3, X, Y), abs_side_direction_bottom[..., None, None]
            ) * region_bottom)
            # 10, 11: direction to center short line along long side (cl_x, cl_y)
            reg_im[10:12] = reg_im[10:12] * (1 - region_slot) + region_slot * (
                vec_left_short_center + vec_right_short_center
            )
            # 12, 13: direction to center long line along short side (cs_x, cs_y)
            reg_im[12:14] = reg_im[12:14] * (1 - region_slot) + region_slot * (
                vec_front_long_center + vec_bottom_long_center
            )
            ## gen centerness
            normed_min_ds = min_dist_along_short / (min_dist_along_short.max() + 1e-3)
            cen_short = (1 - 2 * np.abs(normed_min_ds - 0.5)) * region_slot
            normed_min_dl = min_dist_along_long / (min_dist_along_long.max() + 1e-3)
            cen_long = (1 - 2 * np.abs(normed_min_dl - 0.5)) * region_slot
            centerness = cen_short * cen_long
            centerness /= (centerness.max() + 1e-3)
            cen_im[0] = cen_im[0] * (1 - region_slot) + region_slot * centerness

        tensor_data = {
            'cen': torch.tensor(cen_im),
            'seg': torch.tensor(seg_im),
            'reg': torch.cat([
                torch.tensor(reg_im),
                torch.tensor(height_map)
            ])
        }
        
        return tensor_data


    @staticmethod
    def _get_mean_point(slot_group_points, cen_score_group, seg_score_group):
        selected_inds_points = slot_group_points[:, 2] > 0.75
        if np.sum(selected_inds_points) < 1:
            selected_inds_points = slot_group_points[:, 2] > 0.375
            # seg_score_group *= 0.5
            if np.sum(selected_inds_points) < 1:
                selected_inds_points = slot_group_points[:, 2] > 0.125
                # seg_score_group *= 0.5
        selected_points = slot_group_points[selected_inds_points]
        selected_cen_scores = cen_score_group[selected_inds_points]
        selected_seg_scores = seg_score_group[selected_inds_points]
        selected_conf = selected_cen_scores * selected_seg_scores * selected_points[:, 2]
        mean_point = np.sum(
            selected_points[:, :2] * selected_conf[:, None], axis=0
        ) / np.sum(selected_conf)
        mean_seg_conf = np.mean(selected_seg_scores * selected_points[:, 2])
        return [mean_point[0], mean_point[1], mean_seg_conf]


    def _group_nms(self, cen_scores, seg_scores, corner_points_bev, dist_thresh):
        # prepare scores as positional weights
        scores = cen_scores * seg_scores
        # sort by score
        ranked_ind = np.argsort(scores)[::-1]
        points_0, points_1, points_2, points_3 = corner_points_bev
        # assign slots to groups according to distance
        kept_groups = []
        kept_inds = []
        for i in ranked_ind:
            if i not in kept_inds:
                quad_i = np.float32([
                    points_0[:2, i],
                    points_1[:2, i],
                    points_2[:2, i],
                    points_3[:2, i]
                ])
                mean_quad_i = quad_i.mean(0)
                kept_inds.append(i)
                grouped_inds = [i]
                kept_groups.append(grouped_inds)
                for j in ranked_ind[:]:
                    if j not in kept_inds:
                        quad_j = np.float32([
                            points_0[:2, j],
                            points_1[:2, j],
                            points_2[:2, j],
                            points_3[:2, j]
                        ])
                        mean_quad_j = quad_j.mean(0)
                        dist_ij = np.linalg.norm(mean_quad_i - mean_quad_j)
                        if dist_ij < dist_thresh: 
                            kept_inds.append(j)
                            grouped_inds.append(j)
        # average slots in each group
        mean_slots = []
        for group in kept_groups:
            # anchor the first slot, align other slots
            ind_0 = group[0]
            slot_0 = np.float32([
                points_0[:, ind_0],
                points_1[:, ind_0],
                points_2[:, ind_0],
                points_3[:, ind_0]
            ])
            slot_0_line_01 = slot_0[1, :2] - slot_0[0, :2]
            slot_group = [slot_0]
            for ind in group[1:]:
                slot_line_01 = points_1[:2, ind] - points_0[:2, ind]
                if np.sum(slot_line_01 * slot_0_line_01) < 0:
                    slot = np.float32([
                        points_2[:, ind],
                        points_3[:, ind],
                        points_0[:, ind],
                        points_1[:, ind],
                    ])
                else:
                    slot = np.float32([
                        points_0[:, ind],
                        points_1[:, ind],
                        points_2[:, ind],
                        points_3[:, ind]
                    ])
                slot_group.append(slot)
            # get mean slot
            slot_group = np.float32(slot_group)
            cen_score_group = cen_scores[group]
            seg_score_group = seg_scores[group]
            mean_point_0 = self._get_mean_point(slot_group[:, 0], cen_score_group, seg_score_group)
            mean_point_1 = self._get_mean_point(slot_group[:, 1], cen_score_group, seg_score_group)
            mean_point_2 = self._get_mean_point(slot_group[:, 2], cen_score_group, seg_score_group)
            mean_point_3 = self._get_mean_point(slot_group[:, 3], cen_score_group, seg_score_group)
            mean_slot = np.float32([mean_point_0, mean_point_1, mean_point_2, mean_point_3])
            # record slot
            mean_slots.append(mean_slot)
        
        return mean_slots


    def reverse(self, tensor_dict: Dict[str, Tensor]):
        """One should rearange model outputs to tensor_dict format.

        Parameters
        ----------
        tensor_dict : dict
        """
        cen_pred = tensor_dict['cen'].detach().cpu().numpy()
        seg_pred = tensor_dict['seg'].detach().cpu().numpy()
        reg_pred = tensor_dict['reg'].detach().cpu().numpy()

        # get valid_points
        valid_points_map = cen_pred[0] > self.reverse_pre_conf
        valid_points_bev = self.points_grid_bev[:, valid_points_map]
        
        ## pickup scores
        cen_scores = cen_pred[0][valid_points_map]
        seg_scores = seg_pred[0][valid_points_map]

        ## pickup regressions
        # 0, 1: min, max distance to short side along long side
        dl_min = reg_pred[0][valid_points_map]
        dl_max = reg_pred[1][valid_points_map]
        # 2, 3: min, max distance to long side along short side
        ds_min = reg_pred[2][valid_points_map]
        ds_max = reg_pred[3][valid_points_map]
        # 4, 5, 6: (|nl_x|, |nl_y|, nl_x * nl_y) normalized absolute long side direction
        abs_nl_x = reg_pred[4][valid_points_map]
        abs_nl_y = reg_pred[5][valid_points_map]
        nl_xy = reg_pred[6][valid_points_map]
        # 7, 8, 9: (|ns_x|, |ns_y|, ns_x * ns_y) normalized absolute short side direction
        abs_ns_x = reg_pred[7][valid_points_map]
        abs_ns_y = reg_pred[8][valid_points_map]
        ns_xy = reg_pred[9][valid_points_map]
        # 10, 11: direction to center short line along long side (cl_x, cl_y)
        cl_x = reg_pred[10][valid_points_map]
        cl_y = reg_pred[11][valid_points_map]
        # 12, 13: direction to center long line along short side (cs_x, cs_y)
        cs_x = reg_pred[12][valid_points_map]
        cs_y = reg_pred[13][valid_points_map]
        
        ## get real side directions
        # long side direction
        nl_x = abs_nl_x
        nl_y = _sign(nl_xy) * abs_nl_y
        nl_sign = nl_x * cl_x + nl_y * cl_y
        nl_x *= _sign(nl_sign)
        nl_y *= _sign(nl_sign)
        norm_vec_l = np.float32([nl_x, nl_y])
        norm_vec_l /= np.linalg.norm(norm_vec_l, axis=0)
        # short side direction
        ns_x = abs_ns_x
        ns_y = _sign(ns_xy) * abs_ns_y
        ns_sign = ns_x * cs_x + ns_y * cs_y
        ns_x *= _sign(ns_sign)
        ns_y *= _sign(ns_sign)
        norm_vec_s = np.float32([ns_x, ns_y])
        norm_vec_s /= np.linalg.norm(norm_vec_s, axis=0)

        ## get vectors to four sides, and then get four corner points
        # min, max vec to long and short sides
        vec_l_min = - dl_min * norm_vec_l
        vec_l_max = dl_max * norm_vec_l
        vec_s_min = - ds_min * norm_vec_s
        vec_s_max = ds_max * norm_vec_s
        # get the best slot corner for each point
        vec_cross = np.cross(vec_l_min, vec_s_min, axis=0)
        # add corner confidence
        vec_l_min_conf = np.insert(vec_l_min, 2, 1, axis=0)
        vec_l_max_conf = np.insert(vec_l_max, 2, 0.5, axis=0)
        vec_s_min_conf = np.insert(vec_s_min, 2, 1, axis=0)
        vec_s_max_conf = np.insert(vec_s_max, 2, 0.5, axis=0)
        # get four vectors
        vec2front = vec_l_min_conf # front as vec_l_min
        vec2bottom = vec_l_max_conf
        vec2left = np.where(vec_cross <= 0, vec_s_min_conf, vec_s_max_conf)
        vec2right = np.where(vec_cross > 0, vec_s_min_conf, vec_s_max_conf)
        # get four corners
        points_0 = valid_points_bev + vec2front[:2] + vec2right[:2]
        points_0 = np.insert(points_0, 2, vec2front[2] * vec2right[2], axis=0)
        points_1 = valid_points_bev + vec2front[:2] + vec2left[:2]
        points_1 = np.insert(points_1, 2, vec2front[2] * vec2left[2], axis=0)
        points_2 = valid_points_bev + vec2bottom[:2] + vec2left[:2]
        points_2 = np.insert(points_2, 2, vec2bottom[2] * vec2left[2], axis=0)
        points_3 = valid_points_bev + vec2bottom[:2] + vec2right[:2]
        points_3 = np.insert(points_3, 2, vec2bottom[2] * vec2right[2], axis=0)
        corner_points_bev = [points_0, points_1, points_2, points_3]

        ## groups_nms, get mean slot points_bev
        cx, cy, fx, fy = self.bev_intrinsics
        mean_slots_bev_no_entrance = self._group_nms(
            cen_scores, seg_scores, corner_points_bev, abs(fx * self.reverse_dist_thresh)
        )
        # determine the entrance and calc 3D coordinates
        _, H, W = self.voxel_shape
        SEQ_CANDIDATES = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2]
        ]
        mean_slots_3d = []
        for mean_slot in mean_slots_bev_no_entrance:
            # calc each line center, then calc entrance confidence
            line_centers = [
                mean_slot[[0, 1], :2].mean(0).astype(int),
                mean_slot[[1, 2], :2].mean(0).astype(int),
                mean_slot[[2, 3], :2].mean(0).astype(int),
                mean_slot[[3, 0], :2].mean(0).astype(int)
            ]
            ent_line_count = []
            for center in line_centers:
                pos_w, pos_h = center
                if np.any([pos_w < 0, pos_h < 0, pos_w >= W, pos_h >= H]):
                    ent_line_count.append(0)
                else:
                    w0 = max(pos_w - 1, 0)
                    w1 = min(pos_w + 2, W)
                    h0 = max(pos_h - 1, 0)
                    h1 = min(pos_h + 2, H)
                    ent_line_count.append(seg_pred[3][h0:h1, w0:w1].mean())
            ent_line_count = np.float32(ent_line_count)
            ent_seq = SEQ_CANDIDATES[ent_line_count.argmax()]
            mean_slot = mean_slot[ent_seq]
            # calc height and get 3d points
            heights = []
            for point in mean_slot:
                if np.any([point[0] < 0, point[1] < 0, point[0] >= W, point[1] >= H]):
                    heights.append(None)
                else:
                    heights.append(reg_pred[14][
                        min(max(round(point[1]), 0), H - 1), 
                        min(max(round(point[0]), 0), W - 1)
                    ].mean())
            valid_heights = [height for height in heights if height is not None]
            if len(valid_heights) > 0:
                mean_height = np.mean(valid_heights)
                heights = [mean_height if height is None else height for height in heights]
            else:
                heights = [0, 0, 0, 0]
            # get 3d points
            mean_slot_3d = []
            for point, height in zip(mean_slot, heights):
                mean_slot_3d.append(
                    [(point[1] - cx) / fx, 
                     (point[0] - cy) / fy, 
                     height, 
                     point[2]]
                )
            mean_slots_3d.append(np.float32(mean_slot_3d))
        
        return mean_slots_3d
            
                

