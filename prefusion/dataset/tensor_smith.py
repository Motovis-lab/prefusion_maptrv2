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
)
from .transform import (
    CameraImage, CameraSegMask, CameraDepth,
    Bbox3D,
    Polyline3D, SegBev, ParkingSlot3D
)

__all__ = [
    "CameraImageTensor", "CameraDepthTensor", "CameraSegTensor", "PlanarBbox3D",
    "PlanarBboxBev", "PlanarSegBev", "PlanarPolyline3D", "PlanarPolygon3D", "PlanarParkingSlot3D",
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

    return fx, fy, cx, cy



@TENSOR_SMITHS.register_module()
class PlanarBbox3D(TensorSmith):
    def __init__(self, voxel_shape, voxel_range):
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
    
    def __call__(self, transformable: Bbox3D):
        raise NotImplementedError



@TENSOR_SMITHS.register_module()
class PlanarBboxBev(TensorSmith):
    def __init__(self, voxel_shape, voxel_range):
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range
    
    def __call__(self, transformable: Bbox3D):
        """
        Parameters
        ----------
        elements : List[dict]
            a list of boxes. Each element is a dict of box having the following format
            ```elements[0] = {
                'class': 'class.vehicle.passenger_car',
                'attr': {'attr.time_varying.object.state': 'attr.time_varying.object.state.stationary',
                        'attr.vehicle.is_trunk_open': 'attr.vehicle.is_trunk_open.false',
                        'attr.vehicle.is_door_open': 'attr.vehicle.is_door_open.false'},
                'size': [4.6486, 1.9505, 1.5845],
                'rotation': array([[ 0.93915682, -0.32818596, -0.10138267],
                                [ 0.32677338,  0.94460343, -0.03071667],
                                [ 0.1058472 , -0.00428138,  0.99437319]]),
                'translation': array([[-15.70570354], [ 11.88484971], [ -0.61029085]]), # NOTE: it is a column vector
                'track_id': '10035_0', # NOT USED
                'velocity': array([[0.], [0.], [0.]]) # NOTE: it is a column vector
            }
            ```
        dictionary : dict
            following format
            ```dictionary = {
                'branch_0': {'classes': ['car', 'bus', 'pedestrain', ...], 'attrs': []}
                'branch_1': {'classes': [], 'attrs': []}
                ...
            }
            ```
        flip_aware_class_pairs : List[tuple]
            list of class pairs that are flip-aware
            flip_aware_class_pairs = [('left_arrow', 'right_arrow')]
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        raise NotImplementedError




@TENSOR_SMITHS.register_module()
class PlanarSegBev(TensorSmith):
    def __init__(self, voxel_shape, voxel_range):
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range

    def __call__(self, transformable: SegBev):
        raise NotImplementedError



@TENSOR_SMITHS.register_module()
class PlanarPolyline3D(TensorSmith):
    def __init__(self, voxel_shape, voxel_range):
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
        fx, fy, cx, cy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev

        tensor_data = {}
        for branch in transformable.dictionary:
            polylines = []
            class_inds = []
            attr_lists = []
            attr_available = 'attrs' in transformable.dictionary[branch]
            for element in transformable.elements:
                if element['class'] in transformable.dictionary[branch]['classes']:
                    points = element['points']
                    polylines.append(np.array([
                        points[..., 1] * fy + cy,
                        points[..., 0] * fx + cx,
                        points[..., 2]
                    ]).T)
                    class_inds.append(transformable.dictionary[branch]['classes'].index(element['class']))
                    attr_ind_list = []
                    if 'attr' in element and attr_available:
                        element_attrs = element['attr'].values()
                        for attr in element_attrs:
                            if attr in transformable.dictionary[branch]['attrs']:
                                attr_ind_list.append(transformable.dictionary[branch]['attrs'].index(attr))
                    attr_lists.append(attr_ind_list)
                    # TODO: add ignore_mask according to ignore classes and attrs
            num_class_channels = len(transformable.dictionary[branch]['classes'])
            if attr_available:
                num_attr_channels = len(transformable.dictionary[branch]['attrs'])
            else:
                num_attr_channels = 0
            branch_seg_im = np.zeros((1 + num_class_channels + num_attr_channels, X, Y))

            line_ims = []
            dist_ims = []
            vec_ims = []
            dir_ims = []
            height_ims = []

            linewidth = int(max(0.51, abs(fx * 0.1)))

            for polyline, class_ind, attr_list in zip(polylines, class_inds, attr_lists):
                for line_3d in zip(polyline[:-1], polyline[1:]):
                    line_3d = np.float32(line_3d)
                    line_bev = line_3d[:, :2]
                    polygon = expand_line_2d(line_bev, radius=linewidth)
                    polygon_int = np.round(polygon).astype(int)
                    # seg_bev_im
                    cv2.fillPoly(branch_seg_im[0], [polygon_int], 1)
                    cv2.fillPoly(branch_seg_im[1 + class_ind], [polygon_int], 1)
                    for attr_ind in attr_list:
                        cv2.fillPoly(branch_seg_im[1 + num_class_channels + attr_ind], [polygon_int], 1)
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

            index_im = np.argmin(np.array(dist_ims), axis=0)
            branch_vec_im = np.choose(index_im, vec_ims)
            branch_dist_im = np.choose(index_im, dist_ims)
            branch_dir_im = np.choose(index_im, dir_ims)
            branch_height_im = np.choose(index_im, height_ims)

            branch_reg_im = np.concatenate([
                branch_seg_im[:1] * branch_dist_im[None],
                branch_vec_im,
                branch_dir_im,
                branch_height_im[None]
            ], axis=0)
            # TODO: add branch_ignore_seg_mask according to ignore classes and attrs

            tensor_data[branch] = {
                'seg': torch.tensor(branch_seg_im, dtype=torch.float32),
                'reg': torch.tensor(branch_reg_im, dtype=torch.float32)
            }
        
        return tensor_data


    @staticmethod
    @numba.njit
    def _group_points(dst_points, seg_scores, dist_thresh=0.1):
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


    @staticmethod
    def _angle_hook_dist(oriented_point_1, oriented_point_2, angle=45, angle_weight=4):
        point_1, abs_dir_1 = oriented_point_1
        point_2, abs_dir_2 = oriented_point_2
        vec_1 = np.float32([abs_dir_1[0], abs_dir_1[1] * abs_dir_1[2]])
        vec_2 = np.float32([abs_dir_2[0], abs_dir_2[1] * abs_dir_2[2]])
        cross_thresh = np.sin(np.abs(angle / 2) * np.pi / 180)
        dist_12 = np.linalg.norm(point_2 - point_1)
        cross_12 = np.abs(np.cross(vec_1, vec_2))
        cross_weight = (angle_weight + 1) if cross_12 > cross_thresh else (angle_weight * cross_12 + 1)
        return dist_12 * cross_weight

    def _link_line_points(self, fused_points, fused_vecs, max_adist=5):
        points_ind = np.arange(fused_points.shape[1])
        # get groups of line segments
        line_segments = []
        kept_inds = []

        for i in points_ind:
            if i not in kept_inds:
                kept_inds.append(i)
                grouped_inds = [i]
                line_segments.append(grouped_inds)
                # go forward if direction is similar
                point_i_forward = fused_points[:, i]
                abs_vec_i_forward = fused_vecs[:, i]
                oriented_point_i_forward = [point_i_forward, abs_vec_i_forward]
                vec_i_forward = np.float32([abs_vec_i_forward[0], abs_vec_i_forward[1] * abs_vec_i_forward[2]])
                adist_forward_max_ind = i
                while True:
                    adist_forward_max = max_adist
                    for j in points_ind:
                        if j not in kept_inds:
                            point_j = fused_points[:, j]
                            abs_vec_j = fused_vecs[:, j]
                            oriented_point_j = [point_j, abs_vec_j]
                            point_vec = point_j - point_i_forward
                            ward = np.sum(point_vec * vec_i_forward)
                            if ward > 0:
                                adist_ij = self._angle_hook_dist(oriented_point_i_forward, oriented_point_j)
                                if adist_ij < adist_forward_max:
                                    adist_forward_max = adist_ij
                                    adist_forward_max_ind = j
                    if adist_forward_max >= max_adist:
                        break
                    grouped_inds.append(adist_forward_max_ind)
                    kept_inds.append(adist_forward_max_ind)
                    point_i_forward_old = point_i_forward
                    point_i_forward = fused_points[:, adist_forward_max_ind]
                    abs_vec_i_forward = fused_vecs[:, adist_forward_max_ind]
                    oriented_point_i_forward = [point_i_forward, abs_vec_i_forward]
                    vec_i_forward = np.float32([abs_vec_i_forward[0], abs_vec_i_forward[1] * abs_vec_i_forward[2]])
                    if np.sum(vec_i_forward * (point_i_forward - point_i_forward_old)) <= 0:
                        vec_i_forward *= -1
                    
                # go backward if direction is opposite
                point_i_backward = fused_points[:, i]
                abs_vec_i_backward = fused_vecs[:, i]
                oriented_point_i_backward = [point_i_backward, abs_vec_i_backward]
                vec_i_backward = np.float32([abs_vec_i_backward[0], abs_vec_i_backward[1] * abs_vec_i_backward[2]])
                adist_backward_max_ind = i
                while True:
                    adist_backward_max = max_adist
                    for j in points_ind:
                        if j not in kept_inds:
                            point_j = fused_points[:, j]
                            abs_vec_j = fused_vecs[:, j]
                            oriented_point_j = [point_j, abs_vec_j]
                            point_vec = point_j - point_i_backward
                            ward = np.sum(point_vec * vec_i_backward)
                            if ward <= 0:
                                adist_ij = self._angle_hook_dist(oriented_point_i_backward, oriented_point_j)
                                if adist_ij < adist_backward_max:
                                    adist_backward_max = adist_ij
                                    adist_backward_max_ind = j
                    if adist_backward_max >= max_adist:
                        break
                    grouped_inds.insert(0, adist_backward_max_ind)
                    kept_inds.append(adist_backward_max_ind)
                    point_i_backward_old = point_i_backward
                    point_i_backward = fused_points[:, adist_backward_max_ind]
                    abs_vec_i_backward = fused_vecs[:, adist_backward_max_ind]
                    oriented_point_i_backward = [point_i_backward, abs_vec_i_backward]
                    vec_i_backward = np.float32([abs_vec_i_backward[0], abs_vec_i_backward[1] * abs_vec_i_backward[2]])
                    if np.sum(vec_i_backward * (point_i_backward - point_i_backward_old)) > 0:
                        vec_i_backward *= -1
        
        return line_segments


    def reverse(self, tensor_dict, pre_conf=0.5):
        """
        Parameters
        ----------
        tensor_dict : dict
            ```dict(
                branch_id = dict,
                branch_id = dict
            )```
        pre_conf : float, optional, by default 0.5

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
        polylines_3d = {}
        for branch in tensor_dict:
            seg_pred = tensor_dict[branch]['seg'].detach().cpu().numpy()
            reg_pred = tensor_dict[branch]['reg'].detach().cpu().numpy()

            ## pickup line points
            valid_points_map = seg_pred[0] > pre_conf
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
            dst_points = valid_points_bev + reg_values[0] * vert_vecs
            dst_vecs =  reg_values[3:6]
            dst_heights = reg_values[6]
            
            ## fuse points, group nms
            # group points
            kept_groups = self._group_points(dst_points, seg_scores)
            # fuse points in one group
            fused_classes = []
            fused_points = []
            fused_vecs = []
            fused_heights = []
            for g in kept_groups:
                weights = seg_scores[g]
                total_weight = weights.sum()
                mean_classes = (seg_classes[:, g] * weights[None]).sum(axis=-1) / total_weight
                mean_point = (dst_points[:, g] * weights[None]).sum(axis=-1) / total_weight
                mean_vec = (dst_vecs[:, g] * weights[None]).sum(axis=-1) / total_weight
                mean_height = (dst_heights[g] * weights).sum(axis=-1) / total_weight
                # mean_point = dst_points[:, g].mean(axis=-1)
                # mean_vec = dst_vecs[:, g].mean(axis=-1)
                # mean_height = dst_heights[g].mean(axis=-1)
                fused_classes.append(mean_classes)
                fused_points.append(mean_point)
                fused_vecs.append(mean_vec)
                fused_heights.append(mean_height)
            fused_classes = np.float32(fused_classes).T
            fused_points = np.float32(fused_points).T
            fused_vecs = np.float32(fused_vecs).T
            fused_heights = np.float32(fused_heights)
            
            ## link all points and get 3d polylines
            line_segments = self._link_line_points(fused_points, fused_vecs)
            fx, fy, cx, cy = self.bev_intrinsics
            polylines_3d[branch] = []

            for g in line_segments:
                polyline_3d = np.concatenate([
                    np.stack([(fused_points[1, g] - cx) / fx,
                              (fused_points[0, g] - cy) / fy,
                              fused_heights[g]]),
                    fused_classes[:, g],
                ], axis=0)
                polylines_3d[branch].append(polyline_3d)
        return polylines_3d





@TENSOR_SMITHS.register_module()
class PlanarPolygon3D(TensorSmith):
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
        fx, fy, cx, cy = self.bev_intrinsics
        points_grid_bev = self.points_grid_bev

        tensor_data = {}
        for branch in transformable.dictionary:
            polylines = []
            class_inds = []
            attr_lists = []
            attr_available = 'attrs' in transformable.dictionary[branch]
            for element in transformable.elements:
                if element['class'] in transformable.dictionary[branch]['classes']:
                    points = element['points']
                    polylines.append(np.array([
                        points[..., 1] * fy + cy,
                        points[..., 0] * fx + cx,
                        points[..., 2]
                    ]).T)
                    class_inds.append(transformable.dictionary[branch]['classes'].index(element['class']))
                    attr_ind_list = []
                    if 'attr' in element and attr_available:
                        element_attrs = element['attr'].values()
                        for attr in element_attrs:
                            if attr in transformable.dictionary[branch]['attrs']:
                                attr_ind_list.append(transformable.dictionary[branch]['attrs'].index(attr))
                    attr_lists.append(attr_ind_list)
                    # TODO: add ignore_mask according to ignore classes and attrs
            num_class_channels = len(transformable.dictionary[branch]['classes'])
            if attr_available:
                num_attr_channels = len(transformable.dictionary[branch]['attrs'])
            else:
                num_attr_channels = 0
            branch_seg_im = np.zeros((2 + num_class_channels + num_attr_channels, X, Y))

            line_ims = []
            dist_ims = []
            vec_ims = []
            dir_ims = []
            height_ims = []

            for polyline, class_ind, attr_list in zip(polylines, class_inds, attr_lists):
                polyline_int = np.round(polyline).astype(int)
                # seg_bev_im
                cv2.fillPoly(branch_seg_im[2 + class_ind], [polyline_int], 1)
                for attr_ind in attr_list:
                    cv2.fillPoly(branch_seg_im[2 + num_class_channels + attr_ind], [polyline_int], 1)
                for line_3d in zip(polyline[:-1], polyline[1:]):
                    line_3d = np.float32(line_3d)
                    line_bev = line_3d[:, :2]
                    polygon = expand_line_2d(line_bev, radius=0.5)
                    polygon_int = np.round(polygon).astype(int)
                    # edge_seg_bev_im
                    cv2.fillPoly(branch_seg_im[0], [polygon_int], 1)
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

            index_im = np.argmin(np.array(dist_ims), axis=0)
            branch_vec_im = np.choose(index_im, vec_ims)
            branch_dist_im = np.choose(index_im, dist_ims)
            branch_dir_im = np.choose(index_im, dir_ims)
            branch_height_im = np.choose(index_im, height_ims)

            branch_reg_im = np.concatenate([
                branch_seg_im * branch_dist_im[None],
                branch_vec_im,
                branch_dir_im,
                branch_height_im[None]
            ], axis=0)
            # TODO: add branch_ignore_seg_mask according to ignore classes and attrs

            tensor_data[branch] = {
                'seg': torch.tensor(branch_seg_im, dtype=torch.float32),
                'reg': torch.tensor(branch_reg_im, dtype=torch.float32)
            }
        
        return tensor_data
    
    def reverse(self, tensor_data):
        pass






@TENSOR_SMITHS.register_module()
class PlanarParkingSlot3D(TensorSmith):
    def __init__(self, voxel_shape: tuple, voxel_range: Tuple[list]):
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
        fx, fy, cx, cy = self.bev_intrinsics
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



    def reverse(self, tensor_dict: Dict[str, Tensor], pre_conf=0.3, dist_thresh=1):
        """One should rearange model outputs to tensor_dict format.

        Parameters
        ----------
        tensor_dict : dict
        pre_conf : float, optional, by default 0.3
            filter valid points, by default 0.3
        """
        cen_pred = tensor_dict['cen'].detach().cpu().numpy()
        seg_pred = tensor_dict['seg'].detach().cpu().numpy()
        reg_pred = tensor_dict['reg'].detach().cpu().numpy()

        # get valid_points
        valid_points_map = cen_pred[0] > pre_conf
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
        mean_slots_bev_no_entrance = self._group_nms(
            cen_scores, seg_scores, corner_points_bev, abs(self.bev_intrinsics[0] * dist_thresh)
        )
        # determine the entrance and calc 3D coordinates
        _, H, W = self.voxel_shape
        SEQ_CANDIDATES = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2]
        ]
        fx, fy, cx, cy = self.bev_intrinsics
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
                        min(max(round(point[1]), 0), H), 
                        min(max(round(point[0]), 0), W)
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
            
                

