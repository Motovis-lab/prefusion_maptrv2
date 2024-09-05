from typing import Tuple, Dict, Union, Iterable

import cv2
import torch
import numpy as np

from prefusion.registry import TENSOR_SMITHS
from .utils import (
    expand_line_2d, _sign, INF_DIST,
    vec_point2line_along_direction, 
    dist_point2line_along_direction,
    get_cam_type
)
from .transform import (
    CameraImage, CameraSegMask, CameraDepth,
    Polyline3D, SegBev, ParkingSlot3D
)


class TensorSmith:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@TENSOR_SMITHS.register_module()
class CameraImageTensor(TensorSmith):
    def __init__(self, 
            means: Union[list[float, float, float], tuple[float, float, float], float] = 128, 
            stds: Union[list[float, float, float], tuple[float, float, float], float] = 255
        ):
        """_summary_

        Parameters
        ----------
        means : Union[list[float, float, float], tuple[float, float, float], float], optional
            _description_, by default 128
        stds : Union[list[float, float, float], tuple[float, float, float], float], optional
            _description_, by default 255
        """
        if isinstance(means, Iterable):
            means = np.array(means)[..., None, None]
        if isinstance(stds, Iterable):
            stds = np.array(stds)[..., None, None]
        self.means = means
        self.stds = stds

    def __call__(self, transformable: CameraImage):
        tensor_dict = dict(
            img=torch.tensor((np.float32(transformable.img.transpose_(2, 0, 1)) - self.means) / self.stds),
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
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range

    def __call__(self, transformable: Polyline3D):
        # voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
        # voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        voxel_shape = tuple(self.voxel_shape)
        voxel_range = tuple(self.voxel_range)

        Z, X, Y = voxel_shape
        
        fx = X / (voxel_range[1][1] - voxel_range[1][0])
        fy = Y / (voxel_range[2][1] - voxel_range[2][0])
        cx = - voxel_range[1][0] * fx - 0.5
        cy = - voxel_range[2][0] * fy - 0.5

        xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
        points_grid = np.float32([yy, xx])

        tensor_data = {}
        for branch in transformable.dictionary:
            polylines = []
            class_inds = []
            attr_inds = []
            for element in transformable.elements:
                if element['class'] in branch['classes']:
                    points = element['points']
                    polylines.append(np.array([
                        points[..., 1] * fy + cy,
                        points[..., 0] * fx + cx,
                        points[..., 2]
                    ]).T)
                    class_inds.append(transformable.dictionary[branch]['classes'].index(element['class']))
                    attr_inds.append(transformable.dictionary[branch]['attrs'].index(element['attr']))
                    # TODO: add ignore_mask according to ignore classes and attrs
            num_class_channels = len(transformable.dictionary[branch]['classes'])
            num_attr_channels = len(transformable.dictionary[branch]['attrs'])
            branch_seg_im = np.zeros((1 + num_class_channels + num_attr_channels, X, Y))

            line_ims = []
            dist_ims = []
            vec_ims = []
            dir_ims = []
            height_ims = []

            for polyline, class_ind, attr_ind in zip(polylines, class_inds, attr_inds):
                for line_3d in zip(polyline[:-1], polyline[1:]):
                    line_3d = np.float32(line_3d)
                    line_bev = line_3d[:, :2]
                    polygon = expand_line_2d(line_bev, radius=0.5)
                    polygon_int = np.round(polygon).astype(int)
                    # seg_bev_im
                    cv2.fillPoly(branch_seg_im[0], [polygon_int], 1)
                    cv2.fillPoly(branch_seg_im[1 + class_ind], [polygon_int], 1)
                    cv2.fillPoly(branch_seg_im[1 + num_class_channels + attr_ind], [polygon_int], 1)
                    # line segment
                    line_im = cv2.fillPoly(np.zeros((X, Y)), [polygon_int], 1)
                    line_ims.append(line_im)
                    # line direction regressions
                    line_dir = line_bev[1] - line_bev[0]
                    line_length = np.linalg.norm(line_dir)
                    line_dir /= line_length
                    line_dir_vert = line_dir[::-1] * [1, -1]
                    vec_map = vec_point2line_along_direction(points_grid, line_bev, line_dir_vert)
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
                        points_grid + vec_map - line_bev[0][..., None, None], axis=0
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
                'seg': torch.tensor(branch_seg_im),
                'reg': torch.tensor(branch_reg_im)
            }
        
        return tensor_data



@TENSOR_SMITHS.register_module()
class PlanarPolygon3D(TensorSmith):
    def __init__(self, voxel_shape: list, voxel_range: Tuple[list, list, list]):
        """

        Parameters
        ----------
        voxel_shape : list
            (6, 320, 160) Z, X, Y
        voxel_range : tuple
            ([-0.5, 2.5], [36, -12], [12, -12]) in Z, X, Y coordinate
        """
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range

    def __call__(self, transformable: Polyline3D):
        # voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
        # voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        voxel_shape = tuple(self.voxel_shape)
        voxel_range = tuple(self.voxel_range)

        Z, X, Y = voxel_shape
        
        fx = X / (voxel_range[1][1] - voxel_range[1][0])
        fy = Y / (voxel_range[2][1] - voxel_range[2][0])
        cx = - voxel_range[1][0] * fx - 0.5
        cy = - voxel_range[2][0] * fy - 0.5

        xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
        points_grid = np.float32([yy, xx])

        tensor_data = {}
        for branch in transformable.dictionary:
            polylines = []
            class_inds = []
            attr_inds = []
            for element in transformable.elements:
                if element['class'] in branch['classes']:
                    points = element['points']
                    polylines.append(np.array([
                        points[..., 1] * fy + cy,
                        points[..., 0] * fx + cx,
                        points[..., 2]
                    ]).T)
                    class_inds.append(transformable.dictionary[branch]['classes'].index(element['class']))
                    attr_inds.append(transformable.dictionary[branch]['attrs'].index(element['attr']))
                    # TODO: add ignore_mask according to ignore classes and attrs
            num_class_channels = len(transformable.dictionary[branch]['classes'])
            num_attr_channels = len(transformable.dictionary[branch]['attrs'])
            branch_seg_im = np.zeros((2 + num_class_channels + num_attr_channels, X, Y))

            line_ims = []
            dist_ims = []
            vec_ims = []
            dir_ims = []
            height_ims = []

            for polyline, class_ind, attr_ind in zip(polylines, class_inds, attr_inds):
                polyline_int = np.round(polyline).astype(int)
                # seg_bev_im
                cv2.fillPoly(branch_seg_im[2 + class_ind], [polyline_int], 1)
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
                    vec_map = vec_point2line_along_direction(points_grid, line_bev, line_dir_vert)
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
                        points_grid + vec_map - line_bev[0][..., None, None], axis=0
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
                'seg': torch.tensor(branch_seg_im),
                'reg': torch.tensor(branch_reg_im)
            }
        
        return tensor_data
    
    def inverse(self, tensor_data):
        pass



@TENSOR_SMITHS.register_module()
class PlanarParkingSlot3D(TensorSmith):
    def __init__(self, voxel_shape, voxel_range):
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range
    
    @staticmethod
    def _get_height_map(
            slot_points_3d: np.ndarray[Tuple[4, 3]], 
            points_grid_bev: np.ndarray[Tuple[2, int, int]]
        ):
        vec_01 = slot_points_3d[1] - slot_points_3d[0]
        vec_12 = slot_points_3d[2] - slot_points_3d[1]
        vec_23 = slot_points_3d[3] - slot_points_3d[2]
        vec_30 = slot_points_3d[0] - slot_points_3d[3]
        X, Y = points_grid_bev[0].shape
        # plane based on triangle_012
        a_012, b_012, c_012 = norm_vec_012 = np.cross(vec_01, vec_12)
        d_012 = - (norm_vec_012 * slot_points_3d[1]).sum() / np.linalg.norm(norm_vec_012)
        h_012 = - ((points_grid_bev[::-1] * [[[a_012]], [[b_012]]]).sum(axis=0) + d_012) / c_012
        mask_012 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[0, 1, 2], :2][..., [1, 0]].astype(int)], 1)
        # plane based on triangle_230
        a_230, b_230, c_230 = norm_vec_230 = np.cross(vec_23, vec_30)
        d_230 = - (norm_vec_230 * slot_points_3d[3]).sum() / np.linalg.norm(norm_vec_230)
        h_230 = - ((points_grid_bev[::-1] * [[[a_230]], [[b_230]]]).sum(axis=0) + d_230) / c_230
        mask_230 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[2, 3, 0], :2][..., [1, 0]].astype(int)], 1)
        # plane based on triangle_013
        a_013, b_013, c_013 = norm_vec_013 = np.cross(vec_01, vec_30)
        d_013 = - (norm_vec_013 * slot_points_3d[0]).sum() / np.linalg.norm(norm_vec_013)
        h_013 = - ((points_grid_bev[::-1] * [[[a_013]], [[b_013]]]).sum(axis=0) + d_013) / c_013
        mask_013 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[0, 1, 3], :2][..., [1, 0]].astype(int)], 1)
        # plane based on triangle_123
        a_123, b_123, c_123 = norm_vec_123 = np.cross(vec_12, vec_23)
        d_123 = - (norm_vec_123 * slot_points_3d[2]).sum() / np.linalg.norm(norm_vec_123)
        h_123 = - ((points_grid_bev[::-1] * [[[a_123]], [[b_123]]]).sum(axis=0) + d_123) / c_123
        mask_123 = cv2.fillPoly(np.zeros((X, Y)), [slot_points_3d[[1, 2, 3], :2][..., [1, 0]].astype(int)], 1)
        # average on all four planes
        overlaps = mask_012 + mask_230 + mask_013 + mask_123 + 1e-3
        height_map = (mask_012 * h_012 + mask_230 * h_230 + mask_013 * h_013 + mask_123 * h_123) / overlaps
        return height_map
 

    def __call__(self, transformable: ParkingSlot3D):
        # voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
        # voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
        voxel_shape = tuple(self.voxel_shape)
        voxel_range = tuple(self.voxel_range)

        Z, X, Y = voxel_shape
        
        fx = X / (voxel_range[1][1] - voxel_range[1][0])
        fy = Y / (voxel_range[2][1] - voxel_range[2][0])
        cx = - voxel_range[1][0] * fx - 0.5
        cy = - voxel_range[2][0] * fy - 0.5

        xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
        points_grid_bev = np.float32([yy, xx])

        
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
            linewidht = int(max(1, fx * 0.15))
            cv2.polylines(seg_im[1], [slot_points_bev_int], linewidht, 1)
            # 2: corners
            radius = int(max(1, fx * 0.3))
            for corner in slot_points_bev_int:
                cv2.circle(seg_im[2], corner, radius, 1)
            # 3: entrance _seg
            if entrance_length > side_length:
                cv2.circle(seg_im[3], slot_points_bev[[0, 3]].mean(0).astype(int), radius * 2, 1)
            else:
                cv2.circle(seg_im[3], slot_points_bev[[0, 1]].mean(0).astype(int), radius * 2, 1)
            # add height map
            height_map[0] = self._get_height_map(slot_points_3d, points_grid_bev)
            for point_3d in slot_points_3d:
                cv2.circle(height_map[0], point_3d[[1, 0]].astype(int), radius, point_3d[2])
            
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
