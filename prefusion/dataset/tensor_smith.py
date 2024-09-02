from typing import List, Tuple, Dict, Union, Iterable

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
            means: Union[List[float, float, float], Tuple[float, float, float], float] = 128, 
            stds: Union[List[float, float, float], Tuple[float, float, float], float] = 255
        ):
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
        points_grid = np.float32([xx, yy])

        tensor_data = {}
        for branch in transformable.dictionary:
            polylines = []
            class_inds = []
            attr_inds = []
            for element in transformable.elements:
                if element['class'] in branch['classes']:
                    points = element['points']
                    polylines.append(np.array([
                        points[..., 0] * fx + cx,
                        points[..., 1] * fy + cy,
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
        points_grid = np.float32([xx, yy])

        tensor_data = {}
        for branch in transformable.dictionary:
            polylines = []
            class_inds = []
            attr_inds = []
            for element in transformable.elements:
                if element['class'] in branch['classes']:
                    points = element['points']
                    polylines.append(np.array([
                        points[..., 0] * fx + cx,
                        points[..., 1] * fy + cy,
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
        h_012 = - ((points_grid_bev * [[[a_012]], [[b_012]]]).sum(axis=0) + d_012) / c_012
        mask_012 = cv2.fillPoly(np.zeros((X, Y)), points_grid_bev[[0, 1, 2]].astype(int), 1)
        # plane based on triangle_230
        a_230, b_230, c_230 = norm_vec_230 = np.cross(vec_23, vec_30)
        d_230 = - (norm_vec_230 * slot_points_3d[3]).sum() / np.linalg.norm(norm_vec_230)
        h_230 = - ((points_grid_bev * [[[a_230]], [[b_230]]]).sum(axis=0) + d_230) / c_230
        mask_230 = cv2.fillPoly(np.zeros((X, Y)), points_grid_bev[[2, 3, 0]].astype(int), 1)
        







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
        points_grid_bev = np.float32([xx, yy])

        
        seg_im = np.zeros((4, X, Y))
        cen_im = np.zeros((1, X, Y))
        reg_im = np.zeros((16, X, Y))

        height_map = np.zeros((1, X, Y))
        # TODO: average two folds. [[0, 1, 2], [0, 2, 3]] and [[0, 1, 3], [1, 2, 3]]

        for element in transformable.elements:
            slot_points_3d = np.float32(element['points']) * [[fx, fy, 1]] + [[cx, cy, 0]]
            entrance_length = np.linalg.norm(element['points'][1] - element['points'][0])
            side_length = np.linalg.norm(element['points'][3] - element['points'][0])
            if entrance_length > side_length:
                slot_points_3d = slot_points_3d[[1, 2, 3, 0]]
            slot_points_bev = slot_points_3d[..., :2]
            slot_points_bev_int = np.round(slot_points_bev).astype(int)

            # define short side and long side
            short_sides = np.float32([
                np.float32([slot_points_3d[0], slot_points_3d[1]]), 
                np.float32([slot_points_3d[2], slot_points_3d[3]])
            ])
            long_sides = np.float32([
                np.float32([slot_points_3d[1], slot_points_3d[2]]),
                np.float32([slot_points_3d[3], slot_points_3d[0]])
            ])
            center_long_line = short_sides.mean(1)
            center_short_line = long_sides.mean(1)

            # gen segmentation and height map
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
            # TODO: add height map
            for point_3d in slot_points_3d:
                cv2.circle(height_map[0], point_3d[:2].astype(int), radius, point_3d[2])
            # get plane function
            # slot_points_3d[[0, 1]]
            
            


        
        # tensor_data = {}
        # for branch in transformable.dictionary:
        #     polylines = []
        #     class_inds = []
        #     attr_inds = []
        #     for element in transformable.elements:
        #         if element['class'] in branch['classes']:
        #             points = element['points']
        #             polylines.append(np.array([
        #                 points[..., 0] * fx + cx,
        #                 points[..., 1] * fy + cy,
        #                 points[..., 2]
        #             ]).T)
        #             class_inds.append(transformable.dictionary[branch]['classes'].index(element['class']))
        #             attr_inds.append(transformable.dictionary[branch]['attrs'].index(element['attr']))
        #             # TODO: add ignore_mask according to ignore classes and attrs
        #     num_class_channels = len(transformable.dictionary[branch]['classes'])
        #     num_attr_channels = len(transformable.dictionary[branch]['attrs'])
        #     branch_seg_im = np.zeros((2 + num_class_channels + num_attr_channels, X, Y))

        #     line_ims = []
        #     dist_ims = []
        #     vec_ims = []
        #     dir_ims = []
        #     height_ims = []

        #     for polyline, class_ind, attr_ind in zip(polylines, class_inds, attr_inds):
        #         polyline_int = np.round(polyline).astype(int)
        #         # seg_bev_im
        #         cv2.fillPoly(branch_seg_im[2 + class_ind], [polyline_int], 1)
        #         cv2.fillPoly(branch_seg_im[2 + num_class_channels + attr_ind], [polyline_int], 1)
        #         for line_3d in zip(polyline[:-1], polyline[1:]):
        #             line_3d = np.float32(line_3d)
        #             line_bev = line_3d[:, :2]
        #             polygon = expand_line_2d(line_bev, radius=0.5)
        #             polygon_int = np.round(polygon).astype(int)
        #             # edge_seg_bev_im
        #             cv2.fillPoly(branch_seg_im[0], [polygon_int], 1)
        #             # line segment
        #             line_im = cv2.fillPoly(np.zeros((X, Y)), [polygon_int], 1)
        #             line_ims.append(line_im)
        #             # line direction regressions
        #             line_dir = line_bev[1] - line_bev[0]
        #             line_length = np.linalg.norm(line_dir)
        #             line_dir /= line_length
        #             line_dir_vert = line_dir[::-1] * [1, -1]
        #             vec_map = vec_point2line_along_direction(points_grid, line_bev, line_dir_vert)
        #             dist_im = line_im * np.linalg.norm(vec_map, axis=0) + (1 - line_im) * INF_DIST
        #             vec_im = line_im * vec_map
        #             abs_dir_im = line_im * np.float32([
        #                 np.abs(line_dir[0]),
        #                 np.abs(line_dir[1]),
        #                 line_dir[0] * line_dir[1]
        #             ])[..., None, None]
        #             dist_ims.append(dist_im)
        #             vec_ims.append(vec_im)
        #             dir_ims.append(abs_dir_im)
        #             # height map
        #             h2s = (line_3d[1, 2] - line_3d[0, 2]) / max(1e-3, line_length)
        #             height_im =  line_3d[0, 2] + h2s * line_im * np.linalg.norm(
        #                 points_grid + vec_map - line_bev[0][..., None, None], axis=0
        #             )
        #             height_ims.append(height_im)

        #     index_im = np.argmin(np.array(dist_ims), axis=0)
        #     branch_vec_im = np.choose(index_im, vec_ims)
        #     branch_dist_im = np.choose(index_im, dist_ims)
        #     branch_dir_im = np.choose(index_im, dir_ims)
        #     branch_height_im = np.choose(index_im, height_ims)

        #     branch_reg_im = np.concatenate([
        #         branch_seg_im * branch_dist_im[None],
        #         branch_vec_im,
        #         branch_dir_im,
        #         branch_height_im[None]
        #     ], axis=0)
        #     # TODO: add branch_ignore_seg_mask according to ignore classes and attrs

        #     tensor_data[branch] = {
        #         'seg': torch.tensor(branch_seg_im),
        #         'reg': torch.tensor(branch_reg_im)
        #     }
        
        # return tensor_data
