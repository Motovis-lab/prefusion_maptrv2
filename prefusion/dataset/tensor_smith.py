import cv2
import torch
import numpy as np

from .utils import (
    expand_line_2d, _sign, INF_DIST,
    vec_point2line_along_direction, 
    dist_point2line_along_direction,
    get_cam_type,
    VoxelLookUpTableGenerator
)


class BaseTensorSmith:
    def to_tensor(self, transformable, *args, **kwargs):
        return transformable


class PlanarSegBevSmith(BaseTensorSmith):
    def to_tensor(self, transformable, bev_resolution, bev_range, **kwargs):
        pass


class PlanarPolyline3D(BaseTensorSmith):
    def to_tensor(self, transformable, voxel_shape, voxel_range, **kwargs):
        # voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
        # voxel_range=([-0.5, 2.5], [36, -12], [12, -12])

        Z, X, Y = voxel_shape
        
        fx = X / (voxel_range[1][1] - voxel_range[1][0])
        fy = Y / (voxel_range[2][1] - voxel_range[2][0])
        cx = - voxel_range[1][0] * fx - 0.5
        cy = - voxel_range[2][0] * fy - 0.5

        xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
        points_grid = np.float32([xx, yy])

        # to_tensor(self, bev_resolution, bev_range, **kwargs)
        # W, H = bev_resolution
        # # 'bev_range': [back, front, right, left, bottom, up], # in ego system
        # fx = H / (bev_range[0] - bev_range[1])
        # fy = W / (bev_range[2] - bev_range[3])
        # cx = - bev_range[1] * fx - 0.5
        # cy = - bev_range[3] * fy - 0.5

        # xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        # points_grid = np.float32([xx, yy])

        tensor_data = {}
        for branch in transformable.dictionary:
            polylines = []
            class_inds = []
            attr_inds = []
            for element in transformable.data['elements']:
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
            branch_seg_im = np.zeros((1 + num_class_channels + num_attr_channels, H, W))

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
                    line_im = cv2.fillPoly(np.zeros((H, W)), [polygon_int], 1)
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


class PlanarPolygon3DSmith(BaseTensorSmith):

    def to_tensor(self, transformable, voxel_shape, voxel_range, **kwargs):
        # voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
        # voxel_range=([-0.5, 2.5], [36, -12], [12, -12])

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
            for element in transformable.data['elements']:
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
            branch_seg_im = np.zeros((2 + num_class_channels + num_attr_channels, H, W))

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
                    line_im = cv2.fillPoly(np.zeros((H, W)), [polygon_int], 1)
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
