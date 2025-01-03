from typing import TYPE_CHECKING

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from spconv.pytorch.utils import PointToVoxel

from prefusion.registry import TENSOR_SMITHS
from prefusion.dataset.tensor_smith import TensorSmith

if TYPE_CHECKING:
    from prefusion.dataset.transform import Bbox3D, LidarPoints

__all__ = ["Bbox3DCorners", "Bbox3D_XYZ_LWH_Yaw_VxVy", 'PointsToVoxelsTensor']


@TENSOR_SMITHS.register_module()
class Bbox3DCorners(TensorSmith):
    def __init__(self):
        pass

    def __call__(self, transformable: "Bbox3D"):
        return {
            "classes": [ele["class"] for ele in transformable.elements],
            "bbox3d_corners": torch.tensor(transformable.corners),
        }


@TENSOR_SMITHS.register_module()
class Bbox3D_XYZ_LWH_Yaw_VxVy(TensorSmith):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, transformable: "Bbox3D"):
        return {
            "classes": torch.tensor([self.classes.index(ele["class"]) for ele in transformable.elements],
                                    dtype=torch.int64),
            "xyz_lwh_yaw_vxvy": torch.tensor(
                np.array(
                    [
                        bx["translation"].flatten().tolist()
                        + np.array(bx["size"]).tolist()
                        + Rotation.from_matrix(bx["rotation"]).as_euler("XYZ", degrees=False)[2:].tolist()
                        + bx["velocity"].flatten()[:2].tolist()
                        for bx in transformable.elements
                    ]
                ),
                dtype=torch.float32,
            ),
        }


@TENSOR_SMITHS.register_module()
class PointsToVoxelsTensor(TensorSmith):
    def __init__(self, voxel_size, max_point_per_voxel=10, max_voxels=100000, max_input_points=1200000,
                 point_cloud_range=[]):
        self.voxel_size = voxel_size
        self.max_point_per_voxel = max_point_per_voxel
        self.max_input_points = max_input_points
        self.num_features = 5
        self.max_voxels = max_voxels
        self.point_cloud_range = point_cloud_range
        self.voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            max_num_points_per_voxel=self.max_point_per_voxel,
            max_num_voxels=self.max_voxels,
            num_point_features=self.num_features,
        )

    def __call__(self, transformable: "LidarPoints"):
        points = np.concatenate([transformable.positions, transformable.attributes], 1)
        points = torch.tensor(points)
        features, coors, num_points = self.voxel_generator(points)  # todo: whether clone?
        points_mean = features[:, :, :self.num_features].sum(  # hardvfe
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        points_mean = points_mean.contiguous()
        return {
            "res_voxels": points_mean,
            "res_coors": coors,
            "res_num_points": num_points,
            "points": points
        }
