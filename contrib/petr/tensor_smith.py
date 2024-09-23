from typing import TYPE_CHECKING

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from prefusion.registry import TENSOR_SMITHS
from prefusion.dataset.tensor_smith import TensorSmith

if TYPE_CHECKING:
    from prefusion.dataset.transform import Bbox3D


__all__ = ["Bbox3DCorners", "Bbox3D_XYZ_LWH_Yaw_VxVy"]


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
            "classes": torch.tensor([self.classes.index(ele["class"]) for ele in transformable.elements], dtype=torch.float32),
            "xyz_lwh_yaw_vxvy": torch.tensor(
                np.array(
                    [
                        bx["translation"].flatten().tolist()
                        + bx["size"]
                        + Rotation.from_matrix(bx["rotation"]).as_euler("XYZ", degrees=False)[2:].tolist()
                        + bx["velocity"].flatten()[:2].tolist()
                        for bx in transformable.elements
                    ]
                ),
                dtype=torch.float32,
            ),
        }
