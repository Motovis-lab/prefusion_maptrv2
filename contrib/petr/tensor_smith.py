from typing import TYPE_CHECKING, List
import math

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from prefusion.registry import TENSOR_SMITHS
from prefusion.dataset.tensor_smith import TensorSmith

if TYPE_CHECKING:
    from prefusion.dataset.transform import Bbox3D


__all__ = ["Bbox3DBasic"]


@TENSOR_SMITHS.register_module()
class Bbox3DBasic(TensorSmith):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, transformable: "Bbox3D"):
        return {
            "classes": torch.tensor([self.classes.index(ele["class"]) for ele in transformable.elements], dtype=torch.int64),
            "xyz_lwh_yaw_vx_vy": torch.tensor(
                np.array(
                    [
                        bx["translation"].flatten().tolist()
                        + bx["size"]
                        + [Rotation.from_matrix(bx["rotation"]).as_euler("XYZ", degrees=False)[2]]
                        # + self._encode_yaw(Rotation.from_matrix(bx["rotation"]).as_euler("XYZ", degrees=False)[2])
                        + bx["velocity"].flatten()[:2].tolist()
                        for bx in transformable.elements
                    ]
                ),
                dtype=torch.float32,
            ),
            "corners": torch.tensor(transformable.corners),
        }
    
    @staticmethod
    def _encode_yaw(yaw: float) -> List[float]:
        return [math.sin(yaw), math.cos(yaw)]
