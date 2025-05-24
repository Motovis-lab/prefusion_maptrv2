from typing import TYPE_CHECKING, List, Union, Iterable
import math

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from mmcv.image.geometric import impad_to_multiple

from prefusion.registry import TENSOR_SMITHS
from prefusion.dataset.tensor_smith import TensorSmith

if TYPE_CHECKING:
    from prefusion.dataset.transform import Bbox3D, CameraImage


__all__ = ["Bbox3DBasic"]


@TENSOR_SMITHS.register_module()
class Bbox3DBasic(TensorSmith):
    def __init__(self, classes, voxel_range):
        self.classes = classes
        self.voxel_range = voxel_range

    def __call__(self, transformable: "Bbox3D"):
        def _in_voxel_range(translation: List[float]) -> bool:
            return (
                self.voxel_range[1][1] <= translation[0] <= self.voxel_range[1][0]
                and self.voxel_range[2][1] <= translation[1] <= self.voxel_range[2][0]
                and self.voxel_range[0][0] <= translation[2] <= self.voxel_range[0][1]
            )
        
        indices_within_range = [i for i, bx in enumerate(transformable.elements) if _in_voxel_range(bx["translation"].flatten().tolist())]

        return {
            "classes": torch.tensor([
                self.classes.index(ele["class"]) 
                for i, ele in enumerate(transformable.elements)
                if i in indices_within_range
            ], dtype=torch.int64),
            "xyz_lwh_yaw_vx_vy": torch.tensor(
                np.array(
                    [
                        bx["translation"].flatten().tolist()
                        + bx["size"]
                        + [Rotation.from_matrix(bx["rotation"]).as_euler("XYZ", degrees=False)[2]]
                        # + self._encode_yaw(Rotation.from_matrix(bx["rotation"]).as_euler("XYZ", degrees=False)[2])
                        + bx["velocity"].flatten()[:2].tolist()
                        for i, bx in enumerate(transformable.elements)
                        if i in indices_within_range
                    ]
                ),
                dtype=torch.float32,
            ),
            "corners": torch.tensor(transformable.corners[indices_within_range]),
        }
    
    @staticmethod
    def _encode_yaw(yaw: float) -> List[float]:
        return [math.sin(yaw), math.cos(yaw)]


@TENSOR_SMITHS.register_module()
class DivisibleCameraImageTensor(TensorSmith):
    def __init__(self, 
            means: Union[list[float, float, float], tuple[float, float, float], float] = 128, 
            stds: Union[list[float, float, float], tuple[float, float, float], float] = 255,
            image_size_divisor: int = 32,
            image_pad_value: float = 0.0,
        ):
        if isinstance(means, Iterable):
            means = np.array(means, dtype=np.float32)
        if isinstance(stds, Iterable):
            stds = np.array(stds, dtype=np.float32)
        self.means = means
        self.stds = stds
        self.image_size_divisor = image_size_divisor
        self.image_pad_value = image_pad_value

    def __call__(self, transformable: "CameraImage"):
        tensor_dict = dict(
            img=torch.tensor(np.float32(
                    impad_to_multiple(
                        (transformable.img - self.means) / self.stds, 
                        self.image_size_divisor, 
                        pad_val=self.image_pad_value)).transpose(2, 0, 1)
            ),
            ego_mask=torch.tensor(
                impad_to_multiple(
                    transformable.ego_mask, 
                    self.image_size_divisor, 
                    pad_val=self.image_pad_value)
            ),
        )
        return tensor_dict
