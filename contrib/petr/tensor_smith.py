from typing import TYPE_CHECKING

from prefusion.registry import TENSOR_SMITHS
from prefusion.dataset.tensor_smith import TensorSmith

if TYPE_CHECKING:
    from prefusion.dataset.transform import Bbox3D


__all__ = ["Bbox3DCorners"]


@TENSOR_SMITHS.register_module()
class Bbox3DCorners(TensorSmith):
    def __init__(self):
        pass
    
    def __call__(self, transformable: "Bbox3D"):
        return {
            "classes": [ele['class'] for ele in transformable.elements],
            "bbox3d_corners": transformable.corners
        }
