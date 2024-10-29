from pathlib import Path
from typing import Callable, Any, TYPE_CHECKING, Dict

import mmcv
import numpy as np
from prefusion.dataset.transformable_loader import TransformableLoader
from prefusion.registry import TRANSFORMABLE_LOADERS
from prefusion.dataset import IndexInfo
from prefusion.dataset.transform import (
    Transformable,
    CameraImage, 
    CameraImageSet
)
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.utils import read_ego_mask

__all__ = ["MonoDepthLoader"]

@TRANSFORMABLE_LOADERS.register_module()
class MonoDepthLoader(TransformableLoader):
    
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> CameraImageSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        calib = scene_data["scene_info"]["calibration"]
        camera_images = {
            cam_id: CameraImage(
                name=f"{name}:{cam_id}",
                cam_id=cam_id,
                cam_type=calib[cam_id]["camera_type"],
                img=mmcv.imread(self.data_root / frame_info["camera_image"][cam_id]),
                ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id]),
                extrinsic=calib[cam_id]["extrinsic"],
                intrinsic=calib[cam_id]["intrinsic"],
                tensor_smith=tensor_smith,
            )
            for cam_id in frame_info["camera_image"]
        }
        return CameraImageSet(name, camera_images)