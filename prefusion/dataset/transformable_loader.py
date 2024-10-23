from pathlib import Path
from typing import Callable, Any, TYPE_CHECKING, Dict

import mmcv
import numpy as np

from prefusion.registry import TRANSFORMABLE_LOADERS
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.utils import read_pcd, read_ego_mask
from prefusion.dataset.transform import (
    Transformable,
    CameraImage, 
    CameraImageSet,
    CameraDepth,
    CameraDepthSet,
    CameraSegMask,
    CameraSegMaskSet,
    LidarPoints,
    EgoPose,
    EgoPoseSet,
    Bbox3D,
    Polyline3D,
    Polygon3D,
    ParkingSlot3D,
    OccSdfBev,
    SegBev,
    OccSdf3D,
)


if TYPE_CHECKING:
    from prefusion.dataset import IndexInfo


class TransformableLoader:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> Transformable:
        """Load Transformable data

        Parameters
        ----------
        name : str
            name of the transformable (best to be unique)
        scene_data : Dict
            >>> {
            >>>     scene_info: ... ,
            >>>     meta_info: ... , 
            >>>     frame_info: {
            >>>         1698825828064: ...,
            >>>         1698825828164: ...,
            >>>         ...
            >>>     }
            >>> }

        index_info : IndexInfo
            index info of current group batch
        tensor_smith : TensorSmith, optional
            tensor smith of the corresponding transformable, by default None

        Returns
        -------
        Transformable
            _description_
        """

        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "load" function')


@TRANSFORMABLE_LOADERS.register_module()
class CameraImageSetLoader(TransformableLoader):
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


@TRANSFORMABLE_LOADERS.register_module()
class CameraDepthSetLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> CameraDepthSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        calib = scene_data["scene_info"]["calibration"]

        camera_depths = {
            cam_id: CameraDepth(
                name=f"{name}:{cam_id}",
                cam_id=cam_id,
                cam_type=calib[cam_id]["camera_type"],
                img=np.load(self.data_root / frame_info['camera_image_depth'][cam_id])['depth'][..., None].astype(np.float32),
                ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id]),
                extrinsic=calib[cam_id]["extrinsic"],
                intrinsic=calib[cam_id]["intrinsic"],
                depth_mode="d",
                tensor_smith=tensor_smith,
            )
            for cam_id in frame_info["camera_image_depth"]
        }
        return CameraDepthSet(name, camera_depths)


@TRANSFORMABLE_LOADERS.register_module()
class CameraSegMaskSetLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> CameraSegMaskSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        calib = scene_data["scene_info"]["calibration"]

        camera_segs = {
            cam_id: CameraSegMask(
                name=f"{name}:{cam_id}",
                cam_id=cam_id,
                cam_type=calib[cam_id]["camera_type"],
                img=mmcv.imread(self.data_root / frame_info["camera_image_seg"][cam_id], flag="unchanged"),
                ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id]),
                extrinsic=calib[cam_id]["extrinsic"],
                intrinsic=calib[cam_id]["intrinsic"],
                dictionary=dictionary,
                tensor_smith=tensor_smith,
            )
            for cam_id in frame_info["camera_image_seg"]
        }
        return CameraSegMaskSet(name, camera_segs)


@TRANSFORMABLE_LOADERS.register_module()
class LidarPointsLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> LidarPoints:
        frame = scene_data["frame_info"][index_info.frame_id]
        points = read_pcd(self.data_root / frame["lidar_points"]["lidar1"])
        points = np.pad(points, [[0, 0], [0, 1]], constant_values=0)
        return LidarPoints(name, points[:, :3], points[:, 3:], tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class EgoPoseSetLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> EgoPoseSet:
        scene = scene_data['frame_info']

        def _create_pose(frame_id, rel_pos):
            return EgoPose(
                f"{name}:{rel_pos}:{frame_id}",
                frame_id, 
                scene[frame_id]["ego_pose"]["rotation"], 
                scene[frame_id]["ego_pose"]["translation"], 
                tensor_smith=tensor_smith
            )

        poses = {}

        cnt = 0
        cur = index_info
        while cur.prev is not None:
            rel_pos = f"-{cnt+1}" # relative position
            poses[rel_pos] = _create_pose(cur.prev.frame_id, rel_pos)
            cur = cur.prev
            cnt += 1

        cur = index_info
        poses["0"] = _create_pose(cur.frame_id, "0")

        cnt = 0
        while cur.next is not None:
            rel_pos = f"+{cnt+1}" # relative position
            poses[rel_pos] = _create_pose(cur.next.frame_id, rel_pos)
            cur = cur.next
            cnt += 1

        sorted_poses = dict(sorted(poses.items(), key=lambda x: int(x[0])))

        return EgoPoseSet(name, transformables=sorted_poses)


@TRANSFORMABLE_LOADERS.register_module()
class Bbox3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> Bbox3D:
        elements = scene_data["frame_info"][index_info.frame_id]["3d_boxes"]
        return Bbox3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class Polyline3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> Polyline3D:
        elements = scene_data["frame_info"][index_info.frame_id]["3d_polylines"]
        return Polyline3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class Polygon3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> Polygon3D:
        elements = scene_data["frame_info"][index_info.frame_id]["3d_polylines"]
        return Polygon3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class ParkingSlot3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> ParkingSlot3D:
        elements = scene_data["frame_info"][index_info.frame_id]["3d_polylines"]
        return ParkingSlot3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class OccSdfBevLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> OccSdfBev:
        frame = scene_data["frame_info"][index_info.frame_id]
        occ_path = frame["occ_sdf"]["occ_bev"]
        sdf_path = frame["occ_sdf"]["sdf_bev"]
        height_path = frame["occ_sdf"]["height_bev"]
        return OccSdfBev(
            name=name,
            src_view_range=scene_data["meta_info"]["space_range"]["occ"],  # ego system,
            occ=mmcv.imread(occ_path),
            sdf=mmcv.imread(sdf_path),
            height=mmcv.imread(height_path),
            dictionary=dictionary,
            mask=None,
            tensor_smith=tensor_smith,
        )


@TRANSFORMABLE_LOADERS.register_module()
class SegBevLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> SegBev:
        raise NotImplementedError


@TRANSFORMABLE_LOADERS.register_module()
class OccSdf3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> OccSdf3D:
        raise NotImplementedError
