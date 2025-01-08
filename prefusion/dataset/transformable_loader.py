from copy import deepcopy
from collections import defaultdict, UserDict, Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union, List
import warnings

import cv2
import mmcv
import numpy as np
from scipy.spatial.transform import Rotation

from prefusion.registry import TRANSFORMABLE_LOADERS
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.utils import read_pcd, read_ego_mask, get_reversed_mapping
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
    Variable,
)


if TYPE_CHECKING:
    from prefusion.dataset.index_info import IndexInfo


__all__ = [
    "CameraImageSetLoader", "CameraDepthSetLoader", "CameraSegMaskSetLoader",
    "LidarPointsLoader", "EgoPoseSetLoader",
    "Bbox3DLoader", "AdvancedBbox3DLoader", "NuscenesCameraImageSetLoader",
    "Polyline3DLoader", "Polygon3DLoader", "ParkingSlot3DLoader",
    "OccSdfBevLoader", "SegBevLoader", "OccSdf3DLoader",
]


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


class CameraSetLoader(TransformableLoader):
    def __init__(self, data_root: Path, camera_mapping: dict = None) -> None:
        super().__init__(data_root)
        self.camera_mapping = camera_mapping


@TRANSFORMABLE_LOADERS.register_module()
class CameraImageSetLoader(CameraSetLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> CameraImageSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        calib = scene_data["scene_info"]["calibration"]
        camera_images = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_info["camera_image"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            if cam_id_ori in frame_info["camera_image"]:
                # need to get the real
                camera_images[cam_id] = CameraImage(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=mmcv.imread(self.data_root / frame_info["camera_image"][cam_id_ori]),
                    ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id_ori]),
                    extrinsic=calib[cam_id_ori]["extrinsic"],
                    intrinsic=calib[cam_id_ori]["intrinsic"],
                    tensor_smith=tensor_smith,
                )
        return CameraImageSet(name, camera_images)


@TRANSFORMABLE_LOADERS.register_module()
class CameraTimeImageSetLoader(CameraSetLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> CameraImageSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        calib = scene_data["scene_info"]["calibration"]
        camera_images = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_info["camera_image"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            def Rt2T(R, t):
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                return T

            def T2Rt(T):
                return (T[:3, :3], T[:3, 3])

            Tec = Rt2T(calib[cam_id_ori]["extrinsic"][0], calib[cam_id_ori]["extrinsic"][1])
            Twe0 = Rt2T(frame_info['ego_pose']['rotation'], frame_info['ego_pose']['translation'])
            if cam_id_ori in frame_info["camera_image"]:
                Twe1 = frame_info["camera_image"][cam_id_ori]['Twe']
                Te0c = np.linalg.inv(Twe0) @ Twe1 @ Tec
                camera_images[cam_id] = CameraImage(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=mmcv.imread(self.data_root / frame_info["camera_image"][cam_id_ori]['path']),
                    ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id_ori]),
                    extrinsic=T2Rt(Te0c),
                    intrinsic=calib[cam_id_ori]["intrinsic"],
                    tensor_smith=tensor_smith,
                )
        return CameraImageSet(name, camera_images)



@TRANSFORMABLE_LOADERS.register_module()
class NuscenesCameraImageSetLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> CameraImageSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        camera_images = {
            cam_id: CameraImage(
                name=f"{name}:{cam_id}",
                cam_id=cam_id,
                cam_type=frame_info["camera_image"][cam_id]["calibration"]["camera_type"],
                img=mmcv.imread(self.data_root / frame_info["camera_image"][cam_id]["path"]),
                ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id]),
                extrinsic=frame_info["camera_image"][cam_id]["calibration"]["extrinsic"],
                intrinsic=frame_info["camera_image"][cam_id]["calibration"]["intrinsic"],
                tensor_smith=tensor_smith,
            )
            for cam_id in frame_info["camera_image"]
        }
        return CameraImageSet(name, camera_images)


@TRANSFORMABLE_LOADERS.register_module()
class CameraDepthSetLoader(CameraSetLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> CameraDepthSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        calib = scene_data["scene_info"]["calibration"]
        camera_depths = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_info["camera_image_depth"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            if cam_id_ori in frame_info["camera_image_depth"]:
                camera_depths[cam_id] = CameraDepth(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=np.load(self.data_root / frame_info['camera_image_depth'][cam_id_ori])['depth'][..., None].astype(np.float32),
                    ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id_ori]),
                    extrinsic=calib[cam_id_ori]["extrinsic"],
                    intrinsic=calib[cam_id_ori]["intrinsic"],
                    depth_mode="d",
                    tensor_smith=tensor_smith,
                )
        return CameraDepthSet(name, camera_depths)


@TRANSFORMABLE_LOADERS.register_module()
class CameraSegMaskSetLoader(CameraSetLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> CameraSegMaskSet:
        scene_info = scene_data["scene_info"]
        frame_info = scene_data["frame_info"][index_info.frame_id]
        calib = scene_data["scene_info"]["calibration"]
        camera_segs = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_info["camera_image_seg"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            if cam_id_ori in frame_info["camera_image_seg"]:
                camera_segs[cam_id] = CameraSegMask(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=mmcv.imread(self.data_root / frame_info["camera_image_seg"][cam_id_ori], flag="unchanged"),
                    ego_mask=read_ego_mask(self.data_root / scene_info["camera_mask"][cam_id_ori]),
                    extrinsic=calib[cam_id_ori]["extrinsic"],
                    intrinsic=calib[cam_id_ori]["intrinsic"],
                    dictionary=dictionary,
                    tensor_smith=tensor_smith,
                )
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
        """Basic loader that loads bbox3d info

        Parameters
        ----------
        dictionary : Dict, optional
            class and attr info of the bboxes, by default None
            e.g.
            ```
            dictionary = {
                "classes": ["car", "person", "bicycle"],
                "attrs": ['is_door_open', 'is_trunk_open']
            }
            ```

        Returns
        -------
        Bbox3D
            Bbox3D transformable
        """
        elements = deepcopy(scene_data["frame_info"][index_info.frame_id]["3d_boxes"])
        for ele in elements:
            ele["attr"] = list(ele["attr"].values()) if isinstance(ele["attr"], dict) else ele["attr"]
        return Bbox3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class AdvancedBbox3DLoader(TransformableLoader):

    rot90deg = Rotation.from_euler("XYZ", [0, 0, 90], degrees=True).as_matrix()
    
    def __init__(self, data_root: Path, class_mapping: Dict = None, attr_mapping: Dict = None, axis_rearrange_method="none") -> None:
        """ Advanced Bbox3D Loader
        # CAUTION
        FIXME:
        only a minimum check has been applied to class_mapping and attr_mapping
        there's still some configurations could lead to strange class mapping behavior,
        such as: "new_cls1": ["c1\:\:attr1.True", "c1\:\:attr2.True"]

        Parameters
        ----------
        data_root : Path
            dataset root
        class_mapping : Dict, optional
            mapping info between original class and desired class
            e.g.
            ```
            class_mapping = {
                "car": ["class.vehicle.passenger_car", "class.vehicle.bus"], 
                "person": ["class.pedestrian.pedestrian"], 
                "bicycle": ["class.cycle.bicycle"],
            }
        attr_mapping : Dict, optional
            mapping info between original attr and desired attr
            e.g.
            ```
            attr_mapping = {
                "is_trunk_open": ['attr.vehicle.is_trunk_open.true'],
                "is_with_rider": ['attr.cycle.is_with_rider.true'],
            }
            ```
        axis_rearrange_method: str, optional
            axis rearrange method, choices: ["none", "longer_edge_as_x", "longer_edge_as_y"], default is "none"
            setting this argument to control the x-axis and y-axis orientation.
        """
        super().__init__(data_root)
        self.class_mapping = ClassMapping(class_mapping)
        self.attr_mapping = AttrMapping(attr_mapping)
        self.axis_rearrange_method = axis_rearrange_method
        assert axis_rearrange_method in ["none", "longer_edge_as_x", "longer_edge_as_y"]

    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> Bbox3D:
        updated_dictionary = self._update_dictionary(dictionary)
        elements = deepcopy(scene_data["frame_info"][index_info.frame_id]["3d_boxes"])
        for ele in elements:
            ele["class"] = self.class_mapping.get_mapped_class(ele["class"], ele["attr"])
            ele["attr"] = self.attr_mapping.get_mapped_attr(ele["attr"])
            if self.axis_rearrange_method != "none" and ele["class"] in updated_dictionary["classes"]:
                self.rearrange_axis_(ele)
        return Bbox3D(name, elements, updated_dictionary, tensor_smith=tensor_smith)
    
    def _update_dictionary(self, dictionary: Dict = None) -> Dict:
        if not dictionary:
            dictionary = {"classes": [], "attrs": []}
        else:
            dictionary = {"classes": deepcopy(dictionary.get("classes", [])), "attrs": deepcopy(dictionary.get("attrs", []))}

        if self.class_mapping:
            dictionary.update(classes=list(self.class_mapping.keys()))
        else:
            if not dictionary.get("classes"):
                warnings.warn("Neither class_mapping nor dictionary['classes'] is provided for AdvancedBbox3DLoader", UserWarning)

        if self.attr_mapping:
            dictionary.update(attrs=list(self.attr_mapping.keys()))

        return dictionary
    
    def rearrange_axis_(self, ele):
        cur_longer_edge = "x" if ele["size"][0] >= ele["size"][1] else "y"
        if (self.axis_rearrange_method == "longer_edge_as_y" and cur_longer_edge == "x") or (
            self.axis_rearrange_method == "longer_edge_as_x" and cur_longer_edge == "y"
        ):
            ele["size"] = np.array(ele["size"])[[1, 0, 2]].tolist()  # change type back to <list>
            ele["rotation"] = self._intrinsic_rotate_90_deg(ele["rotation"])
    
    @staticmethod
    def _intrinsic_rotate_90_deg(original_rot_mat):
        return original_rot_mat @ AdvancedBbox3DLoader.rot90deg


@TRANSFORMABLE_LOADERS.register_module()
class Polyline3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> Polyline3D:
        elements = deepcopy(scene_data["frame_info"][index_info.frame_id]["3d_polylines"])
        for ele in elements:
            ele["attr"] = list(ele["attr"].values()) if isinstance(ele["attr"], dict) else ele["attr"]
        return Polyline3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class Polygon3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> Polygon3D:
        elements = deepcopy(scene_data["frame_info"][index_info.frame_id]["3d_polylines"])
        for ele in elements:
            ele["attr"] = list(ele["attr"].values()) if isinstance(ele["attr"], dict) else ele["attr"]
        return Polygon3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class ParkingSlot3DLoader(TransformableLoader):
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, dictionary: Dict = None, **kwargs) -> ParkingSlot3D:
        elements = deepcopy(scene_data["frame_info"][index_info.frame_id]["3d_polylines"])
        for ele in elements:
            ele["attr"] = list(ele["attr"].values()) if isinstance(ele["attr"], dict) else ele["attr"]
        return ParkingSlot3D(name, elements, dictionary, tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class OccSdfBevLoader(TransformableLoader):
    def __init__(self, data_root: Path, src_voxel_range: List = None, load_sdf: bool = False) -> None:
        super().__init__(data_root)
        self.src_voxel_range = src_voxel_range
        self.load_sdf = load_sdf
    
    def _gen_height_and_mask(self, height_lidar, mask_lidar, height_ground, mask_ground):
        mask = mask_lidar + mask_ground
        mask_left = (mask_ground - mask_lidar) > 0.5
        height = height_lidar * mask_lidar + height_ground * mask_left
        return height, mask

    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> OccSdfBev:
        frame = scene_data["frame_info"][index_info.frame_id]
        # get sdf
        if self.load_sdf:
            sdf = cv2.imread(str(self.data_root / frame["occ_sdf"]["sdf"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 860 - 36
        else:
            sdf = None
        # get occ
        # occ_path = str(self.data_root / frame["occ_sdf"]["occ_2d"])
        occ_path = str(self.data_root / frame["occ_sdf"]["occ_map_sdf"])
        occ = mmcv.imread(occ_path)
        # get height
        mask_ground = np.float32(occ[..., 1] > 128)

        # height_ground_path = str(self.data_root / frame["occ_sdf"]["ground"])
        height_ground_path = str(self.data_root / frame["occ_sdf"]["ground_height_map"])
        height_ground = mmcv.imread(height_ground_path, 'unchanged').astype(np.float32) / 3000 - 10

        height_lidar_path = str(self.data_root / frame["occ_sdf"]["bev_height_map"])
        mask_lidar_path = str(self.data_root / frame["occ_sdf"]["bev_lidar_mask"])
        height_lidar = mmcv.imread(height_lidar_path, 'unchanged').astype(np.float32) * 3.0 / 255.0 - 1
        mask_lidar = np.float32(mmcv.imread(mask_lidar_path, 'unchanged') > 128)

        height, mask = self._gen_height_and_mask(height_lidar, mask_lidar, height_ground, mask_ground)

        if self.src_voxel_range is None:
            src_voxel_range = scene_data["meta_info"]["space_range"]["occ"]
        else:
            src_voxel_range = self.src_voxel_range
        return OccSdfBev(
            name=name,
            src_voxel_range=src_voxel_range,  # ego system,
            occ=occ, sdf=sdf, height=height,
            mask=mask,
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


class ClassMapping(UserDict):
    """Assume input to be Dict[str, List[str]]"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchical_mapping = self.get_hierarchical_mapping()

    def get_hierarchical_mapping(self) -> Dict[str, Union[str, Dict[str, str]]]:
        hierarchical_mapping = defaultdict(dict)
        for k, v in self.items():
            for ca in v:
                if "::" in ca:
                    c, a = ca.split("::")
                    if c in hierarchical_mapping and isinstance(hierarchical_mapping[c], str):
                        raise ValueError(f"Source class {c} is used more than once in the mapping.")
                    if c in hierarchical_mapping and a in hierarchical_mapping[c]:
                        raise ValueError(f"Source class {c}::{a} is used more than once in the mapping.")
                    hierarchical_mapping[c][a] = k
                else:
                    if ca in hierarchical_mapping:
                        raise ValueError(f"Source class {ca} is used more than once in the mapping.")
                    hierarchical_mapping[ca] = k
        return hierarchical_mapping

    def get_mapped_class(self, ele_class_name: str, ele_attrs: Dict[str, str] = None) -> str:
        """Get mapped class name from an element's class_name and attrs"""
        if ele_class_name not in self.hierarchical_mapping:
            return ele_class_name  # no matched configuration, keep the original class_name
        
        if isinstance(self.hierarchical_mapping[ele_class_name], str):
            return self.hierarchical_mapping[ele_class_name]
        
        for attr in ele_attrs.values():
            if attr in self.hierarchical_mapping[ele_class_name]:
                return self.hierarchical_mapping[ele_class_name][attr]
        
        return ele_class_name  # no matched configuration, keep the original class_name


class AttrMapping(UserDict):
    """Assume input to be Dict[str, List[str]]"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate_input()

    def validate_input(self, **kwargs):
        """ to prevent the same source attr from being used more than once. 
        Following are the cases we want to prevent:
        case 1:
        ``` 
        {
            "new_attr_name1": ["c1::attr1.True"],
            "new_attr_name2": ["c1::attr1.True"]
        }
        ```
        """
        src_attr_used_cnt = Counter([k for k in get_reversed_mapping(self)])
        for k, v in src_attr_used_cnt.items():
            if v > 1:
                raise ValueError(f"Source attr {k} is used more than once in the mapping.")

    def get_mapped_attr(self, ele_attrs: Dict[str, str] = None) -> List[str]:
        """Get mapped attr name from attrs"""
        reversed_mapping = get_reversed_mapping(self)
        return sorted({reversed_mapping[attr] for attr in ele_attrs.values() if attr in reversed_mapping})



@TRANSFORMABLE_LOADERS.register_module()
class VariableLoader(TransformableLoader):
    def __init__(self, data_root: Path, variable_key: str) -> None:
        super().__init__(data_root)
        self.data_root = data_root
        self.variable_key = variable_key
    def load(self, name: str, scene_data: Dict, index_info: "IndexInfo", tensor_smith: TensorSmith = None) -> ParkingSlot3D:
        variable_value = deepcopy(scene_data["frame_info"][index_info.frame_id][self.variable_key])
        return Variable(name, variable_value, tensor_smith=tensor_smith)
