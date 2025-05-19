import copy
import warnings
from collections import defaultdict, UserDict, Counter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union, List, Optional

import cv2
import mmcv
import numpy as np
from scipy.spatial.transform import Rotation

from prefusion.registry import TRANSFORMABLE_LOADERS
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.utils import read_pcd, read_ego_mask, get_reversed_mapping, read_frame_pickle, PolarDict
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

    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> Transformable:
        """Load Transformable data

        Parameters
        ----------
        name : str
            name of the transformable (best to be unique)
        frame_info : PolarDict
            >>> {
            >>>     '20230823_110018/1742024381864': '20230823_110018/frame_info_pkl/1742024381864.pkl',
            >>>     '20230823_110018/1742024381964': '20230823_110018/frame_info_pkl/1742024381964.pkl',
            >>>     ...
            >>> }
        frame_data : Dict[str, Dict]
            >>> {
            >>>     scene_info: {
            >>>         camera_mask: {
            >>>             camera1: 'ego_mask/camera1.png',
            >>>             camera2: 'ego_mask/camera2.png',
            >>>             ...
            >>>         } ,
            >>>         calibration: {
            >>>             lidar1: [...],
            >>>             camera1: {extrinsic: [...], intrinsic: [...]},
            >>>             ...
            >>>         } , 
            >>>         ...
            >>>     } ,
            >>>     camera_image: {
            >>>         camera1: '20250315_153742_1742024262664_1742024382664/camera/camera1/1742024381864.jpg',
            >>>         camera2: '20250315_153742_1742024262664_1742024382664/camera/camera2/1742024381864.jpg',
            >>>         ...
            >>>     } ,
            >>>     3d_boxes: [
            >>>         {'class': 'class.traffic_facility.hard_barrier',
            >>>         'attr': {'attr.traffic_facility.hard_barrier.type': 'attr.traffic_facility.hard_barrier.type.undefined'},
            >>>         'size': [0.5223, 10.985, 1.1785],
            >>>         'rotation': array([[ 4.27838545e-02,  9.99084344e-01,  1.22456978e-04],
            >>>                 [-9.98961743e-01,  4.27766835e-02,  1.56713547e-02],
            >>>                 [ 1.56517668e-02, -7.92810795e-04,  9.99877189e-01]]),
            >>>         'translation': array([13.53816211,  9.52569891,  0.51868851]),
            >>>         'track_id': '10252_0',
            >>>         'velocity': array([0., 0., 0.])},
            >>>         ... ,
            >>>     ] , 
            >>>     ego_pose: {
            >>>         rotation: [...],
            >>>         translation: [...],
            >>>     } , 
            >>>     ...
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
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> CameraImageSet:
        scene_data = frame_data["scene_info"]
        calib = scene_data["calibration"]
        camera_images = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_data["camera_image"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            if cam_id_ori in frame_data["camera_image"]:
                # need to get the real
                camera_images[cam_id] = CameraImage(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=mmcv.imread(self.data_root / frame_data["camera_image"][cam_id_ori]),
                    ego_mask=read_ego_mask(self.data_root / scene_data["camera_mask"][cam_id_ori]),
                    extrinsic=list(np.array(p) for p in calib[cam_id_ori]["extrinsic"]),
                    intrinsic=np.array(calib[cam_id_ori]["intrinsic"]),
                    tensor_smith=tensor_smith,
                )
        return CameraImageSet(name, camera_images)


@TRANSFORMABLE_LOADERS.register_module()
class CameraTimeImageSetLoader(CameraSetLoader):
    @staticmethod
    def Rt2T(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def T2Rt(T):
        return (T[:3, :3], T[:3, 3])

    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> CameraImageSet:
        scene_data = frame_data["scene_info"]
        calib = scene_data["calibration"]
        camera_images = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_data["camera_image"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            Tec = self.Rt2T(calib[cam_id_ori]["extrinsic"][0], calib[cam_id_ori]["extrinsic"][1])
            Twe0 = self.Rt2T(frame_data['ego_pose']['rotation'], frame_data['ego_pose']['translation'])
            if cam_id_ori in frame_data["camera_image"]:
                Twe1 = frame_data["camera_image"][cam_id_ori]['Twe']
                Te0c = np.linalg.inv(Twe0) @ Twe1 @ Tec
                camera_images[cam_id] = CameraImage(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=mmcv.imread(self.data_root / frame_data["camera_image"][cam_id_ori]['path']),
                    ego_mask=read_ego_mask(self.data_root / scene_data["camera_mask"][cam_id_ori]),
                    extrinsic=self.T2Rt(Te0c),
                    intrinsic=copy.copy(calib[cam_id_ori]["intrinsic"]),
                    tensor_smith=tensor_smith,
                )
        return CameraImageSet(name, camera_images)


@TRANSFORMABLE_LOADERS.register_module()
class NuscenesCameraImageSetLoader(TransformableLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> CameraImageSet:
        scene_data = frame_data["scene_info"]
        camera_images = {
            cam_id: CameraImage(
                name=f"{name}:{cam_id}",
                cam_id=cam_id,
                cam_type=frame_data["camera_image"][cam_id]["calibration"]["camera_type"],
                img=mmcv.imread(self.data_root / frame_data["camera_image"][cam_id]["path"]),
                ego_mask=read_ego_mask(self.data_root / scene_data["camera_mask"][cam_id]),
                extrinsic=list(np.array(p) for p in frame_data["camera_image"][cam_id]["calibration"]["extrinsic"]),
                intrinsic=np.array(frame_data["camera_image"][cam_id]["calibration"]["intrinsic"]),
                tensor_smith=tensor_smith
            )
            for cam_id in frame_data["camera_image"]
        }
        return CameraImageSet(name, camera_images)


@TRANSFORMABLE_LOADERS.register_module()
class CameraDepthSetLoader(CameraSetLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> CameraDepthSet:
        scene_data = frame_data["scene_info"]
        calib = scene_data["calibration"]
        camera_depths = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_data["camera_image_depth"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            if cam_id_ori in frame_data["camera_image_depth"]:
                camera_depths[cam_id] = CameraDepth(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=np.load(self.data_root / frame_data['camera_image_depth'][cam_id_ori])['depth'][..., None].astype(np.float32),
                    ego_mask=read_ego_mask(self.data_root / scene_data["camera_mask"][cam_id_ori]),
                    extrinsic=list(np.array(p) for p in calib[cam_id_ori]["extrinsic"]),
                    intrinsic=np.array(calib[cam_id_ori]["intrinsic"]),
                    depth_mode="d",
                    tensor_smith=tensor_smith,
                )
        return CameraDepthSet(name, camera_depths)


@TRANSFORMABLE_LOADERS.register_module()
class CameraSegMaskSetLoader(CameraSetLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> CameraSegMaskSet:
        scene_data = frame_data["scene_info"]
        calib = scene_data["calibration"]
        camera_segs = {}
        if self.camera_mapping is None:
            self.camera_mapping = {}
            for cam_id in frame_data["camera_image_seg"]:
                self.camera_mapping[cam_id] = cam_id
        for cam_id in self.camera_mapping:
            cam_id_ori = self.camera_mapping[cam_id]
            if cam_id_ori in frame_data["camera_image_seg"]:
                camera_segs[cam_id] = CameraSegMask(
                    name=f"{name}:{cam_id}",
                    cam_id=cam_id,
                    cam_type=calib[cam_id_ori]["camera_type"],
                    img=mmcv.imread(self.data_root / frame_data["camera_image_seg"][cam_id_ori], flag="unchanged"),
                    ego_mask=read_ego_mask(self.data_root / scene_data["camera_mask"][cam_id_ori]),
                    extrinsic=list(np.array(p) for p in calib[cam_id_ori]["extrinsic"]),
                    intrinsic=np.array(calib[cam_id_ori]["intrinsic"]),
                    dictionary=dictionary,
                    tensor_smith=tensor_smith,
                )
        return CameraSegMaskSet(name, camera_segs)


@TRANSFORMABLE_LOADERS.register_module()
class LidarPointsLoader(TransformableLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> LidarPoints:
        points = read_pcd(self.data_root / frame_data["lidar_points"]["lidar1"])
        points = np.pad(points, [[0, 0], [0, 1]], constant_values=0)
        return LidarPoints(name, points[:, :3], points[:, 3:], tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class EgoPoseSetLoader(TransformableLoader):
    def __init__(self, data_root: Path, prev_window_size: int = 0, next_window_size: int = 0):
        """This loader not only load current frame's ego pose, but also load surrounding frames' ego pose within the same group (by default).
        If `self.prev_window_size` or `self.next_window_size` is set, then the ego poses lies in the unioned window will be loaded.

        Args:
            data_root (Path): root directory of the data
            prev_window_size (Optional[int], optional): load previous frames' ego_pose no matter whether these frames are in the same group. Defaults to None.
            next_window_size (Optional[int], optional): load next frames' ego_pose no matter whether these frames are in the same group. Defaults to None.
        """
        super().__init__(data_root)
        self.prev_window_size = prev_window_size
        self.next_window_size = next_window_size

    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> EgoPoseSet:

        def _create_pose(_another_index_info: "IndexInfo", rel_pos: str):
            _another_frame_data = read_frame_pickle(self.data_root / frame_info[_another_index_info.scene_frame_id])
            return EgoPose(
                f"{name}:{rel_pos}:{_another_index_info.frame_id}",
                _another_index_info.frame_id, 
                np.array(_another_frame_data["ego_pose"]["rotation"]),
                _another_frame_data["ego_pose"]["translation"].reshape(3, 1), # it should be a column vector
                tensor_smith=tensor_smith
            )
        
        def _find_num_infos_in_the_group(cur: Union["IndexInfo", None], direction: str) -> int:
            assert direction in ["prev", "next"]
            _attr = "g_" + direction
            if cur is None:
                raise ValueError("cur cannot be None")
            if getattr(cur, _attr) is None:
                return 0
            return 1 + _find_num_infos_in_the_group(getattr(cur, _attr), direction)

        poses = {}

        cnt = 0
        cur = index_info # no need to clone or copy
        num_prev_infos_in_group = _find_num_infos_in_the_group(cur, "prev")
        num_next_infos_in_group = _find_num_infos_in_the_group(cur, "next")
        max_prev = max(num_prev_infos_in_group, self.prev_window_size)
        max_next = max(num_next_infos_in_group, self.next_window_size)

        while (the_prev := cur.g_prev or cur.prev) is not None and cnt < max_prev:
            rel_pos = f"-{cnt+1}" # relative position
            poses[rel_pos] = _create_pose(the_prev, rel_pos)
            cur = the_prev
            cnt += 1

        cur = index_info
        poses["0"] = _create_pose(cur, "0")

        cnt = 0
        while (the_next := cur.g_next or cur.next) is not None and cnt < max_next:
            rel_pos = f"+{cnt+1}" # relative position
            poses[rel_pos] = _create_pose(the_next, rel_pos)
            cur = the_next
            cnt += 1

        sorted_poses = dict(sorted(poses.items(), key=lambda x: int(x[0])))

        return EgoPoseSet(name, transformables=sorted_poses)


@TRANSFORMABLE_LOADERS.register_module()
class Bbox3DLoader(TransformableLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> Bbox3D:
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
        elements = []
        for bx in frame_data["3d_boxes"]:
            ele = {
                "class": bx["class"],
                "attr": list(bx["attr"].values()) if isinstance(bx["attr"], dict) else copy.copy(bx["attr"]),
                "size": copy.copy(bx["size"]),
                "rotation": np.array(bx["rotation"]),
                "track_id": bx["track_id"],
            }
            # ensure translation and velocity to be a column array
            if "translation" in bx:
                ele["translation"] = np.array(bx["translation"]).reshape(3, 1)
            if "velocity" in bx:
                ele["velocity"] = np.array(bx["velocity"]).reshape(3, 1)
            elements.append(ele)
            
        return Bbox3D(name, elements, copy.deepcopy(dictionary), tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class AdvancedBbox3DLoader(TransformableLoader):

    rot90deg = Rotation.from_euler("XYZ", [0, 0, 90], degrees=True).as_matrix()
    
    def __init__(self, data_root: Path, class_mapping: Optional[Dict] = None, attr_mapping: Optional[Dict] = None, axis_rearrange_method="none") -> None:
        """ Advanced Bbox3D Loader
        # CAUTION
        FIXME:
        only a minimum check has been applied to class_mapping and attr_mapping
        there's still some configurations could lead to strange class mapping behavior,
        such as: "new_cls1": ["c1::attr1.True", "c1::attr2.True"]

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

    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> Bbox3D:
        updated_dictionary = self._update_dictionary(dictionary)

        elements = []
        for bx in frame_data["3d_boxes"]:
            ele = {
                "class": self.class_mapping.get_mapped_class(bx["class"], bx["attr"]),
                "attr": self.attr_mapping.get_mapped_attr(bx["attr"]),
                "size": copy.copy(bx["size"]),
                "rotation": np.array(bx["rotation"]),
                "track_id": bx["track_id"],
            }
            
            # ensure translation and velocity to be a column array
            if "translation" in bx:
                ele["translation"] = np.array(bx["translation"]).reshape(3, 1)
            if "velocity" in bx:
                ele["velocity"] = np.array(bx["velocity"]).reshape(3, 1)
            
            # rearrange axis if needed
            if self.axis_rearrange_method != "none" and ele["class"] in updated_dictionary["classes"]:
                self.rearrange_axis_(ele)
            elements.append(ele)

        return Bbox3D(name, elements, updated_dictionary, tensor_smith=tensor_smith)
    
    def _update_dictionary(self, dictionary: Optional[Dict] = None) -> Dict:
        if not dictionary:
            dictionary = {"classes": [], "attrs": []}
        else:
            dictionary = {"classes": copy.copy(dictionary.get("classes", [])), "attrs": copy.copy(dictionary.get("attrs", []))}

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
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> Polyline3D:
        elements = []
        for pl in frame_data["3d_polylines"]:
            ele = {
                "class": pl["class"],
                "attr": list(pl["attr"].values()) if isinstance(pl["attr"], dict) else copy.copy(pl["attr"]),
                "points": np.array(pl["points"]),
            }
            if "track_id" in pl:
                ele["track_id"] = pl["track_id"]
            elements.append(ele)
        return Polyline3D(name, elements, copy.deepcopy(dictionary), tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class Polygon3DLoader(TransformableLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> Polygon3D:
        elements = []
        for pg in frame_data["3d_polylines"]:
            ele = {
                "class": pg["class"],
                "attr": list(pg["attr"].values()) if isinstance(pg["attr"], dict) else copy.copy(pg["attr"]),
                "points": np.array(pg["points"]),
            }
            if "track_id" in pg:
                ele["track_id"] = pg["track_id"]
            elements.append(ele)
        return Polygon3D(name, elements, copy.deepcopy(dictionary), tensor_smith=tensor_smith)


@TRANSFORMABLE_LOADERS.register_module()
class ParkingSlot3DLoader(TransformableLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> ParkingSlot3D:
        elements = []
        for slot in frame_data["3d_polylines"]:
            ele = {
                "class": slot["class"],
                "attr": list(slot["attr"].values()) if isinstance(slot["attr"], dict) else copy.copy(slot["attr"]),
                "points": np.array(slot["points"]),
            }
            if "track_id" in slot:
                ele["track_id"] = slot["track_id"]
            elements.append(ele)
        return ParkingSlot3D(name, elements, copy.deepcopy(dictionary), tensor_smith=tensor_smith)


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

    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> OccSdfBev:
        scene_data = frame_data["scene_info"]

        # get sdf
        if self.load_sdf:
            sdf = cv2.imread(str(self.data_root / frame_data["occ_sdf"]["sdf"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 860 - 36
        else:
            sdf = None
        # get occ
        # occ_path = str(self.data_root / frame["occ_sdf"]["occ_2d"])
        occ_path = str(self.data_root / frame_data["occ_sdf"]["occ_map_sdf"])
        occ = mmcv.imread(occ_path)
        # get height
        mask_ground = np.float32(occ[..., 1] > 128)

        # height_ground_path = str(self.data_root / frame["occ_sdf"]["ground"])
        height_ground_path = str(self.data_root / frame_data["occ_sdf"]["ground_height_map"])
        height_ground = mmcv.imread(height_ground_path, 'unchanged').astype(np.float32) / 3000 - 10

        height_lidar_path = str(self.data_root / frame_data["occ_sdf"]["bev_height_map"])
        mask_lidar_path = str(self.data_root / frame_data["occ_sdf"]["bev_lidar_mask"])
        height_lidar = mmcv.imread(height_lidar_path, 'unchanged').astype(np.float32) * 3.0 / 255.0 - 1
        mask_lidar = np.float32(mmcv.imread(mask_lidar_path, 'unchanged') > 128)

        height, mask = self._gen_height_and_mask(height_lidar, mask_lidar, height_ground, mask_ground)

        if self.src_voxel_range is None:
            src_voxel_range = scene_data["space_range"]["occ"]
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
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> SegBev:
        raise NotImplementedError


@TRANSFORMABLE_LOADERS.register_module()
class OccSdf3DLoader(TransformableLoader):
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, dictionary: Optional[Dict] = None, **kwargs) -> OccSdf3D:
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
        self._reversed_mapping = None
        self.validate_input()

    @property
    def reversed_mapping(self):
        if self._reversed_mapping is None:
            self._reversed_mapping = get_reversed_mapping(self)
        return self._reversed_mapping

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
        src_attr_used_cnt = Counter([k for k in self.reversed_mapping])
        for k, v in src_attr_used_cnt.items():
            if v > 1:
                raise ValueError(f"Source attr {k} is used more than once in the mapping.")

    def get_mapped_attr(self, ele_attrs: Dict[str, str] = None) -> List[str]:
        """Get mapped attr name from attrs"""
        if not self: # empty mapping provided in the cfgs
            return []
        return sorted({self.reversed_mapping[attr] for attr in ele_attrs.values() if attr in self.reversed_mapping})



@TRANSFORMABLE_LOADERS.register_module()
class VariableLoader(TransformableLoader):
    def __init__(self, data_root: Path, variable_key: str) -> None:
        super().__init__(data_root)
        self.data_root = data_root
        self.variable_key = variable_key
    def load(self, name: str, frame_info: PolarDict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: Optional[TensorSmith] = None, **kwargs) -> Transformable:
        variable_value = copy.deepcopy(frame_data[self.variable_key])
        return Variable(name, variable_value, tensor_smith=tensor_smith)
