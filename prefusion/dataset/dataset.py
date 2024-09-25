# Written by AlphaLFC. All rights reserved.

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Sequence
from collections import defaultdict

import mmcv
import mmengine
import torch
import numpy as np
from torch.utils.data import Dataset
from prefusion.registry import DATASETS, MMENGINE_FUNCTIONS

from .transform import (
    CameraImage,
    CameraImageSet,
    LidarPoints,
    CameraSegMask,
    CameraSegMaskSet,
    CameraDepth,
    CameraDepthSet,
    Bbox3D,
    Polyline3D,
    Polygon3D,
    ParkingSlot3D,
    Pose,
    PoseSet,
    Trajectory,
    SegBev,
    OccSdfBev,
    OccSdf3D,
)

from .utils import build_transforms, build_tensor_smiths, build_model_feeder, read_pcd

if TYPE_CHECKING:
    from .tensor_smith import TensorSmith
    from .transform import Transform
    from .model_feeder import BaseModelFeeder

__all__ = ["GroupBatchDataset"]

def get_frame_index(sequence, timestamp):
    for t in sequence:
        if timestamp <= t:
            return t


@MMENGINE_FUNCTIONS.register_module()
def collate_dict(batch):
    return batch[0]


GroupBatch = List[List[Dict]]


def generate_groups(
    total_num_frames: int, 
    group_size: int, 
    frame_interval: int, 
    start_ind: int = 0, 
    random_start_ind: bool = False, 
    pad_mode: str = 'both',
    seed: int = None,
) -> List[Tuple[int]]:
    """
    Generate groups of frames from a single scene.

    Parameters
    ----------
    total_num_frames : int
        The total number of frames in the scene.
    group_size : int
        The number of frames in each group.
    frame_interval : int
        The interval between frames in each group.
    start_ind : int, optional
        The starting index of the first group. Default is 0.
    random_start_ind : bool, optional
        If True, randomly select the starting index of the first group. Default is False.
    pad_mode : str, optional, choices=['prev', 'post', 'both'], default='prev'
        The padding mode for the last group. Default is 'prev'.

    Returns
    -------
    groups : list of tuples
        A list of tuples, where each tuple represents a group of frames.

    Notes
    -----
    - `group_interval = group_size * frame_interval`
    - `group_interval <= total_num_frames` 
    - `start_ind` should be assigned between `[0, group_interval - 1]`
    - When the `start_ind > 0`, we should insert a group including `[0, start_ind)`
    - If the tail of the group, aka `end_ind`, is bigger than `total_num_frames - 1`, we should append a group including `(end_ind, total_num_frames]`
    """
    if seed:
        random.seed(seed)

    total_frame_inds = np.arange(total_num_frames)
    # fill up total_frame_inds if group_interval > group_size
    group_interval = group_size * frame_interval
    if group_interval > total_num_frames:
        out_of_bound = group_interval - total_num_frames
        if pad_mode in ['prev']:
            total_frame_inds = np.insert(total_frame_inds, 0, [0, ] * out_of_bound)
        elif pad_mode in ['post']:
            total_frame_inds = np.append(total_frame_inds, [total_num_frames - 1, ] * out_of_bound)
        elif pad_mode in ['both']:
            out_of_bound_prev = out_of_bound // 2
            out_of_bound_post = out_of_bound - out_of_bound_prev
            total_frame_inds = np.insert(total_frame_inds, 0, [0, ] * out_of_bound_prev)
            total_frame_inds = np.append(total_frame_inds, [total_num_frames - 1, ] * out_of_bound_post)

    total_num_frames_padded = len(total_frame_inds)

    if random_start_ind:
        start_ind = random.randint(0, group_interval - 1)
    assert start_ind >= 0 and start_ind < group_interval
    # get splits
    splits = total_frame_inds[start_ind::group_interval]
    # insert a start_ind < 0
    if splits[0] > 0:
        splits = np.insert(splits, 0, splits[0] - group_interval)
    # append a tail, end_ind
    splits = np.append(splits, splits[-1] + group_interval)

    ind_lists = []
    for start, end in zip(splits[:-1], splits[1:]):
        if start < 0:
            start = 0
            end = group_interval
        if end >= total_num_frames_padded:
            end = total_num_frames_padded
            start = total_num_frames_padded - group_interval
        ind_list = total_frame_inds[start:end].reshape(group_size, frame_interval).T
        ind_lists.extend(ind_list.tolist())
    # sometimes the ind_list may be duplicated, so add a unique operation
    return np.unique(ind_lists, axis=0)


class IndexInfo:
    def __init__(self, scene_id: str, frame_id: str, prev: "IndexInfo" = None, next: "IndexInfo" = None):
        self.scene_id = scene_id
        self.frame_id = frame_id
        self.prev = prev
        self.next = next
        if prev:
            prev.next = self
        if next:
            next.prev = self

    def __repr__(self) -> str:
        return f"{self.scene_id}/{self.frame_id} (prev: {self.prev.scene_frame_id if self.prev else None}, next: {self.next.scene_frame_id if self.next else None})"

    def __eq__(self, other: "IndexInfo") -> bool:
        if other is None:
            return False
        if self.scene_frame_id != other.scene_frame_id:
            return False
        if self.prev is None and other.prev is not None:
            return False
        if self.next is None and other.next is not None:
            return False
        if self.prev is not None and other.prev is not None and self.prev.scene_frame_id != other.prev.scene_frame_id:
            return False
        if self.next is not None and other.next is not None and self.next.scene_frame_id != other.next.scene_frame_id:
            return False
        return True

    @property
    def scene_frame_id(self) -> str:
        return f"{self.scene_id}/{self.frame_id}"

    def as_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "prev": {"scene_id": self.prev.scene_id, "frame_id": self.prev.frame_id} if self.prev else None,
            "next": {"scene_id": self.next.scene_id, "frame_id": self.next.frame_id} if self.next else None,
        }

    @classmethod
    def from_str(self, index_str: str, prev: "IndexInfo" = None, next: "IndexInfo" = None, sep: str = "/"):
        scene_id, frame_id = index_str.split(sep)
        return IndexInfo(scene_id, frame_id, prev=prev, next=next)


class GroupSampler:
    def __init__(self, scene_frame_inds: Dict[str, List[str]], possible_group_sizes: Union[int, Tuple[int]], possible_frame_intervals: Union[int, Tuple[int]] = 1, seed: int = None):
        """Sample groups

        Parameters
        ----------
        scene_frame_inds : Dict[str, List[str]]
            e.g.  
            ```
            {
              "20231101_160337": [ "20231101_160337/1698825817664", "20231101_160337/1698825817764"],
              "20230823_110018": [ "20230823_110018/1692759640764", "20230823_110018/1692759640864"],
            }
            ```
        possible_group_size : Union[int, Tuple[int]]
            if int, will always use this value as group_size;
            if Tuple[int], during train phase, will random pick a value as the group_size for a given epoch.
        possible_frame_interval : Union[int, Tuple[int]], optional
            if int, will always use this value as frame_interval;
            if Tuple[int], during train phase, will random pick a value as the frame_interval for a given epoch.
        seed : int, optional
            Random seed for randomization operations. It's usually for testing and debugging purpose.
        """
        self.scene_frame_inds = scene_frame_inds
        self.possible_group_sizes = [possible_group_sizes] if isinstance(possible_group_sizes, int) else possible_group_sizes
        self.possible_frame_intervals = [possible_frame_intervals] if isinstance(possible_frame_intervals, int) else possible_frame_intervals
        self._cur_train_group_size = self.possible_group_sizes[0]  # train group size of current epoch
        self.seed = seed

    @property
    def group_size(self) -> int:
        return self._cur_train_group_size

    def sample(self, phase: str, output_str_index: bool = False) -> List[List[IndexInfo]]:
        assert phase.lower() in ["train", "val", "test", "test_scene_by_scene"]
        match phase:
            case "train":
                groups = self.sample_train_groups()
            case "val" | "test":
                groups = self.sample_val_groups()
            case "test_scene_by_scene" :
                groups = self.sample_scene_groups()
        if not output_str_index:
            return self._convert_groups_to_info(groups)

    @staticmethod
    def _convert_groups_to_info(groups: List[List[str]]) -> List[List[IndexInfo]]:
        index_info_groups = []
        for grp in groups:
            index_info_grp = []
            prev = None
            for i, frm in enumerate(grp):
                cur = IndexInfo.from_str(frm, prev=prev)
                prev = cur
                index_info_grp.append(cur)
            index_info_groups.append(index_info_grp)
        return index_info_groups

    def sample_train_groups(self) -> List[List[str]]:
        if self.seed: random.seed(self.seed)
        self._cur_train_group_size = random.choice(self.possible_group_sizes)
        return self._generate_groups(self._cur_train_group_size, random_start_ind=True, shuffle=True, seed=self.seed)

    def sample_val_groups(self) -> List[List[str]]:
        return self._generate_groups(self.possible_group_sizes[0], frame_interval=self.possible_frame_intervals[0], start_ind=0, random_start_ind=False, shuffle=False)

    def sample_scene_groups(self) -> List[List[str]]:
        return list(self.scene_frame_inds.values())

    def sample_groups_by_class_balance(self):
        raise NotImplementedError

    def sample_groups_by_meta_info(self):
        raise NotImplementedError
    
    def _generate_groups(
        self, 
        group_size: int, 
        frame_interval: int = None, 
        start_ind: int = 0, 
        random_start_ind: bool = False, 
        shuffle: bool = False, 
        seed: int = None
    ) -> List[List[str]]:
        all_groups = []
        for _, frame_ids in self.scene_frame_inds.items():
            if not frame_interval:
                if self.seed: random.seed(self.seed)
                frame_interval = random.choice(self.possible_frame_intervals)
            inds_list = generate_groups(len(frame_ids), group_size, frame_interval, start_ind=start_ind, random_start_ind=random_start_ind, seed=seed)
            groups = [[frame_ids[i] for i in inds] for inds in inds_list]
            all_groups.extend(groups)
        if shuffle:
            if self.seed: random.seed(self.seed)
            random.shuffle(all_groups)
        return all_groups


@DATASETS.register_module()
class GroupBatchDataset(Dataset):
    """
    A novel dataset class for batching sequence groups for multi-module data.
    """

    # TODO: implement visualization?
    AVAILABLE_TRANSFORMABLE_KEYS = (
        "camera_images",
        "camera_segs",
        "camera_depths",
        "lidar_points",
        "bbox_3d",
        "bbox_bev",
        "square_3d",
        "cylinder_3d",
        "oriented_cylinder_3d",
        "polyline_3d",
        "polygon_3d",
        "parkingslot_3d",
        "ego_poses",
        "trajectory",
        "seg_bev",
        "occ_sdf_bev",
        "occ_sdf_3d",
    )

    def __init__(
        self,
        name,
        data_root: Union[str, Path],
        info_path: Union[str, Path],
        transformable_keys: List[str],
        dictionaries: Dict[ str, Dict ],
        tensor_smiths: Dict[str, Union[dict, "TensorSmith"]] = None,
        transforms: List[Union[dict, "Transform"]] = None,
        model_feeder: Union["BaseModelFeeder", dict] = None,
        phase: str = "train",
        indices_path: Union[str, Path] = None,
        batch_size: int = 1,
        drop_last: bool = False,
        possible_group_sizes: Union[int, Tuple[int]] = 3,
        possible_frame_intervals: Union[int, Tuple[int]] = 1,  # TODO: redefine it by time, like seconds
        group_backtime_prob: float = 0.0,
        seed_dataset: int = None,
    ):
        """
        Initializes the dataset instance.

        Args:
        - name (str): Name of the dataset.
        - data_root (str): Root directory of the dataset.
        - info_path (str): Path to the information file.
        - dictionary (dict): Dictionary for each transformable.
            - dictionary = {
                'bbox_3d': {
                    'branch_0': {
                        'classes': [<>, <>, ...],
                        'attrs:': [<>, <>, ...]
                    },
                    'branch_1': {...}
                },
                'bbox_bev': {...},
                'polyline_3d': {...},
            }
        - transformable_keys (list): List of transformable keys. Keys should be in dictionary.
        - transforms (list): Transform classes for preprocessing transformables. Build by TRANSFORMS.
        - phase (str): Specifies the phase ('train', 'val', 'test' or 'test_scene_by_scene') of the dataset; default is 'train'.
        - indices_path (str, optional): Specified file of indices to load; if None, all frames are automatically fetched from the info_path.
        - batch_size (int, optional): Batch size; defualt is 1.
        - possible_group_sizes (int or list, optional): Size of sequence group; can be a single integer or a list (e.g., [5, 10]); default is 3.
        - possible_frame_intervals (int or list, optional): Interval between frames; default is 1.
        - group_backtime_prob (float): Probability of grouping backtime frames.
        - seed_dataset (int): Random seed for dataset

        Notes:
        - `scene`: a scene contains a sequence of frames
        - `group`: a subset of a scene, when testing, a scene can be one group
        - `frame`: a time instance of a scene
        - one index should get a batch of groups, one group should be a sequential list
        - the length of the dataset should be len(self.groups) // self.batch_size
        - a frame data is a dictionary containing the transformed data.
        """
        super().__init__()
        self.name = name
        self.data_root = Path(data_root)
        assert phase.lower() in ["train", "val", "test", "test_scene_by_scene"]
        self.phase = phase.lower()
        self.info = mmengine.load(str(info_path))
        self._assert_availability(list(dictionaries.keys()))
        self._assert_availability(transformable_keys)
        self.dictionaries = dictionaries
        self.transformable_keys = transformable_keys
        self.tensor_smiths = build_tensor_smiths(tensor_smiths)
        self.transforms = build_transforms(transforms)
        self.model_feeder = build_model_feeder(model_feeder)

        if indices_path is not None:
            indices = [line.strip() for line in open(indices_path, "w")]
            self.scene_frame_inds = self._prepare_scene_frame_inds(self.info, indices=indices)
        else:
            self.scene_frame_inds = self._prepare_scene_frame_inds(self.info)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.group_backtime_prob = group_backtime_prob

        self.seed_dataset = seed_dataset

        self.group_sampler = GroupSampler(self.scene_frame_inds, possible_group_sizes, possible_frame_intervals, seed=seed_dataset)
        self.groups = self.group_sampler.sample(self.phase, output_str_index=False)

    @classmethod
    def _assert_availability(cls, keys: List[str]) -> None:
        for key in keys:
            assert (
                key in cls.AVAILABLE_TRANSFORMABLE_KEYS
            ), f"{key} is not a valid transformable key from {cls.AVAILABLE_TRANSFORMABLE_KEYS}"

    @staticmethod
    def _prepare_scene_frame_inds(info: Dict, indices: List[str] = None) -> Dict[str, List[str]]:
        if indices is None:
            indices = {}
            for scene_id in info:
                frame_list = sorted(info[scene_id]["frame_info"].keys())
                if frame_list:
                    indices[scene_id] = [f"{scene_id}/{frame_id}" for frame_id in frame_list]
            return indices

        available_indices = defaultdict(list)
        for index in indices:
            scene_id, frame_id = index.split("/")
            if scene_id in info:
                if frame_id in info[scene_id]["frame_info"]:
                    available_indices[scene_id].append(index)
        for scene_id in available_indices:
            available_indices[scene_id] = sorted(available_indices[scene_id])

        return available_indices
    
    @property
    def group_size(self):
        return self.group_sampler.group_size

    def __repr__(self):
        return "".join(
            [
                f"An instance of {self.__class__}: (\n",
                f"    name={self.name}, \n",
                f"    num_groups={len(self.groups)}, \n"
                f"    phase={self.phase}, \n"
                f"    batch_size={self.batch_size}, \n"
                f"    group_size={self.group_sampler.possible_group_sizes}, \n"
                f"    frame_interval={self.group_sampler.possible_frame_intervals}, \n"
                f"    group_backtime_prob={self.group_backtime_prob}\n)",
            ]
        )

    def load_all_transformables(self, index_info: IndexInfo) -> dict:
        all_transformables = {}
        for key in self.transformable_keys:
            all_transformables[key] = eval(f"self.load_{key}")(key, index_info)
        return all_transformables

    def __len__(self):
        if self.drop_last:
            return len(self.groups) // self.batch_size
        else:
            return int(np.ceil(len(self.groups) / self.batch_size))
    

    @staticmethod
    def _batch_groups(group_batch_ind, groups, batch_size):
        batched_groups = []
        for batch_idx in range(batch_size):
            group_idx = group_batch_ind * batch_size + batch_idx
            if group_idx >= len(groups):
                group_idx = max(0, 2 * (len(groups) - 1) - group_idx)
            batched_groups.append(groups[group_idx])
        return batched_groups


    def __getitem__(self, idx) -> GroupBatch:

        if idx >= len(self):
            self.groups = self.group_sampler.sample(self.phase, output_str_index=False)
            raise IndexError

        batched_groups = self._batch_groups(idx, self.groups, self.batch_size)

        batch = []
        for group in batched_groups:
            group_of_inputs = []
            for i, index_info in enumerate(group):
                input_dict = {
                    "index_info": index_info,
                    "transformables": self.load_all_transformables(index_info),
                }  # TODO: add ego pose transformation
                group_of_inputs.append(input_dict)
            batch.append(group_of_inputs)

        # apply transforms
        batch_seed = int.from_bytes(os.urandom(2), byteorder="big")
        for group_of_inputs in batch:
            group_seed = int.from_bytes(os.urandom(2), byteorder="big")
            for input_dict in group_of_inputs:
                frame_seed = int.from_bytes(os.urandom(2), byteorder="big")
                transformables = []
                for key in input_dict["transformables"]:
                    transformable = input_dict["transformables"][key]
                    if isinstance(transformable, dict):
                        transformables.extend(transformable.values())
                    else:
                        transformables.append(transformable)
                for transform in self.transforms:
                    transform(*transformables, seeds={"group": group_seed, "batch": batch_seed, "frame": frame_seed})

        group_batch = []
        for frame_batch in zip(*batch):
            group_batch.append(frame_batch)

        model_food = self.model_feeder(group_batch)

        return model_food

    def load_camera_images(self, transformable_key: str, index_info: IndexInfo) -> CameraImageSet:
        scene_info = self.info[index_info.scene_id]["scene_info"]
        frame_info = self.info[index_info.scene_id]["frame_info"][index_info.frame_id]
        calib = self.info[index_info.scene_id]["scene_info"]["calibration"]
        camera_images = {
            cam_id: CameraImage(
                cam_id=cam_id,
                cam_type=calib[cam_id]["camera_type"],
                img=mmcv.imread(self.data_root / frame_info["camera_image"][cam_id]),
                ego_mask=mmcv.imread(self.data_root / scene_info["camera_mask"][cam_id], flag="grayscale"),
                extrinsic=calib[cam_id]["extrinsic"],
                intrinsic=calib[cam_id]["intrinsic"],
                tensor_smith=self.tensor_smiths.get(transformable_key),
            )
            for cam_id in frame_info["camera_image"]
        }
        return CameraImageSet(camera_images)

    def load_lidar_points(self, transformable_key: str, index_info: IndexInfo):
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        points = read_pcd(self.data_root / frame["lidar_points"]["lidar1"])
        return LidarPoints(points[:, :3], points[:, 3], self.tensor_smiths[transformable_key])

    def load_camera_segs(self, transformable_key: str, index_info: IndexInfo) -> CameraSegMaskSet:
        scene_info = self.info[index_info.scene_id]["scene_info"]
        frame_info = self.info[index_info.scene_id]["frame_info"][index_info.frame_id]
        calib = self.info[index_info.scene_id]["scene_info"]["calibration"]

        camera_segs = {
            cam_id: CameraSegMask(
                cam_id=cam_id,
                cam_type=calib[cam_id]["camera_type"],
                img=mmcv.imread(self.data_root / frame_info["camera_image_seg"][cam_id], flag="unchanged"),
                ego_mask=mmcv.imread(self.data_root / scene_info["camera_mask"][cam_id], flag="grayscale"),
                extrinsic=calib[cam_id]["extrinsic"],
                intrinsic=calib[cam_id]["intrinsic"],
                dictionary=self.dictionaries.get(transformable_key),
                tensor_smith=self.tensor_smiths.get(transformable_key),
            )
            for cam_id in frame_info["camera_image_seg"]
        }
        return CameraSegMaskSet(camera_segs)

    def load_camera_depths(self, transformable_key: str, index_info: IndexInfo) -> CameraDepthSet:
        scene_info = self.info[index_info.scene_id]["scene_info"]
        frame_info = self.info[index_info.scene_id]["frame_info"][index_info.frame_id]
        calib = self.info[index_info.scene_id]["scene_info"]["calibration"]

        camera_depths = {
            cam_id: CameraDepth(
                cam_id=cam_id,
                cam_type=calib[cam_id]["camera_type"],
                img=np.load(self.data_root / frame_info['camera_image_depth'][cam_id])['depth'][..., None].astype(np.float32),
                ego_mask=mmcv.imread(self.data_root / scene_info["camera_mask"][cam_id], flag="grayscale"),
                extrinsic=calib[cam_id]["extrinsic"],
                intrinsic=calib[cam_id]["intrinsic"],
                depth_mode="d",
                tensor_smith=self.tensor_smiths.get(transformable_key),
            )
            for cam_id in frame_info["camera_image_depth"]
        }
        return CameraDepthSet(camera_depths)

    def load_bbox_3d(self, transformable_key: str, index_info: IndexInfo) -> Bbox3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_boxes"]
        return Bbox3D(elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key))

    def load_bbox_bev(self, transformable_key: str, index_info: IndexInfo) -> Bbox3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_boxes"]
        return Bbox3D(elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key))

    def load_square_3d(self, transformable_key: str, index_info: IndexInfo) -> Bbox3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_boxes"]
        return Bbox3D(elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key))

    def load_cylinder_3d(self, transformable_key: str, index_info: IndexInfo) -> Bbox3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_boxes"]
        return Bbox3D(elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key))

    def load_oriented_cylinder_3d(self, transformable_key: str, index_info: IndexInfo) -> Bbox3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_boxes"]
        return Bbox3D(elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key))

    def load_polyline_3d(self, transformable_key: str, index_info: IndexInfo) -> Polyline3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_polylines"]
        return Polyline3D(
            elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key)
        )

    def load_polygon_3d(self, transformable_key: str, index_info: IndexInfo) -> Polygon3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_polylines"]
        return Polygon3D(
            elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key)
        )

    def load_parkingslot_3d(self, transformable_key: str, index_info: IndexInfo) -> ParkingSlot3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        elements = frame["3d_polylines"]
        return ParkingSlot3D(
            elements, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key)
        )

    def load_ego_poses(self, transformable_key: str, index_info: IndexInfo) -> PoseSet:
        scene = self.info[index_info.scene_id]['frame_info']

        def _create_pose(frame_id):
            return Pose(
                frame_id, 
                scene[frame_id]["ego_pose"][0], 
                scene[frame_id]["ego_pose"][1], 
                tensor_smith=self.tensor_smiths.get(transformable_key)
            )

        poses = {}

        cnt = 0
        cur = index_info
        while cur.prev is not None:
            poses[f'-{cnt+1}'] = _create_pose(cur.prev.frame_id)
            cur = cur.prev
            cnt += 1

        cur = index_info
        poses['0'] = _create_pose(cur.frame_id)

        cnt = 0
        while cur.next is not None:
            poses[f'+{cnt+1}'] = _create_pose(cur.next.frame_id)
            cur = cur.next
            cnt += 1

        sorted_poses = dict(sorted(poses.items(), key=lambda x: int(x[0])))

        return PoseSet(transformables=sorted_poses)


    def load_trajectory(self, transformable_key: str, index_info: IndexInfo, time_window=2) -> Trajectory:
        cur_frame_id = index_info.frame_id
        scene = self.info[index_info.scene_id]
        frame_list = list(scene["frame_info"].keys())
        cur_ind = frame_list.index(cur_frame_id)
        end_time = int(cur_frame_id) * scene["meta_info"]["time_unit"] + time_window
        end_timestamp = str(int(end_time / scene["meta_info"]["time_unit"]))
        end_frame_id = get_frame_index(frame_list, end_timestamp)
        end_ind = frame_list.index(end_frame_id) + 1
        frame_ids = frame_list[cur_ind:end_ind]

        ego_trajectory = []
        for frame_id in frame_ids:
            # ego_trajectory
            R = scene[frame_id]["ego_pose"]["rotation"]
            t = scene[frame_id]["ego_pose"]["translation"]
            ego_trajectory.append((R, t))

        # TODO: other object trajectories

        trajectories = [ego_trajectory]
        return Trajectory(
            trajectories, self.dictionaries.get(transformable_key), tensor_smith=self.tensor_smiths.get(transformable_key)
        )

    def load_seg_bev(self, transformable_key: str, index_info: IndexInfo) -> SegBev:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        raise NotImplementedError

    def load_occ_sdf_bev(self, transformable_key: str, index_info: IndexInfo) -> OccSdfBev:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        occ_path = frame["occ_sdf"]["occ_bev"]
        sdf_path = frame["occ_sdf"]["sdf_bev"]
        height_path = frame["occ_sdf"]["height_bev"]
        return OccSdfBev(
            src_view_range=scene["meta_info"]["space_range"]["occ"],  # ego system,
            occ=mmcv.imread(occ_path),
            sdf=mmcv.imread(sdf_path),
            height=mmcv.imread(height_path),
            dictionary=self.dictionaries.get(transformable_key),
            mask=None,
            tensor_smith=self.tensor_smiths.get(transformable_key),
        )

    def load_occ_sdf_3d(self, transformable_key: str, index_info: IndexInfo) -> OccSdf3D:
        scene = self.info[index_info.scene_id]
        frame = scene["frame_info"][index_info.frame_id]
        file_path = frame["occ_sdf"]["occ_sdf_3d"]
        raise NotImplementedError
