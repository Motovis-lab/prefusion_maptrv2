# Written by AlphaLFC. All rights reserved.

import os
import copy
import random
import warnings
import logging
import functools
from pathlib import Path
from typing import List, Tuple, Dict, Union, TYPE_CHECKING, Callable
from collections import defaultdict

import mmengine
import numpy as np
from mmengine.logging import print_log
from torch.utils.data import Dataset
from prefusion.registry import DATASETS, MMENGINE_FUNCTIONS, TENSOR_SMITHS, DATASET_TOOLS
from prefusion.dataset.transformable_loader import (
    TransformableLoader,
    CameraImageSetLoader,
    CameraDepthSetLoader,
    CameraSegMaskSetLoader,
    LidarPointsLoader,
    EgoPoseSetLoader,
    Bbox3DLoader,
    Polyline3DLoader,
    Polygon3DLoader,
    ParkingSlot3DLoader,
    SegBevLoader,
    OccSdfBevLoader,
    OccSdf3DLoader,
)
from prefusion.dataset.transform import (
    CameraImageSet,
    CameraDepthSet,
    CameraSegMaskSet,
    LidarPoints,
    EgoPoseSet,
    Bbox3D,
    Polyline3D,
    Polygon3D,
    ParkingSlot3D,
    OccSdfBev,
    OccSdf3D,
    SegBev,
)

from .utils import build_transforms, build_model_feeder, build_tensor_smith, build_transformable_loader, divide

if TYPE_CHECKING:
    from .tensor_smith import TensorSmith
    from .transform import Transform
    from .model_feeder import BaseModelFeeder
    from .transform import Transformable

__all__ = [
    "IndexInfo",
    "GroupBatchDataset"
]


def get_frame_index(sequence, timestamp):
    for t in sequence:
        if timestamp <= t:
            return t


@MMENGINE_FUNCTIONS.register_module()
def collate_dict(batch):
    return batch[0]


GroupBatch = List[List[Dict]]

_print_log = functools.partial(print_log, logger='current')


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


class EndOfAllSubEpochs(Exception):
    pass


@DATASET_TOOLS.register_module()
class SubEpochManager:
    def __init__(
        self,
        batch_size: int,
        num_group_batches_per_subepoch: int,
        drop_last_group_batch: bool = False,
        drop_last_subepoch: bool = False,
        verbose: bool = True,
    ):
        """SubEpochManager stores and manages the state of sub epochs. This feature is mainly used in the case
        that one wants to train an original epoch as a series of sub epochs.

        Parameters
        ----------
        batch_size : int
            batch_size
        num_group_batches_per_subepoch : int
            number of groups in one sub epoch
        drop_last_group_batch : bool, optional
            whether to drop the last group batch when splitting subepochs, by default False
        drop_last_subepoch : bool, optional
            whether to drop the last subepoch if the subepoch's length doesn't meet `num_group_batches_per_subepoch`, by default False
        """
        self.batch_size = batch_size
        self.num_group_batches_per_subepoch = num_group_batches_per_subepoch
        self.drop_last_group_batch = drop_last_group_batch
        self.drop_last_subepoch = drop_last_subepoch
        self.verbose = verbose

    def get_num_group_batches_in_cur_subepoch(self):
        if self.cur_subepoch_idx < self.num_subepochs - 1:
            return self.num_group_batches_per_subepoch
        residual = self.num_total_group_batches % self.num_group_batches_per_subepoch if self.drop_last_subepoch else 0
        num_used_group_batches = self.num_group_batches_per_subepoch * (self.num_subepochs - 1)
        num_group_batches_in_the_last_subepoch = self.num_total_group_batches - num_used_group_batches - residual
        return num_group_batches_in_the_last_subepoch

    def translate_index(self, idx):
        if idx >= self.get_num_group_batches_in_cur_subepoch():
            raise IndexError
        translated_index = idx + self.cur_subepoch_idx * self.num_group_batches_per_subepoch
        self.visited[translated_index] = True
        return translated_index

    def to_next_sub_epoch(self):
        if self.verbose:
            _print_log(f"cur_subepoch_idx is {self.cur_subepoch_idx}, moving to the next subepoch.")
        self.cur_subepoch_idx += 1
        if self.cur_subepoch_idx == self.num_subepochs: # finished all the subepochs
            if self.verbose:
                _print_log(f"Reached the end of all subepochs.")
            raise EndOfAllSubEpochs

    def reset(self, num_total_groups: int):
        """Resets the state of the SubEpochManager

        Parameters
        ----------
        num_total_groups : int
            total number of groups that generated by GroupSampler.
        """
        self.num_total_groups = num_total_groups
        self.num_total_group_batches = divide(self.num_total_groups, self.batch_size, drop_last=self.drop_last_group_batch)
        assert self.num_group_batches_per_subepoch <= self.num_total_group_batches, "num_group_batches_per_subepoch should be no larger than num_total_group_batches"
        self.num_subepochs = divide(self.num_total_group_batches, self.num_group_batches_per_subepoch, drop_last=self.drop_last_subepoch)
        self.cur_subepoch_idx = 0
        if hasattr(self, "visited"):
            self._check_visited_indices()
        self.visited = np.zeros(self._get_num_group_batches_available_to_visit(), dtype=bool)
        if self.verbose:
            _print_log(f"Reset SubEpochManager state to {self}")

    init: Callable = reset

    def _check_visited_indices(self):
        if not self.visited.all():
            indices_not_visited = np.nonzero(~self.visited)[0]
            warnings.warn(f"Some group batches are not visited! (group_batch_index: {indices_not_visited.tolist()})", UserWarning)

    def _get_num_group_batches_available_to_visit(self):
        tmp_cur_subepoch_idx = self.cur_subepoch_idx
        tmp_verbose = self.verbose
        self.cur_subepoch_idx = 0
        self.verbose = False
        _num_group_batches = 0
        while True:
            _num_group_batches += self.get_num_group_batches_in_cur_subepoch()
            try:
                self.to_next_sub_epoch()
            except EndOfAllSubEpochs:
                break
        self.cur_subepoch_idx = tmp_cur_subepoch_idx
        self.verbose = tmp_verbose
        return _num_group_batches

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_total_group_batches={self.num_total_group_batches} "
            f"num_subepochs={self.num_subepochs} "
            f"cur_subepoch_idx={self.cur_subepoch_idx}"
            f")"
        )


@DATASETS.register_module()
class GroupBatchDataset(Dataset):
    """
    A novel dataset class for batching sequence groups for multi-module data.
    """

    # TODO: implement visualization?
    DEFAULT_LOADERS = {
        CameraImageSet.__name__: CameraImageSetLoader,
        CameraDepthSet.__name__: CameraDepthSetLoader,
        CameraSegMaskSet.__name__: CameraSegMaskSetLoader,
        LidarPoints.__name__: LidarPointsLoader,
        EgoPoseSet.__name__: EgoPoseSetLoader,
        Bbox3D.__name__: Bbox3DLoader,
        Polyline3D.__name__: Polyline3DLoader,
        Polygon3D.__name__: Polygon3DLoader,
        ParkingSlot3D.__name__: ParkingSlot3DLoader,
        OccSdfBev.__name__: OccSdfBevLoader,
        OccSdf3D.__name__: OccSdf3DLoader,
        SegBev.__name__: SegBevLoader,
    }

    def __init__(
        self,
        name,
        data_root: Union[str, Path],
        info_path: Union[str, Path],
        transformables: dict,
        transforms: List[Union[dict, "Transform"]] = None,
        model_feeder: Union["BaseModelFeeder", dict] = None,
        subepoch_manager: Union["SubEpochManager", dict] = None,
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
        - transformables (dict): Dict of transformable definitions.
        - transforms (list): Transform classes for preprocessing transformables. Build by TRANSFORMS.
        - phase (str): Specifies the phase ('train', 'val', 'test' or 'test_scene_by_scene') of the dataset; default is 'train'.
        - indices_path (str, optional): Specified file of indices to load; if None, all frames are automatically fetched from the info_path.
        - batch_size (int, optional): Batch size; defualt is 1.
        - possible_group_sizes (int or list, optional): Size of sequence group; can be a single integer or a list (e.g., [5, 10]); default is 3.
        - possible_frame_intervals (int or list, optional): Interval between frames; default is 1.
        - group_backtime_prob (float): Probability of grouping backtime frames.
        - subepoch_manager (SubEpochManager or dict): Only works for training. If set, only `num_group_batches_per_subepoch` will be trained in a epoch and the 
                                                      rest of data will be spit out in the next epoch. 
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
        self.transformables = transformables
        self.transforms = build_transforms(transforms)
        self.model_feeder = build_model_feeder(model_feeder)
        self.subepoch_manager = DATASET_TOOLS.build(subepoch_manager) if subepoch_manager and self.phase == "train" else None

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

        self._sample_groups()

        if self.subepoch_manager is not None:
            _print_log(
                "Since you have set SubEpochManager, "
                "you should consider set a larger max_epochs for your training.", 
                level=logging.WARNING
            )
            self.subepoch_manager.init(len(self.groups))

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
    
    def _sample_groups(self):
        self.groups = self.group_sampler.sample(self.phase, output_str_index=False)
        self.num_total_group_batches = divide(len(self.groups), self.batch_size, drop_last=self.drop_last)

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
        transformables = {}
        for name in self.transformables:
            _t_cfg = copy.deepcopy(self.transformables[name])
            transformable_type = _t_cfg.pop("type")
            loader_cfg = _t_cfg.pop("loader", None)
            loader = self._build_transformable_loader(loader_cfg, transformable_type)
            tensor_smith = build_tensor_smith(_t_cfg.pop("tensor_smith")) if "tensor_smith" in _t_cfg else None
            scene_data = self.info[index_info.scene_id]
            transformables[name] = self._build_transformable(name, scene_data, index_info, loader, tensor_smith=tensor_smith, **_t_cfg)
        
        return transformables
    

    def _build_transformable_loader(self, loader_cfg, transformable_type: str) -> TransformableLoader:
        if loader_cfg:
            loader_cfg.setdefault("data_root", self.data_root)  # fallback with default data_root from Dataset
        else:
            loader_cfg = self._get_default_loader_cfg(transformable_type)
        loader = build_transformable_loader(loader_cfg)
        return loader
    
    def _get_default_loader_cfg(self, transformable_type: str) -> Dict:
        if transformable_type not in self.DEFAULT_LOADERS:
            raise ValueError(f"No default transformable loader for transformable type: {transformable_type}. Please provide one explicitly.")
        return {
            "type": self.DEFAULT_LOADERS[transformable_type],
            "data_root": self.data_root,
        }

    @staticmethod
    def _build_tensor_smith(tensor_smith: dict = None):
        tensor_smith = copy.deepcopy(tensor_smith)
        if isinstance(tensor_smith, dict):
            tensor_smith = TENSOR_SMITHS.build(tensor_smith)
        return tensor_smith

    def __len__(self):
        if self.subepoch_manager is None:
            return self.num_total_group_batches
        return self.subepoch_manager.get_num_group_batches_in_cur_subepoch()

    @staticmethod
    def _batch_groups(group_batch_ind, groups, batch_size):
        batched_groups = []
        for batch_idx in range(batch_size):
            group_idx = group_batch_ind * batch_size + batch_idx
            if group_idx >= len(groups):
                group_idx = max(0, 2 * (len(groups) - 1) - group_idx)
            batched_groups.append(groups[group_idx])
        return batched_groups

    def post_epoch_processing(self):
        if self.subepoch_manager is None: # training on the normal epoch settings
            self._sample_groups()
            return

        try: # training on sub-epochs
            self.subepoch_manager.to_next_sub_epoch()
        except EndOfAllSubEpochs:
            self._sample_groups()
            self.subepoch_manager.reset(len(self.groups))

    def __getitem__(self, idx) -> GroupBatch:
        if self.subepoch_manager is not None:
            idx = self.subepoch_manager.translate_index(idx)

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
        # TODO: set seed using seed_dataset
        batch_seed = int.from_bytes(os.urandom(2), byteorder="big")
        for group_of_inputs in batch:
            group_seed = int.from_bytes(os.urandom(2), byteorder="big")
            for input_dict in group_of_inputs:
                frame_seed = int.from_bytes(os.urandom(2), byteorder="big")
                transformables = input_dict["transformables"].values()
                for transform in self.transforms:
                    transform(*transformables, seeds={"group": group_seed, "batch": batch_seed, "frame": frame_seed})

        group_batch = []
        for frame_batch in zip(*batch):
            group_batch.append(frame_batch)

        model_food = self.model_feeder(group_batch)

        return model_food

    def _build_transformable(self, name: str, scene_data: Dict, index_info: IndexInfo, loader: "TransformableLoader", tensor_smith: "TensorSmith" = None, **kwargs) -> "Transformable":
        return loader.load(name, scene_data, index_info, tensor_smith=tensor_smith, **kwargs)


@DATASETS.register_module()
class ClassBalancedGroupSampler:
    def __init__( self, annotation_loaders: Union[TransformableLoader, List[TransformableLoader]] = Bbox3DLoader, ):
        pass
