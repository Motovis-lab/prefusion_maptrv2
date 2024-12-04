import abc
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Union, TYPE_CHECKING

import numpy as np
from prefusion.registry import GROUP_SAMPLERS
from prefusion.dataset.transformable_loader import Bbox3DLoader
from prefusion.dataset.index_info import IndexInfo

if TYPE_CHECKING:
    from .dataset import TransformableLoader
    from .transform import Transformable

__all__ = ["IndexGroupSampler", "ClassBalancedGroupSampler"]


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


@GROUP_SAMPLERS.register_module()
class IndexGroupSampler:
    def __init__(
        self, 
        phase: str, 
        possible_group_sizes: Union[int, Tuple[int]], 
        possible_frame_intervals: Union[int, Tuple[int]] = 1, 
        indices_path: Union[str, Path] = None,
        seed: int = None,
        **kwargs,
    ):
        """Sample groups

        Parameters
        ----------
        phase : str
            Specifies the phase ('train', 'val', 'test' or 'test_scene_by_scene') of the dataset; default is 'train'.
        possible_group_size : Union[int, Tuple[int]]
            Size of sequence group; can be a single integer or a list (e.g., [5, 10]); default is 3.
            if int, will always use this value as group_size;
            if Tuple[int], during train phase, will random pick a value as the group_size for a given epoch.
        possible_frame_interval : Union[int, Tuple[int]], optional
            Interval between frames; default is 1.
            if int, will always use this value as frame_interval;
            if Tuple[int], during train phase, will random pick a value as the frame_interval for a given epoch.
        indices_path : Union[str, Path], optional
            Specified file of indices to load; if None, all frames are automatically fetched from the info_path.
        seed : int, optional
            Random seed for randomization operations. It's usually for testing and debugging purpose.
        """
        assert phase.lower() in ["train", "val", "test", "test_scene_by_scene"]
        self.phase = phase.lower()
        self.possible_group_sizes = [possible_group_sizes] if isinstance(possible_group_sizes, int) else possible_group_sizes
        self.possible_frame_intervals = [possible_frame_intervals] if isinstance(possible_frame_intervals, int) else possible_frame_intervals
        self._cur_train_group_size = self.possible_group_sizes[0]  # train group size of current epoch
        self.indices_path = indices_path
        self.seed = seed

    @staticmethod
    def _prepare_scene_frame_inds(info: Dict, indices: List[str] = None) -> Dict[str, List[str]]:
        """prepare scene_frame_inds for later usage

        Parameters
        ----------
        info : Dict
            the full info (pkl) of the dataset
        indices : List[str], optional
            _description_, by default None

        Returns
        -------
        scene_frame_inds : Dict[str, List[str]]
            e.g.  
            ```
            {
              "20231101_160337": [ "20231101_160337/1698825817664", "20231101_160337/1698825817764"],
              "20230823_110018": [ "20230823_110018/1692759640764", "20230823_110018/1692759640864"],
            }
            ```
        """
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
    def group_size(self) -> int:
        return self._cur_train_group_size

    def sample(self, info: Dict, output_str_index: bool = False, **kwargs) -> List[List["IndexInfo"]]:
        """Sample groups

        Parameters
        ----------
        info : Dict
            the full info (pkl) of the dataset
        output_str_index: bool, optional
            whether to output str index or IndexInfo; default is False.
        Returns
        -------
        _type_
            _description_
        """
        # generate scene_frame_inds
        if self.indices_path is not None:
            indices = [line.strip() for line in open(self.indices_path, "w")]
            scene_frame_inds = self._prepare_scene_frame_inds(info, indices=indices)
        else:
            scene_frame_inds = self._prepare_scene_frame_inds(info)

        # sample groups
        match self.phase:
            case "train":
                groups = self.sample_train_groups(scene_frame_inds)
            case "val" | "test":
                groups = self.sample_val_groups(scene_frame_inds)
            case "test_scene_by_scene" :
                groups = self.sample_scene_groups(scene_frame_inds)
        if not output_str_index:
            return self._convert_groups_to_info(groups)

    @staticmethod
    def _convert_groups_to_info(groups: List[List[str]]) -> List[List["IndexInfo"]]:
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

    def sample_train_groups(self, scene_frame_inds: Dict[str, List[str]]) -> List[List[str]]:
        if self.seed: random.seed(self.seed)
        self._cur_train_group_size = random.choice(self.possible_group_sizes)
        return self._generate_groups(scene_frame_inds, self._cur_train_group_size, random_start_ind=True, shuffle=True, seed=self.seed)

    def sample_val_groups(self, scene_frame_inds: Dict[str, List[str]]) -> List[List[str]]:
        return self._generate_groups(scene_frame_inds, self.possible_group_sizes[0], frame_interval=self.possible_frame_intervals[0], start_ind=0, random_start_ind=False, shuffle=False)

    def sample_scene_groups(self, scene_frame_inds: Dict[str, List[str]]) -> List[List[str]]:
        return list(scene_frame_inds.values())

    def sample_groups_by_class_balance(self):
        raise NotImplementedError

    def sample_groups_by_meta_info(self):
        raise NotImplementedError
    
    def _generate_groups(
        self, 
        scene_frame_inds: Dict[str, List[str]],
        group_size: int, 
        frame_interval: int = None, 
        start_ind: int = 0, 
        random_start_ind: bool = False, 
        shuffle: bool = False, 
        seed: int = None
    ) -> List[List[str]]:
        all_groups = []
        for _, frame_ids in scene_frame_inds.items():
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


@GROUP_SAMPLERS.register_module()
class ClassBalancedGroupSampler:
    def __init__( self, annotation_loaders: Union["TransformableLoader", List["TransformableLoader"]] = Bbox3DLoader, ):
        pass
