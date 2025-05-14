import abc
import copy
import math
import random
import warnings
from itertools import cycle
from pathlib import Path
from cachetools import cached, Cache
from collections import defaultdict, UserList, Counter
from typing import List, Tuple, Dict, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from copious.data_structure.dict import defaultdict2dict
from loguru import logger
from tabulate import tabulate

from prefusion.registry import GROUP_SAMPLERS
from prefusion.dataset.transformable_loader import Bbox3DLoader
from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.transformable_loader import Bbox3DLoader, Polyline3DLoader, Polygon3DLoader, ParkingSlot3DLoader
from prefusion.dataset.transform import Bbox3D, Polyline3D, Polygon3D, ParkingSlot3D
from prefusion.dataset.utils import build_transformable_loader, PolarDict, load_frame_data_in_the_group, load_scene_data

if TYPE_CHECKING:
    from .dataset import TransformableLoader
    from .transform import Transformable

__all__ = ["IndexGroupSampler", "ClassBalancedGroupSampler"]


class Group(UserList):
    pass


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


def get_scene_frame_inds(frame_info: PolarDict, indices: List[str] = None) -> Dict[str, List[str]]:
    """prepare scene_frame_inds for later usage

    Parameters
    ----------
    frame_info : PolarDict
        the frame info pkl paths of the dataset
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
    indices = indices or frame_info.keys()
    frame_info_keys = set(frame_info.keys())
    indices = sorted([i for i in indices if i in frame_info_keys])

    available_indices = defaultdict(list)
    for scn_frm_id in indices:
        scene_id = scn_frm_id.split("/")[0]
        available_indices[scene_id].append(scn_frm_id)
    for scene_id in available_indices:
        available_indices[scene_id] = sorted(available_indices[scene_id])

    return defaultdict2dict(available_indices)


def convert_str_index_to_index_info(groups: List[Union[List[str], Group[str]]]) -> List[Group["IndexInfo"]]:
    index_info_groups = []
    for grp in groups:
        index_info_grp = []
        prev = None
        for i, scn_frm_str in enumerate(grp):
            cur = IndexInfo.from_str(scn_frm_str, prev=prev)
            prev = cur
            index_info_grp.append(cur)
        index_info_groups.append(Group(index_info_grp))
    return index_info_groups


class GroupSampler:
    def __init__(
        self, 
        phase: str, 
        possible_group_sizes: Union[int, Tuple[int]], 
        possible_frame_intervals: Union[int, Tuple[int]] = 1, 
        seed: int = None,
        **kwargs,
    ):
        self.phase = phase.lower()
        self.possible_group_sizes = [possible_group_sizes] if isinstance(possible_group_sizes, int) else possible_group_sizes
        self.possible_frame_intervals = [possible_frame_intervals] if isinstance(possible_frame_intervals, int) else possible_frame_intervals
        self._cur_train_group_size = self.possible_group_sizes[0]  # train group size of current epoch
        self.seed = seed

    @property
    def group_size(self) -> int:
        return self._cur_train_group_size

    @abc.abstractmethod
    def sample(self, data_root: Path, scene_info: PolarDict, frame_info: PolarDict, **kwargs) -> List[Group["IndexInfo"]]:
        raise NotImplementedError


@GROUP_SAMPLERS.register_module()
class IndexGroupSampler(GroupSampler):
    def __init__(
        self, 
        phase: str, 
        possible_group_sizes: Union[int, Tuple[int]], 
        possible_frame_intervals: Union[int, Tuple[int]] = 1, 
        seed: int = None,
        indices_path: Union[str, Path] = None,
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
        super().__init__(phase, possible_group_sizes, possible_frame_intervals, seed)
        assert phase in ["train", "val", "test"]
        self.indices_path = indices_path

    def sample(self, data_root: Path, scene_info: PolarDict, frame_info: PolarDict, **kwargs) -> List[Group["IndexInfo"]]:
        """Sample groups

        Parameters
        ----------
        frame_info : PolarDict
            the frame info pkl paths of the dataset
        Returns
        -------
        List[Group["IndexInfo"]]
            Generated groups
        """
        # generate scene_frame_inds
        if self.indices_path is not None:
            indices = [line.strip() for line in open(self.indices_path, "r")]
            scene_frame_inds = get_scene_frame_inds(frame_info, indices=indices)
        else:
            scene_frame_inds = get_scene_frame_inds(frame_info)

        # sample groups
        match self.phase:
            case "train":
                groups: List[List[str]] = self.sample_train_groups(scene_frame_inds)
            case "val" | "test":
                groups: List[List[str]] = self.sample_val_groups(scene_frame_inds)

        return convert_str_index_to_index_info(groups)

    def sample_train_groups(self, scene_frame_inds: Dict[str, List[str]]) -> List[List[str]]:
        if self.seed: random.seed(self.seed)
        self._cur_train_group_size = random.choice(self.possible_group_sizes)
        return self._generate_groups(scene_frame_inds, self._cur_train_group_size, random_start_ind=True, shuffle=True, seed=self.seed)

    def sample_val_groups(self, scene_frame_inds: Dict[str, List[str]]) -> List[List[str]]:
        return self._generate_groups(scene_frame_inds, self.possible_group_sizes[0], frame_interval=self.possible_frame_intervals[0], start_ind=0, random_start_ind=False, shuffle=False)
    
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
            groups = [Group([frame_ids[i] for i in inds]) for inds in inds_list]
            all_groups.extend(groups)
        if shuffle:
            if self.seed: random.seed(self.seed)
            random.shuffle(all_groups)
        return all_groups


@GROUP_SAMPLERS.register_module()
class SequentialSceneFrameGroupSampler(GroupSampler):
    def __init__(self, phase: str, seed: int = None, **kwargs):
        """Prepare groups frame by frame, scene by scene.

        Parameters
        ----------
        phase : str
            Specifies the phase ('train', 'val', 'test' or 'test_scene_by_scene') of the dataset; default is 'train'.
        seed : int, optional
            Random seed for randomization operations. It's usually for testing and debugging purpose.
        """
        super().__init__(phase, possible_group_sizes=1, possible_frame_intervals=1, seed=seed)
        assert phase == "test_scene_by_scene"

    def sample(self, data_root: Path, scene_info: PolarDict, frame_info: PolarDict, **kwargs) -> List[Group["IndexInfo"]]:
        scene_frame_inds = get_scene_frame_inds(frame_info)
        groups = convert_str_index_to_index_info(list(scene_frame_inds.values())) # build prev/next connections between adjacent frames in the same scene
        groups = [Group([frm]) for grp in groups for frm in grp] # flatten all scenes into single-frame groups
        return groups


@GROUP_SAMPLERS.register_module()
class ClassBalancedGroupSampler(GroupSampler):
    SUPPORTED_LOADERS = {
        Bbox3D.__name__: Bbox3DLoader,
        Polyline3D.__name__: Polyline3DLoader,
        Polygon3D.__name__: Polygon3DLoader,
        ParkingSlot3D.__name__: ParkingSlot3DLoader,
    }

    def __init__(
        self, 
        phase: str, 
        possible_group_sizes: Union[int, Tuple[int]], 
        possible_frame_intervals: Union[int, Tuple[int]] = 1, 
        seed: int = None,
        transformable_cfg: Dict[str, "Transformable"] = None,
        cbgs_cfg: Union[str, Dict] = None,
        num_processes: int = 0,
        **kwargs
    ):
        super().__init__(phase, possible_group_sizes, possible_frame_intervals, seed)
        assert phase == "train", "ClassBalancedGroupSampler is only designed for train phase."
        assert transformable_cfg is not None, "transformables cannot be None, it's for loading annotation data that has class info."
        assert cbgs_cfg is not None, "cbgs_cfg cannot be None."
        self.transformable_cfg = transformable_cfg
        self.num_processes = num_processes
        self._base_group_sampler = IndexGroupSampler(phase, possible_group_sizes, possible_frame_intervals, seed)
        self.cbgs_cfg = self._cbgs_cfg_with_default_values(cbgs_cfg)
        self.default_data_root = None
    
    def _cbgs_cfg_with_default_values(self, cbgs_cfg: Dict):
        def _set_default(key, value):
            if key not in cbgs_cfg:
                warnings.warn(f"{key} not found in cbgs_cfg, using default value {value}", UserWarning)
                cbgs_cfg[key] = value
        
        _set_default("desired_ratio", 0.25)
        _set_default("counter_type", "frame")
        _set_default("update_stats_during_oversampling", False)
        _set_default("oversampling_consider_no_objects", False)
        _set_default("oversampling_consider_object_attr", False)

        return cbgs_cfg

    @staticmethod
    def _to_df(groups: List[Group], colname: str = "cnt", fill_value: float = 0.0) -> pd.DataFrame:
        gpdf = pd.DataFrame([getattr(grp, colname) for grp in groups]).fillna(fill_value)
        return gpdf


    def sample(self, data_root: Path, scene_info: PolarDict, frame_info: PolarDict, **kwargs) -> List[Group["IndexInfo"]]:
        self.default_data_root = data_root
        groups = self._base_group_sampler.sample(data_root, scene_info, frame_info)
        groups = self.count_class_and_attr_occurrence(data_root, scene_info, frame_info, groups)
        sampled_groups = self.iterative_sample_minority_groups(groups)
        total_groups = groups + sampled_groups
        self.print_cbgs_report(groups, total_groups, self.cbgs_cfg)
        return total_groups
    
    @staticmethod
    def print_cbgs_report(groups_before: List[Group], groups_after: List[Group], cbgs_cfg: Dict):
        num_groups = pd.DataFrame({"before": [len(groups_before)], "after": [len(groups_after)]})
        before = ClassBalancedGroupSampler._to_df(groups_before).sum(axis=0)
        after = ClassBalancedGroupSampler._to_df(groups_after).sum(axis=0)
        combined = pd.DataFrame({"before": before, "after": after})
        combined.loc[:, "before_ratio"] = combined.before.div(combined.before.max())
        combined.loc[:, "after_ratio"] = combined.after.div(combined.after.max())
        logger.info(
            "\n============== CBGS Report ==============\n"
            f"### Number of Groups ###\n"
            f"{tabulate(num_groups.values, headers=num_groups.columns, tablefmt='psql')}\n\n"
            f"### Occurrence ({cbgs_cfg['counter_type']}) per Class ###\n"
            f"{tabulate(combined, headers='keys', tablefmt='psql')}\n"
        )

    def count_class_and_attr_occurrence(self, data_root: Path, scene_info: PolarDict, frame_info: PolarDict, groups: List[Group]) -> List[Group]:
        for group in groups:
            obj_cnt = Counter()
            frm_cnt = Counter()
            for index_info in group:
                transformables = self.load_all_transformables(data_root, scene_info, frame_info, index_info)
                no_objects_found = True
                for _, transformable in transformables.items():
                    classes = [ele['class'] for ele in transformable.elements]
                    attrs = [attr_val for ele in transformable.elements for attr_val in ele['attr']]
                    
                    if classes:
                        obj_cnt.update(classes)
                        frm_cnt.update(set(classes))
                        no_objects_found = False

                    if attrs and self.cbgs_cfg.get("oversampling_consider_object_attr"):
                        obj_cnt.update(attrs)
                        frm_cnt.update(set(attrs))
                        no_objects_found = False

                if self.cbgs_cfg.get("oversampling_consider_no_objects") and no_objects_found:
                    obj_cnt.update(['<NO_OBJECTS>'])
                    frm_cnt.update(['<NO_OBJECTS>'])
            
            group.obj_cnt = obj_cnt
            group.frm_cnt = frm_cnt
            group.grp_cnt = Counter(list(frm_cnt.keys()))
            group.cnt = self._decide_counter(group)

        return groups
    
    def _decide_counter(self, group: Group):
        match self.cbgs_cfg:
            case {"counter_type": "group", **rest}:
                return group.grp_cnt
            case {"counter_type": "frame", **rest}:
                return group.frm_cnt
            case {"counter_type": "object", **rest}:
                return group.obj_cnt
            case _:
                return group.frm_cnt

    def iterative_sample_minority_groups(self, groups: List[Group]) -> List[Group]:
        # Calculate class distribution
        groups_df = self._to_df(groups)
        cnt_per_class = groups_df.sum(axis=0)
        max_class = cnt_per_class.idxmax()

        # Get minority classes
        class_stats_df = cnt_per_class.to_frame(name="cnt")
        class_stats_df.loc[:, "ratio"] = class_stats_df.cnt / class_stats_df.loc[max_class]['cnt']
        minority_classes = class_stats_df[class_stats_df.ratio < self.cbgs_cfg["desired_ratio"]]\
            .reset_index(names=["cls"])\
            .sort_values(["cnt", "cls"])\
            .cls.tolist() # NOTE: it's sorted by `cnt`, then `cls`

        if not minority_classes:
            return []

        # Oversample the minority classes to the same level
        sep = 1
        oversampling_classes, target_classes = minority_classes[:sep], minority_classes[sep:]
        sampled_groups = []
        while len(target_classes) > 0:
            target_class = target_classes[0] # class with minimum cnt in target_classes
            if self.cbgs_cfg["update_stats_during_oversampling"]:
                sampled_groups.extend(self.sample_minority_groups(groups + sampled_groups, oversampling_classes, target_class, target_ratio=1.0))
                groups_df = self._to_df(groups + sampled_groups)
                class_stats_df = groups_df.sum(axis=0).to_frame(name="cnt")
            else:
                sampled_groups.extend(self.sample_minority_groups(groups, oversampling_classes, target_class, target_ratio=1.0))
        
            sep += 1
            oversampling_classes, target_classes = minority_classes[:sep], minority_classes[sep:]

        # Final Oversampling referring to the max class
        if self.cbgs_cfg["update_stats_during_oversampling"]:
            sampled_groups.extend(self.sample_minority_groups(groups + sampled_groups, oversampling_classes, max_class, target_ratio=self.cbgs_cfg["desired_ratio"]))
        else:
            sampled_groups.extend(self.sample_minority_groups(groups, oversampling_classes, max_class, target_ratio=self.cbgs_cfg["desired_ratio"]))

        return sampled_groups

    def format_group_name(self, grp):
        output = grp.data[0].scene_id
        for i in grp.data:
            output+= ("&" + i.frame_id)
        return output

    def sample_minority_groups(self, groups: List[Group], minority_classes: List[str], target_class: str, target_ratio: float = 1.0) -> List[Group]:
        groups_df = self._to_df(groups)
        cnt_per_class = groups_df.sum(axis=0)
        gap_cnt = int(math.ceil(cnt_per_class[target_class] * target_ratio - max([cnt_per_class[c] for c in minority_classes])))
        groups_df.loc[:, 'gp_name'] = [self.format_group_name(grp) for grp in groups]
        groups_df = groups_df.drop_duplicates()  # oversamping in the following step should only select groups from the original pool (while calculating gap may based on updated groups)
        sampled_groups = []
        for minor_cls in minority_classes:
            colname = f"{minor_cls}_{target_class}_ratio"
            groups_df.loc[:, colname] = groups_df[minor_cls] / (groups_df[target_class] + 1e-4)
            candidates = cycle(groups_df[groups_df[colname] > 1e-6][colname].sort_values(ascending=False).index)
            _group_indices = [next(candidates) for _ in range(0, gap_cnt)]
            sampled_groups.extend([copy.deepcopy(groups[idx]) for idx in _group_indices])

        return sampled_groups

    def load_all_transformables(self, data_root: Path, scene_info: PolarDict, frame_info: PolarDict, index_info: "IndexInfo") -> dict:
        transformables = {}
        scene_data = load_scene_data(data_root, scene_info, index_info)
        frame_data = load_frame_data_in_the_group(data_root, frame_info, index_info)
        for name in self.transformable_cfg:
            _t_cfg = self.transformable_cfg[name]
            if _t_cfg["type"] not in self.SUPPORTED_LOADERS:
                continue
            loader_cfg = _t_cfg["loader"] if "loader" in _t_cfg else None
            loader = self._build_transformable_loader(loader_cfg, _t_cfg["type"])
            rest_kwargs = {k: v for k, v in _t_cfg.items() if k not in ["type", "loader", "tensor_smith"]}
            transformables[name] = loader.load(name, scene_data, frame_data, index_info, **rest_kwargs)
        
        return transformables
    
    @cached(cache=Cache(maxsize=float('inf')), key=lambda self_, cfg, type_: (str(sorted((cfg or {}).items())), type_))
    def _build_transformable_loader(self, loader_cfg, transformable_type: str) -> "TransformableLoader":
        if loader_cfg:
            loader_cfg.setdefault("data_root", self.default_data_root)  # fallback with default data_root from Dataset
        else:
            loader_cfg = dict(type=self.SUPPORTED_LOADERS[transformable_type], data_root=self.default_data_root)
        loader = build_transformable_loader(loader_cfg)
        return loader


@GROUP_SAMPLERS.register_module()
class SceneLevelBalancedGroupSampler(GroupSampler):
    SUPPORTED_LOADERS = {
        Bbox3D.__name__: Bbox3DLoader,
        Polyline3D.__name__: Polyline3DLoader,
        Polygon3D.__name__: Polygon3DLoader,
        ParkingSlot3D.__name__: ParkingSlot3DLoader,
    }

    def __init__(
        self, 
        phase: str, 
        possible_group_sizes: Union[int, Tuple[int]], 
        possible_frame_intervals: Union[int, Tuple[int]] = 1, 
        seed: int = None,
        transformable_cfg: Dict[str, "Transformable"] = None,
        cbgs_cfg: Union[str, Dict] = None,
        num_processes: int = 0,
        **kwargs
    ):
        super().__init__(phase, possible_group_sizes, possible_frame_intervals, seed)
        self._base_group_sampler = ClassBalancedGroupSampler(phase, possible_group_sizes, possible_frame_intervals, seed)

    def sample(self, data_root: Path, scene_info: PolarDict, frame_info: PolarDict, **kwargs) -> List[Group["IndexInfo"]]:
        # TODO: [ ] 1. calculate occurred tags for each scene based on the groups generated by self._base_group_sampler
        # TODO: [ ] 2. expand scenes according to tag distribution (e.g. expand groups that in the rare scenes)
        raise NotImplementedError
