# Written by AlphaLFC. All rights reserved.
import os
import copy
import logging
import functools
from cachetools import cached, Cache
from pathlib import Path
from typing import List, Dict, Union, TYPE_CHECKING

import mmengine
from mmengine.logging import print_log
from torch.utils.data import Dataset
from prefusion.registry import DATASETS, MMENGINE_FUNCTIONS, TENSOR_SMITHS
from prefusion.dataset.subepoch_manager import EndOfAllSubEpochs
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

from .utils import (
    build_transforms, 
    build_model_feeder, 
    build_tensor_smith, 
    build_transformable_loader, 
    build_group_sampler,
    build_subepoch_manager, 
    divide,
)

if TYPE_CHECKING:
    from .tensor_smith import TensorSmith
    from .transform import Transform
    from .model_feeder import BaseModelFeeder
    from .transform import Transformable
    from .index_info import IndexInfo
    from .subepoch_manager import SubEpochManager
    from .group_sampler import GroupSampler

__all__ = ["GroupBatchDataset"]


@MMENGINE_FUNCTIONS.register_module()
def collate_dict(batch):
    return batch[0]


GroupBatch = List[List[Dict]]

_print_log = functools.partial(print_log, logger='current')


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
        group_sampler: Union["GroupSampler", dict] = None,
        batch_size: int = 1,
        drop_last: bool = False,
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
        - batch_size (int, optional): Batch size; defualt is 1.
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
        self.info = mmengine.load(str(info_path))
        self.transformables = transformables
        self.transforms = build_transforms(transforms)
        self.model_feeder = build_model_feeder(model_feeder)
        self.group_sampler = build_group_sampler(group_sampler)
        self.subepoch_manager = None
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.group_backtime_prob = group_backtime_prob
        self.seed_dataset = seed_dataset

        self._sample_groups()

        if self.group_sampler.phase == "train":
            self.subepoch_manager = build_subepoch_manager(subepoch_manager, batch_size)
            if self.subepoch_manager is not None:
                _print_log(
                    "Since you have set SubEpochManager, "
                    "you should consider set a larger max_epochs for your training.", 
                    level=logging.WARNING
                )
                self.subepoch_manager.init(len(self.groups))
    
    def _sample_groups(self):
        self.groups = self.group_sampler.sample(self.data_root, self.info)
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
                f"    batch_size={self.batch_size}, \n"
                f"    group_size={self.group_sampler.possible_group_sizes}, \n"
                f"    frame_interval={self.group_sampler.possible_frame_intervals}, \n"
                f"    group_backtime_prob={self.group_backtime_prob}\n)",
            ]
        )

    def load_all_transformables(self, index_info: "IndexInfo") -> dict:
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
    
    @cached(cache=Cache(maxsize=float('inf')), key=lambda self_, cfg, type_: (str(sorted((cfg or {}).items())), type_))
    def _build_transformable_loader(self, loader_cfg: dict, transformable_type: str) -> TransformableLoader:
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
        return self.subepoch_manager.num_group_batches_per_subepoch

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
            # from mmengine.dist import get_dist_info
            # from loguru import logger
            # rank, _ = get_dist_info()
            # logger.debug(f"[rank{rank}] id={id(self.subepoch_manager)}")
            # logger.debug(f"[rank{rank}] drop_last_group_batch={self.subepoch_manager.drop_last_group_batch}")
            # logger.debug(f"[rank{rank}] drop_last_subepoch={self.subepoch_manager.drop_last_subepoch}")
            # logger.debug(f"[rank{rank}] num_total_groups={self.subepoch_manager.num_total_groups}")
            # logger.debug(f"[rank{rank}] num_total_group_batches={self.subepoch_manager.num_total_group_batches}")
            # logger.debug(f"[rank{rank}] num_group_batches_per_subepoch={self.subepoch_manager.num_group_batches_per_subepoch}")
            # logger.debug(f"[rank{rank}] batch_size={self.subepoch_manager.batch_size}")
            # logger.debug(f"[rank{rank}] num_subepochs={self.subepoch_manager.num_subepochs}")
            # logger.debug(f"[rank{rank}] id(visited)={id(self.subepoch_manager.visited)}")
            # logger.debug(f"[rank{rank}] visited={self.subepoch_manager.visited.todict().keys()}")

            self._sample_groups()
            self.subepoch_manager.reset(len(self.groups))


    def __getitem__(self, idx) -> GroupBatch:
        if self.subepoch_manager is not None:
            idx = self.subepoch_manager.translate_index(idx)

        batched_groups = self._batch_groups(idx, self.groups, self.batch_size)

        batch = []
        batch_seed = int.from_bytes(os.urandom(2), byteorder="big")
        for group in batched_groups:
            group_of_inputs = []
            group_seed = int.from_bytes(os.urandom(2), byteorder="big")
            for i, index_info in enumerate(group):
                frame_seed = int.from_bytes(os.urandom(2), byteorder="big")
                transformables = self.load_all_transformables(index_info)
                for transform in self.transforms:
                    transform(*transformables.values(), seeds={"group": group_seed, "batch": batch_seed, "frame": frame_seed})
                input_dict = {
                    "index_info": index_info,
                    "transformables": transformables,
                }
                group_of_inputs.append(input_dict)
            batch.append(group_of_inputs)
        
        group_batch = []
        for frame_batch in zip(*batch):
            group_batch.append(frame_batch)

        model_food = self.model_feeder(group_batch)

        return model_food


    def _build_transformable(self, name: str, scene_data: Dict, index_info: "IndexInfo", loader: "TransformableLoader", tensor_smith: "TensorSmith" = None, **kwargs) -> "Transformable":
        return loader.load(name, scene_data, index_info, tensor_smith=tensor_smith, **kwargs)
