import copy
from typing import List, Union, Dict, TYPE_CHECKING, Tuple, Any
from pathlib import Path
from functools import lru_cache
from cachetools import cached, Cache

import torch
import mmengine
import polars as pl
import mmcv
import numpy as np
from pypcd_imp import pypcd

from prefusion.registry import TRANSFORMS, MODEL_FEEDERS, TRANSFORMABLE_LOADERS, TENSOR_SMITHS, DATASET_TOOLS, GROUP_SAMPLERS

if TYPE_CHECKING:
    from .transform import Transform
    from .model_feeder import BaseModelFeeder
    from .transformable_loader import TransformableLoader
    from .subepoch_manager import SubEpochManager
    from .group_sampler import GroupSampler

INF_DIST = 1e8

def expand_line_2d(line, radius=1):
    vec = line[1] - line[0]
    norm_vec = vec / np.linalg.norm(vec)
    vert_norm_vec = norm_vec[::-1] * [1, -1]
    expand_vec = radius * vert_norm_vec
    point_0 = line[0] + expand_vec
    point_1 = line[0] - expand_vec
    point_2 = line[1] - expand_vec
    point_3 = line[1] + expand_vec
    return np.float32([point_0, point_1, point_2, point_3])


def _add_new_axis(arr, n):
    for _ in range(n):
        arr = arr[..., None]
    return arr

def vec_point2line_along_direction(point, line, direction):
    point = np.float32(point)
    n_extra_dim = len(point.shape) - 1
    line = _add_new_axis(np.float32(line), n_extra_dim)
    vec = _add_new_axis(np.float32(direction), n_extra_dim)

    vec_l = line[1] - line[0]
    vec_p = line[1] - point
    C1 = vec[1] * vec_l[0] - vec[0] * vec_l[1]
    if np.abs(C1) < 1e-5:
        return np.full_like(point, np.inf)
    C2 = (vec[1] * vec_p[0] - vec[0] * vec_p[1]) / C1
    
    return vec_p - vec_l * C2
    

def dist_point2line_along_direction(point, line, direction):
    vec = vec_point2line_along_direction(point, line, direction)
    return np.linalg.norm(vec, axis=0)


def _sign(x):
    return 2 * (x > 0) - 1


def divide(dividend, divisor, drop_last=False):
    if drop_last:
        return dividend // divisor
    else:
        return int(np.ceil(dividend / divisor))


def make_seed(base_number: int, *variables, base=17) -> int:
    """Create a new Integer seed based on `base_number` and extra varying input values such as i=0,1,2,...; j=0,1,2,... 

    Parameters
    ----------
    base_number : int
        _description_
    base : int, optional
        _description_, by default 17

    Returns
    -------
    int
        _description_
    """
    for i, v in enumerate(reversed(variables)):
        base_number += (base ** i) * (v + 1)
    return base_number


def get_cam_type(name):
    if 'perspective' in name.lower():
        return 'PerspectiveCamera'
    elif 'fisheye' in name.lower():
        return 'FisheyeCamera'
    else:
        raise ValueError('Unknown camera type')


@cached(cache=Cache(maxsize=float('inf')), key=lambda cfg: str(sorted((cfg or {}).items())))
def build_tensor_smith(tensor_smith: dict = None):
    tensor_smith = copy.deepcopy(tensor_smith)
    if isinstance(tensor_smith, dict):
        tensor_smith = TENSOR_SMITHS.build(tensor_smith)
    return tensor_smith


def build_transformable_loader(loader: Union[Dict, "TransformableLoader"]) -> "TransformableLoader":
    if isinstance(loader, dict):
        return TRANSFORMABLE_LOADERS.build(loader)
    return loader


def build_transforms(transforms: List[Union[dict, "Transform"]]) -> List["Transform"]: 
    from prefusion.dataset.transform import ToTensor
    built_transforms = []
    if transforms is None:
        transforms = []
    for transform in transforms:
        if isinstance(transform, dict):
            transform = TRANSFORMS.build(transform)
        if isinstance(transform, ToTensor):
            raise ValueError("ToTensor should not be set mannually.")
        built_transforms.append(transform)
    built_transforms.append(ToTensor())
    return built_transforms


def build_model_feeder(model_feeder: Union["BaseModelFeeder", dict]) -> "BaseModelFeeder":
    if model_feeder is None:
        return MODEL_FEEDERS.build(dict(type='BaseModelFeeder'))
    model_feeder = copy.deepcopy(model_feeder)
    if isinstance(model_feeder, dict):
        return MODEL_FEEDERS.build(model_feeder)
    return model_feeder


def build_group_sampler(group_sampler: Union["GroupSampler", dict]) -> "GroupSampler":
    if group_sampler is None:
        raise ValueError("Group sampler is mandantory for dataset.")
    group_sampler = copy.deepcopy(group_sampler)
    if isinstance(group_sampler, dict):
        return GROUP_SAMPLERS.build(group_sampler)
    return group_sampler


def build_subepoch_manager(subepoch_manager: Union["SubEpochManager", dict], batch_size: int):
    if subepoch_manager is None:
        return None
    subepoch_manager = copy.deepcopy(subepoch_manager)
    if isinstance(subepoch_manager, dict):
        subepoch_manager = DATASET_TOOLS.build(subepoch_manager)
    subepoch_manager.set_batch_size(batch_size)
    return subepoch_manager

def read_pcd(path: Union[str, Path], intensity: bool = True) -> np.ndarray:
    """read pcd file

    Parameters
    ----------
    path : Union[str, Path]
        _description_
    intensity : bool, optional
        _description_, by default True
        
    Returns
    -------
    np.ndarray
        - if intensity is false, return (N, 3) array, i.e. [[x,y,z], ...]
        - if intensity is true, return (N, 4) array, i.e. [[x,y,z,intensity], ...]
    """
    pcd = pypcd.PointCloud.from_path(path)
    npdata = np.stack([pcd.pc_data['x'], pcd.pc_data['y'], pcd.pc_data['z']], axis=1)
    if intensity:
        npdata = np.concatenate([npdata, pcd.pc_data['intensity'].reshape(-1, 1)], axis=1)
    return npdata


def read_ego_mask(path):
    ego_mask = mmcv.imread(path, flag="grayscale")
    if ego_mask.max() == 255:
        ego_mask = ego_mask / 255
    return ego_mask


def T4x4(rotation: np.ndarray, translation: np.ndarray):
    """Create a 4x4 transformation matrix

    Parameters
    ----------
    rotation : np.ndarray
        of shape (3, 3)
    translation : np.ndarray
        of shape (3,) or (1, 3) or (3, 1)

    Returns
    -------
    _type_
        _description_
    """
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation.flatten()
    return mat


def get_reversed_mapping(mapping) -> Dict[str, str]:
    """Get reversed mapping from mapping"""
    reversed_mapping = {}
    for k, v in mapping.items():
        for vv in v:
            reversed_mapping[vv] = k
    return reversed_mapping


def choose_index(index_im, choices):
    """Alternative implementation (but support more than 32 choices) of 
    >>> np.choose(index_im, choices)
    """
    max_batch_size = 16
    assert len(choices) > 0
    num_batches = (len(choices) + max_batch_size - 1) // max_batch_size
    chosen = np.zeros_like(choices[0])
    for b in range(num_batches):
        start_ind = b * max_batch_size
        end_ind = min(start_ind + max_batch_size, len(choices))
        choices_batch = choices[start_ind:end_ind]
        index_mask = (index_im >= start_ind) & (index_im < end_ind)
        clipped_index_im = np.clip(index_im - start_ind, 0, len(choices_batch) - 1)
        batch_result = np.choose(clipped_index_im, choices_batch)
        chosen += batch_result * index_mask
    return chosen


def unstack_batch_size(batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    assert isinstance(batch_data, dict)
    assert all(i.ndim >= 2 for i in batch_data.values())
    assert len(set(i.shape[0] for i in batch_data.values())) == 1
    unstacked_data = []
    keys = list(batch_data.keys())
    for inner_data in zip(*batch_data.values()):
        _unstacked = {}
        for key, single_inner_data in zip(keys, inner_data):
            _unstacked[key] = single_inner_data
        unstacked_data.append(_unstacked)
    return unstacked_data


def approx_equal(a, b, eps=1e-4):
    return abs(a - b) < eps


class PolarDict:
    def __init__(self, data: dict, *, separator: str = "/", **kwargs):
        # print(data)
        self.data = pl.json_normalize(data, separator=separator, **kwargs)
        # try:
        #     self.data = pl.json_normalize(data, separator=separator, **kwargs)
        # except Exception as e:
        #     # print("Error in json_normalize:", e)
        #     print("data ===", data, 'finish ===')
        #     raise

    def __getitem__(self, key):
        return self.data[key][0]
    
    def __len__(self) -> int:
        return len(self.keys()) 

    def keys(self) -> List[str]:
        return self.data.columns
    
    def values(self) -> List:
        return [self[key] for key in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        """Return a list of (key, value) tuples in column order."""
        return [(key, self[key]) for key in self.keys()]

    def __iter__(self):
        return iter(self.keys())
    
    def __contains__(self, key):
        return key in self.keys()
    
    def __repr__(self) -> str:
        return f"PolarDict({self.to_python_dict()})"
    
    def to_python_dict(self) -> dict:
        return self.data.to_dicts()[0]


def load_frame_info(path: Union[Path, str]) -> PolarDict:
    info = mmengine.load(path)
    try:
        frame_info = PolarDict({scene_id: scene_data['frame_info'] for scene_id, scene_data in info.items()}) # PolarDict transforms nested dict to flattened dict (sep='/')
    except Exception as e:
        # print("Error in json_normalize:", e)
        print(path)
        print("info.items() ===", info.items(), 'finish ===')
        raise

    
    return frame_info


@lru_cache(maxsize=256)
def read_frame_pickle(path: Union[Path, str]) -> Dict[str, Dict]:
    return mmengine.load(path)
