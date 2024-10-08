from typing import List, Union, Dict, TYPE_CHECKING
from pathlib import Path
import copy

import cv2
import numpy as np
import virtual_camera as vc
from pypcd_imp import pypcd

from prefusion.registry import TRANSFORMS, TENSOR_SMITHS, MODEL_FEEDERS

if TYPE_CHECKING:
    from .transform import Transform
    from .tensor_smith import TensorSmith
    from .model_feeder import BaseModelFeeder

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


def build_tensor_smiths(tensor_smiths: Dict[str, Union[dict, "TensorSmith"]]) -> Dict[str, "TensorSmith"]: 
    if tensor_smiths is None:
        return {}
    built_tensor_smiths = {}
    for transformabel_key, tensor_smith in tensor_smiths.items():
        tensor_smith = copy.deepcopy(tensor_smith)
        if isinstance(tensor_smith, dict):
            tensor_smith = TENSOR_SMITHS.build(tensor_smith)
        built_tensor_smiths[transformabel_key] = tensor_smith
    return built_tensor_smiths


def build_model_feeder(model_feeder: Union["BaseModelFeeder", dict]) -> "BaseModelFeeder":
    if model_feeder is None:
        return MODEL_FEEDERS.build(dict(type='BaseModelFeeder'))
    model_feeder = copy.deepcopy(model_feeder)
    if isinstance(model_feeder, dict):
        return MODEL_FEEDERS.build(model_feeder)
    return model_feeder

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
