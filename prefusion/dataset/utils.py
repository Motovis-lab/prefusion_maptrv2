from typing import List, Union, Dict, TYPE_CHECKING, Tuple
from pathlib import Path
from collections import defaultdict, namedtuple
import copy

import torch
import mmcv
from tqdm import tqdm
import numpy as np
from pypcd_imp import pypcd

from prefusion.registry import TRANSFORMS, MODEL_FEEDERS, TRANSFORMABLE_LOADERS, TENSOR_SMITHS

if TYPE_CHECKING:
    from .transform import Transform
    from .model_feeder import BaseModelFeeder
    from .transformable_loader import TransformableLoader

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


def calculate_bbox3d_ap(
    gt: Dict[str, List[Dict]],
    pred: Dict[str, List[Dict]],
    iou_thresh: float = 0.5,
    max_conf_as_pred_class: bool = True,
    is_first_conf_special: bool = True,
    allow_gt_reuse: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """calculate average precision of predicted 3D bounding boxes against ground truth

    Parameters
    ----------
    gt : Dict[str, List[Dict]]
        >>> {
                '169883828964': [
                    {
                        'confs': [1, 1, 0],  # each value denotes the confidence of the class
                        'corners': torch.Tensor([[ 6.7, -5.7, -0.02],
                                                 [ 6.0, -3.5, -0.03],
                                                 [ 6.0, -3.5,  1.91],
                                                 [ 6.7, -5.7,  1.92],
                                                 [ 1.8, -7.4, -0.05],
                                                 [ 1.0, -5.3, -0.06],
                                                 [ 1.0, -5.3,  1.88],
                                                 [ 1.8, -7.4,  1.89]])  # of shape (8, 3), indicating the location of 8 corners of the bbox
                    },
                    ...
                ],
                
                ...
            }

    pred : Dict[str, List[Dict]]
        >>> {
                '169883828964': [
                    {
                        'confs': [0.9999999, 0.82, 0.03],  # each value denotes the confidence of the class
                        'corners': torch.Tensor([[ 6.7, -5.7, -0.02],
                                                 [ 6.0, -3.5, -0.03],
                                                 [ 6.0, -3.5,  1.91],
                                                 [ 6.7, -5.7,  1.92],
                                                 [ 1.8, -7.4, -0.05],
                                                 [ 1.0, -5.3, -0.06],
                                                 [ 1.0, -5.3,  1.88],
                                                 [ 1.8, -7.4,  1.89]])  # of shape (8, 3), indicating the location of 8 corners of the bbox
                    },
                    ...
                ],
                
                ...
            }

    iou_thresh : float, optional
        the threshold for deciding whethat 2 boxes are matched, by default 0.5

    max_conf_as_pred_class : bool, optional
        whether to use the max confidence as the predicted class, by default True

    is_first_conf_special : bool, optional
        whether the first confidence is special (i.e. denoting is_superclass), by default True
    
    allow_gt_reuse : bool, optional
        whether to allow a ground truth box to be reused (i.e. matched more than once with predictions), by default True

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        pred_confidences, precision, recall and ap, respectively. All of them are sorted according to pred confidences.
    """
    from pytorch3d import box3d_overlap
    assert 0 < iou_thresh < 1, 'iou thresh should be a value between (0, 1)'
    PredResult = namedtuple('PredResult', 'frame_id box_id cls_idx conf matched', defaults=[None, None, None])
    num_classes = len(next(iter(gt.values()))['confs'])
    
    union_frame_ids = set(gt.keys()) | set(pred.keys())
    all_predictions = []  # if max_conf_as_pred_class==False, predicted results will be duplicated for each class
    for frame_id in tqdm(union_frame_ids):
        pred_of_frame = pred.get(frame_id, [])
        gt_of_frame = gt.get(frame_id, [])
        predictions = []

        if not pred_of_frame:
            continue

        if not gt_of_frame:
            for box_id, bx in enumerate(pred_of_frame):
                if max_conf_as_pred_class:
                    start_pos = int(is_first_conf_special)  # if is_first_conf_special==True, ignore the special conf
                    cls_idx = np.argmax(bx["confs"][start_pos:]) + start_pos  # if is_first_conf_special==True, idx should be increased by 1
                    assert cls_idx > 0
                    pred_res = PredResult(frame_id, box_id, cls_idx, bx["confs"][cls_idx], False)
                    predictions.append(pred_res)
                else:
                    predictions.extend([PredResult(frame_id, box_id, cls_idx, bx["confs"][cls_idx], False) for cls_idx in range(num_classes)])
            all_predictions.extend(predictions)
            continue

        pred_confs = torch.stack([bx["confs"] for bx in pred_of_frame])
        pred_corners = torch.stack([bx["corners"] for bx in pred_of_frame])
        gt_confs = torch.stack([bx["confs"] for bx in gt_of_frame])
        gt_corners = torch.stack([bx["corners"] for bx in gt_of_frame])

        _, ious = box3d_overlap(pred_corners, gt_corners)

        if max_conf_as_pred_class:
            start_pos = int(is_first_conf_special)  # if is_first_conf_special==True, ignore the special conf
            pred_conf_max_idx = torch.argmax(pred_confs[:, start_pos:], dim=1) + start_pos  # if is_first_conf_special==True, idx should be increased by 1
            assert (pred_conf_max_idx == 0).sum() == 0
            matching_table = torch.zeros_like(ious)
            for i in range(ious.shape[0]):
                matched_gt_idx = torch.argmax(ious[i])
                pred_cls_idx = pred_conf_max_idx[i]
                pred_conf = pred_confs[i][pred_cls_idx]
                is_gt_the_same_class = gt_confs[matched_gt_idx][pred_cls_idx]
                if is_gt_the_same_class == 1 and pred_conf >= iou_thresh:
                    if allow_gt_reuse or not matching_table[i].any():
                        predictions.append(PredResult(frame_id, i, pred_cls_idx, pred_conf, True))
                        matching_table[i, matched_gt_idx] = 1
                else:
                    predictions.append(PredResult(frame_id, i, pred_cls_idx, pred_conf, False))
        else:
            for cls_idx in range(num_classes):
                matching_table = torch.zeros_like(ious)
                for i in range(ious.shape[0]):
                    matched_gt_idx = torch.argmax(ious[i])
                    pred_conf = pred_confs[i][cls_idx]
                    is_gt_the_same_class = gt_confs[matched_gt_idx][cls_idx]
                    if is_gt_the_same_class == 1 and pred_conf >= iou_thresh:
                        if allow_gt_reuse or not matching_table[i].any():
                            predictions.append(PredResult(frame_id, i, cls_idx, pred_conf, True))
                            matching_table[i, matched_gt_idx] = 1
                    else:
                        predictions.append(PredResult(frame_id, i, cls_idx, pred_conf, False))

    cls_idxes = list(range(num_classes))
    if max_conf_as_pred_class:
        del cls_idxes[0]
    
    results = {}
    for cls_idx in cls_idxes:
        filtered_predictions = [p for p in all_predictions if p.cls_idx == cls_idx]
        sorted_predictions = sorted(filtered_predictions, key=lambda x: x.conf, reverse=True)
        num_gt_boxes = sum([sum(confs[cls_idx] == 1 for confs in gt_of_frame["confs"]) for gt_of_frame in gt.values()])
        tp = torch.zeros(len(sorted_predictions))
        fp = torch.zeros(len(sorted_predictions))
        for i, p in enumerate(sorted_predictions):
            if p.matched:
                tp[i] = 1
            else:
                fp[i] = 1
        cfp = np.cumsum(fp)
        ctp = np.cumsum(tp)
        recall = ctp / (num_gt_boxes + 1e-7)
        precision = ctp / (ctp + cfp + 1e-7)
        ap = voc_ap(precision, recall)
        results[cls_idx] = (sorted_predictions, precision, recall, ap)
    
    return results


def voc_ap(precision, recall, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap