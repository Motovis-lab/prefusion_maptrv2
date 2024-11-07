from typing import Dict, List
import functools

import torch
import pytest
import numpy as np

from contrib.fastray_planar.metric import PlanarBbox3DAveragePrecision, calculate_bbox3d_ap


_approx = functools.partial(pytest.approx, rel=1e-4)

def extend_bev_box_to_3d_(bev_bbox: Dict[str, List[Dict]], zmin=0, zmax=2):
    """assume the order of corners of bev_bbox is as follows:
        1 --- 0
        |     |
        |     |
        |     |
        3 --- 2
    """
    for frame_id, frame_data in bev_bbox.items():
        for box in frame_data:
            box2d = box["corners"]
            box3d = np.concatenate((
                np.concatenate((
                    np.array([box2d[0], box2d[1], box2d[1], box2d[0]]),
                    np.array([box2d[2], box2d[3], box2d[3], box2d[2]]),
                ), axis=0),
                np.array([[zmin], [zmin], [zmax], [zmax], [zmin], [zmin], [zmax], [zmax]])
            ), axis=1)
            box["corners"] = box3d


def convert_to_torch_tensor_(corners_data):
    for frame_id, frame_data in corners_data.items():
        for box in frame_data:
            box["corners"] = torch.tensor(box["corners"]).float()
            box["confs"] = torch.tensor(box["confs"]).float()


@pytest.fixture
def pred_corners():
    boxes = {
        "f001": [
            {"confs": [0.98, 0.75, 0.7], "corners": [[7, 10], [5, 10], [7, 8], [5, 8],]},
            {"confs": [0.97, 0.3, 0.35], "corners": [[5, 3], [5, 5], [4, 3], [4, 5],]},
            {"confs": [0.95, 0.8, 0.61], "corners": [[1, 5], [3, 5], [1, 6], [3, 6],]},
        ],
        "f002": [
            {"confs": [0.99, 0.91, 0.7], "corners": [[8, 5], [4, 5], [8, 1], [4, 1],]},
            {"confs": [0.98, 0.85, 0.3], "corners": [[5, -3], [5, -1], [3, -3], [3, -1],]},
            {"confs": [0.95, 0.6, 0.51], "corners": [[2, 3], [0, 3], [1, 1], [0, 1],]},
            {"confs": [0.99, 0.2, 0.98], "corners": [[6, 7], [6, 9], [2, 7], [2, 9],]},
        ],
    }
    extend_bev_box_to_3d_(boxes)
    convert_to_torch_tensor_(boxes)
    return boxes


@pytest.fixture
def gt_corners():
    boxes = {
        "f001": [
            {"confs": [1, 1, 0], "corners": [[8, 10], [5, 10], [8, 8], [5, 8],]},
            {"confs": [1, 1, 0], "corners": [[8, 3], [8, 5], [6, 3], [6, 5],]},
            {"confs": [1, 1, 0], "corners": [[3, 10], [1, 10], [3, 6], [1, 6],]},
        ],
        "f002": [
            {"confs": [1, 1, 0], "corners": [[7, 4], [4, 4], [7, 1], [4, 1],]},
            {"confs": [1, 0, 1], "corners": [[6, -3], [8, -3], [6, -1], [8, -1],]},
            {"confs": [1, 1, 0], "corners": [[3, 4], [1, 4], [3, 2], [1, 2],]},
            {"confs": [1, 1, 0], "corners": [[5, 7], [5, 9], [1, 7], [1, 9],]},
        ],
    }
    extend_bev_box_to_3d_(boxes)
    convert_to_torch_tensor_(boxes)
    return boxes


def test_calculate_bbox3d_ap(pred_corners, gt_corners):
    results, all_predictions = calculate_bbox3d_ap(gt_corners, pred_corners, iou_thresh=0.5, max_conf_as_pred_class=True)
    sorted_all_predictions = sorted(all_predictions, key=lambda x: x.conf, reverse=True)
    assert [p._asdict() for p in sorted_all_predictions] == [
        {'frame_id': 'f002', 'box_id': 3, 'cls_idx': 2, 'conf': _approx(0.98), 'matched': False},
        {'frame_id': 'f002', 'box_id': 0, 'cls_idx': 1, 'conf': _approx(0.91), 'matched': True},
        {'frame_id': 'f002', 'box_id': 1, 'cls_idx': 1, 'conf': _approx(0.85), 'matched': False},
        {'frame_id': 'f001', 'box_id': 2, 'cls_idx': 1, 'conf': _approx(0.80), 'matched': False},
        {'frame_id': 'f001', 'box_id': 0, 'cls_idx': 1, 'conf': _approx(0.75), 'matched': True},
        {'frame_id': 'f002', 'box_id': 2, 'cls_idx': 1, 'conf': _approx(0.60), 'matched': False},
        {'frame_id': 'f001', 'box_id': 1, 'cls_idx': 2, 'conf': _approx(0.35), 'matched': False},
    ]
    assert results[1]["precision"] == _approx([1, 0.5, 0.333333, 0.5, 0.4])
    assert results[1]["recall"] == _approx([0.1666666, 0.1666666, 0.1666666, 0.3333333, 0.3333333])
    assert results[1]["ap"] == _approx(0.25)
    assert results[2]["precision"] == _approx([0.0, 0.0])
    assert results[2]["recall"] == _approx([0.0, 0.0])
    assert results[2]["ap"] == _approx(0.0)


@pytest.fixture
def val_boxes():
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return [
        {"result_type": "pred", "frame_id": "f001", 'boxes': [{'confs': np.array([0.98, 0.75, 0.7]), 'size': np.array([2, 2, 2]), 'rotation': rot, 'translation': np.array([6, 9, 1])}]},
        {"result_type": "pred", "frame_id": "f002", 'boxes': [{'confs': np.array([0.99, 0.91, 0.7]), 'size': np.array([4, 4, 2]), 'rotation': rot, 'translation': np.array([6, 3, 1])}]},
        {"result_type": "pred", "frame_id": "f003", 'boxes': [{'confs': np.array([0.95, 0.6, 0.51]), 'size': np.array([2, 2, 2]), 'rotation': rot, 'translation': np.array([1, 2, 1])}]},
        {"result_type": "gt", "frame_id": "f001", 'boxes': [{'confs': np.array([1, 1, 0]), 'size': np.array([2, 3, 2]), 'rotation': rot, 'translation': np.array([6.5, 9, 1])}]},
        {"result_type": "gt", "frame_id": "f002", 'boxes': [{'confs': np.array([1, 1, 0]), 'size': np.array([3, 3, 2]), 'rotation': rot, 'translation': np.array([5.5, 2.5, 1])}]},
        {"result_type": "gt", "frame_id": "f004", 'boxes': [{'confs': np.array([1, 1, 0]), 'size': np.array([2, 2, 2]), 'rotation': rot, 'translation': np.array([2, 3, 1])}]},
    ]


def test_planar_bbox3d_average_precision(val_boxes):
    metric = PlanarBbox3DAveragePrecision(
        "bbox3d_ap", 
        tensor_smith_cfg=dict(type='PlanarBbox3D', voxel_shape=(6, 320, 160), voxel_range=([-0.5, 2.5], [36, -12], [12, -12])),
        dictionary={"classes": ["passenger_car", "truck"]},
        max_conf_as_pred_class=True,
    )
    results = metric.compute_metrics(val_boxes)
    assert results["passenger_car_ap"] == _approx(0.6666667)
    assert results["truck_ap"] == 0
