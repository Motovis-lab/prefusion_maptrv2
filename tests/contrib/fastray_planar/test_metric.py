from typing import Dict, List
import pytest
import numpy as np

from contrib.fastray_planar.metric import PlanarBbox3DAveragePrecision, calculate_bbox3d_ap


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


@pytest.fixture
def pred():
    boxes = {
        "f001": [
            {"confs": [0.98, 0.75, 0.7], "corners": [[7, 10], [5, 10], [7, 8], [5, 8],]},
            {"confs": [0.97, 0.3, 0.35], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
            {"confs": [0.95, 0.61, 0.8], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
        ],
        "f002": [
            {"confs": [0.99, 0.91, 0.7], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
            {"confs": [0.98, 0.85, 0.3], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
            {"confs": [0.95, 0.6, 0.51], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
        ],
    }
    extend_bev_box_to_3d_(boxes)
    return boxes


@pytest.fixture
def gt():
    boxes = {
        "f001": [
            {"confs": [1, 1, 0], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
            {"confs": [1, 1, 0], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
            {"confs": [1, 1, 0], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
        ],
        "f002": [
            {"confs": [1, 1, 0], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
            {"confs": [1, 0, 1], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
            {"confs": [1, 1, 0], "corners": [[0, 0], [1, 1], [1, 1], [1, 1],]},
        ],
    }
    extend_bev_box_to_3d_(boxes)
    return boxes


def test_calculate_bbox3d_ap(pred, gt):
    results = calculate_bbox3d_ap(gt, pred, iou_thresh=0.5, max_conf_as_pred_class=True)



def test_planar_bbox3d_average_precision():
    pass


