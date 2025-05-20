import json
from pathlib import Path
from typing import Optional, Sequence, Union, List, Dict, TYPE_CHECKING
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R
from mmengine.hooks.hook import Hook
from mmengine.logging import print_log
from copious.data_structure.dict import defaultdict2dict
from copious.cv.geometry import rt2mat

from prefusion.registry import HOOKS


if TYPE_CHECKING:
    from prefusion.dataset.transform import EgoPose


DATA_BATCH = Optional[Union[dict, tuple, list]]

__all__ = ["DumpPETRDetectionAsNuscenesJsonHook"]


DEFAULT_ATTR = {
    'car': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.moving',
    'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'barrier': '',
    'traffic_cone': '',
}

OBJ_RANGE_THRESH = {
    'car': 50, 
    'truck': 50, 
    'bus': 50, 
    'trailer': 50, 
    'construction_vehicle': 50, 
    'pedestrian': 40, 
    'motorcycle': 40, 
    'bicycle': 40, 
    'traffic_cone': 30, 
    'barrier': 30
}


def format_boxes_as_nuscenes_format(
    token: str,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_class_ids: np.ndarray,
    ego_pose: "EgoPose",
    dictionary: list[str]
) -> List[Dict]:
    """Format predicted boxes as the format that Nuscenes Evaluator expcected in the global (world) coordsys.

    Parameters
    ----------
    token : str
        nuscenes sample token
    pred_boxes : np.ndarray
        of shape (N, 9)
    pred_scores : np.ndarray
        of shape (N, 1)
    pred_class_ids : np.ndarray
        of shape (N, 1)
    ego_pose : EgoPose
        The ego pose (pls. refer to StreamPETRModelFeeder.convert_ego_pose_set_to_lidar_coordsys_)
    T_ego_lidar : np.ndarray
        of shape (4, 4), the transformation matrix from lidar coordsys to ego coordsys
    dictionary : list[str]
        _description_

    Returns
    -------
    List[Dict]
        formatted boxes information in the global (world) coordsys
    """
    formatted_boxes = []
    for bx, sc, lb in zip(pred_boxes, pred_scores, pred_class_ids):
        dist_to_ego_origin = np.linalg.norm(bx[:2], 2)
        det_name = dictionary[lb]
        if dist_to_ego_origin > OBJ_RANGE_THRESH[det_name]:
            continue

        formatted_bx = {
            "sample_token": token,
            "size": bx[[4, 3, 5]].tolist(), # size in nusc is [width, length, height]
            "detection_name": det_name,
            "detection_score": float(sc),
            "attribute_name": get_box_attr(bx[-2:], det_name),
        }

        # convert box location to world coord sys
        T_e_b = rt2mat(R.from_euler("Z", [bx[6]], degrees=False).as_matrix(), bx[:3], as_homo=True)
        T_w_e = rt2mat(ego_pose.rotation, ego_pose.translation, as_homo=True)
        T_w_b = T_w_e @ T_e_b
        formatted_bx.update(
            translation=T_w_b[:3, 3].flatten().tolist(),
            rotation=R.from_matrix(T_w_b[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist(), # nusc uses pyquaternion repr
            velocity=(T_w_e[:3, :3] @ np.array(bx[-2:].tolist() + [0])[:, None]).flatten()[:2].tolist(),
        )
        formatted_boxes.append(formatted_bx)
    return formatted_boxes


def get_box_attr(velo: Union[Sequence, np.ndarray], det_name: str, ):
    if np.sqrt(velo[0]**2 + velo[1]**2) > 0.2:
        if det_name in [
                'car',
                'construction_vehicle',
                'bus',
                'truck',
                'trailer',
        ]:
            attr = 'vehicle.moving'
        elif det_name in ['bicycle', 'motorcycle']:
            attr = 'cycle.with_rider'
        else:
            attr = DEFAULT_ATTR[det_name]
    else:
        if det_name in ['pedestrian']:
            attr = 'pedestrian.standing'
        elif det_name in ['bus']:
            attr = 'vehicle.stopped'
        else:
            attr = DEFAULT_ATTR[det_name]
    return attr


@HOOKS.register_module()
class DumpPETRDetectionAsNuscenesJsonHook(Hook):
    def __init__(
        self,
        det_anno_transformable_keys: List[str],
        pre_conf_thresh: float = 0.3,
    ):
        super().__init__()
        self.pre_conf_thresh = pre_conf_thresh
        self.transformable_keys = det_anno_transformable_keys
        self.results = defaultdict(list)

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Union[dict, Sequence]] = None,
        mode: str = "test",
    ) -> None:
        for idx, (gt_items, pred) in enumerate(zip(data_batch, outputs)):
            token = gt_items['sample_token'].value
            dictionary = gt_items['dictionary']['classes']
            ego_pose = gt_items['ego_poses'].transformables["0"]
            gt_bboxes = gt_items['bbox_3d']
            pred_boxes = pred["bboxes_3d"].numpy()
            pred_scores = pred["scores_3d"].numpy()
            pred_labels = pred["labels_3d"].numpy()
            nusc_fmt_boxes = format_boxes_as_nuscenes_format(token, pred_boxes, pred_scores, pred_labels, ego_pose, dictionary)
            self.results[token].extend(nusc_fmt_boxes)

    def after_test_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        json_save_path = Path(runner.cfg["work_dir"]) / "nusc_det_results.json"
        with open(json_save_path, "w") as f:
            json.dump(
                {
                    "meta": {
                        "use_camera": True,
                        "use_lidar": False,
                        "use_radar": False,
                        "use_map": False,
                        "use_external": False,
                    },
                    "results": defaultdict2dict(self.results),
                },
                f,
            )
        print_log(f"Detection results has been saved to {json_save_path}")