from typing import Dict, List, Tuple
from collections import namedtuple

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.ops import box3d_overlap
from mmengine.evaluator import BaseMetric
from copious.cv.geometry import Box3d
from scipy.spatial.transform import Rotation

from prefusion.registry import METRICS
from prefusion.dataset.utils import build_tensor_smith, unstack_batch_size, approx_equal


__all__ = ["PlanarBbox3DAveragePrecision", "PlanarSegIou"]

PredResult = namedtuple('PredResult', 'frame_id box_id cls_idx conf matched', defaults=[None, None, None])


def calculate_bbox3d_ap(
    gt: Dict[str, List[Dict]],
    pred: Dict[str, List[Dict]],
    iou_thresh: float = 0.5,
    num_confs: int = None,
    max_conf_as_pred_class: bool = True,
    is_first_conf_special: bool = True,
    allow_gt_reuse: bool = True,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], List[PredResult]]:
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

    num_confs : int, optional
        similar to num_classes, but has 1 more value. by default None

    max_conf_as_pred_class : bool, optional
        whether to use the max confidence as the predicted class, by default True

    is_first_conf_special : bool, optional
        whether the first confidence is special (i.e. denoting is_superclass), by default True
    
    allow_gt_reuse : bool, optional
        whether to allow a ground truth box to be reused (i.e. matched more than once with predictions), by default True

    Returns
    -------
    Tuple[Dict[int, Dict[str, torch.Tensor]], List[PredResult]]
        1. pred_confidences, precision, recall and ap for each class respectively (class name as the dict key)
        2. all_predictions (not sorted)
    """
    assert 0 < iou_thresh < 1, 'iou thresh should be a value between (0, 1)'
    num_confs = num_confs or torch.tensor([bx['confs'].tolist() for boxes in gt.values() for bx in boxes]).shape[1]
    
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
                    pred_res = PredResult(frame_id, box_id, cls_idx.item(), bx["confs"][cls_idx].item(), False)
                    predictions.append(pred_res)
                else:
                    predictions.extend([PredResult(frame_id, box_id, cls_idx.item(), bx["confs"][cls_idx].item(), False) for cls_idx in range(num_confs)])
            all_predictions.extend(predictions)
            continue

        pred_confs = torch.stack([bx["confs"] for bx in pred_of_frame])
        pred_corners = torch.stack([bx["corners"] for bx in pred_of_frame])
        gt_confs = torch.stack([bx["confs"] for bx in gt_of_frame])
        gt_corners = torch.stack([bx["corners"] for bx in gt_of_frame])

        _, ious = box3d_overlap(pred_corners, gt_corners, eps=1e-5)  # set eps to 1e-5 to prevent "ValueError: Planes have zero areas"

        if max_conf_as_pred_class:
            start_pos = int(is_first_conf_special)  # if is_first_conf_special==True, ignore the special conf
            pred_conf_max_idx = torch.argmax(pred_confs[:, start_pos:], dim=1) + start_pos  # if is_first_conf_special==True, idx should be increased by 1
            assert (pred_conf_max_idx == 0).sum() == 0
            matching_table = torch.zeros_like(ious)
            for i in range(ious.shape[0]):
                matched_gt_idx = torch.argmax(ious[i])
                pred_cls_idx = pred_conf_max_idx[i]
                pred_conf = pred_confs[i][pred_cls_idx]
                is_gt_the_same_class = approx_equal(gt_confs[matched_gt_idx][pred_cls_idx].item(), 1)
                iou_with_gt = ious[i][matched_gt_idx]
                if is_gt_the_same_class and iou_with_gt >= iou_thresh:
                    if allow_gt_reuse or not matching_table[i].any():
                        predictions.append(PredResult(frame_id, i, pred_cls_idx.item(), pred_conf.item(), True))
                        matching_table[i, matched_gt_idx] = 1
                else:
                    predictions.append(PredResult(frame_id, i, pred_cls_idx.item(), pred_conf.item(), False))
        else:
            for cls_idx in range(num_confs):
                matching_table = torch.zeros_like(ious)
                for i in range(ious.shape[0]):
                    matched_gt_idx = torch.argmax(ious[i])
                    pred_conf = pred_confs[i][cls_idx]
                    is_gt_the_same_class = approx_equal(gt_confs[matched_gt_idx][pred_cls_idx].item(), 1)
                    iou_with_gt = ious[i][matched_gt_idx]
                    if is_gt_the_same_class and iou_with_gt >= iou_thresh:
                        if allow_gt_reuse or not matching_table[i].any():
                            predictions.append(PredResult(frame_id, i, cls_idx.item(), pred_conf.item(), True))
                            matching_table[i, matched_gt_idx] = 1
                    else:
                        predictions.append(PredResult(frame_id, i, cls_idx.item(), pred_conf.item(), False))
        
        all_predictions.extend(predictions)

    cls_idxes = list(range(num_confs))
    if max_conf_as_pred_class:
        del cls_idxes[0]
    
    results = {}
    for cls_idx in cls_idxes:
        filtered_predictions = [p for p in all_predictions if p.cls_idx == cls_idx]
        sorted_predictions = sorted(filtered_predictions, key=lambda x: x.conf, reverse=True)
        num_gt_boxes = sum([sum(approx_equal(bx["confs"][cls_idx], 1) for bx in gt_of_frame) for gt_of_frame in gt.values()])
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
        ap = naive_ap(precision, recall)
        results[cls_idx] = {"predictions": sorted_predictions, "precision": precision, "recall": recall, "ap": ap}
    
    return results, all_predictions


def naive_ap(precisions, recalls):
    # Ensure the lists are of the same length
    if len(precisions) != len(recalls):
        raise ValueError("Precisions and recalls must have the same length")

    # Initialize variables
    average_precision = 0
    previous_recall = 0

    # Iterate through the precision-recall pairs
    for precision, recall in zip(precisions, recalls):
        # Calculate the width of the rectangle
        recall_difference = recall - previous_recall

        # Add the area of the rectangle to the average precision
        average_precision += precision * recall_difference

        # Update the previous recall
        previous_recall = recall

    return average_precision


@METRICS.register_module()
class PlanarBbox3DAveragePrecision(BaseMetric):
    def __init__(
        self, 
        transformable_name: str, 
        tensor_smith_cfg: Dict, 
        dictionary: Dict,
        iou_thresh: float = 0.5,
        num_confs: int = None,
        max_conf_as_pred_class: bool = True,
        is_first_conf_special: bool = True,
        allow_gt_reuse: bool = True,
    ):
        super().__init__(prefix=transformable_name)
        self.transformable_name = transformable_name
        self.tensor_smith = build_tensor_smith(tensor_smith_cfg)
        self.dictionary = dictionary
        self.ap_args = dict(
            iou_thresh=iou_thresh,
            num_confs=num_confs,
            max_conf_as_pred_class=max_conf_as_pred_class,
            is_first_conf_special=is_first_conf_special,
            allow_gt_reuse=allow_gt_reuse,
        )

    def process(self, data_batch, data_samples):
        gt = data_batch["annotations"].get(self.transformable_name)
        pred = [ds.get(self.transformable_name) for ds in data_samples if ds.get(self.transformable_name)][0]

        res = [{"frame_id": ii.frame_id} for ii in data_batch["index_infos"]]

        if gt:
            for i, _gt in enumerate(unstack_batch_size(gt)):
                _reversed_gt = self.tensor_smith.reverse(_gt)
                res[i]["gt"] = _reversed_gt
        
        if pred:
            for i, _pred in enumerate(unstack_batch_size(pred)):
                _pred["seg"] = _pred["seg"].sigmoid()
                _pred["cen"] = _pred["cen"].sigmoid()
                _reversed_pred = self.tensor_smith.reverse(_pred)
                res[i]["pred"] = _reversed_pred
        
        self.results.extend(res)

    def compute_metrics(self, results):
        def _convert(_bx):
            return {
                "confs": torch.tensor(_bx["confs"]).float(),
                "corners": torch.tensor(Box3d(_bx['translation'], _bx['size'], Rotation.from_matrix(_bx['rotation'])).corners).float()
            }

        gt = {res["frame_id"]: [_convert(bx) for bx in res["gt"]] for res in results if "gt" in res}
        pred = {res["frame_id"]: [_convert(bx) for bx in res["pred"]] for res in results if "pred" in res}
        
        if not gt or not pred:
            return {f"{cls_name}_ap": 0.0 for cls_name in self.dictionary["classes"]}
        
        ap_results, all_predictions = calculate_bbox3d_ap(gt, pred, **self.ap_args)
        class_names = ["any"] + self.dictionary["classes"]
        return {f"{class_names[cls_idx]}_ap": ap_res["ap"] for cls_idx, ap_res in ap_results.items()}


@METRICS.register_module()
class PlanarSegIou(BaseMetric):
    def __init__(self):
        super().__init__(prefix="segiou")

    def process(self, data_batch, data_samples):
        for trnsfmbl in data_samples:
            transformable_name, pred = list(trnsfmbl.items())[0]
            gt = data_batch["annotations"].get(transformable_name)

            if not gt or not pred:
                continue

            if "seg" not in gt or "seg" not in pred:
                continue

            for _gt, _pred in zip(gt["seg"], pred["seg"]):
                _pred_sigmoid = _pred[0].sigmoid()
                inter = (_pred_sigmoid * _gt[0]).sum() + 1
                union = (_pred_sigmoid + _gt[0] - _pred_sigmoid * _gt[0]).sum() + 1
                self.results.append({'transformable_name': transformable_name, 'intersection': inter.item(), 'union': union.item()})

    def compute_metrics(self, results):
        res_df = pd.DataFrame.from_records(results)
        metric_df = res_df.groupby(['transformable_name']).agg({'intersection': 'sum', 'union': 'sum'})
        metric_df.loc[:, "seg_iou"] = metric_df['intersection'] / metric_df['union']
        full_result = metric_df.reset_index()[['transformable_name', 'seg_iou']].to_dict('records')
        return {r['transformable_name']: r['seg_iou'] for r in full_result}


@METRICS.register_module()
class PlanarAverageDistance(BaseMetric):
    def __init__(self):
        super().__init__(prefix="average_distance")

    def process(self, data_batch, data_samples):
        pass

    def compute_metrics(self, results):
        pass


@METRICS.register_module()
class PlanarAverageAngle(BaseMetric):
    def __init__(self):
        super().__init__(prefix="average_angle")

    def process(self, data_batch, data_samples):
        pass

    def compute_metrics(self, results):
        pass


@METRICS.register_module()
class PlanarAverageSDF(BaseMetric):
    def __init__(self):
        super().__init__(prefix="average_sdf")

    def process(self, data_batch, data_samples):
        pass

    def compute_metrics(self, results):
        pass