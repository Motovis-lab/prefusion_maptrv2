from typing import Dict, List, Any, Tuple
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
from prefusion.dataset.utils import build_tensor_smith, unstack_batch_size


__all__ = ["PlanarBbox3DAveragePrecision", "PlanarSegIou"]


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


@METRICS.register_module()
class PlanarBbox3DAveragePrecision(BaseMetric):
    def __init__(self, transformable_name: str, tensor_smith_cfg: Dict, dictionary: Dict):
        super().__init__(prefix=transformable_name)
        self.transformable_name = transformable_name
        self.tensor_smith = build_tensor_smith(tensor_smith_cfg)
        self.dictionary = dictionary

    def process(self, data_batch, data_samples):
        gt = data_batch["annotations"].get(self.transformable_name)
        pred = [ds.get(self.transformable_name) for ds in data_samples if ds.get(self.transformable_name)][0]

        if not gt or not pred:
            return

        gt_unstacked = unstack_batch_size(gt)
        pred_unstacked = unstack_batch_size(pred)

        for _gt, _pred, _index_info in zip(gt_unstacked, pred_unstacked, data_batch["index_infos"]):
            _pred["seg"] = _pred["seg"].sigmoid()
            _pred["cen"] = _pred["cen"].sigmoid()
            _reversed_pred = self.tensor_smith.reverse(_pred)
            _reversed_gt = self.tensor_smith.reverse(_gt)
            if _reversed_pred:
                self.results.append({"result_type": "pred", "frame_id": _index_info.frame_id, 'boxes': _reversed_pred})
            if _reversed_gt:
                self.results.append({"result_type": "gt", "frame_id": _index_info.frame_id, 'boxes': _reversed_gt})

    def compute_metrics(self, results):
        def _get_corners(_bx):
            return torch.tensor(Box3d(_bx['translation'], _bx['size'], Rotation.from_matrix(_bx['rotation'])).corners).float()

        gt = {res["frame_id"]: [_get_corners(bx) for bx in res["boxes"]] for res in results if res['result_type'] == "gt"}
        pred = {res["frame_id"]: [_get_corners(bx) for bx in res["boxes"]] for res in results if res['result_type'] == "pred"}
        ap_results = calculate_bbox3d_ap(gt, pred)
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
