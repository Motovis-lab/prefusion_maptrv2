import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from mmengine.logging import MMLogger
import mmengine
import numpy as np
import pyquaternion
from torch import Tensor
from mmengine import Config, load
from mmengine.evaluator import BaseMetric
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.fileio import dump
from mmengine.logging import print_log
import logging
from scipy.spatial.transform import Rotation
from mmengine.structures import BaseDataElement
from .utils import boxes_iou3d_numba as boxes_iou3d
from collections import defaultdict
from prefusion.registry import METRICS


@METRICS.register_module()
class Box3DMetric(BaseMetric):
    def __init__(self, 
                 available_range = None,
                 available_class = [],
                 available_branch=None,
                 iou_thresholds=[0.25, 0.5, 0.75],
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu', 
                 collect_dir: str | None = None) -> None:
        super().__init__(collect_device, prefix, collect_dir) 
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        if available_range is not None:
            self.available = available_range
        else:
            available_range = []
        self.available_class = available_class
        self.available_branch = available_branch
        self.modality = modality
        self.format_only = format_only
        self.iou_thresholds = iou_thresholds
        if self.format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.gt_results = []

    def process(self, data_batch, data_samples):
        """
            each result is frame results 
        """
        for ori_data, det in zip(data_batch, data_samples):
            sample_idx = ori_data['frame_id']
            result = dict()
            boxes, scores, labels = det
            annos = list()
            for i, box in enumerate(boxes):
                name = self.available_class[labels[i]]
                anno = dict(
                    sample_token=sample_idx,
                    detection_box=box,
                    detection_name=name,
                    detection_score=float(scores[i]),
                )
                annos.append(anno)
            result[sample_idx] = annos
            self.results.append(result)
            
            gt_result = dict()
            gt_annos = list()
            for branch in self.available_branch:
                for instance in ori_data['transformables'][branch].data['elements']:
                    if instance['class'] in self.available_class:
                        if self.available[0]<instance['translation'].reshape(3)[0]<self.available[1] and self.available[2]<instance['translation'].reshape(3)[1]<self.available[3]:
                            name = instance['class']
                            box = np.array([*instance['translation'].tolist(), *instance['size'], Rotation.from_matrix(instance['rotation']).as_euler("xyz", degrees=False).tolist()[-1], *instance['velocity'].tolist()[:2]])
                            anno = dict(
                                sample_token=sample_idx,
                                detection_box=box,
                                detection_name=name,
                                detection_score=1.,
                            )
                            gt_annos.append(anno)
            gt_result[sample_idx] = gt_annos
            self.gt_results.append(gt_result)
    
    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
            gt_results = collect_results(
                self.gt_results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)
            gt_results = collect_results(self.gt_results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            gt_results = _to_cpu(gt_results)
            _metrics = self.compute_metrics(results, gt_results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        self.gt_results.clear()
        return metrics[0]

    def compute_metrics(self, results: list, gt_results=None) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        
        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict
    
        class_detections = defaultdict(list)
        class_groundtruths = defaultdict(list)

        for pred, gt in zip(results, gt_results):
            gt_annos = list(gt.values())[0]
            pred_annos = list(pred.values())[0]
            for gt_anno in gt_annos:
                class_groundtruths[gt_anno['detection_name']].append(gt_anno)
            for pred_anno in pred_annos:
                class_detections[pred_anno['detection_name']].append(pred_anno)

        mAPs, all_aps = self.calculate_map_ap_multi_frame(class_groundtruths, class_detections, iou_thresholds=self.iou_thresholds)
        max_class_length = max(len(class_name) for aps in all_aps.values() for class_name in aps.keys())
        formatted_mAPs = "\n" + "\n".join([f"    {iou}: {mAP:.2f}" for iou, mAP in mAPs.items()]) + "\n"

        formatted_all_aps = "\n" + "\n".join([
            f"    {iou}: \n" + 
            "\n".join([f"        {class_name.ljust(max_class_length)}: {ap:.2f}" for class_name, ap in aps.items()]) + 
            "\n    " for iou, aps in all_aps.items()
        ]) + "\n"
        metric_dict["\n" + "mAPs"] = formatted_mAPs
        metric_dict["\n" + "all_aps"] = formatted_all_aps

        return metric_dict

    
    def calculate_ap(self, recalls, precisions):
        # use all point to compute
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # compute AP
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        return ap

    def calculate_map_ap_multi_frame(self, all_frames_gt, all_frames_pred, iou_thresholds=[0.5, 0.75]):
        results = {iou_threshold: {} for iou_threshold in iou_thresholds}

        for class_name in set(list(all_frames_gt.keys()) + list(all_frames_pred.keys())):
            gt_boxes = all_frames_gt[class_name]
            pred_boxes = all_frames_pred[class_name]
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                for iou_threshold in iou_thresholds:
                    results[iou_threshold][class_name] = 0.0
                continue
            
            gt_boxes_array = np.stack([box['detection_box'][:7] 
                                                             for box in gt_boxes], axis=0)
            pred_boxes_array = np.stack([box['detection_box'][:7]
                                                               for box in pred_boxes], axis=0)

            ious = boxes_iou3d(pred_boxes_array, gt_boxes_array)

            confidences = np.array([box['detection_score'] for box in pred_boxes])
            sorted_indices = np.argsort(-confidences)

            for iou_threshold in iou_thresholds:
                tp = np.zeros(len(pred_boxes))
                fp = np.zeros(len(pred_boxes))
                gt_matched = np.zeros(len(gt_boxes), dtype=bool)

                for i in sorted_indices:
                    if np.any(ious[i] >= iou_threshold):
                        max_iou_idx = np.argmax(ious[i])
                        if not gt_matched[max_iou_idx]:
                            tp[i] = 1
                            gt_matched[max_iou_idx] = True
                        else:
                            fp[i] = 1
                    else:
                        fp[i] = 1

                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)

                recalls = tp_cumsum / len(gt_boxes)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

                ap = self.calculate_ap(recalls, precisions)
                results[iou_threshold][class_name] = ap

        mAPs = {iou_threshold: np.mean(list(aps.values())) for iou_threshold, aps in results.items()}

        return mAPs, results

def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data