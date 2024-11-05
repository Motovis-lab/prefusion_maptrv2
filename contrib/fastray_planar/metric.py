from typing import Optional, Dict, List, Any

from mmengine.evaluator import BaseMetric

from prefusion.registry import METRICS, TENSOR_SMITHS
from prefusion.dataset.utils import build_tensor_smith, unstack_batch_size


__all__ = ["PlanarBbox3DAveragePrecision", "PlanarSegIou"]


@METRICS.register_module()
class PlanarBbox3DAveragePrecision(BaseMetric):
    def __init__(self, transformable_name: str, tensor_smith_cfg: Dict):
        super().__init__(prefix=f"{transformable_name}_segiou")
        self.transformable_name = transformable_name
        self.tensor_smith = build_tensor_smith(tensor_smith_cfg)
        self.gt_results: List[Any] = []

    def process(self, data_batch, data_samples):
        gt = data_batch["annotations"].get(self.transformable_name)
        pred = [ds.get(self.transformable_name) for ds in data_samples if ds.get(self.transformable_name)][0]

        if not gt or not pred:
            self.results.append({'intersection': 0, 'union': 1e-6})
            return

        gt_unstacked = unstack_batch_size(gt)
        pred_unstacked = unstack_batch_size(pred)

        for _gt, _pred, _index_info in zip(gt_unstacked, pred_unstacked, data_batch["index_info"]):
            _reversed_pred = self.tensor_smith.reverse(_pred)
            self.results.append({"frame_id": _index_info.frame_id, 'boxes': _reversed_pred})
            _reversed_gt = self.tensor_smith.reverse(_gt)
            self.gt_results.append({"frame_id": _index_info.frame_id, 'boxes': _reversed_gt})

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)


@METRICS.register_module()
class PlanarSegIou(BaseMetric):
    def __init__(self, transformable_name: str, tensor_smith_cfg: Dict):
        super().__init__(prefix=f"{transformable_name}_segiou")
        self.transformable_name = transformable_name
        self.tensor_smith = build_tensor_smith(tensor_smith_cfg)

    def process(self, data_batch, data_samples):
        gt = data_batch["annotations"].get(self.transformable_name)
        pred = [ds.get(self.transformable_name) for ds in data_samples if ds.get(self.transformable_name)][0]

        if not gt or not pred:
            self.results.append({'intersection': 0, 'union': 1e-6})
            return

        for _gt, _pred in zip(gt["seg"], pred["seg"]):
            _pred_sigmoid = _pred[0].sigmoid()
            inter = (_pred_sigmoid * _gt[0]).sum() + 1
            union = (_pred_sigmoid + _gt[0] - _pred_sigmoid * _gt[0]).sum() + 1
            self.results.append({'intersection': inter.item(), 'union': union.item()})

    def compute_metrics(self, results):
        total_inter = sum(r['intersection'] for r in results)
        total_union = sum(r['union'] for r in results)
        return dict(seg_iou=total_inter/total_union)
