from typing import Dict, List, Any

import pandas as pd
from mmengine.evaluator import BaseMetric

from prefusion.registry import METRICS
from prefusion.dataset.utils import build_tensor_smith, unstack_batch_size


__all__ = ["PlanarBbox3DAveragePrecision", "PlanarSegIou"]


@METRICS.register_module()
class PlanarBbox3DAveragePrecision(BaseMetric):
    def __init__(self, transformable_name: str, tensor_smith_cfg: Dict):
        super().__init__(prefix=transformable_name)
        self.transformable_name = transformable_name
        self.tensor_smith = build_tensor_smith(tensor_smith_cfg)

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
            self.results.append({"result_type": "pred", "frame_id": _index_info.frame_id, 'boxes': _reversed_pred})
            self.results.append({"result_type": "gt", "frame_id": _index_info.frame_id, 'boxes': _reversed_gt})

    def compute_metrics(self, results):
        gt = [res for res in results if res['result_type'] == "gt"]
        pred = [res for res in results if res['result_type'] == "pred"]
        return self.calculate_ap(gt, pred)
    
    def calculate_ap(self, gt: List[Dict], pred: List[Dict]) -> Dict:
        return {}


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
