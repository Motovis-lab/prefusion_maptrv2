from prefusion.registry import METRICS
from mmengine.evaluator import BaseMetric

__all__ = ["AccuracyPetr"]

@METRICS.register_module()
class AccuracyPetr(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, data_batch, data_samples):
        final_layer_all_cls_scores = [d for d in data_samples if d['name'] == "all_cls_scores"][0]['content'][-1]
        final_layer_all_bbox_preds = [d for d in data_samples if d['name'] == "all_bbox_preds"][0]['content'][-1]
        for logits, bbox, gt_items in zip(final_layer_all_cls_scores, final_layer_all_bbox_preds, data_batch):
            score = logits.sigmoid()
            pred_cls = score.argmax(dim=1)
            gt_cls = gt_items['meta_info']['bbox_3d']['classes']
    
        self.results.append({
            'batch_size': len(data_batch),
            'correct': len(data_batch),
        })

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)
