from mmdet.evaluation import VOCMetric
from prefusion.registry import METRICS
import torch
from typing import Sequence, Optional, Union, List
import datetime
import random
import cv2
import os
from PIL import Image, ImageDraw, ImageFont


__all__ = ['FusionDetMetric']


@METRICS.register_module()
class FusionDetMetric(VOCMetric):
    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None, 
                 random_show: Optional[float] = 0.,
                 save_dir: Optional[str] = './work_dirs/eval_show',
                 ) -> None:
        self.random_show = random_show
        if self.random_show > 0:
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir    
        super().__init__(
            iou_thrs=iou_thrs,
            scale_ranges=scale_ranges,
            metric=metric,
            proposal_nums=proposal_nums,
            eval_mode=eval_mode,
            collect_device=collect_device,
            prefix=prefix)
        
    def process(self, data_batch, data_samples):
        super().process(data_batch, data_samples)
        for ind, (ann, pred) in enumerate(self.results):
            if random.random() < self.random_show:
                img = data_batch['inputs'][ind].cpu().detach().numpy().transpose(1, 2, 0)[..., ::-1]
                img = Image.fromarray(img.astype('uint8'))
                draw = ImageDraw.Draw(img)
                gt_bboxes = ann['bboxes'].tolist()
                gt_labels = ann['labels'].tolist()
                for b, l in zip(gt_bboxes, gt_labels):
                    draw.rectangle(b, outline=(255, 0, 0), width=2)
                    draw.text((b[0], b[1]-10), str(l), fill=(255, 0, 0))
                
                pred_bboxes = []
                pred_labels = []
                pred_scores = []
                for i in range(len(pred)):
                    tmp = pred[i][:, :-1].tolist()
                    pred_bboxes.extend(tmp)
                    pred_labels.extend([i] * len(tmp))
                    pred_scores.extend(pred[i][:, -1].tolist())
                for b, l, s in zip(pred_bboxes, pred_labels, pred_scores):
                    draw.rectangle(b, outline=(0, 255, 0), width=2)
                    draw.text((b[0], b[1]-10), f"{l}: {s:.2f}", fill=(0, 255, 0))
                filename = f"{self.save_dir}/{data_samples[ind]['img_id'].split('/')[-1]}.jpg"
                img.save(filename)
    
    def draw(self, img, bboxes, labels, scores=None, color=None):
        if scores is None:
            scores = [1.0] * len(bboxes)
        if color is None:
            color = (255, 0, 0)
        img = img.copy()
        for bbox, label, score in zip(bboxes, labels, scores):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{label}"
            text += f": {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 5 if y1 - 5 > 10 else y1 + 15
            cv2.rectangle(img, (text_x, text_y - text_size[1] - 2), 
                  (text_x + text_size[0], text_y + 2), color, -1)
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        return img