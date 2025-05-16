from typing import List, Dict, Any

import torch
import numpy as np
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.structures import BaseDataElement
from mmdet3d.structures.ops.transforms import bbox3d2result

from prefusion import BaseModel
from prefusion.registry import MODELS
from contrib.petr.misc import locations

__all__ = ["StreamPETR"]


@MODELS.register_module()
class FrameBatchMerger(BaseDataPreprocessor):
    def __init__(self, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self._device = device

    def forward(self, data: List[Dict[str, Any]], training: bool = False) -> Dict[str, List[Any]]:
        merged = {}
        for key in data[0].keys():
            merged[key] = [self._cast_data(i[key]) for i in data]
        return merged

    def _cast_data(self, data: Any):
        if isinstance(data, torch.Tensor):
            _dtype = torch.float32 if "float" in str(data.dtype) else data.dtype
            return data.to(dtype=_dtype, device=self._device)
        if isinstance(data, dict):
            return {k: self._cast_data(data[k]) for k in data}
        return data
        # return self.cast_data(merged)  # type: ignore


@MODELS.register_module()
class StreamPETR(BaseModel):
    def __init__(
        self,
        *,
        data_preprocessor=None,
        img_backbone=None,
        img_neck=None,
        roi_head=None,
        box_head=None,
        stride=16,
        position_level=0,
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        data_preprocessor : _type_, optional
            _description_, by default None
        img_backbone : _type_, optional
            _description_, by default None
        img_neck : _type_, optional
            _description_, by default None
        box_head : _type_, optional
            _description_, by default None
        stride : int, optional
            _description_, by default 16
        position_level : int, optional
            用于选择 FPN 结果中的哪一个粒度的特征来做后续的 bbox 预测, by default 0
        """
        super().__init__(data_preprocessor=data_preprocessor)
        assert not any(m is None for m in [img_backbone, img_neck])
        self.img_backbone = MODELS.build(img_backbone)
        self.box_head = MODELS.build(box_head)
        self.roi_head = MODELS.build(roi_head) if roi_head else None
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.stride = stride
        self.position_level = position_level

    def forward(self, *, index_info=None, camera_images=None, bbox_3d=None, ego_poses=None, meta_info=None, mode="loss", **kwargs):
        B, (N, C, H, W) = len(camera_images), camera_images[0].shape
        camera_images = torch.vstack([i.unsqueeze(0) for i in camera_images]).reshape(B * N, C, H, W)
        im_size = camera_images.shape[-2:][::-1]
        img_feats = self.extract_img_feat(camera_images)

        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)

        img_feats = img_feats[self.position_level]
        img_feats = img_feats.reshape(B, N, *img_feats.shape[1:])
        location = self.prepare_location(img_feats, im_size)

        if self.roi_head:
            outs_roi = self.roi_head(location, img_feats)
        else:
            outs_roi = {'topk_indexes': None}

        topk_indexes = outs_roi['topk_indexes']
        _device = img_feats.device
        
        data = {
            "timestamp": torch.tensor([int(ii.frame_id) / 1000 for ii in index_info], device=_device, dtype=torch.float64),
            "prev_exists": torch.tensor([ii.prev is not None for ii in index_info], device=_device, dtype=torch.float32),
            "ego_pose": torch.tensor(np.array([p.transformables['0'].trans_mat for p in ego_poses]), device=_device, dtype=torch.float32),
            "ego_pose_inv": torch.tensor(np.array([np.linalg.inv(p.transformables['0'].trans_mat) for p in ego_poses]), device=_device, dtype=torch.float32),
            "intrinsics": torch.tensor(np.array([m["camera_images"]["intrinsic"] for m in meta_info]), device=_device, dtype=torch.float32),
            "ego2img": torch.tensor(np.array([m["camera_images"]["extrinsic_inv"] for m in meta_info]), device=_device, dtype=torch.float32),
            "lidar2img": torch.tensor(np.array([np.linalg.inv(np.linalg.inv(m['T_ego_lidar']) @ np.array(m['camera_images']['extrinsic'])) for m in meta_info]), device=_device, dtype=torch.float32),
        }

        img_metas = []
        for m in meta_info:
            img_metas.append({
                "pad_shape": [(im_size[1], im_size[0], 3)] * N,
            })
        gt_labels = [m['bbox_3d']['classes'] for m in meta_info]
        outs = self.box_head(img_feats, location, img_metas, bbox_3d, gt_labels, topk_indexes=topk_indexes, **data)
        outs = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in outs.items()}   # convert torch.float16 to torch.float32

        # self.visualize_bbox3d(data, bbox_3d, outs, meta_info, ego_poses)

        if mode == 'tensor':
            bbox_list = self.box_head.get_bboxes(outs, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return bbox_results
        
        if mode == "loss":
            loss_inputs = [bbox_3d, gt_labels, outs]
            losses = self.box_head.loss(*loss_inputs)
            # if self.with_img_roi_head:
            #     loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
            #     losses2d = self.img_roi_head.loss(*loss2d_inputs)
            #     losses.update(losses2d)

            return losses
        if mode == "predict":
            loss_inputs = [bbox_3d, gt_labels, outs]
            losses = self.box_head.loss(*loss_inputs)
            return (
                *[{"name": k, "content": v.cpu() if isinstance(v, torch.Tensor) else v} for k, v in outs.items()],
                BaseDataElement(loss=losses),
            )

    def visualize_bbox3d(self, data, bbox_3d, outs, meta_info, ego_poses, *args, **kwargs):
        ###########################
        # FIXME: Visualize Images
        ###########################
        # import matplotlib.pyplot as plt
        # import cv2
        # nrows, ncols = 4, 3
        # for ts, images, m in zip(data['timestamp'], camera_images.reshape(B, N, C, H, W), meta_info):
        #     fig, ax = plt.subplots(nrows, ncols, figsize=(18, 18))
        #     for i, im in enumerate(images):
        #         restored_im = im.cpu().numpy().transpose(1, 2, 0) * np.array([58.395, 57.120, 57.375]) + np.array([123.675, 116.280, 103.530])
        #         ax[i // ncols][i % ncols].imshow(restored_im.astype(np.uint8)[..., ::-1])
        #         cam_id = m['camera_images']['camera_ids'][i]
        #         rim = cv2.imread(f"/data/datasets/mv4d/20231101_160337/vcamera/{cam_id}/{ts.long().item()}.jpg")
        #         ax[2 + i // ncols][i % ncols].imshow(rim[..., ::-1])
        #     plt.savefig(f"./vis/{ts.item()}.png")
        #     plt.close()
        # a = 100

        ###########################
        # FIXME: Visualize BBoxes
        ###########################
        def _draw_rect(p0, p1, p5, p4, linewidth=1, color='r', alpha=1):
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=linewidth, color=color, alpha=alpha)
            plt.plot((p1[0], p5[0]), (p1[1], p5[1]), linewidth=linewidth, color=color, alpha=alpha)
            plt.plot((p5[0], p4[0]), (p5[1], p4[1]), linewidth=linewidth, color=color, alpha=alpha)
            plt.plot((p4[0], p0[0]), (p4[1], p0[1]), linewidth=linewidth, color=color, alpha=alpha)

        def _draw_boxes(_boxes, color='blue', alpha=0.3):
            for bx in _boxes:
                l, w, h = bx[3:6].tolist()
                translation = bx[:2].detach().cpu().numpy()
                yaw = bx[6].item()
                corners = np.array([
                    [ l / 2, -w / 2],
                    [ l / 2, +w / 2],
                    [-l / 2, +w / 2],
                    [-l / 2, -w / 2],
                ])
                rotmat = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
                corners = corners @ rotmat.T + translation
                _draw_rect(*corners.tolist(), color=color, alpha=alpha)

        import matplotlib.pyplot as plt
        import math
        from contrib.petr.misc import denormalize_bbox
        for ts, gt_boxes, pred_bboxes, pred_scores, m, ep in zip(data['timestamp'], bbox_3d, outs['all_bbox_preds'][-1], outs['all_cls_scores'][-1], meta_info, ego_poses):
            # if int(ts.item()) not in [1698825828064, 1698825829564, 1698825837064, 1698825846064, 1698825852564]:
            #     continue
            _ = plt.figure()
            _draw_boxes(gt_boxes, color='blue')
            denormalized_bbox = denormalize_bbox(pred_bboxes, None)
            sigmoid_scores = pred_scores.sigmoid()
            sorted_scores, sorted_index = sigmoid_scores.max(dim=1)[0].topk(100)
            _index_gt35 = sorted_index[sorted_scores >= 0.35]
            _index_btw25_35 = sorted_index[(sorted_scores >= 0.25) & (sorted_scores < 0.35)]
            _index_btw20_25 = sorted_index[(sorted_scores >= 0.2) & (sorted_scores < 0.25)]
            _draw_boxes(denormalized_bbox[_index_gt35], color="red", alpha=0.5)
            _draw_boxes(denormalized_bbox[_index_btw25_35], color="yellow", alpha=0.3)
            _draw_boxes(denormalized_bbox[_index_btw20_25], color="gray", alpha=0.1)
            
            plt.plot([0, 1], [0, 0], color="r", marker='.')
            plt.plot([0, 0], [0, 1], color="green", marker='.')
            plt.scatter([0], [0], color="black", marker='o')
            
            plt.gca().set_aspect('equal')
            plt.savefig(f"./vis/{ts.item()}.png")
            plt.close()
        a = 100        

    def extract_img_feat(self, img: torch.Tensor):
        """
        Parameters
        ----------
        img : torch.Tensor
            of shape (N, C, H, W), where N is the batch size, C is usually 3, H and W are image height and width.
        """
        return self.img_backbone(img)

    def prepare_location(self, img_feats, im_size):
        pad_w, pad_h = im_size
        batch_size, num_cameras = img_feats.shape[:2]
        location = locations(img_feats.flatten(0, 1), self.stride, pad_h, pad_w)[None].repeat(batch_size * num_cameras, 1, 1, 1)
        return location
