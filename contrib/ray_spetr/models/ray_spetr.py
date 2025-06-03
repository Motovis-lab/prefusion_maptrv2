from contrib.petr import StreamPETR
from prefusion.registry import MODELS
import torch    
from mmdet3d.structures.ops.transforms import bbox3d2result
from mmengine.structures import BaseDataElement
import numpy as np


__all__ = ['RaySPETR']

@MODELS.register_module()
class RaySPETR(StreamPETR):
    def __init__(self, *args, **kwargs):
        super(RaySPETR, self).__init__(*args, **kwargs)

    def forward(self, *, index_info=None, camera_images=None, bbox_3d=None, bbox_2d=None, bbox_center_2d=None, ego_poses=None, meta_info=None, mode="loss", **kwargs):
        B, (N, C, H, W) = len(camera_images), camera_images[0].shape
        camera_images = torch.vstack([i.unsqueeze(0) for i in camera_images]).reshape(B * N, C, H, W)

        # FIXME: Visualize 2D boxes
        # import cv2
        # import matplotlib.pyplot as plt
        # im = (camera_images[3].cpu().numpy().transpose(1, 2, 0) * np.array([58.395, 57.120, 57.375]) + np.array([123.675, 116.280, 103.530])).astype(np.uint8)
        # im = np.ascontiguousarray(im)
        # bboxes = bbox_2d[0][3]
        # for bx in bboxes:
        #     cv2.rectangle(im, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 0), 2)
        # plt.imshow(im)
        # plt.savefig("2dbbox_vis.png")
        # plt.close()
        # FIXME

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
        gt_bboxes_3d = [b.reshape(0, 9) if len(b) == 0 else b for b in bbox_3d]  # 9 is hard coded here for: (x,y,z,l,w,h,yaw,vx,vy)
        try:
            outs = self.box_head(img_feats, location, img_metas, gt_bboxes_3d, gt_labels, topk_indexes=topk_indexes, centerness=outs_roi['sample_weight'], **data)
        except Exception as e:
            from loguru import logger
            logger.error(f"index_info: {index_info}")
            print(e)
            raise
        outs = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in outs.items()}   # convert torch.float16 to torch.float32

        # self.visualize_bbox3d(data, gt_bboxes_3d, outs, meta_info, ego_poses)

        if mode == 'tensor':
            bbox_list = self.box_head.get_bboxes(outs, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            return bbox_results
        
        if mode == "loss":
            loss_inputs = [gt_bboxes_3d, gt_labels, outs]
            try:
                losses = self.box_head.loss(*loss_inputs)
            except Exception as e:
                from loguru import logger
                logger.error(f"index_info: {index_info}")
                print(e)
                raise
            if self.roi_head:
                gt_bboxes = [[boxes.to(device=_device, dtype=torch.float32) for boxes in batch]for batch in bbox_2d]
                gt_labels = [[cls.to(device=_device, dtype=torch.int64) for cls in batch['bbox_2d']['classes']] for batch in meta_info]
                centers2d = [[cnts.to(device=_device, dtype=torch.float32) for cnts in batch]for batch in bbox_center_2d]
                dummy_depths = [[torch.tensor([], device=_device, dtype=torch.float32) for _ in m['bbox_2d']['classes']] for m in meta_info]
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, dummy_depths, outs_roi, img_metas]
                losses2d = self.roi_head.loss(*loss2d_inputs)
                losses.update(losses2d)

            return losses
        if mode == "predict":
            loss_inputs = [gt_bboxes_3d, gt_labels, outs]
            losses = self.box_head.loss(*loss_inputs)
            return (
                *[{"name": k, "content": v.cpu() if isinstance(v, torch.Tensor) else v} for k, v in outs.items()],
                BaseDataElement(loss=losses),
            )