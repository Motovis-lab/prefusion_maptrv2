from contrib.petr import FocalHead
from mmengine.registry import MODELS
import torch
import random
from contrib.petr.misc import apply_center_offset, apply_ltrb


__all__ = ['RayFocalHead']

@MODELS.register_module()
class RayFocalHead(FocalHead):
    """Ray Focal Head for SPETR.

    Args:
        num_propagated (int): Number of propagated frames.
        with_ego_pos (bool): Whether to use ego position.
    """
    
    def __init__(self, *args, **kwargs):
        super(RayFocalHead, self).__init__(*args, **kwargs)

    def forward(self, location: torch.Tensor, img_feats: torch.Tensor):
        """
        Parameters
        ----------
        location : torch.Tensor
            of shape 
        img_feats : torch.Tensor
            of shape (B, N, C, H, W)

        Returns
        -------
        _type_
            _description_
        """
        bs, n, c, h, w = img_feats.shape
        num_tokens = n * h * w

        # focal sampling
        if self.training:
            if self.use_hybrid_tokens:
                sample_ratio = random.uniform(0.2, 1.0)
            else:
                sample_ratio = self.train_ratio
            num_sample_tokens = int(num_tokens * sample_ratio)

        else:
            sample_ratio = self.infer_ratio
            num_sample_tokens = int(num_tokens * sample_ratio)

        x = img_feats.flatten(0, 1)
        cls_feat = self.shared_cls(x)
        cls = self.cls(cls_feat)
        centerness = self.centerness(cls_feat)
        cls_logits = cls.permute(0, 2, 3, 1).reshape(bs * n, -1, self.num_classes)
        centerness = centerness.permute(0, 2, 3, 1).reshape(bs * n, -1, 1)
        pred_bboxes = None
        pred_centers2d = None

        reg_feat = self.shared_reg(x)
        ltrb = self.ltrb(reg_feat).permute(0, 2, 3, 1).contiguous()  # ltrb: Left, Top, Right, Bottom
        ltrb = ltrb.sigmoid()
        centers2d_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()

        centers2d = apply_center_offset(location, centers2d_offset)
        bboxes = apply_ltrb(location, ltrb)

        pred_bboxes = bboxes.view(bs * n, -1, 4)
        pred_centers2d = centers2d.view(bs * n, -1, 2)

        cls_score = cls_logits.topk(1, dim=2).values[..., 0].view(bs, -1, 1)

        sample_weight = cls_score.detach().sigmoid() * centerness.detach().view(bs, -1, 1).sigmoid()

        _, topk_indexes = torch.topk(sample_weight, num_sample_tokens, dim=1)
        
        outs = {
            "enc_cls_scores": cls_logits,
            "enc_bbox_preds": pred_bboxes,
            "pred_centers2d": pred_centers2d,
            "centerness": centerness,
            "topk_indexes": topk_indexes,
            "sample_weight": sample_weight
        }

        return outs