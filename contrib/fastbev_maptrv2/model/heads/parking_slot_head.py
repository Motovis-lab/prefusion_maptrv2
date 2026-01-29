import torch
import torch.nn as nn
from typing import List, Dict

class ParkingSlotHead(nn.Module):
    """
    将 MapTRv2 Decoder 输出 (Q,B,C) 转为点坐标与实例分类。
    约定：
      - Q = N * P（实例数 * 每实例点数）
      - reference_points: (B,Q,4) = (xc,yc,w,h) ∈ [0,1]
      - metas[b]["shape_info"] 包含 {"num_instances": N, "num_point_queries": P, "embed_dim": C}
      - valid_mask: (B,Q) —— True 表示该点存在；False 为 padding
    """
    def __init__(self, embed_dim: int, num_point_queries: int,
                 num_classes: int = 2, mode: str = "delta"):
        super().__init__()
        assert mode in ("delta", "abs")
        self.embed_dim = embed_dim
        self.num_point_queries = num_point_queries
        self.num_classes = num_classes
        self.mode = mode

        self.pt_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2)  # (dx,dy) 或 (x,y)
        )
        self.cls_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes)
        )
        self.tanh = nn.Tanh()

    def forward(self,
                decoder_out: torch.Tensor,        # (Q,B,C)
                reference_points: torch.Tensor,   # (B,Q,4)
                metas: List[Dict],
                valid_mask: torch.Tensor):        # (B,Q)
        Q, B, C = decoder_out.shape
        feat_BQC = decoder_out.permute(1, 0, 2).contiguous()     # (B,Q,C)

        raw_xy = self.pt_mlp(feat_BQC)                           # (B,Q,2)

        if self.mode == "delta":
            dxy = self.tanh(raw_xy)                              # (-1,1)
            xc = reference_points[..., 0:1]
            yc = reference_points[..., 1:2]
            w  = reference_points[..., 2:3]
            h  = reference_points[..., 3:4]
            x_pred = xc + 0.5 * w * dxy[..., 0:1]
            y_pred = yc + 0.5 * h * dxy[..., 1:2]
            pts_xy_BQ2 = torch.cat([x_pred, y_pred], dim=-1)     # (B,Q,2)
        else:
            pts_xy_BQ2 = torch.sigmoid(raw_xy)                   # (B,Q,2)

        preds_pts, preds_cls = [], []
        for b in range(B):
            N = metas[b]["shape_info"]["num_instances"]
            P = metas[b]["shape_info"]["num_point_queries"]
            assert N * P == Q, "collate 时需统一 N/P；保证 Q=N*P"

            pts_b = pts_xy_BQ2[b].view(N, P, 2)                  # (N,P,2)
            feat_b = feat_BQC[b].view(N, P, C)                   # (N,P,C)
            vmask_b = valid_mask[b].view(N, P)                   # (N,P)

            # 实例特征（有效点平均）
            inst_feat = (feat_b * vmask_b.unsqueeze(-1)).sum(dim=1) \
                        / vmask_b.sum(dim=1, keepdim=True).clamp(min=1.0)
            cls_logit = self.cls_mlp(inst_feat)                  # (N,num_classes)

            preds_pts.append(pts_b)
            preds_cls.append(cls_logit)

        points_xy = torch.stack(preds_pts, dim=0)                # (B,N,P,2)
        cls_logits = torch.stack(preds_cls, dim=0)               # (B,N,num_classes)

        return {
            "points_xy": points_xy,    # 归一化坐标
            "cls_logits": cls_logits,  # 实例分类
            "valid_mask": valid_mask   # (B,Q)
        }
