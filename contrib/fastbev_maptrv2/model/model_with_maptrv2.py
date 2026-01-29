# prefusion/contrib/fastbev_maptrv2/model/decoders/maptrv2_adapter.py
import torch
import torch.nn as nn
from typing import Optional, Tuple


class SimpleMapTRv2Adapter(nn.Module):
    """
    占位版 MapTRv2 Decoder：
    - 现在只是用 MLP 简单融合一下 BEV feature + query embedding
    - 目的是先跑通全流程，后面你可以换成真正的 transformer decoder
    """

    def __init__(self, bev_channels: int, embed_dim: int):
        super().__init__()
        # 将 BEV 全局 pool 成一个向量，然后映射到 embed_dim，再加到 query 上
        self.bev_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,C,H,W)->(B,C,1,1)
            nn.Flatten(1),            # (B,C)
            nn.Linear(bev_channels, embed_dim),
        )

    def forward(
        self,
        bev_feat: torch.Tensor,           # (B, C_bev, H, W)
        query: torch.Tensor,              # (Q, B, C_embed)
        reference_points: torch.Tensor,   # (B, Q, 4)   # 暂时没用上
        valid_mask: Optional[torch.Tensor] = None,  # (B, Q)
    ) -> torch.Tensor:
        """
        返回:
            token_feats: (B, Q, C_embed)  # 供 head 使用
        """
        B, C_bev, H, W = bev_feat.shape
        Q, Bq, C_embed = query.shape
        assert B == Bq, "batch size mismatch between bev_feat and query"

        # [B, embed_dim]
        bev_global = self.bev_proj(bev_feat)

        # 将 bev_global 加到每个 query 上：先 reshape query
        # query: (Q,B,C)->(B,Q,C)
        query_bqc = query.permute(1, 0, 2).contiguous()

        bev_global_exp = bev_global.unsqueeze(1)  # (B,1,C)
        token_feats = query_bqc + bev_global_exp  # (B,Q,C)

        return token_feats
