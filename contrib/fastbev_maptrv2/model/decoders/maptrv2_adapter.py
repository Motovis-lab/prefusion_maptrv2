# prefusion/models/decoders/maptrv2_adapter.py
import torch
import torch.nn as nn

# 如果你把 MapTRv2 的 Decoder 类拷进来：
# from third_party.maptrv2.decoder import MapTRDecoder
# 或者你已经把它放到 prefusion.models.third_party 目录：
from prefusion.models.third_party.maptrv2_decoder import MapTRDecoder

class MapTRv2DecoderAdapter(nn.Module):
    """
    轻量适配器：
    - 调整 BEV 特征的通道/维度/排列以满足 decoder 的 cross-attn 需要
    - 调 decoder：out = decoder(query, bev_feat, reference_points=...)
    - 返回 decoder 输出，方便上游 Head 消费
    """
    def __init__(self, embed_dim: int, bev_channels: int, proj_bev_to_embed=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_bev_to_embed = proj_bev_to_embed

        # MapTRv2 Decoder 本体（按你复制/引入的实现初始化）
        self.decoder = MapTRDecoder(
            # 这里按原始实现的 __init__ 需要的参数填（num_layers、attention设置等）
        )

        # 如果你的 BEV 特征通道数 != embed_dim，则投影一下
        self.bev_proj = None
        if proj_bev_to_embed and bev_channels != embed_dim:
            # 2D BEV：Conv1x1；3D BEV （B,C,Z,H,W）的话先把 Z 合并或先 reduce
            self.bev_proj = nn.Conv2d(bev_channels, embed_dim, kernel_size=1)

    def forward(self, query, reference_points, bev_feat):
        """
        Args:
            query: (Q, B, C)
            reference_points: (B, Q, 4)
            bev_feat: 2D: (B, C, H, W) 或 3D: (B, C, Z, H, W)
        """
        B = reference_points.shape[0]
        Q = query.shape[0]
        C = query.shape[2]

        # 处理 BEV 特征维度
        if bev_feat.dim() == 5:
            # (B, C, Z, H, W) → 先把 Z 折叠/最大池化成平面的 (B, C, H, W)
            bev_feat = bev_feat.max(dim=2)[0]  # 或 mean
        assert bev_feat.dim() == 4, "期望 2D BEV (B,C,H,W)"

        if self.bev_proj is not None:
            bev_feat = self.bev_proj(bev_feat)  # (B, embed_dim, H, W)

        # 直接把 bev_feat 作为 *args 传入。注意：MapTRv2 的 layer 实现会决定如何使用该 memory。
        out, updated_ref = self.decoder(
            query,                # (Q, B, C)
            bev_feat,             # *args[0] 作为 memory/key/value
            reference_points=reference_points,  # (B, Q, 4)
            key_padding_mask=None,
            reg_branches=None
        )
        return out, updated_ref
