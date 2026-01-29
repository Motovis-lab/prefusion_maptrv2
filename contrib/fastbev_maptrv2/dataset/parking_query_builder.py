# prefusion/contrib/fastbev_maptrv2/dataset/parking_query_builder.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, List


class ParkingQueryBuilder(nn.Module):
    """
    将停车位 GT(多边形/角点)转换为 MapTRv2 风格的层级 query：
      - Q = N * P = num_instances * num_point_queries
      - 每个 query 是“实例 i 的第 j 个点”
      - reference_points 是实例级 (xc,yc,w,h) 归一化到 [0,1]，在该实例的 P 个点上复制
    """

    def __init__(
        self,
        embed_dim: int,
        num_point_queries: int,
        voxel_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        combine_mode: str = "sum",  # "sum" 或 "concat"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_point_queries = num_point_queries
        self.voxel_range = voxel_range
        self.combine_mode = combine_mode

        # 实例级投影: (x,y,z)->C
        self.instance_proj = nn.Linear(3, embed_dim)
        # 点级投影: (x,y,z)->C
        self.point_proj = nn.Linear(3, embed_dim)

        if combine_mode == "concat":
            self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.concat_proj = None

    @staticmethod
    def _norm01(val: float, a: float, b: float) -> float:
        """支持 a>b 的区间，稳健归一化到 [0,1]。"""
        lo, hi = (a, b) if a <= b else (b, a)
        denom = max(hi - lo, 1e-6)
        return float((val - lo) / denom)

    def build(
        self,
        instances: List[Dict[str, Any]],
        batch_size: int,
        device: torch.device,
    ):
        """
        Args:
            instances: [{"points": (P,3) or list[[x,y,z],...], "label": int, ...}, ...]
        Returns:
            query:            (Q, B, C)
            reference_points: (B, Q, 4)
            valid_mask:       (B, Q)
            meta: {
                "map_inst2query_idx": (N,),
                "shape_info": {"num_instances": N, "num_point_queries": P, ...}
            }
        """
        num_instances = len(instances)
        (z_min, z_max), (x_min, x_max), (y_min, y_max) = self.voxel_range

        total_queries = num_instances * self.num_point_queries
        hier_query_no_bs = torch.zeros((total_queries, self.embed_dim), device=device)
        valid_mask_no_bs = torch.zeros((total_queries,), dtype=torch.bool, device=device)
        ref_points_no_bs = torch.zeros((total_queries, 4), device=device)
        map_inst2query_idx = torch.zeros((num_instances,), dtype=torch.long, device=device)

        q_ptr = 0
        for i, inst in enumerate(instances):
            pts = np.asarray(inst["points"], dtype=np.float32)  # (P,3)
            P = pts.shape[0]

            # 用 x,y 计算中心/宽高
            center = pts[:, :2].mean(axis=0)
            width = pts[:, 0].max() - pts[:, 0].min()
            height = pts[:, 1].max() - pts[:, 1].min()

            x_norm = self._norm01(center[0], x_min, x_max)
            y_norm = self._norm01(center[1], y_min, y_max)
            w_norm = self._norm01(width, 0.0, abs(x_max - x_min))
            h_norm = self._norm01(height, 0.0, abs(y_max - y_min))

            # 实例级 embedding
            inst_feat = torch.tensor([center[0], center[1], 0.0], device=device)
            q_ins = self.instance_proj(inst_feat)  # (C,)

            map_inst2query_idx[i] = q_ptr

            for j in range(self.num_point_queries):
                if j < P:
                    # 真实点
                    pt_xyz = torch.tensor(pts[j, :3], device=device)
                    q_pt = self.point_proj(pt_xyz)

                    if self.combine_mode == "sum":
                        q_ij = q_ins + q_pt
                    elif self.combine_mode == "concat":
                        assert self.concat_proj is not None
                        q_ij = self.concat_proj(torch.cat([q_ins, q_pt], dim=-1))
                    else:
                        raise ValueError(f"Unknown combine_mode={self.combine_mode}")

                    hier_query_no_bs[q_ptr] = q_ij
                    valid_mask_no_bs[q_ptr] = True
                # else: padding，保持 0，mask=False

                ref_points_no_bs[q_ptr, :] = torch.tensor(
                    [x_norm, y_norm, w_norm, h_norm], device=device
                )
                q_ptr += 1

        # query: (Q,B,C)
        query = (
            hier_query_no_bs.unsqueeze(1)
            .expand(-1, batch_size, -1)
            .contiguous()
        )
        # (B,Q,4)
        reference_points = (
            ref_points_no_bs.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        # (B,Q)
        valid_mask = (
            valid_mask_no_bs.unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        shape_info = dict(
            num_instances=num_instances,
            num_point_queries=self.num_point_queries,
            num_query=total_queries,
            embed_dim=self.embed_dim,
        )

        meta = {
            "map_inst2query_idx": map_inst2query_idx,
            "shape_info": shape_info,
        }

        return query, reference_points, valid_mask, meta
