from pathlib import Path
from typing import TYPE_CHECKING, Dict
import numpy as np

from prefusion.registry import TRANSFORMABLE_LOADERS
from prefusion.dataset.tensor_smith import TensorSmith
from prefusion.dataset.utils import read_pcd
from prefusion.dataset.transform import LidarPoints
from prefusion.dataset.transformable_loader import TransformableLoader


if TYPE_CHECKING:
    from prefusion.dataset.index_info import IndexInfo


@TRANSFORMABLE_LOADERS.register_module()
class LidarSweepsLoader(TransformableLoader):
    def __init__(self, data_root: Path, sweep_info_length=None) -> None:
        self.data_root = data_root
        self.sweep_info_length = sweep_info_length

    def load(self, name: str, scene_data: Dict, frame_data: Dict[str, Dict], index_info: "IndexInfo", tensor_smith: TensorSmith = None, **kwargs) -> LidarPoints:
        cur_frame = frame_data[index_info.frame_id]
        sweep_infos = cur_frame['lidar_points']['lidar1_sweeps']

        def Rt2T(R, t):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            return T

        def to_homo(points: np.array) -> np.array:
            if points.ndim == 1:
                points = points[None, :]
            ones = np.ones((len(points), 1), dtype=np.float32)
            return np.concatenate((points, ones), axis=1)

        def transform_pts_with_T(points, T):
            # from pts3d to lidar 3d
            points = np.array(points)
            shape = points.shape
            points = points.reshape(-1, 3)
            points_output = (to_homo(points) @ T.T)[:, :3].reshape(*shape)
            return points_output

        points = read_pcd(self.data_root / cur_frame["lidar_points"]["lidar1"])
        Twe = Rt2T(cur_frame["ego_pose"]["rotation"], cur_frame['ego_pose']["translation"])
        Tew = np.linalg.inv(Twe)
        ts = float(Path(cur_frame["lidar_points"]["lidar1"]).stem) / 1000
        output_points = [
                         np.concatenate([ points, np.zeros_like(points[:, :1])], axis=1)
                         
                         ]
        if self.sweep_info_length is not None:
            sweep_infos = sweep_infos[-self.sweep_info_length:]
        for sweep in sweep_infos:
            path = sweep['path']  # input points in ego coord
            Twei = sweep['Twe']
            sweep_ts = float(sweep['timestamp']) / 1000  # use as s
            Te0ei = Tew @ Twei
            points = read_pcd(self.data_root / path)
            points = np.concatenate([
                transform_pts_with_T(points[:, :3], Te0ei),
                points[:, 3:],
                np.zeros_like(points[:, :1]) + ts - sweep_ts
            ], axis=1)
            output_points += [points]
        output_points = np.concatenate(output_points, axis=0)
        return LidarPoints(name, output_points[:, :3], output_points[:, 3:], tensor_smith=tensor_smith)
