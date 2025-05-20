from typing import TYPE_CHECKING, Union

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from mmdet3d.structures.bbox_3d.utils import limit_period

from prefusion.registry import MODEL_FEEDERS
from prefusion.dataset import BaseModelFeeder
from prefusion.dataset.transform import CameraImageSet, Bbox3D

__all__ = ["StreamPETRModelFeeder"]


if TYPE_CHECKING:
    from prefusion.dataset.transform import EgoPoseSet


@MODEL_FEEDERS.register_module()
class StreamPETRModelFeeder(BaseModelFeeder):
    """StreamPETRModelFeeder.

    Args
    ----
    Any: Any parameter or keyword arguments.
    """
    def __init__(self, *, visible_range=(-25.6, -25.6, -2, 25.6, 25.6, 2), lidar_extrinsics=None, bbox_3d_pos_repr="cuboid_center", **kwargs) -> None:
        super().__init__(**kwargs)
        self.visible_range = visible_range

        assert bbox_3d_pos_repr in ["cuboid_center", "bottom_center"]
        self.bbox_3d_pos_repr = bbox_3d_pos_repr

        # 4x4 transformation matrix from lidar to ego (i.e. what we normally provided in calib)
        self.T_e_l: Union[np.ndarray, None] = np.array(lidar_extrinsics) if lidar_extrinsics is not None else None

    def process(self, frame_batch: list) -> dict | list:
        """
        Process frame_batch, make it ready for model inputs

        Parameters
        ----------
        frame_batch : list
            list of input_dicts

        Returns
        -------
        processed_frame_batch: dict | list
        }
        """
        processed_frame_batch = []
        for frame in frame_batch:
            processed_frame = dict(index_info=frame["index_info"], **{t.name: t for t in frame["transformables"].values()})
            processed_frame["meta_info"] = {}
            for k, trnsfmb in processed_frame.items():
                if isinstance(trnsfmb, CameraImageSet):
                    cam_ids, camera_images = zip(*trnsfmb.transformables.items())
                    img_tensor = torch.vstack(
                        [(t.tensor["img"] * t.tensor["ego_mask"]).unsqueeze(0) for t in camera_images]
                    )
                    processed_frame[k] = img_tensor
                    processed_frame["meta_info"][k] = {
                        "camera_ids": cam_ids,
                        "intrinsic": [self._intrinsic_param_to_4x4_mat(cam_im.intrinsic) for cam_im in camera_images],
                        "extrinsic": [self._extrinsic_param_to_4x4_mat(*cam_im.extrinsic) for cam_im in camera_images],
                        "extrinsic_inv": [
                            np.linalg.inv(self._extrinsic_param_to_4x4_mat(*cam_im.extrinsic))
                            for cam_im in camera_images
                        ],
                    }
                    continue
                if isinstance(trnsfmb, Bbox3D) and len(trnsfmb.elements) > 0:
                    visible_boxes = self.get_boxes_within_visible_range(trnsfmb.tensor)
                    processed_frame[k] = visible_boxes["xyz_lwh_yaw_vx_vy"]
                    processed_frame["meta_info"][k] = {"classes": visible_boxes["classes"]}
                    processed_frame["meta_info"][k].update(bbox3d_corners=visible_boxes["corners"])
            processed_frame_batch.append(processed_frame)

        if self.T_e_l is not None:
            for frame in processed_frame_batch:
                frame["meta_info"]["T_ego_lidar"] = self.T_e_l
                self.convert_model_food_to_lidar_coordsys_(frame)

        # Dedicated data handling for Nuscenes data
        for frame in processed_frame_batch:
            frame.update(dictionary=frame_batch[0]['transformables']['bbox_3d'].dictionary)

        return processed_frame_batch

    @staticmethod
    def _intrinsic_param_to_4x4_mat(param):
        mat = np.eye(4)
        mat[0, 0] = param[2]
        mat[1, 1] = param[3]
        mat[0, 2] = param[0]
        mat[1, 2] = param[1]
        return mat

    @staticmethod
    def _extrinsic_param_to_4x4_mat(rotation, translation):
        mat = np.eye(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = translation
        return mat

    def get_boxes_within_visible_range(self, box3d_tensor_dict: dict):
        box_centers = box3d_tensor_dict["xyz_lwh_yaw_vx_vy"][:, :3]
        visible_mask = (
            (box_centers[:, 0] >= self.visible_range[0]) & (box_centers[:, 0] <= self.visible_range[3])
            & (box_centers[:, 1] >= self.visible_range[1]) & (box_centers[:, 1] <= self.visible_range[4])
            & (box_centers[:, 2] >= self.visible_range[2]) & (box_centers[:, 2] <= self.visible_range[5])
        )
        return {k: v[visible_mask] for k, v in box3d_tensor_dict.items()}

    def convert_model_food_to_lidar_coordsys_(self, frame_data: dict):
        self.convert_ego_pose_set_to_lidar_coordsys_(frame_data["ego_poses"])
        self.convert_bbox3d_to_lidar_coordsys_(frame_data["bbox_3d"])

    def convert_ego_pose_set_to_lidar_coordsys_(self, ego_pose_set: "EgoPoseSet"):
        if self.T_e_l is None:
            return
        for t, pose in ego_pose_set.transformables.items():
            # explaination: what we want is T_w_l = T_w_e @ T_e_l
            # `pose` is T_w_e
            pose.rotation, pose.translation = (
                pose.rotation @ self.T_e_l[:3, :3],
                pose.rotation @ self.T_e_l[:3, 3][:, None] + pose.translation
            )

    def convert_bbox3d_to_lidar_coordsys_(self, bbox3d: torch.Tensor):
        if self.T_e_l is None:
            return
        
        T_l_e = torch.tensor(np.linalg.inv(self.T_e_l), dtype=torch.float32)

        # convert positions from ego coordsys to lidar coordsys
        bbox3d[:, :3] = (T_l_e @ self.to_homo(bbox3d[:, :3]).T).T[:, :3]
        if self.bbox_3d_pos_repr == "bottom_center":
            bbox3d[:, 2] -= bbox3d[:, 5] / 2

        # convert rotations from ego coordsys to lidar coordsys
        rotations = R.from_euler("Z", bbox3d[:, 6].tolist(), degrees=False).as_matrix()
        yaws = R.from_matrix(T_l_e[:3, :3][None, :].numpy() @ rotations).as_euler("XYZ", degrees=False)[:, 2]
        bbox3d[:, 6] = torch.tensor(limit_period(yaws, offset=0.5, period=2 * np.pi), dtype=torch.float32) # limit rad to [-pi, pi]

        # convert velocity from ego coordsys to lidar coordsys (do not apply translation to velocity)
        bbox3d[:, 7:9] = (T_l_e[:2, :2] @ bbox3d[:, 7:9].T).T
    
    @staticmethod
    def to_homo(pts: torch.Tensor):
        return torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
