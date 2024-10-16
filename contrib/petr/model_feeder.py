import torch
import numpy as np

from prefusion.registry import MODEL_FEEDERS
from prefusion.dataset import BaseModelFeeder
from prefusion.dataset.transform import CameraImageSet, Bbox3D

__all__ = ["StreamPETRModelFeeder"]


@MODEL_FEEDERS.register_module()
class StreamPETRModelFeeder(BaseModelFeeder):
    """StreamPETRModelFeeder.

    Args
    ----
    Any: Any parameter or keyword arguments.
    """
    def __init__(self, *, visible_range=(-25.6, -25.6, -2, 25.6, 25.6, 2), **kwargs) -> None:
        super().__init__(**kwargs)
        self.visible_range = visible_range

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
            processed_frame = dict(index_info=frame["index_info"], **{t.name: t for t in frame["transformables"]})
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
                if isinstance(trnsfmb, Bbox3D):
                    visible_boxes = self.get_boxes_within_visible_range(trnsfmb.tensor)
                    processed_frame[k] = visible_boxes["xyz_lwh_yaw_vx_vy"]
                    processed_frame["meta_info"][k] = {"classes": visible_boxes["classes"]}
                    processed_frame["meta_info"][k].update(bbox3d_corners=visible_boxes["corners"])
            processed_frame_batch.append(processed_frame)
        return processed_frame_batch

    @staticmethod
    def _intrinsic_param_to_4x4_mat(param):
        mat = np.eye(4)
        mat[0, 0] = param[2]
        mat[1, 1] = param[3]
        mat[0, 2] = param[0]
        mat[2, 2] = param[1]
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
