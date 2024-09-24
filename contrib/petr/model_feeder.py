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
            processed_frame = dict(index_info=frame["index_info"], **frame["transformables"])
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
                    processed_frame[k] = trnsfmb.tensor["xyz_lwh_yaw_vxvy"]
                    processed_frame["meta_info"][k] = {"classes": trnsfmb.tensor["classes"]}
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
