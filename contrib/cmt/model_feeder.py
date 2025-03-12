import torch
import numpy as np

from prefusion import LidarPoints
from prefusion.registry import MODEL_FEEDERS
from prefusion.dataset import BaseModelFeeder
from prefusion.dataset.transform import CameraImageSet, Bbox3D

__all__ = ["CMTModelFeeder"]


@MODEL_FEEDERS.register_module()
class CMTModelFeeder(BaseModelFeeder):
    """StreamPETRModelFeeder.

    Args
    ----
    Any: Any parameter or keyword arguments.

    """

    def __init__(self, *args, **kwargs):
        self.key_list = kwargs['key_list']

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
            # transformable_dict ={k: v for k, v in zip(self.key_list, frame["transformables"])}
            transformable_dict = frame["transformables"]
            processed_frame = dict(index_info=frame["index_info"], **transformable_dict)
            processed_frame["meta_info"] = {}
            if 'lidar_sweeps' in processed_frame.keys():
                output_lidar_points = {k: v for k, v in processed_frame.items() if
                                       k == 'lidar_points' or 'lidar_sweeps'}
                processed_frame['lidar_points'] = output_lidar_points['lidar_sweeps'].tensor
                processed_frame.pop('lidar_sweeps')
            for k, trnsfmb in processed_frame.items():
                if isinstance(trnsfmb, CameraImageSet):
                    cam_ids, camera_images = zip(*trnsfmb.transformables.items())
                    data = [(t.tensor["img"] * t.tensor["ego_mask"]).unsqueeze(0) for t in camera_images]
                    # import torch.nn.functional as F
                    # data = [i if i.shape[2] == 1080 else F.upsample(i, (1080, 1920))for i in data]
                    img_tensor = torch.vstack(data)
                    processed_frame[k] = img_tensor
                    processed_frame["meta_info"][k] = {
                        "camera_ids": cam_ids,
                        "intrinsic": [self._intrinsic_param_to_4x4_mat(cam_im.intrinsic) for cam_im in camera_images],
                        "extrinsic": [self._extrinsic_param_to_4x4_mat(*cam_im.extrinsic) for cam_im in camera_images],
                        "extrinsic_inv": [
                            np.linalg.inv(self._extrinsic_param_to_4x4_mat(*cam_im.extrinsic))
                            for cam_im in camera_images
                        ],
                        'cam_inv_poly': [cam_im.intrinsic[4:] for cam_im in camera_images],
                    }
                    continue
                if isinstance(trnsfmb, Bbox3D):
                    processed_frame[k] = trnsfmb.tensor["xyz_lwh_yaw_vxvy"]
                    processed_frame["meta_info"][k] = {"classes": trnsfmb.tensor["classes"]}
                # if isinstance(trnsfmb, LidarPoints):
                #     processed_frame[k] = trnsfmb.tensor
            processed_frame_batch.append(processed_frame)
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
