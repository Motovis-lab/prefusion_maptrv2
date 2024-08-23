import torch
import numpy as np


from prefusion.registry import MODEL_FEEDERS



@MODEL_FEEDERS.register_module()
class BaseModelFeeder:
    """BaseModelFeeder.

    An alternative implementation of data_preprocessor.

    Args
    ----
    Any: Any parameter or keyword arguments.

    """

    def __init__(self, *args, **kwargs) -> None:
        pass


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
        processed_frame_batch = frame_batch
        return processed_frame_batch

    def __call__(self, group_batch: list):
        processed_group_batch = []
        for frame_batch in group_batch:
            processed_group_batch.append(self.process(frame_batch))
        
        return processed_group_batch




@MODEL_FEEDERS.register_module()
class FastRayFastBEVFeeders(BaseModelFeeder):

    def __init__(self, 
                 camera_feature_configs: dict,
                 voxel_feature_config: dict, 
                 bbox_3d_config: dict,
                 polyline_3d_config: dict,
                 phase: str = 'train'):
        pass


    def process(self, frame_batch: list) -> dict | list:
        """
        Parameters
        ----------
        frame_batch : list
            list of input_dicts
        training : bool, optional
            _description_, by default False

        Returns
        -------
        dict | list
            _description_

        Notes
        -----
        ```
        input_dict = {
            'scene_id': scene_id,
            'frame_id': frame_id,
            'prev_exists': True,
            'next_exists': True,
            'transformables': {}
        }
        batch_input_dict ={
            'scene_ids': [scene_id, scene_id, ...],
            'frame_ids': [frame_id, frame_id, ...],
            'prev_exists': [True, True, ...],
            'next_exists': [True, True, ...],
            'transformables': {
                'lidar_points': [lidar_points, lidar_points, ...],
                'camera_image_batches': {
                    cam_id: (N, 3, H, W),
                    cam_id: (N, 3, H, W)
                }
            }
            'LUTS': [LUT, LUT, ...],
        }
        ```
        """
        for input_dict in frame_batch:
            input_dict['transformables']['camera_image_set'].to_tensor()
    