import numpy as np

from typing import List, Union
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import MODEL_FEEDERS

__all__ = ["FastRayFastBEVFeeders"]

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
    