import numpy as np

from typing import List, Union
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import MODEL_FEEDERS

from .voxel_lut import VoxelLookUpTableGenerator

__all__ = ["FastRayModelFeeders"]



@MODEL_FEEDERS.register_module()
class FastRayModelFeeders(BaseModelFeeder):

    def __init__(self, 
                 voxel_feature_config: dict, 
                 camera_feature_configs: dict):
        super.__init__()
        self.voxel_feature_config = voxel_feature_config
        self.camera_feature_configs = camera_feature_configs
        # TODO: move cam group to model
        # self.camera_id_groups = camera_id_groups
        self.voxel_lut_gen = VoxelLookUpTableGenerator(
            voxel_feature_config=self.voxel_feature_config,
            camera_feature_configs=self.camera_feature_configs
        )


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
            'index_info': index_info,
            'transformables': {
                <key>: <values>,
            }
        }
        processed_frame_batch = {
            'index_infos': [index_info, index_info, ...],
            'camera_images': {
                'cam_0': (N, 3, H, W),
                'cam_1': (N, 3, H, W),
                ...
            },
            'camera_lookups': {
                'cam_0': {
                    uu:, (N, Z*X*Y),
                    vv:, (N, Z*X*Y),
                    dd:, (N, Z*X*Y),
                    ...
                },
                'cam_1': {
                    uu:, (N, Z*X*Y),
                    vv:, (N, Z*X*Y),
                    dd:, (N, Z*X*Y),
                    ...
                },
                ...
            },
            'delta_poses': [],
            'annotations': {},
        }
        ```
        """
        processed_frame_batch = {
            'index_infos': [],
            'camera_tensors': {},
            'camera_lookups': {},
            'lidar_points': [],
            'delta_poses': [],
            'annotations': {},
        }
        for input_dict in frame_batch:
            transformable_keys = input_dict['transformables'].keys()
            # get camera tensors and lookups
            camera_image_set = input_dict['transformables']['camera_images']
            LUT = self.voxel_lut_gen.generate(camera_image_set)
            for cam_id in camera_image_set:
                if cam_id not in processed_frame_batch['camera_tensors']:
                    processed_frame_batch['camera_tensors'][cam_id] = []
                if cam_id not in processed_frame_batch['camera_lookups']:
                    processed_frame_batch['camera_lookups'][cam_id] = {}
                camera_tensor = camera_image_set.transformables[cam_id].tensor['img']
                processed_frame_batch['camera_tensors'][cam_id].append(camera_tensor)
                camera_lookup = LUT[cam_id]
                camera_lookup_batch = processed_frame_batch['camera_lookups'][cam_id]
                for lut_key in camera_lookup:
                    if lut_key not in camera_lookup_batch:
                        camera_lookup_batch[lut_key] = []
                    camera_lookup_batch[lut_key].append(camera_lookup[lut_key])
            # lidar points
            if 'lidar_points' in input_dict['transformables']:
                raise NotImplementedError
            # delta poses
            # TODO:
            
            # annotations
            for transformable_key in transformable_keys:
                if transformable_key in ['camera_images', 'camera_poses', 'lidar_points']:
                    continue
                annotation_tensor = input_dict['transformables'][transformable_key].tensor                    
                if transformable_key not in processed_frame_batch['annotations']:
                    if isinstance(annotation_tensor, dict):
                        processed_frame_batch['annotations'][transformable_key] = {}
                    else:
                        processed_frame_batch['annotations'][transformable_key] = []
                if isinstance(annotation_tensor, dict):
                    for tensor_key in annotation_tensor:
                        processed_frame_batch['annotations'][transformable_key][tensor_key].append(
                            annotation_tensor[tensor_key])
                else:
                    processed_frame_batch['annotations'][transformable_key].append(annotation_tensor)
                
            
    