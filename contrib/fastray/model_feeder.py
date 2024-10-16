import torch
import numpy as np

from typing import List, Union, TYPE_CHECKING
from collections import defaultdict
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import MODEL_FEEDERS

from .voxel_lut import VoxelLookUpTableGenerator

from prefusion.dataset import (
    CameraImageSet, EgoPoseSet, LidarPoints,
    Bbox3D, Polyline3D, ParkingSlot3D
)

from prefusion.dataset.tensor_smith import (
    PlanarBbox3D, PlanarSquarePillar,
    PlanarCylinder3D, PlanarOrientedCylinder3D,
    PlanarPolyline3D, PlanarPolygon3D, PlanarSegBev,
    PlanarParkingSlot3D
)


# TODO: occ2d, should merge multiple frames to one


__all__ = ["FastRayModelFeeder"]



@MODEL_FEEDERS.register_module()
class FastRayModelFeeder(BaseModelFeeder):
    # TODO: for sdf_2d, we should mix tensor across frames

    def __init__(self, 
                 voxel_feature_config: dict, 
                 camera_feature_configs: dict):
        super().__init__()
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

        Returns
        -------
        dict | list
            processed_frame_batch

        Notes
        -----
        ```
        input_dict = {
            'index_info': index_info,
            'transformables': []
        }
        processed_frame_batch = {
            'index_infos': [index_info, index_info, ...],
            'camera_images': {
                'cam_0': (N, 3, H, W),
                ...
            },
            'camera_lookups': {
                'cam_0': {
                    uu:, (N, Z*X*Y),
                    vv:, (N, Z*X*Y),
                    dd:, (N, Z*X*Y),
                    ...
                ...
            },
            'delta_poses': [],
            'annotations': {
                'bbox_3d_0': {
                    'cen': (N, 1, X, Y)
                    'seg': (N, C, X, Y)
                    'reg': (N, C, X, Y)
                },
                ...
            },
            ...
        }
        ```
        """
        processed_frame_batch = {
            'index_infos': [],
            'camera_tensors': defaultdict(list),
            'camera_lookups': defaultdict(lambda: defaultdict(list)),
            'lidar_points': [],
            'delta_poses': [],
            'annotations': defaultdict(lambda: defaultdict(list)),
        }
        anno_batch_dict = processed_frame_batch['annotations']
        # rearange input_dict into batches
        for input_dict in frame_batch:
            # batching index info
            processed_frame_batch['index_infos'].append(input_dict['index_info'])
            for transformable in input_dict['transformables']:
                match transformable:
                    case CameraImageSet():
                        LUT = self.voxel_lut_gen.generate(transformable)
                        for cam_id in transformable.transformables:
                            camera_tensor = transformable.transformables[cam_id].tensor['img']
                            processed_frame_batch['camera_tensors'][cam_id].append(camera_tensor)
                            camera_lookup = LUT[cam_id]
                            camera_lookup_batch = processed_frame_batch['camera_lookups'][cam_id]
                            for lut_key in camera_lookup:
                                camera_lookup_batch[lut_key].append(camera_lookup[lut_key])
                    case LidarPoints():
                        raise NotImplementedError
                    case EgoPoseSet():
                        cur_pose = transformable.transformables['0']
                        if '-1' not in transformable.transformables:
                            pre_pose = transformable.transformables['0']
                        else:
                            pre_pose = transformable.transformables['-1']
                        delta_rotation = pre_pose.rotation.T @ cur_pose.rotation
                        delta_translation = pre_pose.rotation.T @ (cur_pose.translation - pre_pose.translation)
                        delta_T = torch.eye(4)
                        delta_T[:3, :3] = torch.tensor(delta_rotation)
                        delta_T[:3, 3:] = torch.tensor(delta_translation)
                        processed_frame_batch['delta_poses'].append(delta_T)
                    case Bbox3D() | Polyline3D() | ParkingSlot3D():
                        annotation_tensor = transformable.tensor
                        anno_ts = transformable.tensor_smith
                        match anno_ts:
                            case PlanarBbox3D() | PlanarSquarePillar() | \
                                PlanarCylinder3D() | PlanarOrientedCylinder3D() | PlanarParkingSlot3D():
                                anno_batch_dict[transformable.name]['cen'].append( annotation_tensor['cen'])
                                anno_batch_dict[transformable.name]['seg'].append( annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append( annotation_tensor['reg'])
                            case PlanarPolyline3D() | PlanarPolygon3D():
                                anno_batch_dict[transformable.name]['seg'].append( annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append( annotation_tensor['reg'])
                            case _:
                                anno_batch_dict[transformable.name].append(annotation_tensor)
        # tensorize batches
        for cam_id in processed_frame_batch['camera_tensors']:
            processed_frame_batch['camera_tensors'][cam_id] = torch.stack(
                processed_frame_batch['camera_tensors'][cam_id])
        for cam_id in processed_frame_batch['camera_lookups']:
            for lut_key in processed_frame_batch['camera_lookups'][cam_id]:
                processed_frame_batch['camera_lookups'][cam_id][lut_key] = torch.tensor(
                    np.float32(processed_frame_batch['camera_lookups'][cam_id][lut_key])
                )
        processed_frame_batch['delta_poses'] = torch.stack(processed_frame_batch['delta_poses'])
        for transformable_name, tensor_data in anno_batch_dict.items():
            # stack known one-sub-layer tensor_dict
            if isinstance(tensor_data, dict):
                for task_name, task_tensor_data in tensor_data.items():
                    if isinstance(task_tensor_data, torch.Tensor):
                        anno_batch_dict[transformable_name][task_name] = torch.stack(task_tensor_data)
            # stack tensor batches
            elif isinstance(tensor_data, torch.Tensor):
                anno_batch_dict[transformable_name] = torch.stack(tensor_data)
        return processed_frame_batch
