import torch
import numpy as np

from typing import List, Union, TYPE_CHECKING
from collections import defaultdict
from copious.data_structure.dict import defaultdict2dict
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import MODEL_FEEDERS

from .voxel_lut import VoxelLookUpTableGenerator

from prefusion.dataset import (
    CameraImageSet, EgoPoseSet, LidarPoints,
    Bbox3D, Polyline3D, ParkingSlot3D, OccSdfBev
)

from prefusion.dataset.tensor_smith import (
    PlanarBbox3D, PlanarSquarePillar,
    PlanarCylinder3D, PlanarOrientedCylinder3D,
    PlanarPolyline3D, PlanarPolygon3D,
    PlanarParkingSlot3D, PlanarOccSdfBev
)


# TODO: occ2d, should merge multiple frames to one


__all__ = ["FastRayPlanarModelFeeder"]



@MODEL_FEEDERS.register_module()
class FastRayPlanarModelFeeder(BaseModelFeeder):
    # TODO: for sdf_2d, we should mix tensor across frames

    def __init__(self, 
                 voxel_feature_config: dict, 
                 camera_feature_configs: dict,
                 bilinear_interpolation: bool = True,
                 debug_mode: bool = False):
        super().__init__()
        self.voxel_feature_config = voxel_feature_config
        self.camera_feature_configs = camera_feature_configs
        self.voxel_lut_gen = VoxelLookUpTableGenerator(
            voxel_feature_config=self.voxel_feature_config,
            camera_feature_configs=self.camera_feature_configs,
            bilinear_interpolation=bilinear_interpolation
        )
        self.debug_mode = debug_mode


    def process(self, frame_batch: list) -> Union[dict, list]:
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
            'camera_lookups': [
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
            ],
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
            'camera_lookups': [],
            'lidar_points': [],
            'delta_poses': [],
            'annotations': defaultdict(lambda: defaultdict(list))
        }
        if self.debug_mode:
            processed_frame_batch['transformables'] = []
        anno_batch_dict = processed_frame_batch['annotations']
        # rearange input_dict into batches
        for input_dict in frame_batch:
            # batching index info
            processed_frame_batch['index_infos'].append(input_dict['index_info'])
            # append transformables
            if self.debug_mode:
                processed_frame_batch['transformables'].append(input_dict['transformables'])
            # batching transformables
            for transformable in input_dict['transformables'].values():
                match transformable:
                    case CameraImageSet():
                        camera_lookup = self.voxel_lut_gen.generate(transformable)
                        for cam_id in camera_lookup:
                            for lut_key in camera_lookup[cam_id]:
                                camera_lookup[cam_id][lut_key] = torch.tensor(
                                    camera_lookup[cam_id][lut_key])
                        processed_frame_batch['camera_lookups'].append(camera_lookup)
                        for cam_id in transformable.transformables:
                            camera_tensor = transformable.transformables[cam_id].tensor['img']
                            processed_frame_batch['camera_tensors'][cam_id].append(camera_tensor)
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
                    case Bbox3D() | Polyline3D() | ParkingSlot3D() | OccSdfBev():
                        annotation_tensor = transformable.tensor
                        anno_ts = transformable.tensor_smith
                        match anno_ts:
                            case (PlanarBbox3D() | PlanarSquarePillar() 
                                  | PlanarCylinder3D() | PlanarOrientedCylinder3D() 
                                  | PlanarParkingSlot3D()):
                                anno_batch_dict[transformable.name]['cen'].append(annotation_tensor['cen'])
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarPolyline3D() | PlanarPolygon3D():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarOccSdfBev():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['sdf'].append(annotation_tensor['sdf'])
                                anno_batch_dict[transformable.name]['height'].append(annotation_tensor['height'])
                            case _:
                                anno_batch_dict[transformable.name].append(annotation_tensor)

        # tensorize batches
        for cam_id in processed_frame_batch['camera_tensors']:
            processed_frame_batch['camera_tensors'][cam_id] = torch.stack(
                processed_frame_batch['camera_tensors'][cam_id])
        if processed_frame_batch['delta_poses']:
            processed_frame_batch['delta_poses'] = torch.stack(processed_frame_batch['delta_poses'])
        for transformable_name, data_batch in anno_batch_dict.items():
            # stack known one-sub-layer tensor_dict
            if isinstance(data_batch, dict):
                for task_name, task_data_batch in data_batch.items():
                    if all(isinstance(data, torch.Tensor) for data in task_data_batch):
                        anno_batch_dict[transformable_name][task_name] = torch.stack(task_data_batch)
            # stack tensor batches
            elif all(isinstance(data, torch.Tensor) for data in data_batch):
                anno_batch_dict[transformable_name] = torch.stack(data_batch)
        return defaultdict2dict(processed_frame_batch)

@MODEL_FEEDERS.register_module()
class FastRayLidarPlanarModelFeeder(BaseModelFeeder):
    # TODO: for sdf_2d, we should mix tensor across frames

    def __init__(self,
                 voxel_feature_config: dict,
                 camera_feature_configs: dict,
                 bilinear_interpolation: bool = True,
                 debug_mode: bool = False):
        super().__init__()
        self.voxel_feature_config = voxel_feature_config
        self.camera_feature_configs = camera_feature_configs
        self.voxel_lut_gen = VoxelLookUpTableGenerator(
            voxel_feature_config=self.voxel_feature_config,
            camera_feature_configs=self.camera_feature_configs,
            bilinear_interpolation=bilinear_interpolation
        )
        self.debug_mode = debug_mode

    def process(self, frame_batch: list) -> Union[dict, list]:
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
            'camera_lookups': [
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
            ],
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
            'camera_lookups': [],
            'lidar_points': defaultdict(list),
            'annotations': defaultdict(lambda: defaultdict(list)),
            'delta_poses':[]
        }
        if self.debug_mode:
            processed_frame_batch['transformables'] = []
        anno_batch_dict = processed_frame_batch['annotations']
        # rearange input_dict into batches
        for input_dict in frame_batch:
            # batching index info
            processed_frame_batch['index_infos'].append(input_dict['index_info'])
            # append transformables
            if self.debug_mode:
                processed_frame_batch['transformables'].append(input_dict['transformables'])
            # batching transformables
            for transformable in input_dict['transformables'].values():
                match transformable:
                    case CameraImageSet():
                        camera_lookup = self.voxel_lut_gen.generate(transformable)
                        for cam_id in camera_lookup:
                            for lut_key in camera_lookup[cam_id]:
                                camera_lookup[cam_id][lut_key] = torch.tensor(
                                    camera_lookup[cam_id][lut_key])
                        processed_frame_batch['camera_lookups'].append(camera_lookup)
                        for cam_id in transformable.transformables:
                            camera_tensor = transformable.transformables[cam_id].tensor['img']
                            processed_frame_batch['camera_tensors'][cam_id].append(camera_tensor)
                    case LidarPoints():
                        processed_frame_batch['lidar_points']['res_voxels'].append(transformable.tensor['res_voxels'])
                        processed_frame_batch['lidar_points']['res_coors'].append(transformable.tensor['res_coors'])
                        processed_frame_batch['lidar_points']['res_num_points'].append(transformable.tensor['res_num_points'])
                        # processed_frame_batch['lidar_points']['points'].append(transformable.tensor['points'])
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
                    case Bbox3D() | Polyline3D() | ParkingSlot3D() | OccSdfBev():
                        annotation_tensor = transformable.tensor
                        anno_ts = transformable.tensor_smith
                        match anno_ts:
                            case (PlanarBbox3D() | PlanarSquarePillar()
                                  | PlanarCylinder3D() | PlanarOrientedCylinder3D()
                                  | PlanarParkingSlot3D()):
                                anno_batch_dict[transformable.name]['cen'].append(annotation_tensor['cen'])
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarPolyline3D() | PlanarPolygon3D():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarOccSdfBev():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['sdf'].append(annotation_tensor['sdf'])
                                anno_batch_dict[transformable.name]['height'].append(annotation_tensor['height'])
                            case _:
                                anno_batch_dict[transformable.name].append(annotation_tensor)

        # tensorize batches
        for cam_id in processed_frame_batch['camera_tensors']:
            processed_frame_batch['camera_tensors'][cam_id] = torch.stack(
                processed_frame_batch['camera_tensors'][cam_id])
        if 'delta_poses' in processed_frame_batch['delta_poses']:
            processed_frame_batch['delta_poses'] = torch.stack(processed_frame_batch['delta_poses'])
        for transformable_name, data_batch in anno_batch_dict.items():
            # stack known one-sub-layer tensor_dict
            if isinstance(data_batch, dict):
                for task_name, task_data_batch in data_batch.items():
                    if all(isinstance(data, torch.Tensor) for data in task_data_batch):
                        anno_batch_dict[transformable_name][task_name] = torch.stack(task_data_batch)
            # stack tensor batches
            elif all(isinstance(data, torch.Tensor) for data in data_batch):
                anno_batch_dict[transformable_name] = torch.stack(data_batch)
        return defaultdict2dict(processed_frame_batch)

# @MODEL_FEEDERS.register_module()
# class FastRayLidarPlanarModelFeeder(BaseModelFeeder):
#     # TODO: for sdf_2d, we should mix tensor across frames
#
#     def __init__(self,
#                  voxel_feature_config: dict,
#                  camera_feature_configs: dict,
#                  bilinear_interpolation: bool = True,
#                  debug_mode: bool = False):
#         super().__init__()
#         self.voxel_feature_config = voxel_feature_config
#         self.camera_feature_configs = camera_feature_configs
#         self.voxel_lut_gen = VoxelLookUpTableGenerator(
#             voxel_feature_config=self.voxel_feature_config,
#             camera_feature_configs=self.camera_feature_configs,
#             bilinear_interpolation=bilinear_interpolation
#         )
#         self.debug_mode = debug_mode
#
#     def process(self, frame_batch: list) -> Union[dict, list]:
#         """
#         Parameters
#         ----------
#         frame_batch : list
#             list of input_dicts
#
#         Returns
#         -------
#         dict | list
#             processed_frame_batch
#
#         Notes
#         -----
#         ```
#         input_dict = {
#             'index_info': index_info,
#             'transformables': []
#         }
#         processed_frame_batch = {
#             'index_infos': [index_info, index_info, ...],
#             'camera_images': {
#                 'cam_0': (N, 3, H, W),
#                 ...
#             },
#             'camera_lookups': [
#                 {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
#                 {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
#                 {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
#                 {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
#             ],
#             'delta_poses': [],
#             'annotations': {
#                 'bbox_3d_0': {
#                     'cen': (N, 1, X, Y)
#                     'seg': (N, C, X, Y)
#                     'reg': (N, C, X, Y)
#                 },
#                 ...
#             },
#             ...
#         }
#         ```
#         """
#         processed_frame_batch = {
#             'index_infos': [],
#             'camera_tensors': defaultdict(list),
#             'camera_lookups': [],
#             'lidar_points': defaultdict(list),
#             'delta_poses': [],
#             'annotations': defaultdict(lambda: defaultdict(list))
#         }
#         if self.debug_mode:
#             processed_frame_batch['transformables'] = []
#         anno_batch_dict = processed_frame_batch['annotations']
#         # rearange input_dict into batches
#         for input_dict in frame_batch:
#             # batching index info
#             processed_frame_batch['index_infos'].append(input_dict['index_info'])
#             # append transformables
#             if self.debug_mode:
#                 processed_frame_batch['transformables'].append(input_dict['transformables'])
#             # batching transformables
#             for transformable in input_dict['transformables'].values():
#                 match transformable:
#                     case CameraImageSet():
#                         camera_lookup = self.voxel_lut_gen.generate(transformable)
#                         for cam_id in camera_lookup:
#                             for lut_key in camera_lookup[cam_id]:
#                                 camera_lookup[cam_id][lut_key] = torch.tensor(
#                                     camera_lookup[cam_id][lut_key])
#                         processed_frame_batch['camera_lookups'].append(camera_lookup)
#                         for cam_id in transformable.transformables:
#                             camera_tensor = transformable.transformables[cam_id].tensor['img']
#                             processed_frame_batch['camera_tensors'][cam_id].append(camera_tensor)
#                     case LidarPoints():
#                         processed_frame_batch['lidar_points']['res_voxels'].append(transformable.tensor['res_voxels'])
#                         processed_frame_batch['lidar_points']['res_coors'].append(transformable.tensor['res_coors'])
#                         processed_frame_batch['lidar_points']['res_num_points'].append(transformable.tensor['res_num_points'])
#                         # processed_frame_batch['lidar_points']['points'].append(transformable.tensor['points'])
#                     case EgoPoseSet():
#                         cur_pose = transformable.transformables['0']
#                         if '-1' not in transformable.transformables:
#                             pre_pose = transformable.transformables['0']
#                         else:
#                             pre_pose = transformable.transformables['-1']
#                         delta_rotation = pre_pose.rotation.T @ cur_pose.rotation
#                         delta_translation = pre_pose.rotation.T @ (cur_pose.translation - pre_pose.translation)
#                         delta_T = torch.eye(4)
#                         delta_T[:3, :3] = torch.tensor(delta_rotation)
#                         delta_T[:3, 3:] = torch.tensor(delta_translation)
#                         processed_frame_batch['delta_poses'].append(delta_T)
#                     case Bbox3D() | Polyline3D() | ParkingSlot3D() | OccSdfBev():
#                         annotation_tensor = transformable.tensor
#                         anno_ts = transformable.tensor_smith
#                         match anno_ts:
#                             case (PlanarBbox3D() | PlanarSquarePillar()
#                                   | PlanarCylinder3D() | PlanarOrientedCylinder3D()
#                                   | PlanarParkingSlot3D()):
#                                 anno_batch_dict[transformable.name]['cen'].append(annotation_tensor['cen'])
#                                 anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
#                                 anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
#                             case PlanarPolyline3D() | PlanarPolygon3D():
#                                 anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
#                                 anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
#                             case PlanarOccSdfBev():
#                                 anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
#                                 anno_batch_dict[transformable.name]['sdf'].append(annotation_tensor['sdf'])
#                                 anno_batch_dict[transformable.name]['height'].append(annotation_tensor['height'])
#                             case _:
#                                 anno_batch_dict[transformable.name].append(annotation_tensor)
#
#         # tensorize batches
#         for cam_id in processed_frame_batch['camera_tensors']:
#             processed_frame_batch['camera_tensors'][cam_id] = torch.stack(
#                 processed_frame_batch['camera_tensors'][cam_id])
#         processed_frame_batch['delta_poses'] = torch.stack(processed_frame_batch['delta_poses'])
#         for transformable_name, data_batch in anno_batch_dict.items():
#             # stack known one-sub-layer tensor_dict
#             if isinstance(data_batch, dict):
#                 for task_name, task_data_batch in data_batch.items():
#                     if all(isinstance(data, torch.Tensor) for data in task_data_batch):
#                         anno_batch_dict[transformable_name][task_name] = torch.stack(task_data_batch)
#             # stack tensor batches
#             elif all(isinstance(data, torch.Tensor) for data in data_batch):
#                 anno_batch_dict[transformable_name] = torch.stack(data_batch)
#         return defaultdict2dict(processed_frame_batch)
#

@MODEL_FEEDERS.register_module()
class NuscenesFastRayPlanarModelFeeder(FastRayPlanarModelFeeder):
    def process(self, frame_batch: list) -> Union[dict, list]:
        processed_frame_batch = super().process(frame_batch)
        processed_frame_batch.update(sample_token=[], dictionaries=[], ego_poses=[])
        
        for input_dict in frame_batch:
            processed_frame_batch['sample_token'].append(input_dict["transformables"]["sample_token"])
            processed_frame_batch['dictionaries'].append({k: t.dictionary for k, t in input_dict["transformables"].items() if isinstance(t, Bbox3D)})
            processed_frame_batch['ego_poses'].append(input_dict["transformables"]["ego_poses"])
        
        return processed_frame_batch
