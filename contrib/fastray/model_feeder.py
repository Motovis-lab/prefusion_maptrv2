import torch
import numpy as np

from typing import List, Union, TYPE_CHECKING
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import MODEL_FEEDERS

from .voxel_lut import VoxelLookUpTableGenerator

from prefusion.dataset.tensor_smith import (
    PlanarBbox3D, PlanarSquarePillar,
    PlanarCylinder3D, PlanarOrientedCylinder3D,
    PlanarPolyline3D, PlanarPolygon3D, PlanarSegBev,
    PlanarParkingSlot3D
)


# TODO: occ2d, should merge multiple frames to one


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

        Returns
        -------
        dict | list
            processed_frame_batch

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
                'bbox_3d': {
                    'branch_0': {
                        'cen': (N, 1, X, Y)
                        'seg': (N, C, X, Y)
                        'reg': (N, C, X, Y)
                    },
                    ...
                },
                ...
            },
            ...
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
        anno_batch_dict = processed_frame_batch['annotations']
        # rearange input_dict into batches
        for input_dict in frame_batch:
            # batching index info
            processed_frame_batch['index_infos'].append(input_dict['index_info'])
            # batching camera tensors and lookups
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
            # batching lidar points
            if 'lidar_points' in input_dict['transformables']:
                raise NotImplementedError
            # batching delta poses
            pose_set = input_dict['transformables']['ego_poses']
            cur_pose = pose_set.transformables['0']
            if '-1' not in pose_set.transformables:
                pre_pose = pose_set.transformables['0']
            else:
                pre_pose = pose_set.transformables['-1']
            delta_rotation = pre_pose.rotation.T @ cur_pose.rotation
            delta_translation = pre_pose.rotation.T @ (cur_pose.translation - pre_pose.translation)
            delta_T = torch.eye(4)
            delta_T[:3, :3] = torch.tensor(delta_rotation)
            delta_T[:3, 3] = torch.tensor(delta_translation)
            processed_frame_batch['delta_poses'].append(delta_T)
            # batching annotations
            for transformable_key in input_dict['transformables']:
                if transformable_key in ['camera_images', 'camera_poses', 'lidar_points']:
                    continue
                annotation_tensor = input_dict['transformables'][transformable_key].tensor
                anno_ts = input_dict['transformables'][transformable_key].tensor_smith
                # init placeholder for various annotation types
                if transformable_key not in anno_batch_dict:
                    if isinstance(anno_ts, PlanarParkingSlot3D):
                        anno_batch_dict[transformable_key] = dict(cen=[], seg=[], reg=[])
                    elif isinstance(anno_ts, (
                        PlanarBbox3D, PlanarSquarePillar, PlanarCylinder3D, PlanarOrientedCylinder3D
                    )):
                        for branch_key in annotation_tensor:
                            anno_batch_dict[transformable_key][branch_key] = dict(cen=[], seg=[], reg=[])
                    elif isinstance(anno_ts, (PlanarPolyline3D, PlanarPolygon3D)):
                        for branch_key in annotation_tensor:
                            anno_batch_dict[transformable_key][branch_key] = dict(seg=[], reg=[])
                    else:
                        anno_batch_dict[transformable_key] = []
                if isinstance(anno_ts, PlanarParkingSlot3D):
                    anno_batch_dict[transformable_key]['cen'].append(annotation_tensor['cen'])
                    anno_batch_dict[transformable_key]['seg'].append(annotation_tensor['seg'])
                    anno_batch_dict[transformable_key]['reg'].append(annotation_tensor['reg'])
                elif isinstance(anno_ts, (
                    PlanarBbox3D, PlanarSquarePillar, PlanarCylinder3D, PlanarOrientedCylinder3D
                )):
                    for branch_key in annotation_tensor:
                        anno_batch_dict[transformable_key][branch_key]['cen'].append(
                            annotation_tensor[branch_key]['cen'])
                        anno_batch_dict[transformable_key][branch_key]['seg'].append(
                            annotation_tensor[branch_key]['seg'])
                        anno_batch_dict[transformable_key][branch_key]['reg'].append(
                            annotation_tensor[branch_key]['reg'])
                elif isinstance(anno_ts, (PlanarPolyline3D, PlanarPolygon3D)):
                    for branch_key in annotation_tensor:
                        anno_batch_dict[transformable_key][branch_key]['seg'].append(
                            annotation_tensor[branch_key]['seg'])
                        anno_batch_dict[transformable_key][branch_key]['reg'].append(
                            annotation_tensor[branch_key]['reg'])
                else:
                    anno_batch_dict[transformable_key].append(annotation_tensor)
        # tensorize batches
        for cam_id in processed_frame_batch['camera_tensors']:
            processed_frame_batch['camera_tensors'][cam_id] = torch.tensor(
                processed_frame_batch['camera_tensors'][cam_id])
        for cam_id in processed_frame_batch['camera_lookups']:
            for lut_key in processed_frame_batch['camera_lookups'][cam_id]:
                processed_frame_batch['camera_lookups'][cam_id][lut_key] = torch.tensor(
                    processed_frame_batch['camera_lookups'][cam_id][lut_key])
        processed_frame_batch['delta_poses'] = torch.tensor(processed_frame_batch['delta_poses'])
        for transformable_key in anno_batch_dict:
            if transformable_key in ['bbox_3d', 'square_3d', 'cylinder_3d', 'oriented_cylinder_3d']:
                for branch_key in anno_batch_dict[transformable_key]:
                    anno_batch_dict[transformable_key][branch_key]['cen'] = torch.tensor(
                        anno_batch_dict[transformable_key][branch_key]['cen'])
                    anno_batch_dict[transformable_key][branch_key]['seg'] = torch.tensor(
                        anno_batch_dict[transformable_key][branch_key]['seg'])
                    anno_batch_dict[transformable_key][branch_key]['reg'] = torch.tensor(
                        anno_batch_dict[transformable_key][branch_key]['reg'])
            if transformable_key in ['polyline_3d', 'polygon_3d']:
                for branch_key in anno_batch_dict[transformable_key]:
                    anno_batch_dict[transformable_key][branch_key]['seg'] = torch.tensor(
                        anno_batch_dict[transformable_key][branch_key]['seg'])
                    anno_batch_dict[transformable_key][branch_key]['reg'] = torch.tensor(
                        anno_batch_dict[transformable_key][branch_key]['reg'])
            if transformable_key == 'parking_slot_3d':
                anno_batch_dict[transformable_key]['cen'] = torch.tensor(
                    anno_batch_dict[transformable_key]['cen'])
                anno_batch_dict[transformable_key]['seg'] = torch.tensor(
                    anno_batch_dict[transformable_key]['seg'])
                anno_batch_dict[transformable_key]['reg'] = torch.tensor(
                    anno_batch_dict[transformable_key]['reg'])
        
        return processed_frame_batch