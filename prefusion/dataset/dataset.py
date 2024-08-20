# Written by AlphaLFC. All rights reserved.

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union
from collections import defaultdict

import mmcv
import mmengine
import torch
import numpy as np
from torch.utils.data import Dataset
from mmengine.registry import DATASETS, FUNCTIONS

# 'camera_images', 'lidar_points', 
# 'camera_segs', 'camera_depths',
# 'bbox3d', 'bboxbev', 'square3d', 'cylinder3d', 'oriented_cylinder3d',
# 'polyline3d', 'polygon3d', 'parkingslot3d', 'trajectory',
# 'seg_bev', 'occ_sdf_bev', 'occ_sdf_3d'
from .transform import (
    CameraImage, CameraImageSet, LidarPoints, 
    CameraImageSegMask, CameraImageSegMaskSet, 
    CameraImageDepth, CameraImageDepthSet,
    Bbox3D, BboxBev, Cylinder3D, OrientedCylinder3D, Square3D, 
    Polyline3D, Polygon3D, ParkingSlot3D, Trajectory,
    SegBev, OccSdfBev, OccSdf3D
)

from .utils import get_cam_type, build_transforms


def get_frame_index(sequence, timestamp):
    for t in sequence:
        if timestamp <= t:
            return t


@FUNCTIONS.register_module()
def collate_dict(batch):
    return batch[0]



@DATASETS.register_module()
class GroupBatchDataset(Dataset):
    '''
    A novel dataset class for batching sequence groups for multi-module data.
    '''
    # TODO: implement visualization?
    # TODO: conceptualize and create ModelFood and ModelFoodArranger (naming is temporary), registered ModelFoodArranger should be passed to Dataset.
    AVAILABLE_TRANSFORMABLE_KEYS = (
        'camera_images', 'lidar_points', 
        'camera_segs', 'camera_depths',
        'bbox_3d', 'bbox_bev', 'square_3d', 'cylinder_3d', 'oriented_cylinder_3d',
        'polyline_3d', 'polygon_3d', 'parkingslot_3d', 'trajectory',
        'seg_bev', 'occ_sdf_bev', 'occ_sdf_3d'
    )
    
    def __init__(self, name, *, 
                 data_root, 
                 info_path, 
                 dictionary,  # TODO: to combine with transformable_keys (because they are required to have the same keys)
                 transformable_keys,
                 transforms,
                 phase='train',
                 indices_path=None,
                 batch_size=1,
                 drop_last=False,
                 group_size=3,
                 group_by_scene=False, 
                 frame_interval=1, # TODO: redefine it by time, like seconds
                 group_backtime_prob=0.0,
                 seed_dataset=None):
        """
        Initializes the dataset instance.

        Args:
        - name (str): Name of the dataset.
        - data_root (str): Root directory of the dataset.
        - info_path (str): Path to the information file.
        - dictionary (dict): Dictionary for each transformable.
            - dictionary = {
                'bbox_3d': {
                    'branch_0': {
                        'classes': [<>, <>, ...], 
                        'attrs:': [<>, <>, ...]
                    }, 
                    'branch_1': {...}
                },
                'bbox_bev': {...},
                'polyline_3d': {...},
            }
        - transformable_keys (list): List of transformable keys. Keys should be in dictionary.
        - transforms (list): Transform classes for preprocessing transformables. Build by TRANSFORMS.
        - phase (str): Specifies the phase ('train', 'val', or 'test') of the dataset; default is 'train'.
        - indices_path (str, optional): Specified file of indices to load; if None, all frames are automatically fetched from the info_path.
        - batch_size (int, optional): Batch size; defualt is 1.
        - group_size (int or list, optional): Size of sequence group; can be a single integer or a list (e.g., [5, 10]); default is 3.
        - group_by_scene (bool, optional): Whether to group by scenes, applicable only during the 'test' phase; default is False.
        - frame_interval (int or list, optional): Interval between frames; default is 1.
        - group_backtime_prob (float): Probability of grouping backtime frames.
        - seed_dataset (int): Random seed for dataset

        Notes:
        - `scene`: a scene contains a sequence of frames
        - `group`: a subset of a scene, when testing, a scene can be one group
        - `frame`: a time instance of a scene
        - one index should get a batch of groups, one group should be a sequential list
        - the length of the dataset should be len(self.groups) // self.batch_size
        - a frame data is a dictionary containing the transformed data.
        """
        super().__init__()
        self.name = name
        self.data_root = Path(data_root)
        assert phase.lower() in ['train', 'val', 'test']
        self.phase = phase.lower()
        self.info = mmengine.load(info_path)
        self._assert_availability(list(dictionary.keys()))
        self._assert_availability(transformable_keys)
        self.dictionary = dictionary
        self.transformable_keys = transformable_keys
        self.transforms = build_transforms(transforms)

        if indices_path is not None:
            indices = [line.strip() for line in open(indices_path, 'w')]
            self.scene_ids = self._prepare_indices(self.info, indices=indices)
        else:
            self.scene_ids = self._prepare_indices(self.info)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.group_size = group_size
        self.group_by_scene = group_by_scene
        self.frame_interval = frame_interval
        self.group_backtime_prob = group_backtime_prob
        
        self.seed_dataset = seed_dataset
        self.sample_groups()

    @classmethod
    def _assert_availability(cls, keys: List[str]) -> None:
        for key in keys:
            assert key in cls.AVAILABLE_TRANSFORMABLE_KEYS, \
                f"{key} is not a valid transformable key from {cls.AVAILABLE_TRANSFORMABLE_KEYS}"
    

    @staticmethod
    def _prepare_indices(info: Dict, indices: List[str] = None) -> Dict[str, List[str]]:
        if indices is None:
            indices = {}
            for scene_id in info:
                frame_list = sorted(info[scene_id]['frame_info'].keys())
                if frame_list:
                    indices[scene_id] = [f'{scene_id}/{frame_id}' for frame_id in frame_list]
            return indices
        
        available_indices = defaultdict(list)
        for index in indices:
            scene_id, frame_id = index.split('/')
            if scene_id in info:
                if frame_id in info[scene_id]['frame_info']:
                    available_indices[scene_id].append(index)
        for scene_id in available_indices:
            available_indices[scene_id] = sorted(available_indices[scene_id])
        
        return available_indices
            

    def __repr__(self):
        return ''.join([
            f'An instance of {self.__class__}: (\n',
            f'    name={self.name}, \n',
            f'    num_groups={len(self.groups)}, \n'
            f'    phase={self.phase}, \n'
            f'    batch_size={self.batch_size}, \n'
            f'    group_size={self.group_size}, \n'
            f'    group_by_scene={self.group_by_scene}, \n'
            f'    frame_interval={self.frame_interval}, \n'
            f'    group_backtime_prob={self.group_backtime_prob}\n)'
        ])
    

    def get_input_dict(self, index) -> dict:
        scene_id, frame_id = index.split('/')

        input_dict = {
            'scene_id': scene_id,
            'frame_id': frame_id,
            'prev_exists': True,
            'next_exists': True,
            'transformables': {}
        }  # TODO: add ego pose transformation
        for key in self.transformable_keys:
            input_dict['transformables'][key] = eval(f'self.load_{key}')(index)

        return input_dict



    def sample_groups(self):
        if self.phase in ['train']:
            self._sample_train_groups()
        else:
            if self.phase in ['test'] and self.group_by_scene:
                self._sample_scene_groups()
            else:
                self._sample_val_groups()


    def _sample_train_groups(self):
        if type(self.group_size) in [list, tuple]:
            group_size = random.choice(self.group_size)
        else:
            group_size = self.group_size
        
        if type(self.frame_interval) in [list, tuple]:
            frame_interval = random.choice(self.frame_interval)
        else:
            frame_interval = self.frame_interval
        
        group_interval = group_size * frame_interval

        groups = []
        for scene_id in self.scene_ids:
            first_end_ind = random.randint(frame_interval, group_interval)
            last_end_ind = len(self.scene_ids[scene_id])
            scene_ind_list = range(len(self.scene_ids[scene_id]))
            end_inds = []
            for i in range(frame_interval):
                end_inds.extend(scene_ind_list[first_end_ind + i::group_interval])
            end_inds = sorted(end_inds)
            if end_inds[-1] < last_end_ind:
                end_inds.append(last_end_ind)
            for end_ind in end_inds:
                if end_ind < group_interval:
                    first_ind = 0
                    last_ind = group_interval
                else:
                    first_ind = end_ind - group_interval
                    last_ind = end_ind
                ind_list = range(first_ind, last_ind)[::frame_interval]
                group = [self.scene_ids[scene_id][ind] for ind in ind_list]
                groups.append(group)
        random.shuffle(groups)
        self.groups = groups
                
    
    def _sample_val_groups(self):
        if type(self.group_size) in [list, tuple]:
            group_size = self.group_size[0]
        else:
            group_size = self.group_size
        
        if type(self.frame_interval) in [list, tuple]:
            frame_interval = self.frame_interval[0]
        else:
            frame_interval = self.frame_interval
        
        group_interval = group_size * frame_interval

        groups = []
        for scene_id in self.scene_ids:
            first_end_ind = group_interval
            last_end_ind = len(self.scene_ids[scene_id])
            scene_ind_list = range(len(self.scene_ids[scene_id]))
            end_inds = []
            for i in range(frame_interval):
                end_inds.extend(scene_ind_list[first_end_ind + i::group_interval])
            end_inds = sorted(end_inds)
            if end_inds[-1] < last_end_ind:
                end_inds.append(last_end_ind)
            for end_ind in end_inds:
                if end_ind < group_interval:
                    first_ind = 0
                    last_ind = group_interval
                else:
                    first_ind = end_ind - group_interval
                    last_ind = end_ind
                ind_list = range(first_ind, last_ind)[::frame_interval]
                group = [self.scene_ids[scene_id][ind] for ind in ind_list]
                groups.append(group)
        self.groups = groups
    

    def _sample_scene_groups(self):
        scene_groups = []
        for scene_id in self.scene_ids:
            scene_groups.append(self.scene_ids[scene_id])
        self.groups = scene_groups


    def _sample_groups_by_class_balance(self):
        raise NotImplementedError
    

    def _sample_groups_by_meta_info(self):
        raise NotImplementedError


    def __len__(self):
        if self.drop_last:
            return len(self.groups) // self.batch_size
        else:
            return int(np.ceil(len(self.groups) / self.batch_size))


    def __getitem__(self, idx):
        
        if idx >= len(self):
            self.sample_groups()
            raise IndexError

        batched_groups = []
        for batch_idx in range(self.batch_size):
            group_idx = idx * self.batch_size + batch_idx
            if group_idx >= len(self.groups):
                group_idx = random.randint(0, len(self.groups) - 1)
            batched_groups.append(self.groups[group_idx])
        
        batch = []
        for group in batched_groups:
            group_of_inputs = []
            for i, index in enumerate(group):
                input_dict = self.get_input_dict(index)
                if i == 0:
                    input_dict['prev_exists'] = False
                if i == self.group_size - 1:
                    input_dict['next_exists'] = False
                group_of_inputs.append(input_dict)
            batch.append(group_of_inputs)

        # apply transforms
        batch_seed = int.from_bytes(os.urandom(2), byteorder="big")
        for group_of_inputs in batch:
            group_seed = int.from_bytes(os.urandom(2), byteorder="big")
            for input_dict in group_of_inputs:
                frame_seed = int.from_bytes(os.urandom(2), byteorder="big")
                transformables = []
                for key in input_dict['transformables']:
                    transformable = input_dict['transformables'][key]
                    if isinstance(transformable, dict):
                        transformables.extend(transformable.values())                    
                    else:
                        transformables.append(transformable)
                for transform in self.transforms:
                    transform(*transformables, seeds={'group': group_seed, 'batch': batch_seed, 'frame': frame_seed})

        group_batch = []
        for frame_batch in zip(*batch):
            group_batch.append(frame_batch)
        
        return group_batch


    def load_camera_images(self, index: str) -> CameraImageSet:
        return CameraImageSet.from_info(self.data_root, self.info, index)

    def load_lidar_points(self, index: str):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        pcd_path = self.data_root / frame['lidar_points']['lidar1']
        raise NotImplementedError


    def load_camera_segs(self, index: str) -> CameraImageSegMaskSet:
        return CameraImageSegMaskSet.from_info(self.data_root, self.info, index, self.dictionary)


    def load_camera_depths(self, index: str) -> CameraImageDepthSet:
        return CameraImageDepthSet.from_info(self.data_root, self.info, index, depth_mode='d')


    def load_bbox_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_boxes']
        return Bbox3D(elements, self.dictionary['bbox_3d'])
    

    def load_bbox_bev(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_boxes']
        return BboxBev(elements, self.dictionary['bbox_bev'])

  
    def load_square_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_boxes']
        return Square3D(elements, self.dictionary['square_3d'])


    def load_cylinder_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_boxes']
        return Cylinder3D(elements, self.dictionary['cylinder_3d'])
    

    def load_oriented_cylinder_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_boxes']
        return OrientedCylinder3D(elements, self.dictionary['oriented_cylinder_3d'])
    

    def load_polyline_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_polylines']
        return Polyline3D(elements, self.dictionary['polyline_3d'])
    

    def load_polygon_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_polylines']
        return Polygon3D(elements, self.dictionary['polygon_3d'])
    

    def load_parkingslot_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        elements = frame['3d_polylines']
        return ParkingSlot3D(elements, self.dictionary['parkingslot_3d'])
    

    def load_trajectory(self, index, time_window=2):
        scene_id, cur_frame_id = index.split('/')
        scene = self.info[scene_id]
        frame_list = list(scene['frame_info'].keys())
        cur_ind = frame_list.index(cur_frame_id)
        end_time = int(cur_frame_id) * self.info['meta_info']['time_unit'] + time_window
        end_timestamp = str(int(end_time / self.info['meta_info']['time_unit']))
        end_frame_id = get_frame_index(frame_list, end_timestamp)
        end_ind = frame_list.index(end_frame_id) + 1
        frame_ids = frame_list[cur_ind:end_ind]
        
        ego_trajectory = []
        for frame_id in frame_ids:
            # ego_trajectory
            R = scene[frame_id]['ego_pose']['rotation']
            t = scene[frame_id]['ego_pose']['translation']
            ego_trajectory.append((R, t))
        
        # TODO: other object trajectories
        
        trajectories = [ego_trajectory]
        return Trajectory(trajectories, self.dictionary['trajectory'])
    

    def load_seg_bev(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        raise NotImplementedError
    

    def load_occ_sdf_bev(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        occ_path = frame['occ_sdf']['occ_bev']
        sdf_path = frame['occ_sdf']['sdf_bev']
        height_path = frame['occ_sdf']['height_bev']
        data = {
            'src_view_range': scene['meta_info']['space_range']['occ'], # ego system
            'dst_view_range': scene['meta_info']['space_range']['occ'], # ego system
            # 'occ': <N x H x W>, # H <=> (x_min, x_max), W <=> (y_min, y_max)
            # 'sdf': <1 x H x W>,
            # 'height': <1 x H x W>,
            # 'mask': <1 x H x W>,
        }
        return OccSdfBev(data)
    

    def load_occ_sdf_3d(self, index):
        scene_id, frame_id = index.split('/')
        scene = self.info[scene_id]
        frame = scene['frame_info'][frame_id]
        file_path = frame['occ_sdf']['occ_sdf_3d']
        raise NotImplementedError
    
