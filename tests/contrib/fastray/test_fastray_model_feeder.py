from pathlib import Path

import numpy as np
import pytest

from prefusion.dataset import (
    IndexInfo, 
    CameraImage, CameraImageSet, CameraImageTensor,
    Bbox3D, PlanarBbox3D,
    EgoPose, EgoPoseSet
)
from contrib.fastray.model_feeder import FastRayModelFeeder


def test_fastray_model_feeder():
    voxel_feature_config=dict(
        voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
        voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
        ego_distance_max=40,
        ego_distance_step=5
    )
    default_camera_feature_config = dict(
        ray_distance_num_channel=64,
        ray_distance_start=0.25,
        ray_distance_step=0.25,
        feature_downscale=8,
    )
    camera_feature_configs=dict(
        cam_6=default_camera_feature_config,
    )
    index_infos = [
        IndexInfo("20231101_160337", "1698825817764", 
            next=IndexInfo("20231101_160337", "1698825817864")
        ),
        IndexInfo("20231101_160337", "1698825817864", 
            prev=IndexInfo("20231101_160337", "1698825817764")
        ),
    ]
    camera_images = CameraImageSet("camera_images", dict(
        cam_6=CameraImage(
            name="camera_images:cam_6",
            cam_id='cam_6',
            cam_type='PerspectiveCamera',
            img=np.array(np.random.randint(256, size=(720, 1280, 3)), dtype=np.uint8),
            ego_mask=np.ones((720, 1280)),
            extrinsic=[np.eye(3), np.array([0, 0, 0])],
            intrinsic=[639.5, 359.5, 640, 640],
            tensor_smith=CameraImageTensor()                    
        )
    ))
    camera_images.to_tensor()
    ego_poses = EgoPoseSet(
        "ego_pose_set",
        {
            '0':EgoPose("ego_poses:0:1698825817764", index_infos[0].scene_frame_id, translation=np.zeros((3, 1)), rotation=np.eye(3)),
            '1':EgoPose("ego_poses:0:1698825817864", index_infos[1].scene_frame_id, translation=np.zeros((3, 1)), rotation=np.eye(3)),}
        )
    ego_poses.to_tensor()
    pbox3d = PlanarBbox3D(
        voxel_shape=voxel_feature_config['voxel_shape'],
        voxel_range=voxel_feature_config['voxel_range'],
    )
    bbox3d = Bbox3D(
        "bbox_3d_0",
        elements=[
            {
                'class': 'car',
                'attr': {},
                'size': [5, 2, 1.6],
                'rotation': np.float32([
                    [ 0.8, 0.6, 0],
                    [-0.6, 0.8, 0],
                    [ 0  , 0  , 1]
                ]),
                'translation': np.float32([
                    [3], [-5], [0]
                ]),
                'velocity': np.float32([
                    [3], [-1], [0]
                ]),
            },
        ],
        dictionary={'classes': ['car']},
        tensor_smith=pbox3d
    )
    bbox3d.to_tensor()
    frame_batch = [
        dict(
            index_info = index_infos[0],
            transformables = {
                'camera_images': camera_images, 
                'ego_poses': ego_poses, 
                'bbox_3d_0': bbox3d
            }
        ), 
        dict(
            index_info = index_infos[1],
            transformables = {
                'camera_images': camera_images, 
                'ego_poses': ego_poses, 
                'bbox_3d_0': bbox3d
            }
        )
    ]
    model_feeder = FastRayModelFeeder(
        voxel_feature_config, camera_feature_configs
    )
    processed_frame_batch = model_feeder.process(frame_batch)
    assert processed_frame_batch['camera_tensors']['cam_6'].shape == (2, 3, 720, 1280)
    assert processed_frame_batch['camera_lookups']['cam_6']['uu'].shape == (2, 6*320*160)
    assert processed_frame_batch['annotations']['bbox_3d_0']['reg'].shape == (2, 20, 320, 160)