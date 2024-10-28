import torch

from contrib.fastray_planar.model import (
    VoxelTemporalAlign,
    FastRaySpatialTransform,
)


def test_voxel_temporal_align():
    voxel_shape = (6, 8, 8)  # Z, X, Y in ego system
    voxel_range = ([-0.5, 2.5], [2, -2], [2, -2])
    
    voxel_feats_pre = torch.zeros(2, 4, *voxel_shape)
    voxel_feats_pre[0, 0, 2] = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)
    
    delta_pose = torch.eye(4, dtype=torch.float32)
    delta_pose[:3, :3] = torch.tensor([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1]
    ])
    delta_pose[0, 3] = 2
    delta_pose[1, 3] = -1
    delta_pose[2, 3] = 0.5
    
    delta_poses = torch.stack([delta_pose] * 2)
    
    vta = VoxelTemporalAlign(voxel_shape, 
                             voxel_range,
                             interpolation='nearest')
    voxel_feats_pre_aligned = vta(voxel_feats_pre, delta_poses)

    answer = torch.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)
    
    torch.testing.assert_close(voxel_feats_pre_aligned[0, 0, 1], answer)


def test_fastray_spatial_transform():
    voxel_shape = (2, 8, 8)
    st = FastRaySpatialTransform(voxel_shape, fusion_mode='sampled')
    
    camera_feat_tensor = torch.zeros(1, 3, 4, 4, dtype=torch.float32)
    camera_feat_tensor[0, 0, 1, 1:3] = 28
    camera_feat_tensor[0, 1, 1, 1:3] = 4
    camera_feats_dict = dict(cam_6=camera_feat_tensor)
    camera_lookups = [dict(cam_6=dict(
        valid_map_sampled=torch.tensor([
            [[0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],
        ], dtype=torch.bool).reshape(-1),
        uu=torch.tensor([
            [[0, 0, 0, 1, 2, 3, 3, 0],
             [0, 0, 0, 1, 2, 3, 0, 0],
             [0, 0, 0, 0, 3, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 2, 3, 3, 0],
             [0, 0, 0, 1, 2, 3, 0, 0],
             [0, 0, 0, 0, 3, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],
        ], dtype=torch.long).reshape(-1),
        vv=torch.tensor([
            [[0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]],
        ], dtype=torch.long).reshape(-1)
    ))]

    voxel_feat = st(camera_feats_dict, camera_lookups)
    
    assert voxel_feat.shape == (1, 3, 2, 8, 8)
    answer = torch.tensor([
        [0, 0, 0,28,28, 0, 0, 0],
        [0, 0, 0,28,28, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    torch.testing.assert_close(voxel_feat[0, 0, 0], answer)