import torch

from contrib.fastray_planar.model import (
    VoxelTemporalAlign
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