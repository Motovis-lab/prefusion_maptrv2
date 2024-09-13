import pytest
import numpy as np
import torch
from easydict import EasyDict as edict

from prefusion.dataset.tensor_smith import get_bev_intrinsics, CameraImageTensor

def test_get_bev_intrinsics():
    voxel_shape=(6, 200, 160)
    voxel_range=([-0.5, 2.5], [10, -10], [8, -8])
    bev_intrinsics = get_bev_intrinsics(voxel_shape, voxel_range)
    assert bev_intrinsics == (-10, -10, 99.5, 79.5)


def test_camera_image_tensor():
    camera_image = edict(
        img=np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(np.uint8),
        ego_mask=np.ones((2, 4, 3), dtype=np.uint8)
    )
    tensor_smith = CameraImageTensor(means=[1, 2, 3], stds=[0.1, 0.2, 0.3])
    tensor_dict = tensor_smith(camera_image)
    np.testing.assert_almost_equal(tensor_dict["img"].numpy(), np.array([
        [[-10.,  20.,  50.,  80.],
         [110., 140., 170., 200.]],

        [[ -5.,  10.,  25.,  40.],
         [ 55.,  70.,  85., 100.]],

        [[-3.33333333,  6.66666667, 16.66666667, 26.66666667],
         [36.66666667, 46.66666667, 56.66666667, 66.66666667]]
    ]), decimal=6)
