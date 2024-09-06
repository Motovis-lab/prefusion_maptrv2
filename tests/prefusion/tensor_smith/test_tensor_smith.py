import pytest
import numpy as np
import torch

from prefusion.dataset.tensor_smith import get_bev_intrinsics

def test_get_bev_intrinsics():
    voxel_shape=(6, 200, 160)
    voxel_range=([-0.5, 2.5], [10, -10], [8, -8])
    bev_intrinsics = get_bev_intrinsics(voxel_shape, voxel_range)
    assert bev_intrinsics == (-10, -10, 99.5, 79.5)

