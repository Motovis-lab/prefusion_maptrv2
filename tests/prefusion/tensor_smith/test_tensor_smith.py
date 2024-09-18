import pytest
import numpy as np
import torch
from easydict import EasyDict as edict

from prefusion.dataset.tensor_smith import get_bev_intrinsics, CameraImageTensor, PlanarBbox3D
from prefusion.dataset.transform import Bbox3D, ToTensor

def test_get_bev_intrinsics():
    voxel_shape=(6, 200, 160)
    voxel_range=([-0.5, 2.5], [10, -10], [8, -8])
    bev_intrinsics = get_bev_intrinsics(voxel_shape, voxel_range)
    assert bev_intrinsics == (99.5, 79.5, -10, -10)


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


def test_planar_bbox_3d_tensor_smith():
    return
    bbox3d = Bbox3D(
        elements=[
            {
                'class': 'class.vehicle.passenger_car',
                'attr': {'attr.time_varying.object.state': 'attr.time_varying.object.state.stationary',
                        'attr.vehicle.is_trunk_open': 'attr.vehicle.is_trunk_open.false',
                        'attr.vehicle.is_door_open': 'attr.vehicle.is_door_open.false'},
                'size': [4.6486, 1.9505, 1.5845],
                'rotation': np.array([[ 0.93915682, -0.32818596, -0.10138267],
                                [ 0.32677338,  0.94460343, -0.03071667],
                                [ 0.1058472 , -0.00428138,  0.99437319]]),
                'translation': np.array([[-15.70570354], [ 11.88484971], [ -0.61029085]]), # NOTE: it is a column vector
                'track_id': '10035_0',
                'velocity': np.array([[0.], [0.], [0.]]) # NOTE: it is a column vector
            }
        ],
        dictionary={"det": {
            "classes": ['class.vehicle.passenger_car', 'people', 'bicycle']
        }},
        tensor_smith=PlanarBbox3D(
            voxel_shape=(),
            voxel_range=()
        )
    )
    result_tensor = bbox3d.to_tensor()
    answer_tensor = np.array(
        [
            [],
            [],
            [],
        ]
    )
    np.testing.assert_almost_equal(
        result_tensor,
        answer_tensor,
    )
