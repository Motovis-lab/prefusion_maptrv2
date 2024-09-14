from easydict import EasyDict as edict

import numpy as np
from contrib.petr.tensor_smith import Bbox3DCorners
from prefusion.dataset.transform import Bbox3D


def test_bbox3d_corners():
    bbox3d = Bbox3D([
            {
                "class": "class.vehicle.passenger_car",
                "attr": { },
                "size": [4, 2, 1.5],
                "rotation": np.array(
                    [
                        [-0.7071067811865476, 0.7071067811865476, 0],
                        [-0.7071067811865476, -0.7071067811865476, 0],
                        [0, 0, 1],
                    ]  # rotate clockwise 135 degrees
                ),
                "translation": np.array([[5], [4], [0.7]]),
                "track_id": "1_1",
                "velocity": np.array([[0], [0], [0]]),
            }
        ], {"det": {"classes": ["class.vehicle.passenger_car"]}})
    tensor_smith = Bbox3DCorners()
    tensor_dict = tensor_smith(bbox3d)
    assert tensor_dict['classes'] == ['class.vehicle.passenger_car']
    np.testing.assert_almost_equal(tensor_dict['bbox3d_corners'], np.array([[
        [2.87867966, 3.29289322, -0.05],
        [4.29289322, 1.87867966, -0.05],
        [4.29289322, 1.87867966, 1.45],
        [2.87867966, 3.29289322, 1.45],
        [5.70710678, 6.12132034, -0.05],
        [7.12132034, 4.70710678, -0.05],
        [7.12132034, 4.70710678, 1.45],
        [5.70710678, 6.12132034, 1.45],
    ]]))