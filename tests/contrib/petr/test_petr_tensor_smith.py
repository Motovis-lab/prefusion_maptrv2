import numpy as np
import pytest

from contrib.petr.tensor_smith import Bbox3DBasic
from prefusion.dataset.transform import Bbox3D


@pytest.fixture()
def bbox3d():
    return Bbox3D(
        "bbox_3d",
        [
            {
                "class": "class.vehicle.passenger_car",
                "attr": {},
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
        ],
        {"classes": ["class.vehicle.passenger_car"]},
    )


def test_bbox3d_xyz_lwh_yaw_vx_vy(bbox3d):
    tensor_smith = Bbox3DBasic(classes=["class.road_marker.arrow", "class.vehicle.passenger_car", "class.traffic_facility.box"], voxel_range=[[-100, 100], [100, -100], [100, -100]])
    tensor_dict = tensor_smith(bbox3d)
    assert tensor_dict["classes"].flatten().tolist() == [1]
    np.testing.assert_almost_equal(
        tensor_dict["xyz_lwh_yaw_vx_vy"],
        np.array([[5, 4, 0.7, 4, 2, 1.5, -2.35619449, 0, 0]]),
    )
    np.testing.assert_almost_equal(
        tensor_dict["corners"],
        np.array(
            [
                [
                    [2.87867966, 3.29289322, -0.05],
                    [4.29289322, 1.87867966, -0.05],
                    [4.29289322, 1.87867966, 1.45],
                    [2.87867966, 3.29289322, 1.45],
                    [5.70710678, 6.12132034, -0.05],
                    [7.12132034, 4.70710678, -0.05],
                    [7.12132034, 4.70710678, 1.45],
                    [5.70710678, 6.12132034, 1.45],
                ]
            ]
        ),
    )

