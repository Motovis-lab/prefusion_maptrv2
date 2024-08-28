from pathlib import Path

import pytest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from copious.io.fs import mktmpdir

from prefusion.dataset.transform import (
    transform_method,
    Transformable,
    TransformableSet,
    CameraImageSet,
    CameraImage,
    CameraSegMask,
    CameraSegMaskSet,
    CameraDepth,
    CameraDepthSet,
    LidarPoints,
    Bbox3D,
    Polyline3D,
    ParkingSlot3D,
    OccSdfBev,
)


def test_transform_method():
    class T:
        @transform_method
        def a(self):
            pass

        def b(self):
            pass

    t = T()
    assert hasattr(t.a, "is_transform_method")
    assert not hasattr(t.b, "is_transform_method")


def test_transformableset_getattr():
    trb1, trb2 = Transformable([1, 2, 33]), Transformable([7, 8, -1])
    trb1.img = [1, 2, 33]
    trb2.img = [7, 8, -1]
    ts = TransformableSet({"front": trb1, "back": trb2})
    assert ts.transformables["front"].img == [1, 2, 33]
    assert ts.transformables["back"].img == [7, 8, -1]
    ts.adjust_saturation(saturation=0.5)
    assert ts.transformables["front"].img == [1, 2, 33]
    assert ts.transformables["back"].img == [7, 8, -1]


def test_transformableset_wrong_transformable_type():
    class T:
        pass

    with pytest.raises(TypeError):
        _ = TransformableSet({"t1": T(), "t2": T()})


@pytest.fixture
def img324():
    tmpdir = mktmpdir()
    img = np.arange(3 * 2 * 4, dtype=np.uint8).reshape(2, 4, 3)
    save_path = str(tmpdir / "im324.jpg")
    cv2.imwrite(save_path, img)
    return save_path


@pytest.fixture
def img325():
    tmpdir = mktmpdir()
    img = np.arange(3 * 2 * 5, dtype=np.uint8).reshape(2, 5, 3) * 2
    save_path = str(tmpdir / "im325.jpg")
    cv2.imwrite(save_path, img)
    return save_path


@pytest.fixture
def seg324():
    tmpdir = mktmpdir()
    segmask = np.zeros((2, 4, 3), dtype=np.uint8)
    segmask[0, :, 0] = 1
    segmask[1, :, 1] = 1
    segmask[0, 2, 1] = 1
    save_path = str(tmpdir / "seg324.jpg")
    cv2.imwrite(save_path, segmask)
    return save_path


@pytest.fixture
def seg325():
    tmpdir = mktmpdir()
    segmask = np.zeros((2, 5, 3), dtype=np.uint8)
    segmask[:, 0, 0] = 1
    segmask[:, 2, 1] = 1
    segmask[:, 4, 1] = 1
    save_path = str(tmpdir / "seg325.jpg")
    cv2.imwrite(save_path, segmask)
    return save_path


@pytest.fixture
def camera_ego_mask324():
    tmpdir = mktmpdir()
    img = np.ones((2, 4), dtype=np.uint8)
    save_path = str(tmpdir / "mask324.png")
    cv2.imwrite(save_path, img)
    return save_path


@pytest.fixture
def camera_ego_mask325():
    tmpdir = mktmpdir()
    img = np.ones((2, 5), dtype=np.uint8)
    save_path = str(tmpdir / "mask325.png")
    cv2.imwrite(save_path, img)
    return save_path


@pytest.fixture
def dataset_info(img324, img325, seg324, seg325, camera_ego_mask324, camera_ego_mask325):
    return {
        "20230901_000000": {
            "scene_info": {
                "camera_mask": {
                    "front": camera_ego_mask324,
                    "back": camera_ego_mask325,
                },
                "calibration": {
                    "front": {
                        "extrinsic": (np.eye(3), np.ones(3)),
                        "intrinsic": np.array([1, 1, 10, 10]),
                        "camera_type": "PerspectiveCamera",
                    },
                    "back": {
                        "extrinsic": (np.eye(3), np.ones(3)),
                        "intrinsic": np.array([2, 2, 20, 20]),
                        "camera_type": "FisheyeCamera",
                    },
                },
                "depth_mode": {
                    "front": "d",
                    "back": "z",
                },
                "dictionary": {
                    "camera_segs": ["lane", "arrow", "slot"],
                },
            },
            "frame_info": {
                "1692759619664": {
                    "camera_image": {
                        "front": img324,
                        "back": img325,
                    },
                    "camera_image_seg": {
                        "front": seg324,
                        "back": seg325,
                    },
                    "camera_image_depth": {
                        "front": img324,
                        "back": img325,
                    },
                }
            },
        }
    }


def get_image_related_data(dataset_info, cam_id, transformable_key):
    scene_info = dataset_info["20230901_000000"]["scene_info"]
    frame_info = dataset_info["20230901_000000"]["frame_info"]["1692759619664"]
    calib = scene_info["calibration"][cam_id]
    img = cv2.imread(str(frame_info[transformable_key][cam_id]))
    ego_mask = cv2.imread(str(scene_info["camera_mask"][cam_id]))
    cam_type, intrinsic, extrinsic = calib["camera_type"], calib["intrinsic"], calib["extrinsic"]
    return cam_type, img, ego_mask, extrinsic, intrinsic


def test_camera_image_creation(dataset_info, img324, img325):
    im1 = CameraImage("front", *get_image_related_data(dataset_info, "front", "camera_image"))
    assert im1.cam_id == "front"
    assert im1.cam_type == "PerspectiveCamera"
    np.testing.assert_almost_equal(im1.img, cv2.imread(img324))

    im2 = CameraImage("back", *get_image_related_data(dataset_info, "back", "camera_image"))
    assert im2.cam_id == "back"
    assert im2.cam_type == "FisheyeCamera"
    np.testing.assert_almost_equal(im2.img, cv2.imread(img325))


def test_camera_image_set(dataset_info, img324, img325):
    im1 = CameraImage("front", *get_image_related_data(dataset_info, "front", "camera_image"))
    im2 = CameraImage("back", *get_image_related_data(dataset_info, "back", "camera_image"))
    im_set = CameraImageSet({"front": im1, "back": im2})
    im_set.adjust_brightness(brightness=0.5)
    np.testing.assert_almost_equal(im_set.transformables["front"].img, (cv2.imread(img324) * 0.5).astype(np.uint8))
    np.testing.assert_almost_equal(im_set.transformables["back"].img, (cv2.imread(img325) * 0.5).astype(np.uint8))


def test_camera_image_seg_mask(dataset_info, seg324, seg325):
    seg1 = CameraSegMask(
        "front", *get_image_related_data(dataset_info, "front", "camera_image_seg"), {"seg": ["a", "b"]}
    )
    seg2 = CameraSegMask("back", *get_image_related_data(dataset_info, "back", "camera_image_seg"), {"seg": ["a", "b"]})
    np.testing.assert_almost_equal(seg1.img, cv2.imread(seg324))
    np.testing.assert_almost_equal(seg2.img, cv2.imread(seg325))
    assert seg1.dictionary == seg2.dictionary == {"seg": ["a", "b"]}


def test_camera_image_seg_mask_set(dataset_info):
    seg1 = CameraSegMask(
        "front", *get_image_related_data(dataset_info, "front", "camera_image_seg"), {"seg": ["a", "b"]}
    )
    seg2 = CameraSegMask("back", *get_image_related_data(dataset_info, "back", "camera_image_seg"), {"seg": ["a", "b"]})
    seg_set = CameraSegMaskSet({"front": seg1, "back": seg2})
    rotmat = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, -0.5, 0],
            [0, 0, 1],
        ]
    )
    seg_set.rotate_3d(rotmat)
    np.testing.assert_almost_equal(seg_set.transformables["front"].extrinsic[1], np.array([1, 0, 1]))


def test_camera_image_depth(dataset_info, img324, img325):
    depth1 = CameraDepth("front", *get_image_related_data(dataset_info, "front", "camera_image_depth"), "d")
    depth2 = CameraDepth("back", *get_image_related_data(dataset_info, "back", "camera_image_depth"), "z")
    np.testing.assert_almost_equal(depth1.img, cv2.imread(img324))
    np.testing.assert_almost_equal(depth2.img, cv2.imread(img325))
    assert depth1.depth_mode == "d"
    assert depth2.depth_mode == "z"


def test_camera_image_depth_set(dataset_info):
    depth1 = CameraDepth("front", *get_image_related_data(dataset_info, "front", "camera_image_depth"), "d")
    depth2 = CameraDepth("back", *get_image_related_data(dataset_info, "back", "camera_image_depth"), "z")
    depth_set = CameraDepthSet({"front": depth1, "back": depth2})
    rotmat = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, -0.5, 0],
            [0, 0, 1],
        ]
    )
    depth_set.rotate_3d(rotmat)
    np.testing.assert_almost_equal(depth_set.transformables["front"].extrinsic[1], np.array([1, 0, 1]))


def test_lidar_points_creation(dataset_info):
    lidar_points = LidarPoints(np.array([[0, 0, 0], [1, 0, 1], [-1, -1, 0], [2, 3, 5]]), np.array([10, 20, 15, 5]))
    flip_mat = np.array(
        [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    lidar_points.flip_3d(flip_mat=flip_mat)
    np.testing.assert_almost_equal(lidar_points.positions, np.array([[0, 0, 0], [-1, 0, 1], [1, -1, 0], [-2, 3, 5]]))


@pytest.fixture
def dictionary():
    return {
        "det": {
            "classes": ["class.vehicle.passenger_car", "class.pedestrian.pedestrian", "class.pedestrian.flipped_pedestrian"],
            "attrs": [],
        },
        "map": {
            "classes": ["class.road_marker.lane_line"],
            "attrs": [],
        },
        "arrow": {
            "classes": ["class.parking.parking_slot"],
            "attrs": [],
        },
    }


@pytest.fixture
def boxes_data():
    return [
        {
            "class": "class.vehicle.passenger_car",
            "attr": {
                "attr.time_varying.object.state": "attr.time_varying.object.state.stationary",
                "attr.vehicle.is_trunk_open": "attr.vehicle.is_trunk_open.false",
                "attr.vehicle.is_door_open": "attr.vehicle.is_door_open.false",
            },
            "size": [4.6486, 1.9505, 1.5845],
            "rotation": np.array(
                [
                    [0.93915682, -0.32818596, -0.10138267],
                    [0.32677338, 0.94460343, -0.03071667],
                    [0.1058472, -0.00428138, 0.99437319],
                ]
            ),
            "translation": np.array([[-15.70570354], [11.88484971], [-0.61029085]]),  # VERTICAL
            "track_id": "10035_0",  # NOT USED
            "velocity": np.array([[0.0], [0.0], [0.0]]),
        },
        {
            "class": "class.pedestrian.pedestrian",
            "attr": {},
            "size": [0.5909, 0.7893, 1.7243],
            "rotation": np.array(
                [
                    [9.99939319e-01, -1.10162954e-02, -1.04719755e-05],
                    [1.10162950e-02, 9.99939318e-01, -3.83972435e-05],
                    [1.08943354e-05, 3.82795512e-05, 9.99999999e-01],
                ]
            ),
            "translation": np.array([[33.2981], [19.3322], [1.2403]]),  # VERTICAL
            "track_id": "18_0",  # NOT USED
            "velocity": np.array([[-0.9613], [-0.6015], [-0.0098]]),
        },
        {
            "class": "class.traffic_facility.cone",
            "attr": {},
            "size": [0.3909, 0.393, 0.7243],
            "rotation": np.array(
                [
                    [9.99939319e-01, -1.10162954e-02, -1.04719755e-05],
                    [1.10162950e-02, 9.99939318e-01, -3.83972435e-05],
                    [1.08943354e-05, 3.82795512e-05, 9.99999999e-01],
                ]
            ),
            "translation": np.array([[13.2981], [-9.3322], [1.2403]]),  # VERTICAL
            "track_id": "80_0",  # NOT USED
            "velocity": np.array([[0], [0], [0]]),
        },
    ]


def test_bbox3d_creation(boxes_data, dictionary):
    bbox3d = Bbox3D(boxes_data, dictionary)
    bbox3d.rotate_3d(np.array([[0.5, 0.5, 0], [-0.5, 0.5, 1], [0, 0, 1]]))
    assert [ele["class"] for ele in bbox3d.elements] == ["class.vehicle.passenger_car", "class.pedestrian.pedestrian"]
    np.testing.assert_almost_equal(
        bbox3d.elements[0]["translation"], np.array([[-1.91042692], [13.18498577], [-0.61029085]])
    )


def test_bbox3d_flip_with_flip_aware_class_pairs(boxes_data, dictionary):
    flip_aware_class_pairs = [('class.pedestrian.pedestrian', 'class.pedestrian.flipped_pedestrian')]
    bbox3d = Bbox3D(boxes_data, dictionary, flip_aware_class_pairs=flip_aware_class_pairs)
    flip_mat = np.eye(3)
    flip_mat[1, 1] = -1
    bbox3d.flip_3d(flip_mat)
    assert [ele["class"] for ele in bbox3d.elements] == ["class.vehicle.passenger_car", "class.pedestrian.flipped_pedestrian"]
    np.testing.assert_almost_equal(
        bbox3d.elements[0]["translation"], np.array([[-15.70570354], [-11.88484971], [-0.61029085]])
    )



@pytest.fixture
def polyline_data():
    return [
        {
            "class": "class.parking.parking_slot",
            "attr": {
                "attr.parking.parking_slot.is_mechanical": "attr.parking.parking_slot.is_mechanical.false",
                "attr.parking.parking_slot.is_parkable": "attr.parking.parking_slot.is_parkable.false",
            },
            "points": np.array(
                [
                    [-0.0301, -14.5241, -0.0384],
                    [-2.5247, -15.1224, -0.0368],
                    [-1.3101, -19.9821, -0.025],
                    [1.1847, -19.3627, -0.0065],
                ]
            ),
        },
        {
            "class": "class.parking.parking_slot",
            "attr": {
                "attr.parking.parking_slot.is_mechanical": "attr.parking.parking_slot.is_mechanical.false",
                "attr.parking.parking_slot.is_parkable": "attr.parking.parking_slot.is_parkable.true",
            },
            "points": np.array(
                [
                    [2.4803, -13.9424, -0.0399],
                    [-0.0293, -14.5341, -0.0384],
                    [1.1938, -19.3656, -0.0065],
                    [3.6745, -18.7675, 0.0118],
                ]
            ),
        },
        {
            "class": "class.road_marker.arrow_heading_triangle",
            "attr": {},
            "points": np.array([[-5.849, -10.4515, -0.0431], [-7.1073, -10.515, -0.0516], [-5.9705, -9.99, -0.0424]]),
        },
    ]


def test_polyline3d_creation(polyline_data, dictionary):
    pl = Polyline3D(polyline_data, dictionary)
    pl.rotate_3d(np.array([[0.5, 0.5, 0], [-0.5, 0.5, 1], [0, 0, 1]]))
    assert [ele["class"] for ele in pl.elements] == ["class.parking.parking_slot"] * 2
    np.testing.assert_almost_equal(
        pl.elements[1]["points"],
        np.array(
            [
                [-5.73105e00, -8.25125e00, -3.99000e-02],
                [-7.28170e00, -7.29080e00, -3.84000e-02],
                [-9.08590e00, -1.02862e01, -6.50000e-03],
                [-7.54650e00, -1.12092e01, 1.18000e-02],
            ]
        ),
    )


def test_polyline3d_flip_with_flip_aware_class_pairs(polyline_data, dictionary):
    flip_aware_class_pairs = [('class.parking.parking_slot', 'class.parking.parking_slot.flipped')]
    pl = Polyline3D(polyline_data, dictionary, flip_aware_class_pairs=flip_aware_class_pairs)
    flip_mat = np.eye(3)
    flip_mat[1, 1] = -1
    pl.flip_3d(flip_mat)
    assert [ele["class"] for ele in pl.elements] == ["class.parking.parking_slot.flipped", "class.parking.parking_slot.flipped"]
    np.testing.assert_almost_equal(
        pl.elements[0]["points"],
        np.array(
            [
                [-0.0301, 14.5241, -0.0384],
                [-2.5247, 15.1224, -0.0368],
                [-1.3101, 19.9821, -0.025],
                [1.1847, 19.3627, -0.0065],
            ]
        ),
    )


@pytest.fixture
def vertical_parkslot():
    return [
        {
            "class": "class.parking.parking_slot",
            "attr": {
                "attr.parking.parking_slot.is_mechanical": "attr.parking.parking_slot.is_mechanical.false",
                "attr.parking.parking_slot.is_parkable": "attr.parking.parking_slot.is_parkable.false",
            },
            "points": np.array([
                [1, 1, 0.01],
                [2, 1, 0.02],
                [2, 3, 0.03],
                [1, 3, 0.04]
            ]),
        }
    ]


@pytest.fixture
def horizontal_parkslot():
    return [
        {
            "class": "class.parking.parking_slot",
            "attr": {
                "attr.parking.parking_slot.is_mechanical": "attr.parking.parking_slot.is_mechanical.false",
                "attr.parking.parking_slot.is_parkable": "attr.parking.parking_slot.is_parkable.false",
            },
            "points": np.array([
                [5, 3, 0.01],
                [1, 3, 0.02],
                [1, 1, 0.03],
                [5, 1, 0.04]
            ]),
        }
    ]


def test_parking_slot_3d_creation(vertical_parkslot, horizontal_parkslot, dictionary):
    parking_slots = ParkingSlot3D(vertical_parkslot + horizontal_parkslot, dictionary)
    assert [slot["class"] for slot in parking_slots.elements] == ["class.parking.parking_slot"] * 2



def test_parking_slot_3d_flip_y_horizontal_slot(horizontal_parkslot, dictionary):
    parking_slots = ParkingSlot3D(horizontal_parkslot, dictionary)
    assert [ele["class"] for ele in parking_slots.elements] == ["class.parking.parking_slot"]
    parking_slots.flip_3d(np.array(
        [[1, 0, 0], 
         [0, -1, 0], 
         [0, 0, 1]]
        )
    )
    np.testing.assert_almost_equal(parking_slots.elements[0]['points'], np.array([[1, -3, 0.02], [5, -3, 0.01], [5, -1, 0.04], [1, -1, 0.03]]))


def test_parking_slot_3d_flip_x_horizontal_slot(horizontal_parkslot, dictionary):
    parking_slots = ParkingSlot3D(horizontal_parkslot, dictionary)
    assert [ele["class"] for ele in parking_slots.elements] == ["class.parking.parking_slot"]
    parking_slots.flip_3d(np.array(
        [[-1, 0, 0], 
         [0, 1, 0], 
         [0, 0, 1]]
        )
    )
    np.testing.assert_almost_equal(parking_slots.elements[0]['points'], np.array([[-1, 3, 0.02], [-5, 3, 0.01], [-5, 1, 0.04], [-1, 1, 0.03]]))



def test_parking_slot_3d_flip_y_vertical_slot(vertical_parkslot, dictionary):
    parking_slots = ParkingSlot3D(vertical_parkslot, dictionary)
    assert [ele["class"] for ele in parking_slots.elements] == ["class.parking.parking_slot"]
    parking_slots.flip_3d(np.array(
        [[1, 0, 0], 
         [0, -1, 0], 
         [0, 0, 1]]
        )
    )
    np.testing.assert_almost_equal(parking_slots.elements[0]['points'], np.array([[2, -1, 0.02], [1, -1, 0.01], [1, -3, 0.04], [2, -3, 0.03]]))



def test_parking_slot_3d_flip_x_vertical_slot(vertical_parkslot, dictionary):
    parking_slots = ParkingSlot3D(vertical_parkslot, dictionary)
    assert [ele["class"] for ele in parking_slots.elements] == ["class.parking.parking_slot"]
    parking_slots.flip_3d(np.array(
        [[-1, 0, 0], 
         [0, 1, 0], 
         [0, 0, 1]]
        )
    )
    np.testing.assert_almost_equal(parking_slots.elements[0]['points'], np.array([[-2, 1, 0.02], [-1, 1, 0.01], [-1, 3, 0.04], [-2, 3, 0.03]]))


def test_parking_slot_3d_flip_y_multiple_slots(vertical_parkslot, horizontal_parkslot, dictionary):
    parking_slots = ParkingSlot3D(vertical_parkslot + horizontal_parkslot, dictionary)
    assert [ele["class"] for ele in parking_slots.elements] == ["class.parking.parking_slot"] * 2
    parking_slots.flip_3d(np.array(
        [[1, 0, 0], 
         [0, -1, 0], 
         [0, 0, 1]]
        )
    )
    np.testing.assert_almost_equal(parking_slots.elements[0]['points'], np.array([[2, -1, 0.02], [1, -1, 0.01], [1, -3, 0.04], [2, -3, 0.03]]))
    np.testing.assert_almost_equal(parking_slots.elements[1]['points'], np.array([[1, -3, 0.02], [5, -3, 0.01], [5, -1, 0.04], [1, -1, 0.03]]))


@pytest.fixture
def occ_sdf_bev():
    return OccSdfBev(
        src_view_range=[-50, 50, -50, 50, -2, 2],
        occ=np.array([
            [[1, 1, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 0]],

            [[1, 1, 1, 1],
             [0, 0, 0, 0],
             [1, 0, 0, 0]],
        ]),
        sdf=np.array([
            [[0.1, 0.1, 0.1, 0.1],
             [0.4, 0.3, 0.1, 0.2],
             [0.2, 0.9, 0.3, 1.1]]
        ]),
        height=np.array([
            [[ 0.1, 0.1, 0.1, 0.2],
             [ 0.0, 0.1, 0.2, 0.3],
             [-0.1, 0.0, 0.2, 0.4]]
        ]),
        dictionary={},
    )

def test_occ_sdf_bev(occ_sdf_bev):
    np.testing.assert_almost_equal(occ_sdf_bev.mask, np.ones((1, 3, 4)))
    assert occ_sdf_bev._bev_shape == (3, 4)


def test_occ_sdf_bev_flip_x(occ_sdf_bev):
    flip_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    occ_sdf_bev.flip_3d(flip_mat)
    np.testing.assert_almost_equal(occ_sdf_bev.occ, np.array([
        [[0, 1, 1, 1],
         [0, 1, 0, 0],
         [0, 0, 0, 0]],

        [[1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 1]],
    ]))
    np.testing.assert_almost_equal(occ_sdf_bev.sdf, np.array([
        [[0.1, 0.1, 0.1, 0.1],
         [0.2, 0.1, 0.3, 0.4],
         [1.1, 0.3, 0.9, 0.2]]
    ]))
    np.testing.assert_almost_equal(occ_sdf_bev.height, np.array([
        [[ 0.2, 0.1, 0.1, 0.1],
         [ 0.3, 0.2, 0.1, 0.0],
         [0.4, 0.2, 0.0, -0.1]]
    ]))
