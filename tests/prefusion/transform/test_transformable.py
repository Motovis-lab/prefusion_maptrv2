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
)


def test_transform_method():
    class T:
        @transform_method
        def a(self): pass
        def b(self): pass

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
                "camera_mask": { "front": camera_ego_mask324, "back": camera_ego_mask325, },
                "calibration": {
                    "front": { "extrinsic": (np.eye(3), np.ones(3)), "intrinsic": np.array([1, 1, 10, 10]), "camera_type": "PerspectiveCamera", },
                    "back": { "extrinsic": (np.eye(3), np.ones(3)), "intrinsic": np.array([2, 2, 20, 20]), "camera_type": "FisheyeCamera", }
                },
                "depth_mode": { "front": "d", "back": "z", },
                "dictionary": { "camera_segs": ["lane", "arrow", "slot"], }
            },
            "frame_info": {
                "1692759619664": {
                    "camera_image": { "front": img324, "back": img325, },
                    "camera_image_seg": { "front": seg324, "back": seg325, },
                    "camera_image_depth": { "front": img324, "back": img325, },
                }
            }
        }
    }


def get_image_related_data(dataset_info, cam_id, transformable_key):
    scene_info = dataset_info['20230901_000000']['scene_info']
    frame_info = dataset_info['20230901_000000']['frame_info']["1692759619664"]
    calib = scene_info['calibration'][cam_id]
    img = cv2.imread(str(frame_info[transformable_key][cam_id]))
    ego_mask = cv2.imread(str(scene_info['camera_mask'][cam_id]))
    cam_type, intrinsic, extrinsic = calib['camera_type'], calib['intrinsic'], calib['extrinsic']
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
    seg1 = CameraSegMask("front", *get_image_related_data(dataset_info, "front", "camera_image_seg"), {'seg': ['a', 'b']})
    seg2 = CameraSegMask("back", *get_image_related_data(dataset_info, "back", "camera_image_seg"), {'seg': ['a', 'b']})
    np.testing.assert_almost_equal(seg1.img, cv2.imread(seg324))
    np.testing.assert_almost_equal(seg2.img, cv2.imread(seg325))
    assert seg1.dictionary == seg2.dictionary == {'seg': ['a', 'b']}


def test_camera_image_seg_mask_set(dataset_info):
    seg1 = CameraSegMask("front", *get_image_related_data(dataset_info, "front", "camera_image_seg"), {'seg': ['a', 'b']})
    seg2 = CameraSegMask("back", *get_image_related_data(dataset_info, "back", "camera_image_seg"), {'seg': ['a', 'b']})
    seg_set = CameraSegMaskSet({"front": seg1, "back": seg2})
    rotmat = np.array([
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [0, 0, 1],
    ])
    seg_set.rotate_3d(rotmat)
    np.testing.assert_almost_equal(seg_set.transformables["front"].extrinsic[1], np.array([1, 0, 1]))


def test_camera_image_depth(dataset_info, img324, img325):
    depth1 = CameraDepth("front", *get_image_related_data(dataset_info, "front", "camera_image_depth"), 'd')
    depth2 = CameraDepth("back", *get_image_related_data(dataset_info, "back", "camera_image_depth"), 'z')
    np.testing.assert_almost_equal(depth1.img, cv2.imread(img324))
    np.testing.assert_almost_equal(depth2.img, cv2.imread(img325))
    assert depth1.depth_mode == 'd'
    assert depth2.depth_mode == 'z'


def test_camera_image_depth_set(dataset_info):
    depth1 = CameraDepth("front", *get_image_related_data(dataset_info, "front", "camera_image_depth"), 'd')
    depth2 = CameraDepth("back", *get_image_related_data(dataset_info, "back", "camera_image_depth"), 'z')
    depth_set = CameraDepthSet({"front": depth1, "back": depth2})
    rotmat = np.array([
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [0, 0, 1],
    ])
    depth_set.rotate_3d(rotmat)
    np.testing.assert_almost_equal(depth_set.transformables["front"].extrinsic[1], np.array([1, 0, 1]))


