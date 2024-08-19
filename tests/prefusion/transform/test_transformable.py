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
    CameraImageSegMask,
    CameraImageSegMaskSet,
    CameraImageDepth,
    CameraImageDepthSet,
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
    ts = TransformableSet([Transformable([1, 2, 33]), Transformable([7, 8, -1])])
    assert ts.transformables[0].data == [1, 2, 33]
    assert ts.transformables[1].data == [7, 8, -1]
    ts.adjust_saturation(saturation=0.5)
    assert ts.transformables[0].data == [1, 2, 33]
    assert ts.transformables[1].data == [7, 8, -1]
    assert ts.get_data() is ts.transformables


def test_transformableset_wrong_transformable_type():
    class T:
        pass
    with pytest.raises(TypeError):
        _ = TransformableSet([T(), T()])


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
                        "camera_type": "PerspectiveCamera",
                    },
                    "back": {
                        "extrinsic": (np.eye(3), np.ones(3)),
                        "camera_type": "FisheyeCamera",
                    }
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
            }
        }
    }


def test_camera_image_from_info(dataset_info, img324, img325):
    im1 = CameraImage.from_info("front", Path("/"), dataset_info, "20230901_000000/1692759619664")
    assert im1.data['cam_id'] == "front"
    assert im1.data['cam_type'] == "PerspectiveCamera"
    np.testing.assert_almost_equal(im1.data['img'], cv2.imread(img324))
    
    im2 = CameraImage.from_info("back", Path("/"), dataset_info, "20230901_000000/1692759619664")
    assert im2.data['cam_id'] == "back"
    assert im2.data['cam_type'] == "FisheyeCamera"
    np.testing.assert_almost_equal(im2.data['img'], cv2.imread(img325))


def test_camera_image_set(dataset_info, img324, img325):
    im1 = CameraImage.from_info("front", Path("/"), dataset_info, "20230901_000000/1692759619664")
    im2 = CameraImage.from_info("back", Path("/"), dataset_info, "20230901_000000/1692759619664")
    im_set = CameraImageSet([im1, im2])
    im_set.adjust_brightness(brightness=0.5)
    np.testing.assert_almost_equal(im_set.transformables[0].data['img'], (cv2.imread(img324) * 0.5).astype(np.uint8))
    np.testing.assert_almost_equal(im_set.transformables[1].data['img'], (cv2.imread(img325) * 0.5).astype(np.uint8))

def test_camera_image_set_2(dataset_info, img324, img325):
    im_set = CameraImageSet.from_info(Path("/"), dataset_info, "20230901_000000/1692759619664")
    im_set.adjust_brightness(brightness=0.5)
    np.testing.assert_almost_equal(im_set.transformables[0].data['img'], (cv2.imread(img324) * 0.5).astype(np.uint8))
    np.testing.assert_almost_equal(im_set.transformables[1].data['img'], (cv2.imread(img325) * 0.5).astype(np.uint8))


def test_camera_image_seg_mask(dataset_info, seg324, seg325):
    dictionary = {"camera_segs": ["lane", "arrow", "slot"]}
    seg1 = CameraImageSegMask.from_info("front", Path("/"), dataset_info, "20230901_000000/1692759619664", dictionary)
    seg2 = CameraImageSegMask.from_info("back", Path("/"), dataset_info, "20230901_000000/1692759619664", dictionary)
    np.testing.assert_almost_equal(seg1.data['img'], cv2.imread(seg324))
    np.testing.assert_almost_equal(seg2.data['img'], cv2.imread(seg325))


def test_camera_image_seg_mask_set(dataset_info):
    dictionary = {"camera_segs": ["lane", "arrow", "slot"]}
    seg_set = CameraImageSegMaskSet.from_info(Path("/"), dataset_info, "20230901_000000/1692759619664", dictionary)
    rotmat = np.array([
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [0, 0, 1],
    ])
    seg_set.rotate_3d(rotmat)
    np.testing.assert_almost_equal(seg_set.transformables[0].data['extrinsic'][1], np.array([1, 0, 1]))


def test_camera_image_depth(dataset_info, img324, img325):
    seg1 = CameraImageDepth.from_info("front", Path("/"), dataset_info, "20230901_000000/1692759619664", depth_mode='d')
    seg2 = CameraImageDepth.from_info("back", Path("/"), dataset_info, "20230901_000000/1692759619664", depth_mode='d')
    np.testing.assert_almost_equal(seg1.data['dep_img'], cv2.imread(img324))
    np.testing.assert_almost_equal(seg2.data['dep_img'], cv2.imread(img325))


def test_camera_image_depth_set(dataset_info):
    seg_set = CameraImageDepthSet.from_info(Path("/"), dataset_info, "20230901_000000/1692759619664", depth_mode='d')
    rotmat = np.array([
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [0, 0, 1],
    ])
    seg_set.rotate_3d(rotmat)
    np.testing.assert_almost_equal(seg_set.transformables[0].data['extrinsic'][1], np.array([1, 0, 1]))
