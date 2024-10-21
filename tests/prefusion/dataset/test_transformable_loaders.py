from pathlib import Path
from typing import Any
import pickle

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from copious.io.fs import mktmpdir
from prefusion.dataset.dataset import IndexInfo
from prefusion.dataset.transformable_loader import CameraImageSetLoader, CameraDepthSetLoader, CameraSegMaskSetLoader, EgoPoseSetLoader, Bbox3DLoader


class DummyTransform:
    def __init__(self, scope="frame") -> None:
        self.scope = scope

    def __call__(self, *transformables, **kwargs):
        return transformables


class DummyImgTensorSmith:
    def __call__(self, transformable, **kwds: Any) -> Any:
        return {"img": transformable.img}


class DummyAnnoTensorSmith:
    def __call__(self, transformable, **kwds: Any) -> Any:
        return {"seg": [[0, 1], [2, 3]], "reg": [0, 1, 2, 3]}


def test_load_camera_image_set():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    loader = CameraImageSetLoader(data_root)
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    camera_images = loader.load("camera_images", info_data["20231101_160337"], ii, tensor_smith=DummyImgTensorSmith())
    assert camera_images.transformables['camera1'].img.sum() == 699534854
    assert isinstance(camera_images.transformables['camera5'].tensor_smith, DummyImgTensorSmith)
    assert camera_images.transformables['camera8'].ego_mask.sum() == 1365268
    assert_almost_equal(
        camera_images.transformables['camera11'].intrinsic, 
        np.array([967.5516, 516.1143, 469.18085, 468.7578, 0.05346, -0.00585, -0.000539, -0.000161]),
        decimal=4
    )


def test_load_camera_seg_mask():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    loader = CameraSegMaskSetLoader(data_root)
    ii = IndexInfo('20231101_160337', '1698825817864')
    dic = {"classes": ["pedestrian", "passenger_car", "arrow"]}
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
        info_data["20231101_160337"]["frame_info"]["1698825817864"]["camera_image_seg"] = {}
        info_data["20231101_160337"]["frame_info"]["1698825817864"]["camera_image_seg"]["camera1"] = Path("seg/fisheye_semantic_segmentation/camera1/1698825817864.png")
        info_data["20231101_160337"]["frame_info"]["1698825817864"]["camera_image_seg"]["camera5"] = Path("seg/fisheye_semantic_segmentation/camera5/1698825817864.png")
        info_data["20231101_160337"]["frame_info"]["1698825817864"]["camera_image_seg"]["camera8"] = Path("seg/fisheye_semantic_segmentation/camera8/1698825817864.png")
        info_data["20231101_160337"]["frame_info"]["1698825817864"]["camera_image_seg"]["camera11"] = Path("seg/fisheye_semantic_segmentation/camera11/1698825817864.png")

    with pytest.raises(AttributeError):
        loader.load("camera_segs", info_data["20231101_160337"], ii, tensor_smith=DummyImgTensorSmith())

    camera_segs = loader.load("camera_segs", info_data["20231101_160337"], ii, tensor_smith=DummyImgTensorSmith(), dictionary=dic)
    assert camera_segs.transformables['camera1'].dictionary == dic
    assert camera_segs.transformables['camera1'].img.sum() == 1160515
    assert isinstance(camera_segs.transformables['camera5'].tensor_smith, DummyImgTensorSmith)
    assert camera_segs.transformables['camera8'].ego_mask.sum() == 1365268
    assert_almost_equal(
        camera_segs.transformables['camera11'].intrinsic, 
        np.array([967.5516, 516.1143, 469.18085, 468.7578, 0.05346, -0.00585, -0.000539, -0.000161]),
        decimal=4
    )


@pytest.fixture
def info_pkl_with_depth_path():
    import mmengine
    import mmcv
    info_data = mmengine.load("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl")

    tmpdir = mktmpdir()
    camera_image_depth = {}
    self_mask = {}
    IMG_KEYS = [
        'camera1', 'camera11', 'camera12', 'camera13', 'camera15', 'camera2', 
        'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8'
    ]
    for camera_t in IMG_KEYS:
        camera_image_depth.update({
            camera_t : f"{camera_t}_depth.npz"
        })
        self_mask.update({
            camera_t: f"{camera_t}_mask.png"
        })
        np.savez_compressed(str(tmpdir / f"{camera_t}_depth.npz"), depth=np.ones((5, 10)).astype(np.float16))
        mmcv.imwrite(np.ones((5, 10)), str(tmpdir / f"{camera_t}_mask.png"))

    info_data["20231101_160337"]["frame_info"]["1698825817864"].update({'camera_image_depth': camera_image_depth})
    info_data["20231101_160337"]['scene_info']['camera_mask'] =  self_mask
    
    depth_path = tmpdir / "mv4d-infos-for-test-depth.pkl"
    mmengine.dump(info_data, depth_path)
    
    return depth_path


def test_load_camera_depths(info_pkl_with_depth_path):
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open(info_pkl_with_depth_path, "rb") as f:
        info_data = pickle.load(f)
    data_root = info_pkl_with_depth_path.parent
    loader = CameraDepthSetLoader(data_root)

    camera_depth = loader.load('camera_depths', info_data["20231101_160337"], ii)
    assert len(camera_depth.transformables) == 12

    depth_fish_front = np.load(data_root / Path("camera1_depth.npz"))['depth'][..., None].astype(np.float32)
    assert np.all(camera_depth.transformables['camera1'].img == depth_fish_front)

    depth_fish_front = np.load(data_root / Path("camera2_depth.npz"))['depth'][..., None].astype(np.float32)
    assert np.all(camera_depth.transformables['camera2'].img == depth_fish_front)

    depth_fish_front = np.load(data_root / Path("camera6_depth.npz"))['depth'][..., None].astype(np.float32)
    assert np.all(camera_depth.transformables['camera6'].img == depth_fish_front)


def test_load_ego_poses():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    loader = EgoPoseSetLoader(data_root)
    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)

    ego_pose_set = loader.load('ego_poses', info_data["20231101_160337"], index_info)
    assert len(ego_pose_set.transformables) == 3
    assert list(ego_pose_set.transformables.keys()) == ['-1', '0', '+1']
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"

    index_info2 = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764", prev=IndexInfo("20231101_160337", "1698825817664")), next=IndexInfo("20231101_160337", "1698825817964"))
    ego_pose_set = loader.load('ego_poses', info_data["20231101_160337"], index_info2)
    assert len(ego_pose_set.transformables) == 4
    assert list(ego_pose_set.transformables.keys()) == ['-2', '-1', '0', '+1']
    assert ego_pose_set.transformables['-2'].timestamp == "1698825817664"
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"

    index_info3 = IndexInfo(
        prev=IndexInfo("20231101_160337", "1698825817764", prev=IndexInfo("20231101_160337", "1698825817664")),
        scene_id="20231101_160337",
        frame_id="1698825817864",
        next=IndexInfo("20231101_160337", "1698825817964", next=IndexInfo("20231101_160337", "1698825818064"))
    )
    ego_pose_set = loader.load('ego_poses', info_data["20231101_160337"], index_info3)
    assert len(ego_pose_set.transformables) == 5
    assert list(ego_pose_set.transformables.keys()) == ['-2', '-1', '0', '+1', '+2']
    assert ego_pose_set.transformables['-2'].timestamp == "1698825817664"
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"
    assert ego_pose_set.transformables['+2'].timestamp == "1698825818064"


def test_load_bbox_3d():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    loader = Bbox3DLoader(data_root)
    ii = IndexInfo('20231101_160337', '1698825817864')
    dic = {"classes": ["class.pedestrian.pedestrian", "class.road_marker.arrow"]}
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)

    with pytest.raises(AttributeError):
        loader.load("bbox_3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith())

    bbox_3d = loader.load("bbox_3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith(), dictionary=dic)
    assert bbox_3d.dictionary == dic
    assert isinstance(bbox_3d.tensor_smith, DummyAnnoTensorSmith)
    assert len(bbox_3d.elements) == 7
    assert bbox_3d.elements[-1]['size'] == [3.0765, 0.5656, 0.0195]
