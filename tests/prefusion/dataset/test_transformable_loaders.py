from pathlib import Path
from typing import Any
import functools
import pickle

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from copious.io.fs import mktmpdir
from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.transformable_loader import (
    CameraImageSetLoader, 
    CameraDepthSetLoader, 
    CameraSegMaskSetLoader, 
    EgoPoseSetLoader, 
    Bbox3DLoader,
    AdvancedBbox3DLoader,
    ClassMapping,
    AttrMapping,
)

_approx = functools.partial(pytest.approx, rel=1e-4)


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
    assert camera_images.transformables['camera1'].img.sum() == 1752500
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

    with pytest.raises(AssertionError):
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

    with pytest.raises(AssertionError):
        loader.load("bbox_3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith())

    bbox_3d = loader.load("bbox_3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith(), dictionary=dic)
    assert bbox_3d.dictionary == dic
    assert isinstance(bbox_3d.tensor_smith, DummyAnnoTensorSmith)
    assert len(bbox_3d.elements) == 7
    assert bbox_3d.elements[-1]['size'] == [3.0765, 0.5656, 0.0195]


def test_advanced_bbox3d_loader_mapping():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    dic = {"classes": ["hahaha", "hello-world"]}
    loader = AdvancedBbox3DLoader(data_root, class_mapping={
        "speed_bump": ["class.traffic_facility.speed_bump"],
        "passenger_car": ["class.vehicle.passenger_car"],
        # "pedestrian": ["class.pedestrian.pedestrian"],
        "arrow": ["class.road_marker.arrow::attr.road_marker.arrow.type.ahead", "class.road_marker.arrow::attr.road_marker.arrow.type.ahead_left"],
        "text_icon": ["class.parking.text_icon::attr.parking.text_icon.type.number", "class.parking.text_icon::attr.parking.text_icon.type.text"],
    }, attr_mapping={
        "is_door_open": ["attr.vehicle.is_door_open.true"],
        "static": ["attr.time_varying.object.state.stationary"],
    })
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)

    adv_bbox3d = loader.load("adv_bbox3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith(), dictionary=dic) # won't honor dic (dic is a distraction here)
    assert adv_bbox3d.dictionary == {
        "classes": ["speed_bump", "passenger_car", "arrow", "text_icon"],
        "attrs": ["is_door_open", "static"]
    }
    assert isinstance(adv_bbox3d.tensor_smith, DummyAnnoTensorSmith)
    assert len(adv_bbox3d.elements) == 28 # 33 - 5(pedestrian)
    assert sum(ele['class'] == "speed_bump" for ele in adv_bbox3d.elements) == 1
    assert sum(ele['class'] == "passenger_car" for ele in adv_bbox3d.elements) == 13
    assert sum(ele['class'] == "arrow" for ele in adv_bbox3d.elements) == 2
    assert sum(ele['class'] == "text_icon" for ele in adv_bbox3d.elements) == 12


def test_class_mapping_validate_input():
    with pytest.raises(ValueError):
        _ = ClassMapping({"new_class_name": ["c1::attr1.True", "c1::attr1.True"]})
    with pytest.raises(ValueError):
        _ = ClassMapping({"new_class_name1": ["c1::attr1.True"], "new_class_name2": ["c1"]})
    with pytest.raises(ValueError):
        _ = ClassMapping({"new_class_name1": ["c1"], "new_class_name2": ["c1::attr1.True"]})
    with pytest.raises(ValueError):
        _ = ClassMapping({"new_class_name1": ["c1::attr1.True", "c1"]})

    clsmap = ClassMapping({"new_class_name1": ["c1::attr1.EnumA", "c1::attr1.EnumB"]})
    assert clsmap.hierarchical_mapping == {
        "c1": {
            "attr1.EnumA": "new_class_name1",
            "attr1.EnumB": "new_class_name1",
        },
    }

    clsmap = ClassMapping({"new_class_name1": ["c1::attr1.True"], "new_class_name2": ["c2::attr1.True"]})
    assert clsmap.hierarchical_mapping == {
        "c1": {"attr1.True": "new_class_name1"},
        "c2": {"attr1.True": "new_class_name2"},
    }


def test_class_mapping_get_mapped_class():
    clsmap = ClassMapping({"new_class_name1": ["c1::attr1.True"], "new_class_name2": ["c2::attr1.True"]})
    assert clsmap.get_mapped_class("c1", {"attr1": "attr1.True"}) == "new_class_name1"
    assert clsmap.get_mapped_class("c2", {"attr1": "attr1.True"}) == "new_class_name2"
    assert clsmap.get_mapped_class("c2", {"attr2": "attr2.True"}) == "c2"
    assert clsmap.get_mapped_class("c3", {"attr1": "attr1.True"}) == "c3"


def test_attr_mapping():
    attrmap = AttrMapping({"new_attr_name1": ["attr1.True", "attr2.True"], "new_attr_name2": ["attr3.True"]})
    assert attrmap.get_mapped_attr({"attr1": "attr1.True", "attr2": "attr2.True"}) == ["new_attr_name1"]
    assert attrmap.get_mapped_attr({"attr1": "attr1.True", "attr2": "attr2.True", "attr3": "attr3.True"}) == ["new_attr_name1", "new_attr_name2"]
    assert attrmap.get_mapped_attr({"attr1": "attr1.True", "attr3": "attr3.True"}) == ["new_attr_name1", "new_attr_name2"]
    assert attrmap.get_mapped_attr({"attr1": "attr1.False", "attr3": "attr3.True", "attr4": "attr4.True"}) == ["new_attr_name2"]


def test_advanced_bbox3d_loader_rearrange_axis_longer_edge_as_y():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    clsmap = {
        "speed_bump": ["class.traffic_facility.speed_bump"],
        "arrow": ["class.road_marker.arrow::attr.road_marker.arrow.type.ahead", "class.road_marker.arrow::attr.road_marker.arrow.type.ahead_left"],
        "text_icon": ["class.parking.text_icon::attr.parking.text_icon.type.number", "class.parking.text_icon::attr.parking.text_icon.type.text"],
    }
    loader = AdvancedBbox3DLoader(data_root, class_mapping=clsmap, axis_rearrange_method="longer_edge_as_y")
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    adv_bbox3d = loader.load("adv_bbox3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith())
    assert adv_bbox3d.dictionary == { "classes": ["speed_bump", "arrow", "text_icon"], "attrs": [] }
    assert len(adv_bbox3d.elements) == 15
    assert sum(ele['class'] == "speed_bump" for ele in adv_bbox3d.elements) == 1
    assert sum(ele['class'] == "arrow" for ele in adv_bbox3d.elements) == 2
    assert sum(ele['class'] == "text_icon" for ele in adv_bbox3d.elements) == 12

    speed_bump = [ele for ele in adv_bbox3d.elements if ele['class'] == 'speed_bump'][0]
    arrow_0, arrow_1 = [ele for ele in adv_bbox3d.elements if ele['class'] == 'arrow']
    text_icon_0, text_icon_14 = adv_bbox3d.elements[0], adv_bbox3d.elements[-1]
    
    assert speed_bump["size"] == _approx([0.4303, 3.3381, 0.0359])
    assert_almost_equal(speed_bump["rotation"], np.array([
        [-0.24632949, -0.96913558,  0.00989984],
        [ 0.96917362, -0.24626153,  0.00759955],
        [-0.00492705,  0.01146666,  0.99992212],
    ]))
    assert arrow_0["size"] == _approx([1.0291, 3.2605, 0.0243])
    assert arrow_1["size"] == _approx([0.5656, 3.0765, 0.0195])
    assert text_icon_0["size"] == _approx([0.2333, 0.6493, 0.0358])
    assert text_icon_14["size"] == _approx([0.2097, 0.7179, 0.0332])


def test_advanced_bbox3d_loader_rearrange_axis_longer_edge_as_x():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    clsmap = {
        "speed_bump": ["class.traffic_facility.speed_bump"],
        "arrow": ["class.road_marker.arrow::attr.road_marker.arrow.type.ahead", "class.road_marker.arrow::attr.road_marker.arrow.type.ahead_left"],
        "text_icon": ["class.parking.text_icon::attr.parking.text_icon.type.number", "class.parking.text_icon::attr.parking.text_icon.type.text"],
    }
    loader = AdvancedBbox3DLoader(data_root, class_mapping=clsmap, axis_rearrange_method="longer_edge_as_x")
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    adv_bbox3d = loader.load("adv_bbox3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith())
    assert adv_bbox3d.dictionary == { "classes": ["speed_bump", "arrow", "text_icon"], "attrs": [] }
    assert len(adv_bbox3d.elements) == 15
    assert sum(ele['class'] == "speed_bump" for ele in adv_bbox3d.elements) == 1
    assert sum(ele['class'] == "arrow" for ele in adv_bbox3d.elements) == 2
    assert sum(ele['class'] == "text_icon" for ele in adv_bbox3d.elements) == 12

    speed_bump = [ele for ele in adv_bbox3d.elements if ele['class'] == 'speed_bump'][0]
    arrow_0, arrow_1 = [ele for ele in adv_bbox3d.elements if ele['class'] == 'arrow']
    text_icon_0, text_icon_14 = adv_bbox3d.elements[0], adv_bbox3d.elements[-1]
    
    assert speed_bump["size"] == _approx([3.3381, 0.4303, 0.0359])
    assert_almost_equal(speed_bump["rotation"], np.array([
        [-0.96913558,  0.24632949,  0.00989984],
        [-0.24626153, -0.96917362,  0.00759955],
        [ 0.01146666,  0.00492705,  0.99992212],
    ]))
    assert arrow_0["size"] == _approx([3.2605, 1.0291, 0.0243])
    assert arrow_1["size"] == _approx([3.0765, 0.5656, 0.0195])
    assert text_icon_0["size"] == _approx([0.6493, 0.2333, 0.0358])
    assert text_icon_14["size"] == _approx([0.7179, 0.2097, 0.0332])

def test_advanced_bbox3d_loader_rearrange_axis_corners():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    loader = Bbox3DLoader(data_root)
    bbox3d = loader.load("bbox3d", info_data["20231101_160337"], ii, tensor_smith=DummyAnnoTensorSmith(), dictionary={"classes": ["class.traffic_facility.speed_bump"]})
    assert len(bbox3d.elements) == 1

    clsmap = {
        "speed_bump": ["class.traffic_facility.speed_bump"],
    }
    adv_loader = AdvancedBbox3DLoader(data_root, class_mapping=clsmap, axis_rearrange_method="longer_edge_as_x")
    bbox3d_rearranged = adv_loader.load("bbox3d_rearranged", info_data["20231101_160337"], ii)
    assert bbox3d_rearranged.dictionary == { "classes": ["speed_bump"], "attrs": [] }
    assert len(bbox3d_rearranged.elements) == 1

    # calcualted corner positions should be the same in ego-coord-sys but in shifted order
    assert_almost_equal(bbox3d.corners[0][[1, 5, 4, 0], :], bbox3d_rearranged.corners[0][[0, 1, 5, 4], :])
    # np.array([
    #    [ 1.56436025,  0.6194041 , -0.03814708],
    #    [-1.67071123, -0.20264152,  0.00012977],
    #    [-1.67035583, -0.20236869,  0.03602697],
    #    [ 1.56471565,  0.61967693, -0.00224988],
    #    [ 1.67035583,  0.20236869, -0.03602697],
    #    [-1.56471565, -0.61967693,  0.00224988],
    #    [-1.56436025, -0.6194041 ,  0.03814708],
    #    [ 1.67071123,  0.20264152, -0.00012977],
    # ])


def test_advanced_bbox3d_loader_with_clsmap_and_attrmap():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    clsmap = {"static_car": ["class.vehicle.passenger_car::attr.time_varying.object.state.stationary"]}
    attrmap = {"is_door_closed": ["attr.vehicle.is_door_open.false"], "is_static": ['attr.time_varying.object.state.stationary']}
    loader = AdvancedBbox3DLoader(data_root, class_mapping=clsmap, attr_mapping=attrmap, axis_rearrange_method="none")
    bbox3d = loader.load("bbox3d", info_data["20231101_160337"], ii, dictionary={"classes": ["class.vehicle.passenger_car"]})
    assert bbox3d.dictionary == { "classes": ["static_car"], "attrs": ["is_door_closed", "is_static"] }
    assert len(bbox3d.elements) == 12
    assert bbox3d.elements[0]["class"] == "static_car"
    assert bbox3d.elements[0]["attr"] == ["is_door_closed", "is_static"]


def test_advanced_bbox3d_loader_no_clsmap():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    attrmap = {"hello": ["world"], "is_static": ['attr.time_varying.object.state.stationary']}
    loader = AdvancedBbox3DLoader(data_root, attr_mapping=attrmap, axis_rearrange_method="none")
    bbox3d = loader.load("bbox3d", info_data["20231101_160337"], ii, dictionary={"classes": ["class.traffic_facility.speed_bump"]})
    assert bbox3d.dictionary == { "classes": ["class.traffic_facility.speed_bump"], "attrs": ["hello", "is_static"] }
    assert len(bbox3d.elements) == 1


def test_advanced_bbox3d_loader_no_clsmap_2():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    attrmap = {"is_door_closed": ["attr.vehicle.is_door_open.false"], "is_static": ['attr.time_varying.object.state.stationary']}
    loader = AdvancedBbox3DLoader(data_root, attr_mapping=attrmap, axis_rearrange_method="none")
    bbox3d = loader.load("bbox3d", info_data["20231101_160337"], ii, dictionary={"classes": ["class.vehicle.passenger_car"]})
    assert bbox3d.dictionary == { "classes": ["class.vehicle.passenger_car"], "attrs": ["is_door_closed", "is_static"] }
    assert len(bbox3d.elements) == 13
    assert bbox3d.elements[0]["class"] == "class.vehicle.passenger_car"
    assert bbox3d.elements[0]["attr"] == ["is_door_closed", "is_static"]
    assert bbox3d.elements[-1]["class"] == "class.vehicle.passenger_car"
    assert bbox3d.elements[-1]["attr"] == ["is_door_closed"]


def test_advanced_bbox3d_loader_no_clsmap_no_attrmap():
    data_root = Path("tests/prefusion/dataset/example_inputs")
    ii = IndexInfo('20231101_160337', '1698825817864')
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl", "rb") as f:
        info_data = pickle.load(f)
    loader = AdvancedBbox3DLoader(data_root, axis_rearrange_method="none")
    bbox3d = loader.load("bbox3d", info_data["20231101_160337"], ii, dictionary={"classes": ["class.traffic_facility.speed_bump"]})
    assert bbox3d.dictionary == { "classes": ["class.traffic_facility.speed_bump"], "attrs": [] }
    assert len(bbox3d.elements) == 1
