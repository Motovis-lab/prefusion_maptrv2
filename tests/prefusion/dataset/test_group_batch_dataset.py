import shutil
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import cv2
from copious.io.fs import mktmpdir, parent_ensured_path

from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.group_sampler import IndexGroupSampler, SequentialSceneFrameGroupSampler
from prefusion.dataset.dataset import GroupBatchDataset
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import TRANSFORMABLE_LOADERS
from prefusion.dataset.utils import load_frame_data_within_group


def test_batch_groups_1():
    batched_groups = GroupBatchDataset._batch_groups(
        group_batch_ind=1, 
        groups=[1, 2, 3], 
        batch_size=2
    )
    answer = [3, 2]
    assert batched_groups == answer


def test_batch_groups_2():
    batched_groups = GroupBatchDataset._batch_groups(
        group_batch_ind=1, 
        groups=[1, 2, 3, 4], 
        batch_size=3
    )
    answer = [4, 3, 2]
    assert batched_groups == answer


def test_batch_groups_3():
    batched_groups = GroupBatchDataset._batch_groups(
        group_batch_ind=0, 
        groups=[1, 2, 3, 4], 
        batch_size=3
    )
    answer = [1, 2, 3]
    assert batched_groups == answer


def test_batch_groups_4():
    batched_groups = GroupBatchDataset._batch_groups(
        group_batch_ind=0, 
        groups=[[1], [2], [3], [4]], 
        batch_size=1
    )
    assert batched_groups == [[1]]


def test_batch_groups_5():
    groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [8, 9, 10]]
    get_batched_groups = partial(GroupBatchDataset._batch_groups, groups=groups, batch_size=2)
    assert get_batched_groups(0) == [[1, 2, 3], [4, 5, 6]]
    assert get_batched_groups(1) == [[7, 8, 9], [8, 9, 10]]


class DummyTransform:
    def __init__(self, scope="frame") -> None:
        self.scope = scope

    def __call__(self, *transformables, **kwargs):
        return transformables


def test_load_all_transformables():
    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    data_root = Path("tests/prefusion/dataset/example_inputs")
    info_path = Path("tests/prefusion/dataset/mv4d-infos-for-test-001_separated.pkl")
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=data_root,
        info_path=info_path,
        transformables=dict(
            my_camera_images=dict(type="CameraImageSet"),
            my_ego_poses=dict(type="EgoPoseSet")
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        group_sampler=IndexGroupSampler(phase="val", possible_group_sizes=(4,), possible_frame_intervals=(2,)),
        batch_size=2,
    )

    with open(info_path, "rb") as f:
        info = pickle.load(f)
    frame_info = {f"{scn_id}/{frm_id}": path for scn_id, scn_data in info.items() for frm_id, path in scn_data['frame_info'].items()}
    group = [IndexInfo("20231101_160337", "1698825817764"), IndexInfo("20231101_160337", "1698825817864"), IndexInfo("20231101_160337", "1698825817964")]
    frame_data_within_group = load_frame_data_within_group(data_root, frame_info, group)
    all_transformables = dataset.load_all_transformables(frame_data_within_group, index_info)

    camera_image_set = all_transformables["my_camera_images"]
    assert camera_image_set.transformables['camera1'].img.sum() == 1752500
    assert camera_image_set.transformables['camera8'].ego_mask.sum() == 1365268
    np.testing.assert_almost_equal(
        camera_image_set.transformables['camera11'].intrinsic, 
        np.array([967.5516, 516.1143, 469.18085, 468.7578, 0.05346, -0.00585, -0.000539, -0.000161]),
        decimal=4
    )

    ego_pose_set = all_transformables["my_ego_poses"]
    assert len(ego_pose_set.transformables) == 3
    assert list(ego_pose_set.transformables.keys()) == ['-1', '0', '+1']
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"



@TRANSFORMABLE_LOADERS.register_module()
class CustomizedEgoPoseLoader:
    def __init__(self, data_root): self.data_root = data_root

    def load(self, name: str, frame_data_within_group, index_info, **kwargs):
        cur_frame = frame_data_within_group[index_info.frame_id]
        return cur_frame["ego_pose"]["rotation"], cur_frame["ego_pose"]["translation"]


def test_load_all_transformables_customized_loader():
    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    data_root = Path("tests/prefusion/dataset/example_inputs")
    info_path = Path("tests/prefusion/dataset/mv4d-infos-for-test-001_separated.pkl")
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=data_root,
        info_path=info_path,
        transformables=dict(
            single_frame_ego_pose=dict(
                type="EgoPose",
                loader=dict(type="CustomizedEgoPoseLoader")
            )
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        group_sampler=IndexGroupSampler(phase="val", possible_group_sizes=[4], possible_frame_intervals=[2]),
        batch_size=2,
    )

    with open(info_path, "rb") as f:
        info = pickle.load(f)
    frame_info = {f"{scn_id}/{frm_id}": path for scn_id, scn_data in info.items() for frm_id, path in scn_data['frame_info'].items()}
    group = [IndexInfo("20231101_160337", "1698825817764"), IndexInfo("20231101_160337", "1698825817864"), IndexInfo("20231101_160337", "1698825817964")]
    frame_data_within_group = load_frame_data_within_group(data_root, frame_info, group)
    all_transformables = dataset.load_all_transformables(frame_data_within_group, index_info)

    ego_pose = all_transformables["single_frame_ego_pose"]
    np.testing.assert_almost_equal(ego_pose[0], np.array(
        [[ 9.99999639e-01,  6.82115545e-04,  5.07486245e-04],
       [-6.78401337e-04,  9.99973246e-01, -7.28335824e-03],
       [-5.12440759e-04,  7.28301133e-03,  9.99973347e-01]]))
    np.testing.assert_almost_equal(ego_pose[1], np.array([-0.02978186,  0.7788203 ,  0.01499793]))


def test_dataset_with_test_scene_by_scene_bs1():
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=Path("tests/prefusion/dataset/example_inputs"),
        info_path=Path("tests/prefusion/dataset/mv4d-infos-for-test-001_separated.pkl"),
        transformables=dict(
            my_camera_images=dict(type="CameraImageSet"),
            my_ego_poses=dict(type="EgoPoseSet")
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        group_sampler=SequentialSceneFrameGroupSampler(phase="test_scene_by_scene"),
        batch_size=1,
    )

    assert len(dataset) == 17

    group_batch = dataset[2]
    assert len(group_batch) == 1
    assert len(group_batch[0]) == 1
    assert list(group_batch[0][0]['transformables']['my_ego_poses'].transformables.keys()) == [
        '-2', '-1', '0', '+1', '+2', '+3', '+4', '+5', '+6', '+7', '+8', '+9', '+10', '+11', '+12', '+13', '+14'
    ]


def create_dummy_camera_image_files(data_root: Path, ts: int):
    for cam in ["camera1", "camera5", "camera8", "camera11"]:
        save_path = str(parent_ensured_path(data_root / "camera" / cam / f"{ts}.jpg"))
        dummy_im_data = np.arange(40 * 3, dtype=np.uint8).reshape(5, 8, 3)
        cv2.imwrite(save_path, dummy_im_data)

def test_dataset_with_test_scene_by_scene_bs2():
    tmpdir = mktmpdir()
    create_dummy_camera_image_files(tmpdir, 1698825817864)
    create_dummy_camera_image_files(tmpdir, 1698825817964)
    shutil.copytree("tests/prefusion/dataset/example_inputs/self_mask", tmpdir / "self_mask")
    shutil.copytree("tests/prefusion/dataset/example_inputs/20231101_160337", tmpdir / "20231101_160337")

    dataset = GroupBatchDataset(
        name="gbd",
        data_root=tmpdir,
        info_path=Path("tests/prefusion/dataset/mv4d-infos-for-test-001_separated.pkl"),
        transformables=dict(
            my_camera_images=dict(type="CameraImageSet", loader=dict(type="CameraImageSetLoader", data_root=tmpdir)), # if providing no loader, dataset._build_transformable_loader would cache and reuse other test caes's loader
            my_ego_poses=dict(type="EgoPoseSet")
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        group_sampler=SequentialSceneFrameGroupSampler(phase="test_scene_by_scene"),
        batch_size=2,
    )

    assert len(dataset) == 9

    group_batch = dataset[1]
    assert len(group_batch) == 1
    assert len(group_batch[0]) == 2
    assert list(group_batch[0][0]['transformables']['my_ego_poses'].transformables.keys()) == [
        '-2', '-1', '0', '+1', '+2', '+3', '+4', '+5', '+6', '+7', '+8', '+9', '+10', '+11', '+12', '+13', '+14'
    ]


def test_dataset_with_test_scene_by_scene_2_scenes():
    with open("tests/prefusion/dataset/mv4d-infos-for-test-001_separated.pkl", "rb") as f:
        info = pickle.load(f)
    scene_info = info["20231101_160337"]["scene_info"]
    frame_info = info["20231101_160337"]["frame_info"]
    info["S0"] = {"frame_info": {k: v for k, v in list(frame_info.items())[:10]}, "scene_info": scene_info}
    info["S1"] = {"frame_info": {k: v for k, v in list(frame_info.items())[10:]}, "scene_info": scene_info}
    del info["20231101_160337"]
    tmpdir = mktmpdir()
    with open(tmpdir / "mv4d-infos-for-test-001_separated.pkl", "wb") as f:
        pickle.dump(info, f)
    
    create_dummy_camera_image_files(tmpdir, 1698825817864)
    create_dummy_camera_image_files(tmpdir, 1698825819264)
    shutil.copytree("tests/prefusion/dataset/example_inputs/self_mask", tmpdir / "self_mask")
    shutil.copytree("tests/prefusion/dataset/example_inputs/20231101_160337", tmpdir / "20231101_160337")

    dataset = GroupBatchDataset(
        name="gbd",
        data_root=tmpdir,
        info_path=tmpdir / "mv4d-infos-for-test-001_separated.pkl",
        transformables=dict(
            my_camera_images=dict(type="CameraImageSet", loader=dict(type="CameraImageSetLoader", data_root=tmpdir)), # if providing no loader, dataset._build_transformable_loader would cache and reuse other test caes's loader
            my_ego_poses=dict(type="EgoPoseSet")
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        group_sampler=SequentialSceneFrameGroupSampler(phase="test_scene_by_scene"),
        batch_size=1,
    )

    assert len(dataset) == 17
    assert list(dataset[2][0][0]['transformables']['my_ego_poses'].transformables.keys()) == [
        '-2', '-1', '0', '+1', '+2', '+3', '+4', '+5', '+6', '+7'
    ]
    assert list(dataset[-1][0][0]['transformables']['my_ego_poses'].transformables.keys()) == [
        '-6', '-5', '-4', '-3', '-2', '-1', '0'
    ]
