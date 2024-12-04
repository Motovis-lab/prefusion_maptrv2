from pathlib import Path

import numpy as np
import pytest

from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.group_sampler import IndexGroupSampler
from prefusion.dataset.dataset import GroupBatchDataset
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import TRANSFORMABLE_LOADERS


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


class DummyTransform:
    def __init__(self, scope="frame") -> None:
        self.scope = scope

    def __call__(self, *transformables, **kwargs):
        return transformables


def test_load_all_transformables():
    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=Path("tests/prefusion/dataset/example_inputs"),
        info_path=Path("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl"),
        transformables=dict(
            my_camera_images=dict(type="CameraImageSet"),
            my_ego_poses=dict(type="EgoPoseSet")
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        group_sampler=IndexGroupSampler(phase="val", possible_group_sizes=[4], possible_frame_intervals=[2]),
        batch_size=2,
    )

    all_transformables = dataset.load_all_transformables(index_info)

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

    def load(self, name: str, scene_data, index_info, **kwargs):
        scene = scene_data['frame_info']
        return scene[index_info.frame_id]["ego_pose"]["rotation"], scene[index_info.frame_id]["ego_pose"]["translation"]


def test_load_all_transformables_customized_loader():
    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=Path("tests/prefusion/dataset/example_inputs"),
        info_path=Path("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl"),
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

    all_transformables = dataset.load_all_transformables(index_info)

    ego_pose = all_transformables["single_frame_ego_pose"]
    np.testing.assert_almost_equal(ego_pose[0], np.array(
        [[ 9.99999639e-01,  6.82115545e-04,  5.07486245e-04],
       [-6.78401337e-04,  9.99973246e-01, -7.28335824e-03],
       [-5.12440759e-04,  7.28301133e-03,  9.99973347e-01]]))
    np.testing.assert_almost_equal(ego_pose[1], np.array([-0.02978186,  0.7788203 ,  0.01499793]))
